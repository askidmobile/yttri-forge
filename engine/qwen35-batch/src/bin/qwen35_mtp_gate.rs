//! qwen35_mtp_gate — шлюз совпадения выхода: один и тот же промпт прогоняется
//! с MTP и без него, токены сравниваются позиция в позицию.
//!
//! Запуск:
//!   qwen35_mtp_gate TEXT.gguf MTP.gguf [--prompt N] [--new M] [--slots S] [--control]
//!
//! Ширина драфта берётся из env `MTP_WIDTH` (в scheduler она кэшируется
//! в OnceLock — одно значение на процесс, поэтому матрица прогоняется по
//! процессу на сочетание). Адаптивная ширина гасится `MTP_ADAPTIVE=0`,
//! иначе она меняет ширину под собой и замер перестаёт отвечать на вопрос.
//!
//! Промпт нужной длины набирается из корпуса `GATE_CORPUS`
//! (по умолчанию /root/ppl-corpus.txt), зациклённого до N токенов, — важно,
//! чтобы логиты имели реалистичные зазоры между кандидатами.
//!
//! `--control` прогоняет baseline дважды: если два одинаковых прогона в одном
//! процессе уже расходятся, вопрос про MTP не имеет смысла.

use anyhow::{anyhow, Context, Result};
use candle_core::Device;
use qwen35_batch::model::{BatchModel, Sampler};
use qwen35_batch::real::{tokenizer, Qwen35BatchAdapter};
use qwen35_batch::scheduler::{BatchScheduler, StepOutcome};
use qwen35_batch::slot::SlotStatus;
use serde_json::json;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Greedy + запись зазора между первым и вторым кандидатом на каждом сэмпле.
/// Совпадение выхода само по себе ничего не стоит, если все зазоры широкие:
/// расхождение форм батча перекидывает argmax только на почти-ничьих.
/// Ties разрешаются как в GreedySampler — строгое `>`, побеждает первый.
#[derive(Default, Clone)]
struct GapSampler {
    gaps: Arc<Mutex<Vec<f32>>>,
}

impl Sampler for GapSampler {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        let (mut best, mut v1, mut v2) = (0u32, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for (i, &v) in logits.iter().enumerate() {
            if v > v1 {
                v2 = v1;
                v1 = v;
                best = i as u32;
            } else if v > v2 {
                v2 = v;
            }
        }
        self.gaps.lock().unwrap().push(v1 - v2);
        best
    }
}

/// Итог одного прогона: токены каждого слота + счётчики спекуляции.
struct RunOut {
    tokens: Vec<Vec<u32>>,
    drafted: usize,
    accepted: usize,
    enabled: bool,
    used: bool,
    /// Полное время прогона (с), ВКЛЮЧАЯ prefill — на длинном контексте им и занято.
    wall_s: f64,
    /// Скорость декода (ток/с): только decode-фаза, без prefill.
    decode_tps: f64,
    /// Скорость prefill (ток/с).
    prefill_tps: f64,
    /// Зазор top1-top2 на каждом вызове сэмплера (baseline: по токену на позицию).
    gaps: Vec<f32>,
}

fn run(
    sched: &mut BatchScheduler<Qwen35BatchAdapter>,
    prompt: &[u32],
    max_new: usize,
    slots: usize,
) -> Result<RunOut> {
    let t0 = Instant::now();
    let st0 = sched.stats_snapshot();
    let gaps = Arc::new(Mutex::new(Vec::new()));
    sched.set_sampler(Box::new(GapSampler { gaps: Arc::clone(&gaps) }));
    for _ in 0..slots {
        sched.submit(prompt.to_vec(), max_new);
    }
    let mut tokens = vec![Vec::new(); slots];
    let (mut drafted, mut accepted) = (0usize, 0usize);
    let (mut enabled, mut used) = (false, false);
    loop {
        let finished: Vec<usize> = sched
            .slots_mut()
            .iter()
            .filter(|s| s.status == SlotStatus::Finished)
            .map(|s| s.idx)
            .collect();
        for slot in finished {
            tokens[slot] = sched.slots_mut()[slot].generated_tokens().to_vec();
            if let Some(m) = sched.speculative_metrics(slot) {
                drafted += m.drafted;
                accepted += m.accepted;
                enabled |= m.enabled;
                used |= m.used;
            }
            sched.slots_mut()[slot].reset();
            sched.model_mut().reset_slot(slot)?;
        }
        if sched.step()? == StepOutcome::Idle
            && sched.slots_mut().iter().all(|s| s.status == SlotStatus::Idle)
        {
            break;
        }
    }
    let gaps = gaps.lock().unwrap().clone();
    // Дельта по статистике: она копится в scheduler'е между прогонами.
    let st = sched.stats_snapshot();
    let d_tok = st.total_decode_tokens - st0.total_decode_tokens;
    let d_dec = (st.decode_ns - st0.decode_ns) as f64 / 1e9;
    let d_pre = (st.prefill_ns - st0.prefill_ns) as f64 / 1e9;
    Ok(RunOut {
        tokens,
        drafted,
        accepted,
        enabled,
        used,
        wall_s: t0.elapsed().as_secs_f64(),
        decode_tps: if d_dec > 0.0 { d_tok as f64 / d_dec } else { 0.0 },
        prefill_tps: if d_pre > 0.0 { prompt.len() as f64 / d_pre } else { 0.0 },
        gaps,
    })
}

/// Позиция первого расхождения; None — последовательности совпадают целиком.
fn first_diff(a: &[u32], b: &[u32]) -> Option<usize> {
    let n = a.len().min(b.len());
    (0..n)
        .find(|&i| a[i] != b[i])
        .or(if a.len() == b.len() { None } else { Some(n) })
}

/// Промпт длиной ровно `n` токенов из зациклённого корпуса.
fn build_prompt(model: &Path, n: usize, offset: usize) -> Result<Vec<u32>> {
    let tok = tokenizer::load_from_gguf_path(model)?;
    let path = std::env::var("GATE_CORPUS")
        .unwrap_or_else(|_| "/root/ppl-corpus.txt".to_string());
    let text = std::fs::read_to_string(&path).with_context(|| format!("корпус {path}"))?;
    let ids = tok
        .encode(text.as_str(), false)
        .map_err(|e| anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    if ids.is_empty() {
        return Err(anyhow!("корпус {path} дал пустой набор токенов"));
    }
    let skip = offset % ids.len();
    Ok(ids.into_iter().cycle().skip(skip).take(n).collect())
}

fn main() -> Result<()> {
    let mut argv = std::env::args().skip(1);
    let text = argv
        .next()
        .context("usage: qwen35_mtp_gate TEXT.gguf MTP.gguf [--prompt N] [--new M] [--slots S] [--offset K] [--control] [--no-mtp] [--ignore-eos]")?;
    let mtp = argv.next().context("missing MTP.gguf")?;
    let (mut prompt_len, mut max_new, mut slots, mut control) = (1024usize, 64usize, 1usize, false);
    // Смещение по корпусу: разные промпты той же длины дают независимые
    // выборки первого расхождения — из них считается частота.
    let mut offset = 0usize;
    // --no-mtp: только baseline. Нужен, чтобы снять пик VRAM без MTP отдельным
    // процессом (внутри одного процесса MTP грузится поверх и пик уже общий).
    let mut no_mtp = false;
    // --ignore-eos: замеру скорости нужна одинаковая длина генерации на всех
    // точках, а EOS обрывает её через несколько токенов. Подменяем eos на
    // несуществующий id — слот идёт ровно max_new токенов.
    let mut ignore_eos = false;
    let rest: Vec<String> = argv.collect();
    let mut i = 0;
    while i < rest.len() {
        let need = |i: usize| -> Result<usize> {
            rest.get(i + 1)
                .ok_or_else(|| anyhow!("{} без значения", rest[i]))?
                .parse()
                .map_err(|e| anyhow!("{}: {e}", rest[i]))
        };
        match rest[i].as_str() {
            "--prompt" => {
                prompt_len = need(i)?;
                i += 2;
            }
            "--new" => {
                max_new = need(i)?;
                i += 2;
            }
            "--slots" => {
                slots = need(i)?;
                i += 2;
            }
            "--offset" => {
                offset = need(i)?;
                i += 2;
            }
            "--ignore-eos" => {
                ignore_eos = true;
                i += 1;
            }
            "--no-mtp" => {
                no_mtp = true;
                i += 1;
            }
            "--control" => {
                control = true;
                i += 1;
            }
            other => return Err(anyhow!("неизвестный аргумент {other}")),
        }
    }

    let width: usize = std::env::var("MTP_WIDTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let adaptive = std::env::var("MTP_ADAPTIVE").map(|v| v != "0").unwrap_or(true);

    let prompt = build_prompt(Path::new(&text), prompt_len, offset)?;
    let device = Device::new_cuda(0)?;
    let adapter = Qwen35BatchAdapter::load(Path::new(&text), device, slots)?;
    let eos = if ignore_eos { u32::MAX } else { adapter.eos() };
    let vocab = adapter.vocab_size();
    let mut sched = BatchScheduler::new(adapter, slots, eos, vocab);

    // Baseline: MTP не загружен вовсе — обычный батчевый декод шириной 1.
    let base = run(&mut sched, &prompt, max_new, slots)?;
    let control_diff = if control {
        let again = run(&mut sched, &prompt, max_new, slots)?;
        let d = first_diff(&base.tokens[0], &again.tokens[0]);
        println!("[gate] control baseline#2 diff={d:?}");
        d
    } else {
        None
    };

    if no_mtp {
        println!(
            "[gate] BASELINE prompt={prompt_len} decode={:.2} ток/с prefill={:.0} ток/с wall={:.1} с",
            base.decode_tps, base.prefill_tps, base.wall_s
        );
        println!(
            "{}",
            json!({
                "schema_version": "qwen35-mtp-gate-v3",
                "mode": "baseline_only",
                "prompt_tokens": prompt_len,
                "max_new": max_new,
                "base_decode_tps": base.decode_tps,
                "base_prefill_tps": base.prefill_tps,
                "base_wall_s": base.wall_s,
            })
        );
        return Ok(());
    }
    sched.model_mut().load_mtp(Path::new(&mtp))?;
    let spec = run(&mut sched, &prompt, max_new, slots)?;

    let diff = first_diff(&base.tokens[0], &spec.tokens[0]);
    let matched = diff.unwrap_or(base.tokens[0].len());
    println!(
        "[gate] prompt={prompt_len} offset={offset} width={width} adaptive={adaptive} \
         base_len={} mtp_len={} drafted={} accepted={} enabled={} used={} \
         base_decode={:.2} ток/с mtp_decode={:.2} ток/с ускорение={:.2}x \
         base_wall={:.1} с mtp_wall={:.1} с",
        base.tokens[0].len(),
        spec.tokens[0].len(),
        spec.drafted,
        spec.accepted,
        spec.enabled,
        spec.used,
        base.decode_tps,
        spec.decode_tps,
        if base.decode_tps > 0.0 { spec.decode_tps / base.decode_tps } else { 0.0 },
        base.wall_s,
        spec.wall_s,
    );
    match diff {
        None => println!("[gate] VERDICT match (совпало {matched} токенов)"),
        Some(pos) => {
            println!("[gate] VERDICT diverge@{pos} (совпало {matched} из {})", base.tokens[0].len());
            let lo = pos.saturating_sub(3);
            let hi = (pos + 4).min(base.tokens[0].len().min(spec.tokens[0].len()));
            println!("  base[{lo}..{hi}]: {:?}", &base.tokens[0][lo..hi]);
            println!("  mtp [{lo}..{hi}]: {:?}", &spec.tokens[0][lo..hi]);
        }
    }
    // Насколько близко к ничьей шло сэмплирование в baseline: если все зазоры
    // широкие, совпадение выхода ничего не доказывает про устойчивость.
    let mut sorted: Vec<(usize, f32)> = base.gaps.iter().copied().enumerate().collect();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
    let tight = base.gaps.iter().filter(|&&g| g < 0.1).count();
    println!(
        "[gate] gaps n={} min={:.4} p50={:.3} <0.1: {} узких: {:?}",
        base.gaps.len(),
        sorted.first().map(|x| x.1).unwrap_or(f32::NAN),
        sorted.get(sorted.len() / 2).map(|x| x.1).unwrap_or(f32::NAN),
        tight,
        sorted.iter().take(5).map(|(i, g)| (*i, (g * 1000.0).round() / 1000.0)).collect::<Vec<_>>(),
    );
    if let Some(pos) = diff {
        if let Some(g) = base.gaps.get(pos) {
            println!("[gate] зазор в точке расхождения #{pos}: {g:.4}");
        }
    }

    // Совпадение при нулевом drafted означает, что MTP не работал вовсе —
    // такой "успех" ничего не подтверждает.
    if spec.drafted == 0 {
        println!("[gate] WARN drafted=0 — MTP не участвовал в декоде");
    }

    println!(
        "{}",
        json!({
            "schema_version": "qwen35-mtp-gate-v2",
            "prompt_tokens": prompt_len,
            "offset": offset,
            "max_new": max_new,
            "slots": slots,
            "width": width,
            "adaptive": adaptive,
            "match": diff.is_none(),
            "first_diff": diff,
            "matched": matched,
            "base_len": base.tokens[0].len(),
            "mtp_len": spec.tokens[0].len(),
            "drafted": spec.drafted,
            "accepted": spec.accepted,
            "mtp_enabled": spec.enabled,
            "mtp_used": spec.used,
            "control_first_diff": control_diff,
            "gap_min": sorted.first().map(|x| x.1),
            "gap_p50": sorted.get(sorted.len() / 2).map(|x| x.1),
            "gap_tight_lt_0_1": tight,
            "gap_at_diff": diff.and_then(|p| base.gaps.get(p).copied()),
            "base_gaps": base.gaps,
            "base_decode_tps": base.decode_tps,
            "mtp_decode_tps": spec.decode_tps,
            "speedup": if base.decode_tps > 0.0 { spec.decode_tps / base.decode_tps } else { 0.0 },
            "base_prefill_tps": base.prefill_tps,
            "base_wall_s": base.wall_s,
            "mtp_wall_s": spec.wall_s,
            "acceptance": if spec.drafted > 0 { spec.accepted as f64 / spec.drafted as f64 } else { 0.0 },
            "base_tokens": base.tokens[0],
            "mtp_tokens": spec.tokens[0],
        })
    );
    Ok(())
}
