//! qwen35_verify_rows — сверка многострочной проверки спекуляции против
//! эквивалентных одиночных декодов.
//!
//! `speculative_verify(slot, inputs, pos)` обязан вернуть ровно те логиты,
//! которые дали бы K последовательных `decode_batch` из того же состояния:
//! спекуляция принимает токен ЦЕЛЕВОЙ модели, черновик лишь угадывает.
//! Гейт `qwen35_mtp_gate` показал, что при ширине 1 выход совпадает с
//! baseline, а при ширине >= 2 расходится — то есть врёт именно
//! многострочный путь. Этот пробник сравнивает логиты напрямую, без
//! сэмплера и без спекуляции, и печатает первую разошедшуюся строку.
//!
//! Запуск (сервер должен быть остановлен — нужна VRAM):
//!   qwen35_verify_rows TEXT.gguf MTP.gguf [--prompt N] [--k K]
//!
//! `--k` — число строк проверки (ширина черновика + 1 вход).

use anyhow::{anyhow, Context, Result};
use candle_core::Device;
use qwen35_batch::model::{BatchModel, DecodeBatch, DecodeItem, PrefillChunk};
use qwen35_batch::real::{tokenizer, Qwen35BatchAdapter};
use std::path::Path;

/// Максимум абсолютной разницы и позиция, где она достигнута.
fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut best = 0.0f32;
    let mut at = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > best {
            best = d;
            at = i;
        }
    }
    (best, at)
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i as u32;
        }
    }
    best
}

/// Зазор между первым и вторым кандидатом: широкий зазор означает, что
/// расхождение логитов реально меняет выбор, а не балансирует на ничьей.
fn top2_gap(v: &[f32]) -> f32 {
    let (mut v1, mut v2) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &x in v {
        if x > v1 {
            v2 = v1;
            v1 = x;
        } else if x > v2 {
            v2 = x;
        }
    }
    v1 - v2
}

fn build_prompt(model: &Path, n: usize, offset: usize) -> Result<Vec<u32>> {
    let tok = tokenizer::load_from_gguf_path(model)?;
    let path = std::env::var("QWEN36_GATE_CORPUS")
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
        .context("usage: qwen35_verify_rows TEXT.gguf MTP.gguf [--prompt N] [--k K]")?;
    let mtp = argv.next().context("missing MTP.gguf")?;
    let (mut prompt_len, mut k, mut offset) = (512usize, 4usize, 0usize);
    let rest: Vec<String> = argv.collect();
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--prompt" => {
                prompt_len = rest[i + 1].parse()?;
                i += 2;
            }
            "--k" => {
                k = rest[i + 1].parse()?;
                i += 2;
            }
            "--offset" => {
                offset = rest[i + 1].parse()?;
                i += 2;
            }
            other => return Err(anyhow!("неизвестный аргумент {other}")),
        }
    }

    // ROWS_FORCE_DMMV=1: оба пути через dequant+cuBLAS вместо MMVQ. Проверяет
    // гипотезу о том, что разницу даёт геометрия запуска MMVQ по b_size.
    if std::env::var("ROWS_FORCE_DMMV").as_deref() == Ok("1") {
        candle_core::quantized::cuda::set_force_dmmv(true);
        println!("[rows] FORCE_DMMV включён");
    }
    let device = Device::new_cuda(0)?;
    let mut model = Qwen35BatchAdapter::load(Path::new(&text), device, 1)?;
    model.load_mtp(Path::new(&mtp))?;
    let prompt = build_prompt(Path::new(&text), prompt_len, offset)?;
    println!("[rows] prompt={} k={} offset={offset}", prompt.len(), k);

    // Префилл слота 0 одним чанком: дальше оба пути стартуют из этого состояния.
    let chunk = PrefillChunk {
        slot_idx: 0,
        reset_first: true,
        tokens: prompt.clone(),
        start_pos: 0,
        is_final: true,
    };
    let last = model.prefill_chunk(&chunk)?;
    let first_token = argmax(&last);
    let pos = prompt.len();
    println!("[rows] после префилла: pos={pos} первый токен={first_token}");

    // ── Путь A: K одиночных декодов. Каждый шаг подаёт argmax предыдущего,
    // ровно как baseline-декод планировщика.
    let mut inputs = Vec::with_capacity(k);
    let mut single: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut tok = first_token;
    for step in 0..k {
        inputs.push(tok);
        let batch = DecodeBatch {
            items: vec![DecodeItem {
                slot_idx: 0,
                token: tok,
                pos: pos + step,
            }],
        };
        let rows = model.decode_batch(&batch)?;
        let logits = rows
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("decode_batch вернул пусто"))?;
        tok = argmax(&logits);
        single.push(logits);
    }
    let single_tokens: Vec<u32> = single.iter().map(|l| argmax(l)).collect();
    println!("[rows] одиночные декоды: входы={inputs:?} выходы={single_tokens:?}");

    // ── Путь B: та же K строк одной многострочной проверкой. Состояние
    // возвращаем к концу префилла, чтобы стартовать из той же точки.
    model.reset_slot(0)?;
    let chunk = PrefillChunk {
        slot_idx: 0,
        reset_first: true,
        tokens: prompt.clone(),
        start_pos: 0,
        is_final: true,
    };
    let _ = model.prefill_chunk(&chunk)?;
    model.speculative_begin(0)?;
    let multi = model.speculative_verify(0, &inputs, pos)?;
    let multi_tokens: Vec<u32> = multi.iter().map(|l| argmax(l)).collect();
    println!("[rows] многострочная проверка: выходы={multi_tokens:?}");
    model.speculative_rollback(0)?;

    // ── Сравнение построчно.
    let mut first_bad: Option<usize> = None;
    for row in 0..k {
        let (d, at) = max_abs_diff(&single[row], &multi[row]);
        let same_tok = single_tokens[row] == multi_tokens[row];
        println!(
            "[rows] строка {row}: max|Δlogit|={d:.6} (позиция {at}) argmax {} vs {} {} зазор_одиночного={:.4}",
            single_tokens[row],
            multi_tokens[row],
            if same_tok { "СОВПАЛ" } else { "РАЗОШЁЛСЯ" },
            top2_gap(&single[row]),
        );
        if !same_tok && first_bad.is_none() {
            first_bad = Some(row);
        }
    }

    match first_bad {
        None => println!("[rows] ВЕРДИКТ: argmax совпал на всех {k} строках"),
        Some(r) => println!("[rows] ВЕРДИКТ: первая расхождение на строке {r} из {k}"),
    }

    // Направление «построчная проверка»: k вызовов speculative_verify по одной
    // строке вместо одного батча. Если геометрия матвека — единственная
    // причина, логиты обязаны совпасть с одиночным декодом бит-в-бит.
    if std::env::var("ROWS_SERIAL_VERIFY").as_deref() == Ok("1") {
        model.reset_slot(0)?;
        let chunk = PrefillChunk {
            slot_idx: 0,
            reset_first: true,
            tokens: prompt.clone(),
            start_pos: 0,
            is_final: true,
        };
        let _ = model.prefill_chunk(&chunk)?;
        let mut worst = 0.0f32;
        let mut mismatch = 0usize;
        for row in 0..k {
            model.speculative_begin(0)?;
            let one = model.speculative_verify(0, &inputs[row..=row], pos + row)?;
            model.speculative_rollback(0)?;
            // Состояние двигаем обычным декодом: у пробника нет черновика,
            // а commit без него падает («committed length exceeds draft KV»).
            let batch = DecodeBatch {
                items: vec![DecodeItem { slot_idx: 0, token: inputs[row], pos: pos + row }],
            };
            let _ = model.decode_batch(&batch)?;
            let (d, _) = max_abs_diff(&single[row], &one[0]);
            if d > worst {
                worst = d;
            }
            if argmax(&single[row]) != argmax(&one[0]) {
                mismatch += 1;
            }
            println!("[rows] построчная строка {row}: max|Δ|={d:.6}");
        }
        println!(
            "[rows] ПОСТРОЧНАЯ ПРОВЕРКА: max|Δ|={worst:.6} расхождений argmax={mismatch} из {k}"
        );
        return Ok(());
    }

    // Повторяемость самого пути проверки: два одинаковых вызова из одного
    // состояния. Разные логиты = путь недетерминирован (гонка/грязный буфер).
    // Одинаковые = путь стабилен, но систематически считает иначе, чем декод.
    model.reset_slot(0)?;
    let chunk = PrefillChunk {
        slot_idx: 0,
        reset_first: true,
        tokens: prompt.clone(),
        start_pos: 0,
        is_final: true,
    };
    let _ = model.prefill_chunk(&chunk)?;
    model.speculative_begin(0)?;
    let multi2 = model.speculative_verify(0, &inputs, pos)?;
    model.speculative_rollback(0)?;
    let mut repeat_max = 0.0f32;
    for row in 0..k {
        let (d, _) = max_abs_diff(&multi[row], &multi2[row]);
        if d > repeat_max {
            repeat_max = d;
        }
    }
    println!(
        "[rows] повтор той же проверки: max|Δ|={repeat_max:.6} ({})",
        if repeat_max == 0.0 { "путь ДЕТЕРМИНИРОВАН" } else { "путь НЕДЕТЕРМИНИРОВАН" }
    );

    // Один раунд почти всегда совпадает по argmax: |Δlogit| ~0.3 переворачивает
    // выбор только там, где зазор до второго кандидата меньше. Считаем, как
    // часто это происходит, на длинной серии раундов — каждый раунд стартует
    // из состояния, дошедшего сюда одиночными декодами.
    let rounds: usize = std::env::var("ROWS_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if rounds == 0 {
        return Ok(());
    }
    println!("[rows] === серия из {rounds} раундов по {k} строк ===");
    model.reset_slot(0)?;
    let chunk = PrefillChunk {
        slot_idx: 0,
        reset_first: true,
        tokens: prompt.clone(),
        start_pos: 0,
        is_final: true,
    };
    let last = model.prefill_chunk(&chunk)?;
    let mut tok = argmax(&last);
    let mut cur = pos;
    let mut risky = 0usize;   // |Δ| >= зазор: погрешность способна перевернуть выбор
    let mut flipped = 0usize; // argmax реально разошёлся
    let mut worst_d = 0.0f32;
    for round in 0..rounds {
        // K одиночных декодов из текущего состояния.
        let mut ins = Vec::with_capacity(k);
        let mut singles: Vec<Vec<f32>> = Vec::with_capacity(k);
        let mut t = tok;
        for step in 0..k {
            ins.push(t);
            let batch = DecodeBatch {
                items: vec![DecodeItem { slot_idx: 0, token: t, pos: cur + step }],
            };
            let logits = model
                .decode_batch(&batch)?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("decode_batch вернул пусто"))?;
            t = argmax(&logits);
            singles.push(logits);
        }
        // Откат к началу раунда и та же K строк одной проверкой.
        model.reset_slot(0)?;
        let chunk = PrefillChunk {
            slot_idx: 0,
            reset_first: true,
            tokens: prompt.clone(),
            start_pos: 0,
            is_final: true,
        };
        let _ = model.prefill_chunk(&chunk)?;
        // Догоняем состояние до начала раунда одиночными шагами.
        let mut warm = argmax(&last);
        for step in 0..(cur - pos) {
            let batch = DecodeBatch {
                items: vec![DecodeItem { slot_idx: 0, token: warm, pos: pos + step }],
            };
            let logits = model
                .decode_batch(&batch)?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("decode_batch вернул пусто"))?;
            warm = argmax(&logits);
        }
        model.speculative_begin(0)?;
        let multi = model.speculative_verify(0, &ins, cur)?;
        model.speculative_rollback(0)?;
        for row in 0..k {
            let (d, _) = max_abs_diff(&singles[row], &multi[row]);
            let gap = top2_gap(&singles[row]);
            if d > worst_d {
                worst_d = d;
            }
            if d >= gap {
                risky += 1;
            }
            if argmax(&singles[row]) != argmax(&multi[row]) {
                flipped += 1;
                if flipped <= 3 {
                    println!(
                        "[rows] ПЕРЕВОРОТ раунд {round} строка {row}: |Δ|={d:.4} зазор={gap:.4} {} -> {}",
                        argmax(&singles[row]),
                        argmax(&multi[row])
                    );
                }
            }
        }
        // Двигаем состояние вперёд одиночными декодами (эталонный путь).
        model.reset_slot(0)?;
        let chunk = PrefillChunk {
            slot_idx: 0,
            reset_first: true,
            tokens: prompt.clone(),
            start_pos: 0,
            is_final: true,
        };
        let l2 = model.prefill_chunk(&chunk)?;
        let mut w = argmax(&l2);
        for step in 0..(cur - pos + k) {
            let batch = DecodeBatch {
                items: vec![DecodeItem { slot_idx: 0, token: w, pos: pos + step }],
            };
            let logits = model
                .decode_batch(&batch)?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("decode_batch вернул пусто"))?;
            w = argmax(&logits);
        }
        tok = w;
        cur += k;
    }
    let total = rounds * k;
    println!(
        "[rows] ИТОГ: строк={total} перевёрнуто={flipped} опасных(|Δ|>=зазор)={risky} max|Δlogit|={worst_d:.4}"
    );
    Ok(())
}
