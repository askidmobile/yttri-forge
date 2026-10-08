//! Паритет decider на настоящих весах против torch-f32 (T-750, Ф3c, C2 и C5; PD-052). Игнорируется: нужны стенд
//! `model.gguf` и файл эталона, оба вне репозитория.
//!
//! Эталон — `export_gguf.run_reference` из yttri-decider: HF XLM-R и головы в torch f32 на **деквантованных байтах того же
//! `model.gguf`** (B = 1 без паддинга, как рантайм). Расхождение даёт только вычисление, а не квантование.
//!
//! # Откуда берутся файлы (всё в git, `yttri-decider/bench`)
//!
//! ```text
//! export_gguf.py stand --from models/export-0.1.0 --out <стенд>        # стенд 0.1.0: три новые головы молчат (out_proj = 0)
//! make_stand_live.py <стенд>/model.gguf <live>/model.gguf              # будит молчащие головы: логиты зависят от входа
//! gen_parity.py --gguf <live>/model.gguf --out <live>/parity-live.json # эталон torch-f32, ≥ 1000 входов
//! ```
//!
//! Паритет на стенде с молчащими головами пуст: логиты головы постоянны, поэтому max |Δz| = 0 и смен вердикта нет сами
//! собой. Тест это замечает и падает (`degenerate`), так что годится только стенд `live` (или настоящая модель).
//!
//! Переменные окружения — только здесь, в продуктовом коде env не вводим:
//!
//! * `YTTRI_DECIDER_GGUF` — путь к `model.gguf` (стенд `live`, см. выше);
//! * `YTTRI_DECIDER_PARITY` — JSON эталона: `{"model_sha256", "outputs": [[имя, ширина]…], "cases": [{"name", "kind", "ids",
//!   "logits": {"logits_<qid>": […]}}]}` (генератор — `gen_parity.py`; состав: state_v1.json, тексты, id по Ципфу и равномерные,
//!   `<pad>` внутри входа, граничные длины; не меньше 1000 входов);
//! * `YTTRI_DECIDER_OUT` — куда записать отчёт (числа, без id и текста), необязательно;
//! * `YTTRI_DECIDER_THREADS` — размер пула rayon для CPU, по умолчанию 4;
//! * `YTTRI_FORGE_REV` — ревизия forge для отчёта (тест её не знает); **обязательна, если задан `YTTRI_DECIDER_OUT`**.
//!
//! Допуски PD-052: max |Δz| ≤ 1e-3 на каждой голове, смен вердикта 0 (FR-017 допускает 2 на 1000; здесь жёстче), max |Δp| —
//! справочно (допуск `self_check` 0,02 проверяется отдельно). Каждый замер — отдельный процесс (`--exact --test-threads 1`).
//!
//! ```text
//! cargo test -p candle-transformers --release --test yttri_decider_parity -- --ignored --nocapture --exact parity_cpu --test-threads 1
//! cargo test -p candle-transformers --release --features cuda --test yttri_decider_parity -- --ignored --nocapture --exact parity_cuda --test-threads 1
//! ```

use candle::{Device, Result};
use candle_transformers::models::yttri_decider::YttriDecider;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Instant;

const MAX_DZ: f64 = 1e-3;
const MIN_CASES: usize = 1000;
/// Голова, логиты эталона которой на всём наборе меняются меньше, вырождена: сверять по ней нечего.
const MIN_SPREAD: f64 = 1e-3;
// пороги вердикта как в decider.rs / validate_decider.py: noul — p ≥ t_yes «да», p ≤ t_no «нет»; choice — уверенный argmax
const T_YES: f64 = 0.8;
const T_NO: f64 = 0.1;
const CHOICE_THRESHOLD: f64 = 0.5;

fn env_path(name: &str) -> PathBuf {
    match std::env::var(name) {
        Ok(v) if !v.is_empty() => PathBuf::from(v),
        _ => panic!("не задана переменная окружения {name} (см. документацию теста)"),
    }
}

fn probs(z: &[f64]) -> Vec<f64> {
    if z.len() == 1 {
        return vec![1.0 / (1.0 + (-z[0]).exp())];
    }
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

#[derive(Debug, PartialEq)]
enum Verdict {
    Yes,
    No,
    Unsure,
    Class(usize),
}

fn verdict(z: &[f64]) -> Verdict {
    let p = probs(z);
    if p.len() == 1 {
        return if p[0] >= T_YES {
            Verdict::Yes
        } else if p[0] <= T_NO {
            Verdict::No
        } else {
            Verdict::Unsure
        };
    }
    let (arg, max) = p
        .iter()
        .enumerate()
        .fold((0, f64::MIN), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
    let n = p.len() as f64;
    if (max - 1.0 / n) / (1.0 - 1.0 / n) >= CHOICE_THRESHOLD {
        Verdict::Class(arg)
    } else {
        Verdict::Unsure
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    // nearest-rank, как в замерах S0a
    let i = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1;
    sorted[i]
}

#[derive(Default)]
struct HeadStats {
    dz: Vec<f64>,
    max_dp: f64,
    flips: usize,
    min_margin: f64,
    /// Наименьшее и наибольшее значение каждого выхода эталона по всем входам.
    lo: Vec<f64>,
    hi: Vec<f64>,
}

impl HeadStats {
    fn seen(&mut self, z_ref: &[f64]) {
        if self.lo.is_empty() {
            self.lo = z_ref.to_vec();
            self.hi = z_ref.to_vec();
        }
        for (j, &v) in z_ref.iter().enumerate() {
            self.lo[j] = self.lo[j].min(v);
            self.hi[j] = self.hi[j].max(v);
        }
    }

    /// Размах логитов эталона: наибольший по выходам (max − min по всем входам).
    fn spread(&self) -> f64 {
        self.lo
            .iter()
            .zip(&self.hi)
            .map(|(lo, hi)| hi - lo)
            .fold(0.0, f64::max)
    }
}

fn run(device: Device, label: &str) -> Result<()> {
    let gguf = env_path("YTTRI_DECIDER_GGUF");
    let doc: Value =
        serde_json::from_slice(&std::fs::read(env_path("YTTRI_DECIDER_PARITY")).unwrap()).unwrap();
    let threads: usize = std::env::var("YTTRI_DECIDER_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let on_cpu = device.is_cpu();

    let file_sha: String = Sha256::digest(std::fs::read(&gguf).unwrap())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        doc["model_sha256"].as_str().unwrap(),
        file_sha,
        "эталон посчитан на другом model.gguf"
    );

    let t_load = Instant::now();
    let model = YttriDecider::load(&gguf, &device)?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
    let outputs = model.outputs().to_vec();
    let want: Vec<(String, usize)> = doc["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| {
            (
                o[0].as_str().unwrap().to_string(),
                o[1].as_u64().unwrap() as usize,
            )
        })
        .collect();
    assert_eq!(
        outputs, want,
        "порядок и ширины голов модели и эталона различаются"
    );

    let cases = doc["cases"].as_array().unwrap();
    assert!(
        cases.len() >= MIN_CASES,
        "в эталоне {} входов, нужно не меньше {MIN_CASES}",
        cases.len()
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let mut stats: Vec<HeadStats> = outputs
        .iter()
        .map(|_| HeadStats {
            min_margin: f64::MAX,
            ..Default::default()
        })
        .collect();
    let mut ms = Vec::with_capacity(cases.len());
    let mut worst = (0f64, String::new());

    for (n, c) in cases.iter().enumerate() {
        let ids: Vec<u32> = c["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let t = Instant::now();
        let got = if on_cpu {
            pool.install(|| model.forward(&ids))?
        } else {
            model.forward(&ids)?
        };
        if n >= 10 {
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        for (h, (name, width)) in outputs.iter().enumerate() {
            let z_ref: Vec<f64> = c["logits"][name]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            let z_got: Vec<f64> = got[h].iter().map(|&x| x as f64).collect();
            assert_eq!((z_ref.len(), z_got.len()), (*width, *width));
            let dz = z_ref
                .iter()
                .zip(&z_got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            if dz > worst.0 {
                worst = (dz, format!("{name} {}", c["name"]));
            }
            let dp = probs(&z_ref)
                .iter()
                .zip(probs(&z_got))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            let s = &mut stats[h];
            s.seen(&z_ref);
            s.dz.push(dz);
            s.max_dp = s.max_dp.max(dp);
            s.flips += (verdict(&z_ref) != verdict(&z_got)) as usize;
            let margin = if *width == 1 {
                // зазор до границ t_yes / t_no в логитах
                [(T_YES / (1.0 - T_YES)).ln(), (T_NO / (1.0 - T_NO)).ln()]
                    .iter()
                    .map(|b| (z_ref[0] - b).abs())
                    .fold(f64::MAX, f64::min)
            } else {
                let mut top = z_ref.clone();
                top.sort_by(|a, b| b.partial_cmp(a).unwrap());
                top[0] - top[1]
            };
            s.min_margin = s.min_margin.min(margin);
        }
    }

    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut per_head = serde_json::Map::new();
    let mut ok = true;
    for (h, (name, width)) in outputs.iter().enumerate() {
        let s = &mut stats[h];
        s.dz.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (max, mean) = (
            *s.dz.last().unwrap(),
            s.dz.iter().sum::<f64>() / s.dz.len() as f64,
        );
        let spread = s.spread();
        let degenerate = spread <= MIN_SPREAD;
        println!(
            "{label} {name:<30} n={width:<3} max|dz| {max:.3e}  mean {mean:.2e}  p99 {:.2e}  max|dp| {:.2e}  смен вердикта {}  мин. зазор {:.3}  размах эталона {spread:.3}{}",
            percentile(&s.dz, 0.99),
            s.max_dp,
            s.flips,
            s.min_margin,
            if degenerate { "  ВЫРОЖДЕНА" } else { "" }
        );
        ok &= max <= MAX_DZ && s.flips == 0 && !degenerate;
        per_head.insert(
            name.clone(),
            json!({"n": width, "max_dz": max, "mean_dz": mean, "p99_dz": percentile(&s.dz, 0.99), "max_dp": s.max_dp, "flips": s.flips, "min_margin": s.min_margin, "ref_spread": spread, "degenerate": degenerate}),
        );
    }
    println!(
        "{label}: {} входов, пул {} потоков, загрузка {load_ms:.1} мс, прогон p50 {:.2} / p95 {:.2} мс; худший |dz| {:.3e} ({})",
        cases.len(),
        if on_cpu { threads } else { 0 },
        percentile(&ms, 0.5),
        percentile(&ms, 0.95),
        worst.0,
        worst.1
    );
    if let Ok(out) = std::env::var("YTTRI_DECIDER_OUT") {
        let rev = std::env::var("YTTRI_FORGE_REV").expect(
            "задан YTTRI_DECIDER_OUT, а YTTRI_FORGE_REV нет: отчёт без ревизии forge не принимается",
        );
        let report = json!({
            "device": label, "forge_rev": rev, "threads": if on_cpu { threads } else { 0 }, "cases": cases.len(), "model_sha256": file_sha,
            "tolerance": {"max_dz": MAX_DZ, "max_flips": 0}, "load_ms": load_ms,
            "forward_ms": {"p50": percentile(&ms, 0.5), "p95": percentile(&ms, 0.95)}, "heads": per_head,
        });
        std::fs::write(out, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
    assert!(
        ok,
        "паритет {label}: допуск max|dz| ≤ {MAX_DZ:e} и 0 смен вердикта нарушен, либо голова вырождена (размах логитов эталона ≤ {MIN_SPREAD:e}: стенд с молчащими головами, нужен make_stand_live.py)"
    );
    Ok(())
}

#[test]
#[ignore = "нужны YTTRI_DECIDER_GGUF и YTTRI_DECIDER_PARITY"]
fn parity_cpu() -> Result<()> {
    run(Device::Cpu, "cpu")
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "нужны YTTRI_DECIDER_GGUF, YTTRI_DECIDER_PARITY и видеокарта NVIDIA"]
fn parity_cuda() -> Result<()> {
    run(Device::new_cuda(0)?, "cuda")
}
