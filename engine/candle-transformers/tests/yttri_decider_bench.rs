//! Задержка, память и устойчивость decider (T-750, Ф3c, C2 и C5; D-032: p95 ≤ 40 мс на CPU при L = 256 и пуле из 4
//! потоков). Игнорируется: мерить имеет смысл только в release и на тихой машине.
//!
//! Время — «ids → логиты» одного вызова `forward` (нормализация, маска и токенизация не входят). Статистика — p50 / p95 / p99
//! по правилу nearest-rank, 10 прогревов и 200 замеров, id каждый раз разные.
//!
//! Переменные окружения — только в тестах:
//!
//! * `YTTRI_DECIDER_GGUF` — путь к `model.gguf` (стенд или боевая модель);
//! * `YTTRI_DECIDER_PARITY` — необязательно: JSON эталона паритета, из него берётся поток реальных токенов (иначе id по Ципфу
//!   от генератора — для латентности безразлично, для RSS нет: страницы таблицы эмбеддингов подгружаются по обращению);
//! * `YTTRI_DECIDER_OUT` — куда дописать строки отчёта (JSON Lines), необязательно.
//!
//! Продуктовые флаги сборки (релиз Yttri идёт без `target-cpu`): `--config 'build.rustflags=[]'`.
//!
//! ```text
//! cargo test -p candle-transformers --release --config 'build.rustflags=[]' --test yttri_decider_bench bench_cpu -- --ignored --nocapture
//! cargo test -p candle-transformers --release --features cuda --config 'build.rustflags=[]' --test yttri_decider_bench bench_cuda -- --ignored --nocapture
//! ```

use candle::{Device, Result};
use candle_transformers::models::yttri_decider::YttriDecider;
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const WARMUP: usize = 10;
const RUNS: usize = 200;
const NFR_P95_MS: f64 = 40.0;

fn gguf() -> PathBuf {
    PathBuf::from(std::env::var("YTTRI_DECIDER_GGUF").expect("не задана YTTRI_DECIDER_GGUF"))
}

fn report(line: Value) {
    println!("REPORT {line}");
    if let Ok(path) = std::env::var("YTTRI_DECIDER_OUT") {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        writeln!(f, "{line}").unwrap();
    }
}

// ------------------------------------------------------------------------------------------------ статистика

struct Stats {
    p50: f64,
    p95: f64,
    p99: f64,
    max: f64,
}

fn stats(xs: &[f64]) -> Stats {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = |q: f64| s[((q * s.len() as f64).ceil() as usize).clamp(1, s.len()) - 1];
    Stats {
        p50: rank(0.5),
        p95: rank(0.95),
        p99: rank(0.99),
        max: *s.last().unwrap(),
    }
}

impl Stats {
    fn line(&self) -> String {
        format!(
            "p50 {:7.2}  p95 {:7.2}  p99 {:7.2}  max {:7.2} мс",
            self.p50, self.p95, self.p99, self.max
        )
    }
}

// ------------------------------------------------------------------------------------------------ входы

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn unit(&mut self) -> f64 {
        self.next() as f64 / (1u64 << 53) as f64
    }
}

/// Поток реальных токенов из эталона паритета; без него — id по Ципфу от генератора.
struct Ids {
    pool: Vec<u32>,
    rng: Lcg,
    vocab: u32,
}

impl Ids {
    fn new(vocab: u32) -> Self {
        let mut pool = Vec::new();
        if let Ok(p) = std::env::var("YTTRI_DECIDER_PARITY") {
            let doc: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
            for c in doc["cases"].as_array().unwrap() {
                if ["state_v1", "corpus", "zipf"].contains(&c["kind"].as_str().unwrap_or("")) {
                    let ids = c["ids"].as_array().unwrap();
                    pool.extend(
                        ids[1..ids.len() - 1]
                            .iter()
                            .map(|x| x.as_u64().unwrap() as u32),
                    );
                }
            }
        }
        Ids {
            pool,
            rng: Lcg(0x5eed),
            vocab,
        }
    }

    /// Вход длины `l`: `<s>`, l − 2 токенов, `</s>`.
    fn next(&mut self, l: usize) -> Vec<u32> {
        let mut ids = Vec::with_capacity(l);
        ids.push(0);
        if self.pool.len() > l {
            let start = self.rng.next() as usize % (self.pool.len() - l);
            ids.extend_from_slice(&self.pool[start..start + l - 2]);
        } else {
            let span = (self.vocab - 3) as f64;
            ids.extend((0..l - 2).map(|_| 3 + (self.rng.unit().powi(3) * span) as u32));
        }
        ids.push(2);
        ids
    }
}

// ------------------------------------------------------------------------------------------------ память процесса

#[cfg(windows)]
fn rss_mb() -> f64 {
    #[repr(C)]
    struct Pmc {
        cb: u32,
        page_fault_count: u32,
        peak_working_set: usize,
        working_set: usize,
        q: [usize; 4],
        pagefile: usize,
        peak_pagefile: usize,
        private: usize,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(h: isize, p: *mut Pmc, cb: u32) -> i32;
    }
    // SAFETY: структура совпадает с PROCESS_MEMORY_COUNTERS_EX
    unsafe {
        let mut p: Pmc = std::mem::zeroed();
        p.cb = std::mem::size_of::<Pmc>() as u32;
        if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut p, p.cb) == 0 {
            return 0.0;
        }
        p.working_set as f64 / 1_048_576.0
    }
}

#[cfg(target_os = "linux")]
fn rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| {
                l.split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse::<f64>().ok())
            })
        })
        .map_or(0.0, |kb| kb / 1024.0)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn rss_mb() -> f64 {
    0.0
}

// ------------------------------------------------------------------------------------------------ CPU

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
}

fn measure_cpu(
    model: &YttriDecider,
    ids: &mut Ids,
    p: &rayon::ThreadPool,
    l: usize,
    gap: Duration,
) -> Vec<f64> {
    let mut ms = Vec::with_capacity(RUNS);
    for i in 0..WARMUP + RUNS {
        let x = ids.next(l);
        let t = Instant::now();
        p.install(|| model.forward(&x)).unwrap();
        let dt = t.elapsed().as_secs_f64() * 1e3;
        if i >= WARMUP {
            ms.push(dt);
        }
        if !gap.is_zero() {
            std::thread::sleep(gap);
        }
    }
    ms
}

#[test]
#[ignore = "замер: release и YTTRI_DECIDER_GGUF"]
fn bench_cpu() -> Result<()> {
    let rss0 = rss_mb();
    let t = Instant::now();
    let model = YttriDecider::load(gguf(), &Device::Cpu)?;
    let load_ms = t.elapsed().as_secs_f64() * 1e3;
    let rss_load = rss_mb();
    let mut ids = Ids::new(model.config().vocab as u32);
    println!(
        "CPU: загрузка {load_ms:.1} мс; RSS {rss0:.1} → {rss_load:.1} МБ; резидентно {:.1} МБ",
        model.resident_bytes() as f64 / 1_048_576.0
    );
    let mut nfr = f64::NAN;
    for threads in [2usize, 4, 6] {
        let p = pool(threads);
        for l in [64usize, 128, 256] {
            let s = stats(&measure_cpu(&model, &mut ids, &p, l, Duration::ZERO));
            println!("LAT cpu потоков {threads} L={l:<3}  {}", s.line());
            report(
                json!({"bench": "cpu", "threads": threads, "l": l, "p50": s.p50, "p95": s.p95, "p99": s.p99, "max": s.max, "load_ms": load_ms}),
            );
            if threads == 4 && l == 256 {
                nfr = s.p95;
            }
        }
    }
    println!("RSS после загрузки {rss_load:.1} МБ, после замеров {:.1} МБ (страницы таблицы эмбеддингов подгружаются по обращению)", rss_mb());
    report(
        json!({"bench": "cpu-rss", "after_load_mb": rss_load, "after_runs_mb": rss_mb(), "resident_mb": model.resident_bytes() as f64 / 1_048_576.0}),
    );
    assert!(
        nfr <= NFR_P95_MS,
        "D-032: p95 {nfr:.2} мс > {NFR_P95_MS} мс (L = 256, 4 потока)"
    );
    Ok(())
}

/// Устойчивость при L = 256 и 4 потоках: пауза между вызовами (письма приходят по одному) и фоновая арифметика на соседних потоках
/// (остальная нагрузка приложения). Тот же сценарий, что в S0a.
#[test]
#[ignore = "замер: release и YTTRI_DECIDER_GGUF"]
fn bench_cpu_stability() -> Result<()> {
    let model = YttriDecider::load(gguf(), &Device::Cpu)?;
    let mut ids = Ids::new(model.config().vocab as u32);
    let p = pool(4);
    let s = stats(&measure_cpu(
        &model,
        &mut ids,
        &p,
        256,
        Duration::from_millis(200),
    ));
    println!("LAT cpu 4 потока L=256, пауза 200 мс  {}", s.line());
    report(json!({"bench": "cpu-gap200", "p50": s.p50, "p95": s.p95, "p99": s.p99}));
    for bg in [2usize, 4, 6] {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let workers: Vec<_> = (0..bg)
            .map(|_| {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let mut x = 1.000_001f64;
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        for _ in 0..10_000 {
                            x = (x * 1.000_000_1).sin().abs() + 1.0;
                        }
                    }
                    x
                })
            })
            .collect();
        let s = stats(&measure_cpu(&model, &mut ids, &p, 256, Duration::ZERO));
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        workers.into_iter().for_each(|w| {
            w.join().unwrap();
        });
        println!(
            "LAT cpu 4 потока L=256, +{bg} фоновых потоков  {}",
            s.line()
        );
        report(json!({"bench": format!("cpu-bg{bg}"), "p50": s.p50, "p95": s.p95, "p99": s.p99}));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ CUDA
