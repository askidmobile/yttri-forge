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
//! * `YTTRI_DECIDER_OUT` — куда записать строки отчёта (JSON Lines), необязательно; файл усекается на первой записи
//!   процесса, дописывается дальше;
//! * `YTTRI_FORGE_REV` — ревизия forge для каждой строки отчёта (тест её не знает); **обязательна, если задан
//!   `YTTRI_DECIDER_OUT`**: строка без ревизии не привязана к коду.
//!
//! Продуктовые флаги сборки (релиз Yttri идёт без `target-cpu`): `--config 'build.rustflags=[]'`. Каждый замер — отдельный
//! процесс, один на устройстве и CPU (`--exact --test-threads 1`, как требует `engine/AGENTS.md`): фильтр `bench_cpu` без
//! `--exact` совпал бы по подстроке ещё и с `bench_cpu_stability` и `bench_cpu_leak`, и libtest запустил бы все три
//! параллельно — замер D-032 шёл бы под фоном соседей.
//!
//! ```text
//! cargo test -p candle-transformers --release --config 'build.rustflags=[]' --test yttri_decider_bench -- --ignored --nocapture --exact bench_cpu --test-threads 1
//! cargo test -p candle-transformers --release --config 'build.rustflags=[]' --test yttri_decider_bench -- --ignored --nocapture --exact bench_cpu_stability --test-threads 1
//! cargo test -p candle-transformers --release --config 'build.rustflags=[]' --test yttri_decider_bench -- --ignored --nocapture --exact bench_cpu_leak --test-threads 1
//! cargo test -p candle-transformers --release --features cuda --config 'build.rustflags=[]' --test yttri_decider_bench -- --ignored --nocapture --exact bench_cuda --test-threads 1
//! ```

use candle::{Device, Result};
use candle_transformers::models::yttri_decider::YttriDecider;
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const WARMUP: usize = 10;
const RUNS: usize = 200;
const NFR_P95_MS: f64 = 40.0;

fn gguf() -> PathBuf {
    PathBuf::from(std::env::var("YTTRI_DECIDER_GGUF").expect("не задана YTTRI_DECIDER_GGUF"))
}

/// Строка отчёта: на экран и, если задан `YTTRI_DECIDER_OUT`, в файл (первая запись процесса его усекает). Ревизия forge
/// нужна, только когда пишется файл: прежние файлы отчётов копили строки без неё рядом со свежими, и числа документации
/// брались из строк неизвестной ревизии.
fn report(mut line: Value) {
    let rev = std::env::var("YTTRI_FORGE_REV").ok();
    if let (Some(obj), Some(rev)) = (line.as_object_mut(), &rev) {
        obj.insert("forge_rev".into(), json!(rev));
    }
    println!("REPORT {line}");
    if let Ok(path) = std::env::var("YTTRI_DECIDER_OUT") {
        assert!(
            rev.is_some(),
            "задан YTTRI_DECIDER_OUT, а YTTRI_FORGE_REV нет: строка отчёта без ревизии forge не принимается"
        );
        static FIRST: AtomicBool = AtomicBool::new(true);
        let mut f = if FIRST.swap(false, Ordering::Relaxed) {
            std::fs::File::create(path).unwrap()
        } else {
            std::fs::OpenOptions::new().append(true).open(path).unwrap()
        };
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

/// (RSS, закоммиченная частная память) процесса, МБ. RSS на Windows и Linux включает страницы отображённого файла
/// (таблица эмбеддингов), частная память — только кучу и стеки: по ней видна утечка.
#[cfg(windows)]
fn proc_mem() -> (f64, f64) {
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
            return (0.0, 0.0);
        }
        (
            p.working_set as f64 / 1_048_576.0,
            p.private as f64 / 1_048_576.0,
        )
    }
}

#[cfg(target_os = "linux")]
fn proc_mem() -> (f64, f64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let kb = |key: &str| {
        status
            .lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| {
                l.split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse::<f64>().ok())
            })
            .map_or(0.0, |kb| kb / 1024.0)
    };
    (kb("VmRSS:"), kb("RssAnon:"))
}

#[cfg(not(any(windows, target_os = "linux")))]
fn proc_mem() -> (f64, f64) {
    (0.0, 0.0)
}

fn rss_mb() -> f64 {
    proc_mem().0
}

fn private_mb() -> f64 {
    proc_mem().1
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

/// Фон — синтетические потоки на обычном приоритете, тот же генератор, что в S0a (`bench/src/main.rs`, `sweep --bg`):
/// цепочка умножение-сложение, флаг остановки проверяется раз в 2000 шагов.
fn spawn_background(n: usize, stop: &Arc<AtomicBool>) -> Vec<JoinHandle<()>> {
    (0..n)
        .map(|_| {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut x = 1.0f64;
                while !stop.load(Ordering::Relaxed) {
                    for _ in 0..2000 {
                        x = x * 1.000_000_1 + 1e-9;
                    }
                    std::hint::black_box(x);
                }
            })
        })
        .collect()
}

/// Устойчивость при L = 256 и 4 потоках: пауза между вызовами (письма приходят по одному) и фоновая арифметика на соседних
/// потоках (остальная нагрузка приложения), тот же сценарий и генератор фона, что в S0a.
///
/// Норма D-032 (p95 ≤ 40 мс) проверяется при 2 и 4 занятых фоновых потоках (на 8 ядрах это 6 и 8 потоков в сумме). При 6
/// фоновых потоках (10 потоков на 8 ядер) её не гарантирует никто: хвост задаёт самый медленный из потоков пула, статически
/// поделивших работу `gemm`, и вытесненный ОС поток задерживает весь проход. Это число только печатается; политика размера
/// пула на занятой машине — G5. Замерено на i7-9700 (L = 256, пул 4, три прогона этого теста; в скобках разброс ревью C2/C5):
/// 2 фоновых потока p95 17–25 мс, 4 фоновых — 25–29 мс (21–31), 6 фоновых — 50–53 мс (39–56); без фона 17,3–17,8, пауза 200 мс
/// около 19. «6 потоков выдерживает (38,4 мс)» из S0a — одиночная выборка ниже всех повторных замеров.
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
    let mut under_nfr = Vec::new();
    for bg in [2usize, 4, 6] {
        let stop = Arc::new(AtomicBool::new(false));
        let workers = spawn_background(bg, &stop);
        let s = stats(&measure_cpu(&model, &mut ids, &p, 256, Duration::ZERO));
        stop.store(true, Ordering::Relaxed);
        workers.into_iter().for_each(|w| w.join().unwrap());
        println!(
            "LAT cpu 4 потока L=256, +{bg} фоновых потоков  {}",
            s.line()
        );
        report(json!({"bench": format!("cpu-bg{bg}"), "p50": s.p50, "p95": s.p95, "p99": s.p99}));
        if bg <= 4 {
            under_nfr.push((bg, s.p95));
        }
    }
    for (bg, p95) in under_nfr {
        assert!(
            p95 <= NFR_P95_MS,
            "D-032: p95 {p95:.2} мс > {NFR_P95_MS} мс при {bg} фоновых потоках (L = 256, 4 потока)"
        );
    }
    Ok(())
}

/// Утечка: после прогрева (страницы таблицы, буферы потоков, кэши gemm) частная память не растёт на 5000 прогонах разной длины
/// (норма G8, как у e5: рост ≤ 5 МБ). RSS при этом может расти: в него входят страницы таблицы, которых ещё не касались.
#[test]
#[ignore = "замер: release и YTTRI_DECIDER_GGUF"]
fn bench_cpu_leak() -> Result<()> {
    let model = YttriDecider::load(gguf(), &Device::Cpu)?;
    let mut ids = Ids::new(model.config().vocab as u32);
    let p = pool(4);
    let mut rng = Lcg(0x1eaf);
    let mut run = |n: usize| {
        for _ in 0..n {
            let l = 2 + rng.next() as usize % 255;
            p.install(|| model.forward(&ids.next(l))).unwrap();
        }
    };
    run(1000);
    let (rss0, private0) = (rss_mb(), private_mb());
    for block in 1..=5 {
        run(1000);
        println!(
            "после {} прогонов: RSS {:.1} МБ, частная {:.1} МБ",
            1000 + block * 1000,
            rss_mb(),
            private_mb()
        );
    }
    let growth = private_mb() - private0;
    report(
        json!({"bench": "cpu-leak", "runs": 6000, "private_start_mb": private0, "private_end_mb": private_mb(), "private_growth_mb": growth, "rss_start_mb": rss0, "rss_end_mb": rss_mb()}),
    );
    assert!(
        growth <= 5.0,
        "частная память выросла на {growth:.1} МБ за 5000 прогонов"
    );
    Ok(())
}

// ------------------------------------------------------------------------------------------------ CUDA

#[cfg(feature = "cuda")]
fn vram_used_mb(dev: &Device) -> Option<f64> {
    match dev {
        Device::Cuda(cd) => cd
            .cuda_stream()
            .context()
            .mem_get_info()
            .ok()
            .map(|(free, total)| (total - free) as f64 / 1_048_576.0),
        _ => None,
    }
}

/// Соседнее устройство №1 «уже поднимает среду» (как e5 в Yttri): decider берёт своё устройство и видит активную среду.
#[cfg(feature = "cuda")]
fn cuda_probe(dev: &Device) -> Result<()> {
    let a = candle::Tensor::from_vec(vec![0.5f32; 64], (8, 8), dev)?;
    let c = a.matmul(&a)?;
    let _ = candle_nn::ops::softmax_last_dim(&c)?.tanh()?;
    dev.synchronize()
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "замер: release, видеокарта NVIDIA и YTTRI_DECIDER_GGUF"]
fn bench_cuda() -> Result<()> {
    let rss_start = rss_mb();
    // среда CUDA уже активна «соседом»: так decider используется в Yttri (candle_device::cuda_runtime_active())
    let t = Instant::now();
    let neighbour = Device::new_cuda(0)?;
    cuda_probe(&neighbour)?;
    let env_ms = t.elapsed().as_secs_f64() * 1e3;
    let (rss_env, vram_env) = (rss_mb(), vram_used_mb(&neighbour));
    println!("CUDA: среда поднята соседом за {env_ms:.0} мс, RSS {rss_start:.1} → {rss_env:.1} МБ, VRAM занято {vram_env:?} МБ");

    let t = Instant::now();
    let dev = Device::new_cuda(0)?; // своё устройство decider (отдельный контекст на ту же карту — как у моделей forge)
    let model = YttriDecider::load(gguf(), &dev)?;
    dev.synchronize()?;
    let load_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut ids = Ids::new(model.config().vocab as u32);
    let first = {
        let x = ids.next(256);
        let t = Instant::now();
        model.forward(&x)?;
        t.elapsed().as_secs_f64() * 1e3
    };
    let (rss_load, vram_load) = (rss_mb(), vram_used_mb(&dev));
    println!(
        "CUDA: готовность {load_ms:.0} мс, первый проход L=256 {first:.1} мс; RSS {rss_env:.1} → {rss_load:.1} МБ (+{:.1}); VRAM {vram_env:?} → {vram_load:?} МБ; веса на устройстве {:.1} МБ",
        rss_load - rss_env,
        model.resident_bytes() as f64 / 1_048_576.0
    );
    report(json!({
        "bench": "cuda-memory", "env_ms": env_ms, "load_ms": load_ms, "first_forward_ms": first, "rss_env_mb": rss_env, "rss_after_load_mb": rss_load,
        "vram_env_mb": vram_env, "vram_after_load_mb": vram_load, "weights_mb": model.resident_bytes() as f64 / 1_048_576.0
    }));

    // подряд
    for l in [64usize, 128, 256] {
        for _ in 0..WARMUP {
            model.forward(&ids.next(l))?;
        }
        let ms: Vec<f64> = (0..RUNS)
            .map(|_| {
                let x = ids.next(l);
                let t = Instant::now();
                model.forward(&x).unwrap();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        let s = stats(&ms);
        println!("LAT cuda подряд L={l:<3}  {}", s.line());
        report(
            json!({"bench": "cuda-consecutive", "l": l, "p50": s.p50, "p95": s.p95, "p99": s.p99, "max": s.max}),
        );
    }
    // память после 600 вызовов подряд (рост относительно «после загрузки» — буферы cuBLAS и аллокатор устройства)
    let (rss_runs, vram_runs) = (rss_mb(), vram_used_mb(&dev));
    println!(
        "CUDA: после замеров RSS {rss_runs:.1} МБ (+{:.1} к среде соседа), VRAM {vram_runs:?} МБ (+{:?} к среде соседа)",
        rss_runs - rss_env,
        vram_runs.zip(vram_env).map(|(a, b)| a - b)
    );
    report(
        json!({"bench": "cuda-memory-after-runs", "rss_mb": rss_runs, "rss_delta_mb": rss_runs - rss_env, "vram_mb": vram_runs}),
    );
    // редкие вызовы: пауза между ними, L × 60 вызовов после прогрева. Видеокарта сбрасывает такты за доли секунды, поэтому
    // медленный режим (≈ 30 мс при L = 256 против ≈ 3 мс подряд) начинается с паузы между вызовами, а не с простоя в секунды
    let gap_run = |ids: &mut Ids, l: usize, gap_ms: u64| -> Stats {
        let ms: Vec<f64> = (0..WARMUP + 60)
            .map(|i| {
                let x = ids.next(l);
                let t = Instant::now();
                model.forward(&x).unwrap();
                let dt = t.elapsed().as_secs_f64() * 1e3;
                std::thread::sleep(Duration::from_millis(gap_ms));
                (i, dt)
            })
            .filter(|(i, _)| *i >= WARMUP)
            .map(|(_, dt)| dt)
            .collect();
        stats(&ms)
    };
    for l in [64usize, 128, 256] {
        let s = gap_run(&mut ids, l, 1000);
        println!("LAT cuda пауза 1 с L={l:<3}  {}", s.line());
        report(
            json!({"bench": "cuda-gap1s", "l": l, "p50": s.p50, "p95": s.p95, "p99": s.p99, "max": s.max}),
        );
    }
    // где именно начинается медленный режим: L = 256, паузы от 50 до 500 мс
    for gap_ms in [50u64, 100, 200, 500] {
        let s = gap_run(&mut ids, 256, gap_ms);
        println!("LAT cuda пауза {gap_ms} мс L=256  {}", s.line());
        report(
            json!({"bench": "cuda-gap", "gap_ms": gap_ms, "l": 256, "p50": s.p50, "p95": s.p95, "p99": s.p99, "max": s.max}),
        );
    }
    // длинный простой (10 с): тот же медленный режим, что и при паузах выше; норма D-036 (p95 ≤ 40 мс) проверяется здесь
    let ms: Vec<f64> = (0..20)
        .map(|_| {
            std::thread::sleep(Duration::from_secs(10));
            let x = ids.next(256);
            let t = Instant::now();
            model.forward(&x).unwrap();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    let s = stats(&ms);
    println!(
        "LAT cuda после простоя 10 с L=256 (20 вызовов)  {}",
        s.line()
    );
    report(
        json!({"bench": "cuda-idle10s", "l": 256, "p50": s.p50, "p95": s.p95, "p99": s.p99, "max": s.max}),
    );
    assert!(
        s.p95 <= NFR_P95_MS,
        "D-036: GPU-путь после простоя p95 {:.2} мс > {NFR_P95_MS} мс",
        s.p95
    );
    drop(neighbour);
    Ok(())
}
