//! Интеграционный тест реальной Qwen3.5-4B GGUF через `real-model` адаптер.
//!
//! Запуск:
//! ```sh
//! YTTRI_MODEL_DIR=/Volumes/Askid\ Dev/Projects/Yttri/frontend/src-tauri/resources/models/qwen3.5-4b \
//! cargo test -p qwen35-batch --features real-model \
//!     --test real_qwen35_batch --release -- --ignored --nocapture
//! ```
//!
//! Проверяем:
//! 1. B=1 batched-path == single-stream greedy.
//! 2. B=2/4: aggregate tok/s vs sequential, per-request tok/s.
//! 3. Greedy parity B=4 vs B=1 (batched vs sequential).
//!
//! Путь: `Qwen35BatchAdapter` с true batched decode под `BatchScheduler`; одна
//! копия весов, per-slot DeltaNet/KV state — см. adapter.rs.

#![cfg(feature = "real-model")]

use std::path::PathBuf;
use std::time::Instant;

use qwen35_batch::model::{DecodeBatch, DecodeItem, PrefillChunk};
use qwen35_batch::real::tokenizer::{self, ChatMsg};
use qwen35_batch::real::Qwen35BatchAdapter;
use qwen35_batch::scheduler::BatchScheduler;
use qwen35_batch::{BatchModel, DEFAULT_NUM_SLOTS};

fn model_dir() -> PathBuf {
    std::env::var("YTTRI_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let win_path = PathBuf::from("D:\\Models\\lmstudio-community\\Qwen3.5-4B-GGUF");
            if win_path.exists() {
                win_path
            } else {
                PathBuf::from(
                    "/Volumes/Askid Dev/Projects/Yttri/frontend/src-tauri/resources/models/qwen3.5-4b",
                )
            }
        })
}

fn gguf_path() -> PathBuf {
    std::env::var("QWEN35_TEST_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| model_dir().join("Qwen3.5-4B-Q4_K_M.gguf"))
}

fn accelerator_device() -> candle_core::Device {
    // Force CPU path (тест batched decode fallback без Metal/CUDA batched ctx).
    // На macOS: metal feature скомпилирован, но Device::Cpu → metal_ctx/metal_ctx_batched
    // = None → forward_decode_batch использует CPU fallback (cpu_state_batched).
    if std::env::var("YTTRI_FORCE_CPU")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        eprintln!("[real] YTTRI_FORCE_CPU=1 → Device::Cpu (CPU batched decode fallback)");
        return candle_core::Device::Cpu;
    }
    #[cfg(feature = "cuda")]
    {
        return candle_core::Device::new_cuda(0).expect("CUDA device");
    }
    #[cfg(all(not(feature = "cuda"), target_os = "macos"))]
    {
        qwen35_batch::real::metal_utils::configure_metal_env();
        let device = candle_core::Device::new_metal(0).expect("Metal device");
        qwen35_batch::real::metal_utils::metal_probe(&device).expect("Metal probe");
        return device;
    }
    #[cfg(all(not(feature = "cuda"), not(target_os = "macos")))]
    panic!("real Qwen3.5 benchmark requires feature `cuda` on this platform");
}

/// Детерминированные token IDs для parity (содержательно не важно — важна
/// идентичность batched vs sequential путей). Для качественной генерации
/// нужен настоящий tokenizer (см. REAL_MODEL.md §7).
fn dummy_prompt_ids(seed: u32, len: usize) -> Vec<u32> {
    (0..len)
        .map(|i| (seed.wrapping_add(i as u32 * 7) % 1000) + 10)
        .collect()
}

#[test]
#[ignore = "требует GGUF Qwen3.5-4B на диске + GPU (долгая загрузка ~2.5 ГБ)"]
fn real_qwen35_load_and_single_forward() {
    let _ = env_logger::try_init();
    let gguf = gguf_path();
    assert!(gguf.exists(), "GGUF не найден: {:?}", gguf);

    let device = accelerator_device();

    let mut adapter =
        Qwen35BatchAdapter::load(&gguf, device, DEFAULT_NUM_SLOTS).expect("load Qwen35 adapter");
    let vocab = adapter.vocab_size();
    eprintln!("[real] loaded, vocab={vocab}, eos={}", adapter.eos());

    let prompt = dummy_prompt_ids(42, 8);
    let logits = adapter
        .prefill_chunk(&PrefillChunk {
            slot_idx: 0,
            reset_first: true,
            tokens: prompt,
            start_pos: 0,
            is_final: true,
        })
        .expect("prefill");
    assert_eq!(logits.len(), vocab, "logits size != vocab");
    eprintln!("[real] prefill logits len={} OK", logits.len());
}

#[test]
#[ignore = "требует GGUF + GPU; full-logits B=1 vs B=3"]
fn real_qwen35_batched_logits_equal_single() {
    let gguf = gguf_path();
    assert!(gguf.exists(), "GGUF не найден: {:?}", gguf);
    let device = accelerator_device();
    let mut adapter = Qwen35BatchAdapter::load(&gguf, device, 4).expect("load adapter");
    let prompt = dummy_prompt_ids(42, 8);
    let mut first_logits = None;
    for slot_idx in 0..4 {
        let logits = adapter
            .prefill_chunk(&PrefillChunk {
                slot_idx,
                reset_first: true,
                tokens: prompt.clone(),
                start_pos: 0,
                is_final: true,
            })
            .expect("prefill");
        if let Some(first) = &first_logits {
            assert_eq!(&logits, first, "prefill logits differ for slot {slot_idx}");
        } else {
            first_logits = Some(logits);
        }
    }
    let token = first_logits
        .unwrap()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32;
    let decode = |slots: &[usize]| DecodeBatch {
        items: slots
            .iter()
            .map(|&slot_idx| DecodeItem {
                slot_idx,
                token,
                pos: prompt.len(),
            })
            .collect(),
    };
    let single = adapter
        .decode_batch(&decode(&[0]))
        .expect("B=1 decode")
        .pop()
        .unwrap();
    let batched = adapter
        .decode_batch(&decode(&[1, 2, 3]))
        .expect("B=3 decode");
    for (index, logits) in batched.iter().enumerate() {
        assert_eq!(logits, &single, "B=3 row {index} differs from B=1 logits");
    }
    eprintln!(
        "[logits-parity] B=1 vs B=3: {} logits BIT-EXACT",
        single.len()
    );
}

#[test]
#[ignore = "требует GGUF + GPU; same-slot reset/replay full-logits parity"]
fn real_qwen35_same_slot_replay_is_bit_exact() {
    let gguf = gguf_path();
    assert!(gguf.exists(), "GGUF не найден: {:?}", gguf);
    let device = accelerator_device();
    let mut adapter = Qwen35BatchAdapter::load(&gguf, device, 1).expect("load adapter");
    let prompt = dummy_prompt_ids(42, 25);
    let steps = 16;

    let mut logits = adapter
        .prefill_chunk(&PrefillChunk {
            slot_idx: 0,
            reset_first: true,
            tokens: prompt.clone(),
            start_pos: 0,
            is_final: true,
        })
        .expect("first prefill");
    let mut expected = Vec::with_capacity(steps);
    let mut forced = Vec::with_capacity(steps - 1);
    for step in 0..steps {
        expected.push(logits);
        if step + 1 == steps {
            break;
        }
        let token = expected[step]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        forced.push(token);
        logits = adapter
            .decode_batch(&DecodeBatch {
                items: vec![DecodeItem {
                    slot_idx: 0,
                    token,
                    pos: prompt.len() + step,
                }],
            })
            .expect("first decode")
            .pop()
            .unwrap();
    }

    adapter.reset_slot(0).expect("reset slot");
    let mut actual = adapter
        .prefill_chunk(&PrefillChunk {
            slot_idx: 0,
            reset_first: true,
            tokens: prompt.clone(),
            start_pos: 0,
            is_final: true,
        })
        .expect("second prefill");
    for step in 0..steps {
        assert_eq!(
            actual, expected[step],
            "full logits differ at replay step {step}"
        );
        if step + 1 == steps {
            break;
        }
        actual = adapter
            .decode_batch(&DecodeBatch {
                items: vec![DecodeItem {
                    slot_idx: 0,
                    token: forced[step],
                    pos: prompt.len() + step,
                }],
            })
            .expect("second decode")
            .pop()
            .unwrap();
    }
}

#[derive(Debug)]
struct BenchResult {
    batch: usize,
    outputs: Vec<Vec<u32>>,
    wall_ns: u128,
    prefill_ns: u128,
    decode_ns: u128,
    decode_steps: usize,
    max_concurrent: usize,
    rss_mb: f64,
    vram_mb: f64,
    vram_delta_mb: f64,
}

#[cfg(unix)]
fn current_rss_mb() -> f64 {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output();
    output
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|kb| kb as f64 / 1024.0)
        .unwrap_or(f64::NAN)
}

#[cfg(windows)]
fn current_rss_mb() -> f64 {
    let pid = std::process::id().to_string();
    let script = format!("(Get-Process -Id {pid}).WorkingSet64");
    std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", &script])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|bytes| bytes as f64 / 1024.0 / 1024.0)
        .unwrap_or(f64::NAN)
}

/// Общая занятая VRAM GPU по nvidia-smi. Это не process-exclusive метрика
/// (Windows WDDM не всегда отдаёт per-process memory), поэтому дополнительно
/// сохраняем delta относительно значения непосредственно перед load adapter.
fn current_vram_mb() -> f64 {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().next()?.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

fn run_bench_loaded(
    sched: &mut BatchScheduler<Qwen35BatchAdapter>,
    prompts: &[Vec<u32>],
    max_new: usize,
    batch: usize,
    vram_before_load_mb: f64,
) -> BenchResult {
    let before = sched.stats().clone();
    let t0 = Instant::now();
    let mut outputs = Vec::with_capacity(prompts.len());
    for chunk in prompts.chunks(batch) {
        outputs.extend(
            sched
                .run_with_collection(chunk.to_vec(), max_new)
                .unwrap_or_else(|e| panic!("run B={batch}: {e}")),
        );
    }
    let wall_ns = t0.elapsed().as_nanos();
    let after = sched.stats();
    let rss_mb = current_rss_mb();
    let vram_mb = current_vram_mb();

    BenchResult {
        batch,
        outputs,
        wall_ns,
        prefill_ns: after.prefill_ns - before.prefill_ns,
        decode_ns: after.decode_ns - before.decode_ns,
        decode_steps: after.decode_steps - before.decode_steps,
        max_concurrent: batch.min(prompts.len()),
        rss_mb,
        vram_mb,
        vram_delta_mb: vram_mb - vram_before_load_mb,
    }
}

/// Регрессия slot-indirection (Phase 6 quality-gate fix): batch shrink.
///
/// Два промпта декодируются одновременно (B=2): один длинный (max_new=12),
/// другой короткий (max_new=3). Короткий завершается первым → batch сжимается
/// (B=2→1). Длинный слот продолжает в одиночном батче. Без slot indirection
/// оставшийся слот читал бы чужой persistent state (batch_idx=0 → чужой slot)
/// и дрейфовал — именно это ломало quality-gate case3.
///
/// Паритет: длинный слот batched == sequential (его собственный B=1 прогон).
/// Короткий слот тоже проверяем для полноты.
#[test]
#[ignore = "требует GGUF + GPU; регрессия slot-indirection (batch shrink)"]
fn real_qwen35_batched_equals_sequential_parity_shrink() {
    let _ = env_logger::try_init();
    let gguf = gguf_path();
    assert!(gguf.exists(), "GGUF не найден: {:?}", gguf);

    let device = accelerator_device();

    // Два разных промпта с разной длиной генерации → batch shrink.
    let prompts: Vec<(Vec<u32>, usize)> = vec![
        (dummy_prompt_ids(42, 8), 12), // длинный
        (dummy_prompt_ids(137, 5), 3), // короткий → раннее завершение → shrink
    ];

    // Одна загрузка весов на весь тест. Старый harness держал B=2 adapter и
    // загружал дополнительные B=1 adapters; параллельный запуск второго ignored
    // test создавал до трёх копий GGUF и вызывал WDDM paging.
    let adapter = Qwen35BatchAdapter::load(&gguf, device, 2).expect("load adapter");
    let vocab = adapter.vocab_size();
    let mut sched = BatchScheduler::new(adapter, 2, u32::MAX, vocab);
    let batched = sched
        .run_with_per_request_max(prompts.clone())
        .expect("batched");

    // Sequential reference на том же adapter: по одному активному prompt.
    let mut seq = Vec::with_capacity(prompts.len());
    for (p, m) in prompts {
        seq.extend(sched.run_with_per_request_max(vec![(p, m)]).expect("seq"));
    }

    assert_eq!(batched.len(), seq.len());
    for (i, (b, s)) in batched.iter().zip(seq.iter()).enumerate() {
        assert_eq!(
            b, s,
            "shrink parity fail на prompt {}: batched {:?} vs seq {:?}",
            i, b, s
        );
    }
    eprintln!("[shrink-parity] длинный(B=2→1) и короткий совпали с sequential: BIT-EXACT OK");
    eprintln!("  длинный слот: {} токенов", batched[0].len());
    eprintln!("  короткий слот: {} токенов", batched[1].len());
    eprintln!(
        "  batch shrunk: {} decode_steps (B=2 + B=1 после shrink), max_concurrent={}",
        sched.stats().decode_steps,
        sched.stats().max_concurrent_decode
    );
}

#[test]
#[ignore = "требует GGUF + GPU; долгий parity+bench прогон"]
fn real_qwen35_batched_equals_sequential_parity() {
    let _ = env_logger::try_init();
    let gguf = gguf_path();
    assert!(gguf.exists(), "GGUF не найден: {:?}", gguf);

    let device = accelerator_device();

    let max_new = 16;
    let prompts: Vec<Vec<u32>> = (0..4)
        .map(|s| dummy_prompt_ids(s * 13 + 1, 6 + (s as usize)))
        .collect();
    let total_tokens = max_new * prompts.len();
    let decode_only_tokens = (max_new - 1) * prompts.len();

    // Одна загрузка модели, capacity=4. B меняется только размером одновременно
    // поданной группы; packed weights и CUDA context не дублируются.
    let vram_before_load_mb = current_vram_mb();
    let adapter = Qwen35BatchAdapter::load(&gguf, device, 4).expect("load adapter");
    let vocab = adapter.vocab_size();
    let mut sched = BatchScheduler::new(adapter, 4, u32::MAX, vocab);
    let mut results = Vec::new();
    for batch in [1usize, 2, 4] {
        results.push(run_bench_loaded(
            &mut sched,
            &prompts,
            max_new,
            batch,
            vram_before_load_mb,
        ));
    }

    let reference = &results[0].outputs;
    for result in &results[1..] {
        assert_eq!(
            result.outputs, *reference,
            "greedy parity B={} vs B=1 нарушена",
            result.batch
        );
    }
    eprintln!("[parity] B=1/2/4 greedy outputs: BIT-EXACT OK");

    // Memory guard: один loaded adapter обязан держать одну копию GGUF.
    // Короткий B=1→B=4 matrix может добавить только small state/KV scratch,
    // не ещё один packed-weight set. Предыдущий harness запускал два ignored
    // tests параллельно и создавал несколько adapters, что уходило в WDDM paging.
    let model_mib = std::fs::metadata(&gguf).expect("GGUF metadata").len() as f64 / 1048576.0;
    let deltas: Vec<f64> = results.iter().map(|r| r.vram_delta_mb).collect();
    if deltas.iter().all(|v| v.is_finite()) {
        let min_delta = deltas.iter().copied().fold(f64::INFINITY, f64::min);
        let max_delta = deltas.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_delta <= model_mib + 1024.0,
            "VRAM delta {max_delta:.1} MiB exceeds GGUF {model_mib:.1} MiB + 1024 MiB; duplicate model allocation likely"
        );
        assert!(
            max_delta - min_delta <= 256.0,
            "VRAM grew {:.1} MiB across short B=1→B=4 matrix; expected <=256 MiB",
            max_delta - min_delta
        );
    }

    let baseline_tps = total_tokens as f64 / (results[0].wall_ns as f64 / 1e9);
    eprintln!(
        "[bench] fair matrix: loads excluded; {} prompts × {} generated tokens",
        prompts.len(),
        max_new
    );
    eprintln!(
        "[bench] B | wall_ms | aggregate_tok/s | per_request_tok/s | decode_only_tok/s | vs_B1 | peak_concurrent | RSS_MB | VRAM_MB | VRAM_delta_MB"
    );
    for r in &results {
        let aggregate_tps = total_tokens as f64 / (r.wall_ns as f64 / 1e9);
        let per_request_tps = aggregate_tps / r.max_concurrent.max(1) as f64;
        let decode_tps = decode_only_tokens as f64 / (r.decode_ns as f64 / 1e9);
        eprintln!(
            "[bench] {} | {:.1} | {:.2} | {:.2} | {:.2} | {:.3}x | {} | {:.1} | {:.1} | {:+.1}",
            r.batch,
            r.wall_ns as f64 / 1e6,
            aggregate_tps,
            per_request_tps,
            decode_tps,
            aggregate_tps / baseline_tps,
            r.max_concurrent,
            r.rss_mb,
            r.vram_mb,
            r.vram_delta_mb,
        );
        eprintln!(
            "[bench-detail] B={} prefill_ms={:.1} decode_ms={:.1} decode_steps={}",
            r.batch,
            r.prefill_ns as f64 / 1e6,
            r.decode_ns as f64 / 1e6,
            r.decode_steps,
        );
    }
}

/// Длинный реальный throughput-замер: 4 разных развёрнутых промпта (ChatML через
/// настоящий токенизатор, ~150–250 токенов каждый), генерация 256 токенов,
/// замер B=1/2/3/4 на реальном тексте. Печатает первый ответ полностью (для
/// оценки качества), throughput по aggregate / per-request / decode-only.
///
/// Loads вне таймера. EOS отключён (u32::MAX) — все слоты досчитывают 256 токенов,
/// чтобы замер был честным по фиксированной нагрузке (без batch shrink).
#[test]
#[ignore = "требует GGUF + GPU; длинный throughput bench B=1/2/3/4 (4×256 токенов)"]
fn real_qwen35_long_throughput_b1234() {
    let _ = env_logger::try_init();
    let gguf = gguf_path();
    assert!(gguf.exists(), "GGUF не найден: {:?}", gguf);

    let tokenizer = tokenizer::load_from_gguf_path(&gguf).expect("load tokenizer");
    let device = accelerator_device();

    // Четыре разных развёрнутых промпта → 4 разных длинных ответа.
    // Просим развёрнутый ответ (несколько пунктов) → модель генерит 100+ токенов.
    let messages_per_case: Vec<Vec<ChatMsg>> = vec![
        vec![
            ChatMsg { role: "system", content: "Ты опытный Rust-инженер. Отвечай подробно и технически точно на русском." },
            ChatMsg { role: "user", content: "Объясни, чем отличается Arc<Mutex<T>> от Arc<RwLock<T>> в Rust. Перечисли 4 ключевых отличия с примерами кода, расскажи когда выбирать каждый из них, и какие есть подводные камни при использовании в async-коде." },
        ],
        vec![
            ChatMsg { role: "system", content: "Ты преподаватель информатики. Отвечай подробно на русском." },
            ChatMsg { role: "user", content: "Опиши алгоритм быстрой сортировки (quicksort). Объясни временную сложность в лучшем, среднем и худшем случае, стратегии выбора опорного элемента (pivot), и как избежать худшего случая. Приведи 3 примера выбора pivot с оценкой." },
        ],
        vec![
            ChatMsg { role: "system", content: "Ты биолог-популяризатор. Отвечай подробно и понятно на русском." },
            ChatMsg { role: "user", content: "Расскажи про процесс клеточного дыхания. Опиши три стадии (гликолиз, цикл Кребса, окислительное фосфорилирование), где каждая происходит в клетке, сколько АТФ получается на каждой стадии, и какую роль играет кислород. Дай итоговую сводку по выходу АТФ." },
        ],
        vec![
            ChatMsg { role: "system", content: "Ты финансовый аналитик. Отвечай подробно на русском." },
            ChatMsg { role: "user", content: "Сравни ETF и взаимные фонды (mutual funds) для долгосрочного инвестора. Перечисли 5 критериев сравнения (комиссии, ликвидность, налоговая эффективность, прозрачность, минимальный порог входа), объясни каждый, и дай рекомендацию для инвестора с горизонтом 10 лет." },
        ],
    ];
    let prompts: Vec<Vec<u32>> = messages_per_case
        .iter()
        .map(|msgs| {
            let text = tokenizer::build_chatml_text(msgs);
            tokenizer::encode_no_think(&tokenizer, &text)
                .unwrap_or_else(|e| panic!("encode prompt: {e}"))
        })
        .collect();

    const MAX_NEW: usize = 256;
    let total_tokens = MAX_NEW * prompts.len();
    let decode_only_tokens = (MAX_NEW - 1) * prompts.len();

    eprintln!(
        "[long-bench] {} реальных промптов, ChatML, prompt_tokens={:?}, max_new={MAX_NEW}",
        prompts.len(),
        prompts.iter().map(|p| p.len()).collect::<Vec<_>>()
    );

    let vram_before_load_mb = current_vram_mb();
    let adapter = Qwen35BatchAdapter::load(&gguf, device, 4).expect("load adapter");
    let vocab = adapter.vocab_size();
    let mut sched = BatchScheduler::new(adapter, 4, u32::MAX, vocab);
    let mut results = Vec::new();
    for batch in [1usize, 2, 3, 4] {
        results.push(run_bench_loaded(
            &mut sched,
            &prompts,
            MAX_NEW,
            batch,
            vram_before_load_mb,
        ));
    }

    // Печатаем первый (полный) ответ из B=1 — для оценки качества текста.
    let first_text = {
        let ids = &results[0].outputs[0];
        let raw = tokenizer::decode_text(&tokenizer, ids).expect("decode first answer");
        tokenizer::strip_thinking(&raw)
    };
    eprintln!(
        "\n[long-bench] пример ответа (case 0, Rust Arc/Mutex vs RwLock, B=1):\n{first_text}"
    );
    eprintln!(
        "[long-bench] длина ответа: {} токенов\n",
        results[0].outputs[0].len()
    );

    let baseline_tps = total_tokens as f64 / (results[0].wall_ns as f64 / 1e9);
    let baseline_decode_tps = decode_only_tokens as f64 / (results[0].decode_ns as f64 / 1e9);
    eprintln!("[long-bench] B | wall_ms | aggregate_tok/s | per_request_tok/s | decode_only_tok/s | vs_B1_agg | vs_B1_decode | peak_concurrent | RSS_MB");
    for r in &results {
        let aggregate_tps = total_tokens as f64 / (r.wall_ns as f64 / 1e9);
        let per_request_tps = aggregate_tps / r.max_concurrent.max(1) as f64;
        let decode_tps = decode_only_tokens as f64 / (r.decode_ns as f64 / 1e9);
        eprintln!(
            "[long-bench] {} | {:.1} | {:.2} | {:.2} | {:.2} | {:.3}x | {:.3}x | {} | {:.1}",
            r.batch,
            r.wall_ns as f64 / 1e6,
            aggregate_tps,
            per_request_tps,
            decode_tps,
            aggregate_tps / baseline_tps,
            decode_tps / baseline_decode_tps,
            r.max_concurrent,
            r.rss_mb,
        );
        eprintln!(
            "[long-bench-detail] B={} prefill_ms={:.1} decode_ms={:.1} decode_steps={} gen_lens={:?}",
            r.batch,
            r.prefill_ns as f64 / 1e6,
            r.decode_ns as f64 / 1e6,
            r.decode_steps,
            r.outputs.iter().map(|o| o.len()).collect::<Vec<_>>()
        );
    }
}
