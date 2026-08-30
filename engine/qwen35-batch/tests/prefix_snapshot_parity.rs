//! Этап 1 prefix cache: паритет restore снимка против полного префила.
//!
//! Сценарий из TASK.md:
//!   1. слот 0 — эталон: полный префил P + [tok] непрерывными чанками БЕЗ
//!      restore (состояние живёт в буферах между чанками), логиты последней
//!      позиции (позиция tok);
//!   2. слот 1 — прогнать префикс P, забрать `StateSnapshot` (attention — из
//!      страничного пула, DeltaNet — из single-slot CUDA state);
//!   3. слот 2 — чистый: inject снимка → restore → один чанк [tok];
//!   4. логиты шага 3 сравниваются с эталоном шага 1. Требование — побитовое
//!      совпадение (одинаковые ядра, одинаковый порядок); при расхождении
//!      тест печатает максимум модуля разности и argmax.
//!
//! Гибридность покрыта автоматически: модель содержит и DeltaNet-, и
//! attention-слои, снимок обязан захватить оба типа (assert ниже).
//!
//! Запуск на стенде:
//! ```sh
//! YTTRI_MODEL_DIR=/path/to/model \
//! cargo test -p qwen35-batch --features real-model,cuda \
//!     --test prefix_snapshot_parity --release -- --ignored --nocapture
//! ```
//!
//! Длины префикса: `QWEN36_PCP_SHORT` (дефолт 512), `QWEN36_PCP_LONG`
//! (дефолт 16384). Размер чанка: `QWEN36_PCP_CHUNK` (дефолт 512 — как
//! `prefill_chunk_size`). На paged-пути (CUDA, графы включены — дефолт)
//! длинный префикс exercising пул и постраничные копии снимка.

#![cfg(feature = "real-model")]

use std::path::PathBuf;

use qwen35_batch::model::{BatchModel, PrefillChunk};
use qwen35_batch::real::model_weights::{BlockStateSnap, StateSnapshot};
use qwen35_batch::real::Qwen35BatchAdapter;

fn gguf_path() -> PathBuf {
    std::env::var("QWEN35_TEST_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let model_dir = std::env::var("YTTRI_MODEL_DIR").map(PathBuf::from).unwrap_or_else(|_| {
                PathBuf::from(
                    "/Volumes/Askid Dev/Projects/Yttri/frontend/src-tauri/resources/models/qwen3.5-4b",
                )
            });
            model_dir.join("Qwen3.5-4B-Q4_K_M.gguf")
        })
}

fn accelerator_device() -> candle_core::Device {
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
    {
        // CPU-путь: paged-пул недоступен, снимок attention пойдёт из
        // single-slot кэша (eager-путь). Долгий префил на CPU очень медленный.
        candle_core::Device::Cpu
    }
}

fn env_len(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

/// Детерминированные token IDs, валидные для словаря данной модели.
fn prompt_ids(seed: u32, len: usize, vocab: usize) -> Vec<u32> {
    let span = vocab.saturating_sub(2000).max(1);
    (0..len)
        .map(|i| {
            let x = seed
                .wrapping_add((i as u32).wrapping_mul(2654435761))
                .wrapping_mul(40503);
            (x as usize % span) + 1000
        })
        .map(|v| v.min(vocab - 1) as u32)
        .collect()
}

/// Прогнать токены через `prefill_chunk` чанками; последний чанк — final
/// (там снимок/seed). Возвращает логиты последней позиции.
fn prefill_chunked(
    adapter: &mut Qwen35BatchAdapter,
    slot: usize,
    tokens: &[u32],
) -> anyhow::Result<Vec<f32>> {
    let chunk_size = env_len("QWEN36_PCP_CHUNK", 512);
    let mut logits = Vec::new();
    let mut start = 0usize;
    while start < tokens.len() {
        let end = (start + chunk_size).min(tokens.len());
        let chunk = PrefillChunk {
            slot_idx: slot,
            reset_first: start == 0,
            tokens: tokens[start..end].to_vec(),
            start_pos: start,
            is_final: end == tokens.len(),
        };
        logits = BatchModel::prefill_chunk(adapter, &chunk)?;
        start = end;
    }
    Ok(logits)
}

fn assert_snapshot_coverage(snap: &StateSnapshot, label: &str) {
    let dn = snap
        .blocks
        .iter()
        .filter(|b| matches!(b, BlockStateSnap::DeltaNet(_)))
        .count();
    let attn_some = snap
        .blocks
        .iter()
        .filter(|b| matches!(b, BlockStateSnap::Attention(Some(_))))
        .count();
    let attn_none = snap
        .blocks
        .iter()
        .filter(|b| matches!(b, BlockStateSnap::Attention(None)))
        .count();
    eprintln!(
        "[{label}] снимок: pos={} blocks={} deltanet={dn} attention_с_kv={attn_some} attention_пустых={attn_none}",
        snap.position,
        snap.blocks.len()
    );
    assert!(dn > 0, "{label}: снимок не содержит DeltaNet-слоёв");
    assert!(
        attn_some > 0,
        "{label}: ни один attention-слой не попал в снимок — KV не снят ни из single-slot, ни из страничного пула"
    );
}

fn compare_logits(reference: &[f32], primed: &[f32], label: &str) {
    assert_eq!(
        reference.len(),
        primed.len(),
        "{label}: длина логитов разошлась"
    );
    let mismatch = reference
        .iter()
        .zip(primed)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    if mismatch == 0 {
        eprintln!("[{label}] ПОБИТОВОЕ СОВПАДЕНИЕ: {} логитов", reference.len());
        return;
    }
    let max_abs = reference
        .iter()
        .zip(primed)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.total_cmp(y))
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    let (ar, ap) = (argmax(reference), argmax(primed));
    panic!(
        "{label}: побитового совпадения нет: {mismatch}/{} логитов расходятся, \
         max_abs_diff={max_abs:.3e}, argmax ref={ar} ({:.4}) vs primed={ap} ({:.4})",
        reference.len(),
        reference[ar],
        primed[ap]
    );
}

fn run_parity(prefix_len: usize, label: &str) {
    let gguf = gguf_path();
    assert!(gguf.exists(), "GGUF не найден: {:?} (QWEN35_TEST_GGUF / YTTRI_MODEL_DIR)", gguf);
    let device = accelerator_device();
    let mut adapter = Qwen35BatchAdapter::load(&gguf, device, 3).expect("load Qwen35 adapter");
    let vocab = adapter.vocab_size();
    let eos = adapter.eos();
    eprintln!("[{label}] модель загружена, vocab={vocab}, префикс={prefix_len}");

    let p = prompt_ids(42, prefix_len, vocab);
    // Детерминированный «следующий токен», не EOS.
    let tok = {
        let t = ((p[0] as usize * 7919 + 13) % vocab.saturating_sub(1000)) + 500;
        (if t == eos as usize { t + 1 } else { t }).min(vocab - 1) as u32
    };

    // 1. Эталон: полный префил P + [tok], без restore.
    let mut full = p.clone();
    full.push(tok);
    adapter.reset_slot(0).expect("reset slot 0");
    let logits_ref = prefill_chunked(&mut adapter, 0, &full).expect("эталонный префил");
    eprintln!("[{label}] эталон: префил {} токенов, logits[{}]={:.4}", full.len(), argmax_of(&logits_ref), logits_ref[argmax_of(&logits_ref)]);

    // 2. Префикс P + снимок.
    adapter.reset_slot(1).expect("reset slot 1");
    let _ = prefill_chunked(&mut adapter, 1, &p).expect("префил префикса");
    let snap = adapter
        .slot_snapshot(1)
        .expect("снимок после финального чанка префикса");
    assert_eq!(snap.position, p.len(), "позиция снимка != длине префикса");
    assert_snapshot_coverage(&snap, label);

    // 3. Primed: чистый слот, inject, restore, один следующий токен.
    adapter.reset_slot(2).expect("reset slot 2");
    adapter.inject_slot_snapshot(2, snap);
    let primed_chunk = PrefillChunk {
        slot_idx: 2,
        reset_first: false,
        tokens: vec![tok],
        start_pos: p.len(),
        is_final: true,
    };
    let logits_primed =
        BatchModel::prefill_chunk(&mut adapter, &primed_chunk).expect("primed чанк");

    // 4. Сравнение с эталоном.
    compare_logits(&logits_ref, &logits_primed, label);
}

fn argmax_of(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[test]
#[ignore = "требует GGUF на диске + GPU; короткий префикс (сотни токенов)"]
fn prefix_snapshot_parity_short() {
    run_parity(env_len("QWEN36_PCP_SHORT", 512), "short");
}

#[test]
#[ignore = "требует GGUF на диске + GPU; длинный префикс (десятки тысяч токенов, paged-путь)"]
fn prefix_snapshot_parity_long() {
    run_parity(env_len("QWEN36_PCP_LONG", 16384), "long");
}
