# Batched Decode — CUDA (RTX 3060) Results

Дата: 2026-08-02
Бренч: `feat/qwen35-batching` (HEAD `d44ef32e`)
Хост: `yttri-win` (Windows, RTX 3060 12GB, driver 591.86)
Бэкенд: CUDA 12.4 (cudarc, candle-core/cuda + candle-kernels)
Модель: `D:\Models\lmstudio-community\Qwen3.5-4B-GGUF\Qwen3.5-4B-Q4_K_M.gguf` (2582 MB, Q4_K_M)
Команда-обёртка: `C:\scripts\with_msvc.ps1` (MSVC + CUDA 12.4 pin, `PYO3_NO_PYTHON=1`)

## 1. CUDA kernel parity (unit tests)

`cargo test -p qwen35-batch --features real-model,cuda --lib --release batched -- --nocapture`

- `batched_b1_matches_single` (B=1) — PASS, bit-exact
- `batched_vs_single_parity_b3` (B=3) — PASS, bit-exact
- `batched_vs_single_parity_b4` (B=4) — PASS, bit-exact

Scheduler tests (lib):
- `batched_shrink_parity_early_finish` — PASS
- `batched_more_requests_than_slots_recycles` — PASS
- `batched_equals_sequential_parity` — PASS

Время: 0.33s. CUDA batched kernel (`delta_rule_batched_cuda.rs`) и slot-indirection (slot_ids) подтверждены bit-exact против single-token reference.

## 2. Real-model parity (integration)

`cargo test ... --test real_qwen35_batch --release -- real_qwen35_batched_equals_sequential_parity --ignored --nocapture`

- `real_qwen35_batched_equals_sequential_parity` (B=1/2/4) — PASS, BIT-EXACT OK
- `real_qwen35_batched_equals_sequential_parity_shrink` (B=2→1) — PASS, BIT-EXACT OK
  - длинный слот: 12 токенов, короткий слот: 3 токена
  - batch shrunk: 11 decode_steps (B=2 + B=1 после shrink), max_concurrent=2

Время: 20.05s. End-to-end batched decode с реальной моделью и реальным slot-indirection/shrink — bit-exact на CUDA.

## 3. Quality gate (semantic)

`cargo test ... --test real_qwen35_quality --release -- real_qwen35_quality_gate --ignored --nocapture`

Parity B=1 == B=2 == B=4: текстовых совпадений OK (4 случаев). Все 4 случая семантически корректны:

- case 0 (ru_factual_dialogue): ртуть (Hg), tпл −38.83 °C — ✓ OK
- case 1 (en_translation_to_ru): «Быстрый коричневый лис перепрыгнул через ленивого пса.» — ✓ OK
- case 2 (extraction): «Name: Anna Petrova\nDate: 5 March 2024» — ✓ OK
- case 3 (reasoning_summary): фотосинтез — солнечный свет + вода + CO2 → O2 + глюкоза — ✓ OK

Время: 12.50s.

## 4. Long throughput benchmark (B=1/2/3/4)

`cargo test ... --test real_qwen35_batch --release -- real_qwen35_long_throughput_b1234 --ignored --nocapture`

4 реальных промпта, ChatML, max_new=256, prompt_tokens=[103, 92, 113, 109]. Пример ответа (case 0, Rust Arc/Mutex vs RwLock) — осмысленный технический ответ 256 токенов.

| B | wall_ms | aggregate tok/s | per_request tok/s | decode_only tok/s | vs B=1 agg | vs B=1 decode | peak_concurrent | RSS_MB |
|---|---------|-----------------|-------------------|-------------------|------------|---------------|-----------------|--------|
| 1 | 18951.7 | 54.03 | 54.03 | 58.16 | 1.000x | 1.000x | 1 | 3193.6 |
| 2 | 14089.4 | 72.68 | 36.34 | 80.70 | 1.345x | 1.387x | 2 | 3292.5 |
| 3 | 13733.1 | 74.56 | 24.85 | 82.78 | 1.380x | 1.423x | 3 | 3392.7 |
| 4 | 11560.3 | 88.58 | 22.14 | 100.51 | 1.639x | 1.728x | 4 | 3494.4 |

decode_steps: B=1 → 1020, B=2 → 510, B=3 → 510, B=4 → 255 (fair scheduling, loads excluded).

Время: 65.53s.

### Анализ throughput

- B=4 vs B=1: aggregate ×1.639, decode-only ×1.728 — batching даёт существенный прирост на RTX 3060.
- Per-request throughput падает с ростом B (54.03 → 22.14), что ожидаемо: VRAM/compute делится между слотами.
- VRAM delta от B=1 к B=4: RSS +300 MB (state + KV batch), VRAM стабильно ~5929 MB (Q4_K_M weights доминируют, batch-state мал).
- Decode-only на B=4 достигает 100.51 tok/s против 58.16 у B=1 — batching эффективно утилизирует GPU.

## Сравнение бэкендов (Qwen3.5-4B Q4_K_M, B=4 long bench, decode-only tok/s)

| Бэкенд | Хост | B=1 decode | B=4 decode | × vs B=1 |
|--------|------|-----------|-----------|----------|
| Metal | M4 Pro | 42.07 (agg) | 55.26 (agg) | ×1.314 |
| CPU | M4 Pro | 11.0 | 26.67 | ×2.517 |
| CUDA | RTX 3060 | 58.16 | 100.51 | ×1.728 |

CUDA даёт максимальный decode-only throughput на B=4 (100.51 tok/s) и хороший scaling (×1.728). Metal competitive на B=1 (42 agg), CPU — fallback с сильным batch-scaling (×2.517) но низкой абсолютной скоростью.

## Итог

CUDA batched decode (DeltaNet + Attention) на RTX 3060 полностью валидирован:
- kernel parity bit-exact (B=1/3/4)
- real-model parity bit-exact (B=1/2/4 + shrink)
- quality gate 4/4 семантически корректны, текстовое совпадение B=1==B=2==B=4
- throughput: B=4 decode-only 100.51 tok/s, ×1.728 vs B=1

Все три бэкенда (Metal/CPU/CUDA) теперь валидированы end-to-end на реальной модели.