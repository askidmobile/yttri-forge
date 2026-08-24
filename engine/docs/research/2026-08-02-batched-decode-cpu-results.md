# Batched DeltaNet decode — CPU fallback validation (M4 Pro)

Дата: 2026-08-02
Машина: M4 Pro (macOS), `Device::Cpu` (без Metal/CUDA batched ctx)
Ветка: `feat/qwen35-batching`
Модель: Qwen3.5-4B-Q4_K_M.gguf

## Контекст

Phase 6 validated true batched decode on **Metal** (GPU). Перед CUDA-валидацией
пользователь запросил протестировать **CPU путь** (Candle без Metal) — т.к.
production может работать и без GPU backend.

Проблема: `DeltaNetLayer::forward_decode_batch` имел только Metal + CUDA batched
пути и bail'ил без batched GPU контекста (`metal_ctx_batched`/`cuda_ctx_batched`).
Single-token `forward` имел CPU fallback, но batched — нет.

## Реализация CPU batched fallback

- `DeltaNetLayer.cpu_state_batched: Vec<DeltaNetState>` (len=DECODE_BATCH_CAPACITY=4) —
  per-slot CPU recurrent state (conv_buf + ssm_state), изолированный per slot.
- `forward_decode_batch`: после Metal/CUDA batched путей → CPU fallback:
  batched projections (веса один раз на B слотов) → transfer на CPU → per-slot loop
  (conv1d_step + L2 norm + delta_rule_step + group RMS norm + SiLU gate) через
  `cpu_state_batched[slots[bidx]]` → transfer обратно → batched ssm_out.
  Math per slot идентичен single-token `forward` CPU path.
- `seed_slot_batched`: CPU restore `cpu_state_batched[slot].restore_from(snap)`.
- `clear_state_batched`: зануляет все `cpu_state_batched` слоты.
- Тесты: env var `YTTRI_FORCE_CPU=1` → `Device::Cpu` (metal feature скомпилирован,
  но `metal_ctx`/`metal_ctx_batched` = None → CPU fallback).

## Результаты тестов

### Parity (bit-exact batched vs sequential)

```
[shrink-parity] длинный(B=2→1) и короткий совпали с sequential: BIT-EXACT OK
  длинный слот: 12 токенов
  короткий слот: 3 токенов
  batch shrunk: 11 decode_steps (B=2 + B=1 после shrink), max_concurrent=2
test real_qwen35_batched_equals_sequential_parity_shrink ... ok

[parity] B=1/2/4 greedy outputs: BIT-EXACT OK
test real_qwen35_batched_equals_sequential_parity ... ok
```

### Throughput (4 dummy prompts × 16 generated tokens)

| B | wall_ms | aggregate tok/s | per-request tok/s | decode-only tok/s | vs B=1 (agg) |
|---|---------|-----------------|-------------------|-------------------|--------------|
| 1 | 8226.3  | 7.78            | 7.78              | 11.00             | 1.000×       |
| 2 | 4981.5  | 12.85           | 6.42              | 25.90             | 1.651×       |
| 4 | 3268.0  | 19.58           | 4.90              | 26.67             | 2.517×       |

CPU decode-only tok/s растёт с B (11→26.67), aggregate ×2.517 на B=4 —
batched decode даёт сильный throughput выигрыш на CPU (веса читаются один раз
на B слотов + per-slot recurrent math на CPU cores).

### Quality gate (semantic, 4 ChatML cases)

```
[parity] B=1 == B=2 == B=4: текстовых совпадений OK (4 случаев)

case 0 — ru_factual_dialogue: "ртуть (Hg)" ✓ OK
case 1 — en_translation_to_ru: "Быстрый коричневый лис перепрыгнул через ленивого пса." ✓ OK
case 2 — extraction: "Name: Anna Petrova\nDate: 5 March 2024" ✓ OK
case 3 — reasoning_summary: "Фотосинтез — это процесс, при котором растения используют солнечный свет, воду и углекислый газ для производства кислорода и глюкозы." ✓ OK

test real_qwen35_quality_gate ... ok
```

## Команды запуска

```sh
# Parity (B=1/2/4 + shrink)
YTTRI_FORCE_CPU=1 YTTRI_MODEL_DIR="/Volumes/Askid Dev/Projects/Yttri/frontend/src-tauri/resources/models/qwen3.5-4b" \
cargo test -p qwen35-batch --features "real-model metal" \
    --test real_qwen35_batch --release -- \
    real_qwen35_batched_equals_sequential_parity --ignored --nocapture

# Quality gate
YTTRI_FORCE_CPU=1 YTTRI_MODEL_DIR="/Volumes/Askid Dev/Projects/Yttri/frontend/src-tauri/resources/models/qwen3.5-4b" \
cargo test -p qwen35-batch --features "real-model metal" \
    --test real_qwen35_quality --release -- \
    real_qwen35_quality_gate --ignored --nocapture
```

## Вывод

CPU batched decode fallback реализован и валидирован: bit-exact parity
B=1/2/4 + shrink, semantic quality PASS, aggregate ×2.517 на B=4. Модель
Qwen3.5-4B работает на Candle без Metal/CUDA backend через true batched decode.