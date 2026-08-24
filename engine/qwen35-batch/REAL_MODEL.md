# REAL_MODEL — интеграция реальной Qwen3.5-4B в qwen35-batch

Документ описывает, как подключить реальную GGUF-модель Qwen3.5-4B к
scheduler'у за фичей `real-model`. Scheduler (`BatchScheduler<M: BatchModel>`)
уже протестирован на детерминированном mock'е (`MockRecurrentModel`) —
8 unit + 4 integration-теста зелёные. Дальше — только реализация `BatchModel`
над существующим кодом `quantized_qwen35.rs` из Yttri.

## 0. Статус mock-уровня (готово)

- `src/slot.rs` — lifecycle слота IDLE→PREFILL→DECODE→FINISHED, `current_token`,
  `next_pos`, `push_token`, `full_sequence`. Тесты `slot_lifecycle`,
  `slot_eos_terminates` — ok.
- `src/model.rs` — `BatchModel` trait (`prefill_chunk` → логиты последнего
  токена; `decode_batch` → логиты per-slot; `reset_slot`), `GreedySampler`,
  `MockRecurrentModel`. Паритет: output слота — чистая функция prompt'а
  (state сбрасывается в 0 при admit, эволюционирует только от своих токенов) →
  batched == sequential. Тесты `mock_output_is_pure_function_of_prompt`,
  `mock_concurrent_slots_isolated` — ok.
- `src/scheduler.rs` — `BatchScheduler::step` (prefill phase: один чанк для
  одного Prefilling-слота + сэмпл первого токена из prefill-логитов; decode
  phase: батч всех Decoding-слотов), `run_with_collection` (сбор outputs из
  Finished ДО reset, пере-admit из очереди), `sequential_reference` (baseline).
  `SchedulerStats` (decode_steps, total_decode_tokens, max_concurrent_decode,
  prefill_chunks, wall/prefill/decode_ns, `decode_aggregate_tps`). Тесты
  `batched_equals_sequential_parity`, `batched_more_requests_than_slots_recycles`,
  `stats_recorded`, `first_token_comes_from_prefill_logits` — ok.
- `tests/scheduler_parity.rs` — `parity_batched_vs_sequential_varlen`,
  `max_concurrent_reaches_num_slots`, `recycling_preserves_output_mapping`,
  `eos_terminates_early_when_reached` — ok.

`cargo test -p qwen35-batch` → **12 passed, 0 failed**.

## 0b. Статус real-model (после портирования)

Код модели скопирован в `src/real/` и **компилируется** за фичей `real-model`
(`cargo build -p qwen35-batch --features real-model --tests` — OK, только
dead-code warnings). Адаптер `Qwen35BatchAdapter` (`src/real/adapter.rs`)
реализует `BatchModel` через **time-multiplexing** (см. §0c).

### 0c. Архитектурная стена (главный вывод портирования)

**True batched decode на текущем Candle НЕВОЗМОЖЕН** без переписывания Metal-ядер:
- `DeltaNetLayer::forward` (model_weights.rs:1469): `debug_assert_eq!(seq_len, 1)`
  и `debug_assert_eq!(b_sz, 1)` (1711) — decode принимает строго 1 токен.
- `dispatch_delta_rule` (delta_rule_metal.rs): grid per v-head, F32 per-token
  scratch (`DeltaNetTempBuffers` — фиксированные [channels]/[n_v*hd] размеры);
  batch-оси в 4 ядрах (conv1d_prep, l2_norm_expand, delta_rule, norm_gate) нет.
- `dispatch_gdn_fused` (prefill): batch-aware (n_tokens), но только для prefill
  (seq_len>1); decode-путь через него не идёт.
- Q4K fast-path (`dispatch_q4k_matmul:193`): `n%64==0 && m%32==0` — B=4 даёт
  m=4 → fallback V1 (медленнее, но корректно).

**Реализованный путь — time-multiplexing одной weight copy**:
- Веса — одни (`ModelWeights` shared). Per-slot recurrent state (GDN ssm/conv)
  и KV-cache — в `Vec<Option<StateSnapshot>>`, свопаются через
  `ModelWeights::restore_state` / `snapshot_state` (T-274 prompt-cache API).
- Каждый decode-шаг: restore state слота → `forward([tok], pos)` → snapshot.
- Parity — бит-точная (каждый слот изолирован своим state; тот же forward-путь).
- **Цена**: snapshot ~114 МБ/слот → своп ~B×228 МБ/decode-шаг. Ожидание:
  aggregate SLOWER чем sequential (штраф snapshot/restore), что подтверждает
  D-12 — true continuous-batching (×1.15-1.35) требует batch-axis в Metal-ядрах.

**Вывод для решения**: continuous batching на Candle для Qwen3.5 = major kernel
rewrite (4 delta_rule ядра + fused GDN + Q4K fast-path под m=B). Объём работы
сопоставим с тем, что MLX ecosystem уже сделал (BatchedHybridCache). Альтернатива —
MLX sidecar (Yttri уже использует mlx-swift), где hybrid batching shipped.

### 0d. Результаты реального прогона (2026-08-01)

Оба `#[ignore]`-теста выполнены на реальном GGUF Qwen3.5-4B-Q4_K_M (2.7 ГБ),
Metal, release:

- `real_qwen35_load_and_single_forward` — **ok**. Загрузка zero-copy
  (~2583 МБ в MTLResidencySet), vocab=248320 (shape[0] token_embd.weight,
  padding до 256 для квантизации), eos=248046. Prefill 8 токенов → logits
  len=248320 OK.
- `real_qwen35_batched_equals_sequential_parity` — **ok, BIT-EXACT**.
  Greedy parity B=2 и B=4 vs B=1 — идентичные последовательности токенов
  (time-multiplexed snapshot/restore изолирует state каждого слота).

**Честная bench-матрица** (загрузка модели ВНЕ таймера; 4 prompt'а ×
max_new=16 токенов; greedy, eos=MAX):

| B | wall_ms | aggregate tok/s | per-request tok/s | decode-only tok/s | vs B=1 | peak_concurrent | RSS МБ |
|---|---------|----------------|------------------|-------------------|--------|-----------------|--------|
| 1 | 2145.6  | 29.83          | 29.83            | 34.54             | 1.000× | 1               | 1140.7 |
| 2 | 2012.6  | 31.80          | 15.90            | 36.22             | 1.066× | 2               | 825.2  |
| 4 | 2012.8  | 31.80          | 7.95             | 36.06             | 1.066× | 4               | 902.8  |

Detail: prefill_ms ≈341–401, decode_ms ≈1656–1737, decode_steps = 60/30/15
(B=1/2/4 — 4 промпта × 15 decode-шагов, распределённых по слотам).

**Интерпретация (решающий вывод vs D-12)**:
- **Aggregate tok/s практически плоский**: 29.8 → 31.8 → 31.8 (**×1.07**
  максимум). Это далекó от ×1.35 llama.cpp при B=8. Наблюдаемое ×1.07 —
  экономия на 4 prefill'ах (они не масштабируются с B), а не на decode.
- **Decode-only tok/s плоский**: 34.5 / 36.2 / 36.1 — 4 слота не амортизируют
  чтение Q4-весов: каждый decode-шаг B=4 = 4 отдельных forward + restore/
  snapshot по per-slot state. Per-step время: B=1 28.9 мс, B=4 110.9 мс
  (×3.84 — практически линейно по槽ам).
- **Per-request tok/s деградирует линейно**: 29.8 → 15.9 → 7.95 — каждый
  запрос получает ~1/B от aggregate. True parallelism отсутствует.
- **Parity бит-точный** (time-multiplexed изоляция работает корректно).

**Причина стена**: Copper decode path без batch-оси (§0c). Time-multiplexing
не даёт batching-выгоды, а лишь interleaving + штраф snapshot/restore
(~114 МБ/слот своп). Это эмпирически подтверждает D-12: на Candle true
continuous batching (×1.15–1.35) требует batch-axis в 4 delta_rule Metal-
ядрах + fused GDN + Q4K fast-path. Без них multi-slot = sequential с
overhead'ом, не parallelism.

## 1. Источник кода модели

Код живёт в приложении Yttri (не в candle-fork):

- `frontend/src-tauri/src/modules/ai/local_llm/quantized_qwen35.rs` (4854 строки) —
  `ModelWeights::from_gguf_zero_copy`, `forward(x [1,seq], index_pos)`,
  DeltaNet/Attention blocks, embedding, lm_head, RoPE F32.
- `frontend/src-tauri/src/modules/ai/local_llm/metal/delta_rule_metal.rs` (604) —
  `dispatch_delta_rule(metal_device, pipelines, layer_state, temp, params, qkv_t, z_t, beta_t, alpha_t)`,
  4 ядра (conv1d_prep, l2_norm_expand, delta_rule, norm_gate), F32 буферы,
  zero-wrap `temp.gated_output` (SHARED scratch).
- `metal/gated_delta_net_fused.rs` (386) + `.metal` (399) — fused GDN prefill
  (`dispatch_gdn_fused`).
- `frontend/src-tauri/src/modules/ai/local_llm/metal/mod.rs` — re-export'ы.
- `frontend/src-tauri/src/modules/ml_common/metal_utils.rs` — env-сетап Metal,
  `with_autoreleasepool`, probe/teardown.
- GGUF: `/Volumes/Askid Dev/Projects/Yttri/frontend/src-tauri/resources/models/qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf` (+ mmproj).
- candle-fork патчится через `[patch]` в `frontend/src-tauri/Cargo.toml:478-483`.

## 2. Что перенести в crate

Скопировать в `qwen35-batch/src/real/` (новый модуль за `#[cfg(feature="real-model")]`):

- `quantized_qwen35.rs` → `real/model_weights.rs`
- `metal/delta_rule_metal.rs` → `real/delta_rule_metal.rs`
- `metal/gated_delta_net_fused.rs` + `.metal` → `real/gated_delta_net_fused.rs`(+`.metal`)
- `metal/mod.rs` → `real/metal_mod.rs`
- минимальный `metal_utils` (env-сетап + autoreleasepool) → `real/metal_utils.rs`

### Зависимости для `real-model` (уже в Cargo.toml)

`memmap2` 0.9.3, `tokenizers` 0.22.0; candle-core (path, `metal`), candle-nn,
candle-metal-kernels (path); macOS-only objc2 0.6.3 / objc2-metal 0.3.2 /
objc2-foundation 0.3.2. Проверить, что `candle-metal-kernels` экспортирует
Q4K-ядра (V1/V2/V3/V4) и SDPA — иначе добавить feature.

## 3. Что вырезать / переписать (Tauri-зависимости)

1. `crate::modules::ml_common::metal_utils::with_autoreleasepool` →
   локальный `objc2::rc::autoreleasepool` (уже objc2 в deps). См. metal_utils.rs
   в Yttri — там autoreleasepool + `CANDLE_METAL_COMPUTE_PER_BUFFER=15`,
   `CANDLE_METAL_COMMAND_POOL_SIZE=1` перед созданием device.
2. `super::metal` / `super::cuda` cfg-блоки: `cuda_backend` cfg = false →
   безопасно оставить/вырезать. Метал-путь — единственный целевой.
3. `dispatch_q4k_matmul` (app lines 165–223): V2Opt/V3/V4 fast-path требует
   `n % 64 == 0 && m % 32 == 0` (m = second-to-last dim xs). Для batched decode
   `xs = [1, B, K]` → m = B. При B < 32 → fallback на V1. **Решение ниже (§6).**
4. Логгирование/трейсинг Yttri (`tracing::`, `crate::modules::...`) → `log::`.

## 4. Адаптер `BatchModel` — `real/adapter.rs`

```rust,ignore,path=null,start=null
pub struct Qwen35BatchAdapter {
    weights: ModelWeights,            // одна weight copy
    device: Device,                   // Metal
    // per-slot state × N
    gdn_state: Vec<Vec<DeltaNetMetalState>>, // [slot][layer]
    kv_cache: Vec<Vec<Option<(Tensor, Tensor)>>>, // [slot][attn_layer]
    kv_len: Vec<usize>,               // [slot]
    slot_pos: Vec<usize>,             // [slot] index_pos для RoPE
    vocab: usize,
    eos: u32,
    pad: usize,                       // m-паддинг для Q4K fast-path (§6)
}
```

`DeltaNetMetalState { ssm_state, conv_state, conv_weights, dt_bias, ssm_a, norm_weight }`
— все `Arc<Buffer>`; per-slot = own `ssm_state`/`conv_state` (нули) + shared
`Arc`-clones весов. Фабрика per-slot state:
`create_layer_metal_state(metal_device, params, conv_weights_data, dt_bias_data, ssm_a_data, norm_weight_data)`
— вызвать N×24 раз при init.

### `prefill_chunk(chunk)` (§7)

- `reset_first` → обнулить `gdn_state[slot]`, `kv_cache[slot]`, `kv_len[slot]=0`,
  `slot_pos[slot]=chunk.start_pos`.
- reuse `ModelWeights::forward_prefill` путь (fused GDN `dispatch_gdn_fused` с
  per-slot `layer_state`, attention с per-slot KV). Возвращает logits последнего
  токена чанка. Для `PREFILL_CHUNK == usize::MAX` (весь prompt) — один вызов.
- **Критично**: НЕ повторно forward'ить последний токен prompt'а — scheduler
  уже сэмплирует первый токен из этих logits (см. тест
  `first_token_comes_from_prefill_logits`).

### `decode_batch(batch)` (§7–8)

- Собрать input-эмбеддинги всех активных слотов → `[1, B, hidden]`.
- **Batched weight matmuls** (embedding, RmsNorm, residual, QMatMul, MLP,
  lm_head) через padded `[1, pad, *]` layout (§6).
- **Per-slot stateful ops**:
  - DeltaNet (24 слоя): сузить строки слота из padded-буфера →
    `dispatch_delta_rule(device, pipelines, &mut gdn_state[slot][layer], ...)` →
    `slice_set` output-row обратно в padded-буфер. **SHARED scratch
    `temp.gated_output`**: consumer должен забрать output-row ДО следующего
    slot'а/layer'а — Metal command ordering внутри одного command stream
    гарантирует это; синхронизировать `command_buffer.wait_until_completed()`
    между layer'ами при необходимости.
  - Attention (8 слоёв): per-slot KV append + `candle_nn::ops::sdpa` (handles
    GQA, F16 — как app lines 2371–2376), затем batched gate+wo.
- Вернуть logits per-item (`Vec<Vec<f32>>` в порядке `batch.items`).

### `reset_slot(idx)`

Обнулить `gdn_state[idx]` (ssm/conv → zeros), `kv_cache[idx] → None`,
`kv_len[idx]=0`, `slot_pos[idx]=0`. Scheduler вызывает это в admit'е
через `reset_first` первого чанка.

## 5. Размерности (4B)

- 32 слоя: 24 DeltaNet + 8 Attention (`full_attention_interval=4`,
  `(idx+1)%4!=0` → DeltaNet).
- hidden 2560; DeltaNet: n_v_heads=32, n_k_heads=16, head_k_dim=128,
  head_v_dim=128, key_dim=2048, value_dim=4096, channels=8192, conv_kernel=4.
- Attention: n_head=8, n_kv_head=4, head_dim=256, rope_dim=64 (partial 25%),
  RoPE F32 cos/sin.
- KV-cache F16 per attn-слой ~4KB/token; GDN state fp32 ~2.1MB/слой.
- Per-slot overhead: 4 слота ≈ +200MB поверх одной weight copy.
- `context_length = min(GGUF, YTTRI_CONTEXT_LIMIT=81920)`.

## 6. Q4K fast-path и паддинг (критично для производительности)

`dispatch_q4k_matmul` (app lines 185–218): V2Opt/V3/V4 fast-path требует
`n % 64 == 0 && m % 32 == 0` (m = second-to-last dim xs). Для batched decode
reshape `[1, B, K]` → m = B.

- **B=1/2/4 < 32** → fallback V1 `qmatmul.forward` (медленнее на batched, но
  корректно). Паритет B=1 vs single-stream ожидается бит-точным (те же веса,
  тот же V1-путь).
- **Паддинг до m=32**: добить строки token=0 (garbage outputs discard'ятся;
  `RmsNorm(0)=0`, `SiLU(0)=0` безопасны) → активирует V2 fast-path. Сделать
  `pad` конфигурируемым (`Qwen35BatchAdapter::new(..., pad: usize)`).
- **Бенч**: V1-m4 vs V2-padded-32 при B=4 — меряем aggregate tok/s; ожидание
  (bandwidth-bound Pro-class) ×1.15–1.35 у V2-padded над sequential.

## 7. Порядок реализации (рекомендованный)

1. Скопировать файлы модели в `real/`, вырезать Tauri-deps (§3), добиться
   `cargo build -p qwen35-batch --features real-model` (без адаптера).
2. Loader: mmap GGUF → `gguf_file::Content::read` → arch check "qwen35" →
   tokenizer из `tokenizer.huggingface.json` metadata (`tokenizers::Tokenizer::from_bytes`,
   fallback `build_tokenizer_from_ggml`) → `ModelWeights::from_gguf_zero_copy`
   на Metal → EOS из `tokenizer.ggml.eos_token_id` (fallback 151645).
   (app engine.rs:1541–1658.)
3. `Qwen35BatchAdapter::new(device, gguf_path, num_slots, pad)` — weights + N×
   per-slot state (§4).
4. `impl BatchModel`: `prefill_chunk` (single-slot forward_prefill reuse) →
   тест B=1 parity vs single-stream (бит-точность, V1 fallback).
5. `decode_batch`: batched matmuls + per-slot DeltaNet/Attention → тест B=2/4
   parity (prefix-match rate из-за m-паддинга) + bench.
6. Заполнить `tests/real_qwen35_batch.rs` (`real_qwen35_batch_parity_and_bench`).

## 8. Контракт SHARED scratch (опасное место)

`dispatch_delta_rule` пишет в `temp.gated_output` — SHARED scratch. Per-slot
вызовы в одном layer'е должны consume output-row до следующего slot'а.
Metal command ordering внутри одного command stream гарантирует порядок, НО:
- если слоты идут в разных command buffers — нужен явный fence/wait;
- между layer'ами — `command_buffer.wait_until_completed()` или
  вынести output-row в per-slot Tensor до следующего dispatch.

В прототипе держать один command stream per decode-шаг; мерять корректность
parity — если state протекает, паддинг/порядок — первый подозреваемый.

## 9. Запуск реальной модели

```sh
YTTRI_MODEL_DIR=/Volumes/Askid\ Dev/Projects/Yttri/frontend/src-tauri/resources/models/qwen3.5-4b \
cargo test -p qwen35-batch --features real-model \
    --test real_qwen35_batch --release -- --ignored --nocapture
```

Env Metal (перед запуском, см. metal_utils.rs):
`CANDLE_METAL_COMPUTE_PER_BUFFER=15`, `CANDLE_METAL_COMMAND_POOL_SIZE=1`,
`YTTRI_Q4K_KERNEL=v2` (default).

## 10. Что считаем (критерии для отчёта vs D-12)

- **Aggregate tok/s** при B=1/2/4 decode (один loaded model, 4 потока).
- **Per-request tok/s** (справедливость разделения bandwidth).
- **RSS** (одна weight copy + per-slot state).
- **Parity**: B=1 batched == single-stream (бит-точность, V1); B=4 vs B=1 —
  prefix-match rate (допускаем argmax-расхождения из-за V2-padded m=32 vs V1 m=1).
- **Sanity**: translate-стиль проверка качества генерации (greedy, несколько
  prompt'ов).

Сравнение с D-12 (llama.cpp batched bench в wiki): S_TG B=1 43.9 → B=8 59.2 t/s
(×1.35) на M4 Pro. Гипотеза для Candle: ×1.15–1.35 aggregate при B=4 на
Pro-class (bandwidth-bound). Прототип должен это подтвердить/опровергнуть.
