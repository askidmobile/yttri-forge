# True batched DeltaNet decode — Metal результаты (M4 Pro)

Дата: 2026-08-02
Машина: Apple M4 Pro, macOS, unified memory
Backend: Metal (`--features "real-model metal"`)
Модель: Qwen3.5-4B-Q4_K_M (`/Volumes/Askid Dev/Projects/Yttri/frontend/src-tauri/resources/models/qwen3.5-4b`)
Ветка: `feat/qwen35-batching`
Коммит фикса slot-indirection: `8a3b1624`

## 1. Quality gate (`real_qwen35_quality_gate`) — PASS

4 реальных ChatML-промпта через токенизатор из GGUF, B=1/B=2/B=4, MAX_NEW=72.

Проверки: greedy parity (B=2/B=4 == B=1 по тексту), семантика (непустой ответ, не эхо, не зацикливание, ожидаемые ключевые слова).

| case | промпт (последний user) | ответ модели | статус |
|---|---|---|---|
| ru_factual_dialogue | Какой металл жидкий при комнатной температуре? | «При комнатной температуре жидким является **ртуть** (химический символ **Hg**). Это единственное природное металл…» | ✓ OK |
| en_translation_to_ru | The quick brown fox jumps over the lazy dog. | «Быстрый коричневый лис перепрыгнул через ленивого пса.» | ✓ OK |
| extraction | Extract name+date… Anna Petrova … 5 March 2024 | «Name: Anna Petrova\nDate: 5 March 2024» | ✓ OK |
| reasoning_summary | Summarize photosynthesis… | «Фотосинтез — это процесс, при котором растения используют солнечный свет, воду и углекислый газ для производства кислорода и глюкозы.» | ✓ OK |

Parity: B=1 == B=2 == B=4 (текстовых совпадений OK, 4 случаев).
Время: 13.44s (включая 3 загрузки весов: B=1/B=2/B=4).

## 2. Parity + benchmark (`real_qwen35_batched_equals_sequential_parity`) — PASS

4 dummy prompts × 16 generated tokens, greedy, bit-exact.

| B | wall_ms | aggregate tok/s | decode-only tok/s | vs B=1 |
|---|---|---|---|---|
| 1 | 1923.1 | 33.28 | 44.55 | 1.000× |
| 2 | 1670.6 | 38.31 | 49.17 | 1.151× |
| 4 | 1324.6 | 48.32 | 62.46 | 1.452× |

## 3. Shrink parity (`real_qwen35_batched_equals_sequential_parity_shrink`) — PASS

Регрессия slot-indirection: длинный слот (12 токенов) + короткий (3 токена), B=2→1 после раннего завершения короткого.
Длинный слот bit-exact vs sequential. 11 decode_steps, max_concurrent=2.

## 4. Длинный throughput (`real_qwen35_long_throughput_b1234`) — PASS

4 реальных развёрнутых промпта (Rust Arc/Mutex vs RwLock, quicksort, клеточное дыхание, ETF vs mutual funds),
ChatML, prompt_tokens=[103, 92, 113, 109], генерация 256 токенов, EOS отключён (u32::MAX).

### Пример ответа (case 0, Rust Arc/Mutex vs RwLock, B=1)

> Это отличный вопрос, затрагивающий одну из самых частых ошибок при работе с синхронизацией в Rust, особенно в контексте асинхронного программирования.
> Короткий ответ: **`Arc<Mutex<T>>` блокирует всех, кто пытается получить доступ к данным одновременно**, в то время как **`Arc<RwLock<T>>` позволяет нескольким читателям работать параллельно**, блокируя только одного писателя.
> Вот подробный разбор четырех ключевых отличий, примеры кода, рекомендации по выбору и подводные камни в async-коде.
> ---
> ### 4 ключевых отличия
> #### 1. Модель доступа (Readers vs Writers)
> * **`Mutex<T>`**: Строго мультиэксклюзивный. Если один поток (или асинхронная задача) захватывает замок, **никто** не может получить доступ к данным, пока он не будет освобожден…
> * **`RwLock<T>`**: Поддерживает мульти-чтение и мульти-запись (но не одновременно)…

(Ответ прошёл 256 токенов, не оборвался на EOS — развёрнутое техническое объяснение на русском с заголовками и примерами.)

### Throughput B=1/2/3/4

| B | wall_ms | aggregate tok/s | per-request tok/s | decode-only tok/s | vs B=1 (agg) | vs B=1 (decode) | peak | RSS MB |
|---|---|---|---|---|---|---|---|---|
| 1 | 24341 | 42.07 | 42.07 | 43.97 | 1.000× | 1.000× | 1 | 1299 |
| 2 | 22171 | 46.19 | 23.09 | 48.28 | 1.098× | 1.098× | 2 | 1126 |
| 3 | 20884 | 49.03 | 16.34 | 51.47 | 1.166× | 1.171× | 3 | 1242 |
| 4 | 18531 | 55.26 | 13.81 | 58.31 | 1.314× | 1.326× | 4 | 1307 |

decode_steps: B=1→1020, B=2→510, B=3→510, B=4→255. gen_lens=[256,256,256,256] (все слоты досчитали).
prefill ~910–1020ms (не батчится, per-slot).

### Интерпретация

- Aggregate throughput растёт: 42→46→49→55 tok/s. **B=4 = ×1.314 aggregate vs B=1** на реальном тексте.
- Per-request throughput падает (42→23→16→14) — ожидаемый tradeoff continuous batching: выигрыш в aggregate, проигрыш в latency.
- Decode-only ×1.326 — чуть больший выигрыш, так как prefill не батчится.
- B=3 < B=4 (×1.17 vs ×1.31): 3 слота не полностью насыщают bandwidth + Q4_K fast path (кратный 32) не активен для B=3.

## Команды запуска

```sh
YTTRI_MODEL_DIR="/Volumes/Askid Dev/Projects/Yttri/frontend/src-tauri/resources/models/qwen3.5-4b" \
cargo test -p qwen35-batch --features "real-model metal" --test real_qwen35_quality --release -- --ignored --nocapture real_qwen35_quality_gate

YTTRI_MODEL_DIR=... cargo test -p qwen35-batch --features "real-model metal" --test real_qwen35_batch --release -- --ignored --nocapture real_qwen35_batched_equals_sequential_parity

YTTRI_MODEL_DIR=... cargo test -p qwen35-batch --features "real-model metal" --test real_qwen35_batch --release -- --ignored --nocapture real_qwen35_batched_equals_sequential_parity_shrink

YTTRI_MODEL_DIR=... cargo test -p qwen35-batch --features "real-model metal" --test real_qwen35_batch --release -- --ignored --nocapture real_qwen35_long_throughput_b1234
```