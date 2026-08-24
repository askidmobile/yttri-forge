# Specification: CUDA-graph prefill

**Date:** 2026-08-25
**Priority:** P1
**Type:** Performance feature (candle-fork + qwen36-server)

## 1. Problem

Host launch overhead префилла: ~4 мс × ~240 launches на чанк 512 = ~1010 мс
host-времени при GPU-работе ~850 мс. Декод уже графирован (`QWEN36_CUDA_GRAPHS=1`,
DecodeGraphState) и снял аналогичный overhead. Префилл — последний большой
участок host-bound выполнения.

Бенчмарк-цель: llama.cpp prefill @8K = 2480 ток/с; наш eager = ~1810 ток/с.

## 2. Goal

Префилл через CUDA graph replay. Целевой wall @8K ≤ 1.6 с (≥2500 ток/с) —
паритет с llama.cpp; амбициозная цель ≥3000 ток/с (преимущество за счёт
fused DeltaNet, которого у них нет).

## 3. Current state

- Decode graphs: `adapter.decode_batch_graphed` + `DecodeGraphState` +
  `paged_kv_cuda` — рабочий референс решения проблем capture (стабильные
  буферы, htod-cache guard, staging ids, logits_out D2D).
- Prefill: `ModelWeights::forward_embeds_hidden_inner` / `forward_inner`
  → `block.forward_prefill_with_rope` × N слоёв; DeltaNet fused CUDA;
  attention FA2; MoE indexed kernels.
- KV single-slot F16 растёт realloc'ом (ломает capture) — нужен статический буфер.
- Инструментация: QWEN36_GPROF=3 фазовые тайминги; бенчи bench_4b/knee/gprof_run.

## 4. User scenarios

### S1: Замер выигрыша (P1)
Prefill @8K на 4B: replay vs eager. **AC:** wall ≤1.7с (≥2400 ток/с) хотя бы в
одной из 4 конфигураций; лог `[pg] replay T=512 hit=1`.

### S2: Корректность (P1)
Графированный вывод ≡ eager. **AC:** logits last-token MAE ≤ 1e-2 vs eager;
генерация 128 токенов связная (golden smoke).

### S3: Откат (P0)
`QWEN36_DISABLE_PREFILL_GRAPHS=1` → точный eager baseline.

### S4: Многоразмерность (P1)
Промпты 4062 (8×512+398) и 1000+хвосты. **AC:** хвостовые чанки либо replay
из кэша, либо корректный eager без деградации общего wall >5%.

## 5. Functional requirements

### Must Have
- **FR-A1**: Две стратегии форм, env-выбираемые:
  - `per-T LRU×8`: capture под каждый уникальный T чанка, eviction LRU;
  - `buckets`: формы {128,256,512,1024}, паддинг входа нулями + causal-маска,
    запрещающая внимание на паддинг (позиции RoPE реальные).
- **FR-A2**: Два режима выхода, env-выбираемые:
  - `last`: head только над последней позицией (logits [1,vocab]);
  - `full`: полный [T,vocab] (совместимость MTP catch-up).
- **FR-A3**: Статический KV-буфер attention слоёв на max_ctx (prefill-graphs
  режим), чтобы указатели были стабильны между replay.
- **FR-A4**: DeltaNet state (ssm/conv) — стабильные batched-буферы (уже есть
  cuda_ctx_batched), сброс на reset_first вне графа.
- **FR-A5**: Кэш графов: key=(T или bucket, slot), LRU 8; инвалидация при seed/reset.
- **FR-A6**: Fallback на eager при любой ошибке capture/replay — БЕЗ отключения
  фичи навсегда (в отличие от decode-graphs), retry на следующем чанке.
- **FR-A7**: Метрики: `[pg] mode=… T=… capture/hit/miss/fallback` per chunk;
  суммарный отчёт в конце запроса.

### Should Have
- **FR-S1**: Матрица 2×2 (стратегия × режим выхода) прогоняется одним скриптом;
  победитель становится default после эксперимента.
- **FR-S2**: 27B: авто-отключение если статический KV не влезает (как decode).

## 6. Non-functional
- Точность: MAE ≤1e-2 vs eager (F16-пути внутри графа идентичны).
- VRAM: статический KV + graph pool; авто-off при нехватке.
- Совместимость: sm_86, CUDA 12.4; Windows WDDM.

## 7. Architecture

```mermaid
flowchart TD
  REQ[PrefillChunk T] --> SEL{"strategy / exit-mode"}
  SEL -->|hit| REP[cuGraphLaunch replay]
  SEL -->|miss| CAP[capture: stage inputs → forward_prefill\n→ head(last|full) → end_capture]
  CAP --> STORE[LRU store]
  REP --> OUT[logits_out D2D→D2H]
  ERR[any error] --> FB[eager forward_prefill]
```

## 8. Out of scope
- Графирование MTP draft/catch-up.
- Multi-slot параллельный prefill в одном графе (b=1 per slot).
- Metal path.

## 9. Spec decisions

| # | Вопрос | Решение | Дата |
|---|--------|---------|------|
| D-201 | Логиты | Оба режима реализовать, выбрать по A/B (заказчик) | 2026-08-25 |
| D-202 | Формы | Обе стратегии реализовать, выбрать по A/B (заказчик) | 2026-08-25 |

## 10. Success criteria

- [ ] 4 конфигурации замерены на 4B @8K+2K, таблица в research.md
- [ ] Лучший режим: prefill ≥2400 ток/с (цель ≥2500)
- [ ] Decode не деградировал; откат-тест точный baseline
- [ ] Golden-smoke генерация связная во всех режимах
