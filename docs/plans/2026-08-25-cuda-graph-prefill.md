# Plan: CUDA-graph prefill

**Date:** 2026-08-25
**Status:** 📝 Planning
**Priority:** P1
**Specification:** [docs/specs/2026-08-25-cuda-graph-prefill.md](../specs/2026-08-25-cuda-graph-prefill.md)

## Goal

Префилл через cuGraphLaunch: убрать ~1010 мс host-launch overhead на чанк 512.
Цель @8K ≥2400 ток/с (llama.cpp parity), амбиция ≥3000 (fused DeltaNet).

## Current state

- Decode graphs — рабочий референс: `DecodeGraphState`, paged pool, htod-cache
  guard, staging ids, logits_out D2D, retry-fallback семантика отличается.
- Prefill: single-slot F16 KV растёт realloc (ломает capture); RoPE cos/sin
  индексируются host-offset (`narrow(0, index_pos, T)` — меняется между
  запросами → нельзя фиксировать в графе); DeltaNet fused CUDA готов;
  MTP catch-up требует hidden всех позиций.

## Solution architecture

```mermaid
flowchart TD
  REQ[PrefillChunk T slot] --> EN{"graphs on?\nMTP off?\nvision off?"}
  EN -->|no| EAGER[eager forward_prefill]
  EN -->|yes| LR{"LRU hit (T,slot)?"}
  LR -->|hit| REP[stage ids/cos/sin → launch → logits_out]
  LR -->|miss| CAP[capture: begin → staged fwd → head(last|full)\n→ logits_out D2D → end_capture]
  CAP --> ST[LRU push, evict >8]
  REP --> OUT[D2H logits_out]
  ERR[error] --> FB[eager this chunk, keep feature on]
```

### Ключевые решения

| # | Решение | Обоснование |
|---|---------|-------------|
| PD-201 | RoPE: staging cos/sin H2D вне графа (буферы [maxT, rd/2]) вместо devpos-index_select | минимальный дифф forward; decode-devpos остаётся для декода |
| PD-202 | Статический KV: env QWEN36_PGRAPH_STATIC_KV=1 → первый prefill выделяет cap=max_ctx сразу; рост→ERROR | realloc ломает capture; 4B@16K ≈ 1.3 ГиБ F16 — влезает |
| PD-203 | Логиты last: head над hidden[..,-1] внутри графа | D2H 1 МБ vs 500 МБ |
| PD-204 | MTP active → eager (граф не совместим с catch-up hidden) | спека FR-A2 full-режим покрывает будущий MTP-совместимый вариант |
| PD-205 | Fallback: eager этот чанк + счётчик; graphs НЕ выключаются навсегда | отличие от decode (там выключение) |
| PD-206 | Buckets: реализуются ТОЛЬКО если per-T не достиг цели или miss-rate хвостов >15% | candle_flash_attn не принимает маску → buckets требуют SDPA-mask path (дорого). YAGNI-gate |

## Implementation phases

### Phase 1: Статические буферы стабильности (estimate: 6h)
- [ ] `GatedAttentionLayer`: static-KV режим (cap=max_ctx, alloc at first use,
      append без growth; ERROR при переполнении) → `model_weights.rs`
- [ ] RoPE staging: `prefill_cos/sin_stage: Option<Tensor>` буферы [maxT, rd/2]
      + ветка forward, читающая их вместо narrow → `model_weights.rs`
- [ ] Env: QWEN36_PGRAPH_STATIC_KV=1 включает оба
- **Check:** eager prefill @8K с флагом — wall не хуже baseline ±5%, логи чистые

### Phase 2: Capture/Replay + LRU (estimate: 12h)
- [ ] `PrefillGraphState { exec, cu_graph, stream, t, slot, ids_t, cos_t, sin_t,
      logits_out }` в adapter.rs (аналог DecodeGraphState)
- [ ] `forward_prefill_graphed(ids, index_pos)` в ModelWeights: полный проход
      embeds→блоки→norm→head(last) c capture-безопасными путями
- [ ] Adapter: `prefill_chunk_graphed` + LRU(8) + fallback-retry (PD-205)
- [ ] Метрики `[pg]` per chunk + сводка за запрос
- **Check:** лог hit=1 на втором чанке; illegal address отсутствует;
  logits MAE ≤1e-2 vs eager

### Phase 3: Full-logits режим (estimate: 4h)
- [ ] Exit-mode env QWEN36_PG_LOGITS=last|full; full → head над всем [T,vocab],
      logits_out [T,vocab]; MTP catch-up работает в этом режиме
- **Check:** с MTP=1+SLOTS=2 full-режим: генерация корректна

### Phase 4: A/B стенд (estimate: 6h)
- [ ] `scripts/pgraph_matrix.ps1`: {disabled,last,full} × {per-T} на 4B
      @2K/8K/16K + golden-smoke ×3 промпта
- [ ] Таблица результатов → research.md; выбор default
- [ ] Buckets: gate-решение по miss-rate хвостов (реализация только при
      необходимости — PD-206)
- **Check:** таблица заполнена, победитель выбран данными

### Phase 5: Default + док (estimate: 2h)
- [ ] Победитель = default env; README/план обновлены
- **Check:** свежий клон получает выигрыш «из коробки»

## Traceability

| FR | Phases |
|---|---|
| FR-A1 стратегии | 2 (per-T), 4 (buckets gate) |
| FR-A2 режимы выхода | 2 (last), 3 (full) |
| FR-A3 статич.KV | 1 |
| FR-A4 DeltaNet stable | 1, 2 |
| FR-A5 LRU | 2 |
| FR-A6 fallback-retry | 2 |
| FR-A7 метрики | 2, 4 |

## Risks

| Risk | Mitigation |
|---|---|
| Аллокации внутри capture рвут граф (cudarc pool) | decode-референс доказал работоспособность; те же guard'ы |
| DeltaNet conv_state tail-copy указатели нестабильны | qkv_conv аллоцируется в capture → graph memory node (CUDA 12.4) |
| Статич. KV 27B не влезает | FR-S2 авто-off; 4B приоритет |
| RoPE staging забыт в какой-то ветке (mrope/vision) | vision/MTP → eager (gate); grep-аудит narrow(0, index_pos |
| Windows WDDM evicts graph pool | trim после capture; мониторинг VRAM в бенче |

## Plan decisions

| # | Решение | Дата |
|---|---|---|
| PD-201..206 | см. таблицу выше | 2026-08-25 |

## Deferred

| Вопрос | Когда |
|---|---|
| Buckets реализация | gate по данным Phase 4 |
| MTP-совместимый граф | после паритета префилла |
