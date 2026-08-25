# Plan: CUDA-graph prefill

**Date:** 2026-08-25
**Status:** 🚧 Phase 1-2 done
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

### Phase 2: Capture/Replay + LRU (estimate: 12h) — 🔶 ядро готово (623b82a6..42603918)
Готово: forward_prefill_graphed (embeds→блоки paged→norm→head-last),
HybridBlock::forward_prefill_paged (DeltaNet fused in-graph + attention
forward_attn_prefill_paged: F16 dual-read proj, RoPE devpos, q8 round-trip,
append_multi T строк, FA2 varlen seqlens_k=kv0+T). Сборка BUILD=0 на yttri-win.
Остаток фазы: adapter capture/replay + LRU + rope/kv_len staging.
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


## Миграция репозитория (2026-08-25)

candle-fork ЗАМОРОЖЕН (05aa926a — последняя точка). Вся разработка в yttri-forge:
- engine/ = subtree форка (история сохранена, a29344e9)
- qwen36-server Cargo.toml → path ../../yttri-forge/engine/*
- Транспорт на yttri-win: git bundle (GitHub credentials недоступны из SSH-сессии)
- Сборка BUILD=0 из нового пути подтверждена

Правила процесса (урок 2026-08-25): НЕ коммитить в candle-fork; сборки/тесты только yttri-win.


## Результаты Phase 2 (2026-08-25, 4B Q4_K_M, RTX 3060)

Adapter-слой готов: capture/replay, LRU×8 по (T, slot), staging вне графа
(ids, rope_pos, kv_len[slot], seqlens_q=[0,T], block_table), fallback без
выключения графов, метрики `[pg]`. Режимы: `QWEN36_PGRAPH=off|on|check`.

### Что пришлось починить в ядре Phase 2a (иначе первый replay = мусор)

| Баг | Следствие |
|---|---|
| `launch_increment_t` не передавал `b`/`t` в ядро; инкремент вообще не вызывался | kv_len ломался на случайных значениях |
| RoPE через `apply_partial_rotary_emb_devpos` (рассчитан на [B,nh,1,hd]) | broadcast давал форму [T,nh,T,half] |
| `seqlens_q` брался статический `[0,1]` | FA2 видела 1 q-токен вместо T |
| `seqlens_k_for_prefill(seq_len, seq_len)` вместо `(b, T)` | narrow за границу буфера |
| FA2 без `window_size_right=Some(0)` | нет causal-маски — токены чанка видели будущее |

`seqlens_k` теперь считается один раз на проход, а не в каждом слое.

### Паритет (mode=check: граф → откат snapshot'ом → eager на том же чанке)

| Условие | MAE | argmax |
|---|---|---|
| T=11/19, F16 KV | **0.000e0** (бит-в-бит, включая replay) | OK |
| T=512 ×3 + T=374, F16 KV | 6.6e-2 … 2.5e-1 (max diff 0.45…1.53) | OK на всех чанках |
| любой T, q8 round-trip KV | 4.2e-1 | argmax flip на near-tie |

Вывод: **весь вклад в расхождение давал q8 round-trip KV**. Пул и так F16 —
поэтому дефолт переключён на чистый F16 (`QWEN36_PGRAPH_Q8KV=1` возвращает
старое поведение). Остаточные 0.1 на T=512 — разный порядок редукции
paged-varlen FA2 vs плотной `flash_attn` (при T=11 обе бит-в-бит).

E2E на 1910-токенном промпте (32 токена генерации): graph-префилл и eager
дают связный идентичный текст до ~60 символов, дальше расходятся на near-tie.

### Производительность (4B, чанк 512)

| | время |
|---|---|
| capture (первый T) | 12.3 с (загрузка/JIT 4381-4669 нод), далее ~275 мс |
| replay | 233–246 мс |
| eager-проход тем же путём | 275–282 мс |
| prefill wall 1910 ток., eager | 1.14 с |
| prefill wall 1910 ток., graph | 1.16 с |

Граф снимает ~48 мс/чанк host-launch overhead (17%), но на 4B префилл
compute-bound — суммарного выигрыша нет. Цель плана (снять ~1010 мс/чанк)
относится к 27B, где launch overhead доминирует; там же и надо мерить.
На 12 GB графы для 27B выключаются VRAM-гейтом (`GRAPH_MIN_FREE_BYTES`) —
нужен либо 48GB-стенд, либо пересмотр гейта.

### Побочная находка: ytf16-сайдкар ломает префилл (не связано с графами)

`Qwen3.5-4B-Q4_K_M.ytf16` подхватывается автоматически рядом с GGUF
(`[ytf] mapped 152 tensors (2488 MiB F16)`). При этом **eager-префилл выдаёт
248320/248320 не-финитных логитов** для чанков T≥64 (T=11/19 — норма);
сэмплер получает NaN и первый токен мусорный, дальше генерация вырождается в
повтор одного токена. Проверено без графов вообще (`plain`), не зависит от
`QWEN36_DISABLE_FLASH_PREFILL` и `QWEN36_DELTA_WARPS`.
С `QWEN36_DISABLE_YTF16=1` — 0 не-финитных, ответ связный.

Все замеры паритета/качества выше сделаны с `QWEN36_DISABLE_YTF16=1`.
**Починено в тот же день**: два бага конвертера (смещения safetensors + порядок
голов v/z/a/b/out) — см. [stage1-f16-sidecar](2026-08-24-stage1-f16-sidecar.md).
С исправленным сайдкаром паритет графа сохраняется: T=11 бит-в-бит, T=512
MAE 0.05…0.13, argmax совпадает на всех чанках.
Добавлено предупреждение `[pfa] WARN non-finite logits` — ловит класс целиком.

### Осталось

- Phase 3 (full-logits + MTP), Phase 4 (A/B стенд), Phase 5 (дефолт)
- Замер выигрыша на launch-bound конфигурации (27B / 48GB)
- Гейт `pg_paged_only`: слот с KV только в пуле закрывает eager-decode ошибкой;
  снять можно обратной миграцией paged→batched, если понадобится fallback


## Почему graph-префилл не даёт скорости (2026-08-25, вечер)

Разбор чанка `[pfa]` расставил всё по местам:

| | eager | graph |
|---|---|---|
| fwd (постановка в очередь) | 44.3 мс | 0.0001 |
| logits_d2h (догон GPU) | 159.6 | 0.1 |
| replay `[pg] run` | — | 205.0 |
| **GPU-часть суммарно** | **203.9** | **205.0** |

`fwd` асинхронный — работа уезжает в очередь и считается параллельно, поэтому
GPU-часть eager это `fwd + logits_d2h`, а не один `logits_d2h`. Сравнив
205 мс replay с одним лишь `logits_d2h`, я получил несуществующие «34 мс
лишней работы в графе» — их нет, GPU-работа в обоих режимах одинакова.

Вывод: **префилл упирается в GPU, а не в запуски**, поэтому графы здесь не
ускоряют (в отличие от декода, где шаг крошечный и правит launch overhead).
Ценность graph-префилла остаётся в другом: он снимает хост с критического пути
(полезно при нескольких слотах и на слабом CPU) и даёт готовый механизм на
будущее, когда GPU-часть станет дешевле.

### Что было найдено по дороге

- `kv_append_paged_f16_multi` копировал T строк циклом внутри блока при
  `grid = (n_kv, b, 1)` — четыре блока на 28 SM, 362 мкс на запуск при ~1 МБ
  данных. Токен вынесен на ось z сетки: чанк в графе 211.1 → 204.4 мс. **Принято.**
- `max_seqlen_k` подавался как всё окно (16384 при реальных ~2000). Сделана
  корзина по длине KV с ключом графа — корректно работает, но времени не дала:
  205.0 против 204.4. **Откачено** (лишние графы в пуле без выигрыша).
