# Q4_K_M, 128k/256k vs llama.cpp: GQA decode и prefill coalescing

Дата: 2026-11-01. Стенд: RTX 2080 Ti 11 GiB, PCIe 3.0 x16,
2× Xeon E5-2686 v4, `numactl --cpunodebind=1 --membind=1`, 36 Rayon threads.
Модель: `Ornith-1.5-35B-Q4_K_M.gguf`, 41 MoE-слоёв, top-8.
Общие env: `MOE_CPU=1`, `KV_POOL_Q8=0`, `CUDA_GRAPHS=1`, `PGRAPH=on`,
`PREFILL_CHUNK=2048`, `CONTEXT_LIMIT=GRAPH_WINDOW=CTX`, temp=0, 128 output tokens.

## Итоговые замеры

| Контекст | Метрика | yforge | llama.cpp | Отношение |
|---|---|---:|---:|---:|
| 128k, prompt 130885 | Prefill | **346.5 t/s** | 303.28 t/s | **1.143×** |
| 128k | Decode | 10.52 t/s | **20.32 t/s** | 0.518× |
| 256k, prompt 261885 | Prefill | **310.7 t/s** | 285.0 t/s | **1.090×** |
| 256k | Decode | 8.60 t/s | **23.78 t/s** | 0.362× |

Более ранние точки этой сессии на 128k: 207.4/6.91 после перехода на paged eager,
248.8/6.88 после coalescing H2D, 344.6/6.92 после `PREFILL_CHUNK=2048`,
346.0/8.65 после GQA kernel, 346.5/10.52 после 64 splits и pinned staging.
256k до pinned/GQA был 311.0/6.87; итог — 310.7/8.60.

## Принятые изменения

- `0f454f38`: `MOE_CPU=1` остаётся на paged KV, но не пытается захватить CUDA graph.
- `051d1be2`: соседние expert id копируются H2D диапазонами; pointer table при
  release также обновляется диапазонами.
- `553341e0`: `MOE_CPU_TIMING=1` показывает раздельно D2H/shared wait и CPU+H2D.
- `55534cef`: Turing GQA partial kernel объединяет 8 Q-голов одной KV-головы,
  K/V tile загружается в shared один раз. Combine layout не изменён.
- `c343e282`: 64 splits для 128k, 256 для >128k.
- `95849c33`: постоянные pinned-буферы ids/weights/x/out и прямой H2D результата.

## Микробенч decode-attention

Одна attention-сцена, `decode_gqa_probe`, 20/16 итераций:

| KV | Старое split-K | GQA, лучший split | Ускорение |
|---|---:|---:|---:|
| 131072 | 6.556 ms | **2.727 ms (64)** | **2.40×** |
| 261900 | 8.677 ms | **5.120 ms (256)** | **1.69×** |

## Decode bottleneck после изменений

На коротком контексте `MOE_CPU_TIMING=1` даёт в среднем примерно:

- D2H + ожидание shared expert до pinned staging: 0.43 ms/layer;
- после pinned staging: **0.145 ms/layer**;
- CPU-MoE + постановка H2D: около **1.18 ms/layer**.

Суммарно CPU/boundary остаётся около 54 ms/token; это главный разрыв с llama.
GQA fold уже снял большую часть long-context attention overhead, поэтому следующий
шаг должен уменьшать зависимую CPU-MoE стоимость или перекрывать её GPU-работой,
а не менять expert cache.

## Воспроизводимость

- Скрипты: `/tmp/longstream35.py`, `/tmp/longstream35-llama.py`, `/tmp/parity20.py`.
- JSON: `/tmp/yforge-pinned-128k.json`, `/tmp/yforge-final-256k.json`,
  `/tmp/llama-256-final.json`.
- Parity 505/512: до и после GQA/pinned — `1, 2, 3, 4, 5, 6, 7,`;
  итоговые long-context ответы связные.

## Отклонено: GPU cache на целевом контексте

`MOE_CPU=0`, `EXPERT_CACHE_MIB=4400`, `PREFILL_CHUNK=2048` дал short decode
15.43 t/s (в пределах шума с CPU/pinned 15.37), но 128k prefill упал с
`CUDA_ERROR_OUT_OF_MEMORY` в MoE `broadcast_mul` после первого чанка при
остатке 64 MiB. Конфигурация отклонена: она не обслуживает одновременно 128k/256k.
