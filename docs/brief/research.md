# Research — yttri-forge

## Замеры-обоснование (2026-08-24, RTX 3060 12GB)

Полные данные: qwen36-server `docs/plans/2026-08-24-qwen35-4b-benchmark.md`

### Qwen3.5-4B Q4_K_M

| Метрика @8K | Ours (SLOTS=2+graphs) | llama.cpp b10472 |
|---|---|---|
| Decode | 63.1 ток/с | 81.9 ток/с |
| Prefill wall | 2.22с (~1810 ток/с) | 1.62с (~2480 ток/с) |
| MTP draft-mtp | наш adaptive: нейтрален | их: ×2.1 (accept 94%) |

### Профиль GPU-блоков декода (gprof, eager prime)

DeltaNet 24 слоя = 8.3мс (0.35мс/слой); Attention 8 слоёв = 7.8мс (~0.97мс/слой).
Host launch overhead снят CUDA graphs; остаток — скорость ядер.

### Выводы для формата

1. Декод bandwidth-bound: IQ2 (2.06 bpw) даёт максимальную скорость по природе;
   гоняться за Q8 ради decode бессмысленно даже на 48GB.
2. Префилл compute-bound: F16 tensor-core GEMM >> integer MMQ. W4A16/W8A16 —
   рычаг ×2–4 на префилле. Это главный аргумент собственного формата.
3. Attention-блоки дороже DeltaNet в 2.8× на слой — первоочередные кандидаты
   на F16-сайдкар (проекции attn.* дают максимум эффекта при минимуме размера).

## Аналоги

| Проект | Формат | Урок для нас |
|---|---|---|
| llama.cpp GGUF | K-quants/IQ + MMQ/MMVQ | эталон decode; их MMQ портирован у нас |
| vLLM/AWQ | W4A16 pack + dequant-in-registers | готовые AWQ ядра для порта (этап 2) |
| TensorRT-LLM | собственные форматы per-layer | доказывает жизнеспособность подхода «формат под железо» |


## Этап 0: профиль префилла (gate R-TARGET) — ПРОЙДЕН (2026-08-24)

Qwen3.5-4B Q4_K_M @8K, GPROF=3 sync-замеры (wall 2.71с с синками):

| Фаза | мс | Доля |
|---|---|---|
| DeltaNet proj (matmul) | 790 | 35% |
| DeltaNet head/ssm_out (matmul) | 543 | 24% |
| DeltaNet fused (рекуррентность) | 563 | 25% |
| Attention FA2 | 365 | 16% |
| Прочее (attn proj/wo/kv) | 21 | 1% |

**Matmul-проекции = 59% ≥ 40% gate.** F16-сайдкар попадает в цель.
Открытие: ssm_out — скрытый matmul 24%, включён в дефолтную маску
(`--f16-attn-all` → переименовать смысл в "все тяжёлые проекции":
attn.wq/wk/wv/wo, deltanet.wqkv/wgate/w_beta/w_alpha/ssm_out).
Рекуррентность (25%) и FA2 (16%) — вне scope этапа 1.
Инструментация оставлена в форке под QWEN36_GPROF=3 ([pfp-delta]/[pfp-attn]).
