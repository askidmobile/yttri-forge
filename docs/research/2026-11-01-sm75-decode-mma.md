# Turing decode-attention MMA для Q4-K MoE

Дата: 2026-11-01. Стенд: RTX 2080 Ti 11 GiB, CUDA 12.4, sm_75.

## Задача

Скалярное paged GQA-ядро на 256k читало K/V и считало каждый Q/K dot
float-инструкциями. Для десяти full-attention слоёв это около `53.7 ms/token`
и больше всего бюджета decode у llama.cpp. Нужно сохранить F16 paged KV и
существующий layout результатов, но использовать Turing tensor cores.

## Ядро

`flash_decode_paged_gqa_mma_partial` обслуживает один split, один batch и одну
KV-head одним warp'ом. Для каждой группы из 16 KV-токенов:

1. `K[16×256] × Q^T[256×8]` считается через `mma.m16n8k8` и даёт восемь
   независимых softmax для восьми Q-heads.
2. Online max/sum обновляется для каждой головы; `P` округляется в F16 один раз
   и из тех же значений считается denominator.
3. `P^T[16×16] × V[16×256]` даёт выход восьми Q-heads. Восемь младших строк
   матрицы `P^T` содержат головы, восемь старших нулевые.
4. Partial записывается в прежний формат `[2 + hd]`; отдельное combine-ядро
   объединяет только splits, как и раньше.

Новый путь включается через `DECODE_GQA_MMA=1`. При `0` остаётся исправленный
скалярный GQA-путь; `DECODE_GQA_FOLD=0` возвращает прежнее split-K ядро.

## Parity

`decode_gqa_probe` перед benchmark запускает две проверки:

- 64 токена с восемью различными Q-heads и аналитическим softmax;
- 257 токенов и частичные страницы: сверка со скалярным ядром.

На стенде:

| Проверка | Максимальная ошибка |
|---|---:|
| 64 токена | `1.53e-4` |
| 257 токенов | `1.5e-5` |
| `compute-sanitizer --tool memcheck` | 0 ошибок |

Реальный HTTP parity на 505/512 токенах: `1, 2, 3, 4, 5, 6, 7,`.

## Микробенч

Одна attention-сцена, 30 итераций после прогрева:

| KV | Скалярный GQA | MMA, splits 128 |
|---|---:|---:|
| 131072 | 2.705 ms | **0.966 ms** |
| 261900 | 5.366 ms | **1.883 ms** |

На 128k ускорение `2.80×`, на 256k `2.85×`. Автоматический выбор для MMA:
128 splits в обоих целевых контекстах. При 256k это также ограничивает размер
каждого CTA примерно 2046 токенами.

## End-to-end

Первый чистый прогон 128k, `MOE_CPU_GPU_EXPERTS=5`, без expert cache:

| Метрика | yforge MMA | llama.cpp |
|---|---:|---:|
| Prompt tokens | 130870 | около 130885 |
| Prefill | **346.6 t/s** | 303.28 t/s |
| Decode | 15.87 t/s | 20.32 t/s |

Относительно старого ошибочного GQA-пути decode вырос `13.35 → 15.87 t/s`.
После исправления Q-heads старые значения нельзя использовать как корректный
baseline.

256k, тот же `5/3` без overlap:

| Метрика | yforge MMA | llama.cpp |
|---|---:|---:|
| Prompt tokens | 261870 | около 261885 |
| Prefill | **311.8 t/s** | 285.0 t/s |
| Decode | 13.78 t/s | 23.78 t/s |

После этих двух прогонов MoE-путь был изменён: CPU и host submission GPU-ветки
выполняются параллельно, SwiGLU считается внутри gate/up chunk, а weighted sum
делается одним `moe_weighted_sum_kernel`. На коротком 512-токенном запросе
split `5/3` без overlap давал `18.40 t/s`, итоговый opt-in `4/4` с этими
изменениями — `19.45 t/s`.

Ещё один найденный дефект: cudarc `alloc_pinned` всегда создаёт
`CU_MEMHOSTALLOC_WRITECOMBINED`. Для D2H-буферов `ids/weights/x`, которые затем
читает CPU, WC вреден. Они переведены на page-locked память с flags=0; выход
CPU→GPU остался WC. CPU-путь упал до `0.56–0.62 ms/layer`, короткий decode
вырос до **`20.64 t/s`**.

Для expert-store память имеет противоположный профиль доступа: CPU читает
собственную RAM-копию, а GPU читает device-mapped буфер. Поэтому pinned буфер
экспертов переведён в `DEVICEMAP|WRITECOMBINED`; короткий decode вырос до
`20.80 t/s`.

Итоговые long-context замеры этой конфигурации без expert cache:

| Контекст | Prefill yforge | Prefill llama | Decode yforge | Decode llama |
|---|---:|---:|---:|---:|
| 128k | **348.2 t/s** | 303.3 t/s | 17.13 t/s | **20.32 t/s** |
| 256k | **312.4 t/s** | 285.0 t/s | 14.50 t/s | **23.78 t/s** |

Prefill выигран на обоих контекстах. Decode ускорен относительно исходного
корректного state, но преимущества над llama.cpp пока нет. Для следующих
изменений нужен CUDA graph/async pipeline GPU-сегментов; дальнейший перебор
CPU-ядер и raw layout не окупился.

## CUDA graph GPU-подфазы MoE

`MOE_GPU_GRAPH=1` захватывает только чистый GPU-подграф одного MoE-слоя:
shared expert, gate/up GPU-экспертов, SwiGLU, down и weighted reduction.
CPU-эксперты, D2H/H2D и merge остаются снаружи. Стабильные device-буферы
`x/ids/weights` обновляются D2D перед replay.

На коротком запросе все слои захватились, parity сохранился, decode вырос
`20.80 → 22.01 t/s` (+5.8%). На длинных контекстах:

| Контекст | Без graph | Graph |
|---|---:|---:|
| 128k | 17.13 t/s | **17.63 t/s** |
| 256k | 14.50 t/s | **14.79 t/s** |

`MOE_GPU_GRAPH=1` остаётся opt-in. Prefill не деградировал:
347.3/312.2 t/s.

Warm-freeze cache также проверен: динамический прогрев 32 шага, затем
замороженный набор без LRU. С graph он несовместим из-за side-stream activity
во время capture; без graph короткий decode составил `18.86 t/s` и оказался
хуже no-cache варианта. Cache закрыт.

## Отклонено в ходе интеграции

- Dynamic expert cache 2 GiB с отдельным copy stream: hit 64–71%, но decode
  упал до `14.42 t/s` из-за LRU/table rewrites.
- Статический cache без LRU дал только `19.07` вместо `18.91 t/s` при цене
  2 GiB VRAM и не был включён.
- Второй native hot pool на 36 потоков поднял CPU-MoE до `1.2–1.4 ms/layer`
  и уронил decode до `12.28 t/s`; собственный scheduler дороже Rayon.
- Специализированный GPU Q4_K down kernel для `K=512` побитово совпал со
  старым, но E2E остался в шуме (`19.38`): ядро удалено.

## Ограничения

- MMA снимает стоимость attention, но не закрывает CPU-MoE и GPU expert PCIe.
- `DECODE_GQA_MMA=1` пока opt-in. Оба контекста замерены, но decode всё ещё
  уступает llama.cpp.
- Для 256k недостаточно одного attention-ядра: нужен измеримый выигрыш в
  MoE critical path либо дальнейшее сокращение PV/attention.
