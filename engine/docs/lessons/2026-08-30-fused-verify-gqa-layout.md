# Слитая MTP-проверка: GQA-layout важнее специальной causal mask

**Date:** 2026-08-30  
**Source:** `learn-from-work` после исправления fused verify attention

## Что произошло

Слитая проверка MTP одним вызовом FA2 ускорила verify примерно на 30%, но
изменила логиты: доля принятых черновиков упала, вывод расходился с эталоном с
первого символа, а иногда содержал повреждённый текст. Специальная causal mask
делила физический индекс строки на число GQA-групп.

## Root cause

Ошибка возникала раньше маски — при интерпретации памяти Q. Исходный contiguous
тензор имеет порядок `[position, kv_head, group, d]`. Слитая ветка объявляла
`row = position * ngroups + group`, но задавала strides как для порядка
`[row, kv_head, d]`, то есть фактически требовала
`[position, group, kv_head, d]`.

Из-за этого адрес зависел от `position + kv_head`, а разные пары
`(position, kv_head)` читали и записывали перекрывающиеся участки. Возникала
гонка записей в output. Деление глобального `row_idx` в `mask.h` не могло
исправить неверный layout и aliasing.

## Исправление

- Q оставлен в исходном `[position, query_head, d]` без reinterpretation.
- Один fused FA2 launch представляет T позиций как batch из T
  последовательностей с `seqlen_q = 1`.
- Все элементы batch используют одну page table (`batch_stride = 0`), но имеют
  точные K-длины `base+1, base+2, ..., base+T` через cumulative seqlens.
- Число K-split сохранено таким же, как у построчного эталона.
- Специальная ветка causal mask удалена; `k = 1` остаётся на старом пути.

## Проверка

Модель-независимый CUDA-тест использует production GQA-геометрию
`n_head=24`, `n_kv_head=4`, `head_dim=256`, seed 7 и сравнивает итоговый F16
attention output с построчным эталоном:

- prefix 452: `max_abs_diff = 0.000000000e0`;
- prefix 32685: `max_abs_diff = 0.000000000e0`.

## References

- `qwen35-batch/src/real/model_weights.rs`
- `qwen35-batch/src/real/paged_attn.rs`
- `qwen35-batch/src/real/paged_kv_cuda.rs`
- `qwen35-batch/tests/fused_verify_attn.rs`
- `candle-flash-attn/kernels/flash_api.cu`
- `candle-kernels/src/quantized.cu`
