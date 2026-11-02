# Консультация ASTRA и Opus по mapped-router

Дата: 2026-11-02. Исходный `HEAD`: `82c392a3`. Незакоммиченный diff:
`engine/candle-kernels/src/moe_router.cu`, `engine/qwen35-batch/src/real/moe.rs`.

Консультации запускались по коду и заметкам, без сборки и замеров:

- `codex exec -m gpt-6-astra -c model_reasoning_effort=xhigh -s read-only`;
- `claude -p --model claude-opus-5-5 --effort xhigh --permission-mode plan`.

## Вывод

Обе модели независимо пришли к одному решению: diff в текущем виде не
коммитить. Mapped-router и mapped-merge — отдельный opt-in эксперимент.
Async/mapped по умолчанию и новый split `topk/4` отклонены обеими.

`take_async_cpu_error()` не имеет ни одного вызова: ошибка CPU-worker
всплывает только следующим запросом, а текущий токен читает старый `out`.

## Оценка выигрыша

| Изменение | Оценка ASTRA | Оценка Opus |
|---|---|---|
| Router/export вместо трёх D2H | 10–35 µs/слой | −40…−120 µs/слой |
| Mapped merge вместо H2D и двух add | 5–15 µs/слой | −20…+60 µs/слой |
| Итого с учётом overlap | 0,5–2 ms/token | 1,2–5,3 ms/token |

Расходятся в ширине интервала, сходятся в том, что выигрыш накладных
расходов, а не PCIe: per-token объём mapped-обмена 0,66 MiB против
276,75 MiB весов GPU-экспертов при `4/4`.

## Где остаток на 256k

| Резерв | Потенциал | Основание |
|---|---:|---|
| Page-aligned loader (`82c392a3`) | 8,91 ms/token | `10 × (1,882 − 0,991)` мкс |
| CPU-MoE: scope, гранулярность, SIMD, баланс | 4–8 ms/token | чистый проход 0,641 ms против 0,87–0,93 на сервере |
| Submission gaps, staging, handshake | 3–6 ms/token | 3 D2D на слой, `zeros`, `trace.copy_in` |

Прогноз после loader: 128k `49,48 ms/token` = 20,21 t/s при цели 20,32;
256k `54,99 ms/token` = 18,19 t/s при цели 23,78. То есть 128k закрывается
почти целиком одним loader, 256k требует ещё около 13 ms/token, и обе
модели называют этот запас недоказанным.

## Согласованные решения по diff

| Изменение | Решение |
|---|---|
| `MappedHostBuffer`, export-ядро, `moe_merge_mapped` | Оставить, только opt-in |
| Удаление `partials`/`to_vec()` в reduction | Принять |
| Последовательное квантование активаций перед down | Принять |
| `rayon::scope` вокруг полного CPU-слоя | Принять |
| Удаление AVX2 x8 | Отклонить, вернуть |
| `MOE_CPU_ASYNC=1`, `MOE_CPU_MAPPED=1` по умолчанию | Отклонить |
| `MOE_CPU_GPU_EXPERTS=topk/4` по умолчанию | Отклонить |
| Кэш env через `OnceLock` | Вернуть |

Обязательные исправления до коммита:

1. ограниченное ожидание export-флага вместо бесконечного spin;
2. запись NaN в `out` при ошибке worker до публикации флага;
3. `cpu + gpu + shared`, как в прежнем пути — ассоциативность F32 иначе
   ломает побитовый parity;
4. `mapped_router` только для `ForwardMode::DecodeBatch`;
5. обычная запись флага вместо `atomicExch_system`.

## Профиль перед следующей гипотезой

Один прогон, 64 decode-шага на заполненном 256k KV, `82c392a3` с новым
loader, `MOE_CPU_MAPPED=0`. Метки: enqueue, graph submit, wait submit,
merge submit; dequeue, export ready, store lock, publish; quant, gate/up,
down, reduction; граница шага.

```
nsys profile --trace=cuda,nvtx,osrt --sample=process-tree \
  --cpuctxsw=process-tree --cuda-graph-trace=node \
  --capture-range=nvtx --nvtx-capture='decode_steady@yforge' \
  --gpu-metrics-device=0 -o decode256k
```

Разделять: worker ждёт GPU (export wait), GPU ждёт CPU (merge уже
поставлен), host submission gap, memory stall (kernel идёт, SM простаивают).

## Что осталось непроверенным

- Долгоконтекстные числа 128k/256k выше сняты на
  `Ornith-1.5-35B-Q4_K_M`, а не на 9B: это 41 MoE-слой, `top-8`,
  `hidden=2048`, `expert_ffn=512`. Короткий decode 84,7 против 80,6 t/s из
  `2026-10-30` относится к 9B. Оба профиля нужно держать раздельно.
- Геометрия attention согласована: код включает MMA при `n_head / n_kv == 8`,
  `decode_gqa_probe` использует 16 Q-heads и 2 KV-heads. Фактические heads
  загруженного GGUF на стенде отдельно не проверены.
- Перенос микробенча loader в E2E не измерен.
- Baseline llama.cpp: 49,21 ms/token на 128k при 42,05 на 256k выглядит
  аномально, обе модели требуют переснять его тем же запросом.
