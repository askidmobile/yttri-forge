# yttri-forge

Приватный проект: собственный конвертер весов и загрузчик тензоров для
инференс-стека Yttri (candle-fork + qwen36-server).

## Мотивация (кратко)

Текущий стек потребляет GGUF (формат llama.cpp) и их кванты, а ядра портируются
из llama.cpp — вечное отставание ×1.3 по декоду длинного контекста. Собственный
конвейер `BASE (safetensors) → наш формат → наши ядра` даёт:

- per-tensor форматы: декод bandwidth-bound → 2–4 bit; префилл compute-bound →
  W4A16/W8A8 на tensor cores;
- layout под наши ядра (interleave/pad/K-split) — +10–30% без новых алгоритмов;
- калибровка под задачи (coding) вместо generic IQ2;
- MTP-head в F16 → acceptance выше.

GGUF остаётся fallback. Этап 1 — гибрид (GGUF + F16-сайдкар для горячих
тензоров), этап 2 — W4A16 ядра, этап 3 — полный собственный формат.

## Структура

```
docs/
  brief/       бриф проекта (create-brief)
  specs/       спецификации фич
  plans/       планы реализации
```

## Связанные репозитории

| Репо | Роль |
|---|---|
| askidmobile/candle (fork, master d614d47d) | движок, ядра |
| askidmobile/qwen36-server (main ef76eb7) | сервер, планировщик |
