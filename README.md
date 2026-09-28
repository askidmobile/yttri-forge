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

## Структура движка

`engine/` — полный candle-fork, мигрирован через git subtree
(`a29344e9`, исходная точка 05aa926a). ВСЯ разработка движка теперь здесь.
askidmobile/candle заморожен как референс.

Сборка на yttri-win: `D:\Projects\yttri-inference\inference-build.bat`
указывает на `<clone>/engine/qwen35-batch`.

| Репо | Роль |
|---|---|
| askidmobile/candle (ЗАМОРОЖЕН на 05aa926a) | исторический референс |
| askidmobile/qwen36-server (main ef76eb7) | сервер, планировщик |

<a id="donate"></a>

## Поддержать проект

Если проект вам пригодился, его можно поддержать переводом **USDT в сети TON**:

```text
UQAqmiUAf-kCo7pw1HM4KDrf4r8XCAsDDolODnZJnAydZ37O
```

> [!WARNING]
> Отправляйте только **USDT** и только в **сети TON**. Монеты TON, другие
> токены и переводы из других сетей (TRC-20, ERC-20, BEP-20) на этот адрес
> не зачисляются — средства можно потерять.

**Donate:** USDT on the TON network only, to the address above.
