# AGENTS.md — yttri-forge

Приватный проект: конвертер весов и загрузчик тензоров для инференс-стека Yttri
(candle-fork + qwen36-server). Источник правды по продукту — `docs/brief/PROJECT-BRIEF.md`.

## Состояние репозитория

Кода пока нет — только документация. Не ищи build/test/lint: появятся вместе с
Rust-конвертером (этап 1). Весь текущий контекст в `docs/brief/`.

## Жёсткие ограничения (из decision log)

- Конвертер — только Rust, вход — только HF safetensors (BD-003, BD-004).
- Этап 1 — гибрид: **prefill читает F16-сайдкар, decode — GGUF-квант** (dual-read).
  Никогда не читать F16 на декоде — bandwidth-bound, декод рухнет (R-DUAL).
- Этап 1 строго до аренды 48GB — замеры на слабом железе (BD-006).
- Обратный конвертер наш→GGUF не делаем (BD-002).
- Этап 0 (профиль префилла @8K) — gate перед конвертером: matmul-проекции
  должны быть ≥40% времени префилла, иначе пересмотр цели (R-TARGET).

## Workflow (ai-spec-kit, установлен user-scope)

Пайплайн: `brief → spec → plan → implement → review`. Артефакты ложатся в `docs/`:

| Команда | Результат |
|---|---|
| `/create-brief` | `docs/brief/` (уже есть) |
| `/create-spec <фича>` | `docs/specs/<фича>.md` |
| `/create-spec-plan <фича>` | `docs/plans/<фича>.md` |
| `/create-spec-implement <фича>` | пошаговая реализация по фазам плана |
| `/create-spec-review <фича>` | ретроспектива после реализации |
| `/commit` | conventional commit в стиле репо |

Нетривиальную фичу не кодить сразу — сначала spec + plan.

## Управление задачами

- `TASKS.md` — источник правды по «что в работе». **Руками не редактировать** —
  только через `/tasks` (скил `task-tracker`, скрипт `tasks.py`).
- `/tasks` — активные задачи; `/tasks add "..."`, `/tasks update T-XXX "✅ Done"`,
  `/tasks archive-done`. Завершённые уходят в `TASKS_ARCHIVE.md`.

## Коммиты

Conventional Commits, описание на русском: `docs(brief): ...`, `init: ...`
(см. `git log`). Один логический шаг — один коммит.

## Язык

Все коммуникации, документация и комментарии в коде — на русском.
