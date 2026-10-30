# CUDA-оптимизации: проверка CPU и Metal перед слиянием

Дата: 2026-10-30. Источник: подготовка Turing-веток к слиянию в `main`.

## Что обнаружено

Проверенный на RTX 2080 Ti CUDA-профиль не гарантирует сборку остальных
бэкендов. `cargo test --lib` сервера обнаружил 28 ошибок компиляции движка:
CUDA-профилирование, роутер MoE и перенос состояния обращались к `cudarc`,
CUDA-полям и страничному пулу без соответствующих `cfg`.

После исправления библиотеки интеграционные тесты дополнительно выявили
`fa_prefill_probe` и `fa_decode_probe`: Cargo автоматически собирал эти
CUDA-бинари без фичи `cuda`, поскольку у них не было `required-features`.

## Исправление

- CUDA-события и методы сброса/переноса CUDA-состояния закрыты `cfg(feature = "cuda")`.
- Валидация pool-backed записей на CPU/Metal возвращает `false` без обращения к CUDA-пулу.
- Прямой D2D seed разрешён только при `device.is_cuda()`. На CPU/Metal
  сохраняется snapshot, который требуется для засевания batched-состояния.
- Оба FA-пробника требуют `real-model` и `cuda` в Cargo.toml.

Повторяющееся правило: после изменения общего адаптера проверять сборку без
CUDA и декларации всех новых бинарей. Успешный CUDA-бенчмарк этого не проверяет.

## Проверка

| Проверка | Результат |
|---|---|
| Сервер: `cargo test --lib` | 99 passed, 1 ignored |
| Сервер: `cargo test --test api_test --test multimodal_api_test` | 26 + 5 passed |
| Движок: `cargo test -p qwen35-batch --features real-model --test scheduler_parity --test qwen35moe_reference --test expert_store --test model_profile` | 4 + 11 + 0 + 12 passed; expert_store закрыт CUDA-фичей |
| Сервер: `cargo check --features metal --lib` | passed на macOS; проверка компиляции |
| Сервер: `cargo check --release --features cuda-turing --bins` | passed на стенде Михаила, CUDA 12.4 / sm_75 |
| Квантование: `python3 tests/test_ds41_vram_alloc.py` | 27 passed |

Изменения CUDA-гейтов не меняют CUDA-ядра. Результаты предыдущего A/B и
ограничения аппаратной проверки остаются в журнале
[замеров RTX 2080 Ti](../research/2026-10-30-mikhail-server-results.md).

## Исходники

- [Адаптер](../../engine/qwen35-batch/src/real/adapter.rs)
- [Профилирование и состояние модели](../../engine/qwen35-batch/src/real/model_weights.rs)
- [Роутер MoE](../../engine/qwen35-batch/src/real/moe.rs)
- [Feature-гейты бинарей](../../engine/qwen35-batch/Cargo.toml)
