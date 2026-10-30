# Урок 2026-10-30: managed-fallback для host-хранилища экспертов

## Контекст
Запуск Ornith-1.5-35B-A3B Q4_K_M на арендованном стенде RTX 2080 Ti
(`MOE_EXPERTS=auto`, эксперты 18.6 ГиБ в RAM) упал на первом же
`malloc_host(210 МиБ, DEVICEMAP)`:

```
malloc_host(220200960 байт, DEVICEMAP) не удался:
DriverError(CUDA_ERROR_OPERATING_SYSTEM, ...)
```

Причина — `ulimit -l` = 8192 КиБ (hard limit тот же): pinned-выделения
CUDA ограничены memlock пользователя, а sudo на стенде недоступен.
На машинах разработчика лимит бесконечный, поэтому проблема не проявлялась.

## Правка
`engine/qwen35-batch/src/real/expert_store.rs`: `HostBuf::alloc` теперь
пробует два пути:

1. `malloc_host(DEVICEMAP)` — основной, полный zero-copy;
2. fallback `malloc_managed(CU_MEM_ATTACH_HOST)` — страницы остаются на
   хосте, ядро decode читает их через PCIe, `memcpy_htod` при prefill
   работает без изменений; UVA-указатель валиден на хосте, поэтому
   заполнение из GGUF и `expert_slice` не меняются.

Без обоих путей — прежний fail-closed (FR-008). Выбранный бэкенд пишется
в лог (`[moe] matrix: … pinned|managed`) и предупреждение о fallback.

## Почему managed, а не pageable+copy
- pageable-буфер нельзя отобразить в адресное пространство GPU — сломался бы
  zero-copy decode (FR-004) и таблицы указателей;
- managed ATTACH_HOST сохраняет семантику «GPU читает хост через PCIe»
  без миграции страниц (в отличие от ATTACH_GLOBAL);
- malloc не требует memlock — работает на любом арендованном стенде.

## Проверка
- `rustfmt --check` (синтаксис) локально; модуль CUDA-only, полный `cargo
  check --features cuda-turing` — на стенде (см. research 35B).
- Ожидаемая полоса managed ≈ pinned (обе упираются в PCIe 3.0 x16);
  подтверждается бенчем 35B на стенде.
