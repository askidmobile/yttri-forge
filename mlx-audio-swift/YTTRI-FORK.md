# Форк mlx-audio-swift для Yttri

**Апстрим:** https://github.com/Blaizzy/mlx-audio-swift
**Тег:** v0.1.3, коммит `d302a5c6080d2bb97bae38c7418f82abb76013b6`
**Лицензия:** MIT (файл `LICENSE` апстрима сохранён)
**Кто потребляет:** `Yttri/frontend/src-tauri/resources/mlx-runtime/sidecar`
(`Package.swift`, продукт `MLXAudioSTT`) — ASR-слот сайдкара, тип
`nemotron` (T-558).

## Зачем форк

Nemotron-слоту нужны две точки входа, которые в апстриме `internal`:

| Символ | Файл | Зачем |
|--------|------|-------|
| `NemotronASRModel.fromDirectory(_:computeDType:)` | `Sources/MLXAudioSTT/Models/NemotronASR/NemotronASRModel.swift` | грузим веса из каталога приложения (S3-доставка), а не через HuggingFace-загрузчик |
| `NemotronASRModel.makeStreamSession(language:chunkMs:)` | `.../NemotronASRStreamSession.swift` | cache-aware сессия на канал живой записи (FR-002) |

Патч — только модификаторы доступа и комментарии `// yttri:`, логика не
тронута. Ровно эти две строки и нужно переносить при обновлении апстрима.

## Обновление на новую версию апстрима

```bash
git clone --branch <тег> --depth 1 https://github.com/Blaizzy/mlx-audio-swift.git /tmp/mas
rsync -a --delete --exclude .git /tmp/mas/ mlx-audio-swift/   # YTTRI-FORK.md сохранить
# заново пометить два символа public (см. таблицу), обновить тег/коммит выше
```

Обновление пина = **повторный гейт качества FR-010**: пин защищает сборку,
но не результат распознавания.

## PR в апстрим

Патч видимости стоит предложить наверх (issue/PR «make NemotronASR
fromDirectory / makeStreamSession public») — тогда форк можно снять.
