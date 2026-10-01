# Yttri Forge

Русский | [English](README.en.md)

**Сервер локального инференса на Rust и CUDA: один бинарь `yforge`, GGUF-модели, API OpenAI и Anthropic, оптимизация длинного контекста и параллельных запросов.**

Yttri Forge развивается как CUDA-движок для Windows и Linux, ориентированный на запуск языковых моделей на NVIDIA GPU, в том числе RTX 3060 12 ГБ. Здесь находятся движок и конвертер весов. HTTP-сервер `yforge` собирается из отдельного репозитория [qwen36-server](https://github.com/askidmobile/qwen36-server).

[Сайт](https://forge.yttri.online/) · [Замеры](https://forge.yttri.online/#bench) · [Модели](https://forge.yttri.online/#models) · [Установка Windows/Linux](https://forge.yttri.online/#install) · [Сервер](https://github.com/askidmobile/qwen36-server)

`yforge` выполняет инференс без Python-сервера и контейнера: нужны веса модели, NVIDIA-драйвер и подходящий CUDA runtime. Python используется в отдельных инструментах и бенчмарках. Собственные ядра, int8 KV-кеш, CUDA Graphs и MTP дают выигрыш в конкретных проверенных конфигурациях; результаты и условия приведены ниже.

## С чего начать

- **Запустить модель и подключить клиент:** перейдите к [быстрому старту](#быстрый-старт-сервера)
- **Развернуть Windows-сервер с Open WebUI:** используйте [DEPLOY.md](https://github.com/askidmobile/qwen36-server/blob/main/docs/DEPLOY.md)
- **Изучить движок или конвертер:** начните с [архитектуры](#архитектура) и [forge-convert](#конвертер-весов-forge-convert)
- **Посмотреть результаты:** откройте [производительность](#производительность) и [публичные скрипты сравнения](https://github.com/askidmobile/qwen36-server/tree/main/scripts/bench)
- **Изучить протоколы:** [CLI и настройка](https://github.com/askidmobile/qwen36-server#readme), [API и media-контракты](https://github.com/askidmobile/qwen36-server/blob/main/docs/engine-api.md)
- **Прочитать историю разработки:** [статья на vc.ru](https://vc.ru/id6112515/3167949-sozdanie-lokalnogo-inference-stack-dlya-llm-na-rtx-3060)

## Что есть в стеке

| Компонент | Возможности |
| --- | --- |
| Движок | Исполнение поддерживаемых GGUF-моделей; CUDA-путь с FlashAttention-2, квантованным матричным умножением, CUDA Graphs и страничным KV-кешем |
| Планировщик | Continuous batching для архитектур `qwen35` / `qwen35moe`, очередь запросов и повторное использование префикса |
| Сервер `yforge` | OpenAI Chat Completions, OpenAI Responses, Anthropic Messages; потоковая выдача SSE и API-ключи |
| Мультимодальность | Обработка изображений и видео для совместимых моделей и профилей с необходимыми компонентами |
| MTP | Спекулятивное декодирование с отдельным совместимым MTP-компонентом; включается явно |
| Веб-интерфейс | Встроенный fallback-чат и прокси к Unsloth Studio; отдельное развёртывание Open WebUI описано в документации сервера |
| `forge-convert` | Конвертация исходных safetensors в F16-сайдкар `.ytf16` и самостоятельный контейнер `.ytf` v2; инструменты проверки тензоров |

Доступность оптимизаций зависит от backend, архитектуры модели и конфигурации. Совместимые HTTP-интерфейсы не означают полную реализацию всех функций облачных API.

**Модель выбирается при запуске.** Для смены модели перезапустите сервер с другим `--model` или профилем. В текущем HTTP-роутере ручки горячей смены и выгрузки модели отключены; описание горячей смены на сайте и в старых инструкциях относится к прежнему состоянию.

## Репозитории

| Репозиторий | Для чего нужен |
| --- | --- |
| **[yttri-forge](https://github.com/askidmobile/yttri-forge)**, этот репозиторий | Активная разработка движка в `engine/`, конвертер `forge-convert`, эксперименты с форматом весов |
| **[qwen36-server](https://github.com/askidmobile/qwen36-server)** | HTTP API, конфигурация, интеграция планировщика и исполняемый файл `yforge` |
| [askidmobile/candle](https://github.com/askidmobile/candle) | Исторический форк; актуальная разработка перенесена в `yttri-forge/engine/` |

Для сборки сервера нужны **оба первых репозитория**. В `qwen36-server/Cargo.toml` движок подключён локальными `path`-зависимостями. При расположении клонов рядом редактировать эти пути не требуется:

```text
workspace/
├── yttri-forge/
│   ├── engine/          # отдельный Cargo workspace: форк Candle
│   └── forge-convert/   # конвертер весов
└── qwen36-server/       # сервер → target/release/yforge(.exe)
```

## Модели и оборудование

Основной сценарий сервера: GGUF-модели семейств **Qwen 3.5 / 3.6 / 3.8 и Ornith 1.0 / 1.5**, использующие поддерживаемые архитектуры `qwen35` / `qwen35moe`. В сервере также есть отдельный путь `gemma4`.

Полная [таблица проверенных моделей на сайте](https://forge.yttri.online/#models) включает Qwen 3.5 4B/9B, Qwen 3.6 27B/35B-A3B, Qwen 3.8 27B, Ornith 1.5 9B и Gemma 4 E4B-it с конкретными квантами и стендами. Ориентиры для первого запуска:

- **Ornith-1.5-9B Q4_K_M:** рабочий профиль на RTX 3060 12 ГБ в [инструкции развёртывания](https://github.com/askidmobile/qwen36-server/blob/main/docs/DEPLOY.md)
- **Gemma 4 E4B Q4_K_M** и **Qwen3.6 35B-A3B IQ2_XXS:** результаты загрузки и работы на CUDA в [отчёте о проверке runtime](https://github.com/askidmobile/qwen36-server/blob/main/docs/lessons/2026-08-21-gguf-runtime-support-gate.md)

Наличие файла `.gguf` само по себе не означает совместимость. Проверяются архитектура, квантование, токенизатор, шаблон чата и нужные компоненты. Поддержка одного варианта модели не гарантирует работу всех размеров и квантов этого семейства. Для vision/video нужен совместимый vision-компонент; для MTP нужна соответствующая голова. Для первого запуска достаточно текстовой модели, без MTP.

### Требования

- **Проверенный стенд:** Windows, RTX 3060 12 ГБ, 64 ГБ RAM; версии инструментов и профиль приведены в `DEPLOY.md`
- **Ориентир инструкции развёртывания:** NVIDIA от 8 ГБ VRAM, от 16 ГБ RAM, около 20 ГБ диска под сборку и кеши **плюс** место под модели. Это не гарантия, что любая модель или длинный контекст поместятся
- **Для CUDA-сборки на Windows:** Rust MSVC, Visual Studio 2022 Build Tools с C++, CUDA Toolkit и совместимый NVIDIA-драйвер. Проверенная конфигурация проекта использует CUDA 13.2; особенности runtime 12.4 описаны в `DEPLOY.md`
- **Границы платформ:** согласно [решению BD-008 от 25 сентября 2026](docs/brief/decisions.md), Forge — CUDA-движок для Windows/Linux. На macOS продукт Yttri использует отдельный MLX-сайдкар, который не входит в Forge
- **Разработка и диагностика:** в `qwen36-server/Cargo.toml` сохранены опция Metal и сборка без GPU; [README сервера](https://github.com/askidmobile/qwen36-server#readme) описывает их как отдельные варианты запуска. Наличие этих путей в коде не означает, что Forge поставляется как macOS/Metal-продукт; CUDA-замеры к ним не относятся

VRAM расходуется на веса, KV-пул, временные буферы и графы. Длина контекста и число слотов влияют на этот бюджет. Начинайте с одного слота и небольшого окна; профиль на 128K требует отдельной настройки и не является универсальным значением для 12 ГБ.

## Быстрый старт сервера

Ниже команды сборки для **Windows/Linux + NVIDIA** и полный пример запуска на Windows с уже установленными инструментами. Для установки с нуля, автозапуска и Open WebUI используйте [полную инструкцию](https://github.com/askidmobile/qwen36-server/blob/main/docs/DEPLOY.md). Её раскладка каталогов отличается от соседних клонов ниже: не смешивайте два варианта путей.

### 1. Клонируйте движок и сервер рядом

В новом рабочем каталоге:

```console
git clone https://github.com/askidmobile/yttri-forge.git
git clone https://github.com/askidmobile/qwen36-server.git
cd qwen36-server
```

### 2. Соберите `yforge`

**Windows + CUDA**

В **cmd.exe** из каталога `qwen36-server`:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set CUDA_COMPUTE_CAP=86
cargo build --release --features cuda --bin yforge
target\release\yforge.exe --help
```

Пути должны соответствовать вашей установке. `CUDA_COMPUTE_CAP=86` здесь задан для RTX 3060; для другой карты выберите её compute capability. Результат сборки: `target\release\yforge.exe`.

**Linux + CUDA**

После установки NVIDIA-драйвера, CUDA Toolkit, Rust и инструментов C/C++ используйте соседние клоны из шага 1. В каталоге `qwen36-server`:

```bash
# Укажите фактический путь к CUDA и compute capability своей GPU.
export CUDA_PATH=/usr/local/cuda-12.8
export CUDA_COMPUTE_CAP=86
cargo build --release --features cuda --bin yforge
./target/release/yforge --help
```

В `local.env` ниже замените Windows-путь к модели на свой Linux-путь. Для запуска используйте `./target/release/yforge --env local.env`; остальные параметры конфигурации те же. Установка инструментов и запуск как systemd-службы описаны в [Linux-вкладке сайта](https://forge.yttri.online/#install).

### 3. Подготовьте модель и конфигурацию

Скачайте совместимый GGUF, например вариант Ornith-1.5-9B Q4_K_M, и проверьте условия распространения весов у автора модели. Веса не входят в репозиторий. Подготовку моделей и конвертацию из safetensors описывает [раздел «Модель» в DEPLOY.md](https://github.com/askidmobile/qwen36-server/blob/main/docs/DEPLOY.md#6-модель).

Создайте `local.env` рядом с `Cargo.toml` сервера. Замените путь к модели и значение ключа; не добавляйте этот файл с настоящим ключом в Git:

```dotenv
MODEL=D:\Models\Ornith-1.5-9B-Q4_K_M.gguf
API_KEYS='[{"key":"REPLACE_WITH_YOUR_RANDOM_KEY","name":"local"}]'
HOST=127.0.0.1
PORT=18099
CTX=8192
CONTEXT_LIMIT=8192
SLOTS=1
MAX_TOKENS=2048
KV_POOL_Q8=1
KV_CACHE_DTYPE=q8
CUDA_GRAPHS=1
MTP=0
```

Это консервативный стартовый пример, а не профиль максимальной скорости. `CTX` и `CONTEXT_LIMIT` заданы согласованно; увеличение одного рекламируемого окна не увеличивает автоматически реальный лимит движка.

```bat
target\release\yforge.exe --env local.env --dry-run
target\release\yforge.exe --env local.env
```

`--dry-run` показывает конфигурацию без загрузки модели. Успешный `--dry-run` ещё не проверяет генерацию: дождитесь сообщения о готовности сервера и выполните запрос ниже.

Приоритет настроек: **CLI → окружение процесса → env-файл**. Старые значения в окружении могут перекрыть `local.env`.

### 4. Проверьте API

В отдельном окне **PowerShell**:

```powershell
$headers = @{ Authorization = 'Bearer REPLACE_WITH_YOUR_RANDOM_KEY' }
$models = Invoke-RestMethod -Uri 'http://127.0.0.1:18099/v1/models' -Headers $headers
$models.data

$body = @{
    model = $models.data[0].id
    messages = @(@{ role = 'user'; content = 'Say hello in one sentence.' })
    max_tokens = 256
    stream = $false
} | ConvertTo-Json -Depth 5

$response = Invoke-RestMethod `
    -Uri 'http://127.0.0.1:18099/v1/chat/completions' `
    -Method Post -Headers $headers -ContentType 'application/json' -Body $body
$response.choices[0].message
```

Используйте тот же ключ, что в `local.env`. Для клиента с OpenAI-совместимым подключением укажите Base URL `http://127.0.0.1:18099/v1`, свой ключ и `id` из `/v1/models`.

Веб-интерфейс доступен по `http://127.0.0.1:18099/`: сервер проксирует настроенный Studio backend, а при его недоступности показывает встроенный чат. Open WebUI устанавливается отдельно.

### Если запуск не удался

| Симптом | Что проверить |
| --- | --- |
| Cargo не находит `qwen35-batch` или Candle | Оба клона лежат рядом; `yttri-forge/engine/` существует |
| `nvcc` / `cl` не найдены | Пути к CUDA и окружение VS Build Tools |
| Windows-процесс завершается до полезного лога | Зависимости CUDA runtime и порядок `PATH`, см. раздел 9 `DEPLOY.md` |
| HTTP 401 | Ключ в запросе совпадает с `API_KEYS`; передан `Authorization: Bearer …` |
| OOM, резкое замедление или предупреждение об окне KV | Уменьшите контекст и число слотов; проверьте VRAM и shared GPU memory |
| Ответ оборвался во время рассуждений | Проверьте клиентский `max_tokens`: он ограничивает и доступный бюджет генерации |

Оставляйте `HOST=127.0.0.1` для локального использования. Для доступа из сети отдельно настройте контроль доступа и HTTPS-прокси; не публикуйте API напрямую в интернет. Не включайте логирование тел запросов для приватных данных без необходимости.

## Архитектура

```text
Чат / IDE / AI-клиент
        ↓ HTTP + API-ключ
qwen36-server / yforge
        ↓ обработка API, очередь и планирование
yttri-forge/engine/qwen35-batch + Candle
        ↓ загрузка весов, KV-кеш, вычислительные ядра
NVIDIA GPU / CUDA
```

- `engine/` содержит форк [Hugging Face Candle](https://github.com/huggingface/candle) и остаётся отдельным Cargo workspace
- `engine/qwen35-batch/` содержит runtime и планировщик; `qwen35` / `qwen35moe` проходят через `BatchedEngine` даже при одном слоте
- В сервере `gemma4` использует отдельный последовательный `CandleEngine`; возможности batching не следует переносить на него автоматически
- `forge-convert/` входит в корневой Cargo workspace и готовит контейнеры весов
- `docs/brief/`, `docs/specs/` и `docs/plans/` сохраняют обоснования и историю экспериментов; ранние документы могут описывать уже пересмотренные решения

## Конвертер весов `forge-convert`

Для обычного запуска GGUF-сервера этот шаг **не нужен**. Конвертер предназначен для работы с собственным конвейером весов и исследований.

Из корня `yttri-forge`:

```console
cargo build --release -p forge-convert
cargo run --release -p forge-convert -- --help
cargo test -p forge-convert
```

Реализованные режимы CLI:

- `--f16-heavy`: F16-сайдкар `.ytf16` для выбранных тяжёлых проекций; в движке есть отдельный dual-read путь для prefill
- `--pack`: самостоятельный `.ytf` v2 с типами тензоров, метаданными и встроенным токенизатором
- `--pack-vision` и `--pack-mtp`: упаковка соответствующих компонентов с эталонным GGUF
- `--verify-ytf … --gguf …`: потензорная проверка контейнера относительно эталона
- `--list-gguf`: просмотр имён, форм и типов тензоров

Исходники: [CLI](forge-convert/src/main.rs), [упаковка](forge-convert/src/pack.rs), [загрузчик контейнеров](engine/qwen35-batch/src/real/ytf16.rs). Наличие режима упаковки не гарантирует совместимость произвольной модели. GGUF остаётся самым простым маршрутом для первого запуска.

## Производительность

Результаты ниже взяты из **опубликованных отчётов проекта**, а не являются обещанием скорости на любой системе или замером текущего `main`.

### Последняя проверка RTX 3060 в опубликованном журнале

[Журнал измерений Ornith](https://github.com/askidmobile/qwen36-server/blob/main/docs/research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md) начинается 16 сентября, но содержит последующие эксперименты. Его заключительный **§103 от 18 сентября 2026** описывает проверку после исправления OOM при длинном префилле.

Стенд: **RTX 3060 12 ГБ**, модель **Ornith-1.5-9B Q4_K_M**, рабочий профиль с q8 KV-пулом на 128K и int8-QK. Пять запросов выполнены последовательно: 30K, 62K, 30K, 30K и 127K.

| Проверка после исправления | Результат yforge |
| --- | ---: |
| Decode на заключительном промпте ~127K | **35,8 ток/с** |
| TTFT на этом промпте | **115,33 с** |
| VRAM после каждого из пяти запросов | **7863 МиБ** |
| Decode на трёх промптах ~30K | 48,8–49,3 ток/с |

**7863 МиБ — память после запроса, не пиковое потребление.** TTFT включает время до первого токена и не является скоростью prefill. Это результат конкретной последовательности запросов после исправления управления памятью, а не нагрузочный тест, гарантия для любого промпта или полный тест качества. Точные входы, профиль и этапы изменений описаны в §§95–103 журнала.

### Сравнение с llama.cpp

[Интерактивные таблицы на сайте](https://forge.yttri.online/#bench) разделяют два стенда. На RTX 3060 12 ГБ с Ornith-1.5-9B Q4_K_M, промптом 30 232 токена и flash-attention у обоих движков:

| Метрика | yforge | llama.cpp b10375 |
| --- | ---: | ---: |
| Prefill, F16 KV, пять пар | 19,661 с | 19,716 с |
| Decode, F16 KV, пять пар | 47,11 ток/с | 46,81 ток/с |
| Decode, int8 KV, окно 128K, три пары | 45,60 ток/с | 45,20 ток/с |
| Prefill, int8 KV, окно 128K | 20,40 с | 19,80 с |

Это близкий результат: в q8-prefill yforge уступает примерно 3%. Таблица сравнения и заключительная проверка памяти §103 относятся к разным этапам; смешивать их в один прогон нельзя.

На втором стенде, обозначенном в [отчёте от 18 сентября](https://github.com/askidmobile/qwen36-server/blob/main/docs/research/2026-09-18-prod-parity-vs-llamacpp.md) как RTX 4090 **48 ГБ**, использованы Qwen3.8-27B Q8_0 и одинаковое окно 256K у обоих движков:

| Метрика | yforge | llama.cpp b10375 |
| --- | ---: | ---: |
| Decode при 256K | 23,4 ток/с | 20,1 ток/с |
| Prefill при 8K | 2179 ток/с | 1826 ток/с |
| Установившийся TTFT, короткий промпт | 0,250 с | 0,342 с |
| VRAM при 256K | 35 241 МиБ | 36 659 МиБ |

Отсюда опубликованные **+16% decode на 256K, −27% TTFT и около −1,4 ГиБ VRAM**. Это результаты указанного 48-ГБ стенда, не обычной 24-ГБ RTX 4090 и не RTX 3060. MTP измерялся отдельно: +22% на 8K и +43% на 32K к собственному baseline yforge; в сравниваемой сборке llama.cpp той же MTP-головы нет.

### Воспроизвести

[Публичный benchmark-комплект](https://github.com/askidmobile/qwen36-server/tree/main/scripts/bench) добавлен в репозиторий 19 сентября. В нём есть `final_verify.py`, `batch_bench.py`, `mtp_bench.py`, `peak_vram.py` и другие проверки. Сначала прочитайте [требования и методику](https://github.com/askidmobile/qwen36-server/blob/main/scripts/bench/README.md): пути к моделям, бинарям и логам привязаны к исходному Linux-стенду и требуют адаптации. Скрипты запускают и останавливают процессы; используйте отдельный тестовый стенд.

После настройки путей, из корня `qwen36-server`:

```bash
python3 scripts/bench/final_verify.py
```

Для сопоставимого собственного теста фиксируйте коммиты обоих репозиториев, версию сравниваемого движка, GPU/драйвер, хеш GGUF, тип KV, контекст, слоты, MTP, состояние prefix cache, точные входы и длину вывода. Измеряйте отдельно загрузку, time to first token, prefill, decode и пик VRAM. Проверяйте качество ответов вместе со скоростью. MTP может как ускорить генерацию, так и замедлить её.

## Статус и дальнейшая работа

Это активно развивающийся инженерный проект. Код, инструкции развёртывания и исследовательские отчёты обновляются независимо, поэтому сохраняйте версии, на которых получен ваш результат.

- **Уже в коде:** GGUF runtime, CUDA-оптимизации, серверные API, F16-сайдкар, упаковка и чтение `.ytf` v2
- **Исследования и история решений:** [планы движка](docs/plans/), [исследования сервера](https://github.com/askidmobile/qwen36-server/tree/main/docs/research), [планы сервера](https://github.com/askidmobile/qwen36-server/tree/main/docs/plans)
- **Пересмотренные идеи:** первоначальный этап W4A16 отмечен в [TASKS.md](TASKS.md) отменённым по результатам эксперимента. Ранний бриф не является обещанием внедрить этот этап или получить указанный в нём прирост

Возможности, перечисленные в планах, следует считать исследовательскими до появления реализации и проверки на целевом железе. Сроки и универсальная поддержка всех моделей не заявляются.

## Сообщить о проблеме

Для ошибок движка и конвертации используйте [Issues этого репозитория](https://github.com/askidmobile/yttri-forge/issues); для HTTP API, конфигурации и запуска сервера — [Issues qwen36-server](https://github.com/askidmobile/qwen36-server/issues).

Приложите ОС, GPU и объём VRAM/RAM, версии драйвера/CUDA, коммиты обоих репозиториев, точное имя и квант модели, команду запуска и минимальный пример запроса. Перед публикацией удалите API-ключи, личные пути, приватные промпты и другие чувствительные данные из конфигурации и логов.

## Лицензия

Код распространяется по двойной лицензии на выбор: [MIT](LICENSE-MIT) или [Apache-2.0](LICENSE-APACHE) (`MIT OR Apache-2.0`). Движок в `engine/` — форк [Candle](https://github.com/huggingface/candle) на тех же условиях; его лицензии и копирайты авторов Candle сохранены в `engine/`.

Если явно не указано иное, любой вклад, намеренно отправленный для включения в проект, распространяется на условиях той же двойной лицензии без дополнительных условий.

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option. Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

Веса моделей распространяются на собственных условиях их авторов; лицензия кода проекта их не заменяет.

<a id="donate"></a>

## Поддержать проект

Если проект вам пригодился, его можно поддержать. Донаты идут на аренду GPU для тестирования новых моделей.

Адрес для перевода **USDT в сети TON**:

```text
UQAqmiUAf-kCo7pw1HM4KDrf4r8XCAsDDolODnZJnAydZ37O
```

> [!WARNING]
> Отправляйте только **USDT** и только в **сети TON**. Монеты TON, другие токены и переводы из других сетей (TRC-20, ERC-20, BEP-20) на этот адрес не зачисляются — средства можно потерять.

**Donate:** USDT on the TON network only, to the address above. Donations pay for GPU rental to test new models.
