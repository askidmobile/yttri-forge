# Подключение pi.dev к стенду yforge

Как открыть локальный OpenAI-совместимый endpoint yforge на чужой машине
через SSH-туннель и подключить его к pi.dev.

## 1. Сервер на стенде

```bash
# на стенде (tinycoder.dev)
nohup numactl --cpunodebind=1 --membind=1 \
  ~/yforge-e1/target-e1/release/yforge --env ~/pi.env \
  > ~/pi-server.log 2>&1 &
```

`~/pi.env` — профиль из `2026-11-03-rtx2080ti-35b-q4.md`, порт 18099,
`CTX=131072`. Проверка:

```bash
curl -s -H "Authorization: Bearer yttri-local-2026" \
  http://127.0.0.1:18099/v1/models
```

Ответ содержит `"id":"ornith-1.5-35b"`, `"context_length":131072`,
`"capabilities":{"tools":true,...}`.

## 2. Туннель на рабочую машину

```bash
ssh -f -N -L 18099:127.0.0.1:18099 -p 24 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  anton@tinycoder.dev
```

`-f` уводит ssh в фон, `-N` не запускает команду, `-L` пробрасывает порт.
Проверка: `lsof -nP -iTCP:18099 -sTCP:LISTEN` и `curl` по
`http://127.0.0.1:18099/v1/models`.

Остановить: `pkill -f "L 18099:127.0.0.1:18099"`.

## 3. Провайдер для pi.dev

| Поле | Значение |
|---|---|
| API | OpenAI-compatible (`chat_completions`, `responses`, `messages`) |
| Base URL | `http://127.0.0.1:18099/v1` |
| API key | `yttri-local-2026` |
| Model id | `ornith-1.5-35b` |
| Контекст | 131 072 (нативный 262 144) |
| Streaming | да |
| Tools | да |
| Слоты | 1 — параллельные запросы не обслуживаются |

Параметры сэмплинга на сервере зафиксированы (`sampling_locked: true`),
клиентские `temperature/top_p/top_k` игнорируются — это нормально.

Режим «thinking» по умолчанию включён и тратит токены на рассуждения.
Для агентских циклов его лучше выключать:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

## 4. Скорость

| Сценарий | Скорость |
|---|---:|
| короткий промпт, decode | ~34 т/с |
| 128K контекст, decode | ~30 т/с |
| 128K префил | ~347 т/с |
| 256K префил (если CTX=262144) | ~316 т/с |

Один слот: пока идёт генерация, второй запрос ждёт.
