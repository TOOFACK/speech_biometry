# Speech Biometry API

Сервис биометрической идентификации по голосу. Позволяет регистрировать голосовые профили спикеров и идентифицировать их по аудиозаписям.

## Технологии

- **FastAPI** — HTTP API
- **SpeechBrain ECAPA-TDNN** — модель извлечения голосовых эмбеддингов (предобучена на VoxCeleb)
- **PyTorch / TorchAudio** — инференс и обработка аудио
- **NumPy** — хранение профилей (NPZ)

## Запуск

```bash
docker compose up --build -d
```

Сервис доступен на `http://localhost:8000`. При первом запуске загружается модель (~170MB).

Swagger UI: `http://localhost:8000/docs`

## API

### `GET /` — информация о сервисе

**Ответ:**
```json
{
  "message": "Speaker Identification API",
  "endpoints": {
    "POST /process": "Upload audio file. With speaker_id — enroll, without — identify.",
    "GET /speakers": "List enrolled speakers."
  }
}
```

---

### `POST /process` — регистрация или идентификация спикера

Единый эндпоинт. Поведение зависит от наличия параметра `speaker_id`.

**Параметры (multipart/form-data):**

| Параметр     | Тип    | Обязательный | Описание |
|-------------|--------|-------------|----------|
| `file`       | file   | да          | Аудиофайл (wav, ogg, mp3), 0.8–12 сек речи |
| `speaker_id` | string | нет         | ID спикера. Передан — регистрация, пропущен — идентификация |

#### Режим регистрации (enrollment)

Передаём `speaker_id` — голос привязывается к этому ID. При повторной регистрации эмбеддинг усредняется с существующим.

```bash
curl -X POST http://localhost:8000/process \
  -F "file=@voice.wav" \
  -F "speaker_id=ivan"
```

**Ответ (201):**
```json
{
  "status": "enrolled",
  "speaker_id": "ivan",
  "total_speakers": 3
}
```

#### Режим идентификации (identification)

Без `speaker_id` — сервис сравнивает голос со всеми зарегистрированными профилями и возвращает рейтинг совпадений.

```bash
curl -X POST http://localhost:8000/process \
  -F "file=@unknown_voice.wav"
```

**Ответ (200):**
```json
{
  "scores": [
    {"speaker_id": "ivan", "score": 0.8734},
    {"speaker_id": "anna", "score": 0.4521},
    {"speaker_id": "petr", "score": 0.3102}
  ]
}
```

`score` — косинусное сходство (0–1). Чем выше, тем больше совпадение.

---

### `GET /speakers` — список зарегистрированных спикеров

```bash
curl http://localhost:8000/speakers
```

**Ответ:**
```json
{
  "speakers": ["ivan", "anna", "petr"]
}
```

---

## Ошибки

| Код | Причина |
|-----|---------|
| 400 | Слишком короткое аудио, не обнаружена речь, пустая база спикеров |
| 500 | Внутренняя ошибка обработки |

**Пример ошибки:**
```json
{
  "detail": "Too little speech detected in audio. Try a longer or louder recording."
}
```

## Требования к аудио

- **Форматы:** WAV, OGG, MP3
- **Длительность речи:** 0.8 – 12 секунд (после удаления тишины)
- **Качество:** чем чище запись, тем выше точность
- **Содержание:** не зависит от произносимых слов — идентификация по тембру голоса

## Хранение данных

Профили спикеров хранятся в `db/profiles.npz` (volume в Docker). Каждый профиль — 192-мерный вектор (768 байт). При удалении файла все регистрации сбрасываются.
