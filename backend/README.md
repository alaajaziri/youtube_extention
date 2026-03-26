# Local API for YouTube Audio Search

## Setup (first time only)

Run from the **repo root**:

```bash
# Windows
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements.txt

# Linux / macOS
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
```

> The first run downloads large ML models (CLAP, CLIP, Whisper). Wait until
> `Application startup complete` appears before using the extension.

## Run

```bash
# Windows
.venv\Scripts\uvicorn app:app --reload --host 127.0.0.1 --port 8000 --app-dir backend

# Linux / macOS
.venv/bin/uvicorn app:app --reload --host 127.0.0.1 --port 8000 --app-dir backend
```

## Endpoints

- `GET /health`
- `POST /index`
- `POST /search`

### `POST /index` body

```json
{
  "videoUrl": "https://youtu.be/55oPmIBq3Ok",
  "windowSeconds": 2.0,
  "hopSeconds": 1.0,
  "batchSize": 16
}
```

### `POST /search` body

```json
{
  "videoUrl": "https://youtu.be/55oPmIBq3Ok",
  "query": "guitar",
  "topK": 5
}
```
