# YouTube Audio Search MVP

This workspace now has:

- `backend/`: FastAPI service that indexes video audio and searches by text query.
- `extension/`: Chrome extension popup + YouTube seeking integration.

## 1. Start backend

### Windows

```bat
start_backend.bat
```

### Linux / macOS

```bash
bash start_backend.sh
```

### Manual setup (any platform)

```bash
# From the repo root, create a virtual environment (only needed once)
python -m venv .venv

# Windows
.venv\Scripts\pip install -r backend\requirements.txt
.venv\Scripts\uvicorn app:app --reload --host 127.0.0.1 --port 8000 --app-dir backend

# Linux / macOS
.venv/bin/pip install -r backend/requirements.txt
.venv/bin/uvicorn app:app --reload --host 127.0.0.1 --port 8000 --app-dir backend
```

> **Note:** The first run downloads several large ML models (CLAP, CLIP, Whisper).
> This can take several minutes. Wait until you see `Application startup complete` before using the extension.

## 2. Load extension

- Open `chrome://extensions`
- Enable Developer mode
- Click Load unpacked
- Select `extension/`

## 3. Test flow

- Open a YouTube video
- Open extension popup
- Click Index This Video
- Enter query (e.g. `guitar`)
- Click Search
- Click Jump on a result
