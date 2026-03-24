# YouTube Audio Search MVP

This workspace now has:

- `backend/`: FastAPI service that indexes video audio and searches by text query.
- `extension/`: Chrome extension popup + YouTube seeking integration.

## 1. Start backend

```bash
cd backend
..\.venv\Scripts\pip install -r requirements.txt
..\.venv\Scripts\uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

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
