# Local API for YouTube Audio Search

## Setup

```bash
cd backend
..\.venv\Scripts\pip install -r requirements.txt
```

## Run

```bash
..\.venv\Scripts\uvicorn app:app --reload --host 127.0.0.1 --port 8000
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
