#!/usr/bin/env bash

# Move to repo root (works regardless of where the script is called from)
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/python" ]; then
    echo "[setup] Creating virtual environment..."
    python3 -m venv .venv || {
        echo "[error] Failed to create virtual environment. Is python3 installed?"
        exit 1
    }
fi

if [ ! -f ".venv/bin/uvicorn" ]; then
    echo "[setup] Installing dependencies..."
    .venv/bin/pip install -r backend/requirements.txt || {
        echo "[error] Failed to install dependencies."
        exit 1
    }
fi

echo "[info] Starting backend on http://127.0.0.1:8000 ..."
echo "[info] Wait for 'Application startup complete' before using the extension."
.venv/bin/uvicorn app:app --reload --host 127.0.0.1 --port 8000 --app-dir backend || {
    echo "[error] Backend failed to start. Check the output above for details."
    exit 1
}
