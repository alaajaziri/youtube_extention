@echo off
cd /d %~dp0

if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [error] Failed to create virtual environment. Is Python installed?
        pause
        exit /b 1
    )
)

if not exist ".venv\Scripts\uvicorn.exe" (
    echo [setup] Installing dependencies...
    .venv\Scripts\pip install -r backend\requirements.txt
    if errorlevel 1 (
        echo [error] Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo [info] Starting backend on http://127.0.0.1:8000 ...
echo [info] Wait for "Application startup complete" before using the extension.
.venv\Scripts\uvicorn app:app --reload --host 127.0.0.1 --port 8000 --app-dir backend
