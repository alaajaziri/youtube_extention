@echo off
cd /d %~dp0backend
..\.venv\Scripts\uvicorn app:app --reload --host 127.0.0.1 --port 8000
