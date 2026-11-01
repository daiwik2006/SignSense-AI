@echo off
setlocal
title Sign Language Detection Server
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo.
    echo Run these once, then come back:
    echo   python -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist "models\sign_classifier.keras" (
    echo [WARNING] No trained model found at models\sign_classifier.keras
    echo The server may start but predict nothing. Train first:
    echo   .venv\Scripts\python.exe scripts\preprocess_data.py
    echo   .venv\Scripts\python.exe scripts\train_static_model.py
)

echo Starting Sign Language Detection server...
echo.
echo Wait for the line  "Uvicorn running on http://0.0.0.0:8000"  below,
echo then open  http://localhost:8000  in your browser.
echo Press Ctrl+C to stop.
echo.
call ".venv\Scripts\python.exe" -m app.main
if errorlevel 1 (
    echo.
    echo [ERROR] Server exited with an error - see the message above.
    pause
)
endlocal