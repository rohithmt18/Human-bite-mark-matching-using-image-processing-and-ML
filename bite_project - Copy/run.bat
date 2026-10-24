@echo off
title BiteID Forensic Analysis System
echo ========================================================
echo   Starting BiteID Forensic Analysis System
echo ========================================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in venv\Scripts\python.exe
    pause
    exit /b 1
)

echo [*] Launching BiteID at http://127.0.0.1:5000 ...
echo [*] Press CTRL+C in this terminal to stop the server.
echo.

start "" http://127.0.0.1:5000
venv\Scripts\python.exe app.py

pause
