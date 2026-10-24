Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting BiteID Forensic Analysis System              " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "[ERROR] Virtual environment not found in venv\Scripts\python.exe" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "[*] Opening browser at http://127.0.0.1:5000..." -ForegroundColor Green
Start-Process "http://127.0.0.1:5000"

Write-Host "[*] Starting Flask application..." -ForegroundColor Green
& ".\venv\Scripts\python.exe" "app.py"
