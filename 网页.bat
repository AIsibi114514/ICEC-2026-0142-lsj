@echo off
rem One-click launcher for the web UI. Double-click this file.
chcp 65001 >nul
setlocal
set "ROOT=%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python not found in PATH. Install Python 3.10+ first.
    pause
    exit /b 1
)

echo Starting NoticeCatcher web UI ...
python "%ROOT%02-20Days-Challenge\run.py" web
if errorlevel 1 pause
endlocal
