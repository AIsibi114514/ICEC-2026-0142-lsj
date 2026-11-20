@echo off
rem NoticeCatcher launcher. Usage: notice.bat run ^| status ^| digest ^| selftest ^| web
chcp 65001 >nul
setlocal
set "ROOT=%~dp0"
set "APP=%ROOT%02-20Days-Challenge\run.py"

if not exist "%APP%" (
    echo [ERROR] run.py not found: %APP%
    exit /b 1
)

rem Prefer .python-path.txt written by tools\register_task.ps1.
rem PATH on this machine may only contain the Microsoft Store stub,
rem which exits 0 while doing nothing -- so verify the interpreter really runs.
set "PY=python"
python -c "import sys" >nul 2>nul
if errorlevel 1 (
    if exist "%ROOT%.python-path.txt" (
        set /p PY=<"%ROOT%.python-path.txt"
    )
)

"%PY%" -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] No working Python found.
    echo         Run: powershell -ExecutionPolicy Bypass -File tools\register_task.ps1
    exit /b 1
)

"%PY%" "%APP%" %*
endlocal
