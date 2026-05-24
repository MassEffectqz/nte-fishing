@echo off
>nul 2>&1 net session || (
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)
cd /d "%~dp0"
python -X utf8 main.py
pause