@echo off
title LLM Bench
REM Run from this file's folder, even when double-clicked from elsewhere.
cd /d "%~dp0"

where python >nul 2>nul || (
  echo Python is not installed. Get it from https://www.python.org/downloads/
  echo During setup, tick "Add python.exe to PATH".
  pause
  exit /b 1
)

echo Checking requirements...
python -m pip install -q --disable-pip-version-check -r requirements.txt

python run.py
