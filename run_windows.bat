@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher was not found. Install Python 3.12 64-bit from https://www.python.org/downloads/
  pause
  exit /b 1
)
py -3.12 --version >nul 2>nul
if errorlevel 1 (
  echo Python 3.12 was not found. Install Python 3.12 64-bit from https://www.python.org/downloads/
  pause
  exit /b 1
)

if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,12) else 1)"
  if errorlevel 1 rmdir /s /q .venv
)
if not exist .venv py -3.12 -m venv .venv
if errorlevel 1 (
  echo Could not create the virtual environment.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto install_failed
pip install -r requirements.txt
if errorlevel 1 goto install_failed
python -m streamlit run app.py
pause
exit /b 0

:install_failed
echo Dependency installation failed. Check the error above and your internet connection.
pause
exit /b 1
