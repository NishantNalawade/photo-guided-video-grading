@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run run_windows.bat once to create the app environment, then close the app.
  pause
  exit /b 1
)
echo Installing NVIDIA CUDA dependencies into the app environment.
echo This is a large download. A compatible NVIDIA GPU and recent driver are required.
".venv\Scripts\python.exe" -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-gpu.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "from video_processing import cuda_stack; t, k, d = cuda_stack(); print('CUDA grading ready:', t.cuda.get_device_name(0))"
if errorlevel 1 goto failed
echo GPU setup succeeded. Restart run_windows.bat.
pause
exit /b 0

:failed
echo GPU setup or self-test failed. Read the error above.
echo Check your NVIDIA driver and the PyTorch CUDA compatibility instructions in README.md.
pause
exit /b 1
