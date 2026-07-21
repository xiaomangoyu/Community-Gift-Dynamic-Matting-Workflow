@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating the local Python environment...
  py -3.12 -m venv .venv 2>nul
  if errorlevel 1 py -3 -m venv .venv
  if errorlevel 1 goto :python_error
)

echo Installing or checking CUDA demo dependencies...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements-gpu.txt
if errorlevel 1 goto :dependency_error

echo.
echo Checking NVIDIA CUDA backend...
".venv\Scripts\python.exe" tools\check_cuda.py --device cuda
if errorlevel 1 echo CUDA is unavailable. Processing will safely fall back to CPU.

echo.
echo Processing Pak20 images and videos with GPU-first scheduling...
".venv\Scripts\python.exe" tools\process_matting_demo.py --device auto --encoder auto %*
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo Matting demo completed successfully.
) else (
  echo Matting demo completed with one or more failures. Check matting_demo_manifest.json.
)
pause
exit /b %RESULT%

:python_error
echo Python 3 was not found. Install 64-bit Python and enable Add Python to PATH.
pause
exit /b 2

:dependency_error
echo CUDA dependency installation failed. Check the network connection and try again.
pause
exit /b 2

