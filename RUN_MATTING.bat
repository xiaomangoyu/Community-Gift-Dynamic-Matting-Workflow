@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating the local Python environment...
  py -3.12 -m venv .venv 2>nul
  if errorlevel 1 py -3 -m venv .venv
  if errorlevel 1 goto :python_error
)

echo Installing or checking matting dependencies...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :dependency_error

echo.
echo Processing images and videos. This may take several minutes.
".venv\Scripts\python.exe" tools\process_matting.py %*
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo Matting completed successfully.
) else (
  echo Matting completed with one or more failures. Check matting_outputs\manifest.json.
)
pause
exit /b %RESULT%

:python_error
echo Python 3 was not found. Install 64-bit Python and enable Add Python to PATH.
pause
exit /b 2

:dependency_error
echo Dependency installation failed. Check the network connection and try again.
pause
exit /b 2
