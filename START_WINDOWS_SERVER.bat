@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "PORT=3001"

for /f %%P in ('powershell -NoProfile -Command "$c=Get-NetTCPConnection -LocalPort 3001 -State Listen -ErrorAction SilentlyContinue ^| Select-Object -First 1; if($c){$c.OwningProcess}"') do set "PORT_PID=%%P"
if defined PORT_PID (
  echo Port 3001 is already in use by PID !PORT_PID!.
  powershell -NoProfile -Command "$p=Get-CimInstance Win32_Process -Filter 'ProcessId=!PORT_PID!' -ErrorAction SilentlyContinue; if($p){Write-Host ('Command: ' + $p.CommandLine)}"
  set /p "STOP_OLD=Stop this exact process and restart on port 3001? [y/N]: "
  if /I "!STOP_OLD!"=="Y" (
    taskkill /PID !PORT_PID! /F >nul
  ) else (
    set "PORT=3002"
    powershell -NoProfile -Command "if(Get-NetTCPConnection -LocalPort 3002 -State Listen -ErrorAction SilentlyContinue){exit 1}"
    if errorlevel 1 (
      echo Port 3002 is also occupied. No process was stopped.
      pause
      exit /b 2
    )
  )
)

echo.
echo Matting Preview Viewer
echo Local URL: http://localhost:%PORT%/matting_demo.html
echo LAN URL:   http://YOUR-WINDOWS-IP:%PORT%/matting_demo.html
echo.
echo Keep this window open while others are viewing the site.
echo Press Ctrl+C to stop the server.
echo.

where py >nul 2>&1
if not errorlevel 1 (
  py -3 -m http.server %PORT% --bind 0.0.0.0
  goto :end
)

where python >nul 2>&1
if not errorlevel 1 (
  python -m http.server %PORT% --bind 0.0.0.0
  goto :end
)

echo Python 3 was not found.
echo Install Python 3 from https://www.python.org/downloads/windows/
echo During installation, enable "Add Python to PATH", then run this file again.
pause

:end
endlocal
