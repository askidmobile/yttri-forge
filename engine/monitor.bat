@echo off
REM Monitor: launch server, poll every 30s for 10min, capture log on death.
taskkill /f /im qwen36-server.exe 2>nul
ping -n 4 localhost >nul
del /q D:\Projects\yttri-build\live.log 2>nul
start "qwen36" /b cmd /c "D:\Projects\yttri-build\run_live_inner.bat >D:\Projects\yttri-build\live.log 2>&1"
for /L %%i in (1,1,20) do (
  ping -n 31 localhost >nul
  echo --- poll %%i (%%i x 30s) ---
  tasklist /fi "imagename eq qwen36-server.exe" 2>nul | findstr qwen36-server
  if errorlevel 1 (
    echo PROCESS DIED at poll %%i
    echo === LOG ===
    type D:\Projects\yttri-build\live.log
    echo === END ===
    exit /b 0
  )
)
echo === STILL ALIVE after 600s ===
type D:\Projects\yttri-build\live.log
exit /b 0