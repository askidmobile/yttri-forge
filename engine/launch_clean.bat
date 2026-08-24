@echo off
REM Clean launch: kill all, start fresh, wait 300s, check, do NOT kill.
taskkill /f /im qwen36-server.exe 2>nul
timeout /t 3 /nobreak >nul
del /q D:\Projects\yttri-build\live.log 2>nul
start "qwen36" /b cmd /c "D:\Projects\yttri-build\run_live_inner.bat >D:\Projects\yttri-build\live.log 2>&1"
echo Waiting 300s for model load...
timeout /t 300 /nobreak >nul
echo === PROCESS ===
tasklist /fi "imagename eq qwen36-server.exe" 2>nul
echo === LOG ===
type D:\Projects\yttri-build\live.log
echo === END ===
exit /b 0