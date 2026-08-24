@echo off
REM Debug launch: starts server in background, waits 15s, kills, dumps log.
REM Never hangs SSH — always returns after timeout.
taskkill /f /im qwen36-server.exe 2>nul
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;%PATH%"
set "QWEN36_MODEL=D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q2_K_XL.gguf"
set QWEN36_SLOTS=1
set QWEN36_CTX=2048
set QWEN36_PORT=18099
set QWEN36_API_KEY=test
set RUST_BACKTRACE=1
del /q D:\Projects\yttri-build\debug.log 2>nul
start /b cmd /c "D:\Projects\yttri-build\qwen36-server\target\release\qwen36-server.exe > D:\Projects\yttri-build\debug.log 2>&1"
timeout /t 120 /nobreak >nul
tasklist /fi "imagename eq qwen36-server.exe" 2>nul
echo === LOG START ===
type D:\Projects\yttri-build\debug.log
echo === LOG END ===
taskkill /f /im qwen36-server.exe 2>nul
exit /b 0