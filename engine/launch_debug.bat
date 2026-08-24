@echo off
REM Launch server in background, wait 180s for model load, check status, do NOT kill.
REM SSH-friendly: always returns after timeout.
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
echo Waiting 180s for model load...
timeout /t 180 /nobreak >nul
echo === PROCESS ===
tasklist /fi "imagename eq qwen36-server.exe" 2>nul
echo === LOG ===
type D:\Projects\yttri-build\debug.log
echo === END ===
exit /b 0