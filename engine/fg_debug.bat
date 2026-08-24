@echo off
REM Foreground run with line-buffered stderr redirect. Runs 30s max, then kills.
REM Captures panic/crash output. SSH-friendly (always returns).
taskkill /f /im qwen36-server.exe 2>nul
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;%PATH%"
set "QWEN36_MODEL=D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q2_K_XL.gguf"
set QWEN36_SLOTS=1
set QWEN36_CTX=2048
set QWEN36_PORT=18099
set QWEN36_API_KEY=test
set RUST_BACKTRACE=full
del /q D:\Projects\yttri-build\fg.log 2>nul
echo Starting server (30s capture)...
start /b cmd /c "D:\Projects\yttri-build\qwen36-server\target\release\qwen36-server.exe 1>D:\Projects\yttri-build\fg.log 2>&1"
timeout /t 30 /nobreak >nul
echo === PROCESS ===
tasklist /fi "imagename eq qwen36-server.exe" 2>nul
echo === LOG ===
type D:\Projects\yttri-build\fg.log
echo === END ===
taskkill /f /im qwen36-server.exe 2>nul
exit /b 0