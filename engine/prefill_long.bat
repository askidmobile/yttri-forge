@echo off
cd /d D:\Projects\yttri-build
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;%PATH%"
set "QWEN36_MODEL=D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q2_K_XL.gguf"
set QWEN36_API_KEY=test
set QWEN36_SLOTS=1
set QWEN36_CTX=8192
set QWEN36_PORT=18099
set RUST_BACKTRACE=1
del server_prefill.log 2>nul
start "srv" /b cmd /c "qwen36-server\target\release\qwen36-server.exe >server_prefill.log 2>&1"
echo launched with ctx=8192, waiting for load...
set /a tries=0
:wait
timeout /t 3 /nobreak >nul
set /a tries+=1
findstr /i /c:"loaded:" server_prefill.log >nul 2>&1
if not errorlevel 1 goto loaded
findstr /i /c:"panic" server_prefill.log >nul 2>&1
if not errorlevel 1 goto panic
if %tries% lss 80 goto wait
echo TIMEOUT after %tries% tries
goto showlog
:loaded
echo === LOADED ===
timeout /t 2 /nobreak >nul
echo === VRAM BEFORE ===
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader
echo === CURL LONG PROMPT (600s) ===
curl -s -m 600 http://localhost:18099/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer test" -d @longprompt.json
echo.
echo === CURL EXIT: %ERRORLEVEL% ===
echo === VRAM AFTER ===
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader
echo === LOG TAIL ===
powershell -NoProfile -Command "Get-Content server_prefill.log -Tail 50"
goto kill
:panic
echo === PANIC ===
:showlog
type server_prefill.log
:kill
taskkill /f /im qwen36-server.exe >nul 2>&1
echo done