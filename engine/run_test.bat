@echo off
setlocal

set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;%PATH%"
set "QWEN36_MODEL=D:\Models\unsloth\Qwen3.6-27B-GGUF\Qwen3.6-27B-UD-Q2_K_XL.gguf"
set "QWEN36_SLOTS=1"
set "QWEN36_CTX=2048"
set "QWEN36_PORT=18099"
set "QWEN36_API_KEY=test"
set "OUTFILE=D:\Projects\yttri-build\test_results.txt"
set "LOGFILE=D:\Projects\yttri-build\server_test_log.txt"

if exist "%OUTFILE%" del "%OUTFILE%"
if exist "%LOGFILE%" del "%LOGFILE%"

echo === Starting server (bg, 90s timeout) === > "%OUTFILE%"
echo QWEN36_MODEL=%QWEN36_MODEL% >> "%OUTFILE%"
echo QWEN36_SLOTS=%QWEN36_SLOTS% >> "%OUTFILE%"
echo QWEN36_CTX=%QWEN36_CTX% >> "%OUTFILE%"
echo QWEN36_API_KEY=%QWEN36_API_KEY% >> "%OUTFILE%"
echo. >> "%OUTFILE%"

start /b "" cmd /c "D:\Projects\yttri-build\qwen36-server\target\release\qwen36-server.exe > %LOGFILE% 2>&1"

set /a WAIT=0
:waitloop
timeout /t 3 /nobreak > nul
set /a WAIT+=3
findstr /c:"loaded:" "%LOGFILE%" > nul 2>&1
if %errorlevel% equ 0 (
    echo Model loaded after %WAIT%s >> "%OUTFILE%"
    goto :test
)
tasklist /fi "imagename eq qwen36-server.exe" 2>nul | findstr /i "qwen36-server" > nul
if %errorlevel% neq 0 (
    echo Server process died after %WAIT%s >> "%OUTFILE%"
    goto :showlog
)
if %WAIT% geq 90 (
    echo TIMEOUT: model did not load in %WAIT%s >> "%OUTFILE%"
    goto :showlog
)
goto :waitloop

:test
echo. >> "%OUTFILE%"
echo === API Request 1: simple chat === >> "%OUTFILE%"
curl -s --max-time 60 http://localhost:18099/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer test" -d "{\"model\":\"qwen36\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word\"}],\"max_tokens\":5,\"temperature\":0}" >> "%OUTFILE%" 2>&1
echo. >> "%OUTFILE%"
echo. >> "%OUTFILE%"

echo === API Request 2: longer generation (32 tokens) === >> "%OUTFILE%"
curl -s --max-time 120 http://localhost:18099/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer test" -d "{\"model\":\"qwen36\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a short story about a robot.\"}],\"max_tokens\":32,\"temperature\":0.3}" >> "%OUTFILE%" 2>&1
echo. >> "%OUTFILE%"
echo. >> "%OUTFILE%"

:showlog
echo. >> "%OUTFILE%"
echo === Server log (full) === >> "%OUTFILE%"
type "%LOGFILE%" >> "%OUTFILE%" 2>&1

taskkill /f /im qwen36-server.exe > nul 2>&1
echo. >> "%OUTFILE%"
echo === Server killed === >> "%OUTFILE%"

type "%OUTFILE%"
