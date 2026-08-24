@echo off
REM Download IQ2_XXS model from HuggingFace, then launch server detached.
REM Step 1: Download if not exists.
set "MODEL_DIR=D:\Models\unsloth\Qwen3.6-27B-GGUF"
set "MODEL_FILE=Qwen3.6-27B-UD-IQ2_XXS.gguf"
set "MODEL_PATH=%MODEL_DIR%\%MODEL_FILE%"

if exist "%MODEL_PATH%" (
    echo Model already exists: %MODEL_PATH%
    goto launch
)

echo Downloading %MODEL_FILE% ...
mkdir "%MODEL_DIR%" 2>nul
C:\Windows\System32\curl.exe -L -o "%MODEL_PATH%" "https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-UD-IQ2_XXS.gguf"
echo Download exit code: %errorlevel%
dir "%MODEL_PATH%"

:launch
REM Step 2: Kill old server, launch with IQ2_XXS detached.
taskkill /f /im qwen36-server.exe 2>nul
ping -n 4 localhost >nul
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;%PATH%"
set "QWEN36_MODEL=%MODEL_PATH%"
set QWEN36_SLOTS=1
set QWEN36_CTX=2048
set QWEN36_PORT=18099
set QWEN36_API_KEY=test
set RUST_BACKTRACE=full
del /q D:\Projects\yttri-build\iq2xxs.log 2>nul
start "q36" /b cmd /c "D:\Projects\yttri-build\qwen36-server\target\release\qwen36-server.exe >D:\Projects\yttri-build\iq2xxs.log 2>&1"
echo Server launched. PID:
tasklist /fi "imagename eq qwen36-server.exe" 2>nul
exit /b 0