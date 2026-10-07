@echo off
rem Автономный A/B декод-ядра DeltaNet: боевые раскладки против приёма llama.cpp
rem и tile-варианта (транспозиция состояния через shared).
rem
rem Стенд #include-ит рабочий delta_rule_batched.cu, поэтому лежит вне src/.
rem ВНИМАНИЕ: обязательна проверка ошибок запуска (chk). Без неё молчаливый
rem отказ запуска при 64 КБ dynamic shared даёт ложные 11x.
rem
rem Использование: build_delta_decode_tile_ab.bat <путь к candle-kernels/src>
setlocal
if "%~1"=="" ( echo usage: %~nx0 ^<path to candle-kernels/src^> & exit /b 2 )
set "SRC=%~1"
if not exist "%SRC%\delta_rule_batched.cu" ( echo no delta_rule_batched.cu in %SRC% & exit /b 2 )
set "VC="
for %%R in ("C:\Program Files\Microsoft Visual Studio\2022" "C:\Program Files (x86)\Microsoft Visual Studio\2022") do (
  for /d %%E in ("%%~R\*") do if exist "%%~E\VC\Auxiliary\Build\vcvars64.bat" set "VC=%%~E\VC\Auxiliary\Build\vcvars64.bat"
)
if "%VC%"=="" ( echo NO_VCVARS & exit /b 3 )
call "%VC%" >nul || ( echo VCVARS_FAILED & exit /b 4 )
nvcc -std=c++17 -arch=sm_86 -O3 -I"%SRC%" -o "%~dp0delta_decode_tile_ab.exe" "%~dp0delta_decode_tile_ab.cu"
if errorlevel 1 ( echo NVCC_FAILED & exit /b 5 )
"%~dp0delta_decode_tile_ab.exe"
echo RUN_RC=%ERRORLEVEL%
endlocal
