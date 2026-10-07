@echo off
rem Автономная GPU-проверка покрытия столбцов в delta_rule_kernel_batched_splitc.
rem Собирает delta_splitc_coverage.cu (он #include-ит рабочий delta_rule_batched.cu)
rem и сравнивает ядро с последовательным CPU-эталоном.
rem
rem Использование:  build_delta_splitc_coverage.bat <путь к candle-kernels/src>
rem Пример:        build_delta_splitc_coverage.bat D:\Projects\yttri-forge\engine\candle-kernels\src
setlocal
if "%~1"=="" ( echo usage: %~nx0 ^<path to candle-kernels/src^> & exit /b 2 )
set "SRC=%~1"
if not exist "%SRC%\delta_rule_batched.cu" ( echo no delta_rule_batched.cu in %SRC% & exit /b 2 )

echo [1] find vcvars64.bat
set "VC="
for %%R in ("C:\Program Files\Microsoft Visual Studio\2022" "C:\Program Files (x86)\Microsoft Visual Studio\2022") do (
  for /d %%E in ("%%~R\*") do if exist "%%~E\VC\Auxiliary\Build\vcvars64.bat" set "VC=%%~E\VC\Auxiliary\Build\vcvars64.bat"
)
if "%VC%"=="" ( echo NO_VCVARS & exit /b 3 )
echo     VC=%VC%

echo [2] init MSVC env
call "%VC%" >nul || ( echo VCVARS_FAILED & exit /b 4 )

echo [3] nvcc build
nvcc -std=c++17 -arch=sm_86 -I"%SRC%" -o "%~dp0delta_splitc_coverage.exe" "%~dp0delta_splitc_coverage.cu"
if errorlevel 1 ( echo NVCC_FAILED & exit /b 5 )

echo [4] run
"%~dp0delta_splitc_coverage.exe"
echo RUN_RC=%ERRORLEVEL%
endlocal
