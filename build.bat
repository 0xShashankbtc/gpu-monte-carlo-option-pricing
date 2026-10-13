@echo off
setlocal

echo ====================================================================
echo  Building GPU Monte-Carlo Option Pricing and Portfolio Risk Analysis
echo ====================================================================

if not exist "C:\Temp" mkdir "C:\Temp"
set "TEMP=C:\Temp"
set "TMP=C:\Temp"

set "MSVC_BIN=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64"
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
set "PATH=%MSVC_BIN%;%CUDA_BIN%;%PATH%"

echo Compiler:
where nvcc
where cl

echo.
echo Compiling CUDA source files...
nvcc -O3 -std=c++17 -Xcompiler "/Zc:preprocessor /EHsc" -arch=sm_86 ^
    src\main.cu src\option_mc.cu src\portfolio_mc.cu ^
    -I src -I include ^
    -o option_pricer.exe

if %ERRORLEVEL% equ 0 (
    echo.
    echo ====================================================================
    echo  BUILD SUCCESSFUL: option_pricer.exe created!
    echo ====================================================================
) else (
    echo.
    echo ====================================================================
    echo  BUILD FAILED! Please check the error messages above.
    echo ====================================================================
    exit /b %ERRORLEVEL%
)

endlocal
