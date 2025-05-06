@echo off
REM Exit on error
setlocal enabledelayedexpansion
set ERRLEV=0

REM Set environment name 
set ENV_NAME=proj_4228

REM Create the conda environment
call conda create -n %ENV_NAME% python=3.10 -y
if errorlevel 1 goto :error

REM Activate the environment
call conda activate %ENV_NAME%
if errorlevel 1 goto :error

REM Install metadrive in editable mode
if exist metadrive (
    cd metadrive
    pip install -e .
    cd ..
) else (
    echo Directory 'metadrive' not found!
    goto :error
)

REM Install remaining dependencies
pip install -r requirements.txt
if errorlevel 1 goto :error

echo Environment '%ENV_NAME%' setup complete.
goto :eof

:error
echo Error occurred during environment setup.
exit /b 1
