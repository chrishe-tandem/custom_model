@echo off
REM Setup script for XGBoost training conda environment (Windows)

set ENV_NAME=xgboost_training
set ENV_FILE=environment.yml

echo ==========================================
echo Setting up Conda Environment
echo ==========================================

REM Check if conda is installed
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: conda is not installed or not in PATH
    echo Please install Miniconda or Anaconda first
    exit /b 1
)

echo Conda version:
conda --version

REM Check if environment already exists
conda env list | findstr /C:"%ENV_NAME%" >nul
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Environment '%ENV_NAME%' already exists.
    set /p RECREATE="Do you want to remove it and recreate? (y/n): "
    if /i "%RECREATE%"=="y" (
        echo Removing existing environment...
        conda env remove -n %ENV_NAME% -y
    ) else (
        echo Keeping existing environment. Activate with: conda activate %ENV_NAME%
        exit /b 0
    )
)

REM Create environment from yml file
echo.
echo Creating conda environment from %ENV_FILE%...
conda env create -f %ENV_FILE%

echo.
echo ==========================================
echo Environment created successfully!
echo ==========================================
echo.
echo To activate the environment, run:
echo   conda activate %ENV_NAME%
echo.
echo To deactivate, run:
echo   conda deactivate
echo.

pause

