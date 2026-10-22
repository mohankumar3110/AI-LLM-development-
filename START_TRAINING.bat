@echo off
REM ========================================================================
REM   Llama 3.2 Fine-tuning Master Pipeline
REM   Run this file to train your model automatically
REM ========================================================================

echo.
echo ========================================================================
echo   LLAMA 3.2 FINE-TUNING - AUTOMATIC SETUP & TRAINING
echo ========================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python from: https://www.python.org/
    pause
    exit /b 1
)

echo Detected Python version:
python --version
echo.

REM Navigate to script directory
cd /d "%~dp0"

REM Install requirements if not already installed
echo.
echo ========================================================================
echo STEP 1: Installing/Updating Requirements
echo ========================================================================
pip install -r requirements.txt -q
echo Done!

REM Run master pipeline
echo.
echo ========================================================================
echo STEP 2: Running Training Pipeline
echo ========================================================================
echo.
python MASTER_PIPELINE.py

REM Show final message
echo.
echo ========================================================================
echo TRAINING COMPLETE!
echo ========================================================================
echo.
echo Your trained model is saved in: checkpoints/final_adapter/
echo.
echo Next steps:
echo 1. Test the model: python inference.py "Your prompt here"
echo 2. Evaluate results: python evaluate.py
echo 3. Check logs: type logs\train.log
echo.
echo Press any key to close this window...
pause
