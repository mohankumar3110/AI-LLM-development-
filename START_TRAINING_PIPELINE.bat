@echo off
REM 🚀 LLAMA BLENDER TRAINING - COMPLETE PIPELINE
REM This script runs the entire training process

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ================================================================================
echo 🚀 LLAMA BLENDER COMPLETE TRAINING PIPELINE
echo ================================================================================
echo.

REM Run the complete training pipeline
python RUN_COMPLETE_TRAINING.py

if errorlevel 1 (
    echo.
    echo ❌ Training pipeline failed
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo ✅ Training complete!
    echo.
    pause
)
