@echo off
echo ==========================================
echo Chatty Voice Assistant (Offline Mode)
echo ==========================================
echo.

rem Set environment variables for offline operation
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1
set HF_HUB_DISABLE_PROGRESS_BARS=1
set HF_HUB_DISABLE_TELEMETRY=1

rem Verify models are available
set "HF_CACHE_DIR=%USERPROFILE%\.cache\huggingface"
if not exist "%HF_CACHE_DIR%\hub\models--hexgrad--Kokoro-82M" (
    echo ERROR: Kokoro TTS models not found in local cache
    echo Expected location: %HF_CACHE_DIR%\hub\models--hexgrad--Kokoro-82M
    echo.
    echo Please run 'package_models.bat extract' first to install offline models
    echo or copy the devpacks folder from a system with internet access.
    echo.
    pause
    exit /b 1
)

if not exist "%HF_CACHE_DIR%\hub\models--Systran--faster-whisper-base.en" (
    echo ERROR: Whisper STT models not found in local cache
    echo Expected location: %HF_CACHE_DIR%\hub\models--Systran--faster-whisper-base.en
    echo.
    echo Please run 'package_models.bat extract' first to install offline models
    echo or copy the devpacks folder from a system with internet access.
    echo.
    pause
    exit /b 1
)

echo ✓ Kokoro TTS models found
echo ✓ Whisper STT models found
echo ✓ Offline mode enabled
echo.

rem Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo WARNING: No virtual environment found. Using system Python.
    echo.
)

rem Start the pipeline
echo Starting Chatty pipeline in offline mode...
echo.
python pipeline.py

pause