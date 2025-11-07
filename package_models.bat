@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Chatty Model Package Manager
echo ==========================================
echo.

set "HF_CACHE_DIR=%USERPROFILE%\.cache\huggingface"
set "WORKSPACE_ROOT=%~dp0"

if "%1"=="extract" goto :extract
if "%1"=="package" goto :package
if "%1"=="" goto :menu

:menu
echo Choose operation:
echo 1. Package models to devpacks
echo 2. Extract devpacks to system
echo 3. Exit
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" goto :package
if "%choice%"=="2" goto :extract
if "%choice%"=="3" goto :end
echo Invalid choice, please try again.
goto :menu

:package
echo.
echo ==========================================
echo PACKAGING MODELS TO DEVPACKS
echo ==========================================
echo.

echo Checking for HuggingFace cache directory...
if not exist "%HF_CACHE_DIR%" (
    echo ERROR: HuggingFace cache directory not found at: %HF_CACHE_DIR%
    echo Please run the pipeline at least once to download models.
    goto :end
)

echo Found HuggingFace cache at: %HF_CACHE_DIR%
echo.

rem Create devpacks directory if it doesn't exist
if not exist "%WORKSPACE_ROOT%devpacks" mkdir "%WORKSPACE_ROOT%devpacks"

rem Package Kokoro TTS model
echo Packaging Kokoro TTS model...
set "KOKORO_SOURCE=%HF_CACHE_DIR%\hub\models--hexgrad--Kokoro-82M"
if exist "%KOKORO_SOURCE%" (
    powershell -Command "Compress-Archive -Path '%KOKORO_SOURCE%' -DestinationPath '%WORKSPACE_ROOT%devpacks\devpack_kokoro_tts.zip' -Force"
    echo   ✓ Created: devpack_kokoro_tts.zip
) else (
    echo   ✗ Kokoro model not found at: %KOKORO_SOURCE%
)

rem Package Whisper STT model
echo Packaging Whisper STT model...
set "WHISPER_SOURCE=%HF_CACHE_DIR%\hub\models--Systran--faster-whisper-base.en"
if exist "%WHISPER_SOURCE%" (
    powershell -Command "Compress-Archive -Path '%WHISPER_SOURCE%' -DestinationPath '%WORKSPACE_ROOT%devpacks\devpack_whisper_stt.zip' -Force"
    echo   ✓ Created: devpack_whisper_stt.zip
) else (
    echo   ✗ Whisper model not found at: %WHISPER_SOURCE%
)

rem Package any additional HuggingFace models (excluding the ones already packaged)
echo Packaging additional HuggingFace models...
set "OTHER_MODELS_TEMP=%TEMP%\chatty_other_models"
if exist "%OTHER_MODELS_TEMP%" rmdir /s /q "%OTHER_MODELS_TEMP%"
mkdir "%OTHER_MODELS_TEMP%"

xcopy "%HF_CACHE_DIR%\*" "%OTHER_MODELS_TEMP%\" /E /I /Q 2>nul
rmdir /s /q "%OTHER_MODELS_TEMP%\hub\models--hexgrad--Kokoro-82M" 2>nul
rmdir /s /q "%OTHER_MODELS_TEMP%\hub\models--Systran--faster-whisper-base.en" 2>nul

rem Check if there are any other models
dir /b "%OTHER_MODELS_TEMP%\hub\models--*" >nul 2>nul
if %errorlevel%==0 (
    powershell -Command "Compress-Archive -Path '%OTHER_MODELS_TEMP%\*' -DestinationPath '%WORKSPACE_ROOT%devpacks\devpack_other_hf_models.zip' -Force"
    echo   ✓ Created: devpack_other_hf_models.zip
) else (
    echo   ℹ No additional HuggingFace models found
)

rmdir /s /q "%OTHER_MODELS_TEMP%" 2>nul

echo.
echo ==========================================
echo PACKAGING COMPLETE
echo ==========================================
echo.
echo Devpacks created in: %WORKSPACE_ROOT%devpacks\
dir /b "%WORKSPACE_ROOT%devpacks\devpack_*.zip" 2>nul
echo.
echo To deploy on corporate network:
echo 1. Copy the devpacks folder to target system
echo 2. Run: %0 extract
echo 3. Set environment variable: set HF_HOME=%USERPROFILE%\.cache\huggingface
echo.
goto :end

:extract
echo.
echo ==========================================
echo EXTRACTING DEVPACKS TO SYSTEM
echo ==========================================
echo.

if not exist "%WORKSPACE_ROOT%devpacks" (
    echo ERROR: devpacks directory not found in workspace root
    echo Please ensure devpack files are present in: %WORKSPACE_ROOT%devpacks\
    goto :end
)

echo Creating HuggingFace cache directory structure...
if not exist "%HF_CACHE_DIR%\hub" mkdir "%HF_CACHE_DIR%\hub"

rem Extract Kokoro TTS model
if exist "%WORKSPACE_ROOT%devpacks\devpack_kokoro_tts.zip" (
    echo Extracting Kokoro TTS model...
    powershell -Command "Expand-Archive -Path '%WORKSPACE_ROOT%devpacks\devpack_kokoro_tts.zip' -DestinationPath '%HF_CACHE_DIR%\hub\' -Force"
    echo   ✓ Extracted: Kokoro TTS model
) else (
    echo   ℹ Kokoro devpack not found
)

rem Extract Whisper STT model
if exist "%WORKSPACE_ROOT%devpacks\devpack_whisper_stt.zip" (
    echo Extracting Whisper STT model...
    powershell -Command "Expand-Archive -Path '%WORKSPACE_ROOT%devpacks\devpack_whisper_stt.zip' -DestinationPath '%HF_CACHE_DIR%\hub\' -Force"
    echo   ✓ Extracted: Whisper STT model
) else (
    echo   ℹ Whisper devpack not found
)

rem Extract other HuggingFace models
if exist "%WORKSPACE_ROOT%devpacks\devpack_other_hf_models.zip" (
    echo Extracting additional HuggingFace models...
    powershell -Command "Expand-Archive -Path '%WORKSPACE_ROOT%devpacks\devpack_other_hf_models.zip' -DestinationPath '%HF_CACHE_DIR%\' -Force"
    echo   ✓ Extracted: Additional HuggingFace models
) else (
    echo   ℹ Additional models devpack not found
)

echo.
echo ==========================================
echo EXTRACTION COMPLETE
echo ==========================================
echo.
echo Models extracted to: %HF_CACHE_DIR%
echo.
echo To use offline mode, set environment variables:
echo   set HF_HOME=%HF_CACHE_DIR%
echo   set HF_HUB_OFFLINE=1
echo   set TRANSFORMERS_OFFLINE=1
echo   set HF_DATASETS_OFFLINE=1
echo.
echo Or add to your batch file / system environment variables for persistence.
echo For immediate use, you can run:
echo   set HF_HUB_OFFLINE=1 ^& python pipeline.py
echo.
goto :end

:end
echo Script completed.
pause