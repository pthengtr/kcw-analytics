@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM SYP: PARTS9 (local SQL) -> Google Drive 01_raw (raw_syp_*.csv)
REM Schedule this BEFORE the HQ full pipeline.

cd /d "%~dp0.."

for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if /I "%%A"=="KCW_ANALYTICS_PYTHON" set "KCW_ANALYTICS_PYTHON=%%B"
    if /I "%%A"=="KCW_ANALYTICS_LOG_DIR" set "KCW_ANALYTICS_LOG_DIR=%%B"
    if /I "%%A"=="KCW_DRIVE_ROOT" set "KCW_DRIVE_ROOT=%%B"
)

if "%KCW_ANALYTICS_PYTHON%"=="" (
    echo Missing KCW_ANALYTICS_PYTHON in .env
    exit /b 1
)

if "%KCW_ANALYTICS_LOG_DIR%"=="" set "KCW_ANALYTICS_LOG_DIR=%cd%\logs"
if not exist "%KCW_ANALYTICS_LOG_DIR%" mkdir "%KCW_ANALYTICS_LOG_DIR%"

set "LOG=%KCW_ANALYTICS_LOG_DIR%\extract_syp.log"

echo ==========================================
echo SYP PARTS9 -^> Drive raw
echo Python: %KCW_ANALYTICS_PYTHON%
echo Repo: %cd%
echo Log: %LOG%
echo ==========================================

"%KCW_ANALYTICS_PYTHON%" -m src.kcw.pipeline extract --site syp > "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: SYP extract
    echo Check log: "%LOG%"
    exit /b %ERRORLEVEL%
)

echo DONE: SYP extract
exit /b 0
