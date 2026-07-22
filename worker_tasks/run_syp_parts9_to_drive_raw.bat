@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM SYP: PARTS9 (local SQL) -> Google Drive 01_raw (raw_syp_*.csv)
REM Runs the original-style notebook via nbconvert (same approach as your
REM old local BAT that never raised Errno 9 / false verify failures).
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

set "NBDIR=%cd%\notebooks"
set "NBNAME=51_syp_parts9_to_drive_raw.ipynb"
set "NB=%NBDIR%\%NBNAME%"
set "OUT=%KCW_ANALYTICS_LOG_DIR%\%NBNAME:.ipynb=.executed.ipynb%"
set "LOG=%KCW_ANALYTICS_LOG_DIR%\extract_syp.log"

echo ==========================================
echo SYP PARTS9 -^> Drive raw (notebook, original-style)
echo Python: %KCW_ANALYTICS_PYTHON%
echo Notebook: %NB%
echo Log: %LOG%
echo ==========================================

if not exist "%NB%" (
    echo FAILED: notebook not found "%NB%"
    exit /b 1
)

"%KCW_ANALYTICS_PYTHON%" -m jupyter nbconvert ^
    --to notebook ^
    --execute ^
    --ExecutePreprocessor.kernel_name=python3 ^
    "%NB%" ^
    --output "%OUT%" > "%LOG%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo FAILED: SYP extract notebook
    echo Check log: "%LOG%"
    exit /b %ERRORLEVEL%
)

echo DONE: SYP extract
echo Check Drive timestamps for all five raw_syp_*.csv files
exit /b 0
