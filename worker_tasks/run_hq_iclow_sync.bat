@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM HQ: PARTS9 ICLOW -> Drive 01_raw -> Supabase raw_kcw
REM Stock-order / pending-receive tracker (ค้างรับ = ORDERED=Y RECEIVED=N).
REM Focused sync (does not extract sales/ICMAS/PIMAS/etc).

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

set "LOG=%KCW_ANALYTICS_LOG_DIR%\sync_hq_iclow.log"

echo ==========================================
echo HQ ICLOW sync -^> Drive + Supabase
echo Python: %KCW_ANALYTICS_PYTHON%
echo Repo: %cd%
echo Log: %LOG%
echo ==========================================

"%KCW_ANALYTICS_PYTHON%" -c "from src.kcw import paths; print('raw_dir=', paths.raw_dir())" > "%LOG%" 2>&1

"%KCW_ANALYTICS_PYTHON%" -m src.kcw.pipeline sync-iclow >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: HQ ICLOW sync
    echo Check log: "%LOG%"
    exit /b %ERRORLEVEL%
)

echo DONE: HQ ICLOW sync
echo Check Drive raw_hq_iclow_stock_orders.csv
echo Check Supabase raw_kcw.raw_hq_iclow_stock_orders
exit /b 0
