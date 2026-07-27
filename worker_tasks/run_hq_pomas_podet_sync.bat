@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM HQ: PARTS9 POMAS/PODET -> Drive 01_raw -> Supabase raw_kcw
REM Focused PO sync (does not extract sales/ICMAS/PIMAS/etc).

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

set "LOG=%KCW_ANALYTICS_LOG_DIR%\sync_hq_pomas_podet.log"

echo ==========================================
echo HQ POMAS/PODET sync -^> Drive + Supabase
echo Python: %KCW_ANALYTICS_PYTHON%
echo Repo: %cd%
echo Log: %LOG%
echo ==========================================

"%KCW_ANALYTICS_PYTHON%" -c "from src.kcw import paths; print('raw_dir=', paths.raw_dir())" > "%LOG%" 2>&1

"%KCW_ANALYTICS_PYTHON%" -m src.kcw.pipeline sync-pomas-podet --site hq >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: HQ POMAS/PODET sync
    echo Check log: "%LOG%"
    exit /b %ERRORLEVEL%
)

echo DONE: HQ POMAS/PODET sync
echo Check Drive raw_hq_pomas_purchase_orders.csv / raw_hq_podet_purchase_order_lines.csv
echo Check Supabase raw_kcw.raw_hq_pomas_purchase_orders / raw_hq_podet_purchase_order_lines
exit /b 0
