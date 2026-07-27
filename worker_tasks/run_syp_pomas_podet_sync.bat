@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM SYP: PARTS9 POMAS/PODET -> Drive 01_raw -> Supabase raw_kcw
REM Focused PO sync (does not extract sales/ICMAS).
REM Requires SUPABASE_DB_URL (or DB_PASSWORD + host vars) on this machine.

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

set "LOG=%KCW_ANALYTICS_LOG_DIR%\sync_syp_pomas_podet.log"

echo ==========================================
echo SYP POMAS/PODET sync -^> Drive + Supabase
echo Python: %KCW_ANALYTICS_PYTHON%
echo Repo: %cd%
echo Log: %LOG%
echo ==========================================

"%KCW_ANALYTICS_PYTHON%" -c "from src.kcw import paths; print('raw_dir=', paths.raw_dir())" > "%LOG%" 2>&1

"%KCW_ANALYTICS_PYTHON%" -m src.kcw.pipeline sync-pomas-podet --site syp >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: SYP POMAS/PODET sync
    echo Check log: "%LOG%"
    exit /b %ERRORLEVEL%
)

echo DONE: SYP POMAS/PODET sync
echo Check Drive raw_syp_pomas_purchase_orders.csv / raw_syp_podet_purchase_order_lines.csv
echo Check Supabase raw_kcw.raw_syp_pomas_purchase_orders / raw_syp_podet_purchase_order_lines
exit /b 0
