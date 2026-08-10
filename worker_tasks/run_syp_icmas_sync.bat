@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM SYP: PARTS9 ICMAS -> Drive 01_raw -> Supabase raw_kcw
REM Product masters (raw_syp_icmas_products).
REM Focused sync (does not extract sales/ICLOW/PIMAS/etc).
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

set "LOG=%KCW_ANALYTICS_LOG_DIR%\sync_syp_icmas.log"

echo ==========================================
echo SYP ICMAS sync -^> Drive + Supabase
echo Python: %KCW_ANALYTICS_PYTHON%
echo Repo: %cd%
echo Log: %LOG%
echo ==========================================

"%KCW_ANALYTICS_PYTHON%" -c "from src.kcw import paths; print('raw_dir=', paths.raw_dir())" > "%LOG%" 2>&1

"%KCW_ANALYTICS_PYTHON%" -m src.kcw.pipeline sync-icmas --site syp >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: SYP ICMAS sync
    echo Check log: "%LOG%"
    exit /b %ERRORLEVEL%
)

echo DONE: SYP ICMAS sync
echo Check Drive raw_syp_icmas_products.csv
echo Check Supabase raw_kcw.raw_syp_icmas_products
exit /b 0
