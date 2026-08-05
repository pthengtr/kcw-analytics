@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM HQ: PARTS9 SIDET/SIMAS -> Drive 01_raw -> Supabase raw_kcw (latest 6 months)
REM Focused sales sync (HQ only; SYP sales stay on Drive for curated notebooks).

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

set "LOG=%KCW_ANALYTICS_LOG_DIR%\sync_hq_simas_sidet.log"

echo ==========================================
echo HQ SIMAS/SIDET sync -^> Drive + Supabase
echo Window: latest 6 months from max BILLDATE
echo Python: %KCW_ANALYTICS_PYTHON%
echo Repo: %cd%
echo Log: %LOG%
echo ==========================================

"%KCW_ANALYTICS_PYTHON%" -c "from src.kcw import paths; print('raw_dir=', paths.raw_dir())" > "%LOG%" 2>&1

"%KCW_ANALYTICS_PYTHON%" -m src.kcw.pipeline sync-simas-sidet >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: HQ SIMAS/SIDET sync
    echo Check log: "%LOG%"
    exit /b %ERRORLEVEL%
)

echo DONE: HQ SIMAS/SIDET sync
echo Check Drive raw_hq_sidet_sales_lines.csv / raw_hq_simas_sales_bills.csv
echo Check Supabase raw_kcw.raw_hq_sidet_sales_lines / raw_hq_simas_sales_bills
exit /b 0
