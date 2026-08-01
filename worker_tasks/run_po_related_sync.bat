@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM One trigger: PO-related data for both HQ and SYP
REM   - POMAS / PODET  (purchase orders)
REM   - ICMAS          (product / stock masters)
REM   - ICLOW          (stock-order / pending-receive tracker, ค้างรับ)
REM PARTS9 -> Drive 01_raw -> Supabase raw_kcw
REM
REM Requires network reachability to BOTH PARTS9 servers from this machine
REM (HQ: KSS / PARTS9_HQ_*; SYP: KSS-PC / PARTS9_SYP_*), plus Supabase DB URL.
REM To sync one site only, pass --site hq or --site syp after this BAT, or run:
REM   python -m src.kcw.pipeline sync-po-related --site hq

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

set "LOG=%KCW_ANALYTICS_LOG_DIR%\sync_po_related.log"
set "SITE_ARG="
if /I "%~1"=="--site" (
    if /I "%~2"=="hq" set "SITE_ARG=--site hq"
    if /I "%~2"=="syp" set "SITE_ARG=--site syp"
    if "!SITE_ARG!"=="" (
        echo Usage: %~nx0 [--site hq^|syp]
        echo Default ^(no args^): sync both HQ then SYP
        exit /b 1
    )
)

echo ==========================================
echo PO-related sync -^> Drive + Supabase
echo Tables: POMAS/PODET + ICMAS + ICLOW
if "!SITE_ARG!"=="" (
    echo Sites: HQ then SYP
) else (
    echo Sites: %~2
)
echo Python: %KCW_ANALYTICS_PYTHON%
echo Repo: %cd%
echo Log: %LOG%
echo ==========================================

"%KCW_ANALYTICS_PYTHON%" -c "from src.kcw import paths; print('raw_dir=', paths.raw_dir())" > "%LOG%" 2>&1

"%KCW_ANALYTICS_PYTHON%" -m src.kcw.pipeline sync-po-related !SITE_ARG! >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: PO-related sync
    echo Check log: "%LOG%"
    echo --- last lines of sync_po_related.log ---
    powershell -NoProfile -Command "Get-Content -LiteralPath '%LOG%' -Tail 40 -ErrorAction SilentlyContinue"
    echo ------------------------------------
    exit /b %ERRORLEVEL%
)

echo DONE: PO-related sync
echo Check Drive raw_{hq,syp}_pomas_purchase_orders.csv / podet / icmas_products / iclow_stock_orders
echo Check Supabase raw_kcw matching tables for both sites
exit /b 0
