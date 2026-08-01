@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM One trigger: PO-related data for both HQ and SYP + inventory on-hand qty
REM   1) POMAS / PODET  (purchase orders)     — both sites
REM   2) ICLOW          (stock-order / ค้างรับ) — both sites
REM   3) inventory sync (notebook 50 -> curated_kcw.inventory_qty_latest)
REM      Note: inventory uses BRANCH + KSS_* from .env (not ICMAS raw upload).
REM
REM Requires network reachability to BOTH PARTS9 servers for steps 1-2
REM (HQ: KSS / PARTS9_HQ_*; SYP: KSS-PC / PARTS9_SYP_*), plus Supabase DB URL.
REM To sync PO/ICLOW for one site only:
REM   run_po_related_sync.bat --site hq
REM   (inventory sync still runs afterward unless you skip with --skip-inventory)

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
set "SKIP_INVENTORY=0"

:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="--site" (
    if /I "%~2"=="hq" set "SITE_ARG=--site hq"
    if /I "%~2"=="syp" set "SITE_ARG=--site syp"
    if "!SITE_ARG!"=="" (
        echo Usage: %~nx0 [--site hq^|syp] [--skip-inventory]
        echo Default ^(no args^): sync PO+ICLOW for both HQ then SYP, then inventory
        exit /b 1
    )
    shift
    shift
    goto :parse_args
)
if /I "%~1"=="--skip-inventory" (
    set "SKIP_INVENTORY=1"
    shift
    goto :parse_args
)
echo Unknown argument: %~1
echo Usage: %~nx0 [--site hq^|syp] [--skip-inventory]
exit /b 1

:args_done

echo ==========================================
echo PO-related sync -^> Drive + Supabase
echo Tables: POMAS/PODET + ICLOW
if "!SITE_ARG!"=="" (
    echo Sites: HQ then SYP
) else (
    echo Sites: !SITE_ARG:--site =!
)
if "!SKIP_INVENTORY!"=="1" (
    echo Inventory: skipped
) else (
    echo Inventory: run_inventory_sync.bat after PO/ICLOW
)
echo Python: %KCW_ANALYTICS_PYTHON%
echo Repo: %cd%
echo Log: %LOG%
echo ==========================================

"%KCW_ANALYTICS_PYTHON%" -c "from src.kcw import paths; print('raw_dir=', paths.raw_dir())" > "%LOG%" 2>&1

echo.
echo --- 1/2: POMAS/PODET + ICLOW ---
"%KCW_ANALYTICS_PYTHON%" -m src.kcw.pipeline sync-po-related !SITE_ARG! >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: PO/ICLOW sync
    echo Check log: "%LOG%"
    echo --- last lines of sync_po_related.log ---
    powershell -NoProfile -Command "Get-Content -LiteralPath '%LOG%' -Tail 40 -ErrorAction SilentlyContinue"
    echo ------------------------------------
    exit /b %ERRORLEVEL%
)
echo DONE: POMAS/PODET + ICLOW

if "!SKIP_INVENTORY!"=="1" goto :all_done

echo.
echo --- 2/2: inventory on-hand qty ---
call "%~dp0run_inventory_sync.bat"
if errorlevel 1 (
    echo FAILED: inventory sync
    exit /b 1
)

:all_done
echo.
echo ALL DONE: PO-related sync
echo Check Drive raw_{hq,syp}_pomas / podet / iclow_stock_orders
echo Check Supabase raw_kcw matching tables + curated_kcw.inventory_qty_latest
exit /b 0
