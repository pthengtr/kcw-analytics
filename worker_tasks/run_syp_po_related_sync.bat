@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM SYP only: PO-related data + inventory on-hand qty
REM   1) POMAS / PODET  (purchase orders)
REM   2) ICLOW          (stock-order / ค้างรับ)
REM   3) inventory sync (notebook 50 -> curated_kcw.inventory_qty_latest)
REM      Note: inventory uses BRANCH + KSS_* from .env (not ICMAS raw upload).
REM
REM Requires network reachability to SYP PARTS9 (KSS-PC / PARTS9_SYP_*), plus Supabase DB URL.
REM For HQ, use run_hq_po_related_sync.bat on a PC that can reach KSS.
REM Skip inventory: run_syp_po_related_sync.bat --skip-inventory

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

set "LOG=%KCW_ANALYTICS_LOG_DIR%\sync_syp_po_related.log"
set "SKIP_INVENTORY=0"

:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="--skip-inventory" (
    set "SKIP_INVENTORY=1"
    shift
    goto :parse_args
)
echo Unknown argument: %~1
echo Usage: %~nx0 [--skip-inventory]
exit /b 1

:args_done

echo ==========================================
echo SYP PO-related sync -^> Drive + Supabase
echo Tables: POMAS/PODET + ICLOW
echo Site: syp
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
echo --- 1/2: POMAS/PODET + ICLOW ^(syp^) ---
"%KCW_ANALYTICS_PYTHON%" -m src.kcw.pipeline sync-po-related --site syp >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: SYP PO/ICLOW sync
    echo Check log: "%LOG%"
    echo --- last lines of sync_syp_po_related.log ---
    powershell -NoProfile -Command "Get-Content -LiteralPath '%LOG%' -Tail 40 -ErrorAction SilentlyContinue"
    echo ------------------------------------
    exit /b %ERRORLEVEL%
)
echo DONE: SYP POMAS/PODET + ICLOW

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
echo ALL DONE: SYP PO-related sync
echo Check Drive raw_syp_pomas / podet / iclow_stock_orders
echo Check Supabase raw_kcw matching tables + curated_kcw.inventory_qty_latest
exit /b 0
