@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM HQ only: PO-related data + HQ sales raw + inventory on-hand qty
REM   1) POMAS / PODET  (purchase orders)
REM   2) ICLOW          (stock-order / ค้างรับ)
REM   3) SIDET / SIMAS  (HQ sales lines + bills -> Supabase raw, latest 6 months)
REM   4) inventory sync (notebook 50 -> curated_kcw.inventory_qty_latest)
REM      Note: inventory uses BRANCH + KSS_* from .env (not ICMAS raw upload).
REM
REM Requires network reachability to HQ PARTS9 (KSS / PARTS9_HQ_*), plus Supabase DB URL.
REM For SYP, use run_syp_po_related_sync.bat on a PC that can reach KSS-PC.
REM Skip inventory: run_hq_po_related_sync.bat --skip-inventory

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

set "LOG=%KCW_ANALYTICS_LOG_DIR%\sync_hq_po_related.log"
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
echo HQ PO-related sync -^> Drive + Supabase
echo Tables: POMAS/PODET + ICLOW + SIDET/SIMAS ^(6 months^)
echo Site: hq
if "!SKIP_INVENTORY!"=="1" (
    echo Inventory: skipped
) else (
    echo Inventory: run_inventory_sync.bat after PO/ICLOW/sales
)
echo Python: %KCW_ANALYTICS_PYTHON%
echo Repo: %cd%
echo Log: %LOG%
echo ==========================================

"%KCW_ANALYTICS_PYTHON%" -c "from src.kcw import paths; print('raw_dir=', paths.raw_dir())" > "%LOG%" 2>&1

echo.
echo --- 1/2: POMAS/PODET + ICLOW + SIDET/SIMAS ^(hq^) ---
"%KCW_ANALYTICS_PYTHON%" -m src.kcw.pipeline sync-po-related --site hq >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: HQ PO/ICLOW/sales sync
    echo Check log: "%LOG%"
    echo --- last lines of sync_hq_po_related.log ---
    powershell -NoProfile -Command "Get-Content -LiteralPath '%LOG%' -Tail 40 -ErrorAction SilentlyContinue"
    echo ------------------------------------
    exit /b %ERRORLEVEL%
)
echo DONE: HQ POMAS/PODET + ICLOW + SIDET/SIMAS

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
echo ALL DONE: HQ PO-related sync
echo Check Drive raw_hq_pomas / podet / iclow / sidet / simas
echo Check Supabase raw_kcw matching tables + curated_kcw.inventory_qty_latest
exit /b 0
