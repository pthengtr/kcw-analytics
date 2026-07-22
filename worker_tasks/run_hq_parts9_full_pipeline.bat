@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM HQ B: raw extract (HQ A) + full daily notebook pipeline
REM Default Task Scheduler entry on HQ until Claude Cowork cutover.
REM Schedule AFTER SYP raw BAT so raw_syp_*.csv already exist on Drive.

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
set "PY=%KCW_ANALYTICS_PYTHON%"

echo ==========================================
echo HQ full pipeline (raw + notebooks)
echo Python: %PY%
echo Repo: %cd%
echo Notebooks: %NBDIR%
echo Logs: %KCW_ANALYTICS_LOG_DIR%
echo ==========================================

echo.
echo --- HQ A: extract raw ---
call "%~dp0run_hq_parts9_to_drive_raw.bat"
if errorlevel 1 goto :fail

call :run_nb "00_archive_output.ipynb" fail
if errorlevel 1 goto :fail

REM 51 still builds curated facts/dims (and re-reads/writes HQ raw). Safe during transition.
call :run_nb "51_parts9_to_drive.ipynb" fail
if errorlevel 1 goto :fail

call :run_nb "20_vat_sales_nonvat_purchase_report.ipynb" fail
if errorlevel 1 goto :fail

REM Prefer CLI catch-up (idempotent). Falls back to notebook if CLI fails.
echo.
echo ------------------------------------------
echo Running: TAR catch-up (CLI)
echo ------------------------------------------
set "TAR_LOG=%KCW_ANALYTICS_LOG_DIR%\tar_catchup.log"
set "PYTHONPATH=%cd%;%PYTHONPATH%"
"%PY%" -m src.kcw.pipeline tar --catch-up > "%TAR_LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo CLI TAR catch-up failed - falling back to notebook
    echo Check log: "%TAR_LOG%"
    echo --- last lines of tar_catchup.log ---
    powershell -NoProfile -Command "Get-Content -LiteralPath '%TAR_LOG%' -Tail 40 -ErrorAction SilentlyContinue"
    echo ------------------------------------
    call :run_nb "21_tar_daily_supabase.ipynb" continue
) else (
    echo DONE: TAR catch-up CLI
)

call :run_nb "21_tar_daily_report.ipynb" continue

call :run_nb "30_generate_bills_summary.ipynb" fail
if errorlevel 1 goto :fail

call :run_nb "31_vat_purchase_report_excel.ipynb" fail
if errorlevel 1 goto :fail

call :run_nb "32_vat_sales_report_excel.ipynb" fail
if errorlevel 1 goto :fail

call :run_nb "33_ar_ap_report.ipynb" fail
if errorlevel 1 goto :fail

call :run_nb "90_csv_to_supabase.ipynb" continue

echo.
echo ALL DONE.
exit /b 0


:run_nb
set "NBNAME=%~1"
set "MODE=%~2"
set "NB=%NBDIR%\%NBNAME%"
set "OUT=%KCW_ANALYTICS_LOG_DIR%\%NBNAME:.ipynb=.executed.ipynb%"
set "LOG=%KCW_ANALYTICS_LOG_DIR%\%NBNAME:.ipynb=.log%"

echo.
echo ------------------------------------------
echo Running: %NBNAME%
echo ------------------------------------------

if not exist "%NB%" (
    echo FAILED: notebook not found "%NB%"
    if /I "%MODE%"=="fail" exit /b 1
    exit /b 0
)

"%PY%" -m jupyter nbconvert ^
    --to notebook ^
    --execute ^
    --ExecutePreprocessor.kernel_name=python3 ^
    "%NB%" ^
    --output "%OUT%" > "%LOG%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo FAILED: %NBNAME%
    echo Check log: "%LOG%"
    if /I "%MODE%"=="fail" exit /b %ERRORLEVEL%
    echo Continue even if error
    exit /b 0
)

echo DONE: %NBNAME%
exit /b 0


:fail
echo.
echo STOPPED because one required step failed.
exit /b 1
