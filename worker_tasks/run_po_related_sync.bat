@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM DEPRECATED entry point — HQ and SYP must run on separate machines.
REM Prefer:
REM   worker_tasks\run_hq_po_related_sync.bat
REM   worker_tasks\run_syp_po_related_sync.bat
REM
REM This wrapper still accepts --site hq|syp for Task Scheduler compatibility.

cd /d "%~dp0.."

set "SITE="
set "SKIP_INVENTORY="

:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="--site" (
    if /I "%~2"=="hq" set "SITE=hq"
    if /I "%~2"=="syp" set "SITE=syp"
    if "!SITE!"=="" (
        echo ERROR: --site must be hq or syp
        echo Prefer: run_hq_po_related_sync.bat  or  run_syp_po_related_sync.bat
        exit /b 1
    )
    shift
    shift
    goto :parse_args
)
if /I "%~1"=="--skip-inventory" (
    set "SKIP_INVENTORY=--skip-inventory"
    shift
    goto :parse_args
)
echo Unknown argument: %~1
echo Usage: %~nx0 --site hq^|syp [--skip-inventory]
echo Prefer: run_hq_po_related_sync.bat  or  run_syp_po_related_sync.bat
exit /b 1

:args_done

if "!SITE!"=="" (
    echo ERROR: --site is required ^(hq and syp must run separately^).
    echo.
    echo   worker_tasks\run_hq_po_related_sync.bat
    echo   worker_tasks\run_syp_po_related_sync.bat
    echo.
    echo Or: %~nx0 --site hq^|syp [--skip-inventory]
    exit /b 1
)

if /I "!SITE!"=="hq" (
    call "%~dp0run_hq_po_related_sync.bat" !SKIP_INVENTORY!
    exit /b %ERRORLEVEL%
)

call "%~dp0run_syp_po_related_sync.bat" !SKIP_INVENTORY!
exit /b %ERRORLEVEL%
