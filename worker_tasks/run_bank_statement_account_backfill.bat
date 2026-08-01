@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM One-time backfill: re-read statement Excel on Drive -> full account_no in Supabase.
REM Dry-run by default. Pass --apply to write (after reviewing log).
REM Needs KCW_DRIVE_ROOT (or paths.yaml) + Supabase DB creds in .env.

cd /d "%~dp0.."

for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if /I "%%A"=="KCW_ANALYTICS_PYTHON" set "KCW_ANALYTICS_PYTHON=%%B"
    if /I "%%A"=="KCW_ANALYTICS_LOG_DIR" set "KCW_ANALYTICS_LOG_DIR=%%B"
)

if "%KCW_ANALYTICS_PYTHON%"=="" (
    echo Missing KCW_ANALYTICS_PYTHON in .env
    exit /b 1
)

if "%KCW_ANALYTICS_LOG_DIR%"=="" (
    set "KCW_ANALYTICS_LOG_DIR=%cd%\logs"
)

if not exist "%KCW_ANALYTICS_LOG_DIR%" (
    mkdir "%KCW_ANALYTICS_LOG_DIR%"
)

set "LOG=%KCW_ANALYTICS_LOG_DIR%\bank_statement_account_backfill.log"

echo ==========================================
echo Bank statement account backfill
echo Python: %KCW_ANALYTICS_PYTHON%
echo Repo: %cd%
echo Log: %LOG%
echo ==========================================

"%KCW_ANALYTICS_PYTHON%" -m src.kcw.pipeline backfill-statement-accounts %* > "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

if %RC% NEQ 0 (
    echo FAILED — see "%LOG%"
    exit /b %RC%
)

echo DONE — see "%LOG%"
exit /b 0
