@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Bank statement monthly Excel report (one sheet per account).
REM CLI: python -m src.kcw.pipeline bank-statement-report [--year Y] [--month M]
REM Same entry point for Task Scheduler, webapp, and chatbot enqueue.
REM Needs SUPABASE_DB_* / SUPABASE_DB_URL and Drive 01_raw (PIMAS/PVMAS) in .env / paths.

cd /d "%~dp0.."

REM Only read KCW_ANALYTICS_PYTHON / LOG_DIR from .env.
REM Do not load all secrets here because special characters can break batch parsing.
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

set "LOG=%KCW_ANALYTICS_LOG_DIR%\bank_statement_report.log"
set "YEAR_ARG="
set "MONTH_ARG="

if not "%~1"=="" set "YEAR_ARG=--year %~1"
if not "%~2"=="" set "MONTH_ARG=--month %~2"

echo ==========================================
echo Running bank statement report
echo Python: %KCW_ANALYTICS_PYTHON%
echo Repo: %cd%
echo Args: %YEAR_ARG% %MONTH_ARG%
echo Log: %LOG%
echo ==========================================

"%KCW_ANALYTICS_PYTHON%" -m src.kcw.pipeline bank-statement-report %YEAR_ARG% %MONTH_ARG% > "%LOG%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo FAILED: bank-statement-report
    echo Check log: "%LOG%"
    exit /b %ERRORLEVEL%
)

echo DONE: bank-statement-report
exit /b 0
