@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Daily bank sync:
REM   1) PARTS9 BRDET/BPDET cheque+transfer registers -> Supabase raw_kcw
REM   2) Drive bank statement Excel (KBANK + KTB) -> Edge Function import-bank-statement
REM
REM Focused cheque sync: worker_tasks/run_hq_brdet_bpdet_sync.bat
REM Statement uploader: scripts/upload_drive_bank_statements.py
REM Parser SoT: kcw-v2 supabase/functions/import-bank-statement (auto_v2)
REM Needs SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY in .env (loaded by the Python script).

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

echo ==========================================
echo Daily bank sync
echo   1) HQ BRDET/BPDET cheque/transfer registers
echo   2) Bank statement Drive -^> Edge Function upload
echo Python: %KCW_ANALYTICS_PYTHON%
echo Repo: %cd%
echo ==========================================

echo.
echo --- 1/2: HQ BRDET/BPDET sync ---
call "%~dp0run_hq_brdet_bpdet_sync.bat"
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: BRDET/BPDET sync
    exit /b %ERRORLEVEL%
)

set "UPLOAD_SCRIPT=%cd%\scripts\upload_drive_bank_statements.py"
set "LOG=%KCW_ANALYTICS_LOG_DIR%\upload_drive_bank_statements.log"

echo.
echo --- 2/2: bank statement Edge upload ---
echo Script: %UPLOAD_SCRIPT%
echo Logs: %LOG%

if not exist "%UPLOAD_SCRIPT%" (
    echo FAILED: uploader script not found
    echo Uploader not found: "%UPLOAD_SCRIPT%" > "%LOG%"
    exit /b 1
)

"%KCW_ANALYTICS_PYTHON%" "%UPLOAD_SCRIPT%" > "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: upload_drive_bank_statements.py
    echo Check log: "%LOG%"
    exit /b %ERRORLEVEL%
)

echo DONE: Drive statements uploaded via Edge Function

REM Regenerate monthly Excel so next-day operator uploads appear in the report.
echo Running bank statement report after import...
call "%~dp0run_bank_statement_report.bat"
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: bank statement report after import
    exit /b %ERRORLEVEL%
)

echo DONE: daily bank sync
exit /b 0
