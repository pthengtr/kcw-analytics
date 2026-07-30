@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Bank statement import: Drive Excel (KBANK + KTB) -> Supabase bank.statement_*
REM Notebook: notebooks/02_bank_statement_import_test.ipynb
REM Needs SUPABASE_DB_* in .env (loaded by the notebook via python-dotenv).

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

set "NBDIR=%cd%\notebooks"
set "NBNAME=02_bank_statement_import_test.ipynb"
set "NB=%NBDIR%\%NBNAME%"
set "OUT=%KCW_ANALYTICS_LOG_DIR%\%NBNAME:.ipynb=.executed.ipynb%"
set "LOG=%KCW_ANALYTICS_LOG_DIR%\%NBNAME:.ipynb=.log%"

echo ==========================================
echo Running bank statement import
echo Python: %KCW_ANALYTICS_PYTHON%
echo Repo: %cd%
echo Notebook: %NB%
echo Logs: %KCW_ANALYTICS_LOG_DIR%
echo ==========================================

if not exist "%NB%" (
    echo FAILED: notebook not found
    echo Notebook not found: "%NB%" > "%LOG%"
    exit /b 1
)

"%KCW_ANALYTICS_PYTHON%" -m jupyter nbconvert ^
    --to notebook ^
    --execute ^
    --ExecutePreprocessor.kernel_name=python3 ^
    "%NB%" ^
    --output "%OUT%" > "%LOG%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo FAILED: %NBNAME%
    echo Check log: "%LOG%"
    exit /b %ERRORLEVEL%
)

echo DONE: %NBNAME%

REM Regenerate monthly Excel so next-day operator uploads appear in the report.
echo Running bank statement report after import...
call "%~dp0run_bank_statement_report.bat"
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: bank statement report after import
    exit /b %ERRORLEVEL%
)

exit /b 0
