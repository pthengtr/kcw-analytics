@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0.."

REM Only read KCW_ANALYTICS_PYTHON from .env.
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
set "FAILED=0"

echo ==========================================
echo Running online sales sync
echo Python: %KCW_ANALYTICS_PYTHON%
echo Repo: %cd%
echo Notebooks: %NBDIR%
echo Logs: %KCW_ANALYTICS_LOG_DIR%
echo ==========================================

call :run_nb shopee 71_online_shopee.ipynb
call :run_nb lazada 72_online_lazada.ipynb
call :run_nb tiktok 73_online_tiktok.ipynb

echo.
if "%FAILED%"=="0" (
    echo ONLINE_SYNC_RESULT: ALL_OK
) else (
    echo ONLINE_SYNC_RESULT: DONE_WITH_FAILURES
)

echo shopee: !STATUS_shopee!
echo lazada: !STATUS_lazada!
echo tiktok: !STATUS_tiktok!

exit /b 0


:run_nb
set "PLATFORM=%~1"
set "NBNAME=%~2"
set "NB=%NBDIR%\%NBNAME%"
set "OUT=%KCW_ANALYTICS_LOG_DIR%\%NBNAME:.ipynb=.executed.ipynb%"
set "LOG=%KCW_ANALYTICS_LOG_DIR%\%NBNAME:.ipynb=.log%"

echo.
echo ------------------------------------------
echo Running %PLATFORM%: %NBNAME%
echo ------------------------------------------

if not exist "%NB%" (
    echo FAILED: %PLATFORM% notebook not found
    echo Notebook not found: "%NB%" > "%LOG%"
    set "STATUS_%PLATFORM%=FAILED - notebook not found"
    set "FAILED=1"
    exit /b 0
)

"%KCW_ANALYTICS_PYTHON%" -m jupyter nbconvert ^
    --to notebook ^
    --execute ^
    --ExecutePreprocessor.kernel_name=python3 ^
    "%NB%" ^
    --output "%OUT%" > "%LOG%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo FAILED: %PLATFORM%
    echo Check log: "%LOG%"
    set "STATUS_%PLATFORM%=FAILED - check log"
    set "FAILED=1"
    exit /b 0
)

echo DONE: %PLATFORM%
set "STATUS_%PLATFORM%=OK"
exit /b 0