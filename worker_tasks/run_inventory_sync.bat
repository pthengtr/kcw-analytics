@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0.."

if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        set "key=%%A"
        set "value=%%B"
        if not "!key!"=="" if not "!key:~0,1!"=="#" (
            set "!key!=!value!"
        )
    )
)

if "%KCW_ANALYTICS_PYTHON%"=="" (
    echo Missing KCW_ANALYTICS_PYTHON in .env
    exit /b 1
)

if "%KCW_ANALYTICS_ROOT%"=="" (
    set "KCW_ANALYTICS_ROOT=%cd%"
)

if "%KCW_ANALYTICS_LOG_DIR%"=="" (
    set "KCW_ANALYTICS_LOG_DIR=%KCW_ANALYTICS_ROOT%\logs"
)

if not exist "%KCW_ANALYTICS_LOG_DIR%" (
    mkdir "%KCW_ANALYTICS_LOG_DIR%"
)

set "NBDIR=%KCW_ANALYTICS_ROOT%\notebooks"
set "NBNAME=50_parts9_to_supabase.ipynb"
set "NB=%NBDIR%\%NBNAME%"
set "OUT=%KCW_ANALYTICS_LOG_DIR%\%NBNAME:.ipynb=.executed.ipynb%"
set "LOG=%KCW_ANALYTICS_LOG_DIR%\%NBNAME:.ipynb=.log%"

echo ==========================================
echo Running inventory sync
echo Python: %KCW_ANALYTICS_PYTHON%
echo Notebook: %NB%
echo Logs: %KCW_ANALYTICS_LOG_DIR%
echo Branch: %KCW_BRANCH%
echo ==========================================

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
exit /b 0