@echo off
setlocal EnableExtensions EnableDelayedExpansion

set PY=C:\Users\Admin\anaconda3\python.exe
set NBDIR=C:\Users\Admin\Notebook\kcw-analytics\notebooks
set LOGDIR=C:\Users\Admin\Notebook\logs

if not exist "%LOGDIR%" (
    mkdir "%LOGDIR%"
)

echo ==========================================
echo Running notebooks sequentially...
echo Python: %PY%
echo Notebooks: %NBDIR%
echo Logs folder: %LOGDIR%
echo ==========================================

call :run_nb "50_parts9_to_supabase.ipynb"
echo Continue even if error

echo.
echo ✅ ALL DONE.
exit /b 0


:run_nb
set NBNAME=%~1
set NB=%NBDIR%\%NBNAME%
set OUT=%LOGDIR%\%NBNAME:.ipynb=.executed.ipynb%
set LOG=%LOGDIR%\%NBNAME:.ipynb=.log%

echo.
echo ------------------------------------------
echo Running: %NBNAME%
echo ------------------------------------------

"%PY%" -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 "%NB%" --output "%OUT%" > "%LOG%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo ❌ FAILED: %NBNAME%
    echo Check log: "%LOG%"
    exit /b %ERRORLEVEL%
)

echo ✅ DONE: %NBNAME%
exit /b 0


:fail
echo.
echo ❌ STOPPED because one notebook failed.
exit /b 1