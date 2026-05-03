@echo off
setlocal EnableExtensions DisableDelayedExpansion

cd /d "%~dp0.."

REM Only read KCW_ANALYTICS_PYTHON from .env.
REM Do not load secrets/DB URLs in batch because special characters can break.
for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if /I "%%A"=="KCW_ANALYTICS_PYTHON" set "KCW_ANALYTICS_PYTHON=%%B"
)

if "%KCW_ANALYTICS_PYTHON%"=="" (
    echo Missing KCW_ANALYTICS_PYTHON in .env
    exit /b 1
)

echo ==========================================
echo Running product image sync
echo Python: %KCW_ANALYTICS_PYTHON%
echo Repo: %cd%
echo ==========================================

"%KCW_ANALYTICS_PYTHON%" worker_tasks\sync_product_images.py

if %ERRORLEVEL% NEQ 0 (
    echo FAILED: product image sync
    exit /b %ERRORLEVEL%
)

echo DONE: product image sync
exit /b 0