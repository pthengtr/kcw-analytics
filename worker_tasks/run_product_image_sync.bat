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

echo ==========================================
echo Running product image sync
echo Python: %KCW_ANALYTICS_PYTHON%
echo Repo: %cd%
echo Legacy dir: %LEGACY_PRODUCT_IMAGE_DIR%
echo ==========================================

"%KCW_ANALYTICS_PYTHON%" worker_tasks\sync_product_images.py
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: product image sync
    exit /b %ERRORLEVEL%
)

echo DONE: product image sync
exit /b 0