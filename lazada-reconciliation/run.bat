@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

echo =====================================================
echo  กระทบยอด Lazada (ไม่แก้ไขไฟล์ต้นฉบับ, ไม่อัปโหลดออกเน็ต)
echo =====================================================
echo.

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    set "PYTHON=py -3"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 (
        set "PYTHON=python"
    ) else (
        echo ไม่พบ Python กรุณาติดตั้ง Python 3.11 ขึ้นไป แล้วลองใหม่
        echo ดูวิธีติดตั้งใน README.md
        pause
        exit /b 1
    )
)

if not exist "input\orders" (
    echo ไม่พบโฟลเดอร์ input\orders
    pause
    exit /b 1
)
if not exist "input\finance" (
    echo ไม่พบโฟลเดอร์ input\finance
    pause
    exit /b 1
)
if not exist "input\wallet" (
    echo ไม่พบโฟลเดอร์ input\wallet
    pause
    exit /b 1
)

dir /b "input\orders\*.xlsx" "input\orders\*.xls" >nul 2>nul
if errorlevel 1 (
    echo ไม่พบไฟล์คำสั่งซื้อใน input\orders
    echo วางไฟล์ Order list export หรือ คำสั่งซื้อทั้งหมด แล้วดับเบิลคลิกใหม่
    pause
    exit /b 1
)
dir /b "input\finance\*.xlsx" "input\finance\*.xls" >nul 2>nul
if errorlevel 1 (
    echo ไม่พบไฟล์รายการทางบัญชีใน input\finance
    echo วางไฟล์ finance export หรือ รายการทางบัญชี แล้วดับเบิลคลิกใหม่
    pause
    exit /b 1
)
dir /b "input\wallet\*.xlsx" "input\wallet\*.xls" >nul 2>nul
if errorlevel 1 (
    echo ไม่พบไฟล์ยอดของฉันใน input\wallet
    echo วางไฟล์ Wallet / Balance Transactions แล้วดับเบิลคลิกใหม่
    pause
    exit /b 1
)

echo กำลังติดตั้งไลบรารีที่จำเป็นถ้ายังไม่มี...
%PYTHON% -m pip install -r requirements.txt -q
if errorlevel 1 (
    echo ติดตั้งไลบรารีไม่สำเร็จ กรุณาตรวจการเชื่อมต่อหรือสิทธิ์ติดตั้ง Python
    pause
    exit /b 1
)

if not exist "output" mkdir "output"

echo กำลังประมวลผล...
%PYTHON% -m src.main --orders "input/orders" --finance "input/finance" --wallet "input/wallet" --output "output"
set "EXITCODE=%ERRORLEVEL%"

if %EXITCODE%==0 (
    echo.
    echo ประมวลผลเสร็จ เปิดไฟล์ในโฟลเดอร์ output
) else if %EXITCODE%==2 (
    echo.
    echo ประมวลผลเสร็จแต่รายงานมี WARNING หรือ FAIL
    echo ห้ามถือว่ากระทบยอดสำเร็จ กรุณาเปิดชีต Validations และ Exceptions
) else (
    echo.
    echo ประมวลผลไม่สำเร็จ
    echo สาเหตุที่พบบ่อย: ไม่พบไฟล์, พบ header ไม่ครบ, หรืออ่านไฟล์ไม่ได้
    echo ดูข้อความด้านบนและไฟล์ .log ในโฟลเดอร์ output
)

echo.
pause
exit /b %EXITCODE%
