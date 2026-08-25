# โปรแกรมกระทบยอด Lazada

เครื่องมือออฟไลน์สำหรับนักบัญชี ใช้กระทบยอดรายได้และเงินรับจากไฟล์ Excel ที่ดาวน์โหลดจาก Lazada Seller Center สามประเภท:

1. คำสั่งซื้อทั้งหมด (Order list)
2. รายการทางบัญชี (Income Order Overview หรือรายการธุรกรรมที่มี Freename)
3. ยอดของฉัน (Wallet / Balance Transactions)

โปรแกรม**ไม่แก้ไขไฟล์ต้นฉบับ** ไม่ใช้ฐานข้อมูล และ**ไม่อัปโหลดหรือส่งข้อมูลออกอินเทอร์เน็ต**

ผลลัพธ์ไม่ใช่คำรับรองทางบัญชี นักบัญชีต้องตรวจรายงานอีกครั้งก่อนลงบัญชี

## 1. วิธีติดตั้ง Python

ต้องการ Python 3.11 ขึ้นไป

- Windows: ดาวน์โหลดจาก [python.org](https://www.python.org/downloads/) ตอนติดตั้งให้ติ๊ก **Add python.exe to PATH**
- ตรวจเวอร์ชัน:

```bat
python --version
```

หรือ

```bat
py -3 --version
```

## 2. วิธีติดตั้ง dependencies

เปิด Command Prompt ที่โฟลเดอร์ `lazada-reconciliation` แล้วรัน:

```bat
python -m pip install -r requirements.txt
```

ไลบรารีหลัก: pandas, openpyxl, xlrd, pytest

`run.bat` จะพยายามติดตั้งให้อัตโนมัติเมื่อดับเบิลคลิก

## 3. วิธีวางไฟล์ทั้ง 3 ประเภท

วางไฟล์ `.xlsx` หรือ `.xls` ตามโฟลเดอร์นี้ **ห้ามแก้ไฟล์ต้นทาง**

```
lazada-reconciliation/
├─ input/
│  ├─ orders/     ← คำสั่งซื้อทั้งหมด / Order list export
│  ├─ finance/    ← รายการทางบัญชี / finance export / Income Overview
│  └─ wallet/     ← ยอดของฉัน / Balance Transactions
```

โปรแกรมตรวจประเภทไฟล์จาก**ชื่อคอลัมน์** ไม่พึ่งชื่อไฟล์ ถ้าในโฟลเดอร์มีหลายไฟล์จะรวมให้ และติดชื่อไฟล์ต้นทางทุกแถว

ตัวอย่างชื่อไฟล์ที่รองรับ เช่น `7LAZ1 คำสั่งซื้อทั้งหมด – aug 2026.xlsx`

## 4. วิธีรันผ่าน run.bat และ command line

### ดับเบิลคลิก `run.bat` (Windows)

โปรแกรมจะแจ้งเป็นภาษาไทยเมื่อ:

- ไม่พบไฟล์
- พบหลายไฟล์ในโฟลเดอร์เดียวกัน (จะรวมไฟล์)
- header ไม่ครบ
- อ่านไฟล์ไม่ได้
- ประมวลผลเสร็จ
- รายงานมี WARNING หรือ FAIL

### Command line

จากโฟลเดอร์ `lazada-reconciliation`:

```bat
python -m src.main --orders "input/orders" --finance "input/finance" --wallet "input/wallet" --output "output" --month "2026-08" --tolerance "0.01"
```

ผลลัพธ์อยู่ที่ `output/lazada_reconciliation_YYYY-MM.xlsx` และไฟล์ `.log` ในโฟลเดอร์เดียวกัน (ไม่มีชื่อ ที่อยู่ หรือเบอร์โทรลูกค้า)

## 5. วิธีอ่านแต่ละชีต

| ชีต | ใช้ทำอะไร |
| --- | --- |
| Executive_Summary | ภาพรวมเดือน, ยอดขาย, เงินเข้า Wallet, ค่าธรรมเนียม, ยอดโอนธนาคาร, สถานะ PASS/WARNING/FAIL |
| Order_Summary | ยอดรวมต่อ `orderNumber`, สถานะ, จำนวนแถวต้นทาง |
| Finance_Summary | ยอดต่อออเดอร์แยกตามประเภทรายการ |
| Order_vs_Finance | full outer join, ผลต่าง, `match_status`, สาเหตุที่เป็นไปได้ |
| Fee_Details | รายการค่าขนส่ง/ค่าบริการ/ค่าธรรมเนียม พร้อม Freename เดิมและ `source_row` |
| Wallet_Summary | Type, Sub Type, Amount ตามเครื่องหมายต้นทาง, Auto Withdrawal, ยอดคงเหลือ |
| Exceptions | รายการผิดปกติและความรุนแรง |
| Unknown_Mappings | ค่าที่ระบบยังไม่มี mapping ให้เพิ่มใน config |
| Validations | ผลการตรวจ 12 ข้อ |
| Methodology | สูตร, tolerance, mapping, สมมติฐาน, เวอร์ชันโปรแกรม |

คอลัมน์ `source_file` / `source_row` ชี้กลับไปยังแถวในไฟล์ต้นทาง แถว `GENERATED TOTAL` เป็นยอดที่สร้างในรายงาน ห้ามนำไปคำนวณซ้ำ

## 6. วิธีเพิ่ม mapping

แก้ไฟล์ [`src/config.py`](src/config.py)

- สถานะออเดอร์: `STATUS_GROUPS`
- ชื่อรายการ Finance (Freename / ชื่อรายการธุรกรรม): `FREENAME_EXACT` และ `FREENAME_CONTAINS`
- Wallet Type + Sub Type: `WALLET_EXACT`

ค่าที่กำหนดได้สำหรับ Finance: `order_income`, `shipping_fee`, `service_fee`, `other_fee`, `refund`, `adjustment`, `unknown`

ค่าที่กำหนดได้สำหรับ Wallet: `settlement_inflow`, `bank_withdrawal`, `opening_balance`, `closing_balance`, `adjustment`, `unknown`

หลังจากเพิ่ม mapping แล้วรันโปรแกรมใหม่ ค่าที่ไม่รู้จักจะไม่ถูกลบทิ้ง แต่จะไปอยู่ชีต Unknown_Mappings

## 7. ความหมายของ MATCHED และ exception

- **MATCHED** — มีทั้งไฟล์คำสั่งซื้อและไฟล์บัญชี และผลต่างอยู่ใน tolerance (ค่าเริ่มต้น 0.01 บาท)
- **NET_SETTLED** — มีทั้งสองไฟล์ แต่ไฟล์บัญชีเป็นยอดสุทธิ (Income Order Overview) จึงไม่เท่า `paidPrice` ผลต่างคือค่าธรรมเนียมโดยประมาณ
- **ORDER_NOT_RELEASED** — มีในคำสั่งซื้อแต่ยังไม่มีเงินเข้า Wallet (ลูกค้าอาจยังไม่ยืนยันรับสินค้า)
- **FINANCE_FROM_OTHER_PERIOD** — มีในไฟล์บัญชีแต่ไม่มีในคำสั่งซื้อเดือนนี้ (อาจเป็นออเดอร์เดือนก่อน)
- **AMOUNT_MISMATCH** — มีทั้งสองไฟล์แต่ยอดไม่ตรง (อาจมีค่าปรับหรือรายการปรับปรุง)
- **CANCELLED_OR_REFUNDED** — เกี่ยวข้องกับยกเลิกหรือคืนเงิน
- **UNKNOWN** — วิเคราะห์ไม่ได้
- **EMPTY_ORDER_NUMBER** — เลขออเดอร์ว่าง
- **AMOUNT_PARSE_FAILED** — แปลงจำนวนเงินไม่สำเร็จ
- **DUPLICATE_ROW** — แถวซ้ำทุกคอลัมน์ (ไม่ถูกลบ)
- **GRAND_TOTAL_EXCLUDED** — ตัดแถว Grand Total ออกแล้ว

## 8. วิธีตรวจสอบผลก่อนนำไปลงบัญชี

1. ดูสถานะรวมบน Executive_Summary ถ้าเป็น FAIL หรือ WARNING ห้ามถือว่ากระทบยอดสำเร็จ
2. เปิด Validations ทีละข้อ โดยเฉพาะยอดก่อน/หลัง group และยอดสุทธิ Finance
3. ไล่ Order_vs_Finance ตาม `match_status` แล้วเปิดไฟล์ต้นทางที่ `source_row`
4. ตรวจ Unknown_Mappings แล้วเพิ่ม mapping ถ้าจำเป็น
5. เทียบยอดโอนเข้าธนาคารกับรายการ Auto Withdrawal และ Statement ของธนาคาร
6. ถ้าไฟล์ Wallet ไม่มียอดยกมา โปรแกรมจะแสดง **ข้อมูลไม่เพียงพอ** ห้ามใส่ 0 เองโดยไม่ตรวจ
7. ให้ผู้มีอำนาจลงบัญชีลงนามตรวจรายงานก่อนบันทึกบัญชี

## 9. ข้อจำกัดและข้อควรระวัง

- รองรับไฟล์ Seller Center ที่เป็น Excel เท่านั้น
- ไฟล์ “รายการทางบัญชี” แบบ Income Order Overview เป็น**ยอดสุทธิ** ไม่แยกค่าขนส่ง/ค่าบริการ ถ้าต้องการแยก fee ให้ดาวน์โหลดไฟล์ธุรกรรมที่มีคอลัมน์ `ชื่อรายการธุรกรรม` หรือ Freename
- ยอด Wallet กับ Finance อาจคนละงวด (วันที่สร้างออเดอร์ vs วันที่ Settlement)
- เลขออเดอร์ 16 หลักถ้า Excel แปลงเป็นตัวเลขอาจเพี้ยน โปรแกรมอ่านเป็นข้อความเมื่อทำได้
- ไม่เก็บชื่อ ที่อยู่ เบอร์โทร อีเมล ลงรายงานหรือ log
- ห้ามแก้ไขไฟล์ใน `input/`
- แถวซ้ำจะถูกรายงานแต่ไม่ลบให้โดยอัตโนมัติ

## 10. คำรับรอง

โปรแกรมนี้เป็นเครื่องมือช่วยรวบรวมและกระทบยอดข้อมูล **ไม่ใช่คำรับรองทางบัญชี** และไม่ใช่ความเห็นของผู้สอบบัญชี นักบัญชีของกิจการต้องตรวจผล ความครบถ้วน และความเหมาะสมของรายการก่อนลงบัญชีทุกครั้ง
