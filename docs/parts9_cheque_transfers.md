# PARTS9: Cheque / Transfer Registers (ทะเบียนเช็ครับ / เช็คจ่าย)

How PARTS9 stores the Excel reports **ทะเบียนเช็ครับ** and **ทะเบียนเช็คจ่าย**, and how they land in Supabase.

**Canonical dictionary:** [kcw-brdet-bpdet-cheque-transfers-data-dictionary.md](https://github.com/pthengtr/kcw-docs/blob/main/dictionaries/kcw-brdet-bpdet-cheque-transfers-data-dictionary.md) (shared for api / v2 / analytics). This file is extract/validation notes.

## Source reports

| Report | File (example) | PARTS9 table |
|--------|----------------|--------------|
| ทะเบียนเช็ครับ | `07_temp/ทะเบียนเช็ครับ.xls` | **`dbo.BRDET`** |
| ทะเบียนเช็คจ่าย | `07_temp/ทะเบียนเช็คจ่าย.xls` | **`dbo.BPDET`** |

Sheet: `Report`. Period on export is in the title row (e.g. ประจำเดือน กรกฎาคม 2569).

Despite the Thai name “เช็ค”, each row is a **bank instrument line**: either a real **cheque**, or a **transfer / other method**. Discriminate with `CHKNO` (below).

## Finding

These registers are **not** `RVMAS` / `PVMAS` alone.

| Direction | Detail table | Typical voucher headers |
|-----------|--------------|-------------------------|
| In (รับ) | `BRDET` | Often `TR*` / sales-linked; not always present as `RVMAS.VOUCNO` |
| Out (จ่าย) | `BPDET` | Often linked to `PVMAS.VOUCNO` (e.g. `KCPN*`) |

Related but different:

| Table | Role |
|-------|------|
| `PVMAS` / `RVMAS` | Payment / receipt voucher **headers** (totals, AP/AR account) |
| `BKTRNS` | Bank statement / reconciliation lines |
| `CHMAS` | Chart / bank **account master** (`TASK='BK'`, accounts like `2101.x`) — not register lines |

### Validation (July 2026 export)

Against Drive Excel vs HQ PARTS9:

- ทะเบียนเช็ครับ ↔ `BRDET` on `VOUCNO` + `CHKAMT`: **exact match** (after dropping repeated header rows in the xls)
- ทะเบียนเช็คจ่าย ↔ `BPDET` on `VOUCNO` + `CHKAMT`: **exact match**

## Cheque vs transfer (`CHKNO`)

`CHKNO` is free text. There is **no reliable separate “is_cheque” flag**; use the value itself:

| `CHKNO` looks like… | Treat as | Examples |
|---------------------|----------|----------|
| Numeric / cheque-style id | **Cheque number** | `10102934`, `8033176` |
| Method / channel label | **Not a cheque** (transfer, shop, cash, …) | `โอน`, `KSHOP`, `จ่ายสด(กรรมการ)` |

`PAYTYPE` exists (`1` / `2`) but does **not** cleanly mean cheque vs transfer — both methods appear under both values. Prefer `CHKNO` for classification.

## Useful columns

BRDET and BPDET share the same shape:

| Column | Role |
|--------|------|
| `VOUCDATE`, `VOUCNO` | Voucher date / number (Excel “วันที่” / “เลขที่ใบสำคัญ…”) |
| `ACCTNO` | Bank / GL account when filled (e.g. `2101.1`) |
| `CARDNAME` | On receive rows often holds the bank account code (e.g. `2101.4`) when `ACCTNO` is blank |
| `PAYTYPE` | Internal pay-type code (do not use alone for cheque vs transfer) |
| **`CHKNO`** | Cheque number **or** method label (`โอน`, `KSHOP`, …) |
| `CHKDATE` | Instrument date (“ลงวันที่”) |
| `CHKAMT` | Amount |
| `BANKNAME` | Bank description (often includes account #) |
| `JOURTYPE` | e.g. `SJ`/`CR` (receive), `CP`/`PJ` (pay) |
| `STATUS` | Clearing-ish marker when set (e.g. `=`) |
| `CANCELED`, `DONE` | Cancel / done flags |

### Excel ↔ DB map

| Excel | DB |
|-------|-----|
| วันที่ | `VOUCDATE` |
| เลขที่ใบสำคัญรับ / จ่าย | `VOUCNO` |
| รหัสบัญชี | `ACCTNO` (or `CARDNAME` on some receive rows) |
| หมายเลขเช็ค | `CHKNO` |
| ลงวันที่ | `CHKDATE` |
| ชื่อธนาคาร | `BANKNAME` |
| จำนวนเงิน | `CHKAMT` |

## Example

Outbound cheque:

- `BPDET.VOUCNO = KCPN6907-011`, `CHKNO = 10102934`, `CHKAMT = 287426.30`
- Header also in `PVMAS` (same `VOUCNO`, `CHKAMT`)

Inbound transfer-style:

- `BRDET.VOUCNO = TR6907-004`, `CHKNO = KSHOP`, `BANKNAME = KBANK … #0648917236`, `CHKAMT = 2400`

## Connection note

HQ PARTS9 on KSS via `mssql_engine("hq")` (`.env` `KSS_*` / `PARTS9_HQ_*`). Database: `PARTS9` on server `KSS`. HQ only — not in SYP minimal extract.

## Pipeline / worker sync

Drive CSVs:

- `raw_hq_brdet_cheques_received.csv`
- `raw_hq_bpdet_cheques_paid.csv`

Supabase (staging replace):

- `raw_kcw.raw_hq_brdet_cheques_received`
- `raw_kcw.raw_hq_bpdet_cheques_paid`

### Daily HQ A pipeline

Included in full HQ extract (`TABLE_SPECS`) and `upload-daily-raw`:

```bash
python -m src.kcw.pipeline extract --site hq
python -m src.kcw.pipeline upload-daily-raw
```

BAT: [`worker_tasks/run_hq_parts9_to_drive_raw.bat`](../worker_tasks/run_hq_parts9_to_drive_raw.bat)

### Focused sync + daily bank sync

```bash
python -m src.kcw.pipeline sync-brdet-bpdet
python -m src.kcw.pipeline upload-brdet-bpdet
```

Worker BATs:

- Focused only: [`worker_tasks/run_hq_brdet_bpdet_sync.bat`](../worker_tasks/run_hq_brdet_bpdet_sync.bat)
- Daily bank sync (BRDET/BPDET + statement Excel import): [`worker_tasks/run_bank_statement_import.bat`](../worker_tasks/run_bank_statement_import.bat)

Apply migration first:

- [`supabase/migrations/20260803093000_create_raw_hq_brdet_bpdet.sql`](../supabase/migrations/20260803093000_create_raw_hq_brdet_bpdet.sql)
