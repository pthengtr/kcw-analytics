# PARTS9: Pending Receive (ค้างรับ)

How the legacy KSS / PARTS9 system flags open-PO items that are still pending receipt.

## Source report

- File: `G:\Shared drives\KCW-Data\kcw_analytics\07_temp\รายการสินค้าค้างรับ.xls`
- Sheet: `Report`
- Title: รายการสินค้าค้างรับ ณ วันที่ (as-of date on export)

## Finding

Pending receive is **not** flagged on `POMAS` / `PODET`.

It is tracked in **`dbo.ICLOW`**:

| Column | Pending value | Meaning |
|--------|---------------|---------|
| `ORDERED` | `Y` | Line was ordered (linked to a PO) |
| `RECEIVED` | `N` | Not yet received |
| `CANCELED` | `N` | Not canceled |

When goods are received, the app sets `RECEIVED = 'Y'` and fills `RCVDDATE` / `RCVDNO`.

### Equivalent query

```sql
SELECT *
FROM dbo.ICLOW
WHERE ORDERED = 'Y'
  AND ISNULL(RECEIVED, 'N') = 'N'
  AND ISNULL(CANCELED, 'N') = 'N';
```

### Validation

Against the Excel export (as of 2026-08-01):

- ICLOW pending rows: **616**
- Exact match to Excel on `DOCDATE` + `BCODE` + `QTY`: **616 / 616**
- Excel-only / ICLOW-only: **0**

## Useful `ICLOW` fields

| Column | Role |
|--------|------|
| `DOCDATE`, `DOCNO` | PO date / PO number (e.g. `PO6905-392`) |
| `VENDOR` | Supplier code (`ACCTNO`) |
| `BCODE`, `PCODE`, `MCODE` | Product codes |
| `DESCR` | Description |
| `QTY`, `UI`, `PRICE` | Ordered qty / unit / price |
| `ORDERED` | `Y` = ordered |
| `RECEIVED` | `N` pending, `Y` received |
| `RCVDDATE`, `RCVDNO` | Receipt date / doc when received |

## What it is *not*

These look related but do **not** define ค้างรับ:

| Field | Why not |
|-------|---------|
| `PODET.DONE` / `POMAS.DONE` | Always `N` in this database |
| `PODET.STATUS` | Product status (aligns with `ICMAS.STATUS`), not receive status |
| `POMAS.BILLED` | Header-level “purchase bill linked”; partial invoices still leave other lines pending in `ICLOW` |

Purchase invoices (`PIMAS` / `PIDET`, linked via `PIMAS.PO` → `POMAS.DOCNO`) show *what was billed*, but the report itself is driven by **`ICLOW.ORDERED` / `RECEIVED`**.

## Example

PO `PO6905-392` (supplier `BK`): 13 PO lines, invoice `254929` covers 7 lines. The 4 lines still on the ค้างรับ report are the ones still `ORDERED='Y'` and `RECEIVED='N'` in `ICLOW`.

## Connection note

HQ PARTS9 on KSS is reachable via `mssql_engine("hq")` using `.env` `KSS_*` (or `PARTS9_HQ_*`) credentials. Database: `PARTS9` on server `KSS`.

## Pipeline / worker sync

Extract full `ICLOW` to Drive and replace Supabase `raw_kcw.raw_{hq|syp}_iclow_stock_orders`:

```bash
python -m src.kcw.pipeline sync-iclow --site hq
python -m src.kcw.pipeline sync-iclow --site syp
```

Worker BATs:
- HQ ICLOW only: [`worker_tasks/run_hq_iclow_sync.bat`](../worker_tasks/run_hq_iclow_sync.bat)
- SYP ICLOW only: [`worker_tasks/run_syp_iclow_sync.bat`](../worker_tasks/run_syp_iclow_sync.bat)
- HQ PO+ICLOW + inventory: [`worker_tasks/run_hq_po_related_sync.bat`](../worker_tasks/run_hq_po_related_sync.bat)
- SYP PO+ICLOW + inventory: [`worker_tasks/run_syp_po_related_sync.bat`](../worker_tasks/run_syp_po_related_sync.bat)

HQ and SYP must run separately (different PARTS9 servers).

Also included in daily HQ/SYP extracts and `upload-daily-raw` (HQ A uploads both sites from Drive).

Apply migrations first:
- [`supabase/migrations/20260801083000_create_raw_hq_iclow.sql`](../supabase/migrations/20260801083000_create_raw_hq_iclow.sql)
- [`supabase/migrations/20260801084500_create_raw_syp_iclow.sql`](../supabase/migrations/20260801084500_create_raw_syp_iclow.sql)
