# kcw-analytics

PARTS9 extracts, Drive curated layers, TAR/billgen, and accounting reports for KCW HQ + SYP.

## Architecture

```
SYP PC (Task Scheduler)          HQ PC (Task Scheduler)
  raw extract BAT ────────┐        HQ A: raw-only BAT ──┐
                          ▼                             ▼
                    Google Drive 01_raw/          (same Drive)
                          │
                          ▼
              HQ B full pipeline BAT  (today)
              or Claude Cowork        (later cutover)
                    curated → TAR catch-up → reports → Supabase
```

Local SQL Server is the only thing that must stay on the shop PCs. Everything after raw CSVs land on Drive can move to Claude Cowork later; until then use Windows Task Scheduler + the BATs below.

## Windows Task Scheduler BATs

| Machine | Script | Purpose |
|---------|--------|---------|
| **SYP** | [`worker_tasks/run_syp_parts9_to_drive_raw.bat`](worker_tasks/run_syp_parts9_to_drive_raw.bat) | `PARTS9` → `raw_syp_*.csv` (POMAS/PODET + sales + ICMAS) |
| **SYP** | [`worker_tasks/run_syp_pomas_podet_sync.bat`](worker_tasks/run_syp_pomas_podet_sync.bat) | POMAS/PODET only → Drive + Supabase `raw_syp_*` |
| **SYP** | [`worker_tasks/run_syp_iclow_sync.bat`](worker_tasks/run_syp_iclow_sync.bat) | ICLOW only → Drive + Supabase `raw_syp_iclow_stock_orders` |
| **HQ A** | [`worker_tasks/run_hq_parts9_to_drive_raw.bat`](worker_tasks/run_hq_parts9_to_drive_raw.bat) | `PARTS9` → `raw_hq_*.csv`, then daily raw → Supabase `raw_kcw` |
| **HQ** | [`worker_tasks/run_hq_pomas_podet_sync.bat`](worker_tasks/run_hq_pomas_podet_sync.bat) | POMAS/PODET only → Drive + Supabase `raw_hq_*` |
| **HQ** | [`worker_tasks/run_hq_iclow_sync.bat`](worker_tasks/run_hq_iclow_sync.bat) | ICLOW only → Drive + Supabase `raw_hq_iclow_stock_orders` |
| **HQ** | [`worker_tasks/run_hq_po_related_sync.bat`](worker_tasks/run_hq_po_related_sync.bat) | POMAS/PODET + ICLOW + SIDET/SIMAS (HQ, 6 months) → Drive + Supabase, then inventory |
| **HQ** | [`worker_tasks/run_hq_simas_sidet_sync.bat`](worker_tasks/run_hq_simas_sidet_sync.bat) | SIDET/SIMAS only (HQ) → Drive + Supabase `raw_kcw` (latest 6 months) |
| **HQ** | [`worker_tasks/run_hq_brdet_bpdet_sync.bat`](worker_tasks/run_hq_brdet_bpdet_sync.bat) | BRDET/BPDET only (HQ) → Drive + Supabase cheque/transfer registers |
| **SYP** | [`worker_tasks/run_syp_po_related_sync.bat`](worker_tasks/run_syp_po_related_sync.bat) | POMAS/PODET + ICLOW (SYP) → Drive + Supabase, then inventory on-hand qty |
| **HQ B** | [`worker_tasks/run_hq_parts9_full_pipeline.bat`](worker_tasks/run_hq_parts9_full_pipeline.bat) | HQ A + archive + curated + VAT/TAR + Excel + upload |
| **HQ** | [`worker_tasks/run_bank_statement_import.bat`](worker_tasks/run_bank_statement_import.bat) | Daily bank sync: BRDET/BPDET + Drive `01_raw/statement` (KBANK+KTB) → `bank.statement_*` |

Schedule **SYP before HQ A/B** (e.g. SYP 06:00, HQ 06:30) so both site raw files exist (HQ uploads SYP POs from Drive).

Copy [`.env.example`](.env.example) → `.env` and optionally [`paths.yaml.example`](paths.yaml.example) → `paths.yaml`.

Required: `KCW_ANALYTICS_PYTHON`. Recommended: `KCW_DRIVE_ROOT` or `KCW_ANALYTICS_DATA_ROOT` (use the `G:\Shared drives\...` path — do not point at a DriveFS AppData cache path), `SUPABASE_DB_URL` or `DB_PASSWORD`, HQ `PARTS9_HQ_*` credentials.

**SYP Task Scheduler** runs `python -m src.kcw.pipeline extract --site syp` (DriveFS-safe TEMP → copy → `os.replace`). Manual notebook: [`51_syp_parts9_to_drive_raw.ipynb`](notebooks/51_syp_parts9_to_drive_raw.ipynb).

Daily extract sets:
- **HQ**: full `TABLE_SPECS` including POMAS/PODET + PIMAS/PIDET + ICLOW + BRDET/BPDET
- **SYP**: POMAS/PODET + SIDET/SIMAS/ICMAS/ICLOW (no PIMAS/PIDET — purchases are HQ-only; no BRDET/BPDET)

## CLI (BAT and Claude Cowork)

Run from repo root:

```bash
python -m src.kcw.pipeline gap-check
python -m src.kcw.pipeline extract --site hq
python -m src.kcw.pipeline extract --site syp
python -m src.kcw.pipeline extract --site hq --tables POMAS,PODET
python -m src.kcw.pipeline sync-pomas-podet --site hq
python -m src.kcw.pipeline sync-pomas-podet --site syp
python -m src.kcw.pipeline sync-iclow --site hq
python -m src.kcw.pipeline sync-iclow --site syp
python -m src.kcw.pipeline sync-po-related --site hq
python -m src.kcw.pipeline sync-po-related --site syp
python -m src.kcw.pipeline sync-simas-sidet
python -m src.kcw.pipeline upload-simas-sidet
python -m src.kcw.pipeline sync-brdet-bpdet
python -m src.kcw.pipeline upload-brdet-bpdet
python -m src.kcw.pipeline upload-iclow --site hq
python -m src.kcw.pipeline upload-iclow
python -m src.kcw.pipeline upload-po-related
python -m src.kcw.pipeline upload-pomas-podet --site hq
python -m src.kcw.pipeline upload-daily-raw
python -m src.kcw.pipeline upload-armas-apmas
python -m src.kcw.pipeline tar --catch-up
python -m src.kcw.pipeline tar --date 2026-07-20
python -m src.kcw.pipeline tar --reprocess 2026-07-20
```

`sync-pomas-podet --site {hq|syp}` extracts only POMAS/PODET to Drive, then uploads that site's CSVs to `raw_kcw` (used by the focused HQ/SYP PO worker BATs).

`sync-iclow --site {hq|syp}` extracts that site's `ICLOW` (stock-order / ค้างรับ tracker) to Drive, then uploads to `raw_kcw.raw_{site}_iclow_stock_orders`. See [`docs/parts9_pending_receive.md`](docs/parts9_pending_receive.md).

`sync-brdet-bpdet` extracts HQ `BRDET`/`BPDET` (ทะเบียนเช็ครับ/จ่าย — cheque **or** transfer lines) to Drive, then uploads to `raw_kcw.raw_hq_brdet_cheques_received` / `raw_hq_bpdet_cheques_paid`. `CHKNO` is either a cheque number or a method label (`โอน`, `KSHOP`, …). See [`docs/parts9_cheque_transfers.md`](docs/parts9_cheque_transfers.md). Also included in `upload-daily-raw` and in daily bank sync [`run_bank_statement_import.bat`](worker_tasks/run_bank_statement_import.bat).

`sync-po-related --site {hq|syp}` extracts `POMAS`/`PODET` + `ICLOW` for that site only (HQ and SYP must run on separate machines). For **HQ**, it also syncs `SIDET`/`SIMAS` to `raw_kcw` (latest 6 months). Worker BAT [`run_hq_po_related_sync.bat`](worker_tasks/run_hq_po_related_sync.bat) runs this single trigger then inventory on-hand qty via [`run_inventory_sync.bat`](worker_tasks/run_inventory_sync.bat) → `curated_kcw.inventory_qty_latest`. SYP BAT [`run_syp_po_related_sync.bat`](worker_tasks/run_syp_po_related_sync.bat) does PO/ICLOW + inventory only (no sales Supabase upload).

`sync-simas-sidet` is a focused alias for HQ sales only (same 6-month upload). Also included in `upload-daily-raw` after HQ A extract.

`upload-daily-raw` (HQ A after extract) replaces these `raw_kcw` tables via staging from Drive CSVs:

| CSV | Table | Notes |
|-----|-------|-------|
| `raw_hq_armas_receivable.csv` / `raw_hq_apmas_payable.csv` | matching `raw_hq_*` | account masters |
| `raw_{hq,syp}_pomas_purchase_orders.csv` | matching site tables | PO headers |
| `raw_{hq,syp}_podet_purchase_order_lines.csv` | matching site tables | PO lines |
| `raw_hq_pimas_purchase_bills.csv` / `raw_hq_pidet_purchase_lines.csv` | HQ only | purchase invoices |
| `raw_{hq,syp}_icmas_products.csv` | matching site tables | product masters |
| `raw_hq_rvmas_notes_vouchers.csv` | `raw_hq_rvmas_notes_vouchers` | receipt vouchers (RC*) |
| `raw_hq_pvmas_notes_vouchers.csv` | `raw_hq_pvmas_notes_vouchers` | payment vouchers (P* / KCPN*) |
| `raw_hq_brdet_cheques_received.csv` / `raw_hq_bpdet_cheques_paid.csv` | HQ only | cheque/transfer registers (`CHKNO` = cheque # or label) |
| `raw_{hq,syp}_iclow_stock_orders.csv` | matching site tables | stock orders / pending receive |
| `raw_hq_sidet_sales_lines.csv` / `raw_hq_simas_sales_bills.csv` | HQ only | sales lines / bills (latest 6 months) |

`upload-armas-apmas` remains available as a narrower alias (ARMAS/APMAS only).
### TAR catch-up (idempotent)

- Starts at `max(fin_* billdate) + 1` through `min(today, max eligible raw BILLDATE)`.
- **Skip-if-done** if any `fin_*` row already exists for that day.
- Stages CSVs then calls `billgen.process_all_bill_types_day` (apply migration [`supabase/migrations/20260722160000_tar_catchup_helpers.sql`](supabase/migrations/20260722160000_tar_catchup_helpers.sql)).
- Re-running the same BAT/CLI is safe. Missed days heal on the next successful catch-up **as long as you have not already processed a later day** (forward-only numbering). To force one day: `--reprocess YYYY-MM-DD` (deletes `fin_*` for that date; does not rewind `bill_seq_control`).

## Notebook series

| Band | Role |
|------|------|
| `00_` / `_archive/` / `_playground` | Scratch — do not schedule |
| `20–21_` | Tax / TAR ops |
| `30–34_` | Monthly accounting |
| `50–51_` | Extract / curated (`51_syp_*` thin wrapper; HQ `51_parts9_to_drive`) |
| `60_` | Statement jobs (`61`; clones moved to `_archive`) |
| `70_` | Online channels |
| `90_` | Loaders |

## Claude Cowork cutover

1. Keep SYP + HQ A BATs on Task Scheduler (raw only).
2. Disable HQ B on Scheduler.
3. Point Cowork at this repo + Drive; run the same CLI steps: `gap-check` → curated/reports notebooks or modules → `tar --catch-up`.
4. Do not invent TAR numbers in chat — always execute `src.kcw.tar` / SQL RPCs.

## Supabase

Apply new migrations before relying on catch-up helpers. Use Supabase MCP/SQL for gap inspection (`billgen.max_fin_billdate()`, `cntar_unmatched_log`); keep writes via RPCs.
