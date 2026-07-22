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
| **SYP** | [`worker_tasks/run_syp_parts9_to_drive_raw.bat`](worker_tasks/run_syp_parts9_to_drive_raw.bat) | `PARTS9` → `raw_syp_*.csv` |
| **HQ A** | [`worker_tasks/run_hq_parts9_to_drive_raw.bat`](worker_tasks/run_hq_parts9_to_drive_raw.bat) | `PARTS9` → `raw_hq_*.csv` only |
| **HQ B** | [`worker_tasks/run_hq_parts9_full_pipeline.bat`](worker_tasks/run_hq_parts9_full_pipeline.bat) | HQ A + archive + curated + VAT/TAR + Excel + upload |

Schedule **SYP before HQ B** (e.g. SYP 06:00, HQ 06:30) so both site raw files exist.

Copy [`.env.example`](.env.example) → `.env` and optionally [`paths.yaml.example`](paths.yaml.example) → `paths.yaml`.

Required: `KCW_ANALYTICS_PYTHON`. Recommended: `KCW_DRIVE_ROOT` or `KCW_ANALYTICS_DATA_ROOT` (use the `G:\Shared drives\...` path — do not point at a DriveFS AppData cache path), `SUPABASE_DB_URL` or `DB_PASSWORD`, HQ `PARTS9_HQ_*` credentials.

Extract for **SYP Task Scheduler** runs [`51_syp_parts9_to_drive_raw.ipynb`](notebooks/51_syp_parts9_to_drive_raw.ipynb) via nbconvert.
That notebook writes each CSV via **local TEMP → copy onto Drive → `os.replace`** (no `fsync`) so large
`raw_syp_sidet_*` / `raw_syp_icmas_*` files update under Google Drive File Stream. Plain in-place `to_csv`
often left those two stale while smaller files updated.

## CLI (BAT and Claude Cowork)

Run from repo root:

```bash
python -m src.kcw.pipeline gap-check
python -m src.kcw.pipeline extract --site hq
python -m src.kcw.pipeline extract --site syp
python -m src.kcw.pipeline tar --catch-up
python -m src.kcw.pipeline tar --date 2026-07-20
python -m src.kcw.pipeline tar --reprocess 2026-07-20
```

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
