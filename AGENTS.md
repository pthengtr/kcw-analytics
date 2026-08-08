# AGENTS.md

## Cursor Cloud specific instructions

`kcw-analytics` is a **Python ETL / bill-generation pipeline** (no web UI, no
build step). The core entrypoint is the CLI `python -m src.kcw.pipeline`
(`gap-check`, `extract`, `tar`). See `README.md` for the canonical command list
and the notebook/BAT orchestration.

### Python environment
- Deps live in a venv at `.venv` (recreated by the startup update script from
  `requirements.txt`). Run tools as `.venv/bin/python ...` or `source .venv/bin/activate`.
- Ubuntu is PEP 668 "externally managed" — do not `pip install` system-wide; use
  the venv.

### Database (local Postgres stands in for Supabase)
- The pipeline talks to Postgres via `SUPABASE_DB_URL` (or `DB_PASSWORD` + host
  env). In this VM a local **PostgreSQL 16** cluster holds the `billgen` schema.
- It is **not** started automatically. Start it each session with:
  `sudo pg_ctlcluster 16 main start` (check: `sudo pg_lsclusters`).
- Local DB `kcw_analytics`, connection string:
  `postgresql://postgres:postgres@127.0.0.1:5432/kcw_analytics`.
- Migrations in `supabase/migrations/*.sql` are already applied to that DB. To
  reapply from scratch, `psql ... -f` each file in filename order.
- Gotcha: the two newest migrations (`*_fact_sales_bills_all*.sql`) assume a
  `curated_kcw` schema and the Supabase roles `anon`/`authenticated`/`service_role`
  that the `billgen` migrations do not create. Those objects were pre-created in
  this VM so all migrations apply cleanly; on a truly fresh Postgres, create the
  `curated_kcw` schema and those three `NOLOGIN` roles before applying migrations
  (they are unrelated to the TAR/billgen flow). Supabase CLI / Docker are not
  installed.

### Local config & sample data (not committed; `.env`, `*.csv` are gitignored)
- `.env` (repo root) sets `SUPABASE_DB_URL`, `KCW_ANALYTICS_DATA_ROOT`, and
  `KCW_ANALYTICS_PYTHON`. `pipeline.py` auto-loads it via python-dotenv.
- `extract --site {hq,syp}` reads Windows SQL Server (PARTS9) over ODBC and
  **cannot run on Linux**. To exercise the downstream `tar` / `gap-check` flow,
  fabricate raw CSVs with `python scripts/dev_seed_sample_data.py` (writes into
  `$KCW_ANALYTICS_DATA_ROOT/01_raw`), then run
  `python -m src.kcw.pipeline tar --catch-up`.
- `raw_dir()` must be a real, existing directory; the path code intentionally
  errors instead of creating placeholder folders.

### Lint / test / build
- No lint config, no automated test suite, and no build/packaging exist. "Build"
  = venv + `pip install -r requirements.txt`. Validate changes by running the CLI
  end-to-end against the local DB (see above).
