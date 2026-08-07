# import-bank-statement (analytics mirror — DO NOT DEPLOY)

**Source of truth: [kcw-v2](https://github.com/pthengtr/kcw-v2)**
(`supabase/functions/import-bank-statement/`).

Production is deployed from kcw-v2 via Supabase CLI. Drive bulk no longer runs a
local parser — HQ uses `scripts/upload_drive_bank_statements.py`.

This folder is a **stale mirror**. Do **not** deploy `index.bundled.ts` from here;
doing so previously reintroduced `auto_v1` / wrong auth and created duplicate lines.

Prefer deleting or sync-freezing this mirror after confirming kcw-v2 is the only
deploy source.
