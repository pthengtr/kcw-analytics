# Bank statement upload & import

Two ingestion paths write the same `bank.statement_*` tables. Both must use
**auto_v2** fingerprints identical to production `bank.fp_build_hash`.

| Path | Entry | Auth |
|------|--------|------|
| Daily HQ BAT (Drive Excel) | `worker_tasks/run_bank_statement_import.bat` → `notebooks/02_bank_statement_import_test.ipynb` | Direct Postgres (`SUPABASE_DB_*`) |
| Web upload (kcw-v2 UI) | Edge Function `import-bank-statement` | JWT + RBAC page `bank_statement_sync` (or admin role) |

## Dual-repo Edge Function risk

**Source of truth for the Edge Function is [kcw-v2](https://github.com/pthengtr/kcw-v2)**
(production deploy v14+: auto_v2 + RBAC). This analytics repo mirrors the function
for docs/BAT parity.

Do **not** deploy an older analytics bundle over production. A prior drift back to
`auto_v1` + `kcw_admin` auth caused overlapping monthly/cumulative files to insert
duplicate `statement_lines` (kcw-v2 #145 / #147 — e.g. 3557 ด.4.xlsx, KBANK7236).

Before deploying from analytics: sync from kcw-v2, rebuild
`index.bundled.ts`, confirm `PARSER_VERSION = "auto_v2"` and
`requireBankStatementSyncPermission`. Prefer deploying from kcw-v2.

See `supabase/functions/import-bank-statement/README.md`.

## Fingerprint identity (auto_v2)

```
account_no | txn_date | amount | direction | stable_transaction_detail | bank_reference | balance_after
```

`stable_transaction_detail` comes from raw_json keys
`รายละเอียด` / `DESCRIPTION` / `PARTICULAR` / `NARRATION` — **not** display
`รายการ` / time. Shared implementations:

- Python: `src/kcw/bank_statement.py::compute_transaction_fingerprint`
- SQL: `bank.fp_build_hash` (canonical); `bank.build_transaction_fingerprint` wraps it
- TS: `supabase/functions/import-bank-statement/fingerprint.ts`

Notebook + BAT set `parser_version: auto_v2` and
`ON CONFLICT (transaction_fingerprint) DO NOTHING`.

## Recommendation (web)

| Piece | Where |
|-------|--------|
| Parse + insert into `bank.statement_*` | Supabase Edge Function `import-bank-statement` (**kcw-v2** owns deploy) |
| Upload UI (file picker, bank select, progress) | **kcw-v2** |

## Endpoint

```
POST {SUPABASE_URL}/functions/v1/import-bank-statement
Authorization: Bearer <user access_token>
apikey: <anon or publishable key>
Content-Type: multipart/form-data
```

### Form fields

| Field | Required | Notes |
|-------|----------|-------|
| `file` | yes | `.xlsx` / `.xls` / `.xlsm`, max 15 MiB |
| `bank_name` | yes | `KBANK` or `KTB` (explicit; do not infer from filename alone) |

### Auth

Caller must be signed in with RBAC permission for page `bank_statement_sync`
(or role `admin`). Implemented in `_shared/rbac-auth.ts` (not `kcw_admin`).

### Success response (200)

```json
{
  "status": "imported",
  "file_id": "uuid",
  "file_hash": "sha256…",
  "bank_name": "KBANK",
  "account_no": "xxx-x-xxxxx-x",
  "original_filename": "KBANK3557_07.xlsx",
  "source_path": "storage://bank-statements/KBANK/2026/08/…",
  "is_new_file": true,
  "row_count": 120,
  "inserted_count": 118,
  "duplicate_count": 2,
  "storage_error": null
}
```

`status` may also be `"skipped"` (same `file_hash` already imported) or `"failed"`.

### Example (kcw-v2)

```ts
const form = new FormData()
form.append("file", file)
form.append("bank_name", "KBANK") // or "KTB"

const { data, error } = await supabase.functions.invoke("import-bank-statement", {
  body: form,
})
```

Do **not** set `Content-Type` manually when using `FormData` — the browser/SDK sets the multipart boundary.

## Behavior (parity: notebook BAT ↔ web)

1. SHA-256 the file bytes → `bank.statement_import_files.file_hash` (dedupe).
2. Parse sheets with `parser_version: auto_v2` (stable transaction fingerprints).
3. Insert lines into `bank.statement_lines` with `ON CONFLICT (transaction_fingerprint) DO NOTHING`.
4. Web path stores the raw Excel in private Storage bucket `bank-statements`
   (`KBANK|KTB/YYYY/MM/{hash16}_{filename}`); BAT path keeps Drive `G:\...` as `source_path`.

The daily HQ BAT (`worker_tasks/run_bank_statement_import.bat`) scans Drive for
offline drops; web uploads no longer need Drive.

## Storage

Bucket: `bank-statements` (private). Migration: `supabase/migrations/20260804040000_bank_statement_storage_bucket.sql`.

Admins can SELECT (download) objects; the Edge Function uploads with the service role.

Rebuild (mirror only — see dual-repo warning above):

```bash
python scripts/bundle_import_bank_statement.py
# then deploy index.bundled.ts as index.ts via Supabase MCP deploy_edge_function
# ONLY after syncing from kcw-v2 — or deploy from kcw-v2 instead
```

Source modules (edit / sync these):

```
supabase/functions/_shared/rbac-auth.ts
supabase/functions/import-bank-statement/fingerprint.ts
supabase/functions/import-bank-statement/index.ts
supabase/functions/import-bank-statement/parser.ts
supabase/functions/import-bank-statement/cors.ts
```

Deployed artifact: `index.bundled.ts` (single-file bundle for MCP deploy).

## Status

- Production function is owned by **kcw-v2** (auto_v2 + RBAC). Do not regress to `kcw_admin` / `auto_v1`.
- Storage bucket `bank-statements` is private (15 MiB limit).
- Analytics notebook/BAT path uses the same Python fingerprint helpers as `bank.fp_build_hash`.
