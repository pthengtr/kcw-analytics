# Bank statement web upload

Upload KBANK / KTB Excel statements from **kcw-v2** without Google Drive.

## Recommendation

| Piece | Where |
|-------|--------|
| Parse + insert into `bank.statement_*` | Supabase Edge Function `import-bank-statement` (this repo) |
| Upload UI (file picker, bank select, progress) | **kcw-v2** |

Doing the parse in kcw-v2 alone is possible (Next.js API + SheetJS), but worse: privileged writes and bank-statement heuristics would live in the frontend repo and drift from the Python Drive importer. Prefer the Edge Function; keep kcw-v2 thin.

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

Caller must be signed in. Email must appear in `public.kcw_admin.user_id` (same check as existing bank RLS SELECT policies).

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

## Behavior (parity with Drive notebook)

1. SHA-256 the file bytes → `bank.statement_import_files.file_hash` (dedupe).
2. Parse sheets with the same header / column / account-metadata heuristics as `notebooks/02_bank_statement_import_test.ipynb` (`parser_version: auto_v1`).
3. Insert lines into `bank.statement_lines` with `ON CONFLICT (transaction_fingerprint) DO NOTHING`.
4. Store the raw Excel in private Storage bucket `bank-statements` (path `KBANK|KTB/YYYY/MM/{hash16}_{filename}`).
5. `source_path` is `storage://bank-statements/...` instead of a Drive `G:\...` path.

The daily HQ BAT (`worker_tasks/run_bank_statement_import.bat`) can keep scanning Drive for offline drops; web uploads no longer need Drive.

## Storage

Bucket: `bank-statements` (private). Migration: `supabase/migrations/20260804040000_bank_statement_storage_bucket.sql`.

Admins can SELECT (download) objects; the Edge Function uploads with the service role.

Redeploy after changes:

```bash
python scripts/bundle_import_bank_statement.py
# then deploy index.bundled.ts as index.ts via Supabase MCP deploy_edge_function
# or: supabase functions deploy import-bank-statement --project-ref jdzitzsucntqbjvwiwxm
```

Source modules (edit these):

```
supabase/functions/import-bank-statement/index.ts
supabase/functions/import-bank-statement/parser.ts
supabase/functions/import-bank-statement/cors.ts
```

Deployed artifact: `index.bundled.ts` (single-file bundle for MCP deploy).

## Status (as of deploy)

- Function `import-bank-statement` is **ACTIVE** on project `jdzitzsucntqbjvwiwxm` (`verify_jwt: true`).
- Storage bucket `bank-statements` is private (15 MiB limit).
