# Bank statement upload & import

**Single parser:** production Edge Function `import-bank-statement` in
**[kcw-v2](https://github.com/pthengtr/kcw-v2)** (`parser_version: auto_v2`).

| Path | Entry | Auth |
|------|--------|------|
| Daily HQ BAT (Drive Excel) | `worker_tasks/run_bank_statement_import.bat` → `scripts/upload_drive_bank_statements.py` | Service-role bearer |
| Web upload (kcw-v2 UI) | Same Edge Function | JWT + RBAC page `bank_statement_sync` (or admin) |

Do **not** parse/insert statements from the old notebook path in production.
`notebooks/02_bank_statement_import_test.ipynb` is archive/reference only.

Do **not** deploy Edge Function bundles from this analytics repo over production —
kcw-v2 owns deploy (local CLI: `supabase functions deploy import-bank-statement`).

## Fingerprint identity (auto_v2)

```
account_no | txn_date | amount | direction | normalized_stable_detail | bank_reference | balance_after
```

`stable_transaction_detail` comes from raw_json keys
`รายละเอียด` / `DESCRIPTION` / `PARTICULAR` / `NARRATION` — **not** display
`รายการ` / time. KTB detail is normalized (strip trailing online transfer ids /
`Tran:` / `Future Amount` noise) so old `DownLoadService` and new Thai Corporate
Online exports of the same txn share one fingerprint.

Canonical TS: kcw-v2 `supabase/functions/import-bank-statement/fingerprint.ts`
(+ SQL `bank.fp_build_hash` for DB-side tooling).

Idempotency:

1. SHA-256 file bytes → `bank.statement_import_files.file_hash` (skip if already imported)
2. Lines `ON CONFLICT (transaction_fingerprint) DO NOTHING` — **does not overwrite** existing `match_*`

## Drive bulk uploader

```bash
# dry-run (list files)
python scripts/upload_drive_bank_statements.py --dry-run

# upload all KBANK + KTB under the Drive statement folder
python scripts/upload_drive_bank_statements.py
```

Requires `.env`:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- optional `BANK_STATEMENT_BASE_DIR` (default
  `G:\Shared drives\KCW-Data\kcw_analytics\01_raw\statement`)

## Web upload (kcw-v2)

```
POST {SUPABASE_URL}/functions/v1/import-bank-statement
Authorization: Bearer <user access_token>
apikey: <anon or publishable key>
Content-Type: multipart/form-data
```

| Field | Required | Notes |
|-------|----------|-------|
| `file` | yes | `.xlsx` / `.xls` / `.xlsm`, max 15 MiB |
| `bank_name` | yes | `KBANK` or `KTB` |

## KTB layouts

| Sheet | Amount | Detail |
|-------|--------|--------|
| `DownLoadService` | signed `Amount` | `Description` |
| `Account_Statement_Report_TH_XLS` | signed `ถอนเงิน/ฝากเงิน` | `รายละเอียด` |

## Storage

Web path stores raw Excel in private bucket `bank-statements`.
Drive path keeps `G:\...` as the operator drop folder; Edge Function still stores
a copy under Storage when upload succeeds.

## Status

- Production function owned by **kcw-v2** (auto_v2 + RBAC + service-role bulk).
- HQ BAT uses thin uploader only — no competing Python parser for inserts.
