# import-bank-statement (analytics mirror)

**Source of truth: [kcw-v2](https://github.com/pthengtr/kcw-v2)**
(`supabase/functions/import-bank-statement/`, RBAC via `_shared/rbac-auth.ts`).

This repo keeps a **mirror** for the Drive/notebook BAT path and docs. Production
Edge Function `import-bank-statement` **v14+** was deployed from kcw-v2 with:

- `PARSER_VERSION = "auto_v2"`
- `fingerprint.ts` (stable detail, not display description)
- RBAC page permission `bank_statement_sync` (not `kcw_admin`)

## Do not deploy without syncing kcw-v2

Deploying a stale analytics bundle previously reintroduced `auto_v1` hashing and
`kcw_admin`-only auth, which caused 120 overlapping-import duplicates
(see kcw-v2 #145 / #147).

Before any production deploy from this repo:

1. Diff/sync `fingerprint.ts`, `parser.ts`, `index.ts`, `_shared/rbac-auth.ts`
   from the current kcw-v2 main (or the deploy PR).
2. Rebuild the MCP artifact: `python scripts/bundle_import_bank_statement.py`
3. Confirm the bundle contains `auto_v2` + `requireBankStatementSyncPermission`
   and does **not** query `kcw_admin`.
4. Prefer deploying from **kcw-v2** when both repos have changes.

## Local modules

| File | Role |
|------|------|
| `fingerprint.ts` | Canonical SHA-256 identity |
| `parser.ts` | Excel parse + calls fingerprint |
| `index.ts` | HTTP + RBAC + upsert |
| `cors.ts` | CORS headers |
| `../_shared/rbac-auth.ts` | `bank_statement_sync` page check |
| `index.bundled.ts` | Generated single-file deploy artifact |

Rebuild:

```bash
python scripts/bundle_import_bank_statement.py
```
