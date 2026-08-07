#!/usr/bin/env python3
"""Rebuild index.bundled.ts for MCP deploy.

Source of truth for import-bank-statement is **kcw-v2**. This analytics copy is a
mirror. Always sync from kcw-v2 before rebuilding/deploying, and never deploy an
older analytics bundle over production (currently auto_v2 + RBAC v14+).

Modules inlined (edit these, then re-run this script):
  supabase/functions/_shared/rbac-auth.ts
  supabase/functions/import-bank-statement/cors.ts
  supabase/functions/import-bank-statement/fingerprint.ts
  supabase/functions/import-bank-statement/parser.ts
  supabase/functions/import-bank-statement/index.ts
"""
from __future__ import annotations

import re
from pathlib import Path

FN = Path(__file__).resolve().parents[1] / "supabase" / "functions"
BASE = FN / "import-bank-statement"
SHARED = FN / "_shared"


def _strip_relative_imports(src: str) -> str:
    return re.sub(
        r'^import\s+[\s\S]*?from\s+["\']\.[^"\']+["\']\s*;?\s*\n',
        "",
        src,
        flags=re.M,
    )


def _demote_exports(src: str, names: list[str]) -> str:
    out = src
    for name in names:
        out = out.replace(f"export const {name}", f"const {name}")
        out = out.replace(f"export type {name}", f"type {name}")
        out = out.replace(f"export async function {name}", f"async function {name}")
        out = out.replace(f"export function {name}", f"function {name}")
    # Drop re-export lines like: export { sha256HexAsync } from "./fingerprint.ts";
    out = re.sub(
        r'^export\s+\{[^}]+\}\s+from\s+["\'][^"\']+["\']\s*;?\s*\n',
        "",
        out,
        flags=re.M,
    )
    return out


def main() -> None:
    cors = (BASE / "cors.ts").read_text()
    fingerprint = (BASE / "fingerprint.ts").read_text()
    parser = (BASE / "parser.ts").read_text()
    index = (BASE / "index.ts").read_text()
    rbac = (SHARED / "rbac-auth.ts").read_text()

    # Keep only the npm supabase-js import in the bundle entry.
    index_no_rel = _strip_relative_imports(index)
    # Drop type-only import of SupabaseClient from rbac (not needed at runtime).
    rbac_local = re.sub(
        r'^import\s+type\s+[\s\S]*?from\s+["\'][^"\']+["\']\s*;?\s*\n',
        "",
        rbac,
        flags=re.M,
    )
    rbac_local = _demote_exports(
        rbac_local,
        [
            "ROLE_ADMIN",
            "BANK_STATEMENT_SYNC_PAGE",
            "RbacAuthResult",
            "requireBankStatementSyncPermission",
        ],
    )
    # rbac uses SupabaseClient type — replace with any for the single-file bundle.
    rbac_local = rbac_local.replace(
        "admin: SupabaseClient,",
        "admin: { from: (table: string) => any },",
    )

    cors_local = cors.replace("export const corsHeaders", "const corsHeaders")

    fp_local = _demote_exports(
        fingerprint,
        [
            "TransactionFingerprintInput",
            "TRANSACTION_DETAIL_COL_PATTERNS",
            "normText",
            "normMoney",
            "sha256HexAsync",
            "buildTransactionFingerprint",
            "extractTransactionDetailFromRaw",
        ],
    )

    parser_no_rel = _strip_relative_imports(parser)
    parser_local = _demote_exports(
        parser_no_rel,
        [
            "PARSER_VERSION",
            "ParsedLine",
            "ParseResult",
            "inferAccountFromFilename",
            "parseStatementBytes",
        ],
    )

    bundled = f"""/**
 * Bundled import-bank-statement Edge Function for deploy.
 *
 * DO NOT DEPLOY from this repo without first syncing sources from kcw-v2.
 * Production is owned by kcw-v2 (auto_v2 + RBAC page bank_statement_sync).
 * Deploying a stale analytics bundle reintroduced auto_v1 / kcw_admin and caused
 * overlapping-import duplicates (see kcw-v2 #145 / #147).
 *
 * Prefer editing modules listed in scripts/bundle_import_bank_statement.py then:
 *   python scripts/bundle_import_bank_statement.py
 */
import {{ createClient }} from "npm:@supabase/supabase-js@2"

{cors_local}

{fp_local}

{parser_local}

{rbac_local}

{index_no_rel}
"""
    out = BASE / "index.bundled.ts"
    out.write_text(bundled)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
