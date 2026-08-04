#!/usr/bin/env python3
"""Rebuild index.bundled.ts from cors.ts + parser.ts + index.ts for MCP deploy."""
from __future__ import annotations

import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "supabase" / "functions" / "import-bank-statement"


def main() -> None:
    cors = (BASE / "cors.ts").read_text()
    parser = (BASE / "parser.ts").read_text()
    index = (BASE / "index.ts").read_text()

    index_no_imports = re.sub(
        r'^import\s+[\s\S]*?from\s+["\'][^"\']+["\']\s*;?\s*\n',
        "",
        index,
        flags=re.M,
    )

    cors_local = cors.replace("export const corsHeaders", "const corsHeaders")
    repls = [
        ("export const PARSER_VERSION", "const PARSER_VERSION"),
        ("export type ParsedLine", "type ParsedLine"),
        ("export type ParseResult", "type ParseResult"),
        ("export async function sha256HexAsync", "async function sha256HexAsync"),
        ("export function normText", "function normText"),
        ("export function normMoney", "function normMoney"),
        ("export function parseDayFirstDate", "function parseDayFirstDate"),
        ("export function inferAccountFromFilename", "function inferAccountFromFilename"),
        ("export async function parseStatementBytes", "async function parseStatementBytes"),
    ]
    parser_local = parser
    for a, b in repls:
        parser_local = parser_local.replace(a, b)

    bundled = f"""/**
 * Bundled import-bank-statement Edge Function for deploy.
 * Prefer editing cors.ts / parser.ts / index.ts then: python scripts/bundle_import_bank_statement.py
 */
import {{ createClient }} from "npm:@supabase/supabase-js@2"

{cors_local}

{parser_local}

{index_no_imports}
"""
    out = BASE / "index.bundled.ts"
    out.write_text(bundled)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
