/**
 * Import a bank statement Excel file (KBANK / KTB) into bank.statement_*.
 *
 * Auth: signed-in user with RBAC page `bank_statement_sync` (or admin role).
 * Body: multipart/form-data with fields:
 *   - file: .xlsx / .xls / .xlsm
 *   - bank_name: KBANK | KTB (required)
 */
import { createClient } from "npm:@supabase/supabase-js@2";
import { requireBankStatementSyncPermission } from "../_shared/rbac-auth.ts";
import { corsHeaders } from "./cors.ts";
import {
  inferAccountFromFilename,
  parseStatementBytes,
  sha256HexAsync,
  type ParsedLine,
} from "./parser.ts";

const ALLOWED_EXT = new Set([".xlsx", ".xls", ".xlsm"]);
const MAX_BYTES = 15 * 1024 * 1024;
const BUCKET = "bank-statements";

type ImportStatus = "imported" | "skipped" | "failed";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}

function extOf(name: string): string {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

function sanitizeFilename(name: string): string {
  return name.replace(/[^\w.\-()\u0E00-\u0E7F]+/g, "_").slice(0, 180);
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  if (req.method !== "POST") {
    return jsonResponse({ error: "Method not allowed" }, 405);
  }

  try {
    const authHeader = req.headers.get("Authorization");
    if (!authHeader?.startsWith("Bearer ")) {
      return jsonResponse({ error: "Missing Authorization bearer token" }, 401);
    }

    const supabaseUrl = Deno.env.get("SUPABASE_URL") ?? "";
    const anonKey = Deno.env.get("SUPABASE_ANON_KEY") ?? "";
    const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
    if (!supabaseUrl || !anonKey || !serviceKey) {
      return jsonResponse({ error: "Server misconfigured" }, 500);
    }

    const userClient = createClient(supabaseUrl, anonKey, {
      global: { headers: { Authorization: authHeader } },
    });
    const admin = createClient(supabaseUrl, serviceKey);

    const token = authHeader.replace(/^Bearer\s+/i, "");
    const {
      data: { user },
      error: userErr,
    } = await userClient.auth.getUser(token);
    if (userErr || !user) {
      return jsonResponse({ error: "Invalid or expired session" }, 401);
    }

    const perm = await requireBankStatementSyncPermission(admin, user.id);
    if (!perm.ok) {
      return jsonResponse({ error: perm.message }, perm.status);
    }

    const contentType = req.headers.get("content-type") ?? "";
    if (!contentType.toLowerCase().includes("multipart/form-data")) {
      return jsonResponse(
        { error: "Expected multipart/form-data with fields: file, bank_name" },
        400,
      );
    }

    const form = await req.formData();
    const bankRaw = String(form.get("bank_name") ?? "").trim().toUpperCase();
    if (bankRaw !== "KBANK" && bankRaw !== "KTB") {
      return jsonResponse({ error: "bank_name must be KBANK or KTB" }, 400);
    }
    const bankName = bankRaw as "KBANK" | "KTB";

    const fileEntry = form.get("file");
    if (!(fileEntry instanceof File)) {
      return jsonResponse({ error: "Missing file field" }, 400);
    }

    const originalFilename = fileEntry.name || "statement.xlsx";
    const ext = extOf(originalFilename);
    if (!ALLOWED_EXT.has(ext)) {
      return jsonResponse(
        { error: `Unsupported file type ${ext || "(none)"}; use .xlsx, .xls, or .xlsm` },
        400,
      );
    }
    if (fileEntry.size <= 0 || fileEntry.size > MAX_BYTES) {
      return jsonResponse(
        { error: `File size must be between 1 byte and ${MAX_BYTES} bytes` },
        400,
      );
    }

    const bytes = new Uint8Array(await fileEntry.arrayBuffer());
    const fileHash = await sha256HexAsync(bytes);
    const accountGuess = inferAccountFromFilename(originalFilename, bankName);

    const { meta, lines } = await parseStatementBytes(bytes, {
      filename: originalFilename,
      bankName,
      accountNo: accountGuess,
    });
    const resolvedAccount = String(meta.account_no ?? accountGuess ?? "") || null;

    const now = new Date();
    const yyyy = now.getUTCFullYear();
    const mm = String(now.getUTCMonth() + 1).padStart(2, "0");
    const storagePath = `${bankName}/${yyyy}/${mm}/${fileHash.slice(0, 16)}_${sanitizeFilename(originalFilename)}`;
    let storageError: string | null = null;
    {
      const { error: upErr } = await admin.storage.from(BUCKET).upload(storagePath, bytes, {
        contentType: fileEntry.type ||
          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        upsert: true,
      });
      if (upErr) {
        storageError = upErr.message;
        console.error("storage upload failed", upErr);
      }
    }
    const sourcePath = storageError
      ? `upload://${originalFilename}`
      : `storage://${BUCKET}/${storagePath}`;

    const bank = admin.schema("bank");

    const { data: existing, error: existErr } = await bank
      .from("statement_import_files")
      .select("id, status, row_count, inserted_count, duplicate_count")
      .eq("file_hash", fileHash)
      .maybeSingle();
    if (existErr) throw existErr;

    let fileId: string;
    let isNewFile = false;

    if (existing?.id) {
      fileId = existing.id;
      const { error: bumpErr } = await bank
        .from("statement_import_files")
        .update({
          last_seen_at: new Date().toISOString(),
          source_path: sourcePath,
          bank_name: bankName,
          account_no: resolvedAccount,
          original_filename: originalFilename,
        })
        .eq("id", fileId);
      if (bumpErr) throw bumpErr;

      if (existing.status === "imported") {
        return jsonResponse({
          status: "skipped" satisfies ImportStatus,
          file_id: fileId,
          file_hash: fileHash,
          bank_name: bankName,
          account_no: resolvedAccount,
          original_filename: originalFilename,
          source_path: sourcePath,
          row_count: existing.row_count ?? lines.length,
          inserted_count: existing.inserted_count ?? 0,
          duplicate_count: existing.duplicate_count ?? 0,
          storage_error: storageError,
          message: "File already imported (same file_hash)",
        });
      }

      if (existing.status === "duplicate" && (existing.row_count ?? 0) > 0) {
        return jsonResponse({
          status: "skipped" satisfies ImportStatus,
          file_id: fileId,
          file_hash: fileHash,
          bank_name: bankName,
          account_no: resolvedAccount,
          original_filename: originalFilename,
          source_path: sourcePath,
          row_count: existing.row_count,
          inserted_count: existing.inserted_count ?? 0,
          duplicate_count: existing.duplicate_count ?? 0,
          storage_error: storageError,
          message: "Duplicate file with prior rows; skipped",
        });
      }
    } else {
      isNewFile = true;
      const { data: inserted, error: insErr } = await bank
        .from("statement_import_files")
        .insert({
          file_hash: fileHash,
          original_filename: originalFilename,
          source_path: sourcePath,
          bank_name: bankName,
          account_no: resolvedAccount,
          status: "pending",
          row_count: 0,
          inserted_count: 0,
          duplicate_count: 0,
          error_count: 0,
          error_message: null,
          raw_metadata: meta,
        })
        .select("id")
        .single();
      if (insErr) throw insErr;
      fileId = inserted.id;
    }

    try {
      await setFileStatus(bank, fileId, {
        status: "importing",
        row_count: lines.length,
        inserted_count: 0,
        duplicate_count: 0,
        error_count: 0,
        error_message: null,
        raw_metadata: meta,
        account_no: resolvedAccount,
      });

      const { inserted_count, duplicate_count } = await insertStatementLines(
        bank,
        fileId,
        lines,
      );

      await setFileStatus(bank, fileId, {
        status: "imported",
        row_count: lines.length,
        inserted_count,
        duplicate_count,
        error_count: 0,
        error_message: null,
      });

      return jsonResponse({
        status: "imported" satisfies ImportStatus,
        file_id: fileId,
        file_hash: fileHash,
        bank_name: bankName,
        account_no: resolvedAccount,
        original_filename: originalFilename,
        source_path: sourcePath,
        is_new_file: isNewFile,
        row_count: lines.length,
        inserted_count,
        duplicate_count,
        storage_error: storageError,
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      await setFileStatus(bank, fileId, {
        status: "failed",
        row_count: lines.length,
        inserted_count: 0,
        duplicate_count: 0,
        error_count: 1,
        error_message: msg,
      });
      return jsonResponse(
        {
          status: "failed" satisfies ImportStatus,
          file_id: fileId,
          file_hash: fileHash,
          bank_name: bankName,
          account_no: resolvedAccount,
          original_filename: originalFilename,
          source_path: sourcePath,
          row_count: lines.length,
          inserted_count: 0,
          duplicate_count: 0,
          error_count: 1,
          error_message: msg,
          storage_error: storageError,
        },
        500,
      );
    }
  } catch (e) {
    console.error(e);
    const msg = e instanceof Error ? e.message : String(e);
    return jsonResponse({ error: msg }, 500);
  }
});

// deno-lint-ignore no-explicit-any
async function setFileStatus(
  bank: any,
  fileId: string,
  fields: {
    status: string;
    row_count: number;
    inserted_count: number;
    duplicate_count: number;
    error_count: number;
    error_message: string | null;
    raw_metadata?: Record<string, unknown>;
    account_no?: string | null;
  },
) {
  const payload: Record<string, unknown> = {
    status: fields.status,
    row_count: fields.row_count,
    inserted_count: fields.inserted_count,
    duplicate_count: fields.duplicate_count,
    error_count: fields.error_count,
    error_message: fields.error_message,
    processed_at: new Date().toISOString(),
    last_seen_at: new Date().toISOString(),
  };
  if (fields.raw_metadata) payload.raw_metadata = fields.raw_metadata;
  if (fields.account_no !== undefined) payload.account_no = fields.account_no;

  const { error } = await bank.from("statement_import_files").update(payload).eq("id", fileId);
  if (error) throw error;
}

// deno-lint-ignore no-explicit-any
async function insertStatementLines(
  bank: any,
  fileId: string,
  lines: ParsedLine[],
): Promise<{ inserted_count: number; duplicate_count: number }> {
  if (!lines.length) return { inserted_count: 0, duplicate_count: 0 };

  const rows = lines.map((x) => ({
    account_no: x.account_no,
    bank_name: x.bank_name,
    txn_date: x.txn_date,
    value_date: x.value_date,
    description: x.description,
    bank_reference: x.bank_reference,
    amount: x.amount,
    direction: x.direction,
    debit: x.debit,
    credit: x.credit,
    balance_after: x.balance_after,
    transaction_fingerprint: x.transaction_fingerprint,
    source_file_id: fileId,
    source_sheet_name: x.source_sheet_name,
    source_row_number: x.source_row_number,
    raw_json: x.raw_json,
    match_status: "pending",
  }));

  let inserted = 0;
  const pageSize = 500;
  for (let i = 0; i < rows.length; i += pageSize) {
    const chunk = rows.slice(i, i + pageSize);
    const { data, error } = await bank
      .from("statement_lines")
      .upsert(chunk, {
        onConflict: "transaction_fingerprint",
        ignoreDuplicates: true,
      })
      .select("transaction_fingerprint");
    if (error) throw error;
    inserted += data?.length ?? 0;
  }

  return {
    inserted_count: inserted,
    duplicate_count: rows.length - inserted,
  };
}
