/**
 * Bundled import-bank-statement Edge Function for deploy.
 * Prefer editing cors.ts / parser.ts / index.ts then: python scripts/bundle_import_bank_statement.py
 */
import { createClient } from "npm:@supabase/supabase-js@2"

const corsHeaders: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
}


/**
 * Port of notebooks/02_bank_statement_import_test.ipynb + src/kcw/bank_statement.py
 * parser_version: auto_v2 (stable transaction fingerprints across overlapping exports)
 */
import * as XLSX from "npm:xlsx@0.18.5"

const PARSER_VERSION = "auto_v2"

const ACCOUNT_METADATA_LABELS = new Set([
  "ACCOUNT NO.",
  "ACCOUNT NO",
  "ACCOUNT NUMBER",
  "เลขที่บัญชี",
  "เลขที่บัญชีเงินฝาก",
])

const ACCOUNT_METADATA_LABEL_PREFIXES = [
  "ACCOUNT NO",
  "ACCOUNT NUMBER",
  "เลขที่บัญชี",
]

type ParsedLine = {
  account_no: string
  bank_name: string | null
  txn_date: string // YYYY-MM-DD
  value_date: string | null
  description: string | null
  bank_reference: string | null
  amount: number
  direction: "in" | "out"
  debit: number | null
  credit: number | null
  balance_after: number | null
  transaction_fingerprint: string
  source_sheet_name: string | null
  source_row_number: number | null
  raw_json: Record<string, unknown>
}

type ParseResult = {
  meta: Record<string, unknown>
  lines: ParsedLine[]
}

function isBlank(v: unknown): boolean {
  if (v === null || v === undefined) return true
  if (typeof v === "number" && Number.isNaN(v)) return true
  if (typeof v === "string" && v.trim() === "") return true
  return false
}

async function sha256HexAsync(data: ArrayBuffer | Uint8Array | string): Promise<string> {
  let bytes: Uint8Array
  if (typeof data === "string") {
    bytes = new TextEncoder().encode(data)
  } else if (data instanceof Uint8Array) {
    bytes = data
  } else {
    bytes = new Uint8Array(data)
  }
  const hash = await crypto.subtle.digest("SHA-256", bytes as BufferSource)
  return Array.from(new Uint8Array(hash))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
}

function normText(x: unknown): string {
  if (isBlank(x)) return ""
  let s = String(x)
  s = s.replace(/\u00A0/g, " ")
  s = s.trim().toUpperCase()
  s = s.replace(/\s+/g, " ")
  return s
}

/** Keys in raw_json that carry stable bank transaction detail (not display labels). */
const STABLE_DETAIL_KEYS = [
  "รายละเอียด", // KBANK Thai export
  "DESCRIPTION", // KTB / English exports
  "DETAIL",
  "PARTICULAR",
] as const

/**
 * Extract stable transaction detail from parsed row cells.
 * KBANK cumulative exports may put a time or item label in `description`; the
 * underlying transfer detail (counterparty, reference text) lives here instead.
 */
function extractStableTransactionDetail(
  raw: Record<string, unknown>,
): string {
  for (const key of STABLE_DETAIL_KEYS) {
    const val = raw[key]
    if (!isBlank(val)) return String(val).trim()
  }
  return ""
}

type TransactionFingerprintInput = {
  accountNo: string
  txnDate: string
  amount: number | string
  direction: "in" | "out"
  /** Stable bank detail text (not the display description column). */
  stableDetail: string | null
  bankReference: string | null
  balanceAfter: number | null
  /** When false, balance is omitted from identity (sheet had no balance column). */
  hasBalanceColumn?: boolean
}

/**
 * Canonical transaction identity for duplicate detection across overlapping exports.
 *
 * Identity fields (in order):
 *   account_no, txn_date, amount, direction, stable_detail, bank_reference, balance_after
 *
 * Display `description` is intentionally excluded — KBANK exports may show the same
 * transfer as a time ("09:12:00") in one file and an item label ("รับโอนเงิน") in another.
 * `balance_after` disambiguates legitimate same-day same-amount sequences.
 */
async function buildTransactionFingerprint(
  input: TransactionFingerprintInput,
): Promise<string> {
  const fpInput = [
    normText(input.accountNo),
    input.txnDate,
    normMoney(input.amount),
    normText(input.direction),
    normText(input.stableDetail),
    normText(input.bankReference),
    input.hasBalanceColumn !== false ? normMoney(input.balanceAfter) : "",
  ].join("|")
  return sha256HexAsync(fpInput)
}

function normMoney(x: unknown): string {
  if (isBlank(x)) return ""
  const cleaned = String(x).replace(/,/g, "").trim()
  if (!cleaned) return ""
  const n = Number(cleaned)
  if (!Number.isFinite(n)) return ""
  const sign = n < 0 ? -1 : 1
  const abs = Math.abs(n)
  // ROUND_HALF_UP to 2dp
  const scaled = abs * 100
  const whole = Math.floor(scaled + 1e-9)
  const frac = scaled - whole
  let cents = frac >= 0.5 - 1e-12 ? whole + 1 : whole
  // handle floating noise near .xx5
  if (Math.abs(frac - 0.5) < 1e-9) cents = whole + 1
  const out = (sign * cents) / 100
  return out.toFixed(2)
}

function parseDayFirstDate(value: unknown): string | null {
  if (isBlank(value)) return null

  if (value instanceof Date && !Number.isNaN(value.getTime())) {
    return toIsoDateUTC(value)
  }

  if (typeof value === "number" && Number.isFinite(value)) {
    // Excel serial date
    const parsed = XLSX.SSF?.parse_date_code?.(value)
    if (parsed) {
      const d = new Date(Date.UTC(parsed.y, parsed.m - 1, parsed.d))
      return toIsoDateUTC(d)
    }
  }

  const s = String(value).trim()
  // DD/MM/YYYY or DD-MM-YYYY (Thai bank exports)
  const m = s.match(/^(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})(?:\s|$)/)
  if (m) {
    let year = Number(m[3])
    if (year < 100) year += 2000
    // Buddhist Era heuristic (e.g. 2568)
    if (year > 2400) year -= 543
    const day = Number(m[1])
    const month = Number(m[2])
    if (month >= 1 && month <= 12 && day >= 1 && day <= 31) {
      return `${year.toString().padStart(4, "0")}-${month.toString().padStart(2, "0")}-${day
        .toString()
        .padStart(2, "0")}`
    }
  }

  // ISO-ish
  const iso = s.match(/^(\d{4})-(\d{2})-(\d{2})/)
  if (iso) return `${iso[1]}-${iso[2]}-${iso[3]}`

  const t = Date.parse(s)
  if (!Number.isNaN(t)) return toIsoDateUTC(new Date(t))
  return null
}

function toIsoDateUTC(d: Date): string {
  // Prefer local Y-M-D components when Date came from SheetJS cellDates
  const y = d.getFullYear()
  const m = d.getMonth() + 1
  const day = d.getDate()
  // If UTC and local disagree wildly, fall back to UTC (Excel serials)
  if (Math.abs(d.getTimezoneOffset()) > 0 && d.getUTCHours() === 0 && d.getHours() !== 0) {
    return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}-${String(
      d.getUTCDate(),
    ).padStart(2, "0")}`
  }
  return `${y.toString().padStart(4, "0")}-${m.toString().padStart(2, "0")}-${day
    .toString()
    .padStart(2, "0")}`
}

function inferAccountFromFilename(filename: string, bankName: string | null): string {
  const base = filename.replace(/\.[^.]+$/, "").toUpperCase()
  if (bankName && base.startsWith(bankName.toUpperCase())) {
    const rest = base.slice(bankName.length)
    const m = rest.match(/^(\d+)/)
    if (m) return m[1]
  }
  const m = base.match(/(\d{3,})/)
  return m ? m[1] : ""
}

function findHeaderRow(grid: unknown[][]): number | null {
  const limit = Math.min(grid.length, 60)
  for (let i = 0; i < limit; i++) {
    const row = grid[i] ?? []
    const joined = row.map((x) => normText(x)).join("|")
    // Keep original casing for Thai keyword checks
    const joinedRaw = row
      .map((x) => (isBlank(x) ? "" : String(x)))
      .join("|")
    let hits = 0
    if (joined.includes("DATE")) hits += 1
    if (joined.includes("DESCRIPTION") || joined.includes("DETAIL") || joined.includes("PARTICULAR")) {
      hits += 1
    }
    if (joined.includes("DEBIT") || joined.includes("WITHDRAW")) hits += 1
    if (joined.includes("CREDIT") || joined.includes("DEPOSIT")) hits += 1
    if (/\bAMOUNT\b/.test(joined)) hits += 1
    if (joined.includes("BAL") || joined.includes("BALANCE")) hits += 1
    if (joinedRaw.includes("วันที่")) hits += 1
    if (joinedRaw.includes("รายการ") || joinedRaw.includes("รายละเอียด")) hits += 1
    if (joinedRaw.includes("เดบิต") || joinedRaw.includes("ถอน")) hits += 1
    if (joinedRaw.includes("เครดิต") || joinedRaw.includes("ฝาก")) hits += 1
    if (joinedRaw.includes("คงเหลือ") || joinedRaw.includes("ยอดคงเหลือ")) hits += 1
    if (hits >= 3) return i
  }
  return null
}

function normCols(cols: unknown[]): string[] {
  const out = cols.map((c) => normText(c))
  const seen: Record<string, number> = {}
  return out.map((c) => {
    if (!(c in seen)) {
      seen[c] = 0
      return c
    }
    seen[c] += 1
    return `${c}_${seen[c]}`
  })
}

function pickCol(cols: string[], patterns: string[]): string | null {
  for (const p of patterns) {
    const rx = new RegExp(p)
    for (const c of cols) {
      if (rx.test(c)) return c
    }
  }
  return null
}

function isAccountMetadataLabel(label: string): boolean {
  if (!label) return false
  if (ACCOUNT_METADATA_LABELS.has(label)) return true
  return ACCOUNT_METADATA_LABEL_PREFIXES.some((prefix) => label.startsWith(prefix))
}

function splitLabelValueCell(cell: unknown): [string, string] {
  if (isBlank(cell)) return ["", ""]
  const s = String(cell).trim()
  for (const sep of [":", "："]) {
    if (s.includes(sep)) {
      const [left, ...rest] = s.split(sep)
      return [normText(left), rest.join(sep).trim()]
    }
  }
  return [normText(s), ""]
}

function extractAccountFromMetadata(grid: unknown[][]): string {
  const limit = Math.min(grid.length, 20)
  for (let i = 0; i < limit; i++) {
    const row = grid[i] ?? []
    for (let j = 0; j < row.length; j++) {
      const cell = row[j]
      const [subLabel, subVal] = splitLabelValueCell(cell)
      if (subVal && isAccountMetadataLabel(subLabel)) return subVal

      const label = normText(cell)
      if (isAccountMetadataLabel(label)) {
        for (let k = j + 1; k < row.length; k++) {
          const val = row[k]
          if (isBlank(val)) continue
          const s = String(val).trim()
          if (s) return s
        }
      }
    }
  }
  return ""
}

function toNumericMoney(val: unknown): number | null {
  if (isBlank(val)) return null
  const s = String(val).replace(/,/g, "").replace(/\u00a0/g, " ").trim()
  if (!s || s.toUpperCase().startsWith("TOTAL")) return null
  const n = Number(s)
  if (!Number.isFinite(n)) return null
  return n
}

function jsonSafeValue(v: unknown): unknown {
  if (v === null || v === undefined) return null
  if (typeof v === "number" && Number.isNaN(v)) return null
  if (v instanceof Date) return v.toISOString()
  if (typeof v === "bigint") return v.toString()
  return v
}

function rowToObject(cols: string[], row: unknown[]): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (let i = 0; i < cols.length; i++) {
    out[String(cols[i] ?? i)] = jsonSafeValue(row[i])
  }
  return out
}

function sheetToGrid(sheet: XLSX.WorkSheet): unknown[][] {
  const ref = sheet["!ref"]
  if (!ref) return []
  return XLSX.utils.sheet_to_json(sheet, {
    header: 1,
    defval: null,
    raw: true,
    blankrows: true,
  }) as unknown[][]
}

async function parseStatementBytes(
  bytes: Uint8Array,
  opts: {
    filename: string
    bankName: string
    accountNo?: string | null
  },
): Promise<ParseResult> {
  const bankName = opts.bankName
  const fallbackAccount = opts.accountNo || inferAccountFromFilename(opts.filename, bankName)

  const wb = XLSX.read(bytes, {
    type: "array",
    cellDates: true,
    raw: true,
  })

  const meta: Record<string, unknown> = {
    sheet_names: wb.SheetNames,
    parser_version: PARSER_VERSION,
    bank_name: bankName,
    source: "edge_upload",
  }

  const lines: ParsedLine[] = []
  let resolvedAccount = fallbackAccount || ""

  for (const sheetName of wb.SheetNames) {
    const sheet = wb.Sheets[sheetName]
    if (!sheet) continue
    const grid = sheetToGrid(sheet)
    if (!grid.length) continue

    const headerRow = findHeaderRow(grid)
    if (headerRow === null) continue

    const metaAccount = extractAccountFromMetadata(grid)
    if (metaAccount) resolvedAccount = metaAccount

    const headerCells = grid[headerRow] ?? []
    const cols = normCols(headerCells)

    const colDate = pickCol(cols, ["^DATE$", "TXN.*DATE", "TRAN.*DATE", "วันที่"])
    const colValueDate = pickCol(cols, ["VALUE.*DATE", "VAL.*DATE", "วันที่.*เงิน"])
    const colDesc = pickCol(cols, ["DESC", "DETAIL", "PARTICULAR", "รายการ", "รายละเอียด"])
    const colDebit = pickCol(cols, ["DEBIT", "WITHDRAW", "DR", "ถอน", "เดบิต"])
    const colCredit = pickCol(cols, ["CREDIT", "DEPOSIT", "CR", "ฝาก", "เครดิต"])
    const colAmount = pickCol(cols, ["^AMOUNT$", "^จำนวนเงิน$"])
    const colBalance = pickCol(cols, ["BAL", "BALANCE", "คงเหลือ", "ยอดคงเหลือ"])
    const colRef = pickCol(cols, ["REF", "REFERENCE", "CHEQUE", "CHQ", "เลขที่", "อ้างอิง", "^CHEQUE NO"])

    if (!colDate || (!colDebit && !colCredit && !colAmount)) continue

    const idx = (name: string | null) => (name ? cols.indexOf(name) : -1)
    const iDate = idx(colDate)
    const iValueDate = idx(colValueDate)
    const iDesc = idx(colDesc)
    const iDebit = idx(colDebit)
    const iCredit = idx(colCredit)
    const iAmount = idx(colAmount)
    const iBalance = idx(colBalance)
    const iRef = idx(colRef)

    // 1-based Excel row of first data row = headerRow+2
    const baseRowNum = headerRow + 2

    for (let r = headerRow + 1; r < grid.length; r++) {
      const row = grid[r] ?? []
      const raw = rowToObject(cols, row)

      const txnDate = parseDayFirstDate(row[iDate])
      if (!txnDate) continue

      const debit = iDebit >= 0 ? toNumericMoney(row[iDebit]) : null
      const credit = iCredit >= 0 ? toNumericMoney(row[iCredit]) : null
      const signedAmount = iAmount >= 0 ? toNumericMoney(row[iAmount]) : null
      const bal = iBalance >= 0 ? toNumericMoney(row[iBalance]) : null

      let direction: "in" | "out" | null = null
      let amount: number | null = null
      let debitVal: number | null = null
      let creditVal: number | null = null

      if (credit !== null && credit !== 0) {
        direction = "in"
        amount = Math.abs(credit)
        creditVal = amount
      } else if (debit !== null && debit !== 0) {
        direction = "out"
        amount = Math.abs(debit)
        debitVal = amount
      } else if (signedAmount !== null && signedAmount !== 0) {
        if (signedAmount > 0) {
          direction = "in"
          amount = Math.abs(signedAmount)
          creditVal = amount
        } else {
          direction = "out"
          amount = Math.abs(signedAmount)
          debitVal = amount
        }
      } else {
        continue
      }

      const descRaw = iDesc >= 0 ? row[iDesc] : null
      const refRaw = iRef >= 0 ? row[iRef] : null
      const description = isBlank(descRaw) ? null : String(descRaw)
      const bankReference = isBlank(refRaw) ? null : String(refRaw)

      let valueDate: string | null = null
      if (iValueDate >= 0) {
        valueDate = parseDayFirstDate(row[iValueDate])
      }

      const stableDetail = extractStableTransactionDetail(raw) || description || ""
      const fp = await buildTransactionFingerprint({
        accountNo: resolvedAccount,
        txnDate,
        amount,
        direction,
        stableDetail,
        bankReference,
        balanceAfter: bal,
        hasBalanceColumn: Boolean(colBalance),
      })

      lines.push({
        account_no: resolvedAccount,
        bank_name: bankName,
        txn_date: txnDate,
        value_date: valueDate,
        description,
        bank_reference: bankReference,
        amount: Number(normMoney(amount)),
        direction,
        debit: debitVal === null ? null : Number(normMoney(debitVal)),
        credit: creditVal === null ? null : Number(normMoney(creditVal)),
        balance_after: bal === null ? null : Number(normMoney(bal)),
        transaction_fingerprint: fp,
        source_sheet_name: sheetName,
        source_row_number: baseRowNum + (r - headerRow - 1),
        raw_json: raw,
      })
    }
  }

  meta.account_no = resolvedAccount
  meta.row_count_detected = lines.length
  return { meta, lines }
}


/**
 * Import a bank statement Excel file (KBANK / KTB) into bank.statement_*.
 *
 * Auth: caller JWT must belong to public.kcw_admin (email match on user_id).
 * Body: multipart/form-data with fields:
 *   - file: .xlsx / .xls / .xlsm
 *   - bank_name: KBANK | KTB (required)
 *
 * Same parse heuristics as notebooks/02_bank_statement_import_test.ipynb
 * (parser_version auto_v2 — stable transaction fingerprints).
 */
const ALLOWED_EXT = new Set([".xlsx", ".xls", ".xlsm"])
const MAX_BYTES = 15 * 1024 * 1024
const BUCKET = "bank-statements"

type ImportStatus = "imported" | "skipped" | "failed"

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  })
}

function extOf(name: string): string {
  const i = name.lastIndexOf(".")
  return i >= 0 ? name.slice(i).toLowerCase() : ""
}

function sanitizeFilename(name: string): string {
  return name.replace(/[^\w.\-()\u0E00-\u0E7F]+/g, "_").slice(0, 180)
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders })
  }

  if (req.method !== "POST") {
    return jsonResponse({ error: "Method not allowed" }, 405)
  }

  try {
    const authHeader = req.headers.get("Authorization")
    if (!authHeader?.startsWith("Bearer ")) {
      return jsonResponse({ error: "Missing Authorization bearer token" }, 401)
    }

    const supabaseUrl = Deno.env.get("SUPABASE_URL") ?? ""
    const anonKey = Deno.env.get("SUPABASE_ANON_KEY") ?? ""
    const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? ""
    if (!supabaseUrl || !anonKey || !serviceKey) {
      return jsonResponse({ error: "Server misconfigured" }, 500)
    }

    const userClient = createClient(supabaseUrl, anonKey, {
      global: { headers: { Authorization: authHeader } },
    })
    const admin = createClient(supabaseUrl, serviceKey)

    const token = authHeader.replace(/^Bearer\s+/i, "")
    const {
      data: { user },
      error: userErr,
    } = await userClient.auth.getUser(token)
    if (userErr || !user) {
      return jsonResponse({ error: "Invalid or expired session" }, 401)
    }

    const email = (user.email ?? "").trim().toLowerCase()
    if (!email) {
      return jsonResponse({ error: "User email required" }, 403)
    }

    const { data: adminRow, error: adminErr } = await admin
      .from("kcw_admin")
      .select("id")
      .ilike("user_id", email)
      .maybeSingle()
    if (adminErr) {
      console.error("kcw_admin lookup failed", adminErr)
      return jsonResponse({ error: "Admin check failed" }, 500)
    }
    if (!adminRow) {
      return jsonResponse({ error: "Forbidden: not a kcw_admin user" }, 403)
    }

    const contentType = req.headers.get("content-type") ?? ""
    if (!contentType.toLowerCase().includes("multipart/form-data")) {
      return jsonResponse(
        {
          error: "Expected multipart/form-data with fields: file, bank_name",
        },
        400,
      )
    }

    const form = await req.formData()
    const bankRaw = String(form.get("bank_name") ?? "").trim().toUpperCase()
    if (bankRaw !== "KBANK" && bankRaw !== "KTB") {
      return jsonResponse({ error: "bank_name must be KBANK or KTB" }, 400)
    }
    const bankName = bankRaw as "KBANK" | "KTB"

    const fileEntry = form.get("file")
    if (!(fileEntry instanceof File)) {
      return jsonResponse({ error: "Missing file field" }, 400)
    }

    const originalFilename = fileEntry.name || "statement.xlsx"
    const ext = extOf(originalFilename)
    if (!ALLOWED_EXT.has(ext)) {
      return jsonResponse(
        { error: `Unsupported file type ${ext || "(none)"}; use .xlsx, .xls, or .xlsm` },
        400,
      )
    }
    if (fileEntry.size <= 0 || fileEntry.size > MAX_BYTES) {
      return jsonResponse(
        { error: `File size must be between 1 byte and ${MAX_BYTES} bytes` },
        400,
      )
    }

    const bytes = new Uint8Array(await fileEntry.arrayBuffer())
    const fileHash = await sha256HexAsync(bytes)
    const accountGuess = inferAccountFromFilename(originalFilename, bankName)

    const { meta, lines } = await parseStatementBytes(bytes, {
      filename: originalFilename,
      bankName,
      accountNo: accountGuess,
    })
    const resolvedAccount = String(meta.account_no ?? accountGuess ?? "") || null

    // Persist raw file for audit / reprocess (best-effort; import still proceeds if storage fails)
    const now = new Date()
    const yyyy = now.getUTCFullYear()
    const mm = String(now.getUTCMonth() + 1).padStart(2, "0")
    const storagePath = `${bankName}/${yyyy}/${mm}/${fileHash.slice(0, 16)}_${sanitizeFilename(originalFilename)}`
    let storageError: string | null = null
    {
      const { error: upErr } = await admin.storage.from(BUCKET).upload(storagePath, bytes, {
        contentType: fileEntry.type ||
          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        upsert: true,
      })
      if (upErr) {
        storageError = upErr.message
        console.error("storage upload failed", upErr)
      }
    }
    const sourcePath = storageError
      ? `upload://${originalFilename}`
      : `storage://${BUCKET}/${storagePath}`

    const bank = admin.schema("bank")

    // Upsert import file by file_hash (same semantics as Python upsert_import_file)
    const { data: existing, error: existErr } = await bank
      .from("statement_import_files")
      .select("id, status, row_count, inserted_count, duplicate_count")
      .eq("file_hash", fileHash)
      .maybeSingle()
    if (existErr) throw existErr

    let fileId: string
    let isNewFile = false

    if (existing?.id) {
      fileId = existing.id
      const { error: bumpErr } = await bank
        .from("statement_import_files")
        .update({
          last_seen_at: new Date().toISOString(),
          source_path: sourcePath,
          bank_name: bankName,
          account_no: resolvedAccount,
          original_filename: originalFilename,
        })
        .eq("id", fileId)
      if (bumpErr) throw bumpErr

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
        })
      }

      if (
        existing.status === "duplicate" &&
        (existing.row_count ?? 0) > 0
      ) {
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
        })
      }
    } else {
      isNewFile = true
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
        .single()
      if (insErr) throw insErr
      fileId = inserted.id
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
      })

      const { inserted_count, duplicate_count } = await insertStatementLines(
        bank,
        fileId,
        lines,
      )

      await setFileStatus(bank, fileId, {
        status: "imported",
        row_count: lines.length,
        inserted_count,
        duplicate_count,
        error_count: 0,
        error_message: null,
      })

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
      })
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      await setFileStatus(bank, fileId, {
        status: "failed",
        row_count: lines.length,
        inserted_count: 0,
        duplicate_count: 0,
        error_count: 1,
        error_message: msg,
      })
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
      )
    }
  } catch (e) {
    console.error(e)
    const msg = e instanceof Error ? e.message : String(e)
    return jsonResponse({ error: msg }, 500)
  }
})

// deno-lint-ignore no-explicit-any
async function setFileStatus(
  bank: any,
  fileId: string,
  fields: {
    status: string
    row_count: number
    inserted_count: number
    duplicate_count: number
    error_count: number
    error_message: string | null
    raw_metadata?: Record<string, unknown>
    account_no?: string | null
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
  }
  if (fields.raw_metadata) payload.raw_metadata = fields.raw_metadata
  if (fields.account_no !== undefined) payload.account_no = fields.account_no

  const { error } = await bank.from("statement_import_files").update(payload).eq("id", fileId)
  if (error) throw error
}

// deno-lint-ignore no-explicit-any
async function insertStatementLines(
  bank: any,
  fileId: string,
  lines: ParsedLine[],
): Promise<{ inserted_count: number; duplicate_count: number }> {
  if (!lines.length) return { inserted_count: 0, duplicate_count: 0 }

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
  }))

  let inserted = 0
  const pageSize = 500
  for (let i = 0; i < rows.length; i += pageSize) {
    const chunk = rows.slice(i, i + pageSize)
    const { data, error } = await bank
      .from("statement_lines")
      .upsert(chunk, {
        onConflict: "transaction_fingerprint",
        ignoreDuplicates: true,
      })
      .select("transaction_fingerprint")
    if (error) throw error
    inserted += data?.length ?? 0
  }

  return {
    inserted_count: inserted,
    duplicate_count: rows.length - inserted,
  }
}

