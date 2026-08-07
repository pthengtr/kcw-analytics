/**
 * Port of notebooks/02_bank_statement_import_test.ipynb + src/kcw/bank_statement.py
 * parser_version: auto_v2 (stable transaction fingerprints across overlapping exports)
 */
import * as XLSX from "npm:xlsx@0.18.5"

export const PARSER_VERSION = "auto_v2"

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

export type ParsedLine = {
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

export type ParseResult = {
  meta: Record<string, unknown>
  lines: ParsedLine[]
}

function isBlank(v: unknown): boolean {
  if (v === null || v === undefined) return true
  if (typeof v === "number" && Number.isNaN(v)) return true
  if (typeof v === "string" && v.trim() === "") return true
  return false
}

export async function sha256HexAsync(data: ArrayBuffer | Uint8Array | string): Promise<string> {
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

export function normText(x: unknown): string {
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
export function extractStableTransactionDetail(
  raw: Record<string, unknown>,
): string {
  for (const key of STABLE_DETAIL_KEYS) {
    const val = raw[key]
    if (!isBlank(val)) return String(val).trim()
  }
  return ""
}

export type TransactionFingerprintInput = {
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
export async function buildTransactionFingerprint(
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

export function normMoney(x: unknown): string {
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

export function parseDayFirstDate(value: unknown): string | null {
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

export function inferAccountFromFilename(filename: string, bankName: string | null): string {
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

export async function parseStatementBytes(
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
