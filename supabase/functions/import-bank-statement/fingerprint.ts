/**
 * Canonical bank transaction identity for duplicate detection across overlapping
 * statement files (KBANK / KTB). Display-oriented fields such as parsed `description`
 * may change between export formats; identity must not depend on them.
 */

export type TransactionFingerprintInput = {
  account_no: string;
  txn_date: string;
  direction: "in" | "out";
  amount: number;
  balance_after: number | null;
  bank_reference: string | null;
  /** Stable bank detail text (e.g. KBANK รายละเอียด, KTB DESCRIPTION), not display label/time. */
  transaction_detail: string | null;
};

function isBlank(v: unknown): boolean {
  if (v === null || v === undefined) return true;
  if (typeof v === "number" && Number.isNaN(v)) return true;
  if (typeof v === "string" && v.trim() === "") return true;
  return false;
}

export function normText(x: unknown): string {
  if (isBlank(x)) return "";
  let s = String(x);
  s = s.replace(/\u00A0/g, " ");
  s = s.trim().toUpperCase();
  s = s.replace(/\s+/g, " ");
  return s;
}

export function normMoney(x: unknown): string {
  if (isBlank(x)) return "";
  const cleaned = String(x).replace(/,/g, "").trim();
  if (!cleaned) return "";
  const n = Number(cleaned);
  if (!Number.isFinite(n)) return "";
  const sign = n < 0 ? -1 : 1;
  const abs = Math.abs(n);
  const scaled = abs * 100;
  const whole = Math.floor(scaled + 1e-9);
  const frac = scaled - whole;
  let cents = frac >= 0.5 - 1e-12 ? whole + 1 : whole;
  if (Math.abs(frac - 0.5) < 1e-9) cents = whole + 1;
  const out = (sign * cents) / 100;
  return out.toFixed(2);
}

export async function sha256HexAsync(
  data: ArrayBuffer | Uint8Array | string,
): Promise<string> {
  let bytes: Uint8Array;
  if (typeof data === "string") {
    bytes = new TextEncoder().encode(data);
  } else if (data instanceof Uint8Array) {
    bytes = data;
  } else {
    bytes = new Uint8Array(data);
  }
  const hash = await crypto.subtle.digest("SHA-256", bytes as BufferSource);
  return Array.from(new Uint8Array(hash))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/** Column header patterns for stable transaction detail (not display label / time). */
export const TRANSACTION_DETAIL_COL_PATTERNS = [
  "รายละเอียด",
  "^DESCRIPTION$",
  "PARTICULAR",
  "NARRATION",
];

const RAW_DETAIL_KEY_PATTERNS = [
  "รายละเอียด",
  "DESCRIPTION",
  "PARTICULAR",
  "NARRATION",
];

/**
 * Build a SHA-256 fingerprint from banking fields that identify one real transaction.
 *
 * Identity components (pipe-separated, then hashed):
 *   account_no | txn_date | amount | direction | transaction_detail | bank_reference | balance_after
 *
 * `description` (parsed display label / time) is intentionally excluded so overlapping
 * KBANK exports that swap รายการ vs เวลา do not create duplicate rows.
 *
 * `balance_after` disambiguates legitimate same-day same-amount sequences.
 * `transaction_detail` and `bank_reference` add stability when present.
 */
export async function buildTransactionFingerprint(
  input: TransactionFingerprintInput,
): Promise<string> {
  const fpInput = [
    normText(input.account_no),
    input.txn_date,
    normMoney(input.amount),
    normText(input.direction),
    normText(input.transaction_detail),
    normText(input.bank_reference),
    input.balance_after === null || input.balance_after === undefined
      ? ""
      : normMoney(input.balance_after),
  ].join("|");
  return sha256HexAsync(fpInput);
}

export function extractTransactionDetailFromRaw(
  raw: Record<string, unknown>,
): string | null {
  for (const [key, val] of Object.entries(raw)) {
    const normalizedKey = normText(key);
    if (
      RAW_DETAIL_KEY_PATTERNS.some(
        (pattern) => normalizedKey === normText(pattern),
      )
    ) {
      if (!isBlank(val)) return String(val);
    }
  }
  return null;
}
