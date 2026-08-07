/**
 * Fingerprint unit tests (Deno). Synced with kcw-v2 fingerprint.test.ts
 * including the 3557 overlapping monthly/cumulative regression.
 */
import { assertEquals } from "jsr:@std/assert@1";
import {
  buildTransactionFingerprint,
  extractTransactionDetailFromRaw,
  sha256HexAsync,
} from "./fingerprint.ts";

const BASE = {
  account_no: "064-8-92039-3",
  txn_date: "2026-05-04",
  direction: "in" as const,
  amount: 50000,
  balance_after: 69237.34,
  bank_reference: null as string | null,
  transaction_detail: "จาก X3557 บจก. เกียรติชัยอะไ++",
};

Deno.test("Test 1 — same identity when display description formatting differs", async () => {
  const fpA = await buildTransactionFingerprint(BASE);
  const fpB = await buildTransactionFingerprint({
    ...BASE,
    transaction_detail: BASE.transaction_detail,
  });
  assertEquals(fpA, fpB);
  assertEquals(fpA.length, 64);

  const fromTime = extractTransactionDetailFromRaw({
    "รายการ": "09:12:00",
    "รายละเอียด": BASE.transaction_detail!,
  });
  const fromLabel = extractTransactionDetailFromRaw({
    "รายการ": "รับโอนเงิน",
    "รายละเอียด": BASE.transaction_detail!,
  });
  assertEquals(fromTime, fromLabel);
  assertEquals(
    await buildTransactionFingerprint({ ...BASE, transaction_detail: fromTime }),
    await buildTransactionFingerprint({ ...BASE, transaction_detail: fromLabel }),
  );
});

Deno.test("Test 2 — different fingerprints for legitimate repeated same-amount transactions", async () => {
  const first = await buildTransactionFingerprint({
    account_no: "064-8-92039-3",
    txn_date: "2026-05-25",
    direction: "in",
    amount: 1000,
    balance_after: 100500,
    bank_reference: null,
    transaction_detail: "จาก X3557 บจก. เกียรติชัยอะไ++",
  });
  const second = await buildTransactionFingerprint({
    account_no: "064-8-92039-3",
    txn_date: "2026-05-25",
    direction: "in",
    amount: 1000,
    balance_after: 101500,
    bank_reference: null,
    transaction_detail: "จาก KTB X8740 MISS NARUMON WITHA++",
  });
  assertEquals(first === second, false);
});

Deno.test("Test 3 — overlapping cumulative statement rows share fingerprints", async () => {
  const earlierImportLine = {
    account_no: "064-8-92039-3",
    txn_date: "2026-05-12",
    direction: "out" as const,
    amount: 55,
    balance_after: 150934.41,
    bank_reference: null as string | null,
    transaction_detail: "โอนไป KTB X2446 น.ส.นฤมล วิทยผโลท++",
  };
  const cumulativeImportLine = { ...earlierImportLine };
  const fpEarlier = await buildTransactionFingerprint(earlierImportLine);
  const fpCumulative = await buildTransactionFingerprint(cumulativeImportLine);
  assertEquals(fpEarlier, fpCumulative);
});

Deno.test("Test 4 — file_hash (SHA-256 of file bytes) is stable for exact re-upload", async () => {
  const bytes = new TextEncoder().encode("same-statement-bytes");
  const hash1 = await sha256HexAsync(bytes);
  const hash2 = await sha256HexAsync(bytes);
  assertEquals(hash1, hash2);
  assertEquals(
    hash1 === (await sha256HexAsync(new TextEncoder().encode("other-bytes"))),
    false,
  );
});

Deno.test("Test 5 — 3557 monthly re-import matches earlier April file on stable detail", async () => {
  // 04_3557.xlsx vs 3557 ด.4.xlsx: same bank txn; display รายการ may be
  // "โอนเงิน" in one export and a time string in another.
  const fromAprilFile = await buildTransactionFingerprint({
    account_no: "141-1-72355-7",
    txn_date: "2026-04-01",
    direction: "out",
    amount: 3866,
    balance_after: 130186.73,
    bank_reference: null,
    transaction_detail: "โอนไป SCB X7654 บริษัท  คูโบต้า ก.++",
  });
  const fromMonth4Reupload = await buildTransactionFingerprint({
    account_no: "141-1-72355-7",
    txn_date: "2026-04-01",
    direction: "out",
    amount: 3866,
    balance_after: 130186.73,
    bank_reference: null,
    transaction_detail: "โอนไป SCB X7654 บริษัท  คูโบต้า ก.++",
  });
  assertEquals(fromAprilFile, fromMonth4Reupload);

  // auto_v1 wrongly hashed description ("โอนเงิน" vs "08:31:00") → miss
  const v1LikeDifferentDescriptions = [
    "141-1-72355-7|2026-04-01|3866.00|OUT|โอนเงิน||130186.73",
    "141-1-72355-7|2026-04-01|3866.00|OUT|08:31:00||130186.73",
  ];
  const v1Hashes = await Promise.all(
    v1LikeDifferentDescriptions.map((s) => sha256HexAsync(s)),
  );
  assertEquals(v1Hashes[0] === v1Hashes[1], false);
});
