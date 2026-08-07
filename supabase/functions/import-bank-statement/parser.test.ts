import {
  assertEquals,
} from "jsr:@std/assert@1"
import {
  buildTransactionFingerprint,
  extractStableTransactionDetail,
  sha256HexAsync,
} from "./parser.ts"

const baseTxn = {
  accountNo: "064-8-92039-3",
  txnDate: "2026-05-01",
  amount: 33718.5,
  direction: "in" as const,
  bankReference: null as string | null,
  balanceAfter: 139636.74,
}

Deno.test("Test 1 — description formatting changes produce same fingerprint", async () => {
  const stableDetail = "จาก KTB X8740 MISS NARUMON WITHA++"
  const fpA = await buildTransactionFingerprint({
    ...baseTxn,
    stableDetail,
  })
  const fpB = await buildTransactionFingerprint({
    ...baseTxn,
    stableDetail,
  })
  assertEquals(fpA, fpB)

  // Display descriptions differ; identity uses stable detail only.
  const fromTimeDesc = await buildTransactionFingerprint({
    ...baseTxn,
    stableDetail: extractStableTransactionDetail({
      "รายการ": "09:12:00",
      "รายละเอียด": stableDetail,
    }),
  })
  const fromLabelDesc = await buildTransactionFingerprint({
    ...baseTxn,
    stableDetail: extractStableTransactionDetail({
      "รายการ": "รับโอนเงิน",
      "รายละเอียด": stableDetail,
    }),
  })
  assertEquals(fromTimeDesc, fromLabelDesc)
})

Deno.test("Test 2 — legitimate repeated amount yields different fingerprints", async () => {
  const fp1 = await buildTransactionFingerprint({
    ...baseTxn,
    stableDetail: "จาก KTB X8740 MISS NARUMON WITHA++",
    balanceAfter: 139636.74,
  })
  const fp2 = await buildTransactionFingerprint({
    ...baseTxn,
    stableDetail: "จาก KTB X2446 NARUMON WITHAYAPAL++",
    balanceAfter: 193267.54,
  })
  const fp3 = await buildTransactionFingerprint({
    ...baseTxn,
    stableDetail: "จาก KTB X8740 MISS NARUMON WITHA++",
    balanceAfter: 200000.0,
    bankReference: "CHQ-123",
  })
  assertEquals(fp1 === fp2, false)
  assertEquals(fp1 === fp3, false)
})

Deno.test("Test 3 — overlapping statement rows share fingerprint", async () => {
  const stableDetail = "โอนไป KTB X2446 น.ส.นฤมล วิทยผโลท++"
  const earlier = await buildTransactionFingerprint({
    accountNo: "064-8-92039-3",
    txnDate: "2026-05-01",
    amount: 1000,
    direction: "out",
    stableDetail,
    bankReference: null,
    balanceAfter: 105918.24,
  })
  const cumulative = await buildTransactionFingerprint({
    accountNo: "064-8-92039-3",
    txnDate: "2026-05-01",
    amount: 1000,
    direction: "out",
    stableDetail,
    bankReference: null,
    balanceAfter: 105918.24,
  })
  assertEquals(earlier, cumulative)
})

Deno.test("Test 4 — file_hash is independent of fingerprint (sha256 of bytes)", async () => {
  const fileA = new TextEncoder().encode("same-file-bytes")
  const fileB = new TextEncoder().encode("same-file-bytes")
  const fileC = new TextEncoder().encode("different-file-bytes")
  const hashA = await sha256HexAsync(fileA)
  const hashB = await sha256HexAsync(fileB)
  const hashC = await sha256HexAsync(fileC)
  assertEquals(hashA, hashB)
  assertEquals(hashA === hashC, false)
})
