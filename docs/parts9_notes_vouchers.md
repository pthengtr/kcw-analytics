# PARTS9: Notes / payment vouchers (`PVMAS` / `RVMAS`)

How PARTS9 stores **โน้ตจ่าย** and **ใบสำคัญจ่าย** (and the receipt twin).

**Canonical dictionary:** [kcw-pvmas-rvmas-notes-vouchers-data-dictionary.md](https://github.com/pthengtr/kcw-docs/blob/main/dictionaries/kcw-pvmas-rvmas-notes-vouchers-data-dictionary.md) (shared for api / v2 / analytics). This file is extract pointers only.

## Finding

There is **no** note table and **no** `PVDET`. Notes and vouchers are stages of `dbo.PVMAS` (pay) / `dbo.RVMAS` (receive). Related purchase bills are on `PIMAS` via `NOTENO` then `VOUCNO2`.

## Extract

| Table | Drive CSV | Supabase |
|-------|-----------|----------|
| `PVMAS` | `raw_hq_pvmas_notes_vouchers.csv` | `raw_kcw.raw_hq_pvmas_notes_vouchers` |
| `RVMAS` | `raw_hq_rvmas_notes_vouchers.csv` | `raw_kcw.raw_hq_rvmas_notes_vouchers` |

HQ full extract only (`TABLE_SPECS`). Not in SYP minimal.

## Explorer

PARTS9 explorer searches `VOUCNO` **and** `NOTENO`, then loads related `PIMAS` bills (up to 80).
