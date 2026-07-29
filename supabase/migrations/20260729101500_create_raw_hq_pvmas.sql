-- HQ PVMAS payment / notes vouchers (PARTS9), matching Drive raw CSV:
--   raw_hq_pvmas_notes_vouchers.csv
-- Twin of RVMAS; includes P* payment vouchers (VOUCNO) and KCPN* series.

create schema if not exists raw_kcw;

create table if not exists raw_kcw.raw_hq_pvmas_notes_vouchers (
    _ingested_at timestamptz not null default now(),
    _source_file text,
    "ID" text,
    "JOURMODE" text,
    "JOURTYPE" text,
    "JOURDATE" text,
    "JOURNO" text,
    "DEPTNO" text,
    "BOOKNO" text,
    "VOUCED" text,
    "VOUCDATE" text,
    "VOUCNO" text,
    "NOTED" text,
    "NOTEDATE" text,
    "NOTENO" text,
    "RCPTNO" text,
    "RCPTDATE" text,
    "ACCTNO" text,
    "ACCTNAME" text,
    "ADDR1" text,
    "ADDR2" text,
    "BILLCNT" text,
    "BILLAMT" text,
    "TAX" text,
    "DISCOUNT" text,
    "NETAMT" text,
    "CASHAMT" text,
    "CHKAMT" text,
    "PAYAMT" text,
    "PAID" text,
    "POSTED1" text,
    "POSTED2" text,
    "CANCELED" text,
    "DONE" text
);

create table if not exists raw_kcw.raw_hq_pvmas_notes_vouchers_stg
    (like raw_kcw.raw_hq_pvmas_notes_vouchers including all);

create index if not exists raw_hq_pvmas_voucno_idx
    on raw_kcw.raw_hq_pvmas_notes_vouchers ("VOUCNO");
create index if not exists raw_hq_pvmas_rcptno_idx
    on raw_kcw.raw_hq_pvmas_notes_vouchers ("RCPTNO");
create index if not exists raw_hq_pvmas_voucdate_idx
    on raw_kcw.raw_hq_pvmas_notes_vouchers ("VOUCDATE");
create index if not exists raw_hq_pvmas_acctno_idx
    on raw_kcw.raw_hq_pvmas_notes_vouchers ("ACCTNO");

grant select on raw_kcw.raw_hq_pvmas_notes_vouchers to anon, authenticated, service_role;
grant select on raw_kcw.raw_hq_pvmas_notes_vouchers_stg to anon, authenticated, service_role;
