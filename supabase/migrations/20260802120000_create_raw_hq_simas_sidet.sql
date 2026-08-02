-- HQ sales raw uploads from Drive CSVs (PARTS9):
--   raw_hq_sidet_sales_lines.csv  (sales invoice lines)
--   raw_hq_simas_sales_bills.csv  (sales bill headers)
-- Daily pipeline uploads the latest 6 months from max BILLDATE only (HQ-only).

create schema if not exists raw_kcw;

-- ---------------------------------------------------------------------------
-- HQ SIDET (sales invoice lines) — 38 PARTS9 columns
-- ---------------------------------------------------------------------------
create table if not exists raw_kcw.raw_hq_sidet_sales_lines (
    _ingested_at timestamptz not null default now(),
    _source_file text,
    "ID" text,
    "JOURMODE" text,
    "JOURTYPE" text,
    "JOURDATE" text,
    "BILLTYPE" text,
    "BILLDATE" text,
    "BILLNO" text,
    "LINE" text,
    "ITEMNO" text,
    "BCODE" text,
    "PCODE" text,
    "MCODE" text,
    "DETAIL" text,
    "WHNUMBER" text,
    "LOCATION1" text,
    "STATUS" text,
    "SERIAL" text,
    "TAXIC" text,
    "EXMPT" text,
    "ISVAT" text,
    "QTY" text,
    "UI" text,
    "MTP" text,
    "PRICE" text,
    "XPRICE" text,
    "DISCNT1" text,
    "DISCNT2" text,
    "DISCNT3" text,
    "DISCNT4" text,
    "DED" text,
    "VAT" text,
    "AMOUNT" text,
    "CHGAMT" text,
    "ACCTNO" text,
    "PAID" text,
    "ACCT_NO" text,
    "CANCELED" text,
    "DONE" text
);

create table if not exists raw_kcw.raw_hq_sidet_sales_lines_stg
    (like raw_kcw.raw_hq_sidet_sales_lines including all);

-- ---------------------------------------------------------------------------
-- HQ SIMAS (sales bill headers) — same shape as PIMAS
-- ---------------------------------------------------------------------------
create table if not exists raw_kcw.raw_hq_simas_sales_bills (
    _ingested_at timestamptz not null default now(),
    _source_file text,
    "ID" text,
    "JOURMODE" text,
    "JOURTYPE" text,
    "JOURDATE" text,
    "JOURNO" text,
    "JOURTIME" text,
    "DEPTNO" text,
    "BOOKNO" text,
    "BILLTYPE" text,
    "BILLDATE" text,
    "BILLTIME" text,
    "BILLNO" text,
    "LINES" text,
    "TAXIC" text,
    "DISCOUNT" text,
    "DEDUCT" text,
    "BEFORETAX" text,
    "VAT" text,
    "TAX" text,
    "AFTERTAX" text,
    "EXEMPT" text,
    "SVCCHG" text,
    "WITHHOLD" text,
    "PAID" text,
    "CASHED" text,
    "CASHAMT" text,
    "CHKAMT" text,
    "DUEAMT" text,
    "PAYSTAT" text,
    "ACCTNO" text,
    "ACCTNAME" text,
    "ADDR1" text,
    "ADDR2" text,
    "PO" text,
    "SALE" text,
    "RE" text,
    "TERM" text,
    "DUEDATE" text,
    "NOTEDATE" text,
    "NOTENO" text,
    "VOUCDATE1" text,
    "VOUCNO1" text,
    "VOUCDATE2" text,
    "VOUCNO2" text,
    "POSTED1" text,
    "POSTED2" text,
    "REMARKS" text,
    "CANCELED" text,
    "DONE" text
);

create table if not exists raw_kcw.raw_hq_simas_sales_bills_stg
    (like raw_kcw.raw_hq_simas_sales_bills including all);

create index if not exists raw_hq_sidet_billdate_idx
    on raw_kcw.raw_hq_sidet_sales_lines ("BILLDATE");
create index if not exists raw_hq_sidet_billno_idx
    on raw_kcw.raw_hq_sidet_sales_lines ("BILLNO");
create index if not exists raw_hq_sidet_bcode_idx
    on raw_kcw.raw_hq_sidet_sales_lines ("BCODE");

create index if not exists raw_hq_simas_billdate_idx
    on raw_kcw.raw_hq_simas_sales_bills ("BILLDATE");
create index if not exists raw_hq_simas_billno_idx
    on raw_kcw.raw_hq_simas_sales_bills ("BILLNO");
create index if not exists raw_hq_simas_po_idx
    on raw_kcw.raw_hq_simas_sales_bills ("PO");

grant select on raw_kcw.raw_hq_sidet_sales_lines to anon, authenticated, service_role;
grant select on raw_kcw.raw_hq_sidet_sales_lines_stg to anon, authenticated, service_role;
grant select on raw_kcw.raw_hq_simas_sales_bills to anon, authenticated, service_role;
grant select on raw_kcw.raw_hq_simas_sales_bills_stg to anon, authenticated, service_role;
