-- Daily raw uploads from Drive CSVs (PARTS9):
--   HQ + SYP purchase orders: POMAS / PODET
--   HQ purchase invoices:     PIMAS / PIDET (PIDET table may already exist)
-- Column shape mirrors other raw_kcw tables: metadata + text payload columns.

create schema if not exists raw_kcw;

-- ---------------------------------------------------------------------------
-- HQ POMAS (purchase order headers)
-- ---------------------------------------------------------------------------
create table if not exists raw_kcw.raw_hq_pomas_purchase_orders (
    _ingested_at timestamptz not null default now(),
    _source_file text,
    "ID" text,
    "JOURMODE" text,
    "DOCDATE" text,
    "DOCNO" text,
    "LINES" text,
    "TAXIC" text,
    "DISCOUNT" text,
    "DEDUCT" text,
    "BEFORETAX" text,
    "VAT" text,
    "TAX" text,
    "AFTERTAX" text,
    "EXEMPT" text,
    "ACCTNO" text,
    "ACCTNAME" text,
    "ADDR1" text,
    "ADDR2" text,
    "ATTN" text,
    "SUBJECT" text,
    "PO" text,
    "SALE" text,
    "SANAME" text,
    "SATITLE" text,
    "RE" text,
    "TERM" text,
    "STAND" text,
    "DELIVER" text,
    "COVER" text,
    "REMARKS" text,
    "LANG" text,
    "BILLDATE" text,
    "BILLNO" text,
    "BILLED" text,
    "CANCELED" text,
    "DONE" text
);

create table if not exists raw_kcw.raw_hq_pomas_purchase_orders_stg
    (like raw_kcw.raw_hq_pomas_purchase_orders including all);

-- ---------------------------------------------------------------------------
-- HQ PODET (purchase order lines)
-- ---------------------------------------------------------------------------
create table if not exists raw_kcw.raw_hq_podet_purchase_order_lines (
    _ingested_at timestamptz not null default now(),
    _source_file text,
    "ID" text,
    "JOURMODE" text,
    "DOCDATE" text,
    "DOCNO" text,
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
    "AMOUNT" text,
    "ACCT_NO" text,
    "CANCELED" text,
    "DONE" text
);

create table if not exists raw_kcw.raw_hq_podet_purchase_order_lines_stg
    (like raw_kcw.raw_hq_podet_purchase_order_lines including all);

-- ---------------------------------------------------------------------------
-- SYP POMAS / PODET (same shape as HQ)
-- ---------------------------------------------------------------------------
create table if not exists raw_kcw.raw_syp_pomas_purchase_orders
    (like raw_kcw.raw_hq_pomas_purchase_orders including all);

create table if not exists raw_kcw.raw_syp_pomas_purchase_orders_stg
    (like raw_kcw.raw_hq_pomas_purchase_orders including all);

create table if not exists raw_kcw.raw_syp_podet_purchase_order_lines
    (like raw_kcw.raw_hq_podet_purchase_order_lines including all);

create table if not exists raw_kcw.raw_syp_podet_purchase_order_lines_stg
    (like raw_kcw.raw_hq_podet_purchase_order_lines including all);

-- ---------------------------------------------------------------------------
-- HQ PIMAS (purchase bill headers) — invoices only at HQ
-- ---------------------------------------------------------------------------
create table if not exists raw_kcw.raw_hq_pimas_purchase_bills (
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

create table if not exists raw_kcw.raw_hq_pimas_purchase_bills_stg
    (like raw_kcw.raw_hq_pimas_purchase_bills including all);

-- ---------------------------------------------------------------------------
-- HQ PIDET (purchase bill lines) — may already exist from notebook loader
-- ---------------------------------------------------------------------------
create table if not exists raw_kcw.raw_hq_pidet_purchase_lines (
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
    "SALEDATE" text,
    "SALENO" text,
    "SALEPRICE" text,
    "ACCT_NO" text,
    "CANCELED" text,
    "DONE" text
);

create table if not exists raw_kcw.raw_hq_pidet_purchase_lines_stg
    (like raw_kcw.raw_hq_pidet_purchase_lines including all);

-- Indexes
create index if not exists raw_hq_pomas_docno_idx
    on raw_kcw.raw_hq_pomas_purchase_orders ("DOCNO");
create index if not exists raw_hq_pomas_docdate_idx
    on raw_kcw.raw_hq_pomas_purchase_orders ("DOCDATE");
create index if not exists raw_hq_podet_docno_idx
    on raw_kcw.raw_hq_podet_purchase_order_lines ("DOCNO");

create index if not exists raw_syp_pomas_docno_idx
    on raw_kcw.raw_syp_pomas_purchase_orders ("DOCNO");
create index if not exists raw_syp_pomas_docdate_idx
    on raw_kcw.raw_syp_pomas_purchase_orders ("DOCDATE");
create index if not exists raw_syp_podet_docno_idx
    on raw_kcw.raw_syp_podet_purchase_order_lines ("DOCNO");

create index if not exists raw_hq_pimas_billno_idx
    on raw_kcw.raw_hq_pimas_purchase_bills ("BILLNO");
create index if not exists raw_hq_pimas_po_idx
    on raw_kcw.raw_hq_pimas_purchase_bills ("PO");
create index if not exists raw_hq_pidet_billno_idx
    on raw_kcw.raw_hq_pidet_purchase_lines ("BILLNO");

grant select on raw_kcw.raw_hq_pomas_purchase_orders to anon, authenticated, service_role;
grant select on raw_kcw.raw_hq_pomas_purchase_orders_stg to anon, authenticated, service_role;
grant select on raw_kcw.raw_hq_podet_purchase_order_lines to anon, authenticated, service_role;
grant select on raw_kcw.raw_hq_podet_purchase_order_lines_stg to anon, authenticated, service_role;
grant select on raw_kcw.raw_syp_pomas_purchase_orders to anon, authenticated, service_role;
grant select on raw_kcw.raw_syp_pomas_purchase_orders_stg to anon, authenticated, service_role;
grant select on raw_kcw.raw_syp_podet_purchase_order_lines to anon, authenticated, service_role;
grant select on raw_kcw.raw_syp_podet_purchase_order_lines_stg to anon, authenticated, service_role;
grant select on raw_kcw.raw_hq_pimas_purchase_bills to anon, authenticated, service_role;
grant select on raw_kcw.raw_hq_pimas_purchase_bills_stg to anon, authenticated, service_role;
grant select on raw_kcw.raw_hq_pidet_purchase_lines to anon, authenticated, service_role;
grant select on raw_kcw.raw_hq_pidet_purchase_lines_stg to anon, authenticated, service_role;
