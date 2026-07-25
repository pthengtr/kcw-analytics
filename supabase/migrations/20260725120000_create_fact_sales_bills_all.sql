-- Bill-header curated fact (SIMAS), matching Drive fact_sales_bills_all.csv
-- Column shape mirrors curated_kcw.fact_sales_all: metadata + text payload columns.

create table if not exists curated_kcw.fact_sales_bills_all (
    _ingested_at timestamptz not null default now(),
    _source_file text,
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
    "DONE" text,
    "BRANCH" text,
    "BILLTYPE_STD" text
);

create table if not exists curated_kcw.fact_sales_bills_all_stg
    (like curated_kcw.fact_sales_bills_all including all);

create index if not exists fact_sales_bills_all_billdate_idx
    on curated_kcw.fact_sales_bills_all ("BILLDATE");

create index if not exists fact_sales_bills_all_branch_billno_idx
    on curated_kcw.fact_sales_bills_all ("BRANCH", "BILLNO");

-- Match grant_raw_curated_kcw_select for sibling curated facts.
grant select on curated_kcw.fact_sales_bills_all to anon, authenticated, service_role;
grant select on curated_kcw.fact_sales_bills_all_stg to anon, authenticated, service_role;
