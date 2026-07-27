-- HQ ARMAS / APMAS account masters (PARTS9), matching Drive raw CSVs:
--   raw_hq_armas_receivable.csv
--   raw_hq_apmas_payable.csv
-- Column shape mirrors other raw_kcw tables: metadata + text payload columns.

create schema if not exists raw_kcw;

create table if not exists raw_kcw.raw_hq_armas_receivable (
    _ingested_at timestamptz not null default now(),
    _source_file text,
    "ID" text,
    "JOURMODE" text,
    "ACCTTYPE" text,
    "ACCTNO" text,
    "ACCTNAME" text,
    "ADDR1" text,
    "ADDR2" text,
    "PHONE" text,
    "MOBILE" text,
    "FAX" text,
    "CONTACT" text,
    "EMAIL" text,
    "TERM" text,
    "ALLOW" text,
    "ATPRICE" text,
    "MARKUP" text,
    "BEGDATE" text,
    "ENDDATE" text,
    "REMARKS" text,
    "CANCELED" text
);

create table if not exists raw_kcw.raw_hq_armas_receivable_stg
    (like raw_kcw.raw_hq_armas_receivable including all);

create table if not exists raw_kcw.raw_hq_apmas_payable (
    _ingested_at timestamptz not null default now(),
    _source_file text,
    "ID" text,
    "JOURMODE" text,
    "ACCTTYPE" text,
    "ACCTNO" text,
    "ACCTNAME" text,
    "ADDR1" text,
    "ADDR2" text,
    "PHONE" text,
    "MOBILE" text,
    "FAX" text,
    "CONTACT" text,
    "EMAIL" text,
    "TERM" text,
    "ALLOW" text,
    "ATPRICE" text,
    "MARKUP" text,
    "BEGDATE" text,
    "ENDDATE" text,
    "REMARKS" text,
    "CANCELED" text
);

create table if not exists raw_kcw.raw_hq_apmas_payable_stg
    (like raw_kcw.raw_hq_apmas_payable including all);

create index if not exists raw_hq_armas_acctno_idx
    on raw_kcw.raw_hq_armas_receivable ("ACCTNO");

create index if not exists raw_hq_apmas_acctno_idx
    on raw_kcw.raw_hq_apmas_payable ("ACCTNO");

grant select on raw_kcw.raw_hq_armas_receivable to anon, authenticated, service_role;
grant select on raw_kcw.raw_hq_armas_receivable_stg to anon, authenticated, service_role;
grant select on raw_kcw.raw_hq_apmas_payable to anon, authenticated, service_role;
grant select on raw_kcw.raw_hq_apmas_payable_stg to anon, authenticated, service_role;
