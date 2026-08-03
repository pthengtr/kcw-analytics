-- HQ BRDET / BPDET cheque+transfer registers (PARTS9), matching Drive raw CSVs:
--   raw_hq_brdet_cheques_received.csv
--   raw_hq_bpdet_cheques_paid.csv
-- Source of Excel reports ทะเบียนเช็ครับ / ทะเบียนเช็คจ่าย.
-- CHKNO holds either a cheque number or a method label (โอน, KSHOP, จ่ายสด, …).

create schema if not exists raw_kcw;

create table if not exists raw_kcw.raw_hq_brdet_cheques_received (
    _ingested_at timestamptz not null default now(),
    _source_file text,
    "ID" text,
    "JOURMODE" text,
    "JOURTYPE" text,
    "VOUCDATE" text,
    "VOUCNO" text,
    "ACCTNO" text,
    "PAYTYPE" text,
    "CHKNO" text,
    "CHKDATE" text,
    "CHKAMT" text,
    "BANKNAME" text,
    "CARDNAME" text,
    "DATEIN" text,
    "STATUS" text,
    "CANCELED" text,
    "DONE" text
);

create table if not exists raw_kcw.raw_hq_brdet_cheques_received_stg
    (like raw_kcw.raw_hq_brdet_cheques_received including all);

create index if not exists raw_hq_brdet_voucno_idx
    on raw_kcw.raw_hq_brdet_cheques_received ("VOUCNO");
create index if not exists raw_hq_brdet_chkno_idx
    on raw_kcw.raw_hq_brdet_cheques_received ("CHKNO");
create index if not exists raw_hq_brdet_voucdate_idx
    on raw_kcw.raw_hq_brdet_cheques_received ("VOUCDATE");
create index if not exists raw_hq_brdet_chkdate_idx
    on raw_kcw.raw_hq_brdet_cheques_received ("CHKDATE");
create index if not exists raw_hq_brdet_acctno_idx
    on raw_kcw.raw_hq_brdet_cheques_received ("ACCTNO");

grant select on raw_kcw.raw_hq_brdet_cheques_received to anon, authenticated, service_role;
grant select on raw_kcw.raw_hq_brdet_cheques_received_stg to anon, authenticated, service_role;

create table if not exists raw_kcw.raw_hq_bpdet_cheques_paid (
    _ingested_at timestamptz not null default now(),
    _source_file text,
    "ID" text,
    "JOURMODE" text,
    "JOURTYPE" text,
    "VOUCDATE" text,
    "VOUCNO" text,
    "ACCTNO" text,
    "PAYTYPE" text,
    "CHKNO" text,
    "CHKDATE" text,
    "CHKAMT" text,
    "BANKNAME" text,
    "CARDNAME" text,
    "DATEIN" text,
    "STATUS" text,
    "CANCELED" text,
    "DONE" text
);

create table if not exists raw_kcw.raw_hq_bpdet_cheques_paid_stg
    (like raw_kcw.raw_hq_bpdet_cheques_paid including all);

create index if not exists raw_hq_bpdet_voucno_idx
    on raw_kcw.raw_hq_bpdet_cheques_paid ("VOUCNO");
create index if not exists raw_hq_bpdet_chkno_idx
    on raw_kcw.raw_hq_bpdet_cheques_paid ("CHKNO");
create index if not exists raw_hq_bpdet_voucdate_idx
    on raw_kcw.raw_hq_bpdet_cheques_paid ("VOUCDATE");
create index if not exists raw_hq_bpdet_chkdate_idx
    on raw_kcw.raw_hq_bpdet_cheques_paid ("CHKDATE");
create index if not exists raw_hq_bpdet_acctno_idx
    on raw_kcw.raw_hq_bpdet_cheques_paid ("ACCTNO");

grant select on raw_kcw.raw_hq_bpdet_cheques_paid to anon, authenticated, service_role;
grant select on raw_kcw.raw_hq_bpdet_cheques_paid_stg to anon, authenticated, service_role;
