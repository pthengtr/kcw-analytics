-- HQ ICLOW stock-order / pending-receive tracker (PARTS9), matching Drive raw CSV:
--   raw_hq_iclow_stock_orders.csv
-- Pending receive (ค้างรับ) = ORDERED='Y' AND RECEIVED='N' AND CANCELED='N'.
-- See docs/parts9_pending_receive.md.

create schema if not exists raw_kcw;

create table if not exists raw_kcw.raw_hq_iclow_stock_orders (
    _ingested_at timestamptz not null default now(),
    _source_file text,
    "ID" text,
    "JOURMODE" text,
    "BILLDATE" text,
    "BILLNO" text,
    "BCODE" text,
    "MCODE" text,
    "PCODE" text,
    "DESCR" text,
    "MODEL" text,
    "BRAND" text,
    "OEM" text,
    "VENDOR" text,
    "MAIN" text,
    "SUB" text,
    "PART" text,
    "STATUS" text,
    "SERIAL" text,
    "TAXIC" text,
    "EXMPT" text,
    "LOCATION1" text,
    "LOCATION2" text,
    "QTYOH" text,
    "QTY" text,
    "UI" text,
    "MTP" text,
    "PRICE" text,
    "AMOUNT" text,
    "ORDERED" text,
    "DOCDATE" text,
    "DOCNO" text,
    "RECEIVED" text,
    "RCVDDATE" text,
    "RCVDNO" text,
    "CANCELED" text,
    "DONE" text
);

create table if not exists raw_kcw.raw_hq_iclow_stock_orders_stg
    (like raw_kcw.raw_hq_iclow_stock_orders including all);

create index if not exists raw_hq_iclow_docno_idx
    on raw_kcw.raw_hq_iclow_stock_orders ("DOCNO");
create index if not exists raw_hq_iclow_docdate_idx
    on raw_kcw.raw_hq_iclow_stock_orders ("DOCDATE");
create index if not exists raw_hq_iclow_bcode_idx
    on raw_kcw.raw_hq_iclow_stock_orders ("BCODE");
create index if not exists raw_hq_iclow_vendor_idx
    on raw_kcw.raw_hq_iclow_stock_orders ("VENDOR");
create index if not exists raw_hq_iclow_ordered_received_idx
    on raw_kcw.raw_hq_iclow_stock_orders ("ORDERED", "RECEIVED");

grant select on raw_kcw.raw_hq_iclow_stock_orders to anon, authenticated, service_role;
grant select on raw_kcw.raw_hq_iclow_stock_orders_stg to anon, authenticated, service_role;
