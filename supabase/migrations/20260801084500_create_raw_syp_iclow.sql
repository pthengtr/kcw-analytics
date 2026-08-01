-- SYP ICLOW stock-order / pending-receive tracker (PARTS9), matching Drive raw CSV:
--   raw_syp_iclow_stock_orders.csv
-- Same shape as raw_hq_iclow_stock_orders.
-- Pending receive (ค้างรับ) = ORDERED='Y' AND RECEIVED='N' AND CANCELED='N'.
-- See docs/parts9_pending_receive.md.

create schema if not exists raw_kcw;

create table if not exists raw_kcw.raw_syp_iclow_stock_orders
    (like raw_kcw.raw_hq_iclow_stock_orders including all);

create table if not exists raw_kcw.raw_syp_iclow_stock_orders_stg
    (like raw_kcw.raw_syp_iclow_stock_orders including all);

create index if not exists raw_syp_iclow_docno_idx
    on raw_kcw.raw_syp_iclow_stock_orders ("DOCNO");
create index if not exists raw_syp_iclow_docdate_idx
    on raw_kcw.raw_syp_iclow_stock_orders ("DOCDATE");
create index if not exists raw_syp_iclow_bcode_idx
    on raw_kcw.raw_syp_iclow_stock_orders ("BCODE");
create index if not exists raw_syp_iclow_vendor_idx
    on raw_kcw.raw_syp_iclow_stock_orders ("VENDOR");
create index if not exists raw_syp_iclow_ordered_received_idx
    on raw_kcw.raw_syp_iclow_stock_orders ("ORDERED", "RECEIVED");

grant select on raw_kcw.raw_syp_iclow_stock_orders to anon, authenticated, service_role;
grant select on raw_kcw.raw_syp_iclow_stock_orders_stg to anon, authenticated, service_role;
