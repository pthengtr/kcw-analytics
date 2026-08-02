-- Fix SIDET raw table column shape to match PARTS9 / Drive CSV.
-- SIDET aligns with PIDET minus purchase-sale refs (SALEDATE/SALENO/SALEPRICE),
-- not SIMAS headers (BOOKNO).

alter table raw_kcw.raw_hq_sidet_sales_lines
    drop column if exists "BOOKNO";

alter table raw_kcw.raw_hq_sidet_sales_lines
    add column if not exists "CHGAMT" text;

drop table if exists raw_kcw.raw_hq_sidet_sales_lines_stg;
create table raw_kcw.raw_hq_sidet_sales_lines_stg
    (like raw_kcw.raw_hq_sidet_sales_lines including all);

grant select on raw_kcw.raw_hq_sidet_sales_lines_stg to anon, authenticated, service_role;
