-- Align SELECT grants with curated_kcw.fact_sales_all / fact_sales_all_stg.
grant select on curated_kcw.fact_sales_bills_all to anon, authenticated, service_role;
grant select on curated_kcw.fact_sales_bills_all_stg to anon, authenticated, service_role;
