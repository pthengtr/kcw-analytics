-- Product insight documents written by the HQ Spark worker.
-- Own schema so this stays out of public. Rows already in local SQLite are not copied.
-- Direct Postgres only (HQ pooler). Not exposed on the Data API.

create schema if not exists product_insight;

revoke all on schema product_insight from public, anon, authenticated;
grant usage on schema product_insight to postgres, service_role;

create table if not exists product_insight.product_insights (
    site text not null,
    bcode text not null,
    generated_at text not null,
    facts_as_of text not null,
    prompt_version text not null,
    model_id text,
    facts_hash text,
    summary text,
    insight_json jsonb not null,
    typical_monthly_qty double precision,
    suggested_cover_weeks double precision,
    safe_holding_qty double precision,
    safe_holding_reason text,
    order_ok text,
    order_ok_reason text,
    dead_stock text,
    dead_stock_reason text,
    suggested_order_qty double precision,
    suggested_order_qty_large double precision,
    order_unit text,
    order_unit_large text,
    last_supplier text,
    last_supplier_acct text,
    last_buy_price double precision,
    last_buy_date text,
    rec_qtymin double precision,
    rec_qtymin_reason text,
    check_stock text,
    stock_anomaly text,
    qtyoh_hq double precision,
    qtyoh_syp double precision,
    qtymin_hq double precision,
    qtymin_syp double precision,
    rec_transfer_qty_to_syp double precision,
    rec_transfer_reason text,
    sales_qty_30d double precision,
    sales_qty_90d double precision,
    sales_qty_12m double precision,
    trend_30d text,
    trend_90d text,
    trend_12m text,
    trend_label text,
    margin_pct_list double precision,
    margin_pct_12m double precision,
    margin_pct_prior_12m double precision,
    margin_delta_pp double precision,
    margin_flag text,
    cost_change_pct_12m double precision,
    price_change_pct_12m double precision,
    primary key (site, bcode)
);

create index if not exists product_insights_trend12_idx on product_insight.product_insights (trend_12m);
create index if not exists product_insights_trend90_idx on product_insight.product_insights (trend_90d);
create index if not exists product_insights_margin_flag_idx on product_insight.product_insights (margin_flag);
create index if not exists product_insights_order_ok_idx on product_insight.product_insights (order_ok);
create index if not exists product_insights_dead_idx on product_insight.product_insights (dead_stock);
create index if not exists product_insights_stock_anom_idx on product_insight.product_insights (stock_anomaly);
create index if not exists product_insights_last_sup_acct_idx on product_insight.product_insights (site, last_supplier_acct);

alter table product_insight.product_insights enable row level security;

revoke all on product_insight.product_insights from public, anon, authenticated;
grant select, insert, update, delete on product_insight.product_insights to postgres, service_role;
