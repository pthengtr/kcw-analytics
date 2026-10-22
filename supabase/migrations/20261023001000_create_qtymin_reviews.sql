-- QTYMIN recommendations written by the insight worker. Not applied to PARTS9.

create table if not exists product_insight.qtymin_reviews (
    site text not null,
    bcode text not null,
    current_qtymin double precision,
    current_meaning text not null,
    decision text not null,
    suggested_qtymin double precision,
    why text,
    watch jsonb,
    insight_generated_at text,
    reviewed_at text not null,
    source text not null,
    primary key (site, bcode)
);

alter table product_insight.qtymin_reviews enable row level security;

revoke all on product_insight.qtymin_reviews from public, anon, authenticated;
grant select, insert, update, delete on product_insight.qtymin_reviews to postgres, service_role;
