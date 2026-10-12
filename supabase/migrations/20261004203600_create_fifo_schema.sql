-- FIFO sale-line cost. Own schema, direct Postgres only (HQ pooler).
-- Not exposed on the Data API. The job replaces a product's rows and does not edit PARTS9.

create schema if not exists fifo;

revoke all on schema fifo from public, anon, authenticated;
grant usage on schema fifo to postgres, service_role;

create table if not exists fifo.sale_cost (
    site text not null,
    sidet_id bigint not null,
    bcode text not null,
    sale_billno text,
    sale_billdate date,
    units double precision not null,
    fifo_unit_cost double precision,
    fifo_ext_cost double precision,
    status text not null,
    rebuilt_at timestamptz not null,
    primary key (site, sidet_id),
    constraint fifo_sale_cost_status check (status in ('OK', 'PARTIAL', 'UNCOSTED'))
);

create index if not exists fifo_sale_cost_bcode_idx
    on fifo.sale_cost (site, bcode);

create table if not exists fifo.sale_cost_alloc (
    site text not null,
    sidet_id bigint not null,
    pidet_id bigint not null,
    pi_billno text,
    pi_billdate date,
    supplier_acctno text,
    qty double precision not null,
    unit_cost double precision not null,
    primary key (site, sidet_id, pidet_id),
    constraint fifo_sale_cost_alloc_sale
        foreign key (site, sidet_id)
        references fifo.sale_cost (site, sidet_id)
        on delete cascade
);

create index if not exists fifo_sale_cost_alloc_bill_idx
    on fifo.sale_cost_alloc (site, pi_billno);

create table if not exists fifo.product (
    site text not null,
    bcode text not null,
    qtyoh2_icmas double precision,
    qtyoh2_ledger double precision not null,
    qtyoh2_stale boolean not null,
    on_hand_qty double precision not null,
    on_hand_covered_qty double precision not null,
    on_hand_value double precision not null,
    unassigned_qty double precision not null,
    history_window text not null,
    cost_complete boolean not null,
    last_movement_at date,
    last_movement_kind text,
    cleanup_recommend boolean not null,
    cleanup_reason text not null,
    rebuilt_at timestamptz not null,
    primary key (site, bcode),
    constraint fifo_product_window check (history_window in ('5y', 'full')),
    constraint fifo_product_cleanup check (
        cleanup_reason in ('zero_oh_no_movement_5y', 'still_on_hand', 'moved_within_5y')
    )
);

create index if not exists fifo_product_cleanup_idx
    on fifo.product (site, cleanup_recommend);

alter table fifo.sale_cost enable row level security;
alter table fifo.sale_cost_alloc enable row level security;
alter table fifo.product enable row level security;

revoke all on fifo.sale_cost from public, anon, authenticated;
revoke all on fifo.sale_cost_alloc from public, anon, authenticated;
revoke all on fifo.product from public, anon, authenticated;

grant select, insert, update, delete on fifo.sale_cost to postgres, service_role;
grant select, insert, update, delete on fifo.sale_cost_alloc to postgres, service_role;
grant select, insert, update, delete on fifo.product to postgres, service_role;
