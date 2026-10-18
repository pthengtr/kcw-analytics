-- VAT-only inventory. Filled by `python -m src.kcw.pipeline vat-inventory`.
-- Pages read it through the security-definer functions below.

create table if not exists fifo.vat_product (
    bcode text primary key,
    descr text not null default '',
    qty double precision not null,
    value double precision not null,
    uncovered_qty double precision not null,
    open_qty double precision not null,
    purchase_qty double precision not null,
    rv_qty double precision not null,
    sale_qty double precision not null,
    tar_qty double precision not null,
    sa_qty double precision not null,
    actual_qty double precision not null,
    rebuilt_at timestamptz not null
);

create table if not exists fifo.vat_layer (
    id bigint generated always as identity primary key,
    bcode text not null references fifo.vat_product (bcode) on delete cascade,
    source text not null,
    billno text not null,
    billdate date not null,
    supplier text not null default '',
    site text not null default '',
    qty double precision not null,
    unit_cost double precision not null
);

create index if not exists fifo_vat_layer_bcode_idx on fifo.vat_layer (bcode, billdate);

create table if not exists fifo.vat_movement (
    id bigint generated always as identity primary key,
    bcode text not null references fifo.vat_product (bcode) on delete cascade,
    billdate date not null,
    billno text not null default '',
    kind text not null,
    site text not null default '',
    qty double precision not null,
    unit_cost double precision,
    balance double precision not null,
    supplier text not null default ''
);

create index if not exists fifo_vat_movement_bcode_idx
    on fifo.vat_movement (bcode, billdate, id);

alter table fifo.vat_product enable row level security;
alter table fifo.vat_layer enable row level security;
alter table fifo.vat_movement enable row level security;

revoke all on fifo.vat_product from public, anon, authenticated;
revoke all on fifo.vat_layer from public, anon, authenticated;
revoke all on fifo.vat_movement from public, anon, authenticated;

grant select, insert, update, delete on fifo.vat_product to postgres, service_role;
grant select, insert, update, delete on fifo.vat_layer to postgres, service_role;
grant select, insert, update, delete on fifo.vat_movement to postgres, service_role;

create or replace function public.fn_vat_inventory_summary()
returns jsonb
language sql
stable
security definer
set search_path = fifo, public
as $$
    select jsonb_build_object(
        'products', count(*),
        'qty', coalesce(sum(qty), 0),
        'value', coalesce(sum(value), 0),
        'rebuilt_at', max(rebuilt_at)
    )
    from fifo.vat_product;
$$;

create or replace function public.fn_vat_inventory_products(
    p_search text default '',
    p_limit integer default 100,
    p_offset integer default 0
)
returns jsonb
language sql
stable
security definer
set search_path = fifo, public
as $$
    with matched as (
        select *
        from fifo.vat_product
        where p_search is null
           or btrim(p_search) = ''
           or bcode ilike '%' || btrim(p_search) || '%'
           or descr ilike '%' || btrim(p_search) || '%'
    )
    select jsonb_build_object(
        'total', (select count(*) from matched),
        'rows', coalesce((
            select jsonb_agg(to_jsonb(page) order by page.value desc, page.bcode)
            from (
                select
                    bcode, descr, qty, value, uncovered_qty, open_qty,
                    purchase_qty, rv_qty, sale_qty, tar_qty, sa_qty, actual_qty
                from matched
                order by value desc, bcode
                limit least(greatest(coalesce(p_limit, 100), 1), 200)
                offset greatest(coalesce(p_offset, 0), 0)
            ) page
        ), '[]'::jsonb)
    );
$$;

create or replace function public.fn_vat_inventory_product(p_bcode text)
returns jsonb
language sql
stable
security definer
set search_path = fifo, public
as $$
    select jsonb_build_object(
        'product', (
            select to_jsonb(p)
            from fifo.vat_product p
            where p.bcode = btrim(p_bcode)
        ),
        'layers', coalesce((
            select jsonb_agg(to_jsonb(l) order by l.billdate, l.billno)
            from fifo.vat_layer l
            where l.bcode = btrim(p_bcode)
        ), '[]'::jsonb),
        'movements', coalesce((
            select jsonb_agg(
                jsonb_build_object(
                    'billdate', m.billdate,
                    'billno', m.billno,
                    'kind', m.kind,
                    'site', m.site,
                    'qty', m.qty,
                    'unit_cost', m.unit_cost,
                    'balance', m.balance,
                    'supplier', m.supplier
                )
                order by m.billdate, m.id
            )
            from fifo.vat_movement m
            where m.bcode = btrim(p_bcode)
        ), '[]'::jsonb)
    );
$$;

revoke all on function public.fn_vat_inventory_summary() from public;
revoke all on function public.fn_vat_inventory_products(text, integer, integer) from public;
revoke all on function public.fn_vat_inventory_product(text) from public;

grant execute on function public.fn_vat_inventory_summary() to authenticated, service_role;
grant execute on function public.fn_vat_inventory_products(text, integer, integer) to authenticated, service_role;
grant execute on function public.fn_vat_inventory_product(text) to authenticated, service_role;

insert into public.kcw_role_page_permissions (role_key, page_key)
select role_key, 'bi_vat_inventory'
from public.kcw_role_page_permissions
where page_key = 'bi_vat'
on conflict do nothing;

insert into public.kcw_role_page_permissions (role_key, page_key)
values ('external', 'bi_vat_inventory')
on conflict do nothing;
