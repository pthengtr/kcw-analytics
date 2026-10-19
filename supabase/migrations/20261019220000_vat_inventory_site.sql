-- One VAT book per branch. A transfer is stock leaving one site and arriving at the other.

do $$
declare
    ref record;
begin
    for ref in
        select con.conname, rel.relname
        from pg_constraint con
        join pg_class rel on rel.oid = con.conrelid
        join pg_namespace nsp on nsp.oid = rel.relnamespace
        where nsp.nspname = 'fifo'
          and con.contype = 'f'
          and con.confrelid = 'fifo.vat_product'::regclass
    loop
        execute format('alter table fifo.%I drop constraint %I', ref.relname, ref.conname);
    end loop;
end $$;

truncate table fifo.vat_movement, fifo.vat_layer, fifo.vat_product;

alter table fifo.vat_product drop constraint if exists vat_product_pkey;
alter table fifo.vat_product add column if not exists site text not null default 'hq';
alter table fifo.vat_product add column if not exists tf_qty double precision not null default 0;
alter table fifo.vat_product drop constraint if exists vat_product_pkey;
alter table fifo.vat_product add primary key (bcode, site);

alter table fifo.vat_layer
    add constraint vat_layer_product_fkey
    foreign key (bcode, site) references fifo.vat_product (bcode, site) on delete cascade;

alter table fifo.vat_movement
    add constraint vat_movement_product_fkey
    foreign key (bcode, site) references fifo.vat_product (bcode, site) on delete cascade;

drop function if exists public.fn_vat_inventory_summary();
drop function if exists public.fn_vat_inventory_products(text, integer, integer);
drop function if exists public.fn_vat_inventory_product(text);
drop function if exists public.fn_vat_inventory_export(text, text, integer, integer);

create or replace function public.fn_vat_inventory_summary(p_site text default 'hq')
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
    from fifo.vat_product
    where site = case when lower(btrim(coalesce(p_site, ''))) = 'syp' then 'syp' else 'hq' end;
$$;

create or replace function public.fn_vat_inventory_products(
    p_search text default '',
    p_limit integer default 100,
    p_offset integer default 0,
    p_site text default 'hq'
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
        where site = case when lower(btrim(coalesce(p_site, ''))) = 'syp' then 'syp' else 'hq' end
          and (
            p_search is null
            or btrim(p_search) = ''
            or bcode ilike '%' || btrim(p_search) || '%'
            or descr ilike '%' || btrim(p_search) || '%'
          )
    )
    select jsonb_build_object(
        'total', (select count(*) from matched),
        'rows', coalesce((
            select jsonb_agg(to_jsonb(page) order by page.value desc, page.bcode)
            from (
                select
                    bcode, site, descr, qty, value, uncovered_qty, open_qty,
                    purchase_qty, rv_qty, sale_qty, tar_qty, sa_qty, tf_qty, actual_qty
                from matched
                order by value desc, bcode
                limit least(greatest(coalesce(p_limit, 100), 1), 200)
                offset greatest(coalesce(p_offset, 0), 0)
            ) page
        ), '[]'::jsonb)
    );
$$;

create or replace function public.fn_vat_inventory_product(
    p_bcode text,
    p_site text default 'hq'
)
returns jsonb
language sql
stable
security definer
set search_path = fifo, public
as $$
    with chosen as (
        select case when lower(btrim(coalesce(p_site, ''))) = 'syp' then 'syp' else 'hq' end as site
    )
    select jsonb_build_object(
        'product', (
            select to_jsonb(p)
            from fifo.vat_product p, chosen
            where p.bcode = btrim(p_bcode)
              and p.site = chosen.site
        ),
        'layers', coalesce((
            select jsonb_agg(to_jsonb(l) order by l.billdate, l.billno)
            from fifo.vat_layer l, chosen
            where l.bcode = btrim(p_bcode)
              and l.site = chosen.site
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
                    'sale_price', m.sale_price,
                    'cost_billno', m.cost_billno,
                    'balance', m.balance,
                    'supplier', m.supplier
                )
                order by m.billdate, m.id
            )
            from fifo.vat_movement m, chosen
            where m.bcode = btrim(p_bcode)
              and m.site = chosen.site
        ), '[]'::jsonb)
    );
$$;

create or replace function public.fn_vat_inventory_export(
    p_search text default '',
    p_bcode text default '',
    p_limit integer default 5000,
    p_offset integer default 0,
    p_site text default 'hq'
)
returns jsonb
language sql
stable
security definer
set search_path = fifo, public
as $$
    with chosen as (
        select case when lower(btrim(coalesce(p_site, ''))) = 'syp' then 'syp' else 'hq' end as site
    ),
    matched as (
        select p.*
        from fifo.vat_product p, chosen
        where p.site = chosen.site
          and case
            when nullif(btrim(coalesce(p_bcode, '')), '') is not null
                then p.bcode = btrim(p_bcode)
            else
                p_search is null
                or btrim(p_search) = ''
                or p.bcode ilike '%' || btrim(p_search) || '%'
                or p.descr ilike '%' || btrim(p_search) || '%'
          end
    )
    select jsonb_build_object(
        'products', case
            when coalesce(p_offset, 0) > 0 then '[]'::jsonb
            else coalesce((
                select jsonb_agg(to_jsonb(page) order by page.value desc, page.bcode)
                from (
                    select
                        bcode, site, descr, qty, value, uncovered_qty, open_qty,
                        purchase_qty, rv_qty, sale_qty, tar_qty, sa_qty, tf_qty, actual_qty
                    from matched
                    order by value desc, bcode
                ) page
            ), '[]'::jsonb)
        end,
        'movements', coalesce((
            select jsonb_agg(row_to_json(move) order by move.bcode, move.billdate, move.id)
            from (
                select
                    m.id,
                    m.bcode,
                    pr.descr,
                    m.billdate,
                    m.billno,
                    m.kind,
                    m.site,
                    m.qty,
                    m.unit_cost,
                    m.sale_price,
                    m.cost_billno,
                    m.balance,
                    m.supplier
                from fifo.vat_movement m
                join fifo.vat_product pr on pr.bcode = m.bcode and pr.site = m.site
                where pr.bcode in (select bcode from matched)
                  and pr.site = (select site from chosen)
                order by m.bcode, m.billdate, m.id
                limit least(greatest(coalesce(p_limit, 5000), 1), 10000)
                offset greatest(coalesce(p_offset, 0), 0)
            ) move
        ), '[]'::jsonb)
    );
$$;

revoke all on function public.fn_vat_inventory_summary(text) from public;
revoke all on function public.fn_vat_inventory_products(text, integer, integer, text) from public;
revoke all on function public.fn_vat_inventory_product(text, text) from public;
revoke all on function public.fn_vat_inventory_export(text, text, integer, integer, text) from public;

grant execute on function public.fn_vat_inventory_summary(text) to authenticated, service_role;
grant execute on function public.fn_vat_inventory_products(text, integer, integer, text) to authenticated, service_role;
grant execute on function public.fn_vat_inventory_product(text, text) to authenticated, service_role;
grant execute on function public.fn_vat_inventory_export(text, text, integer, integer, text) to authenticated, service_role;
