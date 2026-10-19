-- sale_price is the net selling price per smallest unit.
-- cost_billno is the purchase bill, or bills, that backward FIFO assigned to a sale.

alter table fifo.vat_movement
    add column if not exists sale_price double precision;

alter table fifo.vat_movement
    add column if not exists cost_billno text;

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
                    'sale_price', m.sale_price,
                    'cost_billno', m.cost_billno,
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

drop function if exists public.fn_vat_inventory_export(text, text);

create or replace function public.fn_vat_inventory_export(
    p_search text default '',
    p_bcode text default '',
    p_limit integer default 5000,
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
        where case
            when nullif(btrim(coalesce(p_bcode, '')), '') is not null
                then bcode = btrim(p_bcode)
            else
                p_search is null
                or btrim(p_search) = ''
                or bcode ilike '%' || btrim(p_search) || '%'
                or descr ilike '%' || btrim(p_search) || '%'
        end
    )
    select jsonb_build_object(
        'products', case
            when coalesce(p_offset, 0) > 0 then '[]'::jsonb
            else coalesce((
                select jsonb_agg(to_jsonb(page) order by page.value desc, page.bcode)
                from (
                    select
                        bcode, descr, qty, value, uncovered_qty, open_qty,
                        purchase_qty, rv_qty, sale_qty, tar_qty, sa_qty, actual_qty
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
                join fifo.vat_product pr on pr.bcode = m.bcode
                where m.bcode in (select bcode from matched)
                order by m.bcode, m.billdate, m.id
                limit least(greatest(coalesce(p_limit, 5000), 1), 10000)
                offset greatest(coalesce(p_offset, 0), 0)
            ) move
        ), '[]'::jsonb)
    );
$$;

revoke all on function public.fn_vat_inventory_product(text) from public;
revoke all on function public.fn_vat_inventory_export(text, text, integer, integer) from public;

grant execute on function public.fn_vat_inventory_product(text) to authenticated, service_role;
grant execute on function public.fn_vat_inventory_export(text, text, integer, integer) to authenticated, service_role;
