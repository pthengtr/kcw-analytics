-- Credit notes name the sale they reverse.

alter table fifo.vat_movement add column if not exists ref_billno text;

create or replace function public.fn_vat_inventory_product(
    p_bcode text,
    p_site text default 'hq',
    p_year integer default 2026
)
returns jsonb
language sql
stable
security definer
set search_path = fifo, public
as $$
    with scoped as (
        select *
        from fifo.vat_product
        where bcode = btrim(p_bcode)
          and year = coalesce(p_year, 2026)
          and (
              lower(btrim(coalesce(p_site, ''))) in ('', 'all')
              or site = lower(btrim(p_site))
          )
    )
    select jsonb_build_object(
        'product', (
            select jsonb_build_object(
                'bcode', bcode,
                'site', case
                    when lower(btrim(coalesce(p_site, ''))) in ('', 'all') then 'all'
                    else min(site)
                end,
                'descr', max(descr),
                'qty', sum(qty),
                'value', sum(value),
                'uncovered_qty', sum(uncovered_qty),
                'open_qty', sum(open_qty),
                'purchase_qty', sum(purchase_qty),
                'rv_qty', sum(rv_qty),
                'sale_qty', sum(sale_qty),
                'tar_qty', sum(tar_qty),
                'sa_qty', sum(sa_qty),
                'tf_qty', sum(tf_qty),
                'actual_qty', sum(actual_qty),
                'rebuilt_at', max(rebuilt_at)
            )
            from scoped
            group by bcode
        ),
        'layers', coalesce((
            select jsonb_agg(to_jsonb(l) order by l.billdate, l.billno, l.site)
            from fifo.vat_layer l
            where l.bcode = btrim(p_bcode)
              and l.year = coalesce(p_year, 2026)
              and (
                  lower(btrim(coalesce(p_site, ''))) in ('', 'all')
                  or l.site = lower(btrim(p_site))
              )
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
                    'ref_billno', m.ref_billno,
                    'balance', m.balance,
                    'supplier', m.supplier
                )
                order by m.billdate, m.site, m.id
            )
            from fifo.vat_movement m
            where m.bcode = btrim(p_bcode)
              and m.year = coalesce(p_year, 2026)
              and (
                  lower(btrim(coalesce(p_site, ''))) in ('', 'all')
                  or m.site = lower(btrim(p_site))
              )
        ), '[]'::jsonb)
    );
$$;

create or replace function public.fn_vat_inventory_export(
    p_search text default '',
    p_bcode text default '',
    p_limit integer default 5000,
    p_offset integer default 0,
    p_site text default 'hq',
    p_year integer default 2026
)
returns jsonb
language sql
stable
security definer
set search_path = fifo, public
as $$
    with scoped as (
        select *
        from fifo.vat_product
        where year = coalesce(p_year, 2026)
          and (
              lower(btrim(coalesce(p_site, ''))) in ('', 'all')
              or site = lower(btrim(p_site))
          )
          and case
              when nullif(btrim(coalesce(p_bcode, '')), '') is not null
                  then bcode = btrim(p_bcode)
              else
                  p_search is null
                  or btrim(p_search) = ''
                  or bcode ilike '%' || btrim(p_search) || '%'
                  or descr ilike '%' || btrim(p_search) || '%'
          end
    ),
    matched as (
        select
            bcode,
            case
                when lower(btrim(coalesce(p_site, ''))) in ('', 'all') then 'all'
                else min(site)
            end as site,
            max(descr) as descr,
            sum(qty) as qty,
            sum(value) as value,
            sum(uncovered_qty) as uncovered_qty,
            sum(open_qty) as open_qty,
            sum(purchase_qty) as purchase_qty,
            sum(rv_qty) as rv_qty,
            sum(sale_qty) as sale_qty,
            sum(tar_qty) as tar_qty,
            sum(sa_qty) as sa_qty,
            sum(tf_qty) as tf_qty,
            sum(actual_qty) as actual_qty
        from scoped
        group by bcode
    )
    select jsonb_build_object(
        'products', case
            when coalesce(p_offset, 0) > 0 then '[]'::jsonb
            else coalesce((
                select jsonb_agg(to_jsonb(page) order by page.value desc, page.bcode)
                from (select * from matched order by value desc, bcode) page
            ), '[]'::jsonb)
        end,
        'layers', case
            when nullif(btrim(coalesce(p_bcode, '')), '') is null then '[]'::jsonb
            else coalesce((
                select jsonb_agg(row_to_json(layer) order by layer.billdate, layer.billno)
                from (
                    select l.billdate, l.billno, l.source, l.supplier, l.site, l.qty, l.unit_cost
                    from fifo.vat_layer l
                    where l.bcode = btrim(p_bcode)
                      and l.year = coalesce(p_year, 2026)
                      and (
                          lower(btrim(coalesce(p_site, ''))) in ('', 'all')
                          or l.site = lower(btrim(p_site))
                      )
                ) layer
            ), '[]'::jsonb)
        end,
        'movements', case
            when nullif(btrim(coalesce(p_bcode, '')), '') is null then '[]'::jsonb
            else coalesce((
                select jsonb_agg(row_to_json(move) order by move.billdate, move.id)
                from (
                    select
                        m.id, m.bcode, pr.descr, m.billdate, m.billno, m.kind, m.site,
                        m.qty, m.unit_cost, m.sale_price, m.cost_billno, m.ref_billno, m.balance, m.supplier
                    from fifo.vat_movement m
                    join fifo.vat_product pr
                      on pr.bcode = m.bcode and pr.site = m.site and pr.year = m.year
                    where m.bcode = btrim(p_bcode)
                      and m.year = coalesce(p_year, 2026)
                      and (
                          lower(btrim(coalesce(p_site, ''))) in ('', 'all')
                          or m.site = lower(btrim(p_site))
                      )
                    order by m.billdate, m.id
                    limit least(greatest(coalesce(p_limit, 5000), 1), 10000)
                    offset greatest(coalesce(p_offset, 0), 0)
                ) move
            ), '[]'::jsonb)
        end
    );
$$;


revoke all on function public.fn_vat_inventory_product(text, text, integer) from public;
revoke all on function public.fn_vat_inventory_export(text, text, integer, integer, text, integer) from public;

grant execute on function public.fn_vat_inventory_product(text, text, integer) to authenticated, service_role;
grant execute on function public.fn_vat_inventory_export(text, text, integer, integer, text, integer) to authenticated, service_role;

notify pgrst, 'reload schema';
