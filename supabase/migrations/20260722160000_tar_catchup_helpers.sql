-- TAR catch-up helpers: day status + atomic multi-bill-type processing.
-- Existing process_*_day guards (skip empty staging, anti-rerun, forward-only) stay intact.

create or replace function billgen.max_fin_billdate()
returns date
language sql
stable
as $$
    select max(billdate)
    from (
        select billdate from billgen.fin_tar_lines
        union all
        select billdate from billgen.fin_3tar_lines
        union all
        select billdate from billgen.fin_cntar_lines
        union all
        select billdate from billgen.fin_3cntar_lines
    ) t;
$$;

create or replace function billgen.is_day_processed(p_billdate date)
returns boolean
language sql
stable
as $$
    select exists (
        select 1 from billgen.fin_tar_lines where billdate = p_billdate
        union all
        select 1 from billgen.fin_3tar_lines where billdate = p_billdate
        union all
        select 1 from billgen.fin_cntar_lines where billdate = p_billdate
        union all
        select 1 from billgen.fin_3cntar_lines where billdate = p_billdate
    );
$$;

-- True when any fin table has rows for the day but not all four families
-- that typically write on a busy day. Used as a warning for partial failures.
create or replace function billgen.is_day_partial(p_billdate date)
returns boolean
language plpgsql
stable
as $$
declare
    v_tar boolean;
    v_3tar boolean;
    v_cntar boolean;
    v_3cntar boolean;
    v_any boolean;
    v_all_pos boolean;
begin
    select exists(select 1 from billgen.fin_tar_lines where billdate = p_billdate)
      into v_tar;
    select exists(select 1 from billgen.fin_3tar_lines where billdate = p_billdate)
      into v_3tar;
    select exists(select 1 from billgen.fin_cntar_lines where billdate = p_billdate)
      into v_cntar;
    select exists(select 1 from billgen.fin_3cntar_lines where billdate = p_billdate)
      into v_3cntar;

    v_any := v_tar or v_3tar or v_cntar or v_3cntar;
    -- Positives are the hard gate; CNTAR may legitimately be empty that day.
    v_all_pos := v_tar and v_3tar;

    if not v_any then
        return false;
    end if;

    -- Partial if some credit/pos family written but TAR/3TAR incomplete,
    -- or only credits without positives (corrupt mid-run).
    if v_any and not v_all_pos then
        return true;
    end if;

    return false;
end;
$$;

-- Run all four processors in one transaction (caller transaction / function body).
-- Empty staging remains a no-op NOTICE inside each child function.
create or replace function billgen.process_all_bill_types_day(
    p_run_id text,
    p_billdate date
)
returns void
language plpgsql
as $$
begin
    perform billgen.process_tar_day(p_run_id, p_billdate);
    perform billgen.process_3tar_day(p_run_id, p_billdate);
    perform billgen.process_cntar_day(p_run_id, p_billdate);
    perform billgen.process_3cntar_day(p_run_id, p_billdate);
end;
$$;

comment on function billgen.max_fin_billdate() is
    'Latest billdate across all billgen.fin_* tables; used for catch-up start.';
comment on function billgen.is_day_processed(date) is
    'True if any fin_* row exists for billdate — skip-if-done for catch-up.';
comment on function billgen.process_all_bill_types_day(text, date) is
    'Atomic TAR/3TAR/CNTAR/3CNTAR day processing in a single transaction.';
