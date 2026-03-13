-- =========================================================
-- TAR: STG -> FIN BY DAY
-- =========================================================

create or replace function billgen.process_tar_day(p_run_id text, p_billdate date)
returns void
language plpgsql
as $$
declare
    v_stg_count integer;
    v_latest_fin_billdate date;
    v_yyyymm text;
    v_prefix text;
    v_start_seq integer;
    v_chunk_count integer;
begin
    -- 1) rows must exist for that run_id + day
    select count(*)
    into v_stg_count
    from billgen.stg_tar_lines
    where run_id = p_run_id
      and billdate = p_billdate;

    if v_stg_count = 0 then
        raise exception 'No staging rows found for run_id=% and billdate=%', p_run_id, p_billdate;
    end if;

    -- 2) reject rerun for same run_id + day
    if exists (
        select 1
        from billgen.fin_tar_lines
        where run_id = p_run_id
          and billdate = p_billdate
    ) then
        raise exception 'Run % for billdate % already processed into billgen.fin_tar_lines', p_run_id, p_billdate;
    end if;

    -- 3) forward-only date safety
    select max(billdate)
    into v_latest_fin_billdate
    from billgen.fin_tar_lines;

    if v_latest_fin_billdate is not null and p_billdate < v_latest_fin_billdate then
        raise exception 'Staging billdate % is earlier than latest final billdate %',
            p_billdate, v_latest_fin_billdate;
    end if;

    -- 4) Buddhist YYMM
    v_yyyymm :=
        lpad((((extract(year from p_billdate)::int + 543) % 100))::text, 2, '0') ||
        lpad(extract(month from p_billdate)::int::text, 2, '0');

    v_prefix := v_yyyymm || '-';

    -- 5) ensure control row exists
    insert into billgen.bill_seq_control (bill_type, yyyymm, last_seq)
    values ('TAR', v_yyyymm, 0)
    on conflict (bill_type, yyyymm) do nothing;

    -- 6) lock sequence row
    select last_seq
    into v_start_seq
    from billgen.bill_seq_control
    where bill_type = 'TAR'
      and yyyymm = v_yyyymm
    for update;

    -- 7) count generated bills needed
    select coalesce(max(chunk_no), 0)
    into v_chunk_count
    from (
        select ((row_number() over (order by billno, bcode, id) - 1) / 20) + 1 as chunk_no
        from billgen.stg_tar_lines
        where run_id = p_run_id
          and billdate = p_billdate
    ) t;

    if v_start_seq + v_chunk_count > 999 then
        raise exception 'Sequence overflow for TAR %: current last_seq=%, new chunks=%, max=999',
            v_yyyymm, v_start_seq, v_chunk_count;
    end if;

    -- 8) insert final rows
    with ordered as (
        select
            s.*,
            row_number() over (
                order by s.billno, s.bcode, s.id
            ) as rn
        from billgen.stg_tar_lines s
        where s.run_id = p_run_id
          and s.billdate = p_billdate
    ),
    chunked as (
        select
            *,
            ((rn - 1) / 20) + 1 as chunk_no
        from ordered
    )
    insert into billgen.fin_tar_lines (
        run_id, billdate, billno, new_billno, bcode, qty, mtp, ui, price, amount
    )
    select
        run_id,
        billdate,
        billno,
        v_prefix || lpad((v_start_seq + chunk_no)::text, 3, '0') as new_billno,
        bcode,
        qty,
        mtp,
        ui,
        price,
        amount
    from chunked
    order by rn;

    -- 9) bump sequence
    update billgen.bill_seq_control
    set
        last_seq = v_start_seq + v_chunk_count,
        updated_at = now()
    where bill_type = 'TAR'
      and yyyymm = v_yyyymm;
end;
$$;

-- =========================================================
-- 3TAR: STG -> FIN BY DAY
-- =========================================================

create or replace function billgen.process_3tar_day(p_run_id text, p_billdate date)
returns void
language plpgsql
as $$
declare
    v_stg_count integer;
    v_latest_fin_billdate date;
    v_yyyymm text;
    v_prefix text;
    v_start_seq integer;
    v_chunk_count integer;
begin
    -- 1) rows must exist for that run_id + day
    select count(*)
    into v_stg_count
    from billgen.stg_3tar_lines
    where run_id = p_run_id
      and billdate = p_billdate;

    if v_stg_count = 0 then
        raise exception 'No staging rows found for run_id=% and billdate=%', p_run_id, p_billdate;
    end if;

    -- 2) reject rerun for same run_id + day
    if exists (
        select 1
        from billgen.fin_3tar_lines
        where run_id = p_run_id
          and billdate = p_billdate
    ) then
        raise exception 'Run % for billdate % already processed into billgen.fin_3tar_lines', p_run_id, p_billdate;
    end if;

    -- 3) forward-only date safety
    select max(billdate)
    into v_latest_fin_billdate
    from billgen.fin_3tar_lines;

    if v_latest_fin_billdate is not null and p_billdate < v_latest_fin_billdate then
        raise exception 'Staging billdate % is earlier than latest final billdate %',
            p_billdate, v_latest_fin_billdate;
    end if;

    -- 4) Buddhist YYMM
    v_yyyymm :=
        lpad((((extract(year from p_billdate)::int + 543) % 100))::text, 2, '0') ||
        lpad(extract(month from p_billdate)::int::text, 2, '0');

    v_prefix := v_yyyymm || '-';

    -- 5) ensure control row exists
    insert into billgen.bill_seq_control (bill_type, yyyymm, last_seq)
    values ('3TAR', v_yyyymm, 0)
    on conflict (bill_type, yyyymm) do nothing;

    -- 6) lock sequence row
    select last_seq
    into v_start_seq
    from billgen.bill_seq_control
    where bill_type = '3TAR'
      and yyyymm = v_yyyymm
    for update;

    -- 7) count generated bills needed
    select coalesce(max(chunk_no), 0)
    into v_chunk_count
    from (
        select ((row_number() over (order by billno, bcode, id) - 1) / 20) + 1 as chunk_no
        from billgen.stg_3tar_lines
        where run_id = p_run_id
          and billdate = p_billdate
    ) t;

    if v_start_seq + v_chunk_count > 999 then
        raise exception 'Sequence overflow for 3TAR %: current last_seq=%, new chunks=%, max=999',
            v_yyyymm, v_start_seq, v_chunk_count;
    end if;

    -- 8) insert final rows
    with ordered as (
        select
            s.*,
            row_number() over (
                order by s.billno, s.bcode, s.id
            ) as rn
        from billgen.stg_3tar_lines s
        where s.run_id = p_run_id
          and s.billdate = p_billdate
    ),
    chunked as (
        select
            *,
            ((rn - 1) / 20) + 1 as chunk_no
        from ordered
    )
    insert into billgen.fin_3tar_lines (
        run_id, billdate, billno, new_billno, bcode, qty, mtp, ui, price, amount
    )
    select
        run_id,
        billdate,
        billno,
        v_prefix || lpad((v_start_seq + chunk_no)::text, 3, '0') as new_billno,
        bcode,
        qty,
        mtp,
        ui,
        price,
        amount
    from chunked
    order by rn;

    -- 9) bump sequence
    update billgen.bill_seq_control
    set
        last_seq = v_start_seq + v_chunk_count,
        updated_at = now()
    where bill_type = '3TAR'
      and yyyymm = v_yyyymm;
end;
$$;