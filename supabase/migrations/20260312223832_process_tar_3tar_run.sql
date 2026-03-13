-- =========================================================
-- BILL SEQUENCE CONTROL
-- =========================================================

create table if not exists billgen.bill_seq_control (
    bill_type text not null,
    yyyymm text not null,
    last_seq integer not null default 0,
    updated_at timestamptz not null default now(),
    primary key (bill_type, yyyymm)
);

-- =========================================================
-- TAR: STG -> FIN
-- Rules:
-- - one run_id must contain exactly one billdate
-- - reject if staging billdate < latest final billdate
-- - max 20 lines per generated bill
-- - NEW_BILLNO format = YYMM-SEQ
--   where YY = Buddhist year last 2 digits, MM = month, SEQ = 3 digits
-- - reject rerun if run_id already exists in final
-- =========================================================

create or replace function billgen.process_tar_run(p_run_id text)
returns void
language plpgsql
as $$
declare
    v_min_billdate date;
    v_max_billdate date;
    v_billdate date;
    v_latest_fin_billdate date;
    v_yyyymm text;
    v_prefix text;
    v_start_seq integer;
    v_chunk_count integer;
begin
    -- 1) run must exist
    select min(billdate), max(billdate)
    into v_min_billdate, v_max_billdate
    from billgen.stg_tar_lines
    where run_id = p_run_id;

    if v_min_billdate is null then
        raise exception 'No staging rows found for run_id=%', p_run_id;
    end if;

    -- 2) one run_id = one billdate only
    if v_min_billdate <> v_max_billdate then
        raise exception 'Run % contains multiple bill dates (% to %)',
            p_run_id, v_min_billdate, v_max_billdate;
    end if;

    v_billdate := v_min_billdate;

    -- 3) reject rerun into final
    if exists (
        select 1
        from billgen.fin_tar_lines
        where run_id = p_run_id
    ) then
        raise exception 'Run % already processed into billgen.fin_tar_lines', p_run_id;
    end if;

    -- 4) forward-only date safety
    select max(billdate)
    into v_latest_fin_billdate
    from billgen.fin_tar_lines;

    if v_latest_fin_billdate is not null and v_billdate < v_latest_fin_billdate then
        raise exception 'Staging billdate % is earlier than latest final billdate %',
            v_billdate, v_latest_fin_billdate;
    end if;

    -- 5) Buddhist YYMM
    v_yyyymm :=
        lpad((((extract(year from v_billdate)::int + 543) % 100))::text, 2, '0') ||
        lpad(extract(month from v_billdate)::int::text, 2, '0');

    v_prefix := v_yyyymm || '-';

    -- 6) ensure control row exists
    insert into billgen.bill_seq_control (bill_type, yyyymm, last_seq)
    values ('TAR', v_yyyymm, 0)
    on conflict (bill_type, yyyymm) do nothing;

    -- 7) lock sequence row and get starting seq
    select last_seq
    into v_start_seq
    from billgen.bill_seq_control
    where bill_type = 'TAR'
      and yyyymm = v_yyyymm
    for update;

    -- 8) how many generated bills needed
    select coalesce(max(chunk_no), 0)
    into v_chunk_count
    from (
        select ((row_number() over (order by billno, bcode, id) - 1) / 20) + 1 as chunk_no
        from billgen.stg_tar_lines
        where run_id = p_run_id
    ) t;

    -- 9) protect 3-digit sequence overflow
    if v_start_seq + v_chunk_count > 999 then
        raise exception 'Sequence overflow for TAR %: current last_seq=%, new chunks=%, max=999',
            v_yyyymm, v_start_seq, v_chunk_count;
    end if;

    -- 10) insert final rows
    with ordered as (
        select
            s.*,
            row_number() over (
                order by s.billno, s.bcode, s.id
            ) as rn
        from billgen.stg_tar_lines s
        where s.run_id = p_run_id
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

    -- 11) bump sequence control
    update billgen.bill_seq_control
    set
        last_seq = v_start_seq + v_chunk_count,
        updated_at = now()
    where bill_type = 'TAR'
      and yyyymm = v_yyyymm;
end;
$$;

-- =========================================================
-- 3TAR: STG -> FIN
-- Same rules as TAR
-- =========================================================

create or replace function billgen.process_3tar_run(p_run_id text)
returns void
language plpgsql
as $$
declare
    v_min_billdate date;
    v_max_billdate date;
    v_billdate date;
    v_latest_fin_billdate date;
    v_yyyymm text;
    v_prefix text;
    v_start_seq integer;
    v_chunk_count integer;
begin
    -- 1) run must exist
    select min(billdate), max(billdate)
    into v_min_billdate, v_max_billdate
    from billgen.stg_3tar_lines
    where run_id = p_run_id;

    if v_min_billdate is null then
        raise exception 'No staging rows found for run_id=%', p_run_id;
    end if;

    -- 2) one run_id = one billdate only
    if v_min_billdate <> v_max_billdate then
        raise exception 'Run % contains multiple bill dates (% to %)',
            p_run_id, v_min_billdate, v_max_billdate;
    end if;

    v_billdate := v_min_billdate;

    -- 3) reject rerun into final
    if exists (
        select 1
        from billgen.fin_3tar_lines
        where run_id = p_run_id
    ) then
        raise exception 'Run % already processed into billgen.fin_3tar_lines', p_run_id;
    end if;

    -- 4) forward-only date safety
    select max(billdate)
    into v_latest_fin_billdate
    from billgen.fin_3tar_lines;

    if v_latest_fin_billdate is not null and v_billdate < v_latest_fin_billdate then
        raise exception 'Staging billdate % is earlier than latest final billdate %',
            v_billdate, v_latest_fin_billdate;
    end if;

    -- 5) Buddhist YYMM
    v_yyyymm :=
        lpad((((extract(year from v_billdate)::int + 543) % 100))::text, 2, '0') ||
        lpad(extract(month from v_billdate)::int::text, 2, '0');

    v_prefix := v_yyyymm || '-';

    -- 6) ensure control row exists
    insert into billgen.bill_seq_control (bill_type, yyyymm, last_seq)
    values ('3TAR', v_yyyymm, 0)
    on conflict (bill_type, yyyymm) do nothing;

    -- 7) lock sequence row and get starting seq
    select last_seq
    into v_start_seq
    from billgen.bill_seq_control
    where bill_type = '3TAR'
      and yyyymm = v_yyyymm
    for update;

    -- 8) how many generated bills needed
    select coalesce(max(chunk_no), 0)
    into v_chunk_count
    from (
        select ((row_number() over (order by billno, bcode, id) - 1) / 20) + 1 as chunk_no
        from billgen.stg_3tar_lines
        where run_id = p_run_id
    ) t;

    -- 9) protect 3-digit sequence overflow
    if v_start_seq + v_chunk_count > 999 then
        raise exception 'Sequence overflow for 3TAR %: current last_seq=%, new chunks=%, max=999',
            v_yyyymm, v_start_seq, v_chunk_count;
    end if;

    -- 10) insert final rows
    with ordered as (
        select
            s.*,
            row_number() over (
                order by s.billno, s.bcode, s.id
            ) as rn
        from billgen.stg_3tar_lines s
        where s.run_id = p_run_id
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

    -- 11) bump sequence control
    update billgen.bill_seq_control
    set
        last_seq = v_start_seq + v_chunk_count,
        updated_at = now()
    where bill_type = '3TAR'
      and yyyymm = v_yyyymm;
end;
$$;