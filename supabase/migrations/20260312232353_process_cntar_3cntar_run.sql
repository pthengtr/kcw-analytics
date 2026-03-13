-- =========================================================
-- UNMATCHED LOG TABLE
-- Used by CNTAR and 3CNTAR when PO cannot map to positive NEW_BILLNO
-- =========================================================

create table if not exists billgen.cntar_unmatched_log (
    id bigint generated always as identity primary key,
    bill_type text not null,
    run_id text not null,
    billdate date not null,
    billno text,
    po text,
    reason text not null,
    created_at timestamptz default now()
);

create index if not exists idx_cntar_unmatched_log_run_id
    on billgen.cntar_unmatched_log(run_id);

create index if not exists idx_cntar_unmatched_log_billdate
    on billgen.cntar_unmatched_log(billdate);

create index if not exists idx_cntar_unmatched_log_bill_type
    on billgen.cntar_unmatched_log(bill_type);


-- =========================================================
-- CNTAR: STG -> FIN BY DAY
-- SOFT-FAIL VERSION
-- - logs unmatched PO rows into billgen.cntar_unmatched_log
-- - skips unmatched rows
-- - groups matched rows by ref_new_billno
-- - max 20 lines per generated bill inside each ref group
-- - NEW_BILLNO format = CNTARYYMM-001
-- =========================================================

create or replace function billgen.process_cntar_day(p_run_id text, p_billdate date)
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
    v_unmatched_count integer;
    v_matched_count integer;
begin
    -- 1) rows must exist
    select count(*)
    into v_stg_count
    from billgen.stg_cntar_lines
    where run_id = p_run_id
      and billdate = p_billdate;

    if v_stg_count = 0 then
        raise exception 'No staging rows found for run_id=% and billdate=%', p_run_id, p_billdate;
    end if;

    -- 2) reject rerun
    if exists (
        select 1
        from billgen.fin_cntar_lines
        where run_id = p_run_id
          and billdate = p_billdate
    ) then
        raise exception 'Run % for billdate % already processed into fin_cntar_lines', p_run_id, p_billdate;
    end if;

    -- 3) forward-only date safety
    select max(billdate)
    into v_latest_fin_billdate
    from billgen.fin_cntar_lines;

    if v_latest_fin_billdate is not null and p_billdate < v_latest_fin_billdate then
        raise exception 'Staging billdate % is earlier than latest final billdate %',
            p_billdate, v_latest_fin_billdate;
    end if;

    -- 4) Buddhist YYMM
    v_yyyymm :=
        lpad((((extract(year from p_billdate)::int + 543) % 100))::text, 2, '0') ||
        lpad(extract(month from p_billdate)::int::text, 2, '0');

    v_prefix := 'CNTAR' || v_yyyymm || '-';

    -- 5) ensure control row exists
    insert into billgen.bill_seq_control (bill_type, yyyymm, last_seq)
    values ('CNTAR', v_yyyymm, 0)
    on conflict (bill_type, yyyymm) do nothing;

    -- 6) lock sequence row
    select last_seq
    into v_start_seq
    from billgen.bill_seq_control
    where bill_type = 'CNTAR'
      and yyyymm = v_yyyymm
    for update;

    -- 7) clean old unmatched logs for same run/day/type to avoid duplicate warnings on rerun after cleanup
    delete from billgen.cntar_unmatched_log
    where bill_type = 'CNTAR'
      and run_id = p_run_id
      and billdate = p_billdate;

    -- 8) log unmatched rows
    insert into billgen.cntar_unmatched_log (
        bill_type, run_id, billdate, billno, po, reason
    )
    select
        'CNTAR',
        s.run_id,
        s.billdate,
        s.billno,
        s.po,
        'PO not found in billgen.fin_tar_lines'
    from billgen.stg_cntar_lines s
    left join (
        select billno, min(new_billno) as new_billno
        from billgen.fin_tar_lines
        group by billno
    ) p
      on s.po = p.billno
    where s.run_id = p_run_id
      and s.billdate = p_billdate
      and p.new_billno is null;

    select count(*)
    into v_unmatched_count
    from billgen.cntar_unmatched_log
    where bill_type = 'CNTAR'
      and run_id = p_run_id
      and billdate = p_billdate;

    -- 9) count matched rows
    select count(*)
    into v_matched_count
    from billgen.stg_cntar_lines s
    join (
        select billno, min(new_billno) as new_billno
        from billgen.fin_tar_lines
        group by billno
    ) p
      on s.po = p.billno
    where s.run_id = p_run_id
      and s.billdate = p_billdate;

    -- if everything unmatched, just warn and stop cleanly
    if v_matched_count = 0 then
        raise notice 'CNTAR processed 0 rows, skipped % unmatched rows for run_id=% billdate=%',
            v_unmatched_count, p_run_id, p_billdate;
        return;
    end if;

    -- 10) count how many generated bills are needed
    with mapped as (
        select
            s.id,
            s.billdate,
            s.po,
            p.new_billno as ref_new_billno
        from billgen.stg_cntar_lines s
        join (
            select billno, min(new_billno) as new_billno
            from billgen.fin_tar_lines
            group by billno
        ) p
          on s.po = p.billno
        where s.run_id = p_run_id
          and s.billdate = p_billdate
    ),
    ordered as (
        select
            *,
            row_number() over (
                partition by ref_new_billno
                order by po, id
            ) as rn_in_ref
        from mapped
    ),
    chunked as (
        select
            *,
            ((rn_in_ref - 1) / 20) + 1 as chunk_in_ref
        from ordered
    ),
    ref_chunks as (
        select distinct
            ref_new_billno,
            chunk_in_ref
        from chunked
    ),
    numbered as (
        select
            ref_new_billno,
            chunk_in_ref,
            row_number() over (
                order by ref_new_billno, chunk_in_ref
            ) as global_chunk_no
        from ref_chunks
    )
    select count(*)
    into v_chunk_count
    from numbered;

    if v_start_seq + v_chunk_count > 999 then
        raise exception 'Sequence overflow for CNTAR %: current last_seq=%, new chunks=%, max=999',
            v_yyyymm, v_start_seq, v_chunk_count;
    end if;

    -- 11) insert final rows (matched rows only)
    with mapped as (
        select
            s.*,
            p.new_billno as ref_new_billno
        from billgen.stg_cntar_lines s
        join (
            select billno, min(new_billno) as new_billno
            from billgen.fin_tar_lines
            group by billno
        ) p
          on s.po = p.billno
        where s.run_id = p_run_id
          and s.billdate = p_billdate
    ),
    ordered as (
        select
            *,
            row_number() over (
                partition by ref_new_billno
                order by po, billno, bcode, id
            ) as rn_in_ref
        from mapped
    ),
    chunked as (
        select
            *,
            ((rn_in_ref - 1) / 20) + 1 as chunk_in_ref
        from ordered
    ),
    ref_chunks as (
        select distinct
            ref_new_billno,
            chunk_in_ref
        from chunked
    ),
    numbered as (
        select
            ref_new_billno,
            chunk_in_ref,
            row_number() over (
                order by ref_new_billno, chunk_in_ref
            ) as global_chunk_no
        from ref_chunks
    )
    insert into billgen.fin_cntar_lines (
        run_id, billdate, billno, po, new_billno, ref_new_billno,
        bcode, qty, mtp, ui, price, amount
    )
    select
        c.run_id,
        c.billdate,
        c.billno,
        c.po,
        v_prefix || lpad((v_start_seq + n.global_chunk_no)::text, 3, '0') as new_billno,
        c.ref_new_billno,
        c.bcode,
        c.qty,
        c.mtp,
        c.ui,
        c.price,
        c.amount
    from chunked c
    join numbered n
      on c.ref_new_billno = n.ref_new_billno
     and c.chunk_in_ref = n.chunk_in_ref
    order by c.ref_new_billno, c.chunk_in_ref, c.rn_in_ref;

    -- 12) bump sequence
    update billgen.bill_seq_control
    set
        last_seq = v_start_seq + v_chunk_count,
        updated_at = now()
    where bill_type = 'CNTAR'
      and yyyymm = v_yyyymm;

    raise notice 'CNTAR processed % matched rows, skipped % unmatched rows for run_id=% billdate=%',
        v_matched_count, v_unmatched_count, p_run_id, p_billdate;
end;
$$;


-- =========================================================
-- 3CNTAR: STG -> FIN BY DAY
-- SOFT-FAIL VERSION
-- - logs unmatched PO rows into billgen.cntar_unmatched_log
-- - skips unmatched rows
-- - groups matched rows by ref_new_billno
-- - max 20 lines per generated bill inside each ref group
-- - NEW_BILLNO format = 3CNTARYYMM-001
-- =========================================================

create or replace function billgen.process_3cntar_day(p_run_id text, p_billdate date)
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
    v_unmatched_count integer;
    v_matched_count integer;
begin
    -- 1) rows must exist
    select count(*)
    into v_stg_count
    from billgen.stg_3cntar_lines
    where run_id = p_run_id
      and billdate = p_billdate;

    if v_stg_count = 0 then
        raise exception 'No staging rows found for run_id=% and billdate=%', p_run_id, p_billdate;
    end if;

    -- 2) reject rerun
    if exists (
        select 1
        from billgen.fin_3cntar_lines
        where run_id = p_run_id
          and billdate = p_billdate
    ) then
        raise exception 'Run % for billdate % already processed into fin_3cntar_lines', p_run_id, p_billdate;
    end if;

    -- 3) forward-only date safety
    select max(billdate)
    into v_latest_fin_billdate
    from billgen.fin_3cntar_lines;

    if v_latest_fin_billdate is not null and p_billdate < v_latest_fin_billdate then
        raise exception 'Staging billdate % is earlier than latest final billdate %',
            p_billdate, v_latest_fin_billdate;
    end if;

    -- 4) Buddhist YYMM
    v_yyyymm :=
        lpad((((extract(year from p_billdate)::int + 543) % 100))::text, 2, '0') ||
        lpad(extract(month from p_billdate)::int::text, 2, '0');

    v_prefix := '3CNTAR' || v_yyyymm || '-';

    -- 5) ensure control row exists
    insert into billgen.bill_seq_control (bill_type, yyyymm, last_seq)
    values ('3CNTAR', v_yyyymm, 0)
    on conflict (bill_type, yyyymm) do nothing;

    -- 6) lock sequence row
    select last_seq
    into v_start_seq
    from billgen.bill_seq_control
    where bill_type = '3CNTAR'
      and yyyymm = v_yyyymm
    for update;

    -- 7) clean old unmatched logs for same run/day/type
    delete from billgen.cntar_unmatched_log
    where bill_type = '3CNTAR'
      and run_id = p_run_id
      and billdate = p_billdate;

    -- 8) log unmatched rows
    insert into billgen.cntar_unmatched_log (
        bill_type, run_id, billdate, billno, po, reason
    )
    select
        '3CNTAR',
        s.run_id,
        s.billdate,
        s.billno,
        s.po,
        'PO not found in billgen.fin_3tar_lines'
    from billgen.stg_3cntar_lines s
    left join (
        select billno, min(new_billno) as new_billno
        from billgen.fin_3tar_lines
        group by billno
    ) p
      on s.po = p.billno
    where s.run_id = p_run_id
      and s.billdate = p_billdate
      and p.new_billno is null;

    select count(*)
    into v_unmatched_count
    from billgen.cntar_unmatched_log
    where bill_type = '3CNTAR'
      and run_id = p_run_id
      and billdate = p_billdate;

    -- 9) count matched rows
    select count(*)
    into v_matched_count
    from billgen.stg_3cntar_lines s
    join (
        select billno, min(new_billno) as new_billno
        from billgen.fin_3tar_lines
        group by billno
    ) p
      on s.po = p.billno
    where s.run_id = p_run_id
      and s.billdate = p_billdate;

    if v_matched_count = 0 then
        raise notice '3CNTAR processed 0 rows, skipped % unmatched rows for run_id=% billdate=%',
            v_unmatched_count, p_run_id, p_billdate;
        return;
    end if;

    -- 10) count how many generated bills are needed
    with mapped as (
        select
            s.id,
            s.billdate,
            s.po,
            p.new_billno as ref_new_billno
        from billgen.stg_3cntar_lines s
        join (
            select billno, min(new_billno) as new_billno
            from billgen.fin_3tar_lines
            group by billno
        ) p
          on s.po = p.billno
        where s.run_id = p_run_id
          and s.billdate = p_billdate
    ),
    ordered as (
        select
            *,
            row_number() over (
                partition by ref_new_billno
                order by po, id
            ) as rn_in_ref
        from mapped
    ),
    chunked as (
        select
            *,
            ((rn_in_ref - 1) / 20) + 1 as chunk_in_ref
        from ordered
    ),
    ref_chunks as (
        select distinct
            ref_new_billno,
            chunk_in_ref
        from chunked
    ),
    numbered as (
        select
            ref_new_billno,
            chunk_in_ref,
            row_number() over (
                order by ref_new_billno, chunk_in_ref
            ) as global_chunk_no
        from ref_chunks
    )
    select count(*)
    into v_chunk_count
    from numbered;

    if v_start_seq + v_chunk_count > 999 then
        raise exception 'Sequence overflow for 3CNTAR %: current last_seq=%, new chunks=%, max=999',
            v_yyyymm, v_start_seq, v_chunk_count;
    end if;

    -- 11) insert final rows (matched rows only)
    with mapped as (
        select
            s.*,
            p.new_billno as ref_new_billno
        from billgen.stg_3cntar_lines s
        join (
            select billno, min(new_billno) as new_billno
            from billgen.fin_3tar_lines
            group by billno
        ) p
          on s.po = p.billno
        where s.run_id = p_run_id
          and s.billdate = p_billdate
    ),
    ordered as (
        select
            *,
            row_number() over (
                partition by ref_new_billno
                order by po, billno, bcode, id
            ) as rn_in_ref
        from mapped
    ),
    chunked as (
        select
            *,
            ((rn_in_ref - 1) / 20) + 1 as chunk_in_ref
        from ordered
    ),
    ref_chunks as (
        select distinct
            ref_new_billno,
            chunk_in_ref
        from chunked
    ),
    numbered as (
        select
            ref_new_billno,
            chunk_in_ref,
            row_number() over (
                order by ref_new_billno, chunk_in_ref
            ) as global_chunk_no
        from ref_chunks
    )
    insert into billgen.fin_3cntar_lines (
        run_id, billdate, billno, po, new_billno, ref_new_billno,
        bcode, qty, mtp, ui, price, amount
    )
    select
        c.run_id,
        c.billdate,
        c.billno,
        c.po,
        v_prefix || lpad((v_start_seq + n.global_chunk_no)::text, 3, '0') as new_billno,
        c.ref_new_billno,
        c.bcode,
        c.qty,
        c.mtp,
        c.ui,
        c.price,
        c.amount
    from chunked c
    join numbered n
      on c.ref_new_billno = n.ref_new_billno
     and c.chunk_in_ref = n.chunk_in_ref
    order by c.ref_new_billno, c.chunk_in_ref, c.rn_in_ref;

    -- 12) bump sequence
    update billgen.bill_seq_control
    set
        last_seq = v_start_seq + v_chunk_count,
        updated_at = now()
    where bill_type = '3CNTAR'
      and yyyymm = v_yyyymm;

    raise notice '3CNTAR processed % matched rows, skipped % unmatched rows for run_id=% billdate=%',
        v_matched_count, v_unmatched_count, p_run_id, p_billdate;
end;
$$;