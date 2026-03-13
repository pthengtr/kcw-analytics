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
    select count(*) into v_stg_count
    from billgen.stg_tar_lines
    where run_id = p_run_id
      and billdate = p_billdate;

    if v_stg_count = 0 then
        raise exception 'No staging rows found for run_id=% and billdate=%', p_run_id, p_billdate;
    end if;

    if exists (
        select 1
        from billgen.fin_tar_lines
        where run_id = p_run_id
          and billdate = p_billdate
    ) then
        raise exception 'Run % for billdate % already processed into fin_tar_lines', p_run_id, p_billdate;
    end if;

    select max(billdate)
    into v_latest_fin_billdate
    from billgen.fin_tar_lines;

    if v_latest_fin_billdate is not null and p_billdate < v_latest_fin_billdate then
        raise exception 'Staging billdate % is earlier than latest final billdate %',
            p_billdate, v_latest_fin_billdate;
    end if;

    v_yyyymm :=
        lpad((((extract(year from p_billdate)::int + 543) % 100))::text, 2, '0') ||
        lpad(extract(month from p_billdate)::int::text, 2, '0');

    v_prefix := 'TAR' || v_yyyymm || '-';

    insert into billgen.bill_seq_control (bill_type, yyyymm, last_seq)
    values ('TAR', v_yyyymm, 0)
    on conflict (bill_type, yyyymm) do nothing;

    select last_seq
    into v_start_seq
    from billgen.bill_seq_control
    where bill_type = 'TAR'
      and yyyymm = v_yyyymm
    for update;

    select coalesce(max(chunk_no), 0)
    into v_chunk_count
    from (
        select ((row_number() over (order by billno, bcode, id) - 1) / 20) + 1 as chunk_no
        from billgen.stg_tar_lines
        where run_id = p_run_id
          and billdate = p_billdate
    ) t;

    if v_start_seq + v_chunk_count > 999 then
        raise exception 'Sequence overflow TAR %', v_yyyymm;
    end if;

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
        v_prefix || lpad((v_start_seq + chunk_no)::text, 3, '0'),
        bcode,
        qty,
        mtp,
        ui,
        price,
        amount
    from chunked
    order by rn;

    update billgen.bill_seq_control
    set last_seq = v_start_seq + v_chunk_count,
        updated_at = now()
    where bill_type = 'TAR'
      and yyyymm = v_yyyymm;

end;
$$;

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
    select count(*) into v_stg_count
    from billgen.stg_3tar_lines
    where run_id = p_run_id
      and billdate = p_billdate;

    if v_stg_count = 0 then
        raise exception 'No staging rows found for run_id=% and billdate=%', p_run_id, p_billdate;
    end if;

    if exists (
        select 1
        from billgen.fin_3tar_lines
        where run_id = p_run_id
          and billdate = p_billdate
    ) then
        raise exception 'Run % for billdate % already processed into fin_3tar_lines', p_run_id, p_billdate;
    end if;

    select max(billdate)
    into v_latest_fin_billdate
    from billgen.fin_3tar_lines;

    if v_latest_fin_billdate is not null and p_billdate < v_latest_fin_billdate then
        raise exception 'Staging billdate % is earlier than latest final billdate %',
            p_billdate, v_latest_fin_billdate;
    end if;

    v_yyyymm :=
        lpad((((extract(year from p_billdate)::int + 543) % 100))::text, 2, '0') ||
        lpad(extract(month from p_billdate)::int::text, 2, '0');

    v_prefix := '3TAR' || v_yyyymm || '-';

    insert into billgen.bill_seq_control (bill_type, yyyymm, last_seq)
    values ('3TAR', v_yyyymm, 0)
    on conflict (bill_type, yyyymm) do nothing;

    select last_seq
    into v_start_seq
    from billgen.bill_seq_control
    where bill_type = '3TAR'
      and yyyymm = v_yyyymm
    for update;

    select coalesce(max(chunk_no), 0)
    into v_chunk_count
    from (
        select ((row_number() over (order by billno, bcode, id) - 1) / 20) + 1 as chunk_no
        from billgen.stg_3tar_lines
        where run_id = p_run_id
          and billdate = p_billdate
    ) t;

    if v_start_seq + v_chunk_count > 999 then
        raise exception 'Sequence overflow 3TAR %', v_yyyymm;
    end if;

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
        v_prefix || lpad((v_start_seq + chunk_no)::text, 3, '0'),
        bcode,
        qty,
        mtp,
        ui,
        price,
        amount
    from chunked
    order by rn;

    update billgen.bill_seq_control
    set last_seq = v_start_seq + v_chunk_count,
        updated_at = now()
    where bill_type = '3TAR'
      and yyyymm = v_yyyymm;

end;
$$;