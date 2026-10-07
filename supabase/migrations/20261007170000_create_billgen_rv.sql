-- RV / 3RV bill generation. Separate from TAR.
-- Same day rules: 20 lines per bill, Buddhist YYMM prefix, forward-only, skip rerun.

create table if not exists billgen.stg_rv_lines (
    id bigint generated always as identity primary key,
    run_id text not null,
    billdate date not null,
    billno text not null,
    bcode text not null,
    detail text,
    qty numeric(18,4),
    mtp numeric(18,4),
    ui text,
    price numeric(18,4),
    amount numeric(18,4),
    last_cost numeric(18,6),
    loaded_at timestamptz default now()
);

create index if not exists idx_stg_rv_run on billgen.stg_rv_lines(run_id);
create index if not exists idx_stg_rv_billdate on billgen.stg_rv_lines(billdate);

create table if not exists billgen.stg_3rv_lines (
    id bigint generated always as identity primary key,
    run_id text not null,
    billdate date not null,
    billno text not null,
    bcode text not null,
    detail text,
    qty numeric(18,4),
    mtp numeric(18,4),
    ui text,
    price numeric(18,4),
    amount numeric(18,4),
    last_cost numeric(18,6),
    loaded_at timestamptz default now()
);

create index if not exists idx_stg_3rv_run on billgen.stg_3rv_lines(run_id);
create index if not exists idx_stg_3rv_billdate on billgen.stg_3rv_lines(billdate);

create table if not exists billgen.fin_rv_lines (
    id bigint generated always as identity primary key,
    run_id text not null,
    billdate date not null,
    billno text not null,
    new_billno text not null,
    detail text,
    bcode text not null,
    qty numeric(18,4),
    mtp numeric(18,4),
    ui text,
    price numeric(18,4),
    amount numeric(18,4),
    last_cost numeric(18,6),
    created_at timestamptz default now()
);

create index if not exists idx_fin_rv_run on billgen.fin_rv_lines(run_id);
create index if not exists idx_fin_rv_billdate on billgen.fin_rv_lines(billdate);
create index if not exists idx_fin_rv_new_billno on billgen.fin_rv_lines(new_billno);

create table if not exists billgen.fin_3rv_lines (
    id bigint generated always as identity primary key,
    run_id text not null,
    billdate date not null,
    billno text not null,
    new_billno text not null,
    detail text,
    bcode text not null,
    qty numeric(18,4),
    mtp numeric(18,4),
    ui text,
    price numeric(18,4),
    amount numeric(18,4),
    last_cost numeric(18,6),
    created_at timestamptz default now()
);

create index if not exists idx_fin_3rv_run on billgen.fin_3rv_lines(run_id);
create index if not exists idx_fin_3rv_billdate on billgen.fin_3rv_lines(billdate);
create index if not exists idx_fin_3rv_new_billno on billgen.fin_3rv_lines(new_billno);

create or replace function billgen.process_rv_day(p_run_id text, p_billdate date)
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
    from billgen.stg_rv_lines
    where run_id = p_run_id
      and billdate = p_billdate;

    if v_stg_count = 0 then
        raise notice 'No staging rows found for run_id=% and billdate=%', p_run_id, p_billdate;
        return;
    end if;

    if exists (
        select 1 from billgen.fin_rv_lines
        where run_id = p_run_id and billdate = p_billdate
    ) then
        raise exception 'Run % for billdate % already processed into fin_rv_lines', p_run_id, p_billdate;
    end if;

    select max(billdate) into v_latest_fin_billdate from billgen.fin_rv_lines;

    if v_latest_fin_billdate is not null and p_billdate < v_latest_fin_billdate then
        raise exception 'Staging billdate % is earlier than latest final billdate %',
            p_billdate, v_latest_fin_billdate;
    end if;

    v_yyyymm :=
        lpad((((extract(year from p_billdate)::int + 543) % 100))::text, 2, '0') ||
        lpad(extract(month from p_billdate)::int::text, 2, '0');
    v_prefix := 'RV' || v_yyyymm || '-';

    insert into billgen.bill_seq_control (bill_type, yyyymm, last_seq)
    values ('RV', v_yyyymm, 0)
    on conflict (bill_type, yyyymm) do nothing;

    select last_seq into v_start_seq
    from billgen.bill_seq_control
    where bill_type = 'RV' and yyyymm = v_yyyymm
    for update;

    select coalesce(max(chunk_no), 0) into v_chunk_count
    from (
        select ((row_number() over (order by billno, bcode, id) - 1) / 20) + 1 as chunk_no
        from billgen.stg_rv_lines
        where run_id = p_run_id and billdate = p_billdate
    ) t;

    if v_start_seq + v_chunk_count > 999 then
        raise exception 'Sequence overflow RV %', v_yyyymm;
    end if;

    with ordered as (
        select s.*, row_number() over (order by s.billno, s.bcode, s.id) as rn
        from billgen.stg_rv_lines s
        where s.run_id = p_run_id and s.billdate = p_billdate
    ),
    chunked as (
        select *, ((rn - 1) / 20) + 1 as chunk_no from ordered
    )
    insert into billgen.fin_rv_lines (
        run_id, billdate, billno, new_billno, detail,
        bcode, qty, mtp, ui, price, amount, last_cost
    )
    select
        run_id, billdate, billno,
        v_prefix || lpad((v_start_seq + chunk_no)::text, 3, '0'),
        detail, bcode, qty, mtp, ui, price, amount, last_cost
    from chunked
    order by rn;

    update billgen.bill_seq_control
    set last_seq = v_start_seq + v_chunk_count, updated_at = now()
    where bill_type = 'RV' and yyyymm = v_yyyymm;
end;
$$;

create or replace function billgen.process_3rv_day(p_run_id text, p_billdate date)
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
    from billgen.stg_3rv_lines
    where run_id = p_run_id
      and billdate = p_billdate;

    if v_stg_count = 0 then
        raise notice 'No staging rows found for run_id=% and billdate=%', p_run_id, p_billdate;
        return;
    end if;

    if exists (
        select 1 from billgen.fin_3rv_lines
        where run_id = p_run_id and billdate = p_billdate
    ) then
        raise exception 'Run % for billdate % already processed into fin_3rv_lines', p_run_id, p_billdate;
    end if;

    select max(billdate) into v_latest_fin_billdate from billgen.fin_3rv_lines;

    if v_latest_fin_billdate is not null and p_billdate < v_latest_fin_billdate then
        raise exception 'Staging billdate % is earlier than latest final billdate %',
            p_billdate, v_latest_fin_billdate;
    end if;

    v_yyyymm :=
        lpad((((extract(year from p_billdate)::int + 543) % 100))::text, 2, '0') ||
        lpad(extract(month from p_billdate)::int::text, 2, '0');
    v_prefix := '3RV' || v_yyyymm || '-';

    insert into billgen.bill_seq_control (bill_type, yyyymm, last_seq)
    values ('3RV', v_yyyymm, 0)
    on conflict (bill_type, yyyymm) do nothing;

    select last_seq into v_start_seq
    from billgen.bill_seq_control
    where bill_type = '3RV' and yyyymm = v_yyyymm
    for update;

    select coalesce(max(chunk_no), 0) into v_chunk_count
    from (
        select ((row_number() over (order by billno, bcode, id) - 1) / 20) + 1 as chunk_no
        from billgen.stg_3rv_lines
        where run_id = p_run_id and billdate = p_billdate
    ) t;

    if v_start_seq + v_chunk_count > 999 then
        raise exception 'Sequence overflow 3RV %', v_yyyymm;
    end if;

    with ordered as (
        select s.*, row_number() over (order by s.billno, s.bcode, s.id) as rn
        from billgen.stg_3rv_lines s
        where s.run_id = p_run_id and s.billdate = p_billdate
    ),
    chunked as (
        select *, ((rn - 1) / 20) + 1 as chunk_no from ordered
    )
    insert into billgen.fin_3rv_lines (
        run_id, billdate, billno, new_billno, detail,
        bcode, qty, mtp, ui, price, amount, last_cost
    )
    select
        run_id, billdate, billno,
        v_prefix || lpad((v_start_seq + chunk_no)::text, 3, '0'),
        detail, bcode, qty, mtp, ui, price, amount, last_cost
    from chunked
    order by rn;

    update billgen.bill_seq_control
    set last_seq = v_start_seq + v_chunk_count, updated_at = now()
    where bill_type = '3RV' and yyyymm = v_yyyymm;
end;
$$;

create or replace function billgen.process_all_rv_types_day(p_run_id text, p_billdate date)
returns void
language plpgsql
as $$
begin
    perform billgen.process_rv_day(p_run_id, p_billdate);
    perform billgen.process_3rv_day(p_run_id, p_billdate);
end;
$$;

create or replace function billgen.max_fin_rv_billdate()
returns date
language sql
stable
as $$
    select max(billdate) from (
        select billdate from billgen.fin_rv_lines
        union all
        select billdate from billgen.fin_3rv_lines
    ) t;
$$;

create or replace function billgen.is_rv_day_processed(p_billdate date)
returns boolean
language sql
stable
as $$
    select exists (
        select 1 from billgen.fin_rv_lines where billdate = p_billdate
        union all
        select 1 from billgen.fin_3rv_lines where billdate = p_billdate
    );
$$;

comment on function billgen.process_all_rv_types_day(text, date) is
    'Atomic RV/3RV day processing. Does not touch TAR tables.';
