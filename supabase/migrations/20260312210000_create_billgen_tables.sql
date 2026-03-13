-- =========================================================
-- BILL GENERATION TEST STRUCTURE
-- =========================================================

create schema if not exists billgen;

-- =========================================================
-- STAGING TABLES
-- =========================================================

create table if not exists billgen.stg_tar_lines (
    id bigint generated always as identity primary key,
    run_id text not null,
    billdate date not null,
    billno text not null,
    bcode text not null,
    qty numeric(18,4),
    mtp numeric(18,4),
    ui text,
    price numeric(18,4),
    amount numeric(18,4),
    loaded_at timestamptz default now()
);

create index if not exists idx_stg_tar_run
    on billgen.stg_tar_lines(run_id);

create index if not exists idx_stg_tar_billno
    on billgen.stg_tar_lines(billno);


create table if not exists billgen.stg_3tar_lines (
    id bigint generated always as identity primary key,
    run_id text not null,
    billdate date not null,
    billno text not null,
    bcode text not null,
    qty numeric(18,4),
    mtp numeric(18,4),
    ui text,
    price numeric(18,4),
    amount numeric(18,4),
    loaded_at timestamptz default now()
);

create index if not exists idx_stg_3tar_run
    on billgen.stg_3tar_lines(run_id);

create index if not exists idx_stg_3tar_billno
    on billgen.stg_3tar_lines(billno);


create table if not exists billgen.stg_cntar_lines (
    id bigint generated always as identity primary key,
    run_id text not null,
    billdate date not null,
    billno text not null,
    bcode text not null,
    qty numeric(18,4),
    mtp numeric(18,4),
    ui text,
    price numeric(18,4),
    amount numeric(18,4),
    po text,
    loaded_at timestamptz default now()
);

create index if not exists idx_stg_cntar_run
    on billgen.stg_cntar_lines(run_id);

create index if not exists idx_stg_cntar_billno
    on billgen.stg_cntar_lines(billno);

create index if not exists idx_stg_cntar_po
    on billgen.stg_cntar_lines(po);


create table if not exists billgen.stg_3cntar_lines (
    id bigint generated always as identity primary key,
    run_id text not null,
    billdate date not null,
    billno text not null,
    bcode text not null,
    qty numeric(18,4),
    mtp numeric(18,4),
    ui text,
    price numeric(18,4),
    amount numeric(18,4),
    po text,
    loaded_at timestamptz default now()
);

create index if not exists idx_stg_3cntar_run
    on billgen.stg_3cntar_lines(run_id);

create index if not exists idx_stg_3cntar_billno
    on billgen.stg_3cntar_lines(billno);

create index if not exists idx_stg_3cntar_po
    on billgen.stg_3cntar_lines(po);

-- =========================================================
-- FINAL TABLES
-- =========================================================

create table if not exists billgen.fin_tar_lines (
    id bigint generated always as identity primary key,
    run_id text not null,
    billdate date not null,
    billno text not null,
    new_billno text not null,
    bcode text not null,
    qty numeric(18,4),
    mtp numeric(18,4),
    ui text,
    price numeric(18,4),
    amount numeric(18,4),
    created_at timestamptz default now()
);

create index if not exists idx_fin_tar_run
    on billgen.fin_tar_lines(run_id);

create index if not exists idx_fin_tar_billno
    on billgen.fin_tar_lines(billno);

create index if not exists idx_fin_tar_new_billno
    on billgen.fin_tar_lines(new_billno);


create table if not exists billgen.fin_3tar_lines (
    id bigint generated always as identity primary key,
    run_id text not null,
    billdate date not null,
    billno text not null,
    new_billno text not null,
    bcode text not null,
    qty numeric(18,4),
    mtp numeric(18,4),
    ui text,
    price numeric(18,4),
    amount numeric(18,4),
    created_at timestamptz default now()
);

create index if not exists idx_fin_3tar_run
    on billgen.fin_3tar_lines(run_id);

create index if not exists idx_fin_3tar_billno
    on billgen.fin_3tar_lines(billno);

create index if not exists idx_fin_3tar_new_billno
    on billgen.fin_3tar_lines(new_billno);


create table if not exists billgen.fin_cntar_lines (
    id bigint generated always as identity primary key,
    run_id text not null,
    billdate date not null,
    billno text not null,
    po text,
    new_billno text not null,
    ref_new_billno text,
    bcode text not null,
    qty numeric(18,4),
    mtp numeric(18,4),
    ui text,
    price numeric(18,4),
    amount numeric(18,4),
    created_at timestamptz default now()
);

create index if not exists idx_fin_cntar_run
    on billgen.fin_cntar_lines(run_id);

create index if not exists idx_fin_cntar_billno
    on billgen.fin_cntar_lines(billno);

create index if not exists idx_fin_cntar_po
    on billgen.fin_cntar_lines(po);

create index if not exists idx_fin_cntar_new_billno
    on billgen.fin_cntar_lines(new_billno);


create table if not exists billgen.fin_3cntar_lines (
    id bigint generated always as identity primary key,
    run_id text not null,
    billdate date not null,
    billno text not null,
    po text,
    new_billno text not null,
    ref_new_billno text,
    bcode text not null,
    qty numeric(18,4),
    mtp numeric(18,4),
    ui text,
    price numeric(18,4),
    amount numeric(18,4),
    created_at timestamptz default now()
);

create index if not exists idx_fin_3cntar_run
    on billgen.fin_3cntar_lines(run_id);

create index if not exists idx_fin_3cntar_billno
    on billgen.fin_3cntar_lines(billno);

create index if not exists idx_fin_3cntar_po
    on billgen.fin_3cntar_lines(po);

create index if not exists idx_fin_3cntar_new_billno
    on billgen.fin_3cntar_lines(new_billno);