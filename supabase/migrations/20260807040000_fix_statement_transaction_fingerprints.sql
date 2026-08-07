-- Fix bank statement duplicate detection (parser_version auto_v2).
-- Idempotent: safe if bank_statement_fingerprint_v2_cleanup already ran on production.

create extension if not exists pgcrypto;

create or replace function bank.norm_fp_text(t text)
returns text
language sql
immutable
as $$
  select case
    when t is null or btrim(t) = '' then ''
    else upper(regexp_replace(replace(t, chr(160), ' '), '\s+', ' ', 'g'))
  end;
$$;

create or replace function bank.norm_fp_money(x numeric)
returns text
language sql
immutable
as $$
  select case
    when x is null then ''
    else to_char(trunc(x::numeric + 0.005, 2), 'FM999999990.00')
  end;
$$;

create or replace function bank.extract_stable_detail(raw jsonb)
returns text
language plpgsql
immutable
as $$
declare
  keys text[] := array['รายละเอียด', 'DESCRIPTION', 'DETAIL', 'PARTICULAR'];
  k text;
  v text;
begin
  foreach k in array keys loop
    v := raw ->> k;
    if v is not null and btrim(v) <> '' then
      return btrim(v);
    end if;
  end loop;
  return '';
end;
$$;

create or replace function bank.build_transaction_fingerprint(
  p_account_no text,
  p_txn_date date,
  p_amount numeric,
  p_direction text,
  p_raw_json jsonb,
  p_bank_reference text,
  p_balance_after numeric
)
returns text
language sql
immutable
as $$
  select encode(
    digest(
      bank.norm_fp_text(p_account_no) || '|' ||
      p_txn_date::text || '|' ||
      bank.norm_fp_money(p_amount) || '|' ||
      bank.norm_fp_text(p_direction) || '|' ||
      bank.norm_fp_text(bank.extract_stable_detail(p_raw_json)) || '|' ||
      bank.norm_fp_text(p_bank_reference) || '|' ||
      bank.norm_fp_money(p_balance_after),
      'sha256'
    ),
    'hex'
  );
$$;

-- Delete newer duplicate copies from the cumulative May 0393 import.
with dup_file as (
  select id
  from bank.statement_import_files
  where original_filename = 'KBANK0393_31_5_69.xlsx'
),
to_delete as (
  select n.id
  from bank.statement_lines n
  join dup_file df on n.source_file_id = df.id
  where exists (
    select 1
    from bank.statement_lines o
    where o.source_file_id <> df.id
      and o.account_no = n.account_no
      and o.txn_date = n.txn_date
      and o.direction = n.direction
      and o.amount = n.amount
      and o.balance_after = n.balance_after
      and bank.extract_stable_detail(o.raw_json) = bank.extract_stable_detail(n.raw_json)
  )
)
delete from bank.statement_lines sl
where sl.id in (select id from to_delete);

-- Correct import metadata for the cumulative file (no-op if already updated).
update bank.statement_import_files
set inserted_count = 13,
    duplicate_count = 76,
    raw_metadata = coalesce(raw_metadata, '{}'::jsonb) || '{"parser_version":"auto_v2","duplicate_cleanup":"20260807040000"}'::jsonb
where original_filename = 'KBANK0393_31_5_69.xlsx'
  and (inserted_count <> 13 or duplicate_count <> 76);

-- Backfill fingerprints only where they differ from the canonical algorithm.
update bank.statement_lines sl
set transaction_fingerprint = bank.build_transaction_fingerprint(
  sl.account_no,
  sl.txn_date,
  sl.amount,
  sl.direction,
  sl.raw_json,
  sl.bank_reference,
  sl.balance_after
)
where sl.transaction_fingerprint <> bank.build_transaction_fingerprint(
  sl.account_no,
  sl.txn_date,
  sl.amount,
  sl.direction,
  sl.raw_json,
  sl.bank_reference,
  sl.balance_after
);
