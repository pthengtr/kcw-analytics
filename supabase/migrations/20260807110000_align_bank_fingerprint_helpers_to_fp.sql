# Align analytics bank fingerprint SQL helpers with canonical kcw-v2 bank.fp_*.
#
# Production already has BOTH:
#   - bank.fp_*              (kcw-v2 — source of truth for recomputed fingerprints)
#   - bank.build_transaction_fingerprint / extract_stable_detail / norm_fp_*
#     (analytics — older names; must produce identical hashes)
#
# This migration (re)defines fp_* to match kcw-v2 fingerprint.ts, then makes the
# analytics-named helpers thin wrappers so either API matches bank.fp_build_hash.
#
# Keys for stable detail (must match fingerprint.ts / fp_transaction_detail):
#   รายละเอียด, DESCRIPTION, PARTICULAR, NARRATION
# Display columns (รายการ / TIME) are intentionally excluded.

create extension if not exists pgcrypto;

create or replace function bank.fp_norm_text(val text)
returns text
language sql
immutable
as $$
  select regexp_replace(trim(upper(coalesce(replace(val, chr(160), ' '), ''))), '\s+', ' ', 'g');
$$;

create or replace function bank.fp_norm_money(val numeric)
returns text
language sql
immutable
as $$
  select case
    when val is null then ''
    else trim(to_char(round(val, 2), 'FM999999999990.00'))
  end;
$$;

create or replace function bank.fp_transaction_detail(raw jsonb)
returns text
language sql
immutable
as $$
  select coalesce(
    nullif(trim(raw->>'รายละเอียด'), ''),
    nullif(trim(raw->>'DESCRIPTION'), ''),
    nullif(trim(raw->>'PARTICULAR'), ''),
    nullif(trim(raw->>'NARRATION'), ''),
    ''
  );
$$;

create or replace function bank.fp_build_input(
  p_account_no text,
  p_txn_date date,
  p_direction text,
  p_amount numeric,
  p_balance_after numeric,
  p_bank_reference text,
  p_raw jsonb
)
returns text
language sql
immutable
as $$
  select concat_ws(
    '|',
    bank.fp_norm_text(p_account_no),
    p_txn_date::text,
    bank.fp_norm_money(p_amount),
    bank.fp_norm_text(p_direction),
    bank.fp_norm_text(bank.fp_transaction_detail(p_raw)),
    bank.fp_norm_text(coalesce(p_bank_reference, '')),
    bank.fp_norm_money(p_balance_after)
  );
$$;

create or replace function bank.fp_build_hash(
  p_account_no text,
  p_txn_date date,
  p_direction text,
  p_amount numeric,
  p_balance_after numeric,
  p_bank_reference text,
  p_raw jsonb
)
returns text
language sql
immutable
as $$
  select encode(
    digest(
      bank.fp_build_input(
        p_account_no, p_txn_date, p_direction, p_amount,
        p_balance_after, p_bank_reference, p_raw
      ),
      'sha256'
    ),
    'hex'
  );
$$;

create or replace function bank.extract_stable_detail(raw jsonb)
returns text
language sql
immutable
as $$
  select bank.fp_transaction_detail(raw);
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
  select bank.fp_build_hash(
    p_account_no,
    p_txn_date,
    p_direction,
    p_amount,
    p_balance_after,
    p_bank_reference,
    p_raw_json
  );
$$;

create or replace function bank.norm_fp_text(t text)
returns text
language sql
immutable
as $$
  select bank.fp_norm_text(t);
$$;

create or replace function bank.norm_fp_money(x numeric)
returns text
language sql
immutable
as $$
  select bank.fp_norm_money(x);
$$;

comment on function bank.fp_build_hash(text, date, text, numeric, numeric, text, jsonb) is
  'Canonical auto_v2 transaction fingerprint (kcw-v2). Do not diverge.';

comment on function bank.build_transaction_fingerprint(text, date, numeric, text, jsonb, text, numeric) is
  'Analytics-named wrapper of bank.fp_build_hash — identical hashes.';
