-- Private storage for web-uploaded bank statement Excel files.
-- Edge Function import-bank-statement writes with service_role.
-- Authenticated kcw_admin users may read (download) objects.

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'bank-statements',
  'bank-statements',
  false,
  15728640, -- 15 MiB
  array[
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel',
    'application/vnd.ms-excel.sheet.macroEnabled.12',
    'application/octet-stream'
  ]
)
on conflict (id) do update
set
  public = excluded.public,
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

-- Drop + recreate policies idempotently
drop policy if exists "admin_select_bank_statements" on storage.objects;
drop policy if exists "service_role_all_bank_statements" on storage.objects;

create policy "admin_select_bank_statements"
on storage.objects
for select
to authenticated
using (
  bucket_id = 'bank-statements'
  and exists (
    select 1
    from public.kcw_admin a
    where a.user_id = (auth.jwt() ->> 'email')
  )
);

-- service_role bypasses RLS; no extra policy required for Edge Function uploads.
-- Keep an explicit insert/update policy for authenticated admins as a fallback
-- if a future UI uploads directly to Storage then invokes the function.
drop policy if exists "admin_insert_bank_statements" on storage.objects;
create policy "admin_insert_bank_statements"
on storage.objects
for insert
to authenticated
with check (
  bucket_id = 'bank-statements'
  and exists (
    select 1
    from public.kcw_admin a
    where a.user_id = (auth.jwt() ->> 'email')
  )
);

drop policy if exists "admin_update_bank_statements" on storage.objects;
create policy "admin_update_bank_statements"
on storage.objects
for update
to authenticated
using (
  bucket_id = 'bank-statements'
  and exists (
    select 1
    from public.kcw_admin a
    where a.user_id = (auth.jwt() ->> 'email')
  )
)
with check (
  bucket_id = 'bank-statements'
  and exists (
    select 1
    from public.kcw_admin a
    where a.user_id = (auth.jwt() ->> 'email')
  )
);
