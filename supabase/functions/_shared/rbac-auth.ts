import type { SupabaseClient } from "npm:@supabase/supabase-js@2";

export const ROLE_ADMIN = "admin";
export const BANK_STATEMENT_SYNC_PAGE = "bank_statement_sync";

export type RbacAuthResult =
  | { ok: true }
  | { ok: false; status: 401 | 403 | 500; message: string };

/**
 * Layer-2 page permission check for bank statement upload/report Edge Functions.
 * Mirrors `requirePermission` in kcw-v2 (`src/lib/auth/requirePermission.ts`).
 */
export async function requireBankStatementSyncPermission(
  admin: SupabaseClient,
  userId: string,
): Promise<RbacAuthResult> {
  const { data: userRoles, error: rolesError } = await admin
    .from("kcw_user_roles")
    .select("role_key")
    .eq("user_id", userId);

  if (rolesError) {
    console.error("kcw_user_roles lookup failed", rolesError);
    return { ok: false, status: 500, message: "Permission check failed" };
  }

  const roleKeys = (userRoles ?? []).map((r) => String(r.role_key));

  if (roleKeys.includes(ROLE_ADMIN)) {
    return { ok: true };
  }

  if (roleKeys.length === 0) {
    return {
      ok: false,
      status: 403,
      message: "Forbidden: no KCW role assigned",
    };
  }

  const { data: allowed, error: allowedError } = await admin
    .from("kcw_role_page_permissions")
    .select("page_key")
    .in("role_key", roleKeys)
    .eq("page_key", BANK_STATEMENT_SYNC_PAGE)
    .limit(1);

  if (allowedError) {
    console.error("kcw_role_page_permissions lookup failed", allowedError);
    return { ok: false, status: 500, message: "Permission check failed" };
  }

  if (!allowed?.length) {
    return {
      ok: false,
      status: 403,
      message: "Forbidden: bank statement sync not permitted",
    };
  }

  return { ok: true };
}
