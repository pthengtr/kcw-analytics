import io
import csv
import psycopg2
import pandas as pd
import numpy as np


def filter_last_year_from_latest(
    df: pd.DataFrame,
    date_col: str = "BILLDATE",
    *,
    years: int = 1,
    keep_invalid: bool = False,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Keep rows where `date_col` is within `years` years back from the latest date in that column.

    - Parses dates with pd.to_datetime(errors="coerce")
    - If keep_invalid=False (default): drops rows where date_col can't be parsed.
    - If keep_invalid=True: keeps invalid-date rows (NaT) in the output.

    Returns a filtered copy unless inplace=True (then returns the same df reference).
    """
    if date_col not in df.columns:
        raise KeyError(f"Column not found: {date_col}")

    out = df if inplace else df.copy()

    # Parse to datetime safely
    dt = pd.to_datetime(out[date_col], errors="coerce")

    latest = dt.max()
    if pd.isna(latest):
        # No valid dates at all
        return out if keep_invalid else out.iloc[0:0].copy()

    cutoff = latest - pd.DateOffset(years=years)

    mask_recent = dt >= cutoff
    if keep_invalid:
        mask = mask_recent | dt.isna()
    else:
        mask = mask_recent

    return out.loc[mask].copy()


# Example usage:
# df_1y = filter_last_year_from_latest(df, "BILLDATE")
# df_1y = filter_last_year_from_latest(df_pidet, "JOURDATE")


def _qident(name: str) -> str:
    """Quote an identifier safely for Postgres."""
    return '"' + name.replace('"', '""') + '"'


def _get_table_columns(conn, schema: str, table: str) -> list[str]:
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = %s AND table_name = %s
    ORDER BY ordinal_position
    """
    with conn.cursor() as cur:
        cur.execute(sql, (schema, table))
        rows = cur.fetchall()
    return [r[0] for r in rows]


def bulk_upsert_via_stage(
    conn,
    df: pd.DataFrame,
    *,
    target_table: str,
    schema: str = "public",
    stage_suffix: str = "_stage",
    pk_cols: list[str] = None,
    truncate_stage: bool = True,
) -> dict:
    """
    Bulk upsert df into target_table by:
      TRUNCATE stage
      COPY df -> stage
      INSERT target SELECT stage ON CONFLICT(pk) DO UPDATE
    Returns basic stats: rows_in_df, target_table, stage_table.
    """
    if pk_cols is None:
        pk_cols = ["ID"]

    if df is None or len(df) == 0:
        return {"rows_in_df": 0, "target_table": target_table, "stage_table": f"{target_table}{stage_suffix}"}

    stage_table = f"{target_table}{stage_suffix}"

    # Fetch canonical column order from the DB (target table)
    target_cols = _get_table_columns(conn, schema, target_table)
    if not target_cols:
        raise ValueError(f"Target table not found: {schema}.{target_table}")

    stage_cols = _get_table_columns(conn, schema, stage_table)
    if not stage_cols:
        raise ValueError(f"Stage table not found: {schema}.{stage_table}")

    if target_cols != stage_cols:
        raise ValueError(
            f"Target/stage column mismatch.\n"
            f"Target({len(target_cols)}): {target_cols}\n"
            f"Stage({len(stage_cols)}):  {stage_cols}"
        )

    missing = [c for c in target_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in target_cols]
    if missing:
        raise ValueError(f"DF is missing columns required by table: {missing}")
    if extra:
        # Not fatal, but usually indicates you loaded the wrong dataframe
        print(f"Warning: DF has extra columns not in table (they will be ignored): {extra}")

    # Reorder and drop extras; keep exactly table columns
    df2 = df[target_cols].copy()

    # Convert NaN -> None so COPY writes empty fields
    df2 = df2.where(pd.notnull(df2), None)

    fq_target = f"{_qident(schema)}.{_qident(target_table)}"
    fq_stage = f"{_qident(schema)}.{_qident(stage_table)}"

    cols_sql = ", ".join(_qident(c) for c in target_cols)
    pk_sql = ", ".join(_qident(c) for c in pk_cols)

    # Build SET clause excluding PK columns
    non_pk_cols = [c for c in target_cols if c not in set(pk_cols)]
    if not non_pk_cols:
        raise ValueError("No non-PK columns to update. Check pk_cols.")

    set_sql = ", ".join(f"{_qident(c)} = EXCLUDED.{_qident(c)}" for c in non_pk_cols)

    merge_sql = f"""
    INSERT INTO {fq_target} ({cols_sql})
    SELECT {cols_sql}
    FROM {fq_stage}
    ON CONFLICT ({pk_sql}) DO UPDATE
    SET {set_sql}
    """

    # Write DF to CSV buffer (no header), then COPY
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    for row in df2.itertuples(index=False, name=None):
        writer.writerow(["" if v is None else v for v in row])
    buf.seek(0)

    try:
        with conn.cursor() as cur:
            if truncate_stage:
                cur.execute(f"TRUNCATE TABLE {fq_stage};")

            cur.copy_expert(
                f"COPY {fq_stage} ({cols_sql}) FROM STDIN WITH (FORMAT CSV)",
                buf
            )

            cur.execute(merge_sql)

        conn.commit()
        return {
            "rows_in_df": len(df2),
            "target_table": f"{schema}.{target_table}",
            "stage_table": f"{schema}.{stage_table}",
        }

    except Exception:
        conn.rollback()
        raise


def refill_last_cost_from_icmas(
    data: dict,
    df: pd.DataFrame,
    *,
    icmas_key: str = "raw_hq_icmas_products.csv",
    bcode_col: str = "BCODE",
    last_cost_col: str = "LAST_COST",
    icmas_cost_col: str = "COSTNET",
) -> pd.DataFrame:
    """
    Refill LAST_COST when it is 0 or NaN using COSTNET from ICMAS.
    Includes BCODE cleanup to avoid merge mismatch.
    """

    result = df.copy()

    # --- BCODE CLEANUP (VERY IMPORTANT for KCW datasets) ---
    result[bcode_col] = (
        result[bcode_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    icmas = data[icmas_key][[bcode_col, icmas_cost_col]].copy()

    icmas[bcode_col] = (
        icmas[bcode_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --- numeric safety ---
    result[last_cost_col] = pd.to_numeric(result[last_cost_col], errors="coerce")
    icmas[icmas_cost_col] = pd.to_numeric(icmas[icmas_cost_col], errors="coerce")

    # --- merge COSTNET ---
    result = result.merge(
        icmas,
        on=bcode_col,
        how="left",
        suffixes=("", "_ICMAS")
    )

    # --- detect invalid LAST_COST ---
    mask_invalid = result[last_cost_col].isna() | (result[last_cost_col] == 0)

    # --- refill only invalid rows ---
    result.loc[mask_invalid, last_cost_col] = result.loc[
        mask_invalid, icmas_cost_col
    ]

    # --- drop helper column ---
    result = result.drop(columns=[icmas_cost_col])

    return result

def qc_unknown(df, label):
    total = len(df)
    unk = (df["COST_STATUS"] == "UNKNOWN").sum()
    print(f"[{label}] UNKNOWN: {unk:,} / {total:,} ({(unk/total*100 if total else 0):.2f}%)")

_BCODE_RE = r"^\d{8}$"

def _to_numeric_clean(series: pd.Series) -> pd.Series:
    """
    Convert common messy numeric strings to numbers.
    Handles: whitespace, NBSP, commas.
    Non-convertible -> NaN.
    """
    s = series.astype("string").str.strip()
    s = s.str.replace("\u00A0", " ", regex=False)  # non-breaking space
    s = s.str.replace(",", "", regex=False)        # "1,234.50" -> "1234.50"
    return pd.to_numeric(s, errors="coerce")


def add_sales_quality_flags(
    df: pd.DataFrame,
    *,
    bcode_col: str = "BCODE",
    price_col: str = "PRICE",
    amount_col: str = "AMOUNT",
    canceled_col: str = "CANCELED",
    add_row_id: bool = True,
) -> pd.DataFrame:
    """
    Adds Power-BI-friendly numeric columns + data-quality flags, without removing rows.

    Output columns added:
      - BCODE (trimmed)
      - PRICE_NUM, AMOUNT_NUM (numeric)
      - IS_VALID (bool)
      - INVALID_REASON (text, e.g. "BAD_BCODE|BAD_AMOUNT")
      - ROW_ID (optional)
    """
    out = df.copy()

    # --- BCODE clean + valid ---
    b = out[bcode_col].astype("string").str.strip()
    out[bcode_col] = b
    b_ok = b.fillna("").str.match(_BCODE_RE)

    # --- numeric clean (keep raw, add numeric columns) ---
    price_num = _to_numeric_clean(out[price_col])
    amount_num = _to_numeric_clean(out[amount_col])

    # Power BI hates inf/-inf
    price_ok = price_num.notna() & np.isfinite(price_num.to_numpy())
    amount_ok = amount_num.notna() & np.isfinite(amount_num.to_numpy())

    out[f"{price_col}_NUM"] = price_num
    out[f"{amount_col}_NUM"] = amount_num

    # --- canceled flag ---
    c = out[canceled_col].astype("string").str.strip().str.upper()
    canceled_ok = c != "Y"

    # --- overall validity ---
    out["IS_VALID"] = b_ok & price_ok & amount_ok & canceled_ok

    # --- reason text (can contain multiple reasons) ---
    reason = pd.Series("", index=out.index, dtype="string")

    def add_reason(mask, label):
        nonlocal reason
        reason = np.where(
            mask,
            np.where(reason == "", label, reason + "|" + label),
            reason
        )
        reason = pd.Series(reason, index=out.index, dtype="string")

    add_reason(~b_ok, "BAD_BCODE")
    add_reason(~price_ok, "BAD_PRICE")
    add_reason(~amount_ok, "BAD_AMOUNT")
    add_reason(~canceled_ok, "CANCELED")

    out["INVALID_REASON"] = reason.replace("", pd.NA)

    if add_row_id and "ROW_ID" not in out.columns:
        out["ROW_ID"] = np.arange(len(out), dtype=np.int64)

    return out

def enrich_sales_with_last_purchase_cost(
    sales: pd.DataFrame,
    purchases: pd.DataFrame,
    *,
    bcode_col: str = "BCODE",
    sale_date_col: str = "BILLDATE",
    purch_date_col: str = "BILLDATE",
    qty_col: str = "QTY",
    mtp_col: str = "MTP",
    amount_col: str = "AMOUNT",
    taxic_col: str = "TAXIC",  # ✅ NEW
    out_cost_col: str = "LAST_PURCHASE_COST",
    out_pdate_col: str = "LAST_PURCHASE_DATE",
    out_status_col: str = "COST_STATUS",
) -> pd.DataFrame:

    s = sales.copy()
    p = purchases.copy()

    s[bcode_col] = s[bcode_col].astype("string").str.strip()
    p[bcode_col] = p[bcode_col].astype("string").str.strip()

    s[sale_date_col] = pd.to_datetime(s[sale_date_col], errors="coerce")
    p[purch_date_col] = pd.to_datetime(p[purch_date_col], errors="coerce")

    # --- numeric safety ---
    p_qty = pd.to_numeric(p[qty_col], errors="coerce")
    p_mtp = pd.to_numeric(p[mtp_col], errors="coerce")
    p_amt = pd.to_numeric(p[amount_col], errors="coerce")

    # ✅ VAT handling for purchase amount when TAXIC == 'Y'
    if taxic_col in p.columns:
        taxic = p[taxic_col].astype("string").str.strip().str.upper()
        p_amt_net = np.where(taxic.eq("Y"), p_amt * (100.0 / 107.0), p_amt)
    else:
        p_amt_net = p_amt

    denom = p_qty * p_mtp
    p["_UNIT_COST"] = np.where(denom != 0, p_amt_net / denom, np.nan)

    # Keep only valid purchases
    p = p[p[purch_date_col].notna() & p["_UNIT_COST"].notna()].copy()

    # Create separate right-side date column
    p["_PURCH_DATE"] = p[purch_date_col]

    s["_POS"] = np.arange(len(s))
    s_valid = s[s[sale_date_col].notna()].copy()
    s_invalid = s[s[sale_date_col].isna()].copy()

    s_valid = s_valid.sort_values([sale_date_col, bcode_col, "_POS"], kind="mergesort")
    p = p.sort_values(["_PURCH_DATE", bcode_col], kind="mergesort")

    merged = pd.merge_asof(
        s_valid,
        p[[bcode_col, "_PURCH_DATE", "_UNIT_COST"]],
        left_on=sale_date_col,
        right_on="_PURCH_DATE",
        by=bcode_col,
        direction="backward",
        allow_exact_matches=True,
    )

    merged.rename(columns={"_UNIT_COST": out_cost_col, "_PURCH_DATE": out_pdate_col}, inplace=True)
    merged[out_status_col] = np.where(merged[out_cost_col].notna(), "OK", "UNKNOWN")

    if len(s_invalid) > 0:
        s_invalid[out_cost_col] = np.nan
        s_invalid[out_pdate_col] = pd.NaT
        s_invalid[out_status_col] = "UNKNOWN"
        merged = pd.concat([merged, s_invalid], ignore_index=False)

    merged = merged.sort_values("_POS", kind="mergesort").drop(columns=["_POS"])
    return merged

def _clean_str(s: pd.Series) -> pd.Series:
    s = s.astype("string")

    # remove non-breaking space
    s = s.str.replace("\u00A0", " ", regex=False)

    # trim + normalize case
    s = s.str.strip().str.upper()

    # remove common fake-null strings
    s = s.replace({
        "": pd.NA,
        "NAN": pd.NA,
        "NONE": pd.NA,
        "NULL": pd.NA
    })

    return s



def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

# ------------------------
# DimDate
# ------------------------
def build_dim_date_from_sales(sales_all: pd.DataFrame, *, date_col: str = "BILLDATE") -> pd.DataFrame:
    d = _to_dt(sales_all[date_col]).dropna().dt.normalize()
    if d.empty:
        return pd.DataFrame(columns=["Date", "DateKey", "Year", "Month", "Day", "YearMonth", "Quarter", "WeekNum"])

    date_range = pd.date_range(d.min(), d.max(), freq="D")
    dim = pd.DataFrame({"Date": date_range})
    dim["DateKey"] = dim["Date"].dt.strftime("%Y%m%d").astype(int)
    dim["Year"] = dim["Date"].dt.year
    dim["Month"] = dim["Date"].dt.month
    dim["Day"] = dim["Date"].dt.day
    dim["YearMonth"] = dim["Date"].dt.strftime("%Y-%m")
    dim["Quarter"] = dim["Date"].dt.quarter
    dim["WeekNum"] = dim["Date"].dt.isocalendar().week.astype(int)
    return dim

# ------------------------
# DimBranch
# ------------------------
def build_dim_branch(sales_all: pd.DataFrame, *, branch_col: str = "BRANCH") -> pd.DataFrame:
    dim = pd.DataFrame({"BRANCH": _clean_str(sales_all[branch_col])}).dropna()

    dim = (
        dim[dim["BRANCH"] != ""]
        .drop_duplicates()
        .sort_values("BRANCH")
        .reset_index(drop=True)
    )

    # Keep your existing key
    dim["BranchKey"] = dim["BRANCH"]

    # 🔥 Map branch → UUID
    BRANCH_UUID_MAP = {
        "HQ": "c93efb5f-07c9-4229-b6b3-568ce1c0a9ab",
        "SYP": "4975a5a1-90e6-443a-9921-c6c637f4631c",
    }

    dim["branch_uuid"] = dim["BRANCH"].map(BRANCH_UUID_MAP)

    return dim


# ------------------------
# DimProduct (BCODE)
# ------------------------
def build_dim_product(
    sales_all: pd.DataFrame,
    *,
    bcode_col: str = "BCODE",
    detail_col: str = "DETAIL",
    ui_col: str = "UI",
    last_seen_date_col: str = "BILLDATE",
) -> pd.DataFrame:
    df = sales_all.copy()
    df[bcode_col] = _clean_str(df[bcode_col])
    df[detail_col] = _clean_str(df.get(detail_col, ""))
    df[ui_col] = _clean_str(df.get(ui_col, ""))
    df[last_seen_date_col] = _to_dt(df[last_seen_date_col])

    df = df[df[bcode_col].notna() & (df[bcode_col] != "")]
    df = df.sort_values([bcode_col, last_seen_date_col], kind="mergesort")
    last = df.groupby(bcode_col, sort=False).tail(1)

    dim = pd.DataFrame({
        "BCODE": last[bcode_col],
        "DETAIL": last.get(detail_col, pd.Series([pd.NA]*len(last))),
        "UI": last.get(ui_col, pd.Series([pd.NA]*len(last))),
        "LastSeenDate": last[last_seen_date_col].dt.normalize(),
    }).reset_index(drop=True)

    dim["ProductKey"] = dim["BCODE"]
    # add CATEGORY_CODE (first 2 digits) for easy relationship too
    dim["CATEGORY_CODE"] = dim["BCODE"].astype("string").str.slice(0, 2)
    return dim

# ------------------------
# DimCategory (first 2 digits of BCODE)
# ------------------------
def build_dim_category_from_bcode(
    sales_all: pd.DataFrame,
    *,
    bcode_col: str = "BCODE",
) -> pd.DataFrame:
    b = _clean_str(sales_all[bcode_col])
    cat = b.dropna().str.slice(0, 2)
    # keep only exactly 2 digits
    cat = cat[cat.str.match(r"^\d{2}$", na=False)]

    dim = pd.DataFrame({"CATEGORY_CODE": cat}).drop_duplicates().sort_values("CATEGORY_CODE").reset_index(drop=True)
    dim["CategoryKey"] = dim["CATEGORY_CODE"]
    return dim

# ------------------------
# DimAccount
# ------------------------
def build_dim_account(
    sales_all: pd.DataFrame,
    *,
    customer_col: str = "ACCTNO",
    supplier_col: str = "ACCT_NO",
) -> pd.DataFrame:
    # clean series (or empty)
    c = _clean_str(sales_all[customer_col]) if customer_col in sales_all.columns else pd.Series([], dtype="string")
    s = _clean_str(sales_all[supplier_col]) if supplier_col in sales_all.columns else pd.Series([], dtype="string")

    # unique set of accounts from both columns
    all_keys = pd.concat([c, s], ignore_index=True).dropna()
    all_keys = all_keys[all_keys != ""].drop_duplicates().sort_values().reset_index(drop=True)

    dim = pd.DataFrame({"AccountKey": all_keys})

    # role flags
    customer_set = set(c.dropna()[c.dropna() != ""].unique())
    supplier_set = set(s.dropna()[s.dropna() != ""].unique())

    dim["IsCustomer"] = dim["AccountKey"].isin(customer_set)
    dim["IsSupplier"] = dim["AccountKey"].isin(supplier_set)

    # optional: a readable label
    dim["AccountRole"] = np.where(dim["IsCustomer"] & dim["IsSupplier"], "BOTH",
                          np.where(dim["IsCustomer"], "CUSTOMER",
                          np.where(dim["IsSupplier"], "SUPPLIER", "UNKNOWN")))

    return dim


# ------------------------
# DimSupplier (ACCT_NO)
# ------------------------
def build_dim_supplier(sales_all: pd.DataFrame, *, supplier_col="ACCT_NO") -> pd.DataFrame:
    s = _clean_str(sales_all[supplier_col]) if supplier_col in sales_all.columns else pd.Series([pd.NA]*len(sales_all), dtype="string")
    dim = pd.DataFrame({"SupplierKey": s}).dropna()
    dim = dim[dim["SupplierKey"] != ""].drop_duplicates(subset=["SupplierKey"]).sort_values("SupplierKey").reset_index(drop=True)
    dim["SUPPLIER_ACCT_NO"] = dim["SupplierKey"]
    return dim

# -----------------------------
# DIM BILLTYPE (from BILLNO)
# -----------------------------
KNOWN_TYPES = ["TFV", "TAD", "TAR", "TR", "TD", "TF", "CN", "DN"]

def build_dim_billtype(sales_all):
    dim = pd.DataFrame({"BILLTYPE_STD": _clean_str(sales_all["BILLTYPE_STD"]).str.upper()})
    dim = dim.drop_duplicates().sort_values("BILLTYPE_STD").reset_index(drop=True)
    dim["BillTypeKey"] = dim["BILLTYPE_STD"]
    return dim

# ------------------------
# Wrapper
# ------------------------
def build_all_dims(sales_all):
    return {
        "dim_date": build_dim_date_from_sales(sales_all),
        "dim_product": build_dim_product(sales_all),
        "dim_category": build_dim_category_from_bcode(sales_all),
        "dim_account": build_dim_account(sales_all),
        "dim_branch": build_dim_branch(sales_all),
        "dim_billtype": build_dim_billtype(sales_all),
    }
