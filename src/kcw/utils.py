import pandas as pd

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df

def _clean_bcode(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

def _drop_invalid_bcode(df: pd.DataFrame, bcode_col: str = "BCODE") -> pd.DataFrame:
    if bcode_col not in df.columns:
        return df.iloc[0:0].copy()

    df = df.copy()

    # 1) drop true missing values first (keeps <NA> as missing)
    df = df.dropna(subset=[bcode_col]).copy()

    # 2) now clean as string
    df[bcode_col] = (
        df[bcode_col]
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    # 3) drop blanks and common "missing" strings produced by astype(str)
    df = df[df[bcode_col] != ""].copy()
    df = df[~df[bcode_col].str.upper().isin(["<NA>", "NA", "NAN", "NONE", "NULL"])].copy()

    return df

def get_nonvat_sales_lines(
    data: dict,
    year: int,
    source: str,
    month: int | None = None,
):
    source = source.lower()
    if source not in ("hq", "syp"):
        raise ValueError("source must be 'hq' or 'syp'")

    df = data[f"raw_{source}_sidet_sales_lines.csv"].copy()
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    df = _drop_invalid_bcode(df, "BCODE")

    df["BILLDATE"] = pd.to_datetime(df["BILLDATE"], errors="coerce")
    df = df.dropna(subset=["BILLDATE"]).copy()

    sales_isvat = df["ISVAT"].astype(str).str.strip().str.upper()
    mask_nonvat = sales_isvat.eq("N")  # strict non-vat only
    mask_not_canceled = df["CANCELED"].astype(str).str.strip().str.upper() != "Y"
    mask_year = df["BILLDATE"].dt.year == year

    mask = mask_nonvat & mask_not_canceled & mask_year

    if month is not None:
        if not 1 <= month <= 12:
            raise ValueError("month must be 1–12")
        mask &= df["BILLDATE"].dt.month == month

    return df.loc[mask].copy()

def get_vat_sales_lines(
    data: dict,
    year: int,
    source: str,
    month: int | None = None,
):
    """
    Return VAT (ISVAT=Y), non-canceled sales lines for a given year
    and optional month.
    """

    source = source.lower()
    if source not in ("hq", "syp"):
        raise ValueError("source must be 'hq' or 'syp'")

    key = f"raw_{source}_sidet_sales_lines.csv"
    df = data[key].copy()

    # clean columns
    df = _clean_columns(df)
    df = _drop_invalid_bcode(df, "BCODE")

    # parse date
    df["BILLDATE"] = pd.to_datetime(df["BILLDATE"], errors="coerce")

    # base filters
    mask_vat = df["ISVAT"].astype(str).str.strip().str.upper() == "Y"
    mask_not_canceled = df["CANCELED"].astype(str).str.strip().str.upper() != "Y"
    mask_year = df["BILLDATE"].dt.year == year

    mask = mask_vat & mask_not_canceled & mask_year

    if month is not None:
        if not 1 <= month <= 12:
            raise ValueError("month must be 1–12")
        mask = mask & (df["BILLDATE"].dt.month == month)

    return df.loc[mask].copy()

def get_vat_purchase_lines(
    data: dict,
    year: int,
    source: str,
    month: int | None = None,
):
    """
    Return VAT (ISVAT=Y) purchase lines for a given year
    and optional month.
    """

    source = source.lower()
    if source not in ("hq", "syp"):
        raise ValueError("source must be 'hq' or 'syp'")

    key = f"raw_{source}_pidet_purchase_lines.csv"
    df = data[key].copy()

    # clean columns
    df = _clean_columns(df)
    df = _drop_invalid_bcode(df, "BCODE")

    # parse date
    df["BILLDATE"] = pd.to_datetime(df["BILLDATE"], errors="coerce")

    # base filters
    mask_vat = df["ISVAT"].astype(str).str.strip().str.upper() == "Y"
    mask_year = df["BILLDATE"].dt.year == year

    mask = mask_vat & mask_year

    if month is not None:
        if not 1 <= month <= 12:
            raise ValueError("month must be 1–12")
        mask = mask & (df["BILLDATE"].dt.month == month)

    return df.loc[mask].copy()


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df

def _clean_bcode(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

def get_vat_sales_lines_last_purchase_nonvat(
    data: dict,
    year: int,
    source: str,
    *,
    sales_key_template: str = "raw_{source}_sidet_sales_lines.csv",
    purchase_key_template: str = "raw_hq_pidet_purchase_lines.csv",
    date_col: str = "BILLDATE",
    bcode_col: str = "BCODE",
    sales_isvat_col: str = "ISVAT",
    sales_canceled_col: str = "CANCELED",
    purchase_isvat_col: str = "ISVAT",
) -> pd.DataFrame:
    """
    Return VAT sales lines (ISVAT=Y, not canceled) in `year` whose latest purchase
    on/before the sale date (per BCODE) was NON-VAT (purchase ISVAT != 'Y' or missing).
    """

    source = source.lower()
    if source not in ("hq", "syp"):
        raise ValueError("source must be 'hq' or 'syp'")

    sales_key = sales_key_template.format(source=source)
    purchase_key = purchase_key_template.format(source=source)

    sales = _clean_cols(data[sales_key])
    pidet = _clean_cols(data[purchase_key])

    sales = _drop_invalid_bcode(sales, "BCODE")
    pidet = _drop_invalid_bcode(pidet, "BCODE")

    # --- Clean BCODE ---
    sales[bcode_col] = _clean_bcode(sales[bcode_col])
    pidet[bcode_col] = _clean_bcode(pidet[bcode_col])

    # --- Parse dates ---
    sales[date_col] = pd.to_datetime(sales[date_col], errors="coerce")
    pidet[date_col] = pd.to_datetime(pidet[date_col], errors="coerce")

    # --- Drop missing keys ---
    sales = sales.dropna(subset=[date_col, bcode_col]).copy()
    pidet = pidet.dropna(subset=[date_col, bcode_col]).copy()
    sales = sales[sales[bcode_col] != ""].copy()
    pidet = pidet[pidet[bcode_col] != ""].copy()

    # --- Filter SALES: VAT + not canceled + year ---
    mask_sales_vat = sales[sales_isvat_col].astype(str).str.strip().str.upper() == "Y"
    mask_sales_not_canceled = sales[sales_canceled_col].astype(str).str.strip().str.upper() != "Y"
    mask_sales_year = sales[date_col].dt.year == year

    vat_sales_lines = sales[mask_sales_vat & mask_sales_not_canceled & mask_sales_year].copy()

    # --- Clean VAT flag in PIDET ---
    pidet[purchase_isvat_col] = pidet[purchase_isvat_col].astype(str).str.strip().str.upper()

    pidet_key = pidet[[bcode_col, date_col, purchase_isvat_col]].copy()
    pidet_key = pidet_key.rename(columns={purchase_isvat_col: "LAST_PURCHASE_ISVAT"})

    # --- Sort for merge_asof ---
    vat_sales_lines = vat_sales_lines.sort_values([date_col, bcode_col], kind="mergesort").reset_index(drop=True)
    pidet_key = pidet_key.sort_values([date_col, bcode_col], kind="mergesort").reset_index(drop=True)

    merged = pd.merge_asof(
        vat_sales_lines,
        pidet_key,
        left_on=date_col,
        right_on=date_col,
        by=bcode_col,
        direction="backward",
        allow_exact_matches=True
    )

    # --- Keep VAT sales where last purchase is NON-VAT (or missing) ---
    vat_sales_last_purchase_nonvat = merged[
        merged["LAST_PURCHASE_ISVAT"].fillna("").ne("Y")
    ].copy()

    return vat_sales_last_purchase_nonvat


def get_nonvat_sales_lines_last_purchase_vat(
    data: dict,
    year: int,
    source: str,
    *,
    sales_key_template: str = "raw_{source}_sidet_sales_lines.csv",
    purchase_key_template: str = "raw_hq_pidet_purchase_lines.csv",
    date_col: str = "BILLDATE",
    bcode_col: str = "BCODE",
    sales_isvat_col: str = "ISVAT",
    sales_canceled_col: str = "CANCELED",
    purchase_isvat_col: str = "ISVAT",
) -> pd.DataFrame:
    """
    Return NON-VAT sales lines (ISVAT=N, not canceled) in `year` whose latest purchase
    on/before the sale date (per BCODE) WAS VAT (purchase ISVAT == 'Y').
    """

    source = source.lower()
    if source not in ("hq", "syp"):
        raise ValueError("source must be 'hq' or 'syp'")

    sales_key = sales_key_template.format(source=source)
    purchase_key = purchase_key_template.format(source=source)

    sales = _clean_cols(data[sales_key])
    pidet = _clean_cols(data[purchase_key])

    # remove invalid BCODE early
    sales = _drop_invalid_bcode(sales, bcode_col)
    pidet = _drop_invalid_bcode(pidet, bcode_col)

    # clean BCODE
    sales[bcode_col] = _clean_bcode(sales[bcode_col])
    pidet[bcode_col] = _clean_bcode(pidet[bcode_col])

    # parse dates
    sales[date_col] = pd.to_datetime(sales[date_col], errors="coerce")
    pidet[date_col] = pd.to_datetime(pidet[date_col], errors="coerce")

    # drop missing keys
    sales = sales.dropna(subset=[date_col, bcode_col]).copy()
    pidet = pidet.dropna(subset=[date_col, bcode_col]).copy()
    sales = sales[sales[bcode_col] != ""].copy()
    pidet = pidet[pidet[bcode_col] != ""].copy()

    # SALES filter: NON-VAT + not canceled + year
    sales_isvat = sales[sales_isvat_col].astype(str).str.strip().str.upper()
    mask_sales_nonvat = sales_isvat.eq("N") | sales_isvat.eq("")  # optional: treat blank as non-vat
    mask_sales_not_canceled = sales[sales_canceled_col].astype(str).str.strip().str.upper() != "Y"
    mask_sales_year = sales[date_col].dt.year == year

    nonvat_sales_lines = sales[mask_sales_nonvat & mask_sales_not_canceled & mask_sales_year].copy()

    # PIDET VAT flag clean
    pidet[purchase_isvat_col] = pidet[purchase_isvat_col].astype(str).str.strip().str.upper()

    pidet_key = pidet[[bcode_col, date_col, purchase_isvat_col]].copy()
    pidet_key = pidet_key.rename(columns={purchase_isvat_col: "LAST_PURCHASE_ISVAT"})

    # sort for merge_asof
    nonvat_sales_lines = nonvat_sales_lines.sort_values([date_col, bcode_col], kind="mergesort").reset_index(drop=True)
    pidet_key = pidet_key.sort_values([date_col, bcode_col], kind="mergesort").reset_index(drop=True)

    merged = pd.merge_asof(
        nonvat_sales_lines,
        pidet_key,
        left_on=date_col,
        right_on=date_col,
        by=bcode_col,
        direction="backward",
        allow_exact_matches=True
    )

    # Keep NON-VAT sales where last purchase WAS VAT
    nonvat_sales_last_purchase_vat = merged[
        merged["LAST_PURCHASE_ISVAT"].fillna("").eq("Y")
    ].copy()

    return nonvat_sales_last_purchase_vat

def audit_bcode_vat_sales_last_purchase(
    data: dict,
    bcode: str,
    year: int,
    source: str = "hq",
    month: int | None = None,
    show_sales_head: int = 20,
    show_pidet_tail: int = 30,
):
    """
    Audit one BCODE:
    - VAT sales (ISVAT=Y, not canceled) for a given year / optional month
    - PIDET purchase history
    - Show last purchase <= first sale date (asof logic)
    """

    def clean_bcode(s):
        return (
            s.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

    # ------------------
    # SALES (derived internally)
    # ------------------
    sales = get_vat_sales_lines(
        data=data,
        year=year,
        source=source,
        month=month,
    )

    sales["BCODE"] = clean_bcode(sales["BCODE"])
    sales["BILLDATE"] = pd.to_datetime(sales["BILLDATE"], errors="coerce")

    bcode_clean = clean_bcode(pd.Series([bcode])).iloc[0]
    sales_one = sales.loc[sales["BCODE"] == bcode_clean].copy()

    # ------------------
    # PIDET
    # ------------------
    pidet = data[f"raw_{source}_pidet_purchase_lines.csv"].copy()
    pidet.columns = (
        pidet.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    pidet["BCODE"] = clean_bcode(pidet["BCODE"])
    pidet["BILLDATE"] = pd.to_datetime(pidet["BILLDATE"], errors="coerce")
    pidet["ISVAT"] = pidet["ISVAT"].astype(str).str.strip().str.upper()

    pidet_one = pidet.loc[pidet["BCODE"] == bcode_clean].copy()

    # ------------------
    # PRINT SUMMARY
    # ------------------
    print("\n================ BCODE AUDIT ================")
    print("BCODE :", bcode_clean)
    print("Source:", source.upper())
    print("Year  :", year)
    print("Month :", month if month is not None else "ALL")
    print("--------------------------------------------")
    print("VAT sales rows :", len(sales_one))
    print("PIDET rows     :", len(pidet_one))

    # ------------------
    # DATE RANGES
    # ------------------
    if len(sales_one) > 0:
        print(
            "Sales date range:",
            sales_one["BILLDATE"].min(),
            "->",
            sales_one["BILLDATE"].max(),
        )

    if len(pidet_one) > 0:
        print(
            "PIDET date range:",
            pidet_one["BILLDATE"].min(),
            "->",
            pidet_one["BILLDATE"].max(),
        )

    # ------------------
    # DISPLAY SALES
    # ------------------
    print(f"\n--- SALES (first {show_sales_head} by date) ---")
    sales_cols = [c for c in ["BILLNO", "LINE", "BILLDATE", "BCODE"] if c in sales_one.columns]
    if len(sales_one) > 0:
        print(
            sales_one
            .sort_values("BILLDATE")[sales_cols]
            .head(show_sales_head)
        )
    else:
        print("No VAT sales rows for this BCODE.")

    # ------------------
    # DISPLAY PIDET
    # ------------------
    print(f"\n--- PIDET (last {show_pidet_tail} by date) ---")
    pidet_cols = [c for c in ["BILLNO", "BILLDATE", "BCODE", "ISVAT"] if c in pidet_one.columns]
    if len(pidet_one) > 0:
        print(
            pidet_one
            .sort_values("BILLDATE")[pidet_cols]
            .tail(show_pidet_tail)
        )
    else:
        print("No PIDET rows for this BCODE.")

    # ------------------
    # ASOF LOGIC CHECK
    # ------------------
    if len(sales_one) > 0:
        first_sale_date = sales_one["BILLDATE"].min()
        eligible = pidet_one.loc[
            pidet_one["BILLDATE"] <= first_sale_date
        ].sort_values("BILLDATE")

        print("\nFirst sale date:", first_sale_date)
        print("PIDET rows <= first sale:", len(eligible))

        if len(eligible) > 0:
            print("\nLast purchase before first sale (ASOF match):")
            print(
                eligible.tail(1)[["BILLNO", "LINE", "BILLDATE", "ISVAT"]]
            )
        else:
            print(
                "\n⚠ No purchase exists on/before the first sale date "
                "→ merge_asof will produce NaN."
            )

    print("============================================\n")

    return sales_one, pidet_one


def audit_bcode_nonvat_sales_last_purchase_vat(
    data: dict,
    bcode: str,
    year: int,
    source: str = "hq",
    month: int | None = None,
    show_sales_head: int = 20,
    show_pidet_tail: int = 30,
):
    """
    Audit one BCODE:
    - NON-VAT sales (ISVAT=N, not canceled) for a given year / optional month
    - PIDET purchase history
    - Show last purchase <= first sale date (asof logic)
    """

    def clean_bcode(s):
        return (
            s.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

    # ------------------
    # SALES (derived internally)
    # ------------------
    sales = get_nonvat_sales_lines(
        data=data,
        year=year,
        source=source,
        month=month,
    )

    sales["BCODE"] = clean_bcode(sales["BCODE"])
    sales["BILLDATE"] = pd.to_datetime(sales["BILLDATE"], errors="coerce")

    bcode_clean = clean_bcode(pd.Series([bcode])).iloc[0]
    sales_one = sales.loc[sales["BCODE"] == bcode_clean].copy()

    # ------------------
    # PIDET (all history for this source)
    # ------------------
    pidet = data[f"raw_{source}_pidet_purchase_lines.csv"].copy()
    pidet.columns = (
        pidet.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    pidet = _drop_invalid_bcode(pidet, "BCODE")

    pidet["BCODE"] = clean_bcode(pidet["BCODE"])
    pidet["BILLDATE"] = pd.to_datetime(pidet["BILLDATE"], errors="coerce")
    pidet["ISVAT"] = pidet["ISVAT"].astype(str).str.strip().str.upper()

    pidet_one = pidet.loc[pidet["BCODE"] == bcode_clean].copy()

    # ------------------
    # PRINT SUMMARY
    # ------------------
    print("\n================ BCODE AUDIT ================")
    print("BCODE :", bcode_clean)
    print("Source:", source.upper())
    print("Year  :", year)
    print("Month :", month if month is not None else "ALL")
    print("--------------------------------------------")
    print("NON-VAT sales rows :", len(sales_one))
    print("PIDET rows         :", len(pidet_one))

    # ------------------
    # DATE RANGES
    # ------------------
    if len(sales_one) > 0:
        print("Sales date range:", sales_one["BILLDATE"].min(), "->", sales_one["BILLDATE"].max())

    if len(pidet_one) > 0:
        print("PIDET date range:", pidet_one["BILLDATE"].min(), "->", pidet_one["BILLDATE"].max())

    # ------------------
    # DISPLAY SALES
    # ------------------
    print(f"\n--- SALES (first {show_sales_head} by date) ---")
    sales_cols = [c for c in ["BILLNO", "LINE", "BILLDATE", "BCODE", "ISVAT"] if c in sales_one.columns]
    if len(sales_one) > 0:
        print(sales_one.sort_values("BILLDATE")[sales_cols].head(show_sales_head))
    else:
        print("No NON-VAT sales rows for this BCODE.")

    # ------------------
    # DISPLAY PIDET
    # ------------------
    print(f"\n--- PIDET (last {show_pidet_tail} by date) ---")
    pidet_cols = [c for c in ["BILLNO", "LINE", "BILLDATE", "BCODE", "ISVAT"] if c in pidet_one.columns]
    if len(pidet_one) > 0:
        print(pidet_one.sort_values("BILLDATE")[pidet_cols].tail(show_pidet_tail))
    else:
        print("No PIDET rows for this BCODE.")

    # ------------------
    # ASOF LOGIC CHECK + VAT check
    # ------------------
    if len(sales_one) > 0:
        first_sale_date = sales_one["BILLDATE"].min()
        eligible = pidet_one.loc[pidet_one["BILLDATE"] <= first_sale_date].sort_values("BILLDATE")

        print("\nFirst sale date:", first_sale_date)
        print("PIDET rows <= first sale:", len(eligible))

        if len(eligible) > 0:
            last_purchase = eligible.tail(1).copy()
            print("\nLast purchase before first sale (ASOF match):")
            print(last_purchase[["BILLNO", "LINE", "BILLDATE", "ISVAT"]])

            # Your key condition
            if last_purchase["ISVAT"].iloc[0] == "Y":
                print("✅ Last purchase ISVAT = Y (VAT) — matches your target case.")
            else:
                print("⚠ Last purchase ISVAT != Y — this BCODE is NOT in the target case.")
        else:
            print("\n⚠ No purchase exists on/before the first sale date → merge_asof would be NaN.")

    print("============================================\n")

    return sales_one, pidet_one

def get_hq_to_syp_transfer_lines(
    data: dict,
    year: int | None = None,
    show_summary: bool = True
):
    """
    Return HQ → SYP transfer sales lines (BILLNO starts with 'TF').

    Parameters
    ----------
    data : dict
        Loaded CSV data
    year : int | None
        Filter by BILLDATE year (e.g. 2025). If None, no year filter.
    show_summary : bool
        Print row count and unique bill count

    Returns
    -------
    pd.DataFrame
        Transfer (TF) sales lines from HQ SIDET
    """

    df_hq_sidet = data["raw_hq_sidet_sales_lines.csv"].copy()

    # clean column names
    df_hq_sidet = _clean_columns(df_hq_sidet)
    df_hq_sidet = _drop_invalid_bcode(df_hq_sidet, "BCODE")

    # ensure BILLDATE is datetime
    df_hq_sidet["BILLDATE"] = pd.to_datetime(
        df_hq_sidet["BILLDATE"],
        errors="coerce"
    )

    # filter BILLNO starting with 'TFV'
    hq_tf_rows = df_hq_sidet[
        df_hq_sidet["BILLNO"].astype(str).str.startswith("TFV")
    ].copy()

    # optional year filter
    if year is not None:
        hq_tf_rows = hq_tf_rows[
            hq_tf_rows["BILLDATE"].dt.year == year
        ].copy()

    if show_summary:
        print("TF rows count:", hq_tf_rows.shape)
        print("Unique TF bills:", hq_tf_rows["BILLNO"].nunique())

    return hq_tf_rows


def get_syp_received_transfer_lines(
    data: dict,
    year: int | None = None,
    show_summary: bool = True
):
    """
    Return SYP received transfer purchase lines (BILLNO starts with 'TF').

    Parameters
    ----------
    data : dict
        Loaded CSV data
    year : int | None
        Filter by BILLDATE year (e.g. 2025). If None, no year filter.
    show_summary : bool
        Print row count and unique bill count

    Returns
    -------
    pd.DataFrame
        Transfer (TF) purchase lines received by SYP
    """

    df_syp_pidet = data["raw_syp_pidet_purchase_lines.csv"].copy()

    # clean column names
    df_syp_pidet = _clean_columns(df_syp_pidet)
    df_syp_pidet = _drop_invalid_bcode(df_syp_pidet, "BCODE")

    # ensure BILLDATE is datetime
    df_syp_pidet["BILLDATE"] = pd.to_datetime(
        df_syp_pidet["BILLDATE"],
        errors="coerce"
    )

    # filter BILLNO starting with 'TFV'
    syp_tf_rows = df_syp_pidet[
        df_syp_pidet["BILLNO"].astype(str).str.startswith("TFV")
    ].copy()

    # optional year filter
    if year is not None:
        syp_tf_rows = syp_tf_rows[
            syp_tf_rows["BILLDATE"].dt.year == year
        ].copy()

    if show_summary:
        print("TF rows count:", syp_tf_rows.shape)
        print("Unique TF bills:", syp_tf_rows["BILLNO"].nunique())

    return syp_tf_rows


def get_weighted_avg_purchase_unit_cost_all_time(
    pidet_df: pd.DataFrame,
    *,
    bcode_col: str = "BCODE",
    date_col: str = "BILLDATE",
    price_col: str = "PRICE",
    mtp_col: str = "MTP",
) -> pd.Series:
    """
    Return Series indexed by BCODE with WEIGHTED AVERAGE purchase UNIT COST,
    using ALL PIDET history (no year/month limit).

    Weighted avg unit cost per BCODE:
        sum(PRICE) / sum(MTP)

    Notes:
    - Assumes PRICE is the line total for the purchase line and MTP is units.
    - Filters out invalid dates, missing PRICE/MTP, and MTP <= 0.
    """

    df = pidet_df.copy()

    # clean columns
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    # drop invalid BCODE early
    df = _drop_invalid_bcode(df, bcode_col)

    # parse date (keeps "all time" but ensures valid rows)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # numeric
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df[mtp_col] = pd.to_numeric(df[mtp_col], errors="coerce")

    # keep valid amounts
    df = df.dropna(subset=[price_col, mtp_col])
    df = df[df[mtp_col] > 0]

    # weighted avg = sum(price) / sum(mtp)
    sums = df.groupby(bcode_col).agg(
        TOTAL_PRICE=(price_col, "sum"),
        TOTAL_MTP=(mtp_col, "sum")
    )

    avg_cost = (sums["TOTAL_PRICE"] / sums["TOTAL_MTP"]).replace([pd.NA, float("inf")], pd.NA)

    return avg_cost


import pandas as pd

import pandas as pd

def build_inventory_summary_avg_cost(
    in_dfs: list[pd.DataFrame],
    out_dfs: list[pd.DataFrame],
    pidet_all_df: pd.DataFrame,   # ALL PIDET, no year limit
    products_df: pd.DataFrame,    # raw_hq_icmas_products.csv (has DESCR)
    *,
    fallback_to_pidet_detail: bool = True,
) -> pd.DataFrame:
    """
    Output:
      BCODE | DESCR | IN | OUT | AV_COST

    Rules:
    - Include ONLY BCODEs with movement (IN > 0 or OUT > 0)
    - IN/OUT are in UNITS = QTY * MTP
    - AV_COST = weighted average from ALL PIDET history:
        sum(PRICE) / sum(MTP) per BCODE
    - DESCR is sourced from products_df['DESCR'] (ICMAS product master)
      and optionally falls back to pidet_all_df['DETAIL'] if missing.
    """

    def _norm_bcode(s: pd.Series) -> pd.Series:
        return (
            s.astype("string")
             .str.strip()
             .str.replace(r"\.0$", "", regex=True)  # fixes '123.0'
        )

    def _sum_units(dfs: list[pd.DataFrame]) -> pd.Series:
        parts = []
        for df in dfs:
            t = _drop_invalid_bcode(_clean_columns(df.copy()), "BCODE")
            t["BCODE"] = _norm_bcode(t["BCODE"])

            t["QTY"] = pd.to_numeric(t.get("QTY"), errors="coerce")
            t["MTP"] = pd.to_numeric(t.get("MTP"), errors="coerce")

            t["UNITS"] = t["QTY"] * t["MTP"]
            t = t.dropna(subset=["BCODE", "UNITS"])
            parts.append(t[["BCODE", "UNITS"]])

        if not parts:
            return pd.Series(dtype="float64")

        x = pd.concat(parts, ignore_index=True)
        return x.groupby("BCODE")["UNITS"].sum()

    # movement
    in_units = _sum_units(in_dfs)
    out_units = _sum_units(out_dfs)

    movement_bcodes = sorted(set(in_units.index) | set(out_units.index))

    # avg cost from ALL PIDET (weighted)
    avg_cost = get_weighted_avg_purchase_unit_cost_all_time(pidet_all_df)

    # -----------------------------
    # DESCR from products master (ICMAS)
    # -----------------------------
    p = _clean_columns(products_df.copy())
    p = _drop_invalid_bcode(p, "BCODE")
    p["BCODE"] = _norm_bcode(p["BCODE"])

    if "DESCR" not in p.columns:
        raise KeyError(f"products_df must have column 'DESCR'. Found: {list(p.columns)}")

    p["DESCR"] = p["DESCR"].astype("string").str.strip()

    descr_map_products = (
        p.dropna(subset=["BCODE", "DESCR"])
         .drop_duplicates(subset=["BCODE"], keep="first")
         .set_index("BCODE")["DESCR"]
    )

    # optional fallback from PIDET DETAIL
    descr_map_pidet = None
    if fallback_to_pidet_detail:
        q = _clean_columns(pidet_all_df.copy())
        if "BCODE" in q.columns and "DETAIL" in q.columns:
            q = _drop_invalid_bcode(q, "BCODE")
            q["BCODE"] = _norm_bcode(q["BCODE"])
            q["DETAIL"] = q["DETAIL"].astype("string").str.strip()

            descr_map_pidet = (
                q.dropna(subset=["BCODE", "DETAIL"])
                 .drop_duplicates(subset=["BCODE"], keep="first")
                 .set_index("BCODE")["DETAIL"]
            )

    # build result
    result = pd.DataFrame({"BCODE": pd.Series(movement_bcodes, dtype="string")})
    result["BCODE"] = _norm_bcode(result["BCODE"])

    # output column name stays DESCR
    result["DESCR"] = result["BCODE"].map(descr_map_products)

    if fallback_to_pidet_detail and descr_map_pidet is not None:
        result["DESCR"] = result["DESCR"].fillna(result["BCODE"].map(descr_map_pidet))

    result["IN"] = pd.to_numeric(result["BCODE"].map(in_units).fillna(0), errors="coerce").fillna(0)
    result["OUT"] = pd.to_numeric(result["BCODE"].map(out_units).fillna(0), errors="coerce").fillna(0)
    result["AV_COST"] = result["BCODE"].map(avg_cost)

    return result


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df

def _norm_paid(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    # normalize common variants
    s = s.replace({"1": "Y", "0": "N", "TRUE": "Y", "FALSE": "N", "T": "Y", "F": "N"})
    return s

def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def get_ap_unpaid_bills(data: dict, year: int, site: str, show_summary: bool = True) -> pd.DataFrame:
    """
    AP unpaid bills for a given site ('hq' or 'syp') where BILLDATE is in `year`
    and unpaid as-of Dec 31 of `year`:

      - PAID == 'N'
      OR
      - PAID == 'Y' AND VOUCDATE2 not missing AND VOUCDATE2 > year-12-31

    Keeps ALL columns as-is and adds SOURCE column.
    """
    site = site.strip().lower()
    key_map = {
        "hq":  "raw_hq_pimas_purchase_bills.csv",
        "syp": "raw_syp_pimas_purchase_bills.csv",
    }
    if site not in key_map:
        raise ValueError("site must be 'hq' or 'syp'")

    src_key = key_map[site]
    asof = pd.Timestamp(f"{year}-12-31")

    df = data[src_key].copy()
    df = _clean_columns(df)
    df = _ensure_datetime(df, "BILLDATE")
    df = _ensure_datetime(df, "VOUCDATE2")

    if "PAID" not in df.columns:
        raise KeyError("Missing required column: PAID")

    df["PAID_NORM"] = _norm_paid(df["PAID"])

    # only bills in the input year
    df_year = df[df["BILLDATE"].dt.year == year].copy()

    # unpaid as-of rule (EXCLUDES PAID==Y with missing VOUCDATE2)
    unpaid_mask = (
        (df_year["PAID_NORM"] == "N")
        | (
            (df_year["PAID_NORM"] == "Y")
            & df_year["VOUCDATE2"].notna()
            & (df_year["VOUCDATE2"] > asof)
        )
    )

    out = df_year[unpaid_mask].copy()
    out.insert(0, "SOURCE", src_key)  # keep columns, add source

    if show_summary:
        print(f"[AP] site={site} year={year} source={src_key}")
        print("Rows:", out.shape, "| Unique BILLNO:", out["BILLNO"].nunique() if "BILLNO" in out.columns else "n/a")
        # sanity: excluded case
        bad = out[(out["PAID_NORM"] == "Y") & (out["VOUCDATE2"].isna())]
        print("Check (PAID=Y & missing VOUCDATE2) rows:", len(bad))

    return out


def get_ar_unpaid_bills(data: dict, year: int, site: str, show_summary: bool = True) -> pd.DataFrame:
    """
    AR unpaid bills for a given site ('hq' or 'syp') where BILLDATE is in `year`
    and unpaid as-of Dec 31 of `year`:

      - PAID == 'N'
      OR
      - PAID == 'Y' AND VOUCDATE2 not missing AND VOUCDATE2 > year-12-31

    Keeps ALL columns as-is and adds SOURCE column.
    """
    site = site.strip().lower()
    key_map = {
        "hq":  "raw_hq_simas_sales_bills.csv",
        "syp": "raw_syp_simas_sales_bills.csv",
    }
    if site not in key_map:
        raise ValueError("site must be 'hq' or 'syp'")

    src_key = key_map[site]
    asof = pd.Timestamp(f"{year}-12-31")

    df = data[src_key].copy()
    df = _clean_columns(df)
    df = _ensure_datetime(df, "BILLDATE")
    df = _ensure_datetime(df, "VOUCDATE2")

    if "PAID" not in df.columns:
        raise KeyError("Missing required column: PAID")

    df["PAID_NORM"] = _norm_paid(df["PAID"])

    # only bills in the input year
    df_year = df[df["BILLDATE"].dt.year == year].copy()

    # unpaid as-of rule (EXCLUDES PAID==Y with missing VOUCDATE2)
    unpaid_mask = (
        (df_year["PAID_NORM"] == "N")
        | (
            (df_year["PAID_NORM"] == "Y")
            & df_year["VOUCDATE2"].notna()
            & (df_year["VOUCDATE2"] > asof)
        )
    )

    out = df_year[unpaid_mask].copy()
    out.insert(0, "SOURCE", src_key)  # keep columns, add source

    if show_summary:
        print(f"[AR] site={site} year={year} source={src_key}")
        print("Rows:", out.shape, "| Unique BILLNO:", out["BILLNO"].nunique() if "BILLNO" in out.columns else "n/a")
        # sanity: excluded case
        bad = out[(out["PAID_NORM"] == "Y") & (out["VOUCDATE2"].isna())]
        print("Check (PAID=Y & missing VOUCDATE2) rows:", len(bad))

    return out

def build_yearly_inventory_report(
    prev_year_inventory: pd.DataFrame | None,
    current_year_movement: pd.DataFrame,
    *,
    fix_negative_end: bool = True,
) -> pd.DataFrame:
    """
    Build new yearly inventory report.

    Inputs
    ------
    prev_year_inventory : DataFrame | None
        Columns (if provided):
        BCODE, DESCR, END, AV_COST

    current_year_movement : DataFrame
        Columns:
        BCODE, DESCR, IN, OUT, AV_COST

    Output
    ------
    BCODE, DESCR, BEGIN, IN, OUT, END, AV_COST, AMOUNT

    Rules
    -----
    - If prev_year_inventory is None:
        BEGIN = 0
    - BEGIN   = prev END (truth source)
    - IN/OUT  = current year movement (missing => 0)
    - END     = BEGIN + IN - OUT
    - AV_COST = movement AV_COST if available, else prev AV_COST
    - AMOUNT  = END * AV_COST

    Negative END rule (if fix_negative_end=True)
    --------------------------------------------
    - BEGIN is never changed
    - If END < 0:
        IN := OUT - BEGIN
        END := 0
        AMOUNT recalculated
    """

    # --------------------
    # Prepare movement
    # --------------------
    mov = current_year_movement.copy()
    mov.columns = mov.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    if "BCODE" not in mov.columns:
        raise KeyError("current_year_movement is missing required column: BCODE")

    mov["BCODE"] = mov["BCODE"].astype(str).str.strip()

    mov_small = mov[["BCODE", "IN", "OUT", "AV_COST", "DESCR"]].copy()
    mov_small = mov_small.rename(
        columns={"AV_COST": "AV_COST_MOV", "DESCR": "DESCR_MOV"}
    )

    for c in ["IN", "OUT", "AV_COST_MOV"]:
        mov_small[c] = pd.to_numeric(mov_small[c], errors="coerce")

    # --------------------
    # Prepare previous year (optional)
    # --------------------
    if prev_year_inventory is not None:
        prev = prev_year_inventory.copy()
        prev.columns = prev.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

        if "BCODE" not in prev.columns or "END" not in prev.columns:
            raise KeyError("prev_year_inventory must contain BCODE and END")

        prev["BCODE"] = prev["BCODE"].astype(str).str.strip()

        prev_small = prev[["BCODE", "END", "AV_COST", "DESCR"]].copy()
        prev_small = prev_small.rename(
            columns={
                "END": "BEGIN",
                "AV_COST": "AV_COST_PREV",
                "DESCR": "DESCR_PREV",
            }
        )

        prev_small["BEGIN"] = pd.to_numeric(prev_small["BEGIN"], errors="coerce")
        prev_small["AV_COST_PREV"] = pd.to_numeric(prev_small["AV_COST_PREV"], errors="coerce")

        result = prev_small.merge(mov_small, on="BCODE", how="outer")

    else:
        # First year: BEGIN = 0
        result = mov_small.copy()
        result["BEGIN"] = 0
        result["AV_COST_PREV"] = pd.NA
        result["DESCR_PREV"] = pd.NA

    # --------------------
    # Defaults
    # --------------------
    result["BEGIN"] = result["BEGIN"].fillna(0)
    result["IN"] = result["IN"].fillna(0)
    result["OUT"] = result["OUT"].fillna(0)

    # --------------------
    # DESCR: prev first, then movement
    # --------------------
    result["DESCR"] = result["DESCR_PREV"].combine_first(result["DESCR_MOV"])

    # --------------------
    # AV_COST: movement first, fallback to prev
    # --------------------
    result["AV_COST"] = result["AV_COST_MOV"].combine_first(result["AV_COST_PREV"])

    # --------------------
    # Compute END + AMOUNT
    # --------------------
    result["END"] = result["BEGIN"] + result["IN"] - result["OUT"]
    result["AMOUNT"] = result["END"] * pd.to_numeric(result["AV_COST"], errors="coerce")

    # --------------------
    # Fix negative END (BEGIN is truth)
    # --------------------
    if fix_negative_end:
        neg_mask = result["END"] < 0

        # audit trail
        result["IN_ORIG"] = result["IN"]
        result["NEG_END_FIXED"] = False

        # adjust IN so END becomes 0
        result.loc[neg_mask, "IN"] = (
            result.loc[neg_mask, "OUT"]
            - result.loc[neg_mask, "BEGIN"]
        )

        result.loc[neg_mask, "END"] = 0
        result.loc[neg_mask, "AMOUNT"] = 0
        result.loc[neg_mask, "NEG_END_FIXED"] = True

    # --------------------
    # Final output
    # --------------------
    result = result[
        ["BCODE", "DESCR", "BEGIN", "IN", "OUT", "END", "AV_COST", "AMOUNT",
         "IN_ORIG", "NEG_END_FIXED"]
    ].copy()

    return result