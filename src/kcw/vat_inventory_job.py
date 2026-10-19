"""Build the company VAT inventory from PARTS9 and billgen, then store it."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
from io import StringIO

import pandas as pd

from src.kcw.fifo_cost import is_transfer_bill, unit_cost_ex_vat
from src.kcw.vat_inventory import (
    KIND_RV,
    KIND_SA,
    KIND_SALE,
    KIND_TAR,
    KIND_TF,
    Flow,
    Receipt,
    SiteBook,
    VatProduct,
    build_hq_and_syp,
    document_kind,
    is_service_bcode,
    sale_unit_price,
)

_PIDET_SQL = """
SELECT
    p.ID AS line_id,
    LTRIM(RTRIM(CONVERT(nvarchar(50), p.BCODE))) AS bcode,
    CONVERT(nvarchar(50), p.BILLNO) AS billno,
    p.BILLDATE AS billdate,
    CONVERT(float, ISNULL(p.QTY, 0))
        * COALESCE(NULLIF(CONVERT(float, p.MTP), 0), 1.0) AS units,
    CONVERT(float, ISNULL(p.AMOUNT, 0)) AS amount,
    UPPER(LTRIM(RTRIM(ISNULL(p.TAXIC, N'')))) AS taxic,
    UPPER(LTRIM(RTRIM(ISNULL(p.ISVAT, N'')))) AS isvat,
    NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(50), ISNULL(p.ACCTNO, N'')))), N'') AS supplier
FROM dbo.PIDET AS p WITH (NOLOCK)
WHERE UPPER(LTRIM(RTRIM(ISNULL(p.CANCELED, N'')))) <> N'Y'
"""

_SIDET_SQL = """
SELECT
    d.ID AS line_id,
    LTRIM(RTRIM(CONVERT(nvarchar(50), d.BCODE))) AS bcode,
    CONVERT(nvarchar(50), d.BILLNO) AS billno,
    d.BILLDATE AS billdate,
    CONVERT(float, ISNULL(d.QTY, 0))
        * COALESCE(NULLIF(CONVERT(float, d.MTP), 0), 1.0) AS units,
    CONVERT(float, ISNULL(d.AMOUNT, 0)) AS amount,
    CONVERT(float, ISNULL(d.PRICE, 0)) AS price,
    COALESCE(NULLIF(CONVERT(float, d.MTP), 0), 1.0) AS mtp,
    UPPER(LTRIM(RTRIM(ISNULL(d.TAXIC, N'')))) AS taxic
FROM dbo.SIDET AS d WITH (NOLOCK)
WHERE UPPER(LTRIM(RTRIM(ISNULL(d.CANCELED, N'')))) <> N'Y'
  AND LTRIM(RTRIM(ISNULL(d.JOURMODE, N''))) <> N'0'
  AND d.BILLDATE >= '{start}'
  AND d.BILLDATE < '{end}'
"""

_ICMAS_SQL = """
SELECT
    LTRIM(RTRIM(CONVERT(nvarchar(50), BCODE))) AS bcode,
    CONVERT(nvarchar(200), ISNULL(DESCR, N'')) AS descr,
    CONVERT(float, ISNULL(QTYOH2, 0)) AS qtyoh
FROM dbo.ICMAS WITH (NOLOCK)
"""


class _Book:
    __slots__ = ("descr", "hq", "syp")

    def __init__(self) -> None:
        self.descr = ""
        self.hq = SiteBook()
        self.syp = SiteBook()

    def side(self, site: str) -> SiteBook:
        return self.syp if site == "syp" else self.hq


def _as_date(value) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _num(value) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _engine(site: str):
    from src.kcw.extract_parts9 import mssql_engine

    return mssql_engine(site)


def _read(site: str, sql: str, label: str) -> pd.DataFrame:
    print(f"[vat-inventory] read {site} {label}", flush=True)
    frame = pd.read_sql(sql, _engine(site))
    frame.columns = [str(col).lower() for col in frame.columns]
    print(f"[vat-inventory] {site} rows={len(frame)}", flush=True)
    return frame


def _add_icmas(
    books: dict[str, _Book], frame: pd.DataFrame, *, site: str, prefer_descr: bool
) -> None:
    for row in frame.itertuples(index=False):
        bcode = _text(row.bcode)
        if not bcode:
            continue
        book = books[bcode]
        book.side(site).actual += _num(row.qtyoh)
        descr = _text(row.descr)
        if descr and (prefer_descr or not book.descr):
            book.descr = descr


def _add_pidet(
    books: dict[str, _Book],
    frame: pd.DataFrame,
    site: str,
    *,
    year_start: date,
    year_end: date,
) -> None:
    for row in frame.itertuples(index=False):
        bcode = _text(row.bcode)
        billdate = _as_date(row.billdate)
        if not bcode or billdate is None:
            continue
        units = _num(row.units)
        if abs(units) <= 1e-9:
            continue
        side = books[bcode].side(site)
        is_vat = _text(row.isvat).upper() == "Y"
        cost = unit_cost_ex_vat(_num(row.amount), abs(units), _text(row.taxic))
        billno = _text(row.billno)
        if billdate >= year_end:
            continue
        if is_transfer_bill(billno):
            if billdate >= year_start:
                side.ytd_pi += units
            elif units > 1e-9:
                side.pre_transfer.append(
                    Receipt(
                        line_id=int(_num(row.line_id)),
                        billno=billno,
                        billdate=billdate,
                        units=units,
                        unit_cost=cost or 0.0,
                        is_vat=is_vat,
                        supplier=_text(row.supplier) or None,
                        site=site,
                    )
                )
            continue
        if billdate >= year_start:
            side.ytd_pi += units
            if is_vat:
                side.purchases.append(
                    Flow(
                        billdate=billdate,
                        billno=billno,
                        units=units,
                        kind="PI",
                        site=site,
                        unit_cost=cost or 0.0,
                        supplier=_text(row.supplier) or None,
                    )
                )
        elif units > 1e-9:
            side.pre_supplier.append(
                Receipt(
                    line_id=int(_num(row.line_id)),
                    billno=billno,
                    billdate=billdate,
                    units=units,
                    unit_cost=cost or 0.0,
                    is_vat=is_vat,
                    supplier=_text(row.supplier) or None,
                    site=site,
                )
            )


def _add_sidet(
    books: dict[str, _Book],
    frame: pd.DataFrame,
    site: str,
    *,
    year_start: date,
    year_end: date,
) -> None:
    for row in frame.itertuples(index=False):
        bcode = _text(row.bcode)
        billdate = _as_date(row.billdate)
        if not bcode or billdate is None or billdate < year_start or billdate >= year_end:
            continue
        units = _num(row.units)
        if abs(units) <= 1e-9:
            continue
        side = books[bcode].side(site)
        side.ytd_si += units
        billno = _text(row.billno)
        if is_transfer_bill(billno):
            if units > 1e-9:
                side.transfer_out.append(
                    Flow(billdate=billdate, billno=billno, units=units, kind=KIND_TF, site=site)
                )
            continue
        kind = document_kind(billno)
        if kind == KIND_SALE:
            side.sales.append(
                Flow(
                    billdate=billdate,
                    billno=billno,
                    units=units,
                    kind=KIND_SALE,
                    site=site,
                    sale_price=sale_unit_price(
                        amount=_num(row.amount),
                        units=units,
                        taxic=_text(row.taxic),
                        price=_num(row.price),
                        mtp=_num(row.mtp),
                    ),
                )
            )
        elif kind == KIND_SA:
            side.adjustments.append(
                Flow(billdate=billdate, billno=_text(row.billno), units=units, kind=KIND_SA, site=site)
            )


def _add_generated(
    books: dict[str, _Book],
    frame: pd.DataFrame,
    *,
    kind: str,
    site: str,
    year_start: date,
    year_end: date,
) -> None:
    if frame is None or frame.empty:
        return
    for row in frame.itertuples(index=False):
        bcode = _text(row.bcode)
        billdate = _as_date(row.billdate)
        if not bcode or billdate is None or billdate < year_start or billdate >= year_end:
            continue
        qty = _num(row.qty)
        mtp = _num(row.mtp) or 1.0
        units = qty * mtp
        if abs(units) <= 1e-9:
            continue
        cost = _num(row.last_cost) if kind == KIND_RV else None
        sale_price = None
        if kind == KIND_TAR:
            sale_price = sale_unit_price(
                amount=_num(row.amount),
                units=units,
                taxic="Y",
                price=_num(row.price),
                mtp=mtp,
            )
        books[bcode].side(site).__getattribute__("rvs" if kind == KIND_RV else "tars").append(
            Flow(
                billdate=billdate,
                billno=_text(row.new_billno),
                units=units,
                kind=kind,
                site=site,
                unit_cost=cost or None,
                sale_price=sale_price,
            )
        )


def _generated_frames() -> dict[str, pd.DataFrame]:
    from src.kcw.tar import supabase_db_url

    url = supabase_db_url()
    frames: dict[str, pd.DataFrame] = {}
    queries = {
        "rv": """
            select billdate, new_billno, bcode, qty, mtp, price, amount, last_cost
            from billgen.fin_rv_lines
        """,
        "3rv": """
            select billdate, new_billno, bcode, qty, mtp, price, amount, last_cost
            from billgen.fin_3rv_lines
        """,
        "tar": """
            select billdate, new_billno, bcode, qty, mtp, price, amount,
                   null::float8 as last_cost
            from billgen.fin_tar_lines
        """,
        "3tar": """
            select billdate, new_billno, bcode, qty, mtp, price, amount,
                   null::float8 as last_cost
            from billgen.fin_3tar_lines
        """,
        "cntar": """
            select billdate, new_billno, bcode, qty, mtp, price, amount,
                   null::float8 as last_cost
            from billgen.fin_cntar_lines
        """,
        "3cntar": """
            select billdate, new_billno, bcode, qty, mtp, price, amount,
                   null::float8 as last_cost
            from billgen.fin_3cntar_lines
        """,
    }
    import psycopg

    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            for name, sql in queries.items():
                cur.execute(sql)
                columns = [desc[0] for desc in cur.description]
                frames[name] = pd.DataFrame(cur.fetchall(), columns=columns)
                frames[name].columns = [str(col).lower() for col in frames[name].columns]
                print(f"[vat-inventory] {name} rows={len(frames[name])}", flush=True)
    return frames


def build_products(books: dict[str, _Book], *, year_start: date) -> list[VatProduct]:
    products: list[VatProduct] = []
    for bcode, book in books.items():
        if is_service_bcode(bcode):
            continue
        products.extend(
            build_hq_and_syp(
                bcode=bcode,
                descr=book.descr,
                hq=book.hq,
                syp=book.syp,
                year_start=year_start,
            )
        )
    return products


def _num_csv(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _write(products: list[VatProduct], rebuilt_at: datetime, *, year: int) -> None:
    from src.kcw.tar import supabase_db_url

    import psycopg

    product_buf = StringIO()
    product_buf.write(
        "bcode,site,year,descr,qty,value,uncovered_qty,open_qty,purchase_qty,rv_qty,sale_qty,tar_qty,sa_qty,tf_qty,actual_qty,rebuilt_at\n"
    )
    layer_buf = StringIO()
    layer_buf.write("bcode,source,billno,billdate,supplier,site,year,qty,unit_cost\n")
    move_buf = StringIO()
    move_buf.write(
        "bcode,billdate,billno,kind,site,year,qty,unit_cost,balance,supplier,sale_price,cost_billno\n"
    )
    stamp = rebuilt_at.isoformat()
    for product in products:
        product_buf.write(
            ",".join(
                [
                    _csv(product.bcode),
                    _csv(product.site),
                    str(year),
                    _csv(product.descr),
                    _num_csv(product.qty),
                    _num_csv(product.value),
                    _num_csv(product.uncovered_qty),
                    _num_csv(product.open_qty),
                    _num_csv(product.purchase_qty),
                    _num_csv(product.rv_qty),
                    _num_csv(product.sale_qty),
                    _num_csv(product.tar_qty),
                    _num_csv(product.sa_qty),
                    _num_csv(product.tf_qty),
                    _num_csv(product.actual_qty),
                    stamp,
                ]
            )
            + "\n"
        )
        for layer in product.layers:
            layer_buf.write(
                ",".join(
                    [
                        _csv(product.bcode),
                        layer.source,
                        _csv(layer.billno),
                        layer.billdate.isoformat(),
                        _csv(layer.supplier or ""),
                        _csv(layer.site),
                        str(year),
                        _num_csv(layer.qty),
                        _num_csv(layer.unit_cost),
                    ]
                )
                + "\n"
            )
        for move in product.movements:
            move_buf.write(
                ",".join(
                    [
                        _csv(product.bcode),
                        move.billdate.isoformat(),
                        _csv(move.billno),
                        move.kind,
                        _csv(move.site),
                        str(year),
                        _num_csv(move.qty),
                        _num_csv(move.unit_cost),
                        _num_csv(move.balance),
                        _csv(move.supplier or ""),
                        _num_csv(move.sale_price),
                        _csv(move.cost_billno or ""),
                    ]
                )
                + "\n"
            )

    def copy_into(cur, buf: StringIO, sql: str) -> None:
        buf.seek(0)
        with cur.copy(sql) as copy:
            copy.write(buf.read())

    with psycopg.connect(supabase_db_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "alter table fifo.vat_movement "
                "add column if not exists sale_price double precision"
            )
            cur.execute(
                "alter table fifo.vat_movement "
                "add column if not exists cost_billno text"
            )
            cur.execute("delete from fifo.vat_movement where year = %s", (year,))
            cur.execute("delete from fifo.vat_layer where year = %s", (year,))
            cur.execute("delete from fifo.vat_product where year = %s", (year,))
            copy_into(
                cur,
                product_buf,
                """
                COPY fifo.vat_product (
                    bcode, site, year, descr, qty, value, uncovered_qty, open_qty, purchase_qty,
                    rv_qty, sale_qty, tar_qty, sa_qty, tf_qty, actual_qty, rebuilt_at
                ) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)
                """,
            )
            copy_into(
                cur,
                layer_buf,
                """
                COPY fifo.vat_layer (
                    bcode, source, billno, billdate, supplier, site, year, qty, unit_cost
                ) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)
                """,
            )
            copy_into(
                cur,
                move_buf,
                """
                COPY fifo.vat_movement (
                    bcode, billdate, billno, kind, site, year, qty, unit_cost, balance, supplier,
                    sale_price, cost_billno
                ) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)
                """,
            )
        conn.commit()


def _csv(value: str) -> str:
    text = (value or "").replace('"', '""')
    if text == "" or any(ch in text for ch in (",", '"', "\n", "\r")):
        return f'"{text}"'
    return text


def should_rebuild_year(target: int, today_year: int, already_stored: bool) -> bool:
    """A closed year stays as it was. The open year is rebuilt, and a new year is created on the first run after 1 Jan."""

    if target > today_year:
        return False
    if target < today_year and already_stored:
        return False
    return True


def _year_stored(year: int) -> bool:
    from src.kcw.tar import supabase_db_url

    import psycopg

    with psycopg.connect(supabase_db_url(), connect_timeout=20) as conn:
        with conn.cursor() as cur:
            cur.execute("select exists(select 1 from fifo.vat_product where year = %s)", (year,))
            row = cur.fetchone()
    return bool(row and row[0])


def run(year: int | None = None) -> int:
    from zoneinfo import ZoneInfo

    today = datetime.now(ZoneInfo("Asia/Bangkok")).date()
    target = today.year if year is None else int(year)
    if not should_rebuild_year(target, today.year, _year_stored(target) if target < today.year else False):
        print(f"[vat-inventory] year={target} stays frozen", flush=True)
        return 0
    year_start = date(target, 1, 1)
    year_end = date(target + 1, 1, 1)
    print(f"[vat-inventory] build year={target}", flush=True)
    sidet_sql = _SIDET_SQL.format(start=f"{target}0101", end=f"{target + 1}0101")
    books: dict[str, _Book] = defaultdict(_Book)
    for site, prefer in (("hq", True), ("syp", False)):
        _add_icmas(books, _read(site, _ICMAS_SQL, "icmas"), site=site, prefer_descr=prefer)
        _add_pidet(
            books,
            _read(site, _PIDET_SQL, "pidet"),
            site,
            year_start=year_start,
            year_end=year_end,
        )
        _add_sidet(
            books,
            _read(site, sidet_sql, "sidet"),
            site,
            year_start=year_start,
            year_end=year_end,
        )
    generated = _generated_frames()
    _add_generated(
        books, generated["rv"], kind=KIND_RV, site="hq", year_start=year_start, year_end=year_end
    )
    _add_generated(
        books, generated["3rv"], kind=KIND_RV, site="syp", year_start=year_start, year_end=year_end
    )
    for name, site in (("tar", "hq"), ("cntar", "hq"), ("3tar", "syp"), ("3cntar", "syp")):
        _add_generated(
            books, generated[name], kind=KIND_TAR, site=site, year_start=year_start, year_end=year_end
        )
    print(f"[vat-inventory] products in memory={len(books)}", flush=True)
    products = build_products(books, year_start=year_start)
    print(f"[vat-inventory] listed={len(products)}", flush=True)
    _write(products, datetime.now(timezone.utc), year=target)
    qty = sum(row.qty for row in products)
    value = sum(row.value for row in products)
    print(f"[vat-inventory] year={target} qty={qty:.2f} value={value:.2f}", flush=True)
    return 0
