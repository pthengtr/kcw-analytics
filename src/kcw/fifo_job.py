"""Load PARTS9 history, rebuild FIFO cost, and store it in schema fifo.

Backfill reads the whole site once. The daily pass reads only products that
had a costed purchase, a sale, or a stock count on that day.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from src.kcw.fifo_cost import (
    KIND_PURCHASE,
    FifoResult,
    ReceiptLine,
    SaleLine,
    cleanup_decision,
    issue_kind,
    ledger_on_hand,
    pick_last_movement,
    qtyoh2_is_stale,
    rebuild,
    receipt_for_cost,
    sale_for_cost,
)

_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "supabase"
    / "migrations"
    / "20261004203600_create_fifo_schema.sql"
)

_PIDET_SQL = """
SELECT
    p.ID AS line_id,
    LTRIM(RTRIM(CONVERT(nvarchar(50), p.BCODE))) AS bcode,
    p.BILLNO AS billno,
    p.BILLDATE AS billdate,
    CONVERT(float, ISNULL(p.QTY, 0))
        * COALESCE(NULLIF(CONVERT(float, p.MTP), 0), 1.0) AS units,
    CONVERT(float, ISNULL(p.AMOUNT, 0)) AS amount,
    UPPER(LTRIM(RTRIM(ISNULL(p.TAXIC, N'')))) AS taxic,
    COALESCE(
        NULLIF(LTRIM(RTRIM(ISNULL(h.ACCTNO, N''))), N''),
        NULLIF(LTRIM(RTRIM(ISNULL(p.ACCTNO, N''))), N'')
    ) AS supplier
FROM dbo.PIDET AS p WITH (NOLOCK)
OUTER APPLY (
    SELECT TOP 1 ACCTNO
    FROM dbo.PIMAS WITH (NOLOCK)
    WHERE BILLNO = p.BILLNO
      AND BILLDATE = p.BILLDATE
) AS h
WHERE UPPER(LTRIM(RTRIM(ISNULL(p.CANCELED, N'')))) <> N'Y'
"""

_SIDET_SQL = """
SELECT
    d.ID AS line_id,
    LTRIM(RTRIM(CONVERT(nvarchar(50), d.BCODE))) AS bcode,
    d.BILLNO AS billno,
    d.BILLDATE AS billdate,
    CONVERT(float, ISNULL(d.QTY, 0))
        * COALESCE(NULLIF(CONVERT(float, d.MTP), 0), 1.0) AS units,
    LTRIM(RTRIM(ISNULL(d.JOURMODE, N''))) AS jourmode
FROM dbo.SIDET AS d WITH (NOLOCK)
WHERE UPPER(LTRIM(RTRIM(ISNULL(d.CANCELED, N'')))) <> N'Y'
"""

_ICMAS_SQL = """
SELECT
    LTRIM(RTRIM(CONVERT(nvarchar(50), BCODE))) AS bcode,
    CONVERT(float, ISNULL(QTYBEG2, 0)) AS qtybeg,
    CONVERT(float, QTYOH2) AS qtyoh
FROM dbo.ICMAS WITH (NOLOCK)
"""


@dataclass
class _Book:
    qtybeg: float = 0.0
    qtyoh_icmas: float | None = None
    seen_icmas: bool = False
    pi_units: float = 0.0
    si_units: float = 0.0
    receipts: list[ReceiptLine] = field(default_factory=list)
    sales: list[SaleLine] = field(default_factory=list)
    movements: list[tuple[date, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Built:
    bcode: str
    qtyoh2_icmas: float | None
    qtyoh2_ledger: float
    qtyoh2_stale: bool
    last_movement_at: date | None
    last_movement_kind: str | None
    cleanup_recommend: bool
    cleanup_reason: str
    result: FifoResult


def _engine(site: str) -> Engine:
    from src.kcw.extract_parts9 import mssql_engine

    return mssql_engine(site)


def _as_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _books() -> dict[str, _Book]:
    return defaultdict(_Book)


def _add_icmas(books: dict[str, _Book], rows) -> int:
    n = 0
    for row in rows:
        bcode = _text(row.bcode)
        if not bcode:
            continue
        book = books[bcode]
        book.qtybeg = float(row.qtybeg or 0.0)
        book.qtyoh_icmas = None if row.qtyoh is None else float(row.qtyoh)
        book.seen_icmas = True
        n += 1
    return n


def _add_pidet(books: dict[str, _Book], rows) -> int:
    n = 0
    for row in rows:
        bcode = _text(row.bcode)
        if not bcode:
            continue
        units = float(row.units or 0.0)
        book = books[bcode]
        book.pi_units += units
        n += 1
        billdate = _as_date(row.billdate)
        billno = _text(row.billno)
        receipt = receipt_for_cost(
            line_id=int(row.line_id),
            billno=billno,
            billdate=billdate,
            units=units,
            amount=float(row.amount or 0.0),
            taxic=_text(row.taxic),
            supplier=_text(row.supplier),
        )
        if receipt is None:
            continue
        book.receipts.append(receipt)
        book.movements.append((receipt.billdate, KIND_PURCHASE))
    return n


def _add_sidet(books: dict[str, _Book], rows) -> int:
    n = 0
    for row in rows:
        bcode = _text(row.bcode)
        if not bcode:
            continue
        jourmode = _text(row.jourmode)
        if jourmode == "0":
            continue
        units = float(row.units or 0.0)
        book = books[bcode]
        book.si_units += units
        n += 1
        billdate = _as_date(row.billdate)
        billno = _text(row.billno)
        kind = issue_kind(billno, units, jourmode)
        if kind is not None and billdate is not None:
            book.movements.append((billdate, kind))
        sale = sale_for_cost(
            line_id=int(row.line_id),
            billno=billno,
            billdate=billdate,
            units=units,
            jourmode=jourmode,
        )
        if sale is not None:
            book.sales.append(sale)
    return n


def _filtered(sql: str, column: str) -> str:
    clause = f"LTRIM(RTRIM(CONVERT(nvarchar(50), {column}))) IN :codes"
    keyword = "\n  AND " if "\nWHERE " in sql else "\nWHERE "
    return sql + keyword + clause


def _load_all(engine: Engine) -> dict[str, _Book]:
    books = _books()
    with engine.connect() as conn:
        conn = conn.execution_options(yield_per=5000)
        print("fifo load ICMAS", flush=True)
        n = _add_icmas(books, conn.execute(text(_ICMAS_SQL)))
        print(f"fifo ICMAS rows={n}", flush=True)
        print("fifo load PIDET", flush=True)
        n = _add_pidet(books, conn.execute(text(_PIDET_SQL)))
        print(f"fifo PIDET rows={n}", flush=True)
        print("fifo load SIDET", flush=True)
        n = _add_sidet(books, conn.execute(text(_SIDET_SQL)))
        print(f"fifo SIDET rows={n}", flush=True)
    return books


def _load_codes(engine: Engine, bcodes: list[str]) -> dict[str, _Book]:
    books = _books()
    if not bcodes:
        return books
    icmas_sql = text(_filtered(_ICMAS_SQL, "BCODE")).bindparams(bindparam("codes", expanding=True))
    pidet_sql = text(_filtered(_PIDET_SQL, "p.BCODE")).bindparams(bindparam("codes", expanding=True))
    sidet_sql = text(_filtered(_SIDET_SQL, "d.BCODE")).bindparams(bindparam("codes", expanding=True))
    with engine.connect() as conn:
        for start in range(0, len(bcodes), 400):
            chunk = bcodes[start : start + 400]
            params = {"codes": chunk}
            _add_icmas(books, conn.execute(icmas_sql, params))
            _add_pidet(books, conn.execute(pidet_sql, params))
            _add_sidet(books, conn.execute(sidet_sql, params))
    return books


def build_one(bcode: str, book: _Book, *, as_of: date) -> Built:
    ledger = ledger_on_hand(book.qtybeg, book.pi_units, book.si_units)
    result = rebuild(ledger, book.receipts, book.sales, as_of=as_of)
    last_at, last_kind = pick_last_movement(book.movements)
    recommend, reason = cleanup_decision(ledger, last_at, as_of=as_of)
    icmas = book.qtyoh_icmas if book.seen_icmas else None
    return Built(
        bcode=bcode,
        qtyoh2_icmas=icmas,
        qtyoh2_ledger=ledger,
        qtyoh2_stale=qtyoh2_is_stale(icmas, ledger),
        last_movement_at=last_at,
        last_movement_kind=last_kind,
        cleanup_recommend=recommend,
        cleanup_reason=reason,
        result=result,
    )


def _worth_storing(built: Built) -> bool:
    """Skip empty master rows that never moved and have no on-hand."""
    if abs(built.qtyoh2_ledger) > 0.01:
        return True
    if built.result.sales:
        return True
    if built.last_movement_at is not None:
        return True
    return False


def ensure_schema() -> None:
    import psycopg

    from src.kcw.tar import supabase_db_url

    script = _MIGRATION.read_text()
    statements = [part.strip() for part in script.split(";") if part.strip()]
    with psycopg.connect(supabase_db_url(), connect_timeout=20) as conn:
        for statement in statements:
            # Skip comment-only chunks left when a statement starts with --.
            body = "\n".join(
                line for line in statement.splitlines() if not line.strip().startswith("--")
            ).strip()
            if not body:
                continue
            conn.execute(body)
        conn.commit()


def _write(site: str, rows: list[Built], rebuilt_at: datetime) -> None:
    import psycopg

    from src.kcw.tar import supabase_db_url

    if not rows:
        return
    bcodes = [row.bcode for row in rows]
    products = [
        (
            site,
            row.bcode,
            row.qtyoh2_icmas,
            row.qtyoh2_ledger,
            row.qtyoh2_stale,
            row.result.on_hand_qty,
            row.result.on_hand_covered_qty,
            row.result.on_hand_value,
            row.result.unassigned_qty,
            row.result.history_window,
            row.result.cost_complete,
            row.last_movement_at,
            row.last_movement_kind,
            row.cleanup_recommend,
            row.cleanup_reason,
            rebuilt_at,
        )
        for row in rows
    ]
    sales = []
    allocs = []
    for row in rows:
        for sale in row.result.sales:
            sales.append(
                (
                    site,
                    sale.line_id,
                    row.bcode,
                    sale.billno,
                    sale.billdate,
                    sale.units,
                    sale.fifo_unit_cost,
                    sale.fifo_ext_cost,
                    sale.status,
                    rebuilt_at,
                )
            )
            for alloc in sale.allocs:
                allocs.append(
                    (
                        site,
                        sale.line_id,
                        alloc.pidet_id,
                        alloc.pi_billno,
                        alloc.pi_billdate,
                        alloc.supplier,
                        alloc.qty,
                        alloc.unit_cost,
                    )
                )
    with psycopg.connect(supabase_db_url(), connect_timeout=20) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM fifo.sale_cost
                WHERE site = %s AND bcode = ANY(%s)
                """,
                (site, bcodes),
            )
            cur.executemany(
                """
                INSERT INTO fifo.product (
                    site, bcode, qtyoh2_icmas, qtyoh2_ledger, qtyoh2_stale,
                    on_hand_qty, on_hand_covered_qty, on_hand_value, unassigned_qty,
                    history_window, cost_complete, last_movement_at, last_movement_kind,
                    cleanup_recommend, cleanup_reason, rebuilt_at
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (site, bcode) DO UPDATE SET
                    qtyoh2_icmas = EXCLUDED.qtyoh2_icmas,
                    qtyoh2_ledger = EXCLUDED.qtyoh2_ledger,
                    qtyoh2_stale = EXCLUDED.qtyoh2_stale,
                    on_hand_qty = EXCLUDED.on_hand_qty,
                    on_hand_covered_qty = EXCLUDED.on_hand_covered_qty,
                    on_hand_value = EXCLUDED.on_hand_value,
                    unassigned_qty = EXCLUDED.unassigned_qty,
                    history_window = EXCLUDED.history_window,
                    cost_complete = EXCLUDED.cost_complete,
                    last_movement_at = EXCLUDED.last_movement_at,
                    last_movement_kind = EXCLUDED.last_movement_kind,
                    cleanup_recommend = EXCLUDED.cleanup_recommend,
                    cleanup_reason = EXCLUDED.cleanup_reason,
                    rebuilt_at = EXCLUDED.rebuilt_at
                """,
                products,
            )
            _copy_rows(
                cur,
                "fifo.sale_cost",
                (
                    "site",
                    "sidet_id",
                    "bcode",
                    "sale_billno",
                    "sale_billdate",
                    "units",
                    "fifo_unit_cost",
                    "fifo_ext_cost",
                    "status",
                    "rebuilt_at",
                ),
                sales,
            )
            _copy_rows(
                cur,
                "fifo.sale_cost_alloc",
                (
                    "site",
                    "sidet_id",
                    "pidet_id",
                    "pi_billno",
                    "pi_billdate",
                    "supplier_acctno",
                    "qty",
                    "unit_cost",
                ),
                allocs,
            )
        conn.commit()


def _copy_rows(cur, table: str, columns: tuple[str, ...], rows: list[tuple]) -> None:
    if not rows:
        return
    column_list = ", ".join(columns)
    with cur.copy(f"COPY {table} ({column_list}) FROM STDIN") as copy:
        for row in rows:
            copy.write_row(row)


def _store_all(site: str, built_rows: list[Built], *, as_of: date) -> dict[str, int]:
    rebuilt_at = datetime.now(timezone.utc)
    batch = 100
    stored = 0
    sales = 0
    for start in range(0, len(built_rows), batch):
        chunk = built_rows[start : start + batch]
        _write(site, chunk, rebuilt_at)
        stored += len(chunk)
        sales += sum(len(row.result.sales) for row in chunk)
        print(f"fifo wrote products={stored} sale_lines={sales}", flush=True)
    return {"products": stored, "sale_lines": sales, "as_of": as_of.isoformat()}


def run_backfill(site: str, *, as_of: date | None = None) -> int:
    day = as_of or date.today()
    print(f"fifo backfill site={site} as_of={day}", flush=True)
    ensure_schema()
    books = _load_all(_engine(site))
    built = [build_one(bcode, book, as_of=day) for bcode, book in books.items()]
    built = [row for row in built if _worth_storing(row)]
    print(f"fifo products to store={len(built)}", flush=True)
    summary = _store_all(site, built, as_of=day)
    print(f"fifo backfill done {summary}", flush=True)
    return 0


def mover_bcodes(engine: Engine, day: date) -> list[str]:
    start = datetime(day.year, day.month, day.day)
    end = start + timedelta(days=1)
    sql = text(
        """
        SELECT bcode FROM (
            SELECT LTRIM(RTRIM(CONVERT(nvarchar(50), p.BCODE))) AS bcode
            FROM dbo.PIDET AS p WITH (NOLOCK)
            WHERE p.BILLDATE >= :start AND p.BILLDATE < :end
              AND UPPER(LTRIM(RTRIM(ISNULL(p.CANCELED, N'')))) <> N'Y'
              AND ISNULL(p.AMOUNT, 0) <> 0
              AND UPPER(LTRIM(RTRIM(ISNULL(p.BILLNO, N'')))) NOT LIKE N'TF%'
              AND UPPER(LTRIM(RTRIM(ISNULL(p.BILLNO, N'')))) NOT LIKE N'3TF%'
              AND UPPER(LTRIM(RTRIM(ISNULL(p.BILLNO, N'')))) NOT LIKE N'CNTF%'
              AND UPPER(LTRIM(RTRIM(ISNULL(p.BILLNO, N'')))) NOT LIKE N'3CNTF%'
            UNION
            SELECT LTRIM(RTRIM(CONVERT(nvarchar(50), d.BCODE))) AS bcode
            FROM dbo.SIDET AS d WITH (NOLOCK)
            WHERE d.BILLDATE >= :start AND d.BILLDATE < :end
              AND UPPER(LTRIM(RTRIM(ISNULL(d.CANCELED, N'')))) <> N'Y'
              AND LTRIM(RTRIM(ISNULL(d.JOURMODE, N''))) <> N'0'
              AND UPPER(LTRIM(RTRIM(ISNULL(d.BILLNO, N'')))) NOT LIKE N'TF%'
              AND UPPER(LTRIM(RTRIM(ISNULL(d.BILLNO, N'')))) NOT LIKE N'3TF%'
              AND UPPER(LTRIM(RTRIM(ISNULL(d.BILLNO, N'')))) NOT LIKE N'CNTF%'
              AND UPPER(LTRIM(RTRIM(ISNULL(d.BILLNO, N'')))) NOT LIKE N'3CNTF%'
        ) AS moved
        WHERE bcode <> N''
        """
    )
    with engine.connect() as conn:
        found = [_text(row.bcode) for row in conn.execute(sql, {"start": start, "end": end})]
    return sorted({code for code in found if code})


def run_movers(site: str, day: date | None = None) -> int:
    move_day = day or date.today()
    print(f"fifo movers site={site} day={move_day}", flush=True)
    ensure_schema()
    engine = _engine(site)
    codes = mover_bcodes(engine, move_day)
    print(f"fifo mover bcodes={len(codes)}", flush=True)
    if not codes:
        return 0
    books = _load_codes(engine, codes)
    built = [build_one(bcode, books[bcode], as_of=move_day) for bcode in codes if bcode in books]
    summary = _store_all(site, built, as_of=move_day)
    print(f"fifo movers done {summary}", flush=True)
    return 0


def _print_built(built: Built, *, limit: int = 5) -> None:
    result = built.result
    print(
        f"bcode={built.bcode} ledger={built.qtyoh2_ledger:.4f} "
        f"icmas={built.qtyoh2_icmas} stale={built.qtyoh2_stale} "
        f"window={result.history_window} complete={result.cost_complete} "
        f"on_hand_value={result.on_hand_value:.4f} covered={result.on_hand_covered_qty:.4f} "
        f"unassigned={result.unassigned_qty:.4f} "
        f"cleanup={built.cleanup_recommend}:{built.cleanup_reason} "
        f"last={built.last_movement_at} {built.last_movement_kind}"
    )
    for sale in result.sales[:limit]:
        bills = ",".join(f"{alloc.pi_billno}@{alloc.qty:.2f}" for alloc in sale.allocs) or "-"
        unit = None if sale.fifo_unit_cost is None else round(sale.fifo_unit_cost, 4)
        print(
            f"  sale {sale.billdate} {sale.billno} units={sale.units:.2f} "
            f"status={sale.status} unit={unit} pi={bills}"
        )
    if len(result.sales) > limit:
        print(f"  ... {len(result.sales) - limit} more sales")


def run_proof(site: str) -> int:
    """Read a few live products and print the rebuild. Does not write."""
    engine = _engine(site)
    with engine.connect() as conn:
        multi = conn.execute(
            text(
                """
                SELECT TOP 1 LTRIM(RTRIM(CONVERT(nvarchar(50), BCODE))) AS bcode
                FROM dbo.PIDET WITH (NOLOCK)
                WHERE UPPER(LTRIM(RTRIM(ISNULL(CANCELED, N'')))) <> N'Y'
                  AND ISNULL(AMOUNT, 0) > 0
                  AND ISNULL(QTY, 0) > 0
                  AND BILLDATE >= '2024-01-01'
                  AND NULLIF(LTRIM(RTRIM(CONVERT(nvarchar(50), BCODE))), N'') IS NOT NULL
                  AND LTRIM(RTRIM(CONVERT(nvarchar(50), BCODE))) NOT LIKE N'7%'
                  AND UPPER(LTRIM(RTRIM(ISNULL(BILLNO, N'')))) NOT LIKE N'TF%'
                  AND UPPER(LTRIM(RTRIM(ISNULL(BILLNO, N'')))) NOT LIKE N'3TF%'
                GROUP BY LTRIM(RTRIM(CONVERT(nvarchar(50), BCODE)))
                HAVING COUNT(*) >= 2
                ORDER BY COUNT(*) DESC
                """
            )
        ).scalar()
        counted = conn.execute(
            text(
                """
                SELECT TOP 1 LTRIM(RTRIM(CONVERT(nvarchar(50), BCODE))) AS bcode
                FROM dbo.SIDET WITH (NOLOCK)
                WHERE BILLDATE >= '2026-01-01'
                  AND UPPER(LTRIM(RTRIM(ISNULL(CANCELED, N'')))) <> N'Y'
                  AND (
                    UPPER(LTRIM(RTRIM(BILLNO))) LIKE N'SA%'
                    OR UPPER(LTRIM(RTRIM(BILLNO))) LIKE N'3SA%'
                  )
                """
            )
        ).scalar()
    codes = [code for code in (_text(multi), _text(counted)) if code]
    if not codes:
        print("fifo proof: no sample bcodes", file=sys.stderr)
        return 1
    books = _load_codes(engine, codes)
    as_of = date.today()
    print(f"fifo proof site={site} as_of={as_of}", flush=True)
    for code in codes:
        book = books.get(code)
        if book is None:
            print(f"bcode={code} missing")
            continue
        _print_built(build_one(code, book, as_of=as_of))
    return 0
