"""Snapshot ICMAS + SI/PI from PARTS9 into a local SQLite snap for insight generation.

Default HQ snaps also pull SYP (kss-pc) SI/PI so fact packs see both branches.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from sqlalchemy import text

from src.kcw.extract_parts9 import mssql_engine
from src.kcw.product_insight_store import snap_path, snaps_dir, utc_now_iso


ICMAS_COLS = (
    "BCODE",
    "DESCR",
    "BRAND",
    "MODEL",
    "PCODE",
    "MCODE",
    "ACODE",
    "SIZE1",
    "UI1",
    "UI2",
    "MTP2",
    "STATUS",
    "LOCATION1",
    "COSTAVG",
    "COSTLAST",
    "PRICE1",
    "PRICEM1",
    "QTYOH2",
    "QTYMIN",
    "QTYMAX",
)


def _snap_id_now() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")


def _init_snap_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS icmas (
          bcode TEXT PRIMARY KEY,
          descr TEXT,
          brand TEXT,
          model TEXT,
          pcode TEXT,
          mcode TEXT,
          acode TEXT,
          size1 TEXT,
          ui1 TEXT,
          ui2 TEXT,
          mtp2 REAL,
          status TEXT,
          location1 TEXT,
          costavg REAL,
          costlast REAL,
          price1 REAL,
          pricem1 REAL,
          qtyoh2 REAL,
          qtymin REAL,
          qtymax REAL,
          src_site TEXT
        );
        CREATE TABLE IF NOT EXISTS icmas_stock (
          src_site TEXT NOT NULL,
          bcode TEXT NOT NULL,
          qtyoh2 REAL,
          qtymin REAL,
          qtymax REAL,
          costavg REAL,
          costlast REAL,
          price1 REAL,
          pricem1 REAL,
          location1 TEXT,
          PRIMARY KEY (src_site, bcode)
        );
        CREATE TABLE IF NOT EXISTS pimas (
          src_site TEXT NOT NULL,
          billno TEXT NOT NULL,
          billdate TEXT,
          acctno TEXT,
          acctname TEXT,
          PRIMARY KEY (src_site, billno)
        );
        CREATE TABLE IF NOT EXISTS simas (
          src_site TEXT NOT NULL,
          billno TEXT NOT NULL,
          billdate TEXT,
          acctno TEXT,
          acctname TEXT,
          PRIMARY KEY (src_site, billno)
        );
        CREATE TABLE IF NOT EXISTS sidet (
          bcode TEXT NOT NULL,
          billno TEXT,
          billdate TEXT,
          qty REAL,
          ui TEXT,
          price REAL,
          amount REAL,
          jourmode TEXT,
          src_site TEXT NOT NULL DEFAULT 'hq'
        );
        CREATE INDEX IF NOT EXISTS sidet_bcode_idx ON sidet (bcode);
        CREATE INDEX IF NOT EXISTS sidet_billdate_idx ON sidet (billdate);
        CREATE INDEX IF NOT EXISTS sidet_src_idx ON sidet (src_site);
        CREATE TABLE IF NOT EXISTS pidet (
          bcode TEXT NOT NULL,
          billno TEXT,
          billdate TEXT,
          qty REAL,
          ui TEXT,
          price REAL,
          amount REAL,
          billtype TEXT,
          src_site TEXT NOT NULL DEFAULT 'hq'
        );
        CREATE INDEX IF NOT EXISTS pidet_bcode_idx ON pidet (bcode);
        CREATE INDEX IF NOT EXISTS pidet_billdate_idx ON pidet (billdate);
        CREATE INDEX IF NOT EXISTS pidet_src_idx ON pidet (src_site);
        """
    )
    _ensure_ops_tables(conn)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _ensure_ops_tables(conn: sqlite3.Connection) -> None:
    """Idempotent extras so an older snap can be enriched in place."""
    _init_stock_pimas(conn)
    for col, decl in (
        ("pricem1", "REAL"),
        ("qtyoh2", "REAL"),
        ("qtymin", "REAL"),
        ("qtymax", "REAL"),
        ("src_site", "TEXT"),
    ):
        _ensure_column(conn, "icmas", col, decl)
    conn.commit()


def _init_stock_pimas(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS icmas_stock (
          src_site TEXT NOT NULL,
          bcode TEXT NOT NULL,
          qtyoh2 REAL,
          qtymin REAL,
          qtymax REAL,
          costavg REAL,
          costlast REAL,
          price1 REAL,
          pricem1 REAL,
          location1 TEXT,
          PRIMARY KEY (src_site, bcode)
        );
        CREATE TABLE IF NOT EXISTS pimas (
          src_site TEXT NOT NULL,
          billno TEXT NOT NULL,
          billdate TEXT,
          acctno TEXT,
          acctname TEXT,
          PRIMARY KEY (src_site, billno)
        );
        CREATE TABLE IF NOT EXISTS simas (
          src_site TEXT NOT NULL,
          billno TEXT NOT NULL,
          billdate TEXT,
          acctno TEXT,
          acctname TEXT,
          PRIMARY KEY (src_site, billno)
        );
        """
    )


def _ser(v):
    if v is None:
        return None
    if isinstance(v, (datetime, date)):
        return v.isoformat()[:10]
    try:
        from decimal import Decimal

        if isinstance(v, Decimal):
            return float(v)
    except Exception:
        pass
    if isinstance(v, float):
        return float(v)
    if isinstance(v, str):
        return v.strip()
    return v


def _normalize_sites(*, primary: str, include_sites: list[str] | None) -> list[str]:
    primary = primary.lower()
    if primary not in ("hq", "syp"):
        raise ValueError("site must be hq or syp")
    if include_sites is None:
        # HQ insight snaps default to dual-site (KSS + kss-pc).
        sites = ["hq", "syp"] if primary == "hq" else [primary]
    else:
        sites = []
        for s in include_sites:
            s = (s or "").strip().lower()
            if not s:
                continue
            if s not in ("hq", "syp"):
                raise ValueError(f"invalid include site {s!r}")
            if s not in sites:
                sites.append(s)
        if primary not in sites:
            sites.insert(0, primary)
    if not sites:
        sites = [primary]
    return sites


def _extract_si_pi(
    conn: sqlite3.Connection,
    eng,
    *,
    src_site: str,
    cutoff: str,
    replace_from: bool = False,
) -> tuple[int, int]:
    """Pull SIDET(+JOURMODE) and PIDET for one PARTS9 site into the snap.

    If replace_from, delete existing SI/PI rows with billdate >= cutoff for this
    src_site first (used by incremental refresh with a lookback overlap).
    """
    if replace_from:
        d_si = conn.execute(
            "DELETE FROM sidet WHERE src_site=? AND billdate >= ?",
            (src_site, cutoff),
        ).rowcount
        d_pi = conn.execute(
            "DELETE FROM pidet WHERE src_site=? AND billdate >= ?",
            (src_site, cutoff),
        ).rowcount
        conn.commit()
        print(
            f"{src_site} replace_from>={cutoff}: deleted sidet={d_si} pidet={d_pi}",
            flush=True,
        )

    print(f"extracting {src_site} SIDET …", flush=True)
    with eng.connect() as db:
        si_sql = text(
            """
            SELECT LTRIM(RTRIM(d.BCODE)) AS BCODE,
                   LTRIM(RTRIM(d.BILLNO)) AS BILLNO,
                   d.BILLDATE, d.QTY, d.UI, d.PRICE, d.AMOUNT,
                   h.JOURMODE
            FROM dbo.SIDET d
            LEFT JOIN dbo.SIMAS h
              ON LTRIM(RTRIM(h.BILLNO)) = LTRIM(RTRIM(d.BILLNO))
            WHERE d.BILLDATE >= :cutoff
              AND LTRIM(RTRIM(d.BCODE)) <> ''
            """
        )
        result = db.execute(si_sql, {"cutoff": cutoff})
        batch: list[tuple] = []
        n_si = 0
        while True:
            rows = result.fetchmany(5000)
            if not rows:
                break
            for r in rows:
                m = r._mapping
                batch.append(
                    (
                        _ser(m["BCODE"]),
                        _ser(m["BILLNO"]),
                        _ser(m["BILLDATE"]),
                        _ser(m["QTY"]),
                        _ser(m["UI"]),
                        _ser(m["PRICE"]),
                        _ser(m["AMOUNT"]),
                        _ser(m["JOURMODE"]),
                        src_site,
                    )
                )
            if len(batch) >= 20000:
                conn.executemany(
                    "INSERT INTO sidet(bcode,billno,billdate,qty,ui,price,amount,jourmode,src_site) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    batch,
                )
                n_si += len(batch)
                print(f"  {src_site} sidet rows={n_si}", flush=True)
                batch.clear()
                conn.commit()
        if batch:
            conn.executemany(
                "INSERT INTO sidet(bcode,billno,billdate,qty,ui,price,amount,jourmode,src_site) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                batch,
            )
            n_si += len(batch)
            conn.commit()
        print(f"{src_site} sidet_total={n_si}", flush=True)

        print(f"extracting {src_site} PIDET …", flush=True)
        pi_sql = text(
            """
            SELECT LTRIM(RTRIM(BCODE)) AS BCODE,
                   LTRIM(RTRIM(BILLNO)) AS BILLNO,
                   BILLDATE, QTY, UI, PRICE, AMOUNT, BILLTYPE
            FROM dbo.PIDET
            WHERE BILLDATE >= :cutoff
              AND LTRIM(RTRIM(BCODE)) <> ''
            """
        )
        result = db.execute(pi_sql, {"cutoff": cutoff})
        batch = []
        n_pi = 0
        while True:
            rows = result.fetchmany(5000)
            if not rows:
                break
            for r in rows:
                m = r._mapping
                batch.append(
                    (
                        _ser(m["BCODE"]),
                        _ser(m["BILLNO"]),
                        _ser(m["BILLDATE"]),
                        _ser(m["QTY"]),
                        _ser(m["UI"]),
                        _ser(m["PRICE"]),
                        _ser(m["AMOUNT"]),
                        _ser(m["BILLTYPE"]),
                        src_site,
                    )
                )
            if len(batch) >= 20000:
                conn.executemany(
                    "INSERT INTO pidet(bcode,billno,billdate,qty,ui,price,amount,billtype,src_site) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    batch,
                )
                n_pi += len(batch)
                print(f"  {src_site} pidet rows={n_pi}", flush=True)
                batch.clear()
                conn.commit()
        if batch:
            conn.executemany(
                "INSERT INTO pidet(bcode,billno,billdate,qty,ui,price,amount,billtype,src_site) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                batch,
            )
            n_pi += len(batch)
            conn.commit()
        print(f"{src_site} pidet_total={n_pi}", flush=True)
    return n_si, n_pi


def _chunks(items: list[str], size: int = 500):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _distinct_billnos(conn: sqlite3.Connection, table: str, src_site: str) -> list[str]:
    rows = conn.execute(
        f"""
        SELECT DISTINCT billno FROM {table}
        WHERE src_site = ?
          AND billno IS NOT NULL
          AND TRIM(billno) <> ''
        """,
        (src_site,),
    ).fetchall()
    return [str(r[0]).strip() for r in rows if r[0]]


def _extract_account_headers(
    conn: sqlite3.Connection,
    eng,
    *,
    src_site: str,
    table: str,
    label: str,
    cutoff: str | None = None,
    billnos: list[str] | None = None,
) -> int:
    """Pull PIMAS or SIMAS headers into the local snap.

    Prefer billno-scoped pulls (only headers that appear on snap SI/PI lines).
    Falls back to cutoff-wide extract when billnos is None.
    """
    dest = table.lower()
    if dest not in ("pimas", "simas"):
        raise ValueError(f"unsupported account table {table!r}")

    if billnos is not None and not billnos:
        print(f"{src_site} {label}_total=0 (no billnos in snap)", flush=True)
        return 0

    mode = f"billnos={len(billnos)}" if billnos is not None else f"cutoff={cutoff}"
    print(f"extracting {src_site} {label} ({mode}) …", flush=True)
    n = 0

    def _upsert(batch: list[tuple]) -> None:
        nonlocal n
        if not batch:
            return
        conn.executemany(
            f"INSERT OR REPLACE INTO {dest}(src_site,billno,billdate,acctno,acctname) "
            "VALUES (?,?,?,?,?)",
            batch,
        )
        n += len(batch)
        conn.commit()

    with eng.connect() as db:
        if billnos is not None:
            for chunk in _chunks(billnos, 500):
                placeholders = ", ".join(f":b{i}" for i in range(len(chunk)))
                params = {f"b{i}": b for i, b in enumerate(chunk)}
                sql = text(
                    f"""
                    SELECT LTRIM(RTRIM(BILLNO)) AS BILLNO,
                           BILLDATE,
                           LTRIM(RTRIM(ACCTNO)) AS ACCTNO,
                           LTRIM(RTRIM(ACCTNAME)) AS ACCTNAME
                    FROM dbo.{dest.upper()}
                    WHERE LTRIM(RTRIM(BILLNO)) IN ({placeholders})
                    """
                )
                batch: list[tuple] = []
                for r in db.execute(sql, params):
                    m = r._mapping
                    batch.append(
                        (
                            src_site,
                            _ser(m["BILLNO"]),
                            _ser(m["BILLDATE"]),
                            _ser(m["ACCTNO"]),
                            _ser(m["ACCTNAME"]),
                        )
                    )
                _upsert(batch)
                if n and n % 10000 < len(chunk):
                    print(f"  {src_site} {label} rows={n}", flush=True)
        else:
            if not cutoff:
                raise ValueError("cutoff required when billnos is None")
            sql = text(
                f"""
                SELECT LTRIM(RTRIM(BILLNO)) AS BILLNO,
                       BILLDATE,
                       LTRIM(RTRIM(ACCTNO)) AS ACCTNO,
                       LTRIM(RTRIM(ACCTNAME)) AS ACCTNAME
                FROM dbo.{dest.upper()}
                WHERE BILLDATE >= :cutoff
                  AND LTRIM(RTRIM(BILLNO)) <> ''
                """
            )
            result = db.execute(sql, {"cutoff": cutoff})
            batch = []
            while True:
                rows = result.fetchmany(5000)
                if not rows:
                    break
                for r in rows:
                    m = r._mapping
                    batch.append(
                        (
                            src_site,
                            _ser(m["BILLNO"]),
                            _ser(m["BILLDATE"]),
                            _ser(m["ACCTNO"]),
                            _ser(m["ACCTNAME"]),
                        )
                    )
                if len(batch) >= 10000:
                    _upsert(batch)
                    print(f"  {src_site} {label} rows={n}", flush=True)
                    batch = []
            _upsert(batch)

    print(f"{src_site} {label}_total={n}", flush=True)
    return n


def _extract_pimas(
    conn: sqlite3.Connection,
    eng,
    *,
    src_site: str,
    cutoff: str | None = None,
    billnos: list[str] | None = None,
) -> int:
    return _extract_account_headers(
        conn,
        eng,
        src_site=src_site,
        table="pimas",
        label="PIMAS suppliers",
        cutoff=cutoff,
        billnos=billnos,
    )


def _extract_simas(
    conn: sqlite3.Connection,
    eng,
    *,
    src_site: str,
    cutoff: str | None = None,
    billnos: list[str] | None = None,
) -> int:
    return _extract_account_headers(
        conn,
        eng,
        src_site=src_site,
        table="simas",
        label="SIMAS customers",
        cutoff=cutoff,
        billnos=billnos,
    )


def _icmas_tuple(m: dict, src_site: str) -> tuple:
    return (
        (m.get("BCODE") or "").strip(),
        m.get("DESCR"),
        m.get("BRAND"),
        m.get("MODEL"),
        m.get("PCODE"),
        m.get("MCODE"),
        m.get("ACODE"),
        m.get("SIZE1"),
        m.get("UI1"),
        m.get("UI2"),
        m.get("MTP2"),
        m.get("STATUS"),
        m.get("LOCATION1"),
        m.get("COSTAVG"),
        m.get("COSTLAST"),
        m.get("PRICE1"),
        m.get("PRICEM1"),
        m.get("QTYOH2"),
        m.get("QTYMIN"),
        m.get("QTYMAX"),
        src_site,
    )


def _stock_tuple(m: dict, src_site: str) -> tuple:
    return (
        src_site,
        (m.get("BCODE") or "").strip(),
        m.get("QTYOH2"),
        m.get("QTYMIN"),
        m.get("QTYMAX"),
        m.get("COSTAVG"),
        m.get("COSTLAST"),
        m.get("PRICE1"),
        m.get("PRICEM1"),
        m.get("LOCATION1"),
    )


def _extract_icmas_for_bcodes(
    conn: sqlite3.Connection,
    eng,
    *,
    src_site: str,
    bcodes: set[str],
    write_master: bool,
    only_missing: bool = False,
    write_stock: bool = True,
) -> int:
    if not bcodes:
        return 0
    want_master = set(bcodes)
    if only_missing:
        have = {r[0] for r in conn.execute("SELECT bcode FROM icmas")}
        want_master = bcodes - have
    print(
        f"extracting {src_site} ICMAS master={len(want_master) if write_master else 0} "
        f"stock={len(bcodes) if write_stock else 0} …",
        flush=True,
    )
    col_list = ", ".join(ICMAS_COLS)
    ic_sql = text(f"SELECT {col_list} FROM dbo.ICMAS")
    master_batch: list[tuple] = []
    stock_batch: list[tuple] = []
    n_ic = 0
    n_st = 0
    with eng.connect() as db:
        result = db.execute(ic_sql)
        while True:
            rows = result.fetchmany(5000)
            if not rows:
                break
            for r in rows:
                m = {k.upper(): _ser(v) for k, v in r._mapping.items()}
                bcode = (m.get("BCODE") or "").strip()
                if not bcode:
                    continue
                if write_master and bcode in want_master:
                    master_batch.append(_icmas_tuple(m, src_site))
                if write_stock and bcode in bcodes:
                    stock_batch.append(_stock_tuple(m, src_site))
            if len(master_batch) >= 5000:
                conn.executemany(
                    """INSERT OR REPLACE INTO icmas(
                      bcode,descr,brand,model,pcode,mcode,acode,size1,ui1,ui2,mtp2,status,location1,
                      costavg,costlast,price1,pricem1,qtyoh2,qtymin,qtymax,src_site
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    master_batch,
                )
                n_ic += len(master_batch)
                master_batch.clear()
                conn.commit()
            if len(stock_batch) >= 5000:
                conn.executemany(
                    """INSERT OR REPLACE INTO icmas_stock(
                      src_site,bcode,qtyoh2,qtymin,qtymax,costavg,costlast,price1,pricem1,location1
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    stock_batch,
                )
                n_st += len(stock_batch)
                stock_batch.clear()
                conn.commit()
        if master_batch:
            conn.executemany(
                """INSERT OR REPLACE INTO icmas(
                  bcode,descr,brand,model,pcode,mcode,acode,size1,ui1,ui2,mtp2,status,location1,
                  costavg,costlast,price1,pricem1,qtyoh2,qtymin,qtymax,src_site
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                master_batch,
            )
            n_ic += len(master_batch)
            conn.commit()
        if stock_batch:
            conn.executemany(
                """INSERT OR REPLACE INTO icmas_stock(
                  src_site,bcode,qtyoh2,qtymin,qtymax,costavg,costlast,price1,pricem1,location1
                ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                stock_batch,
            )
            n_st += len(stock_batch)
            conn.commit()
    print(f"{src_site} icmas_upserted={n_ic} stock_upserted={n_st}", flush=True)
    return n_ic


def run_snapshot(
    *,
    site: str = "hq",
    years: int = 5,
    snap_id: str | None = None,
    include_sites: list[str] | None = None,
) -> Path:
    """Extract ~N years SI/PI + ICMAS into snaps/<id>/snapshot.sqlite.

    For primary site=hq, defaults to extracting both HQ (KSS) and SYP (kss-pc).
    Pass include_sites=['hq'] to keep HQ-only.
    """
    site = site.lower()
    sites = _normalize_sites(primary=site, include_sites=include_sites)
    snap_id = snap_id or _snap_id_now()
    out = snap_path(snap_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    facts_as_of = utc_now_iso()
    cutoff = (date.today() - timedelta(days=int(years) * 365)).isoformat()

    conn = sqlite3.connect(str(out))
    try:
        _init_snap_schema(conn)
        for k, v in (
            ("site", site),
            ("sites", ",".join(sites)),
            ("facts_as_of", facts_as_of),
            ("cutoff", cutoff),
            ("years", str(years)),
            ("snap_id", snap_id),
        ):
            conn.execute("INSERT INTO meta(key,value) VALUES (?,?)", (k, v))
        conn.commit()

        print(
            f"snapshot={snap_id} primary={site} sources={sites} cutoff={cutoff} facts_as_of={facts_as_of}",
            flush=True,
        )

        totals: dict[str, dict[str, int]] = {}
        for src in sites:
            eng = mssql_engine(src)
            n_si, n_pi = _extract_si_pi(conn, eng, src_site=src, cutoff=cutoff)
            # Only headers that appear on this snap's lines (not full 5y account tables).
            pi_bills = _distinct_billnos(conn, "pidet", src)
            si_bills = _distinct_billnos(conn, "sidet", src)
            n_pm = _extract_pimas(conn, eng, src_site=src, billnos=pi_bills)
            n_sm = _extract_simas(conn, eng, src_site=src, billnos=si_bills)
            totals[src] = {"sidet": n_si, "pidet": n_pi, "pimas": n_pm, "simas": n_sm}

        bcodes = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT bcode FROM sidet UNION SELECT DISTINCT bcode FROM pidet"
            )
        }
        print(f"active_bcodes={len(bcodes)}", flush=True)

        # Prefer primary-site ICMAS masters; always write per-site stock (HQ + SYP QTYOH2).
        primary_eng = mssql_engine(site)
        _extract_icmas_for_bcodes(
            conn,
            primary_eng,
            src_site=site,
            bcodes=bcodes,
            write_master=True,
            only_missing=False,
            write_stock=True,
        )
        for src in sites:
            if src == site:
                continue
            eng = mssql_engine(src)
            _extract_icmas_for_bcodes(
                conn,
                eng,
                src_site=src,
                bcodes=bcodes,
                write_master=True,
                only_missing=True,
                write_stock=True,
            )

        n_ic = conn.execute("SELECT COUNT(*) FROM icmas").fetchone()[0]
        print(f"icmas_total={n_ic}", flush=True)

        meta_path = out.parent / "meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "snap_id": snap_id,
                    "site": site,
                    "sites": sites,
                    "facts_as_of": facts_as_of,
                    "cutoff": cutoff,
                    "years": years,
                    "totals": totals,
                    "path": str(out),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"done snap={out}", flush=True)
        print(f"snaps_dir={snaps_dir()}", flush=True)
        return out
    finally:
        conn.close()


def _write_snap_meta_json(out: Path, patch: dict) -> None:
    meta_path = out.parent / "meta.json"
    extra: dict = {}
    if meta_path.is_file():
        try:
            extra = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            extra = {}
    extra.update(patch)
    meta_path.write_text(json.dumps(extra, indent=2), encoding="utf-8")


def _resolve_snap_sites(meta: dict[str, str], sites: list[str] | None) -> list[str]:
    if sites is None:
        raw = meta.get("sites") or meta.get("site") or "hq,syp"
        sites = [s.strip() for s in raw.split(",") if s.strip()]
    if not sites:
        sites = ["hq", "syp"]
    return sites


def _snap_active_bcodes(conn: sqlite3.Connection) -> set[str]:
    return {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT bcode FROM sidet UNION SELECT DISTINCT bcode FROM pidet"
        )
    }


def enrich_snapshot(
    *,
    snap: str = "latest",
    sites: list[str] | None = None,
    pimas: bool = True,
    simas: bool = True,
    stock: bool = True,
    billno_scoped: bool = True,
) -> Path:
    """Patch an existing snap (no full SI/PI re-extract).

    Selective flags let you refresh only PIMAS, SIMAS, and/or ICMAS stock.
    By default account headers are scoped to billnos already present on snap lines.
    """
    from src.kcw.product_insight_store import resolve_snap_id

    if not (pimas or simas or stock):
        raise ValueError("enrich_snapshot: enable at least one of pimas/simas/stock")

    snap_id = resolve_snap_id(snap)
    out = snap_path(snap_id)
    conn = sqlite3.connect(str(out))
    try:
        _ensure_ops_tables(conn)
        meta = {r[0]: r[1] for r in conn.execute("SELECT key, value FROM meta")}
        cutoff = meta.get("cutoff") or (date.today() - timedelta(days=5 * 365)).isoformat()
        sites = _resolve_snap_sites(meta, sites)
        bcodes = _snap_active_bcodes(conn) if stock else set()
        print(
            f"enrich snap={snap_id} sites={sites} cutoff={cutoff} "
            f"pimas={pimas} simas={simas} stock={stock} "
            f"billno_scoped={billno_scoped} bcodes={len(bcodes)}",
            flush=True,
        )
        totals: dict[str, int] = {}
        for src in sites:
            eng = mssql_engine(src)
            if pimas:
                bills = _distinct_billnos(conn, "pidet", src) if billno_scoped else None
                totals[f"{src}_pimas"] = _extract_pimas(
                    conn,
                    eng,
                    src_site=src,
                    cutoff=None if billno_scoped else cutoff,
                    billnos=bills,
                )
            if simas:
                bills = _distinct_billnos(conn, "sidet", src) if billno_scoped else None
                totals[f"{src}_simas"] = _extract_simas(
                    conn,
                    eng,
                    src_site=src,
                    cutoff=None if billno_scoped else cutoff,
                    billnos=bills,
                )
            if stock:
                write_master = src == (meta.get("site") or "hq")
                _extract_icmas_for_bcodes(
                    conn,
                    eng,
                    src_site=src,
                    bcodes=bcodes,
                    write_master=write_master,
                    only_missing=not write_master,
                    write_stock=True,
                )
        n_st = conn.execute("SELECT src_site, COUNT(*) FROM icmas_stock GROUP BY 1").fetchall()
        print(f"icmas_stock={dict(n_st)} accounts={totals}", flush=True)
        now = utc_now_iso()
        conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
            ("enriched_at", now),
        )
        conn.commit()
        patch = {
            "enriched_at": now,
            "icmas_stock": {k: v for k, v in n_st},
            "enrich_flags": {"pimas": pimas, "simas": simas, "stock": stock, "billno_scoped": billno_scoped},
        }
        if any(k.endswith("_pimas") for k in totals):
            patch["pimas_enrich"] = {k: v for k, v in totals.items() if k.endswith("_pimas")}
        if any(k.endswith("_simas") for k in totals):
            patch["simas_enrich"] = {k: v for k, v in totals.items() if k.endswith("_simas")}
        _write_snap_meta_json(out, patch)
        print(f"enriched snap={out}", flush=True)
        return out
    finally:
        conn.close()


def _incremental_since(conn: sqlite3.Connection, meta: dict[str, str], lookback_days: int = 1) -> str:
    """Date to re-pull from: max SI/PI billdate minus lookback, else facts_as_of/cutoff."""
    row = conn.execute(
        """
        SELECT MAX(d) FROM (
          SELECT MAX(billdate) AS d FROM sidet
          UNION ALL
          SELECT MAX(billdate) AS d FROM pidet
        )
        """
    ).fetchone()
    raw = (row[0] if row else None) or meta.get("facts_as_of") or meta.get("cutoff")
    if not raw:
        return (date.today() - timedelta(days=7)).isoformat()
    try:
        base = date.fromisoformat(str(raw)[:10])
    except ValueError:
        base = date.today() - timedelta(days=7)
    return (base - timedelta(days=max(0, int(lookback_days)))).isoformat()


def refresh_snapshot_incremental(
    *,
    snap: str = "latest",
    sites: list[str] | None = None,
    lookback_days: int = 1,
) -> Path:
    """In-place weekly-style refresh: append recent SI/PI, scoped accounts, restock ICMAS.

    Keeps the same snap id. Deletes+reloads SI/PI from (max_billdate - lookback) so
    late postings do not duplicate. Then pulls PIMAS/SIMAS only for snap billnos and
    refreshes HQ+SYP stock for active bcodes.
    """
    from src.kcw.product_insight_store import resolve_snap_id

    snap_id = resolve_snap_id(snap)
    out = snap_path(snap_id)
    conn = sqlite3.connect(str(out))
    try:
        _ensure_ops_tables(conn)
        meta = {r[0]: r[1] for r in conn.execute("SELECT key, value FROM meta")}
        sites = _resolve_snap_sites(meta, sites)
        since = _incremental_since(conn, meta, lookback_days=lookback_days)
        facts_as_of = utc_now_iso()
        print(
            f"incremental refresh snap={snap_id} sites={sites} since={since} "
            f"lookback_days={lookback_days}",
            flush=True,
        )

        totals: dict[str, dict[str, int]] = {}
        for src in sites:
            eng = mssql_engine(src)
            n_si, n_pi = _extract_si_pi(
                conn, eng, src_site=src, cutoff=since, replace_from=True
            )
            pi_bills = _distinct_billnos(conn, "pidet", src)
            si_bills = _distinct_billnos(conn, "sidet", src)
            # Only re-fetch headers for bills in the refresh window to keep it cheap.
            pi_recent = [
                str(r[0]).strip()
                for r in conn.execute(
                    "SELECT DISTINCT billno FROM pidet WHERE src_site=? AND billdate >= ?",
                    (src, since),
                )
                if r[0]
            ]
            si_recent = [
                str(r[0]).strip()
                for r in conn.execute(
                    "SELECT DISTINCT billno FROM sidet WHERE src_site=? AND billdate >= ?",
                    (src, since),
                )
                if r[0]
            ]
            print(
                f"{src} account refresh recent_pi={len(pi_recent)}/{len(pi_bills)} "
                f"recent_si={len(si_recent)}/{len(si_bills)}",
                flush=True,
            )
            n_pm = _extract_pimas(conn, eng, src_site=src, billnos=pi_recent)
            n_sm = _extract_simas(conn, eng, src_site=src, billnos=si_recent)
            totals[src] = {"sidet": n_si, "pidet": n_pi, "pimas": n_pm, "simas": n_sm}

        bcodes = _snap_active_bcodes(conn)
        print(f"active_bcodes={len(bcodes)}", flush=True)
        primary = meta.get("site") or "hq"
        for src in sites:
            eng = mssql_engine(src)
            _extract_icmas_for_bcodes(
                conn,
                eng,
                src_site=src,
                bcodes=bcodes,
                write_master=(src == primary),
                only_missing=(src != primary),
                write_stock=True,
            )

        conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
            ("facts_as_of", facts_as_of),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
            ("refreshed_at", facts_as_of),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
            ("refresh_since", since),
        )
        conn.commit()
        n_st = conn.execute("SELECT src_site, COUNT(*) FROM icmas_stock GROUP BY 1").fetchall()
        _write_snap_meta_json(
            out,
            {
                "facts_as_of": facts_as_of,
                "refreshed_at": facts_as_of,
                "refresh_since": since,
                "refresh_totals": totals,
                "icmas_stock": {k: v for k, v in n_st},
                "path": str(out),
            },
        )
        print(f"incremental refresh done snap={out} since={since}", flush=True)
        return out
    finally:
        conn.close()
