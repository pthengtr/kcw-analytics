"""Snapshot ICMAS + SI/PI from PARTS9 into a local SQLite snap for insight generation."""

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
          price1 REAL
        );
        CREATE TABLE IF NOT EXISTS sidet (
          bcode TEXT NOT NULL,
          billno TEXT,
          billdate TEXT,
          qty REAL,
          ui TEXT,
          price REAL,
          amount REAL,
          jourmode TEXT
        );
        CREATE INDEX IF NOT EXISTS sidet_bcode_idx ON sidet (bcode);
        CREATE INDEX IF NOT EXISTS sidet_billdate_idx ON sidet (billdate);
        CREATE TABLE IF NOT EXISTS pidet (
          bcode TEXT NOT NULL,
          billno TEXT,
          billdate TEXT,
          qty REAL,
          ui TEXT,
          price REAL,
          amount REAL,
          billtype TEXT
        );
        CREATE INDEX IF NOT EXISTS pidet_bcode_idx ON pidet (bcode);
        CREATE INDEX IF NOT EXISTS pidet_billdate_idx ON pidet (billdate);
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


def run_snapshot(*, site: str = "hq", years: int = 5, snap_id: str | None = None) -> Path:
    """Extract ~N years SI/PI + ICMAS masters into snaps/<id>/snapshot.sqlite."""
    site = site.lower()
    if site not in ("hq", "syp"):
        raise ValueError("site must be hq or syp")
    snap_id = snap_id or _snap_id_now()
    out = snap_path(snap_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    facts_as_of = utc_now_iso()
    cutoff = (date.today() - timedelta(days=int(years) * 365)).isoformat()
    eng = mssql_engine(site)

    conn = sqlite3.connect(str(out))
    try:
        _init_snap_schema(conn)
        for k, v in (
            ("site", site),
            ("facts_as_of", facts_as_of),
            ("cutoff", cutoff),
            ("years", str(years)),
            ("snap_id", snap_id),
        ):
            conn.execute("INSERT INTO meta(key,value) VALUES (?,?)", (k, v))
        conn.commit()

        print(f"snapshot={snap_id} site={site} cutoff={cutoff} facts_as_of={facts_as_of}")
        print("extracting SIDET+SIMAS …")
        with eng.connect() as db:
            # SIDET joined to SIMAS for JOURMODE (channel mapping)
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
                        )
                    )
                if len(batch) >= 20000:
                    conn.executemany(
                        "INSERT INTO sidet(bcode,billno,billdate,qty,ui,price,amount,jourmode) VALUES (?,?,?,?,?,?,?,?)",
                        batch,
                    )
                    n_si += len(batch)
                    print(f"  sidet rows={n_si}", flush=True)
                    batch.clear()
                    conn.commit()
            if batch:
                conn.executemany(
                    "INSERT INTO sidet(bcode,billno,billdate,qty,ui,price,amount,jourmode) VALUES (?,?,?,?,?,?,?,?)",
                    batch,
                )
                n_si += len(batch)
                conn.commit()
            print(f"sidet_total={n_si}")

            print("extracting PIDET …")
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
                        )
                    )
                if len(batch) >= 20000:
                    conn.executemany(
                        "INSERT INTO pidet(bcode,billno,billdate,qty,ui,price,amount,billtype) VALUES (?,?,?,?,?,?,?,?)",
                        batch,
                    )
                    n_pi += len(batch)
                    print(f"  pidet rows={n_pi}", flush=True)
                    batch.clear()
                    conn.commit()
            if batch:
                conn.executemany(
                    "INSERT INTO pidet(bcode,billno,billdate,qty,ui,price,amount,billtype) VALUES (?,?,?,?,?,?,?,?)",
                    batch,
                )
                n_pi += len(batch)
                conn.commit()
            print(f"pidet_total={n_pi}")

            # ICMAS for BCODEs that appear in SI or PI
            print("extracting ICMAS for active BCODEs …")
            bcodes = {
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT bcode FROM sidet UNION SELECT DISTINCT bcode FROM pidet"
                )
            }
            print(f"active_bcodes={len(bcodes)}")
            # Pull full ICMAS once, filter in Python (simpler than huge IN)
            col_list = ", ".join(ICMAS_COLS)
            ic_sql = text(f"SELECT {col_list} FROM dbo.ICMAS")
            result = db.execute(ic_sql)
            batch = []
            n_ic = 0
            while True:
                rows = result.fetchmany(5000)
                if not rows:
                    break
                for r in rows:
                    m = {k.upper(): _ser(v) for k, v in r._mapping.items()}
                    bcode = (m.get("BCODE") or "").strip()
                    if bcode not in bcodes:
                        continue
                    batch.append(
                        (
                            bcode,
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
                        )
                    )
                if len(batch) >= 5000:
                    conn.executemany(
                        """INSERT OR REPLACE INTO icmas(
                          bcode,descr,brand,model,pcode,mcode,acode,size1,ui1,ui2,mtp2,status,location1,costavg,costlast,price1
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        batch,
                    )
                    n_ic += len(batch)
                    batch.clear()
                    conn.commit()
            if batch:
                conn.executemany(
                    """INSERT OR REPLACE INTO icmas(
                      bcode,descr,brand,model,pcode,mcode,acode,size1,ui1,ui2,mtp2,status,location1,costavg,costlast,price1
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    batch,
                )
                n_ic += len(batch)
                conn.commit()
            print(f"icmas_total={n_ic}")

        meta_path = out.parent / "meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "snap_id": snap_id,
                    "site": site,
                    "facts_as_of": facts_as_of,
                    "cutoff": cutoff,
                    "years": years,
                    "path": str(out),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"done snap={out}")
        print(f"snaps_dir={snaps_dir()}")
        return out
    finally:
        conn.close()
