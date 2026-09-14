"""Derive sales channel from live PARTS9 billno + JOURMODE (no BILLTYPE_STD on KSS)."""

from __future__ import annotations


def billtype_std(billno: str | None, jourmode: str | None = None) -> str:
    b = (billno or "").strip().upper()
    j = str(jourmode).strip() if jourmode is not None else ""
    if j == "0":
        return "TAR_OR_JOUR0"
    if b.startswith("CNTAD") or b.startswith("3CNTAD"):
        return "CNTAD"
    if b.startswith("TAD"):
        return "TAD"
    if b.startswith("TFV") or b.startswith("3TFV"):
        return "TFV"
    if (
        b.startswith("TF")
        or b.startswith("3TF")
        or b.startswith("CNTF")
        or b.startswith("3CNTF")
    ):
        return "TF"
    if b.startswith("TD") or b.startswith("3TD"):
        return "TD"
    if b.startswith("TR") or b.startswith("3TR"):
        return "TR"
    if b.startswith("CN") or b.startswith("3CN"):
        return "CN"
    if b.startswith("DN") or b.startswith("3DN"):
        return "DN"
    return "UNKNOWN"


def channel_of(
    billno: str | None,
    jourmode: str | None = None,
    *,
    src_site: str | None = None,
) -> str:
    std = billtype_std(billno, jourmode)
    b = (billno or "").strip().upper()
    src = (src_site or "").strip().lower()
    if std in ("TAR_OR_JOUR0", "TAR"):
        return "excluded"
    if std in ("TF", "TFV") or b.startswith("CNTF") or b.startswith("3CNTF"):
        return "transfer"
    if std in ("TAD", "CNTAD"):
        return "online"
    # SYP PARTS9 lines (and HQ bills prefixed 3*) are branch counter sales.
    if src == "syp" or b.startswith("3"):
        return "syp_store"
    return "hq_store"


CHANNEL_LEGEND = """
CHANNEL LEGEND (must use correctly):
- TAD / CNTAD / channel=online = online sales (ONLINE), not HQ counter
- TF / TFV / channel=transfer = HQ↔SYP stock transfer, NOT customer sales
- hq_store / syp_store = counter/branch customer sales (Facts may include both HQ KSS + SYP kss-pc SI/PI)
- QTYOH2 / QTYMIN in Facts.stock are snapshot-time (facts_as_of) and go stale within days — policy only, not live / not order-now
- channel_qty_5y aggregates qty by channel across sources; prefer recent.* / derived.demand for ops
""".strip()
