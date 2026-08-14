"""Pick a reachable SQL Server host when LAN DHCP / mDNS / Tailscale differ."""

from __future__ import annotations

import socket


def split_hosts(value: str) -> list[str]:
    return [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]


def tcp_open(host: str, port: int = 1433, timeout: float = 1.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def pick_mssql_server(configured: str, *, port: int = 1433) -> str:
    """Return the first host in a comma list that accepts TCP ``port``.

    A single name is returned as-is (tests / Windows NetBIOS). Multiple
    candidates are probed in order so a stale ``192.168.1.99`` can fall through
    to ``KSS.local`` / ``KSS`` (HQ LAN) when a last-known IP is stale. Do not include SYP ``kss-pc`` on HQ.
    """
    hosts = split_hosts(configured or "")
    if not hosts:
        raise ValueError("SQL Server host is empty")
    if len(hosts) == 1:
        return hosts[0]
    for host in hosts:
        if tcp_open(host, port=port):
            return host
    raise ConnectionError(
        "SQL Server port %s not reachable on: %s" % (port, ", ".join(hosts))
    )
