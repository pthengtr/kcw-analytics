#!/usr/bin/env python3
"""Print a reachable KSS address for rclone.

Probes the same comma list as PARTS9 SQL (KSS.local, KSS, last-known LAN IP)
on TCP 445. rclone uses Go DNS and often cannot resolve .local / NetBIOS, so
we print the system-resolved IP of the first host that accepted the port.
"""
from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.kcw.mssql_host import pick_mssql_server  # noqa: E402


def _env_get(key: str) -> str:
    env_path = REPO / ".env"
    if not env_path.is_file():
        return ""
    prefix = key + "="
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip().strip('"').strip("'")
    return ""


def resolve_ipv4(host: str, port: int) -> str:
    parts = host.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        return host
    infos = socket.getaddrinfo(host, port, family=socket.AF_INET, type=socket.SOCK_STREAM)
    return infos[0][4][0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=445)
    args = parser.parse_args()
    configured = (
        _env_get("KSS_SMB_HOST")
        or _env_get("PARTS9_HQ_SERVER")
        or _env_get("KSS_SERVER")
        or "KSS.local,KSS,192.168.1.99"
    )
    host = pick_mssql_server(configured, port=args.port)
    ip = resolve_ipv4(host, args.port)
    print(f"{host} -> {ip}", file=sys.stderr)
    print(ip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
