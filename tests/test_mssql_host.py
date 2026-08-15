from unittest.mock import patch

import pytest

from src.kcw.mssql_host import pick_mssql_server, split_hosts


def test_split_hosts_comma_and_semicolon():
    assert split_hosts("KSS.local, KSS; 192.168.1.99") == [
        "KSS.local",
        "KSS",
        "192.168.1.99",
    ]


def test_single_host_not_probed():
    with patch("src.kcw.mssql_host.tcp_open") as probe:
        assert pick_mssql_server("KSS") == "KSS"
        probe.assert_not_called()


def test_falls_through_to_second_reachable_host():
    def fake(host, port=1433, timeout=1.2):
        return host == "KSS.local"

    with patch("src.kcw.mssql_host.tcp_open", side_effect=fake):
        assert pick_mssql_server("192.168.1.99,KSS.local") == "KSS.local"


def test_smb_port_is_passed_through():
    def fake(host, port=1433, timeout=1.2):
        return port == 445 and host == "KSS.local"

    with patch("src.kcw.mssql_host.tcp_open", side_effect=fake):
        assert pick_mssql_server("192.168.1.99,KSS.local", port=445) == "KSS.local"


def test_raises_when_none_reachable():
    with patch("src.kcw.mssql_host.tcp_open", return_value=False):
        with pytest.raises(ConnectionError):
            pick_mssql_server("192.168.1.99,KSS.local")
