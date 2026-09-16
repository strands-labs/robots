"""``python -m strands_robots dashboard``: dispatched, loopback by default, LAN bind needs a guard."""

from __future__ import annotations

import pytest

from strands_robots.__main__ import _COMMANDS
from strands_robots.dashboard import cli


def test_the_command_is_dispatched():
    assert "dashboard" in _COMMANDS


def test_defaults_are_loopback_8090():
    args = cli.build_parser().parse_args([])
    assert (args.host, args.port, args.open) == ("127.0.0.1", 8090, False)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_binds_need_no_guard(host):
    assert cli.bind_verdict(host, guarded=False) is None


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "::", "robot.local"])
def test_a_lan_bind_without_a_guard_is_refused_with_the_remedy(host):
    verdict = cli.bind_verdict(host, guarded=False)
    assert verdict and "passkey" in verdict and "DASHBOARD_AUTH_TOKEN" in verdict


def test_a_lan_bind_with_a_guard_proceeds():
    assert cli.bind_verdict("0.0.0.0", guarded=True) is None


def test_main_refuses_the_unguarded_lan_bind_before_importing_uvicorn(monkeypatch, capsys):
    monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "0")
    monkeypatch.setattr("strands_robots.dashboard.settings.get", lambda *a, **k: None)
    assert cli.main(["--host", "0.0.0.0"]) == 2
    assert "refusing to bind 0.0.0.0" in capsys.readouterr().err


def test_main_refuses_a_port_outside_the_range(capsys):
    assert cli.main(["--port", "70000"]) == 2
