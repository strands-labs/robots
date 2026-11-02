"""``python -m strands_robots.dashboard``: the one command the docs hero offers."""

from __future__ import annotations

import pytest

from strands_robots.dashboard.__main__ import parse


def test_defaults_are_the_browser_presets_robot_on_8090_loopback() -> None:
    a = parse([])
    assert (a.robots, a.port, a.host, a.model) == ([], 8090, "127.0.0.1", None)  # main() fills in browser_preset()


def test_robots_and_port() -> None:
    a = parse(["so101", "g1", "--port", "8091"])
    assert a.robots == ["so101", "g1"] and a.port == 8091


def test_a_bad_port_is_refused() -> None:
    with pytest.raises(SystemExit):
        parse(["--port", "x"])
