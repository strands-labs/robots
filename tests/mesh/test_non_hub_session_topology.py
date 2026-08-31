"""A non-hub mesh session opens in the mode that receives relayed traffic.

The machine's first process wins the ``STRANDS_MESH_PORT`` listener and becomes
the hub; every later process falls back to connecting to it. Two entry points
build that fallback -- :func:`~strands_robots.mesh.session.get_session` and
:func:`~strands_robots.mesh.session._get_zenoh_session_directly` -- and both
must produce the same topology, because a Zenoh 1.x *peer* refuses traffic
relayed by an intermediary. A peer-mode child therefore hears nothing a sibling
child publishes, which is the teleop failure where a leader published frames to
a follower whose counters stayed at zero.

These pin the config the fallback hands to ``zenoh.open`` rather than a live
session, so they run without the ``zenoh`` extra installed.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import strands_robots.mesh.session as mod

#: The two fallback builders, driven identically. Both are reached only after
#: the hub-listener attempt fails, so ``zenoh.open`` raises once then succeeds.
ENTRY_POINTS = ("get_session", "_get_zenoh_session_directly")


def _drive_fallback(entry: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Run one fallback builder and return the config keys it set.

    Args:
        entry: Attribute name of the session builder on the session module.
        monkeypatch: Used to clear explicit-endpoint env and stub the config.

    Returns:
        The ``insert_json5`` key -> value mapping the builder produced.
    """
    for var in ("ZENOH_LISTEN", "ZENOH_CONNECT", "STRANDS_MESH_BACKEND"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(mod, "_SESSION", None)
    monkeypatch.setattr(mod, "_SESSION_REFS", 0)

    # One dict per built config: the hub-listener attempt builds its own
    # before the fallback does, and only the LAST one is the topology here.
    configs: list[dict[str, str]] = []

    def _build() -> MagicMock:
        recorded: dict[str, str] = {}
        configs.append(recorded)
        cfg = MagicMock()
        cfg.insert_json5.side_effect = lambda key, val: recorded.__setitem__(key, val)
        return cfg

    monkeypatch.setattr(mod, "_build_config", _build)

    zenoh = MagicMock()
    # First open() is the hub-listener attempt; it must fail so the caller
    # takes the fallback branch under test. The second is the fallback itself.
    zenoh.open.side_effect = [RuntimeError("port already bound"), MagicMock()]

    with patch.dict("sys.modules", {"zenoh": zenoh}):
        session: Any = getattr(mod, entry)()

    assert session is not None, f"{entry} did not open the fallback session"
    assert len(configs) == 2, f"{entry} built {len(configs)} configs, expected hub then fallback"
    return configs[-1]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_non_hub_session_opens_as_client_so_the_hub_relays(entry: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback is a client, not a peer, at both sites.

    A peer would also need its own listener, so the presence of
    ``listen/endpoints`` here is the tell for the topology that drops
    sibling-to-sibling traffic.
    """
    recorded = _drive_fallback(entry, monkeypatch)

    assert recorded.get("mode") == json.dumps("client"), recorded
    assert "listen/endpoints" not in recorded, recorded


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_non_hub_session_retries_the_hub_at_both_sites(entry: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither site may give up on the hub endpoint.

    Without the retry a child whose hub process dies keeps a dead session with
    no error surfaced: the peer just goes dark until it is restarted.
    """
    recorded = _drive_fallback(entry, monkeypatch)

    assert recorded.get("connect/exit_on_failure") == "false", recorded
    assert json.loads(recorded["connect/retry"]) == mod.FALLBACK_RETRY, recorded
    assert json.loads(recorded["connect/endpoints"]) == ["tcp/127.0.0.1:7447"], recorded


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_fallback_mode_peer_restores_direct_peer_links(entry: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """``STRANDS_MESH_FALLBACK_MODE=peer`` opts back into the old topology.

    An operator who arranges direct peer links keeps them, and gets the
    ephemeral listener a peer needs in order to be dialled.
    """
    monkeypatch.setenv("STRANDS_MESH_FALLBACK_MODE", "peer")
    recorded = _drive_fallback(entry, monkeypatch)

    assert json.loads(recorded["listen/endpoints"]) == ["tcp/127.0.0.1:0"], recorded
    assert "mode" not in recorded, recorded
    # The retry property is not what peer mode trades away.
    assert recorded.get("connect/exit_on_failure") == "false", recorded


@pytest.mark.parametrize("bad", ["router", "PEERS", " "])
def test_unrecognised_fallback_mode_warns_and_stays_client(
    bad: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A typo must not take a robot's mesh offline, and must not be silent."""
    monkeypatch.setenv("STRANDS_MESH_FALLBACK_MODE", bad)

    with caplog.at_level("WARNING", logger=mod.logger.name):
        assert mod._fallback_mode() == mod.FALLBACK_MODE_DEFAULT

    if bad.strip():
        assert "STRANDS_MESH_FALLBACK_MODE" in caplog.text
    else:
        # An empty value is "unset", not a typo, so it warrants no warning.
        assert caplog.text == ""
