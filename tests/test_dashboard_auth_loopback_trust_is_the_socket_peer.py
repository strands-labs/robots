# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin: the damaged-store enrollment gate trusts the socket peer, not a header.

The re-seal guard in ``begin_registration`` decides whether the caller is "the
person at the machine" and may therefore enroll the first passkey over a
corrupted store. It used to make that decision on ``_client_ip(request)``, whose
first three reads are ``cf-connecting-ip``, ``x-forwarded-for`` and
``x-real-ip`` -- all set by whoever is calling. So a LAN attacker who reached a
dashboard whose store had just been corrupted sent one header,
``CF-Connecting-IP: 127.0.0.1``, and the guard let them through.

That is a one-way door for the deployment rather than a transient: the first
enrollment seals the dashboard, and ``rp_id_verdict`` and the last-passkey rule
are then built around the store the attacker owns. It also contradicted
``_client_ip``'s own stated contract -- "Best-effort client identity for the
per-ip cap only -- NEVER for trust" -- and the pre-existing pin
(``test_a_stranger_cannot_seize_the_dashboard_through_a_disk_error``) sent no
forwarded header, so the suite stayed green over the bypass.

``_socket_peer`` reads only the connection's own peer address, and these cells
send the hostile header explicitly. ``_client_ip`` keeps its header chain,
because the per-ip fairness cap it feeds genuinely wants the caller's best-known
identity; the split is the point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

from strands_robots.dashboard import auth

# Headers a caller can set that _client_ip reads ahead of the socket peer.
_SPOOFABLE = ["cf-connecting-ip", "x-forwarded-for", "x-real-ip"]


class FakeRequest:
    """A request whose socket peer and headers can disagree."""

    def __init__(self, headers: dict[str, str] | None = None, client_host: str = "127.0.0.1") -> None:
        self.headers = {"host": "localhost:8090", **(headers or {})}
        self.client = type("C", (), {"host": client_host})()


@pytest.fixture(autouse=True)
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    for k in ("STRANDS_DASH_AUTH_ENABLED", "STRANDS_DASH_AUTH_RP_ID", "STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    auth._cache = {}
    auth._corrupt = None
    yield
    auth._corrupt = None


def _corrupt_store(tmp_path: Path) -> None:
    """Leave the store as a killed process would: truncated JSON."""
    (tmp_path / "auth.json").write_text('{"credentials": [{"id": "AAA')
    auth._load()


class TestAForwardedHeaderCannotForgeLocality:
    """The regression: a stranger claiming to be loopback in a header must
    still be refused."""

    @pytest.mark.parametrize("header", _SPOOFABLE)
    @pytest.mark.parametrize("claimed", ["127.0.0.1", "::1", "localhost"])
    def test_the_stranger_is_still_refused(self, tmp_path: Path, header: str, claimed: str) -> None:
        _corrupt_store(tmp_path)
        request = FakeRequest({header: claimed}, client_host="203.0.113.9")
        with pytest.raises(HTTPException) as raised:
            auth.begin_registration(request, label="attacker")
        assert raised.value.status_code == 403

    def test_a_chained_forwarded_for_is_no_better(self, tmp_path: Path) -> None:
        _corrupt_store(tmp_path)
        request = FakeRequest({"x-forwarded-for": "127.0.0.1, 10.0.0.1"}, client_host="203.0.113.9")
        with pytest.raises(HTTPException) as raised:
            auth.begin_registration(request, label="attacker")
        assert raised.value.status_code == 403


class TestThePersonAtTheMachineIsStillLetIn:
    """The guard exists so the owner can recover; a hostile header on a
    genuinely local connection must not cost them that."""

    @pytest.mark.parametrize("peer", ["127.0.0.1", "::1"])
    def test_a_local_peer_can_recover(self, tmp_path: Path, peer: str) -> None:
        _corrupt_store(tmp_path)
        opts = auth.begin_registration(FakeRequest(client_host=peer), label="recovery")
        assert opts.get("challenge_id")

    def test_a_local_peer_is_not_locked_out_by_a_hostile_header(self, tmp_path: Path) -> None:
        _corrupt_store(tmp_path)
        request = FakeRequest({"cf-connecting-ip": "203.0.113.9"}, client_host="127.0.0.1")
        opts = auth.begin_registration(request, label="recovery")
        assert opts.get("challenge_id")


class TestTheTwoReadersAreDeliberatelyDifferent:
    """``_client_ip`` keeps its header chain for the fairness cap;
    ``_socket_peer`` never grows one."""

    @pytest.mark.parametrize("header", _SPOOFABLE)
    def test_socket_peer_ignores_the_header_client_ip_honours(self, header: str) -> None:
        request = FakeRequest({header: "198.51.100.7"}, client_host="203.0.113.9")
        assert auth._client_ip(request) == "198.51.100.7"
        assert auth._socket_peer(request) == "203.0.113.9"

    def test_socket_peer_is_none_when_there_is_no_peer(self) -> None:
        assert auth._socket_peer(object()) is None

    def test_a_none_peer_is_not_loopback(self) -> None:
        assert not auth.client_is_loopback(auth._socket_peer(object()))
