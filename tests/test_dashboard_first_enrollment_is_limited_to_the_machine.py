# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin: the passkey that seals the dashboard can only be enrolled from the machine.

``auth_enabled()`` IS ``has_credentials()``, so the FIRST enrollment is the request
that decides who owns the fleet: once it succeeds, every later caller is locked out
and ``rp_id_verdict`` plus the last-passkey rule are built around the store that
enrollment created. It is a one-way door.

Three branches could gate it and only two did. A bootstrap token is checked when one
is configured, and a loopback check applied when the store was *unreadable* and no
token was set. The ordinary case -- a fresh install, an intact empty store, no token
configured -- matched neither, so it was ungated. Measured on ``main`` at ``19329cd``:

    fresh store,   peer 203.0.113.9  ->  ACCEPTED   (challenge issued)
    damaged store, peer 203.0.113.9  ->  refused    (403)

That is backwards. The damaged store is the rarer route and needs a disk error to
happen first; the fresh install is every install's first minute, and it is the one
where seizing enrollment costs an attacker nothing. So the weaker case carried the
guard and the stronger case did not.

The gate is now the one predicate ``first_time and not required`` for both routes,
and these cells pin each way through it. The wording still differs per cause, because
a refusal that blames a disk error where none occurred sends the reader to the wrong
place; that split is asserted here rather than left to prose.

Scope note: this bounds the *first* enrollment only. Later enrollments are a session
decision belonging to the route, and the last cell pins that this gate stops applying
once a credential exists -- an owner adding a phone from their sofa is not this
threat, and a gate that never lifted would be a lockout rather than a guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

from strands_robots.dashboard import auth

# Headers _client_ip reads ahead of the socket peer, and which a caller can therefore set.
_SPOOFABLE = ["cf-connecting-ip", "x-forwarded-for", "x-real-ip"]

_STRANGER = "203.0.113.9"  # TEST-NET-3, never a real peer


class FakeRequest:
    """A request whose socket peer, headers and Host can be set independently."""

    def __init__(
        self,
        client_host: str | None = _STRANGER,
        headers: dict[str, str] | None = None,
        host: str = "localhost:8090",
    ) -> None:
        self.headers = {"host": host, **(headers or {})}
        # client_host=None models a connection with no peer address, which is what an
        # ASGI scope carries for a unix socket or a broken transport.
        self.client = None if client_host is None else type("C", (), {"host": client_host})()


@pytest.fixture(autouse=True)
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    for k in ("STRANDS_DASH_AUTH_ENABLED", "STRANDS_DASH_AUTH_RP_ID", "STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    auth._cache = {}
    auth._corrupt = None
    yield
    auth._corrupt = None


class TestTheFirstEnrollmentOnAFreshStore:
    """No disk error, no token configured: the case that was ungated."""

    def test_a_stranger_cannot_seize_a_fresh_dashboard(self) -> None:
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest(), label="attacker")
        assert e.value.status_code == 403

    def test_the_refusal_names_the_two_ways_forward(self) -> None:
        """A 403 an operator cannot act on is a support ticket, so both remedies are named."""
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest(), label="attacker")
        detail = e.value.detail
        assert "the machine itself" in detail
        assert "STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN" in detail

    def test_the_refusal_does_not_blame_a_disk_error_that_did_not_happen(self) -> None:
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest(), label="attacker")
        assert "unreadable" not in e.value.detail
        assert "corrupt-" not in e.value.detail
        assert auth.store_corruption() is None, "no damage was staged, so none may be reported"

    def test_nothing_is_written_by_a_refused_enrollment(self) -> None:
        """A refusal must not leave the store half-created; auth stays off, not sealed-by-nobody."""
        with pytest.raises(HTTPException):
            auth.begin_registration(FakeRequest(), label="attacker")
        assert auth.auth_enabled() is False
        assert auth._load().get("credentials") == []

    def test_the_person_at_the_machine_still_enrolls_with_no_configuration(self) -> None:
        """The zero-config local path is the common one and must stay frictionless."""
        opts = auth.begin_registration(FakeRequest(client_host="127.0.0.1"), label="owner")
        assert opts.get("challenge_id")

    @pytest.mark.parametrize("peer", ["127.0.0.1", "127.0.0.53", "::1", "localhost"])
    def test_every_spelling_of_the_machine_itself_is_accepted(self, peer: str) -> None:
        opts = auth.begin_registration(FakeRequest(client_host=peer), label="owner")
        assert opts.get("challenge_id")


class TestWhatCountsAsTheMachine:
    """The gate grants ownership, so it reads only facts about the connection."""

    @pytest.mark.parametrize("header", _SPOOFABLE)
    def test_a_forwarded_header_cannot_manufacture_loopback(self, header: str) -> None:
        """_client_ip's own contract: best-effort identity, NEVER for trust.

        The damaged-store route was already pinned against this. The fresh route is the
        one an attacker reaches without needing a disk error first, so it is pinned too.
        """
        request = FakeRequest(client_host=_STRANGER, headers={header: "127.0.0.1"})
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(request, label="attacker")
        assert e.value.status_code == 403

    def test_a_connection_with_no_peer_is_not_the_machine(self) -> None:
        """Fail closed on an unknown peer: absence of evidence is not evidence of locality."""
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest(client_host=None), label="unknown")
        assert e.value.status_code == 403


class TestTheBootstrapTokenIsTheRemoteRoute:
    """A headless robot has no browser on it, so the token is how its owner enrolls."""

    def test_the_right_token_enrolls_from_anywhere(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", "let-me-in")
        opts = auth.begin_registration(FakeRequest(), label="remote-owner", bootstrap="let-me-in")
        assert opts.get("challenge_id"), "the documented remote path must not be closed by this gate"

    def test_a_wrong_token_is_refused_from_anywhere(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", "let-me-in")
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest(), label="attacker", bootstrap="guess")
        assert e.value.status_code == 403

    def test_a_configured_token_is_required_even_at_the_machine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Loopback does not waive a token the operator deliberately set."""
        monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", "let-me-in")
        with pytest.raises(HTTPException):
            auth.begin_registration(FakeRequest(client_host="127.0.0.1"), label="owner")


class TestTheGateIsBoundedToTheFirstEnrollment:
    """It guards a one-way door, not the dashboard's ongoing use."""

    def test_once_a_credential_exists_the_gate_no_longer_applies(self) -> None:
        """An owner adding a second passkey remotely is the route's decision, not this one.

        Pinned because a gate on ``first_time`` that leaked into later enrollments would
        lock an operator out of their own dashboard from anywhere but the console.
        """
        auth._save({"credentials": [{"id": "AAAA", "name": "existing"}]})
        auth._cache = {}
        assert auth.auth_enabled() is True

        opts = auth.begin_registration(FakeRequest(), label="second-key")
        assert opts.get("challenge_id")


class TestWhichRefusalTheCallerGets:
    """Two gates can refuse the same request; the report has to be the useful one."""

    def test_an_unusable_rp_id_is_reported_as_such_not_as_an_ownership_refusal(self) -> None:
        """A bare IP cannot host passkeys at all, so that is the diagnosis worth giving.

        Pins the ordering: the rp_id check runs first, because a caller refused for an
        unusable rp_id would not be able to enroll from the machine either, and telling
        them to move to the console would send them somewhere that also fails.
        """
        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest(host="192.168.1.166:8090"), label="lan")
        assert e.value.status_code == 400
        assert "BOOTSTRAP_TOKEN" not in e.value.detail

    def test_a_damaged_store_still_reports_the_damage_and_where_the_bytes_went(self, tmp_path: Path) -> None:
        """The more specific cause keeps its more specific message."""
        (tmp_path / "auth.json").write_text('{"credentials": [{"id": "AAA')  # truncated JSON
        auth._cache = {}
        auth._corrupt = None
        auth._load()

        with pytest.raises(HTTPException) as e:
            auth.begin_registration(FakeRequest(), label="attacker")
        assert e.value.status_code == 403
        assert "unreadable" in e.value.detail and "corrupt-" in e.value.detail
