"""Edge-path tests for strands_robots.dashboard.auth - the branches the main
suite never reached (measured at 95% with 15 lines dark; these pin the four
that are security behaviour rather than IO error tolerance).

1. A challenge that survived eviction but aged past _CHAL_TTL is still refused
   at pop time - eviction is housekeeping, the TTL is the guarantee.
2. begin_authentication refuses to run a ceremony whose rp_id is a raw IP,
   even when the operator pinned that IP - browsers reject an IP rpId before
   the ceremony starts, so proceeding would mint challenges nothing can answer.

Expected-origin derivation used to be items 3 and 4 here. It is now a matrix of
its own in test_dashboard_auth_expected_origin_is_the_connection.py, which needs
a real Request to observe the transport scheme these edge cases never had.
"""

from __future__ import annotations

import time

import pytest
from fastapi import HTTPException

from strands_robots.dashboard import auth


class FakeRequest:
    def __init__(self, headers=None):
        self.headers = headers or {"host": "localhost:8090"}
        self.client = None


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    for key in ("ENABLED", "RP_ID", "BOOTSTRAP_TOKEN", "ORIGIN", "TOKEN_TTL"):
        monkeypatch.delenv("STRANDS_DASH_AUTH_" + key, raising=False)
    auth._cache = {}
    auth._challenges.clear()
    yield
    auth._challenges.clear()


# --- 1. challenge TTL is enforced at POP, not only by eviction ---------------


def test_expired_challenge_refused_at_pop():
    cid = auth._stash_challenge("auth", b"chal", ip="1.2.3.4")
    auth._challenges[cid]["t"] = time.time() - auth._CHAL_TTL - 1
    with pytest.raises(HTTPException) as e:
        auth._pop_challenge(cid, "auth")
    assert e.value.status_code == 400
    assert "expired" in str(e.value.detail)
    # and it is consumed: a second pop is invalid, not a retry
    with pytest.raises(HTTPException):
        auth._pop_challenge(cid, "auth")


def test_fresh_challenge_pops_once_then_is_gone():
    cid = auth._stash_challenge("auth", b"chal")
    rec = auth._pop_challenge(cid, "auth")
    assert rec["challenge"] == b"chal"
    with pytest.raises(HTTPException):
        auth._pop_challenge(cid, "auth")


# --- 2. an IP rp_id is refused even when pinned -------------------------------


def _enroll_fake_credential():
    store = auth._load()
    store["credentials"] = [{"id": "abc", "public_key": "cGs", "sign_count": 0, "name": "phone"}]
    auth._save(store)


def test_begin_authentication_refuses_pinned_ip_rp_id(monkeypatch):
    _enroll_fake_credential()
    monkeypatch.setenv("STRANDS_DASH_AUTH_RP_ID", "10.0.0.5")
    req = FakeRequest({"host": "10.0.0.5:8090"})
    with pytest.raises(HTTPException) as e:
        auth.begin_authentication(req)
    assert e.value.status_code == 400
    assert "relying-party id" in str(e.value.detail)


def test_begin_authentication_needs_enrollment_first():
    with pytest.raises(HTTPException) as e:
        auth.begin_authentication(FakeRequest())
    assert "no credentials" in str(e.value.detail)


# --- bonus: a garbage TOKEN_TTL cannot break token minting --------------------


def test_token_ttl_garbage_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_TOKEN_TTL", "banana")
    assert auth._token_ttl() == 86400
    token = auth.issue_token("cred1")
    assert auth.session_is_valid(token)
