"""Unit tests for strands_robots.dashboard.auth (WebAuthn + JWT sessions).

No real authenticator is available in CI, so the ceremonies are exercised up
to options generation; the store, JWT session, hot-reload and loopback rules
are covered end to end.
"""

from __future__ import annotations

import json
import os
import stat
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from strands_robots.dashboard import auth


class FakeRequest:
    """A request as it arrived: over a SCHEME, from a PEER, carrying headers.

    Both are properties of the connection rather than of a header, and both are
    read -- the origin a ceremony is verified against comes from the scheme, and
    the first-enrollment gate comes from the socket peer. A stand-in answering
    only one leaves the other reading a default, so it carries both. These cells
    are the owner enrolling at the machine, so the peer is loopback.
    """

    def __init__(self, headers=None, scheme="http", client_host="127.0.0.1"):
        self.headers = headers or {"host": "localhost:8090"}
        self.url = SimpleNamespace(scheme=scheme)
        self.client = type("C", (), {"host": client_host})()


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("STRANDS_DASH_AUTH_RP_ID", raising=False)
    monkeypatch.delenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", raising=False)
    # bust the module cache between tests
    auth._cache = {}
    yield


def test_store_created_with_0600_and_secret(tmp_path):
    assert not auth.has_credentials()
    path = tmp_path / "auth.json"
    assert path.exists()
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600
    data = json.loads(path.read_text())
    assert len(data["jwt_secret"]) > 30


def test_store_hot_reloads_on_file_change(tmp_path):
    auth.has_credentials()  # create
    path = tmp_path / "auth.json"
    data = json.loads(path.read_text())
    data["credentials"] = [{"id": "abc", "public_key": "cGs", "sign_count": 0, "name": "phone"}]
    path.write_text(json.dumps(data))
    os.utime(path, (time.time() + 2, time.time() + 2))
    assert auth.has_credentials()
    assert auth.list_credentials()[0]["name"] == "phone"


def test_token_roundtrip_and_expiry(monkeypatch):
    token = auth.issue_token("cred1", name="phone")
    claims = auth.verify_token(token)
    assert claims["sub"] == "cred1"
    assert auth.session_is_valid(token)
    assert not auth.session_is_valid("garbage")
    assert not auth.session_is_valid("")
    monkeypatch.setenv("STRANDS_DASH_AUTH_TOKEN_TTL", "-10")
    expired = auth.issue_token("cred1")
    with pytest.raises(HTTPException) as exc:
        auth.verify_token(expired)
    assert exc.value.status_code == 401
    assert not auth.session_is_valid(expired)


def test_token_invalid_after_secret_rotation(tmp_path):
    token = auth.issue_token("cred1")
    path = tmp_path / "auth.json"
    data = json.loads(path.read_text())
    data["jwt_secret"] = "rotated-secret-rotated-secret-rotated"
    path.write_text(json.dumps(data))
    os.utime(path, (time.time() + 2, time.time() + 2))
    assert not auth.session_is_valid(token)


def test_client_is_loopback():
    assert auth.client_is_loopback("127.0.0.1")
    assert auth.client_is_loopback("::1")
    assert auth.client_is_loopback("localhost")
    assert not auth.client_is_loopback("192.168.1.50")
    assert not auth.client_is_loopback(None)
    assert not auth.client_is_loopback("evilhost")


def test_rpid_rules():
    assert auth.rpid_is_usable("localhost")
    assert auth.rpid_is_usable("robots.cagatay.my")
    assert not auth.rpid_is_usable("192.168.1.166")
    assert not auth.rpid_is_usable("")


def test_begin_registration_rejects_raw_ip():
    with pytest.raises(HTTPException) as exc:
        auth.begin_registration(FakeRequest({"host": "192.168.1.166:8090"}))
    assert exc.value.status_code == 400


def test_begin_registration_yields_options_and_challenge():
    out = auth.begin_registration(FakeRequest(), label="test key")
    assert out["challenge_id"]
    assert out["options"]["rp"]["id"] == "localhost"
    assert out["options"]["challenge"]


def test_bootstrap_token_gates_first_enrollment(monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", "sekrit")
    with pytest.raises(HTTPException) as exc:
        auth.begin_registration(FakeRequest(), bootstrap="wrong")
    assert exc.value.status_code == 403
    out = auth.begin_registration(FakeRequest(), bootstrap="sekrit")
    assert out["challenge_id"]


def test_begin_authentication_requires_enrollment():
    with pytest.raises(HTTPException) as exc:
        auth.begin_authentication(FakeRequest())
    assert exc.value.status_code == 400


def test_delete_credential_refuses_last(tmp_path):
    path = tmp_path / "auth.json"
    auth.has_credentials()
    data = json.loads(path.read_text())
    data["credentials"] = [{"id": "only", "public_key": "cGs", "sign_count": 0}]
    path.write_text(json.dumps(data))
    os.utime(path, (time.time() + 2, time.time() + 2))
    with pytest.raises(HTTPException) as exc:
        auth.delete_credential("only")
    assert exc.value.status_code == 409
    with pytest.raises(HTTPException) as exc:
        auth.delete_credential("missing")
    assert exc.value.status_code == 404


def test_status_shape():
    out = auth.status(FakeRequest())
    assert out["setup_required"] is True
    assert out["enabled"] is False
    assert out["rp_id"] == "localhost"
    assert out["secure_context"] is True


def test_challenge_pop_semantics():
    cid = auth._stash_challenge("reg", b"chal")
    rec = auth._pop_challenge(cid, "reg")
    assert rec["challenge"] == b"chal"
    with pytest.raises(HTTPException):
        auth._pop_challenge(cid, "reg")  # single use
    cid2 = auth._stash_challenge("reg", b"chal")
    with pytest.raises(HTTPException):
        auth._pop_challenge(cid2, "auth")  # wrong kind


# --- Q24: enrollment turns auth ON; the env flag is only an override --------


def _enroll_fake_credential(tmp_path):
    auth.has_credentials()  # create the store
    path = tmp_path / "auth.json"
    data = json.loads(path.read_text())
    data["credentials"] = [{"id": "abc", "public_key": "cGs", "sign_count": 0, "name": "phone"}]
    path.write_text(json.dumps(data))
    os.utime(path, (time.time() + 2, time.time() + 2))


def test_auth_disabled_with_empty_store_and_no_env():
    assert not auth.auth_enabled()


def test_enrolled_credential_auto_enables_auth(tmp_path):
    _enroll_fake_credential(tmp_path)
    assert auth.auth_enabled()


def test_env_false_overrides_enrolled_credential(tmp_path, monkeypatch):
    _enroll_fake_credential(tmp_path)
    monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "false")
    assert not auth.auth_enabled()


def test_env_true_overrides_empty_store(monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "true")
    assert auth.auth_enabled()


def test_env_whitespace_is_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "  ")
    assert not auth.auth_enabled()
    _enroll_fake_credential(tmp_path)
    assert auth.auth_enabled()


def test_status_reports_enabled_after_enrollment(tmp_path):
    _enroll_fake_credential(tmp_path)
    s = auth.status(FakeRequest())
    assert s["enabled"] is True
    assert s["setup_required"] is False
