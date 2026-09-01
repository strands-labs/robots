"""The ceremony FINISH paths of strands_robots.dashboard.auth.

These were the untested 17% of the module - and they are exactly the lines
that mint sessions. No authenticator exists in CI, so the webauthn library's
verify_* calls are monkeypatched to succeed (or we hand them garbage and
expect OUR refusal first): what these tests pin is the orchestration around
the library - single-use challenges, the duplicate-credential 409, the
rp_id binding recorded at enrollment, the self-healing back-fill for
credentials that predate rp_id recording, sign_count persistence, and that
the token handed back is a session verify_token() actually accepts.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from webauthn.helpers import bytes_to_base64url

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
    auth._cache = {}
    yield


CRED_ID = b"\x01" * 16
CRED_ID_B64 = bytes_to_base64url(CRED_ID)


def _passing_reg_verifier(monkeypatch, cred_id: bytes = CRED_ID):
    monkeypatch.setattr(
        auth,
        "verify_registration_response",
        lambda **kw: SimpleNamespace(
            credential_id=cred_id,
            credential_public_key=b"\x02" * 32,
            sign_count=7,
        ),
    )


def _enroll(monkeypatch, request=None, label="phone") -> str:
    """Run a full (verifier-stubbed) enrollment; returns the credential id."""
    request = request or FakeRequest()
    begun = auth.begin_registration(request, label=label)
    _passing_reg_verifier(monkeypatch)
    out = auth.finish_registration(request, begun["challenge_id"], {"id": CRED_ID_B64})
    return out["credential_id"]


# --- finish_registration ------------------------------------------------------


def test_finish_registration_stores_binding_and_mints_valid_session(monkeypatch):
    request = FakeRequest()
    begun = auth.begin_registration(request, label="phone")
    _passing_reg_verifier(monkeypatch)

    out = auth.finish_registration(request, begun["challenge_id"], {"id": CRED_ID_B64})
    assert out["ok"] is True
    assert out["credential_id"] == CRED_ID_B64

    # The token is not decorative: verify_token must accept it and carry the sub.
    claims = auth.verify_token(out["token"])
    assert claims["sub"] == CRED_ID_B64
    assert claims["name"] == "phone"

    # The rp_id the ceremony verified against is RECORDED on the credential -
    # this is what rp_id_verdict enforces forever after (Q10).
    (cred,) = auth._load()["credentials"]
    assert cred["rp_id"] == "localhost"
    assert cred["sign_count"] == 7
    assert cred["name"] == "phone"


def test_registration_challenge_is_single_use(monkeypatch):
    request = FakeRequest()
    begun = auth.begin_registration(request, label="phone")
    _passing_reg_verifier(monkeypatch)
    auth.finish_registration(request, begun["challenge_id"], {"id": CRED_ID_B64})
    with pytest.raises(HTTPException) as e:
        auth.finish_registration(request, begun["challenge_id"], {"id": CRED_ID_B64})
    assert e.value.status_code in (400, 404)


def test_duplicate_credential_is_409_and_not_stored_twice(monkeypatch):
    request = FakeRequest()
    _enroll(monkeypatch, request)
    # A second ceremony that "verifies" to the SAME credential id must refuse -
    # otherwise a replayed enrollment quietly forks the credential list.
    begun = auth.begin_registration(request, label="again")
    with pytest.raises(HTTPException) as e:
        auth.finish_registration(request, begun["challenge_id"], {"id": CRED_ID_B64})
    assert e.value.status_code == 409
    assert len(auth._load()["credentials"]) == 1


def test_finish_registration_bubbles_verifier_refusal(monkeypatch):
    """When the library refuses, no credential and no token appear."""
    request = FakeRequest()
    begun = auth.begin_registration(request, label="phone")

    def refuse(**kw):
        raise Exception("bad attestation")

    monkeypatch.setattr(auth, "verify_registration_response", refuse)
    with pytest.raises(Exception):
        auth.finish_registration(request, begun["challenge_id"], {"id": CRED_ID_B64})
    assert auth._load().get("credentials", []) == []


# --- finish_authentication ----------------------------------------------------


def _passing_auth_verifier(monkeypatch, new_sign_count=42):
    monkeypatch.setattr(
        auth,
        "verify_authentication_response",
        lambda **kw: SimpleNamespace(
            new_sign_count=new_sign_count,
        ),
    )


def test_finish_authentication_mints_session_and_persists_sign_count(monkeypatch):
    request = FakeRequest()
    _enroll(monkeypatch, request)
    begun = auth.begin_authentication(request)
    _passing_auth_verifier(monkeypatch, new_sign_count=42)

    out = auth.finish_authentication(request, begun["challenge_id"], {"id": CRED_ID_B64})
    assert out["ok"] is True
    claims = auth.verify_token(out["token"])
    assert claims["sub"] == CRED_ID_B64
    assert auth.session_is_valid(out["token"])

    # sign_count moved 7 -> 42 and was SAVED (clone detection depends on it).
    (cred,) = auth._load()["credentials"]
    assert cred["sign_count"] == 42


def test_unknown_credential_is_404_before_any_crypto(monkeypatch):
    request = FakeRequest()
    _enroll(monkeypatch, request)
    begun = auth.begin_authentication(request)

    def explode(**kw):  # pragma: no cover - the point is it is never reached
        raise AssertionError("verifier must not run for an unknown credential")

    monkeypatch.setattr(auth, "verify_authentication_response", explode)
    with pytest.raises(HTTPException) as e:
        auth.finish_authentication(request, begun["challenge_id"], {"id": "someone-elses-id"})
    assert e.value.status_code == 404


def test_authentication_challenge_is_single_use(monkeypatch):
    request = FakeRequest()
    _enroll(monkeypatch, request)
    begun = auth.begin_authentication(request)
    _passing_auth_verifier(monkeypatch)
    auth.finish_authentication(request, begun["challenge_id"], {"id": CRED_ID_B64})
    with pytest.raises(HTTPException) as e:
        auth.finish_authentication(request, begun["challenge_id"], {"id": CRED_ID_B64})
    assert e.value.status_code in (400, 404)


def test_rp_id_backfill_for_pre_rpid_credential(monkeypatch, tmp_path):
    """cagatay's real passkey predates rp_id recording: a login that VERIFIES
    against a rp_id back-fills it (proof, not a guess), tightening the guard."""
    request = FakeRequest()
    _enroll(monkeypatch, request)

    # Erase the binding, as if enrolled before the field existed.
    path = tmp_path / "auth.json"
    data = json.loads(path.read_text())
    del data["credentials"][0]["rp_id"]
    path.write_text(json.dumps(data))
    import os

    os.utime(path, (time.time() + 2, time.time() + 2))
    auth._cache = {}

    begun = auth.begin_authentication(request)
    _passing_auth_verifier(monkeypatch)
    auth.finish_authentication(request, begun["challenge_id"], {"id": CRED_ID_B64})

    (cred,) = auth._load()["credentials"]
    assert cred["rp_id"] == "localhost"


def test_rp_id_backfill_never_overwrites_an_existing_binding(monkeypatch):
    request = FakeRequest()
    _enroll(monkeypatch, request)
    begun = auth.begin_authentication(request)
    _passing_auth_verifier(monkeypatch)
    auth.finish_authentication(request, begun["challenge_id"], {"id": CRED_ID_B64})
    (cred,) = auth._load()["credentials"]
    assert cred["rp_id"] == "localhost"  # from enrollment, not re-derived


# --- delete_credential --------------------------------------------------------


def test_delete_credential_refuses_unknown_and_last(monkeypatch):
    with pytest.raises(HTTPException) as e:
        auth.delete_credential("nope")
    assert e.value.status_code == 404

    _enroll(monkeypatch)
    with pytest.raises(HTTPException) as e:
        auth.delete_credential(CRED_ID_B64)
    # Removing the LAST passkey would re-open setup to anyone: refused.
    assert e.value.status_code == 409
    assert auth.has_credentials()


def test_delete_credential_removes_one_of_two(monkeypatch):
    request = FakeRequest()
    _enroll(monkeypatch, request)
    other = b"\x03" * 16
    begun = auth.begin_registration(request, label="backup")
    _passing_reg_verifier(monkeypatch, cred_id=other)
    auth.finish_registration(request, begun["challenge_id"], {"id": bytes_to_base64url(other)})

    out = auth.delete_credential(CRED_ID_B64)
    assert out == {"ok": True, "removed": CRED_ID_B64, "remaining": 1}
    (cred,) = auth._load()["credentials"]
    assert cred["name"] == "backup"


# --- status() warnings --------------------------------------------------------


def test_status_warns_on_insecure_context():
    out = auth.status(FakeRequest({"host": "robots.example.com"}))
    assert out["secure_context"] is False
    assert "secure context" in out["warning"]


def test_status_warns_on_unusable_rpid():
    # Reached over https, but the host is an IP: rpId can never work.
    out = auth.status(FakeRequest({"host": "192.168.1.50:8090"}, scheme="https"))
    assert out["rpid_usable"] is False
    assert "rpId" in out.get("warning", "")


def test_status_clean_on_localhost():
    out = auth.status(FakeRequest({"host": "localhost:8090"}))
    assert out["secure_context"] is True
    assert out["rpid_usable"] is True
    assert "warning" not in out
