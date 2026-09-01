"""The origin a passkey ceremony is verified against is the connection's, not the caller's claim.

``expected_origin`` is what the WebAuthn library compares the authenticator's
SIGNED ``clientDataJSON.origin`` against, so it decides whether an assertion
minted for one origin may mint a session at another. Deriving it from the
request's own ``Origin`` header makes that comparison a tautology: the library
is told to expect whatever the caller said. ``rp_id`` still binds the
registrable domain -- a browser will not sign for an rpId that is not a
registrable suffix of the page -- so what a tautology loses is the binding
WITHIN one domain: an ``http://`` downgrade and a sibling subdomain both pass a
check whose only purpose is to refuse them.

These use real :class:`starlette.requests.Request` / :class:`WebSocket` objects
rather than a stand-in with a ``headers`` dict, because the two facts under test
-- the transport's own scheme, and that a websocket reports ``wss`` where its
page origin says ``https`` -- exist only on the real object. A fake that answers
headers alone cannot observe either, which is why an expectation taken from a
header could look correct in a test.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.websockets import WebSocket
from webauthn.helpers import bytes_to_base64url

from strands_robots.dashboard import auth

SERVED_AT = "https://dash.example.com"
ELSEWHERE = "https://evil.example.com"
CRED_ID = b"\x03" * 16
CRED_ID_B64 = bytes_to_base64url(CRED_ID)
BOOTSTRAP = "a-token-for-a-remote-first-enrollment"


def contradiction(offered: str) -> str:
    """The refusal an operator has to be able to read: which two origins disagreed.

    Spelled whole, and compared whole. For a URL the POSITION carries the
    meaning, so a substring assertion is satisfied by a message that names the
    two origins the wrong way round, or that carries one inside the other's
    query string -- neither of which tells an operator behind a proxy which end
    to fix.
    """
    return f"Origin {offered!r} is not this deployment's origin {SERVED_AT!r}"


def request(scheme: str = "https", **headers: str) -> Request:
    """A request that really arrived over ``scheme`` carrying ``headers``."""
    return Request(
        {
            "type": "http",
            "scheme": scheme,
            "method": "POST",
            "path": "/",
            "query_string": b"",
            "server": ("10.0.0.1", 443),
            "client": ("203.0.113.9", 51234),
            "headers": [(k.replace("_", "-").encode(), v.encode()) for k, v in headers.items()],
        }
    )


async def _never_receives() -> Any:
    raise AssertionError("deciding an origin must not read from the socket")


async def _never_sends(message: Any) -> None:
    raise AssertionError("deciding an origin must not write to the socket")


def websocket(scheme: str, **headers: str) -> WebSocket:
    """A websocket that really arrived over ``scheme``.

    Its receive/send channels refuse: the decision under test reads the
    connection's scope, and must not depend on talking to the peer.
    """
    return WebSocket(
        {
            "type": "websocket",
            "scheme": scheme,
            "path": "/ws",
            "query_string": b"",
            "server": ("10.0.0.1", 443),
            "headers": [(k.replace("_", "-").encode(), v.encode()) for k, v in headers.items()],
        },
        receive=_never_receives,
        send=_never_sends,
    )


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    for key in ("ENABLED", "RP_ID", "ORIGIN", "BOOTSTRAP_TOKEN", "TOKEN_TTL"):
        monkeypatch.delenv("STRANDS_DASH_AUTH_" + key, raising=False)
    auth._cache = {}
    auth._challenges.clear()
    yield
    auth._challenges.clear()


# --- the expectation is never the caller's claim -------------------------------

#: Each row is a claim a caller can make that the served origin contradicts. The
#: first is not an attack shape but a MISCONFIGURATION with the same signature (a
#: TLS-terminating proxy that speaks http to the app): both are refused, and the
#: refusal names the remedy rather than silently trusting the claim.
CONTRADICTING = [
    ("http downgrade", "http://dash.example.com"),
    ("sibling subdomain", ELSEWHERE),
    ("unrelated origin", "https://anything.at.all"),
    ("port swapped", "https://dash.example.com:8443"),
]


@pytest.mark.parametrize(("label", "claimed"), CONTRADICTING, ids=[r[0] for r in CONTRADICTING])
def test_an_origin_that_contradicts_the_connection_is_refused(label, claimed):
    with pytest.raises(HTTPException) as e:
        auth._derive_origin(request(host="dash.example.com", origin=claimed))
    assert e.value.status_code == 400
    # Attributable: an operator behind a proxy must be able to tell which two
    # origins disagreed, and be pointed at the pin that settles it.
    assert e.value.detail["detail"] == contradiction(claimed)
    assert "STRANDS_DASH_AUTH_ORIGIN" in e.value.detail["hint"]


def test_the_origin_the_dashboard_is_served_at_is_accepted_and_normalised():
    for claimed in (SERVED_AT, SERVED_AT + "/"):
        assert auth._derive_origin(request(host="dash.example.com", origin=claimed)) == SERVED_AT
    # Local development is the common case and must not need configuring.
    assert (
        auth._derive_origin(request("http", host="localhost:8090", origin="http://localhost:8090"))
        == "http://localhost:8090"
    )


def test_a_missing_origin_header_leaves_the_connection_standing():
    # Not a licence to widen it: the authenticator's signed clientData still has
    # to match this, which is the comparison a header-derived value bypassed.
    assert auth._derive_origin(request(host="dash.example.com")) == SERVED_AT


def test_a_forwarded_proto_header_cannot_choose_the_scheme():
    # x-forwarded-proto is set by whoever is calling. A proxy the operator trusts
    # is honoured by the SERVER (uvicorn --proxy-headers --forwarded-allow-ips),
    # which rewrites the connection scheme before the app sees the request.
    reached_over_http = request("http", host="dash.example.com", x_forwarded_proto="https")
    assert auth._derive_origin(reached_over_http) == "http://dash.example.com"


def test_a_websocket_scheme_is_read_as_the_page_origin_it_belongs_to():
    # A page served over https opens wss sockets, and the browser still spells
    # that page's Origin https:// - so the socket spelling is never the answer.
    assert auth._derive_origin(websocket("wss", host="dash.example.com", origin=SERVED_AT)) == SERVED_AT
    assert (
        auth._derive_origin(websocket("ws", host="localhost:8090", origin="http://localhost:8090"))
        == "http://localhost:8090"
    )


def test_a_transport_with_no_scheme_is_refused_rather_than_guessed():
    with pytest.raises(HTTPException) as e:
        auth._derive_origin(SimpleNamespace(headers={"host": "dash.example.com"}))
    assert e.value.status_code == 400
    assert "no usable scheme" in e.value.detail["detail"]


def test_the_configured_origin_outranks_the_headers_and_is_normalised(monkeypatch):
    # The installs that need this pin are the ones whose proxy rewrites Host and
    # Origin, so consulting either here would refuse the deployment it exists for.
    monkeypatch.setenv("STRANDS_DASH_AUTH_ORIGIN", "https://robots.example/")
    rewritten = request("http", host="internal.local", origin="http://internal.local")
    assert auth._derive_origin(rewritten) == "https://robots.example"


VERDICTS = [
    ("matching", "https://a.example", "https://a.example", "https://a.example"),
    ("trailing slash", "https://a.example/", "https://a.example", "https://a.example"),
    ("absent", "", "https://a.example", "https://a.example"),
    ("downgraded", "http://a.example", "https://a.example", None),
    ("sibling", "https://b.a.example", "https://a.example", None),
]


@pytest.mark.parametrize(("label", "offered", "expected", "decided"), VERDICTS, ids=[r[0] for r in VERDICTS])
def test_origin_verdict_reports_the_decision_and_its_reason(label, offered, expected, decided):
    origin, reason = auth.origin_verdict(offered, expected)
    assert origin == decided
    assert reason


# --- the call sites: what the library is actually told to expect ---------------


def _enrolled_credential() -> None:
    store = auth._load()
    store["credentials"] = [
        {
            "id": CRED_ID_B64,
            "public_key": bytes_to_base64url(b"\x02" * 32),
            "sign_count": 0,
            "name": "phone",
            "rp_id": "dash.example.com",
            "created": time.time(),
        }
    ]
    auth._save(store)


def _finish_registration(monkeypatch, begin_on: Request, finish_on: Request) -> dict:
    seen: dict = {}
    # The first enrollment seals the dashboard, so it is limited to the machine itself
    # unless a bootstrap token is configured to check it against. These cells model a
    # REMOTE browser reaching a served origin -- that is the scenario under test -- so
    # the socket peer is deliberately not loopback, and the token is what holds that
    # gate constant and leaves the origin the only variable. Without it the enrollment
    # gate refuses first, and a test about an origin reports on enrollment instead.
    monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", BOOTSTRAP)
    monkeypatch.setattr(
        auth,
        "verify_registration_response",
        lambda **kw: (
            seen.update(kw) or SimpleNamespace(credential_id=CRED_ID, credential_public_key=b"\x02" * 32, sign_count=1)
        ),
    )
    begun = auth.begin_registration(begin_on, label="phone", bootstrap=BOOTSTRAP)
    auth.finish_registration(finish_on, begun["challenge_id"], {"id": CRED_ID_B64})
    return seen


def _finish_authentication(monkeypatch, begin_on: Request, finish_on: Request) -> dict:
    seen: dict = {}
    _enrolled_credential()
    monkeypatch.setattr(
        auth,
        "verify_authentication_response",
        lambda **kw: seen.update(kw) or SimpleNamespace(new_sign_count=2),
    )
    begun = auth.begin_authentication(begin_on)
    auth.finish_authentication(finish_on, begun["challenge_id"], {"id": CRED_ID_B64})
    return seen


CEREMONIES = [("registration", _finish_registration), ("authentication", _finish_authentication)]


@pytest.mark.parametrize(("label", "run"), CEREMONIES, ids=[c[0] for c in CEREMONIES])
def test_a_ceremony_is_verified_against_the_served_origin(label, run, monkeypatch):
    served = request(host="dash.example.com", origin=SERVED_AT)
    seen = run(monkeypatch, served, served)
    assert seen["expected_origin"] == SERVED_AT
    # The rp_id half comes from the stashed challenge, not the request - both
    # expectations have to be facts for the pair to mean anything.
    assert seen["expected_rp_id"] == "dash.example.com"


#: Every ceremony crossed with every claim the served origin contradicts. The claim
#: is the FINISHING request's, so these rows grade the ceremony path rather than
#: :func:`auth._derive_origin` on its own: a finish that decided the expectation
#: from the credential's stored ``rp_id`` -- which is a HOST -- still refuses the
#: rows whose host differs, and admits the two that share ``dash.example.com``. A
#: single differing-host row cannot tell those two situations apart, however the
#: refusal it does get is compared.
REFUSED_CEREMONIES = [
    (f"{ceremony}/{shape}", run, claimed) for ceremony, run in CEREMONIES for shape, claimed in CONTRADICTING
]


@pytest.mark.parametrize(
    ("label", "run", "claimed"),
    REFUSED_CEREMONIES,
    ids=[row[0] for row in REFUSED_CEREMONIES],
)
def test_a_ceremony_finished_from_a_claimed_origin_is_refused(label, run, claimed, monkeypatch):
    served = request(host="dash.example.com", origin=SERVED_AT)
    finished_from = request(host="dash.example.com", origin=claimed)
    with pytest.raises(HTTPException) as e:
        run(monkeypatch, served, finished_from)
    assert e.value.status_code == 400
    assert e.value.detail["detail"] == contradiction(claimed)
