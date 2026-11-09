"""A browser sends a localhost cookie to every localhost port, so the session cookie alone opens nothing: the page's
proof (kept in its own origin's storage) must come with it, on the Host the session was issued on."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")
pytest.importorskip("webauthn")

from starlette.websockets import WebSocketDisconnect  # noqa: E402

from strands_robots import Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402
from strands_robots.dashboard.auth import session  # noqa: E402

from .passkey import Authenticator  # noqa: E402
from .test_auth import BASE, JSON, client, enrol, proved  # noqa: E402

pytestmark = pytest.mark.signed_out
COOKIE = f"{session.COOKIE}_8090"


@pytest.fixture
def signed():
    """A dashboard on :8090 and a page signed in to it: (dashboard, cookie token, proof)."""
    dash = Dashboard(Simulation())
    _, out = enrol(client(dash), Authenticator(BASE), setup=dash.auth.setup_code)
    return dash, out["set-cookie"].split(";")[0].partition("=")[2], out["proof"]


def _cookie(dash, token, base=BASE, name=COOKIE):
    c = client(dash, base)
    c.cookies.set(name, token)
    return c


def test_the_cookie_another_port_was_sent_opens_no_route_and_no_socket(signed) -> None:
    dash, token, proof = signed
    c = _cookie(dash, token)
    assert c.get("/api/scene").status_code == 401
    assert c.post("/api/robots/x/stop", json={}, headers={**JSON, "X-Confirm": "yes"}).status_code == 401
    assert c.get("/auth/status").json()["signed_in"] is False
    with pytest.raises(WebSocketDisconnect) as closed, c.websocket_connect("ws://localhost:8090/ws/state"):
        pass
    assert closed.value.code == 4401
    assert c.get("/api/scene", headers=proved(proof)).status_code == 200


def test_the_cookie_token_as_a_bearer_is_still_only_half_a_session(signed) -> None:
    dash, token, proof = signed
    c = client(dash)
    assert c.get("/api/scene", headers={"authorization": f"Bearer {token}"}).status_code == 401
    assert c.get("/api/scene", headers={"authorization": f"Bearer {token}", **proved(proof)}).status_code == 200


def test_a_session_counts_only_on_the_port_it_was_issued_on(signed) -> None:
    dash, token, proof = signed
    other = _cookie(dash, token, "http://localhost:8091", f"{session.COOKIE}_8091")
    assert other.get("/api/scene", headers=proved(proof)).status_code == 401
    assert session.verify(dash.auth.store.secret, token)["aud"] == "localhost:8090"


def test_a_wrong_proof_an_old_cookie_name_or_a_script_token_in_the_cookie_is_refused(signed) -> None:
    dash, token, proof = signed
    assert _cookie(dash, token).get("/api/scene", headers=proved(proof[:-1] + "A")).status_code == 401
    assert _cookie(dash, token, name=session.COOKIE).get("/api/scene", headers=proved(proof)).status_code == 401
    script = dash.auth.token("ci")
    assert _cookie(dash, script).get("/api/scene", headers=proved(dash.auth.proof(script))).status_code == 401
    assert client(dash).get("/api/scene", headers={"authorization": f"Bearer {script}"}).status_code == 200


def test_a_socket_offers_the_proof_as_its_subprotocol_and_gets_it_back(signed) -> None:
    dash, token, proof = signed
    offered = f"{session.PROTOCOL}{proof}"
    with _cookie(dash, token).websocket_connect("ws://localhost:8090/ws/state", subprotocols=[offered]) as ws:
        assert ws.accepted_subprotocol == offered
        assert ws.receive_json()
    with pytest.raises(WebSocketDisconnect) as closed:
        with client(dash).websocket_connect("ws://localhost:8090/ws/state", subprotocols=[offered]):
            pass
    assert closed.value.code == 4401


def test_sign_in_answers_with_the_proof_never_cached_and_logout_clears_this_ports_cookie() -> None:
    dash, key = Dashboard(Simulation()), Authenticator(BASE)
    c = client(dash)
    begun = c.post("/auth/enrol/begin", json={"setup": dash.auth.setup_code}, headers=JSON).json()
    r = c.post(
        "/auth/enrol/finish", json={"challenge_id": begun["challenge_id"], "credential": key.create(begun["options"])}
    )
    token = r.headers["set-cookie"].split(";")[0].partition("=")[2]
    assert r.headers["cache-control"] == "no-store" and session.proves(dash.auth.store.secret, token, r.json()["proof"])
    out = c.post("/auth/logout", headers=JSON).headers["set-cookie"]
    assert out.startswith(f"{COOKIE}=;") and "Max-Age=0" in out
