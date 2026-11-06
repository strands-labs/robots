"""Passkey sign-in: every /api and /ws route needs a session, first enrolment needs the setup code, X-Confirm alone
moves nothing. A software authenticator (``passkey.py``) answers the ceremonies, so ``py_webauthn`` verifies for real.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")
pytest.importorskip("webauthn")

import test_drivers  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.routing import Mount, Route, WebSocketRoute  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402
from strands_robots.dashboard.auth import Auth, session  # noqa: E402

from .passkey import Authenticator  # noqa: E402

pytestmark = pytest.mark.signed_out
BASE = "http://localhost:8090"
JSON = {"content-type": "application/json"}
PUBLIC = {"/auth/status", "/auth/enrol/begin", "/auth/enrol/finish", "/auth/login/begin", "/auth/login/finish"}
PUBLIC |= {"/auth/logout"}


@pytest.fixture
def world():
    sim = Simulation()
    Robot("so101", sim=sim)
    d, serial = test_drivers.make_driver()
    real = Robot("so101", driver=d, id="real_arm", twin=False)
    dash = Dashboard(sim, robots=[real])
    yield dash, real, serial
    real.close()
    sim.close()


def client(dash: Dashboard, base: str = BASE) -> TestClient:
    return TestClient(dash.app, base_url=base, follow_redirects=False)


def bearer(token: str) -> dict[str, str]:
    return {**JSON, "authorization": f"Bearer {token}"}


def enrol(c: TestClient, key: Authenticator, *, setup: str | None = None, headers=None) -> tuple[int, dict]:
    r = c.post("/auth/enrol/begin", json={"setup": setup, "label": "laptop"}, headers=headers or JSON)
    if r.status_code != 200:
        return r.status_code, r.json()
    begun = r.json()
    r = c.post(
        "/auth/enrol/finish", json={"challenge_id": begun["challenge_id"], "credential": key.create(begun["options"])}
    )
    return r.status_code, {"set-cookie": r.headers.get("set-cookie", ""), **r.json()}


def login(c: TestClient, key: Authenticator, **kw) -> tuple[int, str]:
    begun = c.post("/auth/login/begin", json={}).json()
    r = c.post(
        "/auth/login/finish",
        json={"challenge_id": begun["challenge_id"], "credential": key.get(begun["options"], **kw)},
    )
    cookie = r.headers.get("set-cookie", "")
    return r.status_code, cookie.split(";")[0].partition("=")[2]


def test_every_api_and_socket_route_needs_a_session_and_nothing_else_is_open(world) -> None:
    dash, _, _ = world
    c = client(dash)
    for route in dash.app.routes:
        path = route.path
        if isinstance(route, Mount):
            assert path == "", f"unexpected mount {path}"  # the static shell and its assets
            continue
        concrete = path.replace("{rid}", "so101").replace("{path:path}", "x").replace("{name}", "x")
        concrete = concrete.replace("{device_id}", "x").replace("{cid}", "x").replace("{verb}", "save")
        concrete = concrete.replace("{cam}", "x")
        if path.startswith(("/api", "/ws")):
            if isinstance(route, WebSocketRoute):
                with (
                    pytest.raises(WebSocketDisconnect) as closed,
                    c.websocket_connect(f"ws://localhost:8090{concrete}"),
                ):
                    pass
                assert closed.value.code == 4401, path
                continue
            assert isinstance(route, Route)
            for method in route.methods or ():
                if method == "HEAD":
                    continue
                r = c.request(method, concrete, headers=JSON, content="{}" if method == "POST" else None)
                assert r.status_code == 401, (method, path, r.status_code)
        else:
            assert path in PUBLIC or path.startswith("/vendor/") or path in ("/openapi.json",), f"open route {path}"


def test_x_confirm_alone_no_longer_releases_or_moves_a_real_robot(world) -> None:
    dash, real, serial = world
    c = client(dash)
    real.estop()
    yes = {**JSON, "x-confirm": "yes"}
    assert c.post("/api/robots/real_arm/resume", json={"approved_by": "someone"}, headers=yes).status_code == 401
    assert real.estopped
    before = len(test_drivers.writes(serial))
    assert c.post("/api/robots/real_arm/act", json={"targets": {"shoulder_pan": 0.2}}, headers=yes).status_code == 401
    assert len(test_drivers.writes(serial)) == before  # nothing reached the bus
    token = dash.auth.token("operator-script")
    r = c.post("/api/robots/real_arm/resume", json={"approved_by": "me"}, headers={**bearer(token), "x-confirm": "yes"})
    assert r.status_code == 200 and not real.estopped


def test_first_enrolment_needs_the_printed_setup_code(world, capsys) -> None:
    dash, _, _ = world
    c, key = client(dash), Authenticator(BASE)
    assert c.get("/auth/status").json()["setup_required"] is True
    assert enrol(c, key)[0] == 403
    assert enrol(c, key, setup="guess")[0] == 403
    code = dash.auth.setup_code
    status, out = enrol(c, key, setup=code)
    assert status == 200 and out["signed_in"] is True
    assert dash.auth.setup_code is None and not dash.auth.setup_required
    assert enrol(client(dash), Authenticator(BASE), setup=code)[0] == 403  # one time: more passkeys need a session


def test_the_session_cookie_is_httponly_samesite_strict_and_short_lived(world) -> None:
    dash, _, _ = world
    _, out = enrol(client(dash), Authenticator(BASE), setup=dash.auth.setup_code)
    flags = [f.strip() for f in out["set-cookie"].split(";")]
    assert flags[0].startswith(f"{session.COOKIE}=") and "HttpOnly" in flags and "SameSite=Strict" in flags
    assert f"Max-Age={12 * 3600}" in flags
    token = flags[0].partition("=")[2]
    c = client(dash)
    c.cookies.set(session.COOKIE, token)
    assert c.get("/api/scene").status_code == 200
    assert "Max-Age=0" in c.post("/auth/logout", headers=JSON).headers["set-cookie"]


def test_sign_in_works_and_a_replayed_sign_count_is_refused(world) -> None:
    dash, _, _ = world
    c, key = client(dash), Authenticator(BASE)
    enrol(c, key, setup=dash.auth.setup_code)
    status, token = login(c, key)
    assert status == 200 and c.get("/api/scene", headers=bearer(token)).status_code == 200
    assert login(c, key, count=key.count)[0] == 403  # the same count again: a cloned or replayed authenticator


def test_a_ceremony_from_another_origin_is_refused(world) -> None:
    dash, _, _ = world
    c, key = client(dash), Authenticator(BASE)
    begun = c.post("/auth/enrol/begin", json={"setup": dash.auth.setup_code}, headers=JSON).json()
    cred = key.create(begun["options"], origin="http://localhost:9999")
    assert (
        c.post("/auth/enrol/finish", json={"challenge_id": begun["challenge_id"], "credential": cred}).status_code
        == 403
    )


def test_more_passkeys_need_a_session_and_the_last_one_cannot_go(world) -> None:
    dash, _, _ = world
    c, first, second = client(dash), Authenticator(BASE), Authenticator(BASE)
    enrol(client(dash), first, setup=dash.auth.setup_code)
    assert enrol(c, second)[0] == 403
    _, token = login(c, first)
    assert enrol(c, second, headers=bearer(token))[0] == 200
    listed = c.get("/api/passkeys", headers=bearer(token)).json()["passkeys"]
    assert {p["id"] for p in listed} == {first.id, second.id}
    r = c.post(f"/api/passkeys/{second.id}", json={"name": "phone"}, headers=bearer(token))
    assert r.json()["name"] == "phone"
    assert c.delete(f"/api/passkeys/{first.id}", headers=bearer(token)).status_code == 200
    assert c.get("/api/scene", headers=bearer(token)).status_code == 401  # its sessions ended with it
    _, token = login(c, second)
    assert c.delete(f"/api/passkeys/{second.id}", headers=bearer(token)).status_code == 409


def test_a_token_is_refused_once_expired_or_tampered(tmp_path) -> None:
    auth = Auth(store=tmp_path / "a.json", ttl=60)
    token = auth.token()
    assert auth.claims(token)["sub"] == "token:local"
    assert auth.claims(token[:-2] + ("A" if token[-2] != "A" else "B") + token[-1]) is None
    assert session.verify(auth.store.secret, token, now=time.time() + 61) is None
    assert Auth(store=tmp_path / "other.json").claims(token) is None


def test_the_store_is_readable_by_this_user_only(tmp_path) -> None:
    auth = Auth(store=tmp_path / "a.json")
    auth.token()
    assert (tmp_path / "a.json").stat().st_mode & 0o777 == 0o600


def test_the_page_on_a_raw_loopback_ip_moves_to_localhost(world) -> None:
    dash, _, _ = world
    r = client(dash, "http://127.0.0.1:8090").get("/?setup=abc")
    assert r.status_code == 307 and r.headers["location"] == "http://localhost:8090/?setup=abc"
    assert client(dash).get("/").status_code == 200
    r = client(dash, "http://127.0.0.1:8090").post("/auth/login/begin", json={})
    assert r.status_code == 400 and "localhost" in r.json()["detail"]


def test_turning_sign_in_off_is_refused_with_a_real_robot_unless_insisted(world, caplog) -> None:
    dash, real, _ = world
    with pytest.raises(ValueError, match="unauthenticated_hardware=True"):
        Dashboard(dash.sim, robots=[real], auth=False)
    with caplog.at_level("WARNING"):
        open_dash = Dashboard(dash.sim, robots=[real], auth=False, unauthenticated_hardware=True)
    assert "sign-in is OFF" in caplog.text
    assert client(open_dash).get("/api/scene").status_code == 200
    assert client(Dashboard(dash.sim, auth=False)).get("/api/scene").status_code == 200


def test_start_prints_the_one_time_setup_link(world, capsys) -> None:
    dash, _, _ = world
    server = dash.start()
    try:
        err = capsys.readouterr().err
        assert f"{server.url}/?setup={dash.auth.setup_code}" in err and server.url.startswith("http://localhost:")
    finally:
        server.stop()
