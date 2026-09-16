"""The dashboard server core: every route fails closed, the open posture is loopback-only.

``create_app`` wires the landed auth/settings/redaction modules to paths. What
is graded here is the routing decision itself - which paths are public, when a
caller with nothing in hand is admitted, and that a sealed dashboard refuses
everything but the login screen.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard import access, auth, settings  # noqa: E402
from strands_robots.dashboard.server import create_app, redacted_settings  # noqa: E402


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """A fresh store and settings file, no env override, auth store empty."""
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    settings.clear_overrides()
    settings.load(refresh=True)
    yield tmp_path
    settings.clear_overrides()
    settings.load(refresh=True)


@pytest.fixture()
def client(isolated):
    return TestClient(create_app())


PUBLIC = ("/api/health", "/api/auth/status")
GUARDED_GET = ("/api/whoami", "/api/settings", "/api/auth/credentials")


class TestOpenPosture:
    """Fresh install, loopback caller, nothing enrolled: usable on this machine."""

    def test_health_says_up_and_versioned(self, client):
        body = client.get("/api/health").json()
        assert body["ok"] is True
        assert body["version"]

    def test_status_reports_setup_required_and_open(self, client):
        body = client.get("/api/auth/status").json()
        assert body["setup_required"] is True
        assert body["authenticated"] is False
        assert body["open_posture"] is True

    def test_loopback_caller_is_admitted_as_loopback(self, client):
        assert client.get("/api/whoami").json()["via"] == "loopback"

    def test_a_proxied_loopback_is_not_this_machine(self, client):
        for header in ("x-forwarded-for", "x-real-ip", "forwarded"):
            assert client.get("/api/whoami", headers={header: "203.0.113.9"}).status_code == 401, header

    def test_a_rebound_name_on_the_loopback_socket_is_not_this_machine(self, client):
        """DNS rebinding: evil.example resolves to 127.0.0.1, so the socket peer is
        loopback and no proxy header is set - but Host carries the attacker's name."""
        for host in ("evil.example", "evil.example:8090", "192.168.1.20:8090"):
            assert client.get("/api/whoami", headers={"host": host}).status_code == 401, host

    @pytest.mark.parametrize("host", ["localhost", "localhost:8090", "127.0.0.1:8090", "[::1]:8090", "::1"])
    def test_every_loopback_spelling_of_host_is_this_machine(self, client, host):
        assert client.get("/api/whoami", headers={"host": host}).json()["via"] == "loopback"

    def test_a_page_from_another_origin_is_not_this_machine(self, client):
        """The operator's browser running someone else's script: loopback peer,
        loopback Host, but Origin names the page that made the request."""
        for origin in ("http://evil.example", "https://evil.example:8090", "null"):
            assert client.get("/api/whoami", headers={"origin": origin}).status_code == 401, origin

    def test_the_dashboards_own_page_is_this_machine(self, client):
        assert client.get("/api/whoami", headers={"origin": "http://testserver"}).json()["via"] == "loopback"

    def test_a_cross_origin_write_is_refused_before_any_credential_is_read(self, client):
        """Whatever the credential, a write whose Origin is another page is 403 -
        this is the guard the sealed posture has too, not only the open one."""
        response = client.post(
            "/api/settings", json={"agent": {"temperature": 0.9}}, headers={"origin": "http://evil.example"}
        )
        assert response.status_code == 403
        assert response.json() == {"error": "cross-origin write refused"}
        assert settings.get("agent", "temperature") != 0.9

    def test_a_write_has_to_say_it_is_json(self, client):
        """A text/plain POST is a no-preflight simple request; parsing it would
        make every page on the web a caller of every write route."""
        response = client.post(
            "/api/settings", content='{"agent": {"temperature": 0.9}}', headers={"content-type": "text/plain"}
        )
        assert response.status_code == 415
        assert settings.get("agent", "temperature") != 0.9
        declared = client.post(
            "/api/settings",
            content='{"agent": {"temperature": 0.9}}',
            headers={"content-type": "application/json; charset=utf-8"},
        )
        assert declared.status_code == 200
        assert settings.get("agent", "temperature") == 0.9

    def test_static_ui_is_served(self, client):
        assert client.get("/").status_code == 200
        assert "text/html" in client.get("/").headers["content-type"]
        assert client.get("/static/app.js").status_code == 200


class TestSealed:
    """The moment auth is on, only the login screen answers an anonymous caller."""

    @pytest.fixture(autouse=True)
    def _seal(self, client, monkeypatch):  # after `isolated` has cleared the env
        monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "1")

    @pytest.mark.parametrize("path", PUBLIC)
    def test_public_routes_stay_public(self, client, path):
        assert client.get(path).status_code == 200

    @pytest.mark.parametrize("path", GUARDED_GET)
    def test_guarded_gets_refuse(self, client, path):
        assert client.get(path).status_code == 401

    def test_guarded_posts_refuse(self, client):
        # Each request is bound before the assert: sending one is a side effect, and
        # `assert` is compiled out under `python -O`.
        patch = client.post("/api/settings", json={"agent": {"temperature": 0.1}})
        handoff = client.post("/api/auth/handoff")
        removal = client.delete("/api/auth/credentials/x")
        assert [patch.status_code, handoff.status_code, removal.status_code] == [401, 401, 401]

    def test_query_string_token_is_not_read(self, client, monkeypatch):
        monkeypatch.setattr(
            settings, "get", lambda s, k=None, d=None: "SECRET" if (s, k) == ("security", "auth_token") else d
        )
        assert client.get("/api/whoami?token=SECRET").status_code == 401

    def test_a_forged_bearer_is_refused(self, client):
        assert client.get("/api/whoami", headers={"authorization": "Bearer eyJ.not.real"}).status_code == 401

    def test_a_real_session_is_admitted(self, client):
        token = auth.issue_token("cred-1", name="owner")
        body = client.get("/api/whoami", headers={"authorization": f"Bearer {token}"}).json()
        assert body == {"via": "passkey", "name": "owner", "sub": "cred-1"}

    def test_the_cookie_carries_the_session(self, client):
        token = auth.issue_token("cred-1", name="owner")
        client.cookies.set(access.COOKIE, token)
        assert client.get("/api/whoami").json()["via"] == "passkey"

    def test_refusals_share_one_shape(self, client):
        assert client.get("/api/whoami").json() == {"error": "sign in required"}


class TestStaticToken:
    def test_matching_bearer_is_admitted_as_token(self, client, isolated):
        settings.update({"security": {"auth_token": "S3CRET-TOKEN"}})
        assert client.get("/api/whoami").status_code == 401, "a configured token closes the open posture"
        body = client.get("/api/whoami", headers={"authorization": "Bearer S3CRET-TOKEN"}).json()
        assert body["via"] == "token"

    def test_token_session_cannot_remove_a_passkey(self, client, isolated):
        settings.update({"security": {"auth_token": "S3CRET-TOKEN"}})
        r = client.delete("/api/auth/credentials/x", headers={"authorization": "Bearer S3CRET-TOKEN"})
        assert r.status_code == 403


class TestSettingsRoutes:
    def test_read_redacts_the_secret(self, client, isolated):
        settings.update({"security": {"auth_token": "S3CRET-TOKEN"}})
        body = client.get("/api/settings", headers={"authorization": "Bearer S3CRET-TOKEN"}).json()
        assert body["settings"]["security"]["auth_token"] is True
        assert "S3CRET" not in str(body)

    def test_unknown_keys_are_refused_before_anything_is_written(self, client):
        r = client.post("/api/settings", json={"agent": {"temperature": 0.2, "colour": "red"}})
        assert r.status_code == 400
        assert "agent.colour" in r.json()["error"]
        assert settings.load(refresh=True)["agent"]["temperature"] is None

    def test_bad_values_are_reported_not_stored(self, client):
        r = client.post("/api/settings", json={"agent": {"temperature": "warm"}})
        assert r.status_code == 422
        assert r.json()["errors"]

    def test_good_values_are_stored(self, client):
        r = client.post("/api/settings", json={"agent": {"temperature": 0.3}})
        assert r.status_code == 200
        assert r.json()["changed"] == ["agent.temperature"]
        assert settings.load(refresh=True)["agent"]["temperature"] == 0.3

    def test_a_non_object_body_is_400(self, client):
        assert (
            client.post("/api/settings", content=b"[1]", headers={"content-type": "application/json"}).status_code
            == 400
        )


def test_redacted_settings_keeps_shape_and_hides_only_secrets():
    data = {"security": {"auth_token": "x", "cors_origins": ["a"]}, "agent": {"model_id": "m"}}
    out = redacted_settings(data)
    assert out == {"security": {"auth_token": True, "cors_origins": ["a"]}, "agent": {"model_id": "m"}}
    assert data["security"]["auth_token"] == "x", "input is not mutated"


class TestAuthRoutes:
    def test_second_enrolment_needs_a_session(self, client, monkeypatch):
        monkeypatch.setattr(auth, "has_credentials", lambda: True)
        assert client.post("/api/auth/register/begin", json={"bootstrap": "x"}).status_code == 401

    def test_finish_without_a_credential_is_400(self, client):
        assert client.post("/api/auth/register/finish", json={"challenge_id": "c"}).status_code == 400
        assert client.post("/api/auth/login/finish", json={"credential": {}}).status_code == 400

    def test_logout_clears_the_cookie(self, client):
        r = client.post("/api/auth/logout")
        assert r.status_code == 200
        assert access.COOKIE in r.headers.get("set-cookie", "")
        assert "Max-Age=0" in r.headers["set-cookie"] or "expires" in r.headers["set-cookie"].lower()


class TestThePreSignInRouteSaysOnlyWhatTheLoginScreenNeeds:
    """``/api/auth/status`` answers whoever the socket lets through, so what it
    carries is a routing decision like any other. On a sealed dashboard the
    callers that reach it include a page the operator's own browser was told to
    load - a rebound name arrives on the loopback socket - so the enrolled
    passkeys are not among the fields it publishes.
    """

    @staticmethod
    def _seal_with_two_passkeys(isolated) -> None:
        """An owner who has already enrolled: the store decides the posture."""
        (isolated / "auth.json").write_text(
            json.dumps(
                {
                    "credentials": [
                        {"id": "cred-touchid", "name": "macbook touchid", "created": 1789500000},
                        {"id": "cred-yubikey", "name": "yubikey 5c nano", "created": 1789500900},
                    ]
                }
            )
        )

    def test_the_enrolled_passkeys_are_not_published_to_a_caller_with_no_session(self, client, isolated):
        self._seal_with_two_passkeys(isolated)
        rebound = {"host": "evil.example:8090", "origin": "http://evil.example:8090"}
        body = client.get("/api/auth/status", headers=rebound).json()
        assert "credentials" not in body
        # The one bit the login screen reads survives, and the route that does
        # serve the list still refuses the same caller.
        assert body["setup_required"] is False
        assert client.get("/api/auth/credentials", headers=rebound).status_code == 401

    def test_the_login_screen_still_gets_every_field_it_renders(self, client):
        body = client.get("/api/auth/status").json()
        assert body["setup_required"] is True
        assert body["bootstrap_source"] == "file"
        assert body["authenticated"] is False
        assert body["open_posture"] is True
        assert body["enabled"] is False

    def test_a_field_added_to_the_status_helper_is_not_public_until_it_is_named(self, client, monkeypatch):
        """The fields are named, not subtracted: growing the helper cannot widen
        this route by accident."""
        monkeypatch.setattr(
            auth, "status", lambda request=None: {"setup_required": True, "recovery_phrase": "correct horse"}
        )
        body = client.get("/api/auth/status").json()
        assert "recovery_phrase" not in body
        assert body["setup_required"] is True
