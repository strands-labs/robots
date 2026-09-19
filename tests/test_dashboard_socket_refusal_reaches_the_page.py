"""A refused socket carries its reason to the page, and a stranger's does not.

The dashboard refuses a WebSocket with an application close code - 4401 "sign in
required", 4404 "no such session" - and the route's docstring, its tests and the
page that opens the socket all read that code as the reason - ``static/app.js``
turns 4401 on the agent socket into the login screen and nothing else does. A close sent
*before* ``accept`` is not a close, though: it is a handshake rejection, which
an ASGI server answers with HTTP 403, and a browser reports 1006 for every one
of them alike. So what these cells grade is the ASGI message order, because that
order is the whole of whether the code reaches the wire:

    accept, close(4401)   the page reads 4401
    close(4401)           the page reads 1006, whatever the code said

``TestClient`` cannot see the difference - it lifts the code out of the close
message either way - so the app is driven here through the ASGI interface
directly, with the lifespan already run by ``TestClient`` so ``app.state`` is
populated.

The exception is a caller from another origin. WebSockets are exempt from CORS,
so a page anywhere can open one at ``ws://127.0.0.1:8090``; it is refused at the
handshake, unaccepted, and told nothing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard import settings  # noqa: E402
from strands_robots.dashboard.server import create_app  # noqa: E402

HOST = "127.0.0.1:8090"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    """The real app with its lifespan run, so ``app.state.safety`` answers."""
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    settings.clear_overrides()
    settings.load(refresh=True)
    built = create_app()
    with TestClient(built):
        yield built
    built.state.safety.store.shutdown()


def sent_while_refusing(app: Any, path: str, headers: dict[str, str]) -> list[dict[str, Any]]:
    """Every ASGI message *app* sends while it refuses a socket at *path*.

    Only for paths the app refuses: an accepted telemetry socket streams until
    the client goes away, and this receive channel never says it did.
    """
    messages: list[dict[str, Any]] = []
    scope: dict[str, Any] = {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "scheme": "ws",
        "server": ("127.0.0.1", 8090),
        "client": ("127.0.0.1", 54321),
        "root_path": "",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "subprotocols": [],
        "state": {},
    }

    async def receive() -> dict[str, Any]:
        return {"type": "websocket.connect"}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    asyncio.run(app(scope, receive, send))
    return messages


class TestARefusedSocketSaysWhy:
    @pytest.mark.parametrize(
        ("code", "headers", "auth_on"),
        [
            pytest.param(4401, {"host": HOST}, True, id="sign-in-required"),
            pytest.param(4404, {"host": HOST}, False, id="no-such-session"),
        ],
    )
    def test_the_socket_is_accepted_first_so_the_close_code_reaches_the_page(
        self, app, monkeypatch, code: int, headers: dict[str, str], auth_on: bool
    ) -> None:
        if auth_on:
            monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "1")
        messages = sent_while_refusing(app, "/ws/telemetry/nosuch", headers)
        assert [m["type"] for m in messages] == ["websocket.accept", "websocket.close"]
        assert messages[-1]["code"] == code

    def test_a_socket_from_another_origin_is_refused_at_the_handshake(self, app) -> None:
        """Unaccepted: HTTP 403 to the handshake, and the page is owed no reason."""
        messages = sent_while_refusing(app, "/ws/telemetry/nosuch", {"host": HOST, "origin": "http://evil.example"})
        assert [m["type"] for m in messages] == ["websocket.close"]
        assert messages[-1]["code"] == 4401

    def test_the_agent_socket_delivers_the_4401_the_page_acts_on(self, app, monkeypatch) -> None:
        """``app.js``: ``ws.onclose = ev => { if (ev.code === 4401) showLogin(); ... }``.

        Sent before ``accept`` that code is 1006 by the time it reaches the
        handler, and the operator whose session expired is shown nothing.
        """
        import pathlib

        import strands_robots.dashboard as pkg

        app_js = (pathlib.Path(pkg.__file__).parent / "static" / "app.js").read_text(encoding="utf-8")
        assert "ev.code === 4401" in app_js

        monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "1")
        messages = sent_while_refusing(app, "/ws/agent", {"host": HOST})
        assert [m["type"] for m in messages] == ["websocket.accept", "websocket.close"]
        assert messages[-1]["code"] == 4401
