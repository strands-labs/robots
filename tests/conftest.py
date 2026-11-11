import importlib.util

import pytest

from strands_robots.simulation import Simulation

# The agent-turn tests need strands (the [tools] extra); without it they are not collected.
collect_ignore_glob = (
    []
    if importlib.util.find_spec("strands")
    else ["tools/test_agent_turn.py", "tools/test_sim_tool.py", "tools/test_tool_gate.py"]
)


@pytest.fixture(autouse=True)
def _fresh_default_world():
    """Every test starts without a process-global default world and leaves none behind."""
    if Simulation._default is not None:
        Simulation._default.close()
    Simulation._default = None
    yield
    if Simulation._default is not None:
        Simulation._default.close()
    Simulation._default = None


_LOOPBACK = ("localhost", "127.0.0.1", "::1", "[::1]")


def dashboard_token() -> str:
    """A session token for the dashboards this test starts (they share the test's passkey store)."""
    from strands_robots.dashboard.auth import Auth

    return Auth().token("tests")


def browser_session(host: str) -> tuple[str, str]:
    """A browser session's two halves (cookie token, page proof) as a passkey sign-in on ``host`` hands them out."""
    from strands_robots.dashboard.auth import Auth, session

    auth = Auth()
    token = session.mint(auth.store.secret, "token:tests", auth.ttl, aud=host)
    return token, auth.proof(token)


@pytest.fixture(autouse=True)
def _dashboard_signed_in(request, tmp_path, monkeypatch):
    """Each test gets its own dashboard passkey store; its httpx and websockets calls to a loopback dashboard carry a
    session token, as a signed-in script's would. ``@pytest.mark.signed_out`` sends none."""
    monkeypatch.setenv("STRANDS_ROBOTS_DASHBOARD_AUTH", str(tmp_path / "dashboard-auth.json"))
    if request.node.get_closest_marker("signed_out"):
        return
    try:
        import httpx
    except ImportError:
        return
    send = httpx.Client.send

    def signed(self, req, **kw):
        if req.url.host in _LOOPBACK and "authorization" not in req.headers:
            req.headers["authorization"] = f"Bearer {dashboard_token()}"
        return send(self, req, **kw)

    monkeypatch.setattr(httpx.Client, "send", signed)
    try:
        import websockets.sync.client as wsc
    except ImportError:
        return
    connect = wsc.connect

    def ws_signed(uri, *a, additional_headers=None, **kw):
        headers = dict(additional_headers or {})
        if not any(k.lower() == "authorization" for k in headers):
            headers["Authorization"] = f"Bearer {dashboard_token()}"
        return connect(uri, *a, additional_headers=headers, **kw)

    monkeypatch.setattr(wsc, "connect", ws_signed)


@pytest.fixture(autouse=True)
def _recordings_in_tmp(tmp_path, monkeypatch):
    """The dashboard records only inside STRANDS_ROBOTS_RECORDINGS: each test's own tmp_path."""
    monkeypatch.setenv("STRANDS_ROBOTS_RECORDINGS", str(tmp_path))
