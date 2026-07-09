"""Behavior tests for the Reachy Mini Device Connect driver.

Exercises ``strands_robots.device_connect.reachy_mini_driver.ReachyMiniDriver``
end to end with all I/O mocked (no hardware, daemon, or network):

- Identity / status metadata.
- ``connect()`` auto-detection of Wireless (Zenoh) vs Lite (WebSocket) and the
  fail-safe (treat a daemon-status error as Wireless).
- Real-time commands (look / antennas / body) and unit conversion deg->rad.
- Cached sensor reads (getJoints / getImu), including the no-data branch.
- Motor torque RPCs (enableMotors / disableMotors) and id parsing.
- REST move/lifecycle RPCs (listMoves / wakeUp / sleep / stopMotion /
  getDaemonStatus) and the paths they hit.
- Expression sequences (nod / shake / happy) with sleep stubbed out.
- Caller-authorization fail-closed gating across the full mutating RPC surface.
- ``onEmergencyStop`` acting only on an allowlisted source.

These use the REAL device_connect_edge package so the @rpc caller-identity
hook is exercised; a fixture restores the genuine modules because sibling test
files replace them with MagicMocks at import time.
"""

import asyncio
import importlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _force_real_device_connect_edge():
    """Restore the genuine device_connect_edge modules and re-import the driver.

    Sibling test modules install MagicMock stand-ins in ``sys.modules`` for
    ``device_connect_edge`` at import time. A real module exposes ``__file__``;
    a MagicMock does not, so we drop the fakes, re-import the real package from
    disk, and purge ``strands_robots.device_connect.*`` so it re-binds to the
    real ``@rpc`` / ``DeviceDriver``.
    """
    for key in (
        "device_connect_edge.drivers",
        "device_connect_edge.types",
        "device_connect_edge.device",
        "device_connect_edge",
    ):
        mod = sys.modules.get(key)
        if mod is not None and not hasattr(mod, "__file__"):
            sys.modules.pop(key, None)
    importlib.import_module("device_connect_edge")
    importlib.import_module("device_connect_edge.drivers")
    importlib.import_module("device_connect_edge.types")
    for key in list(sys.modules):
        if key.startswith("strands_robots.device_connect"):
            sys.modules.pop(key, None)


@pytest.fixture
def rmd():
    """The reachy_mini_driver module bound to the real device_connect_edge."""
    _force_real_device_connect_edge()
    from strands_robots.device_connect import reachy_mini_driver as module

    return module


def _bare(rmd, **attrs):
    """A driver instance without running __init__ (no transport needed)."""
    drv = rmd.ReachyMiniDriver.__new__(rmd.ReachyMiniDriver)
    drv._host = "reachy.local"
    drv._api_port = 8000
    drv._latest_joints = None
    drv._latest_imu = None
    drv._hw = None
    for key, value in attrs.items():
        setattr(drv, key, value)
    return drv


def _run(coro):
    return asyncio.run(coro)


# -- identity / status ------------------------------------------------------


def test_identity_reports_host_and_manufacturer(rmd):
    drv = rmd.ReachyMiniDriver(host="bot.local", api_port=9001)
    ident = drv.identity
    assert ident.device_type == "reachy_mini"
    assert ident.manufacturer == "Pollen Robotics"
    assert "bot.local" in ident.model


def test_status_is_idle(rmd):
    assert rmd.ReachyMiniDriver(host="h").status.availability == "idle"


# -- connect / disconnect ---------------------------------------------------


def test_connect_wireless_uses_zenoh_link(rmd):
    drv = rmd.ReachyMiniDriver(host="h")
    zen, ws = MagicMock(return_value=AsyncMock()), MagicMock(return_value=AsyncMock())
    with (
        patch.object(rmd, "api", return_value={"wireless_version": True}),
        patch.object(rmd, "ZenohLink", zen),
        patch.object(rmd, "WebSocketLink", ws),
    ):
        _run(drv.connect())
    assert zen.called and not ws.called
    zen.return_value.start.assert_awaited_once()


def test_connect_lite_uses_websocket_link(rmd):
    drv = rmd.ReachyMiniDriver(host="h")
    zen, ws = MagicMock(return_value=AsyncMock()), MagicMock(return_value=AsyncMock())
    with (
        patch.object(rmd, "api", return_value={"wireless_version": False}),
        patch.object(rmd, "ZenohLink", zen),
        patch.object(rmd, "WebSocketLink", ws),
    ):
        _run(drv.connect())
    assert ws.called and not zen.called


def test_connect_treats_status_error_as_wireless(rmd):
    drv = rmd.ReachyMiniDriver(host="h")
    zen = MagicMock(return_value=AsyncMock())
    with patch.object(rmd, "api", side_effect=OSError("unreachable")), patch.object(rmd, "ZenohLink", zen):
        _run(drv.connect())
    assert zen.called


def test_connect_registers_sensor_callbacks(rmd):
    drv = rmd.ReachyMiniDriver(host="h")
    link = AsyncMock()
    with (
        patch.object(rmd, "api", return_value={"wireless_version": True}),
        patch.object(rmd, "ZenohLink", MagicMock(return_value=link)),
    ):
        _run(drv.connect())
    kwargs = link.start.await_args.kwargs
    kwargs["on_joints"]({"head_joint_positions": [0.1]})
    kwargs["on_imu"]({"temperature": 30})
    assert drv._latest_joints == {"head_joint_positions": [0.1]}
    assert drv._latest_imu == {"temperature": 30}


def test_disconnect_stops_link(rmd):
    drv = _bare(rmd, _hw=AsyncMock())
    _run(drv.disconnect())
    drv._hw.stop.assert_awaited_once()


def test_disconnect_without_link_is_noop(rmd):
    _run(_bare(rmd).disconnect())  # must not raise


def test_send_cmd_without_link_raises(rmd):
    with pytest.raises(RuntimeError, match="not connected"):
        _run(_bare(rmd)._send_cmd({"torque": True}))


# -- real-time movement -----------------------------------------------------


def test_look_sends_head_pose(rmd):
    drv = _bare(rmd, _hw=AsyncMock())
    res = _run(drv.look(pitch=10, roll=2, yaw=5))
    assert res == {"status": "success", "pitch": 10, "roll": 2, "yaw": 5}
    assert "head_pose" in drv._hw.send_cmd.await_args.args[0]


def test_antennas_converts_degrees_to_radians(rmd):
    import math

    drv = _bare(rmd, _hw=AsyncMock())
    res = _run(drv.antennas(left=90, right=-90))
    assert res == {"status": "success", "left": 90, "right": -90}
    cmd = drv._hw.send_cmd.await_args.args[0]["antennas_joint_positions"]
    assert cmd == pytest.approx([math.radians(90), math.radians(-90)])


def test_body_converts_yaw_to_radians(rmd):
    import math

    drv = _bare(rmd, _hw=AsyncMock())
    res = _run(drv.body(yaw=45))
    assert res["yaw"] == 45
    assert drv._hw.send_cmd.await_args.args[0]["body_yaw"] == pytest.approx(math.radians(45))


# -- cached sensor reads ----------------------------------------------------


def test_get_joints_returns_degrees(rmd):
    import math

    drv = _bare(rmd, _latest_joints={"head_joint_positions": [math.radians(30)], "antennas_joint_positions": [0.0]})
    res = _run(drv.getJoints())
    assert res["status"] == "success"
    assert res["head"] == pytest.approx([30.0])
    assert res["antennas"] == pytest.approx([0.0])


def test_get_joints_without_data_errors(rmd):
    assert _run(_bare(rmd).getJoints()) == {"status": "error", "reason": "no joint data"}


def test_get_imu_returns_cached_fields(rmd):
    imu = {"accelerometer": [0, 0, 9.8], "gyroscope": [0, 0, 0], "quaternion": [1, 0, 0, 0], "temperature": 31.2}
    res = _run(_bare(rmd, _latest_imu=imu).getImu())
    assert res["status"] == "success"
    assert res["temperature"] == 31.2 and res["accelerometer"] == [0, 0, 9.8]


def test_get_imu_without_data_errors(rmd):
    assert _run(_bare(rmd).getImu()) == {"status": "error", "reason": "no IMU data"}


# -- motors -----------------------------------------------------------------


def test_enable_motors_parses_ids(rmd):
    drv = _bare(rmd, _hw=AsyncMock())
    res = _run(drv.enableMotors("1, 2 ,3"))
    assert res == {"status": "success", "enabled": "1, 2 ,3"}
    assert drv._hw.send_cmd.await_args.args[0] == {"torque": True, "ids": ["1", "2", "3"]}


def test_disable_motors_defaults_to_all(rmd):
    drv = _bare(rmd, _hw=AsyncMock())
    res = _run(drv.disableMotors())
    assert res == {"status": "success", "disabled": "all"}
    assert drv._hw.send_cmd.await_args.args[0] == {"torque": False, "ids": None}


# -- REST move / lifecycle RPCs --------------------------------------------


def test_list_moves_targets_dances_library(rmd):
    calls = []
    with patch.object(rmd, "api", lambda *a, **k: calls.append(a) or {"moves": []}):
        res = _run(_bare(rmd).listMoves("dance"))
    assert res["status"] == "success"
    assert calls[0][2].endswith("reachy-mini-dances-library")


def test_list_moves_defaults_to_emotions_library(rmd):
    calls = []
    with patch.object(rmd, "api", lambda *a, **k: calls.append(a) or {}):
        _run(_bare(rmd).listMoves())
    assert calls[0][2].endswith("reachy-mini-emotions-library")


@pytest.mark.parametrize(
    ("method", "path", "verb"),
    [
        ("wakeUp", "/api/move/play/wake_up", "POST"),
        ("sleep", "/api/move/play/goto_sleep", "POST"),
        ("stopMotion", "/api/move/stop", "POST"),
    ],
)
def test_lifecycle_rpcs_hit_expected_endpoint(rmd, method, path, verb):
    seen = {}

    def fake_api(host, port, p, m="GET", data=None):
        seen["path"], seen["verb"] = p, m
        return {"ok": True}

    with patch.object(rmd, "api", fake_api):
        res = _run(getattr(_bare(rmd), method)())
    assert res["status"] == "success"
    assert seen == {"path": path, "verb": verb}


def test_get_daemon_status_merges_result(rmd):
    with patch.object(rmd, "api", return_value={"motors_on": True, "freq": 100}):
        res = _run(_bare(rmd).getDaemonStatus())
    assert res == {"status": "success", "motors_on": True, "freq": 100}


# -- expressions ------------------------------------------------------------


@pytest.mark.parametrize(("method", "expression"), [("nod", "nod"), ("shake", "shake"), ("happy", "happy")])
def test_expression_returns_to_neutral(rmd, method, expression):
    drv = _bare(rmd, _hw=AsyncMock())
    with patch.object(rmd.asyncio, "sleep", AsyncMock()):
        res = _run(getattr(drv, method)())
    assert res == {"status": "success", "expression": expression}
    # Each expression runs an oscillation loop then a final neutral command.
    assert drv._hw.send_cmd.await_count >= 3


# -- caller authorization ---------------------------------------------------


# Every state-mutating RPC (real-time head/antenna/body commands, motor
# torque toggles, recorded-move playback, expression sequences, and the
# REST move/lifecycle calls) must consult the caller allowlist. Read-only
# RPCs (getJoints / getImu / listMoves / getDaemonStatus) are intentionally
# ungated and excluded here.
_MUTATING_RPCS = [
    ("look", {"pitch": 5}),
    ("antennas", {"left": 10, "right": -10}),
    ("body", {"yaw": 15}),
    ("enableMotors", {"motor_ids": "1,2"}),
    ("disableMotors", {"motor_ids": ""}),
    ("playMove", {"move_name": "wave"}),
    ("nod", {}),
    ("shake", {}),
    ("happy", {}),
    ("wakeUp", {}),
    ("sleep", {}),
    ("stopMotion", {}),
]


@pytest.mark.parametrize(("method", "kwargs"), _MUTATING_RPCS, ids=[m for m, _ in _MUTATING_RPCS])
def test_mutating_rpc_denied_for_unlisted_caller(rmd, monkeypatch, method, kwargs):
    """Every mutating RPC fail-closes on an unauthorized caller.

    With an allowlist configured and an anonymous caller (no source_device),
    each state-mutating RPC must return the standard authorization error and
    issue neither a real-time hardware command nor a daemon REST call. This
    pins the contract for the whole mutating surface so a new RPC that forgets
    the gate cannot slip through with only ``look`` covered.
    """
    monkeypatch.setenv("DEVICE_CONNECT_RPC_ALLOW", "trusted-*")
    drv = _bare(rmd, _hw=AsyncMock())
    with patch.object(rmd, "api", MagicMock()) as fake_api, patch.object(rmd.asyncio, "sleep", AsyncMock()):
        res = _run(getattr(drv, method)(**kwargs))
    assert res["status"] == "error"
    assert "not authorized" in res["reason"]
    # The rejection names the specific RPC that was denied.
    assert method in res["reason"]
    # Fail-closed: no physical command and no daemon REST call may fire.
    drv._hw.send_cmd.assert_not_awaited()
    fake_api.assert_not_called()


# -- emergency stop ---------------------------------------------------------


def test_emergency_stop_disables_motors_for_allowlisted_source(rmd, monkeypatch):
    monkeypatch.setenv("DEVICE_CONNECT_ESTOP_ALLOW", "safety-*")
    drv = _bare(rmd, _hw=AsyncMock())
    with patch.object(rmd, "api", return_value={"ok": True}):
        _run(drv.onEmergencyStop("safety-007", "emergencyStop", {}))
    # Disables motors (torque off) as part of the stop reaction.
    assert any(c.args[0].get("torque") is False for c in drv._hw.send_cmd.await_args_list)


def test_emergency_stop_ignores_unauthorized_source(rmd, monkeypatch):
    monkeypatch.setenv("DEVICE_CONNECT_ESTOP_ALLOW", "safety-*")
    drv = _bare(rmd, _hw=AsyncMock())
    with patch.object(rmd, "api", return_value={"ok": True}) as fake_api:
        _run(drv.onEmergencyStop("rogue-device", "emergencyStop", {}))
    drv._hw.send_cmd.assert_not_awaited()
    fake_api.assert_not_called()
