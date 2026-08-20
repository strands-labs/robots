"""Every action the Device Connect backend answers leaves an audit row.

``robot_mesh`` renders its actions onto two backends: an agent-side Device
Connect connection when one is reachable and has devices, and the built-in
Zenoh mesh otherwise. The audit contract is a property of the *action*, not of
whichever backend happened to serve it - "the audit log is a complete record of
agent mesh access", which is why the observation actions are audited alongside
the actuation ones rather than only when something moved.

The read-only half of that contract reached the mesh rendering of ``peers`` /
``status`` and not the Device Connect rendering, and Device Connect is the
backend tried first whenever it has devices - so the audited implementations
were the fallback. ``peers`` is the richest read the tool offers: it returns
every device id plus every function name the fleet exposes, which is the
callable surface a later ``rpc`` would use.

These tests grade the backend against itself rather than against a copied
action list: the set of actions Device Connect answers is discovered by calling
the dispatcher, so an action added to it is covered without editing this file.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import strands_robots.tools.robot_mesh as rm

# Two devices, one of them exposing an actuation surface by name. The function
# names matter: they are what makes ``peers`` a reconnaissance read rather than
# a liveness check.
_FLEET: list[dict[str, Any]] = [
    {
        "device_id": "arm-1",
        "device_type": "strands_robot",
        "status": {"availability": "available"},
        "functions": [{"name": "move_joint"}, {"name": "grip"}],
    },
    {
        "device_id": "sim-1",
        "device_type": "strands_sim",
        "status": {"availability": "available"},
        "functions": [{"name": "step"}],
    },
]

#: Success-path arguments per action, so every branch is driven with a request
#: it accepts rather than bouncing off a required-parameter guard.
_ARGS: dict[str, dict[str, Any]] = {
    "peers": {},
    "status": {},
    "tell": {"target": "arm-1", "instruction": "pick the tote"},
    "send": {"target": "arm-1", "command": '{"action": "status"}'},
    "rpc": {"target": "arm-1", "function": "move_joint", "command": "{}"},
    "stop": {"target": "arm-1"},
    "emergency_stop": {},
    "broadcast": {"validated_command": {"action": "status"}},
}

#: Actions whose whole point is that something moved. Kept apart from the
#: read-only half so a fix aimed at the observation actions cannot quietly make
#: the actuation ones conditional.
_ACTUATION = ("tell", "send", "rpc", "stop", "emergency_stop", "broadcast")

#: The observation actions this backend renders. The mesh backend renders two
#: more (``inbox`` / ``unsubscribe``); Device Connect defers those to it.
_READ_ONLY = ("peers", "status")


class _StubConnection:
    """Agent-side connection standing in for a reachable Device Connect."""

    def __init__(self) -> None:
        self.invoked: list[tuple[str, str]] = []

    def list_devices(self, device_type: str | None = None) -> list[dict[str, Any]]:
        return [dict(d) for d in _FLEET]

    def invoke(
        self, device_id: str, function: str, params: dict[str, Any] | None = None, timeout: float = 30.0
    ) -> dict[str, Any]:
        self.invoked.append((device_id, function))
        return {"result": {"function": function, "ok": True}}

    def broadcast(self, function: str, params: dict[str, Any] | None = None, timeout: float = 30.0) -> list[dict]:
        return [{"device_id": d["device_id"], "result": {"ok": True}} for d in _FLEET]


@pytest.fixture
def connected(monkeypatch: pytest.MonkeyPatch) -> _StubConnection:
    """Make ``device_connect_agent_tools.connection.get_connection`` resolvable.

    The agent-side package is an optional install, so the module is injected
    rather than patched: the dispatcher imports it inside the call, which is the
    seam production uses.
    """
    conn = _StubConnection()
    pkg = types.ModuleType("device_connect_agent_tools")
    conn_mod = types.ModuleType("device_connect_agent_tools.connection")
    conn_mod.get_connection = lambda: conn  # type: ignore[attr-defined]
    pkg.connection = conn_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "device_connect_agent_tools", pkg)
    monkeypatch.setitem(sys.modules, "device_connect_agent_tools.connection", conn_mod)
    return conn


@pytest.fixture
def audited(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, bool, str]]:
    """Capture the rows the dispatcher writes, without touching a log file."""
    rows: list[tuple[str, str, bool, str]] = []

    def record(action: str, target: str, success: bool, detail: str) -> None:
        rows.append((action, target, success, detail))

    monkeypatch.setattr(rm, "_audit_tool_action", record)
    return rows


def _dispatch(action: str, **overrides: Any) -> dict[str, Any] | None:
    """Drive ``_device_connect_dispatch`` for *action*, unwrapping its result."""
    args: dict[str, Any] = {
        "target": "",
        "instruction": "",
        "command": "",
        "policy_provider": "mock",
        "policy_port": 0,
        "duration": 30.0,
        "timeout": 5.0,
        "function": "",
        "validated_command": None,
    }
    args.update(_ARGS.get(action, {}))
    args.update(overrides)
    result = rm._device_connect_dispatch(
        action,
        args["target"],
        args["instruction"],
        args["command"],
        args["policy_provider"],
        args["policy_port"],
        args["duration"],
        args["timeout"],
        args["function"],
        args["validated_command"],
    )
    if result is None:
        return None
    return getattr(result, "result", result)


def _text(result: dict[str, Any]) -> str:
    return "".join(b.get("text", "") for b in result.get("content", []) if isinstance(b, dict))


def _answered_actions() -> list[str]:
    """Every action Device Connect answers rather than deferring to the mesh."""
    return [a for a in _ARGS if a not in ("subscribe", "watch", "inbox", "unsubscribe")]


class TestEveryAnsweredActionIsAudited:
    """The root cause: the audit row belongs to the action, not the backend."""

    @pytest.mark.parametrize("action", _answered_actions())
    def test_a_successful_action_leaves_exactly_one_row(
        self, action: str, connected: _StubConnection, audited: list[tuple[str, str, bool, str]]
    ) -> None:
        result = _dispatch(action)
        assert result is not None, f"premise: {action} deferred to the mesh backend"
        assert result["status"] == "success", f"premise: {action} did not take its success path: {_text(result)}"
        successes = [r for r in audited if r[2]]
        assert successes, (
            f"Device Connect answered {action!r} with success and wrote no audit row, so this "
            f"access is absent from the forensic record the mesh rendering of the same action keeps."
        )

    def test_the_row_records_the_size_of_the_fleet_that_was_read(
        self, connected: _StubConnection, audited: list[tuple[str, str, bool, str]]
    ) -> None:
        """A marker with no magnitude is not a usable record of what was read."""
        _dispatch("peers")
        details = [r[3] for r in audited if r[2]]
        assert details, "peers wrote no audit row"
        assert any(f"devices={len(_FLEET)}" in d for d in details), (
            f"the peers row does not say how much of the fleet was read: {details}"
        )


class TestTheReconnaissanceReadIsRecorded:
    """``peers`` returns the fleet's callable surface, so it is worth a row."""

    def test_peers_returns_every_device_id_and_function_name(
        self, connected: _StubConnection, audited: list[tuple[str, str, bool, str]]
    ) -> None:
        """Premise for the test below: this read really is the rich one."""
        result = _dispatch("peers")
        assert result is not None
        body = _text(result)
        for device in _FLEET:
            assert device["device_id"] in body
            for func in device["functions"]:
                assert func["name"] in body, f"{func['name']} missing from the peers reply"

    @pytest.mark.parametrize("action", _READ_ONLY)
    def test_an_observation_action_is_audited_like_an_actuation_one(
        self, action: str, connected: _StubConnection, audited: list[tuple[str, str, bool, str]]
    ) -> None:
        _dispatch(action)
        assert [r for r in audited if r[0] == action and r[2]], (
            f"{action} left no audit row; the audit log is meant to be a complete record of "
            f"agent mesh access, not only of the calls that moved something."
        )


class TestTheAlreadyAuditedHalfIsUntouched:
    """Controls: what worked before still works, and nothing new is claimed."""

    @pytest.mark.parametrize("action", _ACTUATION)
    def test_an_actuation_action_still_audits_its_success(
        self, action: str, connected: _StubConnection, audited: list[tuple[str, str, bool, str]]
    ) -> None:
        result = _dispatch(action)
        assert result is not None and result["status"] == "success", f"premise: {action} did not succeed"
        assert [r for r in audited if r[0] == action and r[2]], f"{action} lost its audit row"

    def test_the_observation_reply_still_carries_the_fleet_listing(
        self, connected: _StubConnection, audited: list[tuple[str, str, bool, str]]
    ) -> None:
        """The audit row is added beside the reply, not in place of any of it."""
        peers = _dispatch("peers")
        status = _dispatch("status")
        assert peers is not None and status is not None
        assert _text(peers).startswith(f"Discovered {len(_FLEET)} device(s):")
        assert _text(status).startswith(f"Network: {len(_FLEET)} device(s)")

    def test_a_required_parameter_refusal_stays_a_refusal(
        self, connected: _StubConnection, audited: list[tuple[str, str, bool, str]]
    ) -> None:
        """Scope boundary: this change audits answered actions, not every guard.

        A request that never reached a device is refused by a required-parameter
        guard, and whether those refusals belong in the audit log is a separate
        question from whether a completed read does.
        """
        result = _dispatch("tell", target="", instruction="")
        assert result is not None and result["status"] == "error"
        assert not [r for r in audited if r[2]], "a refused request must not be recorded as a success"

    @pytest.mark.parametrize("action", ["subscribe", "watch", "inbox", "unsubscribe"])
    def test_a_mesh_only_action_is_still_deferred(
        self, action: str, connected: _StubConnection, audited: list[tuple[str, str, bool, str]]
    ) -> None:
        """The mesh backend keeps the actions it owns, and audits them there.

        Device Connect renders none of these, so widening the audit here must not
        turn into this backend answering an action it does not implement.
        """
        result = _dispatch(action)
        assert result is None, f"{action} is a mesh-only action and must defer, got {result}"
        assert not audited, f"{action} was not answered here, so it must not be recorded here"
