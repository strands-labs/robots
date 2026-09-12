"""``robot_mesh(action="emergency_stop")`` over Device Connect reads each device's answer.

A device answers its ``stop`` RPC with an envelope: an authorization refusal
(``{"status": "error", "reason": "caller not authorized for 'stop'"}``) or a
``stop_policy`` that could not halt a rollout both arrive as a *delivered*
reply, not as a raised ``invoke``. The dispatcher used to count delivery as a
stop, so a fleet whose every device refused was reported as
``E-STOP: N/N devices stopped`` under ``status="success"``. The mesh path
grades the same verdict through :func:`~strands_robots.mesh.core._reports_failure_to_stop`;
this pins the Device Connect path to the same rule.

Hardware-free: ``device_connect_agent_tools.connection`` is a fake module.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import strands_robots.tools.robot_mesh as rm

REFUSAL = {"status": "error", "reason": "caller not authorized for 'stop'", "caller": "my-agent"}
HALTED = {"status": "success", "content": [{"text": "Halted 1 rollout(s): so100"}]}


@pytest.fixture
def dc_fleet(monkeypatch):
    """Two discovered devices whose ``stop`` answers are scripted per device id."""
    answers: dict[str, dict[str, Any]] = {}
    conn = types.SimpleNamespace()
    conn.list_devices = lambda: [{"device_id": device_id} for device_id in answers]
    conn.invoke = lambda device_id, func, params, timeout: {"jsonrpc": "2.0", "result": answers[device_id]}
    module = types.ModuleType("device_connect_agent_tools.connection")
    module.get_connection = lambda: conn
    pkg = types.ModuleType("device_connect_agent_tools")
    pkg.connection = module
    monkeypatch.setitem(sys.modules, "device_connect_agent_tools", pkg)
    monkeypatch.setitem(sys.modules, "device_connect_agent_tools.connection", module)
    monkeypatch.setattr(rm, "_audit_tool_action", lambda *a, **k: None)
    return answers


def _estop() -> dict[str, Any]:
    result = rm._device_connect_dispatch("emergency_stop", "", "", "", "mock", 0, 30.0, 30.0)
    assert result is not None
    return result


def test_a_device_that_refused_its_stop_is_reported_as_not_stopped(dc_fleet):
    dc_fleet["so100-lab-1"] = REFUSAL
    dc_fleet["aloha-lab-2"] = HALTED

    result = _estop()

    text = result["content"][0]["text"]
    assert result["status"] == "error", text
    assert "1/2 devices stopped" in text
    assert "so100-lab-1" in text
    assert "caller not authorized" in text


def test_a_fleet_that_halted_everywhere_is_a_success(dc_fleet):
    dc_fleet["so100-lab-1"] = HALTED
    dc_fleet["aloha-lab-2"] = HALTED

    result = _estop()

    assert result["status"] == "success"
    assert result["content"][0]["text"] == "E-STOP: 2/2 devices stopped"
