"""``robot_mesh`` refuses an unknown action before it touches the mesh.

The action name is the first thing a call gets wrong when it is wrong, and no
other step is owed to an action that does not exist. In a robot-less process
the tool used to record a rate-limit slot, probe Device Connect and bring up a
gateway mesh (one heartbeat wait) for the typo, then answer it with "no local
mesh found. Construct a Robot()/Simulation() first" - a remedy for a different
problem.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.tools.robot_mesh import robot_mesh


def _call(**kwargs: Any) -> dict[str, Any]:
    fn = getattr(robot_mesh, "original", None) or robot_mesh
    return fn(tool_context=MagicMock(name="ToolContext"), **kwargs)


def test_unknown_action_in_a_robot_less_process_is_named_and_the_gateway_is_never_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("STRANDS_MESH", raising=False)
    monkeypatch.setenv("STRANDS_ROBOT_MESH_DC", "off")
    gateway = MagicMock(name="_gateway_mesh", side_effect=AssertionError("gateway built for an unknown action"))
    with (
        patch("strands_robots.mesh.get_local_robots", return_value={}),
        patch("strands_robots.mesh.session.get_peers", return_value=[]),
        patch("strands_robots.tools.robot_mesh._gateway_mesh", gateway),
    ):
        out = _call(action="warp")
    text = out["content"][0]["text"]
    assert out["status"] == "error"
    assert "unknown action: 'warp'" in text, text
    assert "no local mesh" not in text, text
    for verb in ("peers", "status", "tell", "send", "rpc", "broadcast", "stop", "emergency_stop"):
        assert verb in text
    gateway.assert_not_called()
