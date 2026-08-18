"""Enforce the Strands tool-result contract across ``strands_robots``.

Tool results must expose exactly ``{"status", "content"}`` at the top level
(the runtime may add ``toolUseId``). Structured telemetry belongs inside
``content`` as a ``{"json": {...}}`` block, never as extra top-level keys --
extras are dropped by the agent runtime and never reach ``agent.messages``.

The repo-wide ``test_no_extra_top_level_keys`` test is the standing guard: it
statically scans every tool-result-shaped dict literal in the package and
fails if any carries extra top-level keys. A ``**spread`` inside such a dict is
itself a violation: it can inject arbitrary top-level keys that the runtime
drops, and it previously let smuggled telemetry slip past the scan. The
remaining tests apply
``assert_strands_tool_result`` to live results from representative call sites
(teleop stats, simulation policy rollout, action dispatch).
"""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest

import strands_robots
from strands_robots.teleop_mixin import TeleopMixin
from tests.tool_result_contract import (
    VALID_TOP_LEVEL_KEYS,
    assert_strands_tool_result,
    tool_json,
)

_PKG_ROOT = Path(strands_robots.__file__).resolve().parent


def _tool_result_violations() -> list[str]:
    """Statically find tool-result dict literals that break the top-level contract.

    A dict literal counts as tool-result-shaped when it has constant string
    keys including both ``status`` and ``content``. Such a dict violates the
    contract when it carries either:

    * an extra constant key outside ``{"status", "content", "toolUseId"}``, or
    * a ``**spread`` entry -- a spread can inject arbitrary top-level keys that
      the runtime silently drops, so it is never valid at the top level of a
      tool result (telemetry belongs inside a ``content`` json block instead).

    The spread case is the subtle one: a spread key appears in
    ``ast.Dict.keys`` as ``None``, so a naive "all keys are constant strings"
    filter skips the whole dict and lets the smuggled keys through.
    """
    violations: list[str] = []
    for path in _PKG_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            const_keys = {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if not ({"status", "content"} <= const_keys):
                continue
            problems: list[str] = []
            extra = const_keys - set(VALID_TOP_LEVEL_KEYS)
            if extra:
                problems.append(f"extra_keys={sorted(extra)}")
            if any(k is None for k in node.keys):  # a **spread entry
                problems.append("**spread (may inject dropped top-level keys)")
            if problems:
                rel = path.relative_to(_PKG_ROOT.parent)
                violations.append(f"{rel}:{node.lineno} " + " ".join(problems))
    return violations


def test_no_extra_top_level_keys():
    """No tool-result-shaped return in the package may carry extra top-level keys."""
    violations = _tool_result_violations()
    assert not violations, "Tool-result contract violations found:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# Representative call site 1: TeleopMixin._teleop_stats (success/degraded/error)
# ---------------------------------------------------------------------------


class _FakeHost(TeleopMixin):
    def __init__(self):
        self.tool_name_str = "ct_host"
        self.mesh = None
        self.peer_id = None
        self._send_lock = threading.Lock()

    def send_action(self, action, robot_name=None, n_substeps=1):  # noqa: ARG002
        return {"status": "success", "content": [{"text": "ok"}]}


@pytest.mark.parametrize(
    ("frames", "errors", "expected"),
    [
        (100, 0, "success"),  # clean run
        (100, 5, "degraded"),  # some per-frame errors, mostly fine
        (0, 0, "success"),  # no frames, no errors (idle stop)
        (10, 10, "error"),  # every frame errored
        (0, 3, "error"),  # no good frames at all
    ],
)
def test_teleop_stats_contract_and_status(frames, errors, expected):
    host = _FakeHost()
    host._ensure_teleop_state()
    host._teleop_frames = frames
    host._teleop_errors = errors
    host._teleop_start_mono = 1.0  # non-zero so elapsed/hz compute

    result = host._teleop_stats(blocking=True)
    assert_strands_tool_result(result)
    assert result["status"] == expected

    # Telemetry must live in the json content block, not at the top level.
    json_blocks = [b["json"] for b in result["content"] if "json" in b]
    assert len(json_blocks) == 1
    telemetry = json_blocks[0]
    assert telemetry["frames"] == frames
    assert telemetry["errors"] == errors
    assert "hz_actual" in telemetry
    assert telemetry["status"] == expected


def test_teleop_status_and_list_contract():
    host = _FakeHost()
    host._ensure_teleop_state()
    assert_strands_tool_result(host.get_teleoperate_status())
    assert_strands_tool_result(host.list_teleops())


# ---------------------------------------------------------------------------
# Representative call sites 2 & 3: Simulation.run_policy + action dispatch
# ---------------------------------------------------------------------------


@pytest.fixture
def sim():
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import Simulation

    s = Simulation(tool_name="contract_sim", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


def test_run_policy_result_contract(sim):
    result = sim.run_policy(policy_provider="mock", n_steps=2)
    assert_strands_tool_result(result)


def test_dispatch_action_results_contract(sim):
    for action, params in [
        ("list_robots", {}),
        ("reset", {}),
        ("step", {"n_steps": 1}),
        ("get_gravity", {}),
    ]:
        assert_strands_tool_result(sim._dispatch_action(action, params))


# ---------------------------------------------------------------------------
# Regression: session "status" tool results must keep telemetry inside a json
# content block, not as extra top-level keys. These used to smuggle
# ``session_name/pid/uptime/is_running`` (plus ``**session_info``) at the top
# level via a ``**spread``, which the static scan skipped and the runtime
# dropped -- so the agent asked for session status and saw only the text block.
# ---------------------------------------------------------------------------


def test_teleoperate_status_keeps_telemetry_in_json_block(tmp_path, monkeypatch):
    import os

    import strands_robots.tools.lerobot_teleoperate as tele_mod

    pid = os.getpid()  # a real, running pid so the session is not pruned as dead
    monkeypatch.setattr(tele_mod, "SESSION_DIR", tmp_path)
    tele_mod.SessionManager().add_session(
        "live",
        {"pid": pid, "action": "record", "start_time": 0.0, "robot_type": "so101_follower"},
    )

    result = tele_mod.lerobot_teleoperate(action="status", session_name="live")

    assert_strands_tool_result(result)
    telemetry = tool_json(result)
    assert telemetry["session_name"] == "live"
    assert telemetry["pid"] == pid
    assert telemetry["is_running"] is True
    assert "uptime" in telemetry


def test_train_status_keeps_telemetry_in_json_block(tmp_path, monkeypatch):
    import os

    import strands_robots.tools.lerobot_train as train_mod

    pid = os.getpid()  # a real, running pid so the session is not pruned as dead
    monkeypatch.setattr(train_mod, "SESSION_DIR", tmp_path)
    train_mod.SessionManager().add_session(
        "live",
        {"pid": pid, "action": "train", "start_time": 0.0, "policy_type": "act"},
    )

    result = train_mod.lerobot_train(action="status", dataset_root="/x", session_name="live")

    assert_strands_tool_result(result)
    telemetry = tool_json(result)
    assert telemetry["session_name"] == "live"
    assert telemetry["pid"] == pid
    assert telemetry["is_running"] is True
    assert "uptime" in telemetry
