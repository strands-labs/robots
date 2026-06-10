"""Tests for ``Mesh._dispatch`` against sim peers (issue #303).

The HardwareRobot dispatch path is covered by ``test_mesh_rpc.py``. This
module pins the sim-peer branch added by issue #303: ``tell()`` against a
peer whose ``robot`` is a ``Simulation`` (or any ``SimEngine``-shaped
object) routes ``execute`` -> ``run_policy`` and ``start`` -> ``start_policy``
with the issue #300 well-known kwargs (``target_pose`` / ``target_joints`` /
``world_update``) forwarded into ``policy_config``.

Tests are 100% mocked — no MuJoCo / Isaac install required. A
``_FakeSim`` exposes the SimEngine surface duck-typed minimally to what
``_dispatch_sim_policy`` needs.
"""

from __future__ import annotations

from typing import Any

from strands_robots.mesh import Mesh


class _FakeSim:
    """Minimal duck-typed stand-in for ``Simulation`` / ``SimEngine``.

    Records every ``run_policy`` / ``start_policy`` call so tests can
    assert on the forwarded arguments. The presence of ``_world`` +
    ``run_policy`` + ``list_robots`` is what ``Mesh._dispatch`` keys off
    of to pick the sim branch over the HardwareRobot one.
    """

    def __init__(self, robots: list[str] | None = None) -> None:
        # The mesh dispatcher checks for a non-None ``_world`` to confirm
        # the sim has been initialised. Use a sentinel object, not just
        # truthy — matches MuJoCoSimEngine's "_world is None" gate.
        self._world: Any = object()
        self._robots = list(robots if robots is not None else ["so100"])
        self.run_policy_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.start_policy_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.tool_name_str = "fakesim"

    def list_robots(self) -> list[str]:
        return list(self._robots)

    def run_policy(self, robot_name: str, **kwargs: Any) -> dict[str, Any]:
        self.run_policy_calls.append(((robot_name,), kwargs))
        return {"status": "success", "content": [{"text": f"ran {robot_name}"}]}

    def start_policy(self, robot_name: str, **kwargs: Any) -> dict[str, Any]:
        self.start_policy_calls.append(((robot_name,), kwargs))
        return {"status": "success", "content": [{"text": f"started {robot_name}"}]}


# Sim-peer routing
def test_execute_routes_to_run_policy_with_default_robot() -> None:
    """Single-robot sim: omitting ``robot_name`` defaults to the only robot."""
    sim = _FakeSim(robots=["so100"])
    m = Mesh(sim, peer_id="sim-a")
    out = m._dispatch(
        {
            "action": "execute",
            "instruction": "wave",
            "policy_provider": "mock",
        }
    )
    assert out["status"] == "success"
    assert len(sim.run_policy_calls) == 1
    args, kwargs = sim.run_policy_calls[0]
    assert args == ("so100",)
    assert kwargs["instruction"] == "wave"
    assert kwargs["policy_provider"] == "mock"
    # Even when the caller passes no extras, we always forward an empty
    # policy_config dict so the receiving sim sees a stable type.
    assert kwargs["policy_config"] == {}
    assert sim.start_policy_calls == []


def test_start_routes_to_start_policy_async() -> None:
    """``start`` (async) hits ``start_policy``; ``execute`` hits ``run_policy``."""
    sim = _FakeSim(robots=["so100"])
    m = Mesh(sim, peer_id="sim-a")
    out = m._dispatch(
        {
            "action": "start",
            "instruction": "wave",
            "policy_provider": "mock",
        }
    )
    assert out["status"] == "success"
    assert len(sim.start_policy_calls) == 1
    assert sim.run_policy_calls == []


def test_execute_forwards_well_known_kwargs_via_policy_config() -> None:
    """Issue #300 well-known kwargs land in ``policy_config``, not silently dropped.

    See AGENTS.md > Public API Hygiene: "Forward all advertised kwargs
    end-to-end. Silent drops are bugs masquerading as features."
    """
    sim = _FakeSim(robots=["so100"])
    m = Mesh(sim, peer_id="sim-a")

    target_pose = [0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
    target_joints = {"joint_0": 0.5, "joint_1": -0.2}
    world_update = {"obstacles": [{"name": "cube", "pose": [0.5, 0.0, 0.05]}]}

    m._dispatch(
        {
            "action": "execute",
            "instruction": "reach",
            "policy_provider": "curobo",
            "target_pose": target_pose,
            "target_joints": target_joints,
            "world_update": world_update,
        }
    )
    args, kwargs = sim.run_policy_calls[0]
    pc = kwargs["policy_config"]
    assert pc["target_pose"] == target_pose
    assert pc["target_joints"] == target_joints
    assert pc["world_update"] == world_update


def test_execute_forwards_constructor_extras_via_policy_config() -> None:
    """Existing constructor-style extras (model_path, server_address, ...) also flow.

    Confirms we did not regress the existing dispatch contract when adding
    the issue #300 kwargs branch.
    """
    sim = _FakeSim(robots=["so100"])
    m = Mesh(sim, peer_id="sim-a")
    m._dispatch(
        {
            "action": "execute",
            "instruction": "task",
            "policy_provider": "groot",
            "model_path": "nvidia/GR00T-N1.5",
            "server_address": "127.0.0.1:5555",
            "policy_type": "groot",
            "pretrained_name_or_path": "nvidia/GR00T-N1.5",
        }
    )
    pc = sim.run_policy_calls[0][1]["policy_config"]
    assert pc["model_path"] == "nvidia/GR00T-N1.5"
    assert pc["server_address"] == "127.0.0.1:5555"
    assert pc["policy_type"] == "groot"
    assert pc["pretrained_name_or_path"] == "nvidia/GR00T-N1.5"


def test_execute_requires_robot_name_when_multiple_robots() -> None:
    """Ambiguous targets must be explicit — silent default to first robot is forbidden."""
    sim = _FakeSim(robots=["arm_left", "arm_right"])
    m = Mesh(sim, peer_id="sim-a")
    out = m._dispatch(
        {
            "action": "execute",
            "instruction": "reach",
            "policy_provider": "mock",
        }
    )
    assert "error" in out
    assert "robot_name" in out["error"]
    # Sim was not driven.
    assert sim.run_policy_calls == []


def test_execute_with_robot_name_disambiguates() -> None:
    """Explicit ``robot_name`` picks the target arm in a multi-robot sim."""
    sim = _FakeSim(robots=["arm_left", "arm_right"])
    m = Mesh(sim, peer_id="sim-a")
    m._dispatch(
        {
            "action": "execute",
            "instruction": "reach",
            "policy_provider": "mock",
            "robot_name": "arm_right",
        }
    )
    assert len(sim.run_policy_calls) == 1
    assert sim.run_policy_calls[0][0] == ("arm_right",)


def test_execute_rejects_unknown_robot_name() -> None:
    """Wrong robot_name does not silently fall through to the first robot."""
    sim = _FakeSim(robots=["so100"])
    m = Mesh(sim, peer_id="sim-a")
    out = m._dispatch(
        {
            "action": "execute",
            "instruction": "reach",
            "policy_provider": "mock",
            "robot_name": "ghost",
        }
    )
    assert "error" in out
    assert "ghost" in out["error"]
    assert sim.run_policy_calls == []


def test_execute_returns_error_when_world_uninitialised() -> None:
    """Sim with no world is a hard error, not a silent no-op."""
    sim = _FakeSim(robots=["so100"])
    sim._world = None  # type: ignore[assignment]
    m = Mesh(sim, peer_id="sim-a")
    out = m._dispatch(
        {
            "action": "execute",
            "instruction": "reach",
            "policy_provider": "mock",
        }
    )
    assert "error" in out
    assert "world" in out["error"].lower()


def test_execute_returns_error_when_no_robots_in_world() -> None:
    """Sim with a world but zero robots cannot service tell()."""
    sim = _FakeSim(robots=[])
    m = Mesh(sim, peer_id="sim-a")
    out = m._dispatch(
        {
            "action": "execute",
            "instruction": "reach",
            "policy_provider": "mock",
        }
    )
    assert "error" in out
    assert "no robots" in out["error"].lower()


def test_execute_forwards_optional_run_kwargs() -> None:
    """``control_frequency`` / ``action_horizon`` / ``fast_mode`` / ``n_steps`` reach run_policy."""
    sim = _FakeSim(robots=["so100"])
    m = Mesh(sim, peer_id="sim-a")
    m._dispatch(
        {
            "action": "execute",
            "instruction": "wave",
            "policy_provider": "mock",
            "control_frequency": 30.0,
            "action_horizon": 4,
            "fast_mode": True,
            "n_steps": 100,
        }
    )
    kwargs = sim.run_policy_calls[0][1]
    assert kwargs["control_frequency"] == 30.0
    assert kwargs["action_horizon"] == 4
    assert kwargs["fast_mode"] is True
    assert kwargs["n_steps"] == 100


def test_hardware_path_unchanged_when_run_policy_absent() -> None:
    """A peer without ``run_policy`` / ``_world`` still hits the HardwareRobot branch.

    Regression guard: the sim branch must be additive — existing
    HardwareRobot peers see no behaviour change.
    """

    class _FakeHardware:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def _execute_task_sync(
            self,
            instruction: str,
            policy_provider: str,
            policy_port: Any,
            policy_host: str,
            duration: float,
            **kw: Any,
        ) -> dict[str, Any]:
            self.calls.append(
                (
                    "execute",
                    {
                        "instruction": instruction,
                        "policy_provider": policy_provider,
                        "duration": duration,
                    },
                )
            )
            return {"executed": instruction}

    hw = _FakeHardware()
    m = Mesh(hw, peer_id="hw-a")
    out = m._dispatch(
        {
            "action": "execute",
            "instruction": "go",
            "policy_provider": "mock",
            # Sim-only kwargs that should be inert on the hardware path.
            "target_pose": [0.0] * 7,
            "robot_name": "ignored",
        }
    )
    assert out == {"executed": "go"}
    assert len(hw.calls) == 1
