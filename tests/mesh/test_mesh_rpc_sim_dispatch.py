"""Tests for ``Mesh._dispatch`` against sim peers (issue #303).

The HardwareRobot dispatch path is covered by ``test_mesh_rpc.py``. This
module pins the sim-peer branch added by issue #303: ``tell()`` against a
peer whose ``robot`` is a ``Simulation`` (or any ``SimEngine``-shaped
object) routes ``execute`` -> ``run_policy`` and ``start`` -> ``start_policy``
with the issue #300 well-known goal payload (``target_pose`` /
``target_joints`` / ``target_velocity`` / ``world_update``) forwarded into
``policy_kwargs`` - the
runner parameter that reaches ``get_actions(obs, instruction, **kwargs)``,
which is where every provider reads that goal. Constructor extras
(``model_path`` / ``server_address`` / ...) keep travelling in
``policy_config``, which is expanded into the Policy constructor.

Tests are 100% mocked - no MuJoCo / Isaac install required. A
``_FakeSim`` exposes the SimEngine surface duck-typed minimally to what
``_dispatch_sim_policy`` needs.
"""

from __future__ import annotations

import inspect
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
        # truthy - matches MuJoCoSimEngine's "_world is None" gate.
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
    # dict for both sinks so the receiving sim sees a stable type.
    assert kwargs["policy_config"] == {}
    assert kwargs["policy_kwargs"] == {}
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


def test_execute_forwards_the_goal_payload_via_policy_kwargs() -> None:
    """The issue #300 goal lands in the sink the providers read it from.

    Every provider reads the goal inside ``get_actions(**kwargs)``, which the
    runner fills from ``policy_kwargs``; no provider names a goal key on its
    constructor, so routing the goal through ``policy_config`` hands it to a
    forward-compatibility absorber that discards it. The caller then gets the
    planner's "requires at least one of target_pose=... / target_joints=..."
    refusal for a request that carried exactly that.

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
    goal = kwargs["policy_kwargs"]
    assert goal["target_pose"] == target_pose
    assert goal["target_joints"] == target_joints
    assert goal["world_update"] == world_update
    # The constructor sink must not also receive it: create_policy() expands
    # policy_config into the Policy constructor, where a goal key is an
    # unknown kwarg the provider is required to ignore.
    assert kwargs["policy_config"] == {}


def test_start_forwards_the_goal_payload_via_policy_kwargs() -> None:
    """The async ``start`` branch carries the goal, not only ``execute``.

    ``start_policy`` takes the same ``policy_kwargs`` parameter as
    ``run_policy``, so a fire-and-forget ``tell()`` must not be the one route
    that loses the goal.
    """
    sim = _FakeSim(robots=["so100"])
    m = Mesh(sim, peer_id="sim-a")

    target_pose = [0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
    m._dispatch(
        {
            "action": "start",
            "instruction": "reach",
            "policy_provider": "curobo",
            "target_pose": target_pose,
        }
    )
    kwargs = sim.start_policy_calls[0][1]
    assert kwargs["policy_kwargs"]["target_pose"] == target_pose
    assert kwargs["policy_config"] == {}


def test_planner_providers_read_the_goal_per_call_not_at_construction() -> None:
    """Pin the invariant that makes ``policy_kwargs`` the goal's only sink.

    The dispatcher has two sinks and no way to ask a provider which one it
    reads, so the choice is only correct while every provider that consumes
    the issue #300 goal consumes it per call. A provider that grew a
    ``target_pose`` constructor parameter would make that choice ambiguous
    and this test says so before the dispatcher silently picks wrong.

    ``tests/simulation/test_policy_kwargs_forwarding.py`` pins the other half
    of the chain: the runner forwards ``policy_kwargs`` verbatim to every
    ``get_actions`` call.
    """
    from strands_robots.policies.curobo.policy import CuroboPolicy
    from strands_robots.policies.moveit2.policy import MoveIt2Policy
    from strands_robots.policies.wbc.policy import WBCPolicy

    for policy_class in (CuroboPolicy, MoveIt2Policy):
        constructor = inspect.signature(policy_class.__init__).parameters
        named_at_construction = [key for key in Mesh._SIM_WELL_KNOWN_POLICY_KWARGS if key in constructor]
        assert not named_at_construction, (
            f"{policy_class.__name__}.__init__ names {named_at_construction}; the mesh dispatcher "
            "sends the goal to get_actions, so a constructor parameter of the same name would be "
            "filled from a different payload"
        )

    for goal_reader in (CuroboPolicy, MoveIt2Policy, WBCPolicy):
        per_call = inspect.signature(goal_reader.get_actions).parameters
        assert any(p.kind is p.VAR_KEYWORD for p in per_call.values()), (
            f"{goal_reader.__name__}.get_actions takes no **kwargs, so it cannot receive the goal"
        )

    # WBC is the one provider that names a goal key in BOTH places, so
    # "no constructor parameter" is not the invariant that covers it. What
    # makes ``policy_kwargs`` the right sink there is precedence: the
    # constructor value is a documented STATIC default and a per-call kwarg
    # overrides it. Were that reversed, a mesh caller telling a walking peer
    # a new direction would be answered with the one it was built with.
    assert "target_velocity" in inspect.signature(WBCPolicy.__init__).parameters
    policy = WBCPolicy(allow_missing_models=True, target_velocity=[0.1, 0.0, 0.0])
    _, built_with = policy._resolve_command({})
    _, told_over_the_mesh = policy._resolve_command({"target_velocity": [0.9, 0.0, 0.0]})
    assert list(built_with) == [0.1, 0.0, 0.0]
    assert list(told_over_the_mesh) == [0.9, 0.0, 0.0], (
        "a per-call target_velocity must override the constructor default, or the goal the mesh "
        "forwards is discarded in favour of the one the peer was built with"
    )


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
    kwargs = sim.run_policy_calls[0][1]
    pc = kwargs["policy_config"]
    assert pc["model_path"] == "nvidia/GR00T-N1.5"
    assert pc["server_address"] == "127.0.0.1:5555"
    assert pc["policy_type"] == "groot"
    assert pc["pretrained_name_or_path"] == "nvidia/GR00T-N1.5"
    # A constructor extra is not a per-call goal: it must not be re-sent to
    # get_actions on every tick.
    assert kwargs["policy_kwargs"] == {}


def test_execute_requires_robot_name_when_multiple_robots() -> None:
    """Ambiguous targets must be explicit - silent default to first robot is forbidden."""
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

    Regression guard: the sim branch must be additive - existing
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


# The documented goal set, graded against both layers that carry it
def _documented_goal_keys() -> list[str]:
    """The #300 goal keys ``SimEngine.run_policy`` documents, read from its docstring.

    Derived rather than listed so a key added to that contract is graded on
    arrival. ``run_policy``'s ``policy_kwargs`` entry is the surface that
    offers the mesh path as the analogue of the local call, which is what
    makes it the authority on which keys the mesh has to carry.
    """
    import ast
    import re
    from pathlib import Path

    import strands_robots.simulation.base as base_module

    source = Path(base_module.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "run_policy":
            doc = ast.get_docstring(node) or ""
            break
    else:  # pragma: no cover - run_policy is defined on the ABC
        raise AssertionError("SimEngine.run_policy not found in strands_robots.simulation.base")

    start = doc.index("policy_kwargs:")
    # The entry runs to the next same-indent Args label; ``n_episodes`` is
    # the one that follows it. Read to there rather than to a blank line,
    # because the entry itself wraps across several.
    entry = " ".join(doc[start : doc.index("n_episodes:", start)].split())
    return sorted({t for t in re.findall(r"``(\w+)``", entry) if t.startswith("target_") or t == "world_update"})


def test_the_documented_goal_set_is_not_empty() -> None:
    """Non-vacuity: a docstring reflow that hides the entry must not read as clean."""
    keys = _documented_goal_keys()
    assert len(keys) >= 3, f"only {keys} parsed out of run_policy's policy_kwargs entry"
    assert "target_velocity" in keys


def test_the_wire_admits_every_documented_goal_key() -> None:
    """A key the wire validator omits is dropped, not refused.

    ``validate_command`` builds its output key by key from an allowlist, so
    an unlisted field never reaches the dispatcher and nothing says so.
    """
    from strands_robots.mesh import security

    sample: dict[str, Any] = {
        "target_pose": [0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
        "target_joints": {"joint_0": 0.5},
        "target_velocity": [0.5, 0.0, 0.2],
        "world_update": {"obstacles": []},
    }
    dropped = []
    for key in _documented_goal_keys():
        assert key in sample, f"no probe value for {key}; extend `sample`"
        out = security.validate_command(
            {"action": "execute", "instruction": "go", "policy_provider": "mock", key: sample[key]}
        )
        if key not in out:
            dropped.append(key)
    assert not dropped, f"the wire validator drops documented #300 goal keys: {dropped}"


def test_the_dispatcher_forwards_every_documented_goal_key() -> None:
    """A key the wire admits and this tuple omits is dropped one layer later."""
    missing = [key for key in _documented_goal_keys() if key not in Mesh._SIM_WELL_KNOWN_POLICY_KWARGS]
    assert not missing, (
        f"_SIM_WELL_KNOWN_POLICY_KWARGS omits documented #300 goal keys {missing}; "
        f"it carries {list(Mesh._SIM_WELL_KNOWN_POLICY_KWARGS)}"
    )


def test_the_locomotion_goal_reaches_policy_kwargs() -> None:
    """Telling a walking peer where to walk arrives as a per-call goal.

    ``target_velocity`` is the whole locomotion command for WBC / wbc_gait,
    and those providers are reachable over the mesh because the policy
    provider allowlist is derived from the registry. Without this the peer
    ran the rollout with no goal and reported success.
    """
    sim = _FakeSim(robots=["unitree_g1"])
    m = Mesh(sim, peer_id="sim-locomotion")
    out = m._dispatch(
        {
            "action": "execute",
            "instruction": "walk forward",
            "policy_provider": "wbc",
            "target_velocity": [0.5, 0.0, 0.2],
        }
    )
    assert out["status"] == "success"
    _, kwargs = sim.run_policy_calls[0]
    assert kwargs["policy_kwargs"]["target_velocity"] == [0.5, 0.0, 0.2]
    # The goal is a per-call kwarg, never a constructor kwarg.
    assert "target_velocity" not in kwargs.get("policy_config", {})


def test_an_undocumented_goal_shaped_key_is_still_dropped() -> None:
    """The allowlist is not widened generally - only the documented set travels."""
    from strands_robots.mesh import security

    out = security.validate_command(
        {
            "action": "execute",
            "instruction": "go",
            "policy_provider": "mock",
            "target_acceleration": [1.0, 0.0, 0.0],
        }
    )
    assert "target_acceleration" not in out

    sim = _FakeSim(robots=["unitree_g1"])
    m = Mesh(sim, peer_id="sim-unknown-goal")
    m._dispatch({**out, "target_acceleration": [1.0, 0.0, 0.0]})
    _, kwargs = sim.run_policy_calls[0]
    assert "target_acceleration" not in kwargs["policy_kwargs"]
