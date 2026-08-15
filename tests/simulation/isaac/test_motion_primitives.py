"""Joint-space motion primitives on the Isaac backend - ``set_gripper`` / ``rotate_wrist``.

The Isaac adapter (#2154, child of the parity epic #2123) shares its
backend-neutral half with the MuJoCo reference implementation through
:class:`~strands_robots.simulation.motion_primitives_base.MotionPrimitivesCore`,
so the contracts pinned here are cross-backend claims:

  * validation rejections come from the shared core, so their wording is
    byte-identical to MuJoCo's by construction - pinned by comparing the
    Isaac surface's envelope against the core's own output;
  * gripper resolution is registry-metadata-first with the shared name-hint
    fallback, and stale/malformed metadata is a loud error, never a silent
    heuristic fallback (the so101 jaw misclassification, GH #1658/#1661);
  * the drive loops carry the same success / timeout / abort envelope shape.

These tests deliberately do NOT require NVIDIA Isaac Sim (the pattern of
``tests/simulation/isaac/test_isaac_backend.py`` /
``test_urdf_joint_name_demangle.py``): the articulation, the world and the
``ArticulationAction`` type are all faked, so resolution, convergence,
timeout and abort are all exercised on any host. Real-GPU integration
coverage is the tests child of #2123 and lives in
``tests_integ/simulation/test_isaac_motion_primitives_gpu.py``.

Parity scope (#2156): this file mirrors
``tests/simulation/mujoco/test_motion_primitives.py`` case-for-case where
the behavior is backend-neutral. MuJoCo cases that pin a MuJoCo-only
mechanism are deliberately NOT ported emptily (test behavior, not
implementation):

  * ``TestDiscoverySurface``'s ``tool_spec`` cases and the whole
    ``TestDispatchForwarding`` class pin the MuJoCo ``AgentTool`` router
    (``tool_spec`` / ``_dispatch_action``); ``IsaacSimulation`` exposes no
    tool-dispatch surface, so there is nothing to forward through.
  * The MuJoCo servo pins that read ``data.ctrl`` interplay or count
    ``_SUBSTEPS_PER_TICK`` physics substeps have no Isaac counterpart: an
    Isaac control tick IS one ``world.step`` (see
    ``IsaacMotionPrimitivesMixin._primitive_tick``), which
    ``test_target_reasserted_every_tick`` pins from the behavior side.
"""

from __future__ import annotations

import concurrent.futures
import math
import sys
import types
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.simulation import IsaacSimulation, _RobotState
from strands_robots.simulation.motion_primitives_base import MotionPrimitivesCore, _err

# ---------------------------------------------------------------------------
# Fakes: articulation with PD-target servo semantics, world whose step()
# advances the articulation toward its targets, and the one isaacsim type the
# adapter lazily imports.
# ---------------------------------------------------------------------------


class _FakeArticulationAction:
    """Stand-in for ``isaacsim.core.utils.types.ArticulationAction``."""

    def __init__(self, joint_positions=None, joint_indices=None):
        self.joint_positions = joint_positions
        self.joint_indices = joint_indices


class _FakeArticulation:
    """Articulation with joint limits and a per-step PD-servo model.

    ``apply_action`` records the commanded targets; each ``advance()`` (wired
    to the fake world's ``step``) moves every targeted DOF toward its target
    by ``servo_rate`` of the remaining distance. ``servo_rate=0.0`` models an
    arm that never converges (the timeout path).
    """

    def __init__(
        self,
        joint_names: list[str],
        limits: list[tuple[float, float] | None],
        positions: list[float] | None = None,
        servo_rate: float = 0.5,
    ):
        n = len(joint_names)
        assert len(limits) == n
        self.positions = np.array(positions if positions is not None else [0.0] * n, dtype=np.float64)
        self.servo_rate = servo_rate
        self.applied: list[Any] = []
        self._targets: dict[int, float] = {}
        self.dof_properties = np.zeros(n, dtype=[("hasLimits", "?"), ("lower", "f8"), ("upper", "f8")])
        for i, span in enumerate(limits):
            if span is None:
                self.dof_properties["hasLimits"][i] = False
            else:
                self.dof_properties["hasLimits"][i] = True
                self.dof_properties["lower"][i] = span[0]
                self.dof_properties["upper"][i] = span[1]

    def get_joint_positions(self):
        return self.positions.copy()

    def apply_action(self, action) -> None:
        self.applied.append(action)
        for idx, value in zip(np.asarray(action.joint_indices), np.asarray(action.joint_positions)):
            self._targets[int(idx)] = float(value)

    def advance(self) -> None:
        for idx, target in self._targets.items():
            self.positions[idx] += self.servo_rate * (target - self.positions[idx])


class _FakeWorld:
    """World whose ``step`` drives the articulation servo and an optional hook."""

    def __init__(self, articulation: _FakeArticulation, on_step=None):
        self.articulation = articulation
        self.on_step = on_step
        self.steps = 0

    def step(self, render: bool = False) -> None:  # noqa: ARG002 - signature parity
        self.steps += 1
        if self.on_step is not None:
            self.on_step()
        self.articulation.advance()


@pytest.fixture(autouse=True)
def fake_articulation_action(monkeypatch):
    """Provide the ``isaacsim.core.utils.types`` module the adapter imports."""
    names = ("isaacsim", "isaacsim.core", "isaacsim.core.utils", "isaacsim.core.utils.types")
    mods = {}
    for name in names:
        mod = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
        mods[name] = mod
    mods["isaacsim.core.utils.types"].ArticulationAction = _FakeArticulationAction
    mods["isaacsim"].core = mods["isaacsim.core"]
    mods["isaacsim.core"].utils = mods["isaacsim.core.utils"]
    mods["isaacsim.core.utils"].types = mods["isaacsim.core.utils.types"]
    return mods


# A generic hobby-arm vocabulary the name heuristics resolve against.
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_roll", "jaw"]
ARM_LIMITS: list[tuple[float, float] | None] = [
    (-3.1, 3.1),
    (-1.8, 1.8),
    (-2.4, 2.4),
    (-1.7, 1.7),
    (-0.2, 1.5),
]


def _make_sim(
    joint_names: list[str] = ARM_JOINTS,
    limits: list[tuple[float, float] | None] = ARM_LIMITS,
    robot_name: str = "arm",
    data_config: str | None = None,
    positions: list[float] | None = None,
    servo_rate: float = 0.5,
    on_step=None,
) -> tuple[IsaacSimulation, _FakeArticulation]:
    sim = IsaacSimulation()
    art = _FakeArticulation(joint_names, limits, positions=positions, servo_rate=servo_rate)
    sim._world = _FakeWorld(art, on_step=on_step)
    sim._world_created = True
    sim._robots[robot_name] = _RobotState(
        name=robot_name,
        prim_path=f"/World/Robots/{robot_name}",
        joint_names=list(joint_names),
        articulation=art,
        data_config=data_config,
    )
    return sim, art


def _json_block(result: dict) -> dict:
    blocks = [c["json"] for c in result["content"] if "json" in c]
    assert blocks, result
    return blocks[0]


# ---------------------------------------------------------------------------
# Validation: the rejections are the shared core's, byte-identical to MuJoCo.
# ---------------------------------------------------------------------------


class TestValidationReusesTheCore:
    """A rejected parameter answers with the core's own envelope."""

    def test_invalid_state_matches_core_wording(self):
        sim, _ = _make_sim()
        result = sim.set_gripper(robot_name="arm", state="ajar")
        _, expected = MotionPrimitivesCore()._validate_set_gripper_args("ajar", 12)
        assert result == expected
        assert '"open" or "close"' in result["content"][0]["text"]

    def test_missing_state_rejected(self):
        sim, _ = _make_sim()
        assert sim.set_gripper(robot_name="arm")["status"] == "error"

    @pytest.mark.parametrize("steps", [0, -1, 10_001, 2.7, True, None, "12"])
    def test_bad_steps_matches_core_wording(self, steps):
        sim, art = _make_sim()
        result = sim.set_gripper(robot_name="arm", state="open", steps=steps)
        _, expected = MotionPrimitivesCore()._validate_set_gripper_args("open", steps)
        assert result == expected
        assert result["status"] == "error"
        assert art.applied == []  # refused before any write

    def test_missing_target_yaw_matches_core_wording(self):
        sim, _ = _make_sim()
        result = sim.rotate_wrist(robot_name="arm")
        _, _, expected = MotionPrimitivesCore()._validate_rotate_wrist_args(None, 0.02, 200)
        assert result == expected
        assert "target_yaw" in result["content"][0]["text"]

    @pytest.mark.parametrize("tol", [0.0, -0.1, math.nan, math.inf, True, "0.05"])
    def test_bad_tol_matches_core_wording(self, tol):
        sim, art = _make_sim()
        result = sim.rotate_wrist(robot_name="arm", target_yaw=0.3, tol=tol)
        _, _, expected = MotionPrimitivesCore()._validate_rotate_wrist_args(0.3, tol, 200)
        assert result == expected
        assert result["status"] == "error"
        assert art.applied == []


class TestGuards:
    """Backend-owned world / robot resolution, this backend's wording."""

    def test_no_world_errors(self):
        sim = IsaacSimulation()
        result = sim.set_gripper(state="open")
        assert result == _err("No world created.")
        assert sim.rotate_wrist(target_yaw=0.3) == _err("No world created.")

    def test_unknown_robot_errors(self):
        sim, _ = _make_sim()
        result = sim.set_gripper(robot_name="nope", state="open")
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_single_robot_auto_resolves(self):
        sim, _ = _make_sim()
        assert sim.set_gripper(state="close")["status"] == "success"

    def test_multiple_robots_require_a_name(self):
        sim, _ = _make_sim()
        art2 = _FakeArticulation(["a", "b_gripper"], [(-1, 1), (0, 1)])
        sim._robots["other"] = _RobotState(
            name="other", prim_path="/World/Robots/other", joint_names=["a", "b_gripper"], articulation=art2
        )
        result = sim.rotate_wrist(target_yaw=0.1)
        assert result["status"] == "error"
        assert "Multiple robots" in result["content"][0]["text"]

    def test_uninitialized_articulation_errors(self):
        sim, _ = _make_sim()
        sim._robots["arm"].articulation = None
        result = sim.set_gripper(robot_name="arm", state="open")
        assert result["status"] == "error"
        assert "not initialized" in result["content"][0]["text"]

    def test_allowed_after_policy_stopped(self):
        # The flag every Isaac policy-driving loop clears on exit is the whole
        # gate: once it drops, the primitive proceeds (MuJoCo parity).
        sim, _ = _make_sim()
        sim._robots["arm"].policy_running = True
        sim._robots["arm"].policy_running = False
        result = sim.set_gripper(robot_name="arm", state="open", steps=2)
        assert result["status"] == "success", result

    def test_worker_thread_without_pump_is_a_structured_error(self):
        # Isaac writes must happen on the Kit-owning thread. Called off it
        # with no pump, the primitive answers with the recipe instead of the
        # raising marshal step()/reset() use - never-raises contract.
        sim, art = _make_sim()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(sim.set_gripper, robot_name="arm", state="open").result()
        assert result["status"] == "error"
        assert "run_pump_forever" in result["content"][0]["text"]
        assert art.applied == []

    # -- no-running-policy: the shared preamble's guard, every primitive ----

    @pytest.mark.parametrize(
        ("primitive", "kwargs"),
        [("set_gripper", {"state": "open"}), ("rotate_wrist", {"target_yaw": 0.4})],
    )
    def test_refused_while_a_policy_runs_on_the_robot(self, primitive, kwargs):
        # A primitive and the policy loop write the same articulation's PD
        # targets, so the primitive refuses rather than interleaving. The
        # refusal names the primitive the caller called and the reason.
        sim, art = _make_sim()
        sim._robots["arm"].policy_running = True
        result = getattr(sim, primitive)(robot_name="arm", **kwargs)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert f"Cannot '{primitive}' on 'arm' while its policy is running" in text
        assert "race on the articulation's PD targets" in text
        # Nothing reached the articulation: the guard runs before the drive loop.
        assert art.applied == []
        assert sim._world.steps == 0

    def test_the_policy_refusal_is_one_rule_across_the_primitives(self):
        # The guard lives in the shared preamble, so the three primitives cannot
        # drift apart on the wording: the refusals differ only in the action
        # name. Driving the preamble directly keeps this independent of what
        # each primitive needs to set up after it.
        sim, _ = _make_sim()
        sim._robots["arm"].policy_running = True
        texts = {}
        for action in ("move_to", "set_gripper", "rotate_wrist"):
            name, robot, error = sim._primitive_resolve_robot(action, "arm")
            assert (name, robot) == (None, None)
            assert error is not None
            texts[action] = error["content"][0]["text"]
        normalized = {t.replace(f"'{a}'", "'<action>'", 1) for a, t in texts.items()}
        assert len(normalized) == 1, texts

    def test_a_policy_on_another_robot_does_not_refuse(self):
        # Per-robot scope: Isaac policy loops set the flag per robot and write
        # disjoint articulations, so a rollout elsewhere must not block this arm.
        sim, art = _make_sim()
        other_art = _FakeArticulation(["a", "b_gripper"], [(-1.0, 1.0), (0.0, 1.0)])
        sim._robots["other"] = _RobotState(
            name="other",
            prim_path="/World/Robots/other",
            joint_names=["a", "b_gripper"],
            articulation=other_art,
        )
        sim._robots["other"].policy_running = True
        assert sim.set_gripper(robot_name="arm", state="close")["status"] == "success"
        assert art.applied != []
        assert other_art.applied == []

    def test_an_uninitialized_articulation_reports_its_own_reason(self):
        # Guard order: the policy check is the last one in the preamble, so a
        # robot that is not initialized still reports that rather than being
        # masked by a stale policy flag.
        sim, _ = _make_sim()
        sim._robots["arm"].articulation = None
        sim._robots["arm"].policy_running = True
        result = sim.set_gripper(robot_name="arm", state="open")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "not initialized" in text
        assert "policy is running" not in text


# ---------------------------------------------------------------------------
# set_gripper: resolution, range-end mapping, payload.
# ---------------------------------------------------------------------------


class TestSetGripper:
    """Open/close set-point semantics (LOW end = closed, HIGH end = open)."""

    def test_close_drives_toward_low_end(self):
        sim, art = _make_sim(positions=[0, 0, 0, 0, 1.0], servo_rate=0.9)
        result = sim.set_gripper(robot_name="arm", state="close", steps=40)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["targets"]["jaw"] == pytest.approx(-0.2)
        assert payload["setpoint_sources"]["jaw"] == "articulation dof limits"
        assert art.positions[4] < -0.1  # traveled toward the low (closed) end
        assert payload["gripper_joint_positions"]["jaw"] == pytest.approx(art.positions[4])

    def test_open_drives_toward_high_end(self):
        sim, art = _make_sim(positions=[0, 0, 0, 0, -0.2], servo_rate=0.9)
        result = sim.set_gripper(robot_name="arm", state="open", steps=40)
        assert result["status"] == "success", result
        assert _json_block(result)["targets"]["jaw"] == pytest.approx(1.5)
        assert art.positions[4] > 1.0

    def test_target_reasserted_every_tick(self):
        sim, art = _make_sim()
        result = sim.set_gripper(robot_name="arm", state="close", steps=7)
        assert result["status"] == "success", result
        assert len(art.applied) == 7
        # Only the gripper DOF is commanded; the arm stays on its own targets.
        for action in art.applied:
            assert list(np.asarray(action.joint_indices)) == [4]

    def test_unresolvable_gripper_lists_joint_names(self):
        sim, _ = _make_sim(joint_names=["j1", "j2", "j3"], limits=[(-1, 1)] * 3)
        result = sim.set_gripper(robot_name="arm", state="open")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "could not resolve a gripper joint" in text
        assert "['j1', 'j2', 'j3']" in text
        assert "send_action" in text

    def test_namespaced_joint_names_resolve_via_stripping(self):
        sim, _ = _make_sim(
            joint_names=["arm/shoulder", "arm/jaw"],
            limits=[(-1.0, 1.0), (-0.2, 1.5)],
        )
        result = sim.set_gripper(robot_name="arm", state="open", steps=5)
        assert result["status"] == "success", result
        assert _json_block(result)["actuators"] == ["jaw"]

    def test_unusable_limits_are_a_loud_error(self):
        sim, art = _make_sim(limits=[(-3.1, 3.1), (-1.8, 1.8), (-2.4, 2.4), (-1.7, 1.7), None])
        result = sim.set_gripper(robot_name="arm", state="open")
        assert result["status"] == "error"
        assert "no usable open/close set-points" in result["content"][0]["text"]
        assert art.applied == []


class TestGripperRegistryMetadata:
    """Registry gripper metadata beats the name heuristic (GH #1658)."""

    def _patch_registry(self, monkeypatch, meta):
        monkeypatch.setattr(
            "strands_robots.simulation.isaac.motion_primitives.get_robot",
            lambda name: {"gripper": meta} if name == "so101" else None,
        )

    def test_metadata_resolves_a_hint_less_jaw(self, monkeypatch):
        # so101-style numeric vocabulary: no name hint matches, only the
        # registry metadata can say DOF "6" is the jaw.
        self._patch_registry(monkeypatch, {"actuators": ["6"], "closed": "low", "open": "high"})
        sim, art = _make_sim(
            joint_names=["1", "2", "3", "4", "5", "6"],
            limits=[(-1.9, 1.9)] * 5 + [(-0.2, 1.5)],
            data_config="so101",
        )
        result = sim.set_gripper(robot_name="arm", state="open", steps=5)
        assert result["status"] == "success", result
        assert _json_block(result)["actuators"] == ["6"]
        assert list(np.asarray(art.applied[0].joint_indices)) == [5]

    def test_set_gripper_commands_only_metadata_actuators(self, monkeypatch):
        # The heuristic's inverse failure mode (GH #1658) is silent: an ARM
        # joint whose name happens to contain a hint (here the base pan,
        # 'finger_camera_pan') would be classified as a gripper drive and
        # slewed by set_gripper. Metadata is authoritative: ONLY the named
        # joint is commanded.
        self._patch_registry(monkeypatch, {"actuators": ["Jaw"], "closed": "low", "open": "high"})
        sim, art = _make_sim(
            joint_names=["finger_camera_pan", "shoulder_lift", "elbow", "wrist_roll", "Jaw"],
            data_config="so101",
        )
        result = sim.set_gripper(robot_name="arm", state="close", steps=5)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["actuators"] == ["Jaw"], payload
        assert "finger_camera_pan" not in payload["targets"]
        commanded = {int(i) for a in art.applied for i in np.asarray(a.joint_indices)}
        assert commanded == {4}, "set_gripper commanded a hint-colliding arm DOF"

    def test_alias_data_config_resolves_metadata(self):
        # data_config aliases resolve to the canonical registry entry through
        # the REAL registry (no patching): so100_dualcam -> so100, whose
        # gripper metadata names 'Jaw'. The hint-colliding pan joint proves
        # the metadata (not the heuristic) did the resolution - the heuristic
        # would have picked both.
        sim, _ = _make_sim(
            joint_names=["finger_camera_pan", "shoulder_lift", "elbow", "wrist_roll", "Jaw"],
            data_config="so100_dualcam",
        )
        result = sim.set_gripper(robot_name="arm", state="open", steps=5)
        assert result["status"] == "success", result
        assert _json_block(result)["actuators"] == ["Jaw"]

    def test_inverted_open_close_convention_honored(self, monkeypatch):
        self._patch_registry(monkeypatch, {"actuators": ["6"], "closed": "high", "open": "low"})
        sim, _ = _make_sim(
            joint_names=["1", "2", "3", "4", "5", "6"],
            limits=[(-1.9, 1.9)] * 5 + [(-0.2, 1.5)],
            data_config="so101",
        )
        result = sim.set_gripper(robot_name="arm", state="open", steps=5)
        assert result["status"] == "success", result
        assert _json_block(result)["targets"]["6"] == pytest.approx(-0.2)  # open = LOW here

    def test_stale_metadata_is_a_loud_error_not_a_heuristic_fallback(self, monkeypatch):
        # The metadata names a joint the articulation does not have; falling
        # back to the hints would command whatever the heuristic guesses.
        # rotate_wrist shares the classification (GH #1661), so it refuses
        # identically instead of silently picking a wrist from the raw hints.
        self._patch_registry(monkeypatch, {"actuators": ["NoSuchJoint"], "closed": "low", "open": "high"})
        sim, art = _make_sim(data_config="so101")
        for action, fields in (
            ("set_gripper", {"state": "open"}),
            ("rotate_wrist", {"target_yaw": 0.3}),
        ):
            result = getattr(sim, action)(robot_name="arm", **fields)
            assert result["status"] == "error", (action, result)
            text = result["content"][0]["text"]
            assert "stale" in text, (action, text)
            assert "NoSuchJoint" in text, (action, text)
        assert art.applied == []

    def test_malformed_metadata_is_a_loud_error(self, monkeypatch):
        self._patch_registry(monkeypatch, {"actuators": [], "closed": "low", "open": "low"})
        sim, _ = _make_sim(data_config="so101")
        for action, fields in (
            ("set_gripper", {"state": "open"}),
            ("rotate_wrist", {"target_yaw": 0.3}),
        ):
            result = getattr(sim, action)(robot_name="arm", **fields)
            assert result["status"] == "error", (action, result)
            assert "malformed" in result["content"][0]["text"], (action, result)

    def test_no_data_config_still_uses_heuristic(self):
        sim, _ = _make_sim(data_config=None)
        assert sim.set_gripper(robot_name="arm", state="close")["status"] == "success"


# ---------------------------------------------------------------------------
# rotate_wrist: resolution, hold semantics, convergence / timeout / abort.
# ---------------------------------------------------------------------------


class TestRotateWrist:
    """Wrist set-point semantics on a mocked converging articulation."""

    def test_reaches_target_yaw(self):
        sim, art = _make_sim(servo_rate=0.5)
        result = sim.rotate_wrist(robot_name="arm", target_yaw=0.7)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["reached"] is True
        assert payload["wrist_joint"] == "wrist_roll"
        assert payload["final_yaw"] == pytest.approx(0.7, abs=0.03)
        assert art.positions[3] == pytest.approx(0.7, abs=0.03)

    def test_holds_other_joints_at_their_current_positions(self):
        start = [0.4, -0.3, 0.9, 0.0, 0.2]
        sim, art = _make_sim(positions=list(start))
        result = sim.rotate_wrist(robot_name="arm", target_yaw=0.5)
        assert result["status"] == "success", result
        action = art.applied[0]
        held = dict(
            zip(
                (int(i) for i in np.asarray(action.joint_indices)),
                (float(v) for v in np.asarray(action.joint_positions)),
            )
        )
        assert held[3] == pytest.approx(0.5)  # the wrist is commanded
        for dof in (0, 1, 2, 4):  # the rest (jaw included) hold current
            assert held[dof] == pytest.approx(start[dof])
        for dof in (0, 1, 2, 4):
            # Targets ride an ArticulationAction as float32 (parity with
            # send_action), so "held" is exact only to float32 precision.
            assert art.positions[dof] == pytest.approx(start[dof], abs=1e-6)

    def test_out_of_range_target_rejected(self):
        sim, art = _make_sim()
        result = sim.rotate_wrist(robot_name="arm", target_yaw=10.0)
        assert result["status"] == "error"
        assert "outside joint 'wrist_roll' range" in result["content"][0]["text"]
        assert art.applied == []

    def test_timeout_returns_structured_error_with_residual(self):
        # servo_rate=0 models an arm that never moves: the primitive must
        # burn its budget and answer with the shared not-reached envelope.
        sim, _ = _make_sim(servo_rate=0.0)
        result = sim.rotate_wrist(robot_name="arm", target_yaw=0.7, tol=0.01, max_steps=8)
        assert result["status"] == "error"
        assert "did not reach" in result["content"][0]["text"]
        payload = _json_block(result)
        assert payload["reached"] is False
        assert payload["steps"] == 8
        assert payload["yaw_error_rad"] == pytest.approx(0.7)

    def test_jaw_is_never_selected_as_wrist_on_so101_style_model(self, monkeypatch):
        # Numeric joint names carry no wrist hint, so the fallback picks the
        # last NON-GRIPPER joint - which must be "5", not the jaw "6" the raw
        # last-joint heuristic would grab (GH #1661).
        monkeypatch.setattr(
            "strands_robots.simulation.isaac.motion_primitives.get_robot",
            lambda name: (
                {"gripper": {"actuators": ["6"], "closed": "low", "open": "high"}} if name == "so101" else None
            ),
        )
        sim, _ = _make_sim(
            joint_names=["1", "2", "3", "4", "5", "6"],
            limits=[(-1.9, 1.9)] * 6,
            data_config="so101",
        )
        result = sim.rotate_wrist(robot_name="arm", target_yaw=0.3)
        assert result["status"] == "success", result
        assert _json_block(result)["wrist_joint"] == "5"

    def test_no_metadata_fallback_is_last_non_gripper_joint(self):
        sim, _ = _make_sim(joint_names=["base", "lift", "elbow", "jaw"], limits=[(-2, 2)] * 4)
        result = sim.rotate_wrist(robot_name="arm", target_yaw=0.3)
        assert result["status"] == "success", result
        assert _json_block(result)["wrist_joint"] == "elbow"

    def test_hint_colliding_distal_joint_stays_a_wrist_candidate(self, monkeypatch):
        # GH #1661's fallback-shift failure mode: the most distal arm joint is
        # named 'finger_camera_roll'. The raw heuristic excluded it (finger
        # hint) and shifted the last-non-gripper fallback onto the elbow; with
        # registry metadata (gripper = Jaw) the roll joint stays a candidate
        # and is picked. No wrist hint matches, so the fallback path is the
        # one exercised.
        monkeypatch.setattr(
            "strands_robots.simulation.isaac.motion_primitives.get_robot",
            lambda name: (
                {"gripper": {"actuators": ["Jaw"], "closed": "low", "open": "high"}} if name == "so100" else None
            ),
        )
        sim, _ = _make_sim(
            joint_names=["shoulder_pan", "shoulder_lift", "elbow", "finger_camera_roll", "Jaw"],
            data_config="so100",
        )
        result = sim.rotate_wrist(robot_name="arm", target_yaw=0.5)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["wrist_joint"] == "finger_camera_roll", payload
        assert payload["reached"] is True

    def test_gripper_only_articulation_cannot_resolve_a_wrist(self):
        sim, _ = _make_sim(joint_names=["jaw"], limits=[(-0.2, 1.5)])
        result = sim.rotate_wrist(robot_name="arm", target_yaw=0.3)
        assert result["status"] == "error"
        assert "could not resolve a wrist joint" in result["content"][0]["text"]

    def test_unknown_limits_do_not_block_the_move(self):
        # An articulation that reports no limits for the wrist cannot range-
        # check the target, but the move itself is still legitimate.
        sim, _ = _make_sim(limits=[(-3.1, 3.1), (-1.8, 1.8), (-2.4, 2.4), None, (-0.2, 1.5)])
        result = sim.rotate_wrist(robot_name="arm", target_yaw=0.4)
        assert result["status"] == "success", result


class TestMidRunAbort:
    """The loops release the lock per tick; teardown mid-run aborts loudly."""

    def test_robot_removed_mid_run_aborts(self):
        sim_box: dict[str, IsaacSimulation] = {}

        def _remove_robot():
            sim_box["sim"]._robots.pop("arm", None)

        sim, _ = _make_sim(servo_rate=0.0, on_step=_remove_robot)
        sim_box["sim"] = sim
        result = sim.rotate_wrist(robot_name="arm", target_yaw=0.7, max_steps=50)
        assert result == _err("rotate_wrist: robot 'arm' was removed mid-run; aborting.")

    def test_world_destroyed_mid_run_aborts(self):
        sim_box: dict[str, IsaacSimulation] = {}

        def _destroy_world():
            sim_box["sim"]._world_created = False

        sim, _ = _make_sim(servo_rate=0.0, on_step=_destroy_world)
        sim_box["sim"] = sim
        result = sim.set_gripper(robot_name="arm", state="open", steps=50)
        assert result == _err("set_gripper: world was destroyed mid-run; aborting.")
        # One tick ran before the abort became observable, no more.
        assert sim._world.steps == 1

    @pytest.mark.parametrize(
        ("primitive", "kwargs"),
        [("set_gripper", {"state": "open", "steps": 50}), ("rotate_wrist", {"target_yaw": 0.7, "max_steps": 50})],
    )
    def test_a_policy_started_mid_run_aborts(self, primitive, kwargs):
        # The loops release the lock per tick, so a rollout can start under a
        # running primitive. That aborts with the policy reason rather than
        # driving on and reporting a convergence timeout for a race.
        sim_box: dict[str, IsaacSimulation] = {}

        def _start_policy():
            sim_box["sim"]._robots["arm"].policy_running = True

        sim, art = _make_sim(servo_rate=0.0, on_step=_start_policy)
        sim_box["sim"] = sim
        result = getattr(sim, primitive)(robot_name="arm", **kwargs)
        assert result == _err(f"{primitive}: a policy started on 'arm' mid-run; aborting.")
        # One tick ran before the flag became observable, no more - so the race
        # window is one control tick rather than the whole requested budget.
        assert sim._world.steps == 1
        assert len(art.applied) == 1

    def test_the_mid_run_policy_abort_is_one_rule_across_the_primitives(self):
        # Same shared-helper claim as the up-front guard: the abort check lives
        # in _primitive_abort_reason, so the three primitives report identically.
        sim, _ = _make_sim()
        sim._robots["arm"].policy_running = True
        texts = {}
        for action in ("move_to", "set_gripper", "rotate_wrist"):
            abort = sim._primitive_abort_reason(action, "arm")
            assert abort is not None
            texts[action] = abort["content"][0]["text"]
        normalized = {t.replace(f"{a}:", "<action>:", 1) for a, t in texts.items()}
        assert len(normalized) == 1, texts

    def test_a_removed_robot_reports_removal_not_a_policy(self):
        # Guard order in the abort check mirrors the preamble's: a robot that
        # disappeared reports that, even with a stale flag left behind.
        sim, _ = _make_sim()
        sim._robots["arm"].policy_running = True
        sim._robots["arm"].articulation = None
        abort = sim._primitive_abort_reason("set_gripper", "arm")
        assert abort == _err("set_gripper: robot 'arm' was removed mid-run; aborting.")


class TestDiscoverySurface:
    """describe() advertises the primitives like the MuJoCo backend does."""

    def test_describe_advertises_primitives(self):
        sim = IsaacSimulation()
        methods = sim.describe()["methods"]
        assert "set_gripper" in methods
        assert "rotate_wrist" in methods
        assert "open" in methods["set_gripper"]
        assert "target_yaw" in methods["rotate_wrist"]


class TestRecordingInterplay:
    """Primitive motion does NOT feed the dataset recorder (the #1498 bug class).

    Only the policy-rollout per-frame hook records dataset episodes
    (:meth:`~strands_robots.simulation.isaac.recording.IsaacRecordingMixin._make_run_policy_hook`);
    a primitive stepping physics directly must not add frames - a silent
    zero-frame "recording" must never masquerade as a recorded episode.
    Mirrors the MuJoCo pin; the recorder is a mock, so no dataset root is
    ever resolved (nothing touches the shared cache).
    """

    def test_primitive_does_not_feed_dataset_recorder(self):
        from unittest.mock import MagicMock

        sim, _ = _make_sim()
        recorder = MagicMock()
        sim._recording_state_dict = {"recording": True, "dataset_recorder": recorder}
        try:
            result = sim.set_gripper(robot_name="arm", state="close", steps=5)
            assert result["status"] == "success", result
            recorder.add_frame.assert_not_called()
        finally:
            sim._recording_state_dict = {}
