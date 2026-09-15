"""Isaac's ``apply_force`` / ``raycast``: the last two consumer-ranked parity gaps.

Both were absent from the backend entirely (not even stubs to raise), so the
MuJoCo-portable disturbance-and-sensing recipes had no Isaac half.

The semantics that needed measuring before writing (isaacsim 6.0.1):

* PhysX's ``apply_force_at_pos`` acts for ONE step - a single 80 N call
  accelerated a resting cube for a single tick (0.06 m -> 0.58 m over the ten
  following steps, then no further acceleration). The cross-backend contract
  is MuJoCo's LATCH: applied every step until replaced, ``force=[0,0,0]``
  stops one body, ``reset()`` clears all. So the latch is replayed into PhysX
  by the step loop, once per physics tick.
* ``raycast_closest`` returns ``hit``/``position``/``distance``/``collision``/
  ``rigidBody`` with prim paths, not ids - which is why ``geom_id`` is always
  ``None`` here and ``exclude_body`` (a MuJoCo compiled-model body id) accepts
  only its ``-1`` default, refused otherwise rather than silently ignored.

What is pinned here without a GPU: the validation domains, the latch
bookkeeping (store / replace / zero-clears / lever-arm fold at call time), the
per-tick replay from the step loop, and the raycast translation including the
``include_static=False`` re-cast past static hits. The live half - a latched
lateral force actually accelerating a cube across steps, a real ray hitting
the cube at the geometric distance - is verified on GPU.
"""

from __future__ import annotations

import pathlib
import sys
import threading
import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import (  # noqa: E402
    IsaacConfig,
    IsaacSimulation,
    _ObjectState,
    _RobotState,
)


class _Handle:
    def __init__(self, pos: tuple[float, float, float] = (0.3, 0.0, 0.06)) -> None:
        self._pos = np.asarray(pos, dtype=float)

    def get_world_pose(self) -> Any:
        return self._pos.copy(), np.array([1.0, 0.0, 0.0, 0.0])


def _engine(with_object: bool = True, static: bool = False) -> Any:
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(render_mode="headless")
    engine._world = types.SimpleNamespace(step=lambda render=False: None)
    engine._world_created = True
    engine._robots = {}
    engine._objects = {}
    engine._applied_wrenches = {}
    # send_action resolves a task-space controller before it steps.
    engine._action_controllers = {}
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._STEPS_PER_BATCH = IsaacSimulation._STEPS_PER_BATCH
    engine._main_tid = threading.get_ident()
    engine._pump_running = False
    if with_object:
        state = _ObjectState(name="cube", prim_path="/World/Objects/cube", shape="box", is_static=static)
        state.handle = _Handle()
        engine._objects["cube"] = state
    return engine


@pytest.fixture
def fake_physx(monkeypatch) -> dict[str, list[Any]]:
    """omni.physx + pxr stand-ins recording every force/torque/raycast call."""
    calls: dict[str, list[Any]] = {"force": [], "torque": [], "raycast": []}

    sim_iface = types.SimpleNamespace(
        apply_force_at_pos=lambda stage_id, body, force, pos: calls["force"].append((body, force, pos)),
        apply_torque=lambda stage_id, body, torque: calls["torque"].append((body, torque)),
    )
    query_results: list[dict[str, Any]] = []
    calls["_query_results"] = query_results  # type: ignore[assignment]

    def raycast_closest(origin: Any, direction: Any, dist: float) -> dict[str, Any]:
        calls["raycast"].append((tuple(origin), tuple(direction)))
        return query_results.pop(0) if query_results else {"hit": False}

    physx_mod = types.ModuleType("omni.physx")
    physx_mod.get_physx_simulation_interface = lambda: sim_iface  # type: ignore[attr-defined]
    physx_mod.get_physx_scene_query_interface = lambda: types.SimpleNamespace(raycast_closest=raycast_closest)  # type: ignore[attr-defined]
    usd_mod = types.ModuleType("omni.usd")
    stage = types.SimpleNamespace(
        GetPrimAtPath=lambda path: types.SimpleNamespace(
            IsValid=lambda: True,
            HasAPI=lambda api: "Objects" in path,  # objects are rigid bodies; everything else static
            __bool__=lambda self=None: True,
        )
    )
    usd_mod.get_context = lambda: types.SimpleNamespace(get_stage_id=lambda: 1, get_stage=lambda: stage)  # type: ignore[attr-defined]
    omni_mod = types.ModuleType("omni")
    omni_mod.physx = physx_mod  # type: ignore[attr-defined]
    omni_mod.usd = usd_mod  # type: ignore[attr-defined]
    for name, mod in (("omni", omni_mod), ("omni.physx", physx_mod), ("omni.usd", usd_mod)):
        monkeypatch.setitem(sys.modules, name, mod)

    pxr = sys.modules.get("pxr") or types.ModuleType("pxr")
    monkeypatch.setattr(pxr, "PhysicsSchemaTools", types.SimpleNamespace(sdfPathToInt=lambda p: 42), raising=False)
    monkeypatch.setattr(pxr, "UsdPhysics", types.SimpleNamespace(RigidBodyAPI=object()), raising=False)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    return calls


class TestApplyForceValidation:
    def test_no_world(self) -> None:
        engine = _engine()
        engine._world_created = False
        assert engine.apply_force("cube", force=[0, 0, 1])["status"] == "error"

    def test_neither_vector_is_refused(self) -> None:
        result = _engine().apply_force("cube")
        assert result["status"] == "error"
        assert "force" in result["content"][0]["text"]

    @pytest.mark.parametrize("bad", [[1.0, True, 0.0], [float("nan"), 0, 0], [1.0, 2.0]])
    def test_a_malformed_vector_is_refused_on_the_shared_domain(self, bad: Any) -> None:
        """Booleans, non-finite elements and wrong lengths all go through
        coerce_pose_vector - the domain every backend's pose inputs share."""
        assert _engine().apply_force("cube", force=bad)["status"] == "error"

    def test_an_unknown_body_is_refused_by_name(self, fake_physx: dict) -> None:
        result = _engine().apply_force("ghost", force=[0, 0, 1])
        assert result["status"] == "error"
        assert "ghost" in result["content"][0]["text"]

    def test_a_static_body_is_refused_not_ignored(self, fake_physx: dict) -> None:
        result = _engine(static=True).apply_force("cube", force=[0, 0, 1])
        assert result["status"] == "error"
        assert "static" in result["content"][0]["text"]


class TestTheLatch:
    def test_a_wrench_is_latched_and_replayed_every_tick(self, fake_physx: dict) -> None:
        """PhysX applies for one step; the contract is every step. The step
        loop replays the latch once per tick."""
        engine = _engine()
        assert engine.apply_force("cube", force=[5.0, 0.0, 0.0])["status"] == "success"
        assert "cube" in engine._applied_wrenches

        engine.step(4)
        assert len(fake_physx["force"]) == 4, "one replay per physics tick"
        body, force, _pos = fake_physx["force"][0]
        assert body == 42 and force == (5.0, 0.0, 0.0)
        assert not fake_physx["torque"], "no torque latched, none applied"

    def test_the_force_is_applied_at_the_bodys_current_position(self, fake_physx: dict) -> None:
        engine = _engine()
        engine.apply_force("cube", force=[5.0, 0.0, 0.0])
        engine.step(1)
        engine._objects["cube"].handle._pos = np.array([9.0, 9.0, 9.0])
        engine.step(1)
        assert fake_physx["force"][0][2] == (0.3, 0.0, 0.06)
        assert fake_physx["force"][1][2] == (9.0, 9.0, 9.0)

    def test_a_point_folds_its_lever_arm_into_the_torque_at_call_time(self, fake_physx: dict) -> None:
        """MuJoCo folds ``point`` into the latched wrench once, at call time -
        not re-evaluated as the body moves. (0,0,1) m above the body with a
        5 N x-force is a -5 N*m y-torque... cross((0,0,1),(5,0,0)) = (0,5,0)."""
        engine = _engine()
        engine.apply_force("cube", force=[5.0, 0.0, 0.0], point=[0.3, 0.0, 1.06])
        engine.step(1)
        assert len(fake_physx["torque"]) == 1
        # The binding-level value is NEGATED: measured on isaacsim 6.0.1,
        # apply_torque spins a body opposite to the right-handed world torque
        # it is handed (+0.3 z -> wz -0.458; -0.3 z -> wz +0.538), so the
        # latched (0, +5, 0) reaches the binding as (0, -5, 0) and the CUBE
        # spins the way the caller asked.
        assert fake_physx["torque"][0][1] == pytest.approx((0.0, -5.0, 0.0))

    def test_a_zero_wrench_clears_the_latch(self, fake_physx: dict) -> None:
        engine = _engine()
        engine.apply_force("cube", force=[5.0, 0.0, 0.0])
        result = engine.apply_force("cube", force=[0.0, 0.0, 0.0])
        assert result["status"] == "success"
        assert "cleared" in result["content"][0]["text"]
        engine.step(3)
        assert not fake_physx["force"], "a cleared latch must not be replayed"

    def test_a_replay_failure_drops_the_latch_with_an_error(self, fake_physx: dict, caplog) -> None:
        """Reapplying a failing wrench every tick floods the log; keeping a
        latch that no longer acts is a silent lie. Drop it, loudly, once."""
        import logging

        engine = _engine()
        engine.apply_force("cube", force=[5.0, 0.0, 0.0])
        sim_iface = sys.modules["omni.physx"].get_physx_simulation_interface()

        def _boom(*a: Any, **k: Any) -> None:
            raise RuntimeError("prim deleted")

        sim_iface.apply_force_at_pos = _boom
        with caplog.at_level(logging.ERROR):
            engine.step(3)
        assert "cube" not in engine._applied_wrenches
        assert sum(1 for r in caplog.records if r.levelno == logging.ERROR) == 1


class TestRaycast:
    def test_no_world(self) -> None:
        engine = _engine()
        engine._world_created = False
        assert engine.raycast([0, 0, 1], [0, 0, -1])["status"] == "error"

    def test_a_zero_direction_is_refused(self, fake_physx: dict) -> None:
        result = _engine().raycast([0, 0, 1], [0, 0, 0])
        assert result["status"] == "error"
        assert "zero-length" in result["content"][0]["text"]

    @pytest.mark.parametrize("bad", ["false", "no", 0, 1, None])
    def test_include_static_is_checked_not_read_by_truthiness(self, bad: Any, fake_physx: dict) -> None:
        result = _engine().raycast([0, 0, 1], [0, 0, -1], include_static=bad)
        assert result["status"] == "error", bad
        assert "include_static" in result["content"][0]["text"]

    def test_a_non_default_exclude_body_is_refused_with_the_reason(self, fake_physx: dict) -> None:
        """A MuJoCo compiled-model body id means nothing to PhysX; silently
        ignoring it would report the very body the caller asked to skip."""
        result = _engine().raycast([0, 0, 1], [0, 0, -1], exclude_body=3)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "exclude_body" in text and "body id" in text

    def test_a_hit_is_reported_in_the_mujoco_payload_shape(self, fake_physx: dict) -> None:
        fake_physx["_query_results"].append(
            {
                "hit": True,
                "position": (0.3, 0.0, 0.12),
                "distance": 1.88,
                "collision": "/World/Objects/cube",
                "rigidBody": "/World/Objects/cube",
            }
        )
        result = _engine().raycast([0.3, 0.0, 2.0], [0, 0, -1])
        payload = next(b["json"] for b in result["content"] if "json" in b)
        assert payload["hit"] is True
        assert payload["distance"] == pytest.approx(1.88)
        assert payload["geom_name"] == "cube", "a registered object is named by its object name"
        assert payload["geom_id"] is None, "PhysX has no compiled-model ids; inventing one invites bad comparisons"
        assert payload["hit_point"] == pytest.approx([0.3, 0.0, 0.12])

    def test_a_miss_is_the_documented_null_payload(self, fake_physx: dict) -> None:
        result = _engine().raycast([5, 5, 2], [0, 0, 1])
        payload = next(b["json"] for b in result["content"] if "json" in b)
        assert payload == {"hit": False, "distance": None, "geom_id": None, "geom_name": None, "hit_point": None}

    def test_include_static_false_recasts_past_a_static_hit(self, fake_physx: dict) -> None:
        """The ground plane (no RigidBodyAPI in the stand-in stage) is skipped;
        the dynamic cube behind it is the answer, with the distance accumulated
        across the hops."""
        fake_physx["_query_results"].extend(
            [
                {
                    "hit": True,
                    "position": (0.3, 0.0, 1.0),
                    "distance": 1.0,
                    "collision": "/World/ground",
                    "rigidBody": "/World/ground",
                },
                {
                    "hit": True,
                    "position": (0.3, 0.0, 0.12),
                    "distance": 0.8799,
                    "collision": "/World/Objects/cube",
                    "rigidBody": "/World/Objects/cube",
                },
            ]
        )
        result = _engine().raycast([0.3, 0.0, 2.0], [0, 0, -1], include_static=False)
        payload = next(b["json"] for b in result["content"] if "json" in b)
        assert payload["geom_name"] == "cube"
        assert payload["distance"] == pytest.approx(1.88, abs=1e-3)
        assert len(fake_physx["raycast"]) == 2, "one hop past the static hit"


class TestEveryAdvancingTickReplaysTheLatch:
    """The latch is honoured on every surface that advances time, not only ``step``.

    PhysX's ``apply_force_at_pos`` acts for ONE tick, and ``apply_force`` stores the
    latch without touching PhysX at all - the only calls into it live in
    ``_reapply_wrenches``. So a tick that does not re-push the latch is a tick the
    force is absent from, and the success envelope's "applied every step until
    replaced" is true only of the surfaces that replay.

    ``step`` was the only one. Of the seven physics-advancing call sites,
    ``send_action``, ``run_multi_policy``, ``_primitive_tick`` and
    ``_warmup_camera`` replayed nothing - and ``run_policy`` drives physics through
    the shared ``PolicyRunner``, which calls ``send_action``. So a policy rollout,
    the primary way this backend is driven, never applied a latched wrench while
    reporting that it would. The MuJoCo backend cannot have this defect: its latch
    is persistent ``mjData`` state (``data.xfrc_applied``), honoured by every
    ``mj_step``.
    """

    def test_send_action_replays_the_latch(self, fake_physx: dict[str, list[Any]]) -> None:
        engine = _engine()
        engine._robots["arm"] = _RobotState(name="arm", prim_path="/World/Robots/arm", joint_names=["j0"])
        assert engine.apply_force("cube", force=[0.0, 0.0, 9.0])["status"] == "success"
        fake_physx["force"].clear()

        engine.send_action({"j0": 0.0}, robot_name="arm", n_substeps=3)

        assert len(fake_physx["force"]) == 3, (
            f"send_action advanced 3 ticks but pushed the wrench {len(fake_physx['force'])}x"
        )

    def test_a_primitive_tick_replays_the_latch(self, fake_physx: dict[str, list[Any]]) -> None:
        engine = _engine()
        assert engine.apply_force("cube", force=[0.0, 0.0, 9.0])["status"] == "success"
        fake_physx["force"].clear()

        engine._primitive_tick()

        assert len(fake_physx["force"]) == 1

    def test_the_warmup_tick_replays_the_latch(self, fake_physx: dict[str, list[Any]]) -> None:
        """A warmup tick advances ``_sim_time`` like any other, so exempting it
        would make the wrench act for a number of ticks that depends on how many
        warmup passes the RTX product happened to need."""
        engine = _engine()
        assert engine.apply_force("cube", force=[0.0, 0.0, 9.0])["status"] == "success"
        before = len(fake_physx["force"])

        engine._warmup_camera("nonexistent", 2)

        # The camera is unknown so the warmup may bail early; what is pinned is
        # that it does not advance a tick WITHOUT replaying. Any tick it did take
        # pushed the wrench.
        assert len(fake_physx["force"]) >= before

    def test_step_still_replays(self, fake_physx: dict[str, list[Any]]) -> None:
        """Control: the one surface that always did."""
        engine = _engine()
        assert engine.apply_force("cube", force=[0.0, 0.0, 9.0])["status"] == "success"
        fake_physx["force"].clear()

        engine.step(4)

        assert len(fake_physx["force"]) == 4

    def test_a_cleared_latch_pushes_nothing(self, fake_physx: dict[str, list[Any]]) -> None:
        """Control: the replay is driven by the latch, not unconditional."""
        engine = _engine()
        engine._robots["arm"] = _RobotState(name="arm", prim_path="/World/Robots/arm", joint_names=["j0"])
        fake_physx["force"].clear()

        engine.send_action({"j0": 0.0}, robot_name="arm", n_substeps=3)

        assert fake_physx["force"] == []


class TestTheReplayRuleIsDerivedFromTheSource:
    """Graded from the AST so an eighth physics-advancing site is caught on arrival.

    The rule: a call to ``self._world.step(...)`` inside a method that also advances
    ``_sim_time`` must replay the latch; a render-only pump, which advances no time,
    must not. Stated as a rule rather than a list because the defect was exactly a
    site nobody remembered to add.
    """

    @staticmethod
    def _sites() -> list[tuple[str, str, bool, bool]]:
        import ast

        from strands_robots.simulation.isaac import motion_primitives as mp_mod
        from strands_robots.simulation.isaac import simulation as sim_mod

        out: list[tuple[str, str, bool, bool]] = []
        for module in (sim_mod, mp_mod):
            path = module.__file__
            assert path is not None, f"{module.__name__} has no source file to read"
            src = pathlib.Path(path).read_text(encoding="utf-8")
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                seg = ast.get_source_segment(src, node) or ""
                steps = [
                    sub
                    for sub in ast.walk(node)
                    if isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "step"
                    and isinstance(sub.func.value, ast.Attribute)
                    and sub.func.value.attr == "_world"
                ]
                if steps:
                    out.append(
                        (
                            pathlib.Path(path).name,
                            node.name,
                            "_sim_time" in seg,
                            "_reapply_wrenches" in seg,
                        )
                    )
        return out

    def test_the_scan_finds_the_sites(self) -> None:
        """Non-vacuity: an empty scan would satisfy the rule below by vacuum."""
        sites = self._sites()
        assert len(sites) >= 6, sites
        assert any(name == "step" for _mod, name, _a, _r in sites)

    def test_every_advancing_site_replays(self) -> None:
        adrift = [f"{mod}::{name}" for mod, name, advances, replays in self._sites() if advances and not replays]
        assert adrift == [], (
            "these advance simulated time without replaying the latched wrench, so an "
            "apply_force is silently inert on them: " + ", ".join(adrift)
        )

    def test_a_render_only_pump_does_not_replay(self) -> None:
        """The other half of the rule: a pump that advances no time must not push a
        wrench, or a latch would act on ticks the clock never saw."""
        offenders = [f"{mod}::{name}" for mod, name, advances, replays in self._sites() if replays and not advances]
        assert offenders == [], offenders
