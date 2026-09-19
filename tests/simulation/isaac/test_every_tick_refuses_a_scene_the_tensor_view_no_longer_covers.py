"""Every path that advances the clock refuses a stale tensor view, not just ``step``.

:mod:`tests.simulation.isaac.test_step_refuses_a_scene_the_tensor_view_no_longer_covers`
establishes the flag and what ``step`` does about it. This grades the other three
time-advancing paths, which did not check it - and one of them is the path a
rollout actually drives physics through, so the refusal covered the verb an agent
calls least.

Adding or removing a DYNAMIC body invalidates the view PhysX built at
``world.reset()``. Until a ``reset()`` rebuilds it, every robot's
``get_observation()`` comes back empty and a body does not move, so a tick against
it advances ``_sim_time`` over a scene that is not being simulated.

The four sites, and why each answers differently:

* ``step`` - returns an envelope, and now also RE-checks per batch. It releases
  the lock every ``_STEPS_PER_BATCH`` ticks, so a worker thread's dynamic add
  lands mid-``step(N)``; checking only on entry left every remaining batch
  ticking a view that had gone stale under it.
* ``send_action`` - returns an envelope. This is the reachable one:
  ``add_object(is_static=False)`` then ``run_policy(...)`` reaches physics here,
  so a whole rollout - and any dataset recorded from it - ran over an
  un-simulated scene reporting ``status="success"`` while the action silently did
  not apply.
* ``run_multi_policy`` - returns an envelope from its preflight, and RAISES from
  the per-step hop. The hop returns ``None``, so an envelope is not available
  there; raising is the shape its sibling empty-chunk guard already uses, and the
  ``finally`` discards the partial episode so no half-rollout is kept as if it
  had simulated.
* ``_warmup_camera`` - returns ``False``. It is declared ``-> bool`` and
  documents "never raises", so the honest answer is that the camera did not warm
  up. Its budget would otherwise be spent stepping a scene PhysX no longer
  covers, waiting for frames that cannot arrive.

One refusal text, one owner: ``_physics_view_stale_error`` names the calling verb
and the remedy. Four hand-kept copies is how three of them come to omit the part
about ``reset()`` returning robots to their default pose.

Unit-level, as the sibling module is: the Kit leaves are stood in, so what is
graded is which paths refuse and what they say.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import (  # noqa: E402 - after importorskip
    IsaacConfig,
    IsaacSimulation,
    _physics_view_stale_error,
    _RobotState,
)

#: The remedy every refusal owes its caller.
_REMEDY = "reset()"


class _Scene:
    def add(self, handle: Any) -> None:
        return None

    def remove_object(self, name: str) -> None:
        return None


class _World:
    def __init__(self) -> None:
        self.physics_sim_view = object()
        self.step_calls = 0
        self.scene = _Scene()

    def reset(self) -> None:
        return None

    def step(self, render: bool = False) -> None:
        self.step_calls += 1

    def play(self) -> None:
        return None

    def stop(self) -> None:
        return None


class _Articulation:
    dof_names = ["j0"]

    def initialize(self, *a: Any, **k: Any) -> None:
        return None

    def get_joint_positions(self) -> Any:
        return [0.0]

    def get_joint_velocities(self) -> Any:
        return [0.0]

    def set_joint_position_targets(self, *a: Any, **k: Any) -> None:
        return None

    def set_joint_positions(self, *a: Any, **k: Any) -> None:
        return None


def _engine(*, stale: bool) -> Any:
    """A skeleton engine whose world is live and whose view may be stale."""
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(render_mode="headless")
    engine._world = _World()
    engine._world_created = True
    engine._objects = {}
    engine._cameras = {}
    engine._scene_objects = set()
    engine._prim_registry = []
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._recording_state_dict = {}
    engine._applied_wrenches = {}
    engine._action_controllers = {}
    engine._main_tid = threading.get_ident()
    engine._pump_running = False
    engine._physics_view_stale = stale
    robot = _RobotState(name="arm", prim_path="/World/Robots/arm", joint_names=["j0"])
    robot.articulation = _Articulation()
    engine._robots = {"arm": robot}
    return engine


def _text(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result.get("content", []))


class TestSendActionRefuses:
    """The reachable path: this is how a rollout reaches physics."""

    def test_it_returns_an_error(self) -> None:
        engine = _engine(stale=True)

        result = engine.send_action({"j0": 0.1}, "arm")

        assert result["status"] == "error"

    def test_it_does_not_advance_the_clock(self) -> None:
        engine = _engine(stale=True)

        engine.send_action({"j0": 0.1}, "arm")

        assert engine._sim_time == 0.0
        assert engine._step_count == 0
        assert engine._world.step_calls == 0

    def test_the_refusal_names_the_verb_and_the_remedy(self) -> None:
        engine = _engine(stale=True)

        text = _text(engine.send_action({"j0": 0.1}, "arm"))

        assert "send_action" in text
        assert _REMEDY in text

    def test_it_says_the_pose_is_returned_to_default(self) -> None:
        """The part a copied refusal drops. ``reset()`` un-poses the robot, so a
        caller told only to call it loses a pose without being warned."""
        engine = _engine(stale=True)

        assert "default pose" in _text(engine.send_action({"j0": 0.1}, "arm"))

    def test_an_unknown_robot_still_reports_itself(self) -> None:
        """Ordering: the staleness check sits AFTER robot resolution, so a bad
        name is not answered with a lecture about the tensor view."""
        engine = _engine(stale=True)

        text = _text(engine.send_action({"j0": 0.1}, "ghost"))

        assert "ghost" in text
        assert "tensor view" not in text

    def test_a_live_view_does_not_reach_this_refusal(self) -> None:
        """The control, expressed as the guard NOT firing rather than as a
        success: the joint write below it imports ``isaacsim``, which is absent
        here, so this call cannot succeed in a stub environment. What matters is
        that it fails for that reason and not for staleness - a guard that
        refused a live view would be worse than no guard, and asserting on the
        status alone could not tell the two apart."""
        engine = _engine(stale=False)

        text = _text(engine.send_action({"j0": 0.1}, "arm"))

        assert "tensor view" not in text
        assert _REMEDY not in text


class TestRunMultiPolicyRefusesInItsPreflight:
    def test_it_returns_an_error_before_any_step(self) -> None:
        engine = _engine(stale=True)

        result = engine.run_multi_policy({"arm": "mock"}, duration=0.1)

        assert result["status"] == "error"
        assert engine._world.step_calls == 0

    def test_the_refusal_names_the_verb(self) -> None:
        engine = _engine(stale=True)

        text = _text(engine.run_multi_policy({"arm": "mock"}, duration=0.1))

        assert "run_multi_policy" in text
        assert _REMEDY in text

    def test_it_is_refused_ahead_of_the_policy_validation(self) -> None:
        """A stale view is refused even for a policy spec that would itself be
        rejected, so the caller fixes the scene rather than chasing the spec."""
        engine = _engine(stale=True)

        text = _text(engine.run_multi_policy({}, duration=0.1))

        assert "tensor view" in text


class TestWarmupCameraSkips:
    def test_it_reports_that_the_camera_did_not_warm_up(self) -> None:
        engine = _engine(stale=True)

        assert engine._warmup_camera("cam", n_steps=5) is False

    def test_it_spends_no_step_budget(self) -> None:
        engine = _engine(stale=True)

        engine._warmup_camera("cam", n_steps=5)

        assert engine._world.step_calls == 0
        assert engine._sim_time == 0.0

    def test_it_says_why(self, caplog: pytest.LogCaptureFixture) -> None:
        engine = _engine(stale=True)

        with caplog.at_level("WARNING"):
            engine._warmup_camera("cam", n_steps=5)

        assert _REMEDY in caplog.text


class TestThePrimitivesRefuse:
    """``move_to`` / ``rotate_wrist`` / ``set_gripper`` all drive ``_primitive_tick``.

    Pre-fix the loop burned its whole budget on a dead view and then blamed the
    SERVO - measured, ``rotate_wrist`` reported "residual 0.3000 rad" - which
    sends the caller after a tolerance or a gain for a joint that was never going
    to move. That misdirection is the cost, and it is why the refusal has to name
    the view.
    """

    @pytest.mark.parametrize("action", ["move_to", "rotate_wrist", "set_gripper"])
    def test_the_shared_preflight_refuses(self, action: str) -> None:
        engine = _engine(stale=True)

        _name, _robot, error = engine._primitive_resolve_robot(action, "arm")

        assert error is not None
        assert error["status"] == "error"
        assert _REMEDY in _text(error)

    @pytest.mark.parametrize("action", ["move_to", "rotate_wrist", "set_gripper"])
    def test_the_refusal_names_the_action(self, action: str) -> None:
        engine = _engine(stale=True)

        _name, _robot, error = engine._primitive_resolve_robot(action, "arm")

        assert action in _text(error or {})

    def test_a_live_view_resolves_the_robot(self) -> None:
        """The control: the preflight must not refuse a usable scene."""
        engine = _engine(stale=False)

        name, robot, error = engine._primitive_resolve_robot("move_to", "arm")

        assert error is None
        assert name == "arm"
        assert robot is not None

    def test_the_mid_run_check_aborts(self) -> None:
        """The loop releases the lock between ticks, so a worker thread's dynamic
        add lands mid-primitive - the same window the world/robot/policy aborts
        beside it exist for."""
        engine = _engine(stale=True)

        reason = engine._primitive_abort_reason("move_to", "arm")

        assert reason is not None
        assert "tensor view" in _text(reason)

    def test_the_mid_run_check_passes_a_live_view(self) -> None:
        engine = _engine(stale=False)

        assert engine._primitive_abort_reason("move_to", "arm") is None


class TestTheRefusalHasOneOwner:
    """The wording is the remedy, so it is shared rather than copied."""

    def test_the_helper_answers_none_on_a_live_view(self) -> None:
        engine = _engine(stale=False)

        assert _physics_view_stale_error(engine, "step") is None

    def test_it_names_whatever_verb_it_is_given(self) -> None:
        engine = _engine(stale=True)

        text = _text(_physics_view_stale_error(engine, "some_verb") or {})

        assert "some_verb" in text

    def test_step_still_refuses_through_it(self) -> None:
        """The behaviour the sibling module pins, re-checked here because ``step``
        now reaches it through the shared helper rather than an inline copy."""
        engine = _engine(stale=True)

        result = engine.step(1)

        assert result["status"] == "error"
        assert "step" in _text(result)
        assert engine._world.step_calls == 0

    def test_every_time_advancing_site_consults_it(self) -> None:
        """Derived from the source: a fifth stepping path added later is graded on
        arrival rather than at whichever review happens to look.

        Scoped to the functions that advance ``_sim_time`` - the render-convergence
        helpers step for frames rather than for physics and make no claim about
        simulated time, so they are deliberately out of scope."""
        import ast
        import pathlib

        # The WHOLE package, not just simulation.py. Scoping this to one module
        # is how the fifth site was missed: ``motion_primitives._primitive_tick``
        # advances ``_sim_time`` in a different file, and its own comment calls
        # itself a peer of ``step`` and ``send_action`` while consulting neither
        # gate. A package-wide walk is what makes this pin answer for the rule
        # rather than for one file.
        package_file = __import__("strands_robots.simulation.isaac", fromlist=["x"]).__file__
        assert package_file is not None, "the isaac package has no source file to read"
        sources = sorted(pathlib.Path(package_file).parent.glob("*.py"))
        assert len(sources) > 1, "the isaac package collapsed to one module; re-scope this sweep"
        tree = ast.parse("\n".join(path.read_text(encoding="utf-8") for path in sources))

        # A site that advances the clock is one whose body holds BOTH a
        # ``self._sim_time += ...`` and a ``self._world.step(...)``. The nested
        # per-step hops (``_apply_all_and_step``) are reached by walking every
        # FunctionDef, including nested ones, which is why the guard has to be
        # looked for in the same node rather than in the enclosing method.
        def advances(node: ast.FunctionDef) -> bool:
            dumped = ast.dump(node)
            return "_sim_time" in dumped and "attr='step'" in dumped

        # A tick that does not consult the gate itself must be covered by a
        # named shared guard, listed here with WHERE. Explicit rather than a
        # looser predicate, because every predicate broad enough to accept these
        # two also accepts an unguarded site added later - which is precisely how
        # ``_primitive_tick`` went unnoticed.
        _COVERED_ELSEWHERE = {
            # Nested hop of run_multi_policy, which refuses in its preflight and
            # re-checks inside this hop.
            "_apply_all_and_step": "run_multi_policy preflight",
            # The three primitives share _primitive_resolve_robot (preflight) and
            # _primitive_abort_reason (mid-loop), both of which consult the gate.
            "_primitive_tick": "_primitive_resolve_robot / _primitive_abort_reason",
        }

        advancing = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and advances(node)]
        assert advancing, "found no clock-advancing sites at all; this sweep has stopped measuring"

        offenders = [
            node.name
            for node in advancing
            if "_physics_view_stale" not in ast.dump(node) and node.name not in _COVERED_ELSEWHERE
        ]

        assert offenders == [], f"these advance _sim_time without consulting the stale-view gate: {offenders}"

        # And the covering guards really do consult it, so an entry above cannot
        # become a permanent exemption for a guard that was later deleted.
        guards = {
            node.name: ast.dump(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name
            in {
                "run_multi_policy",
                "_primitive_resolve_robot",
                "_primitive_abort_reason",
            }
        }
        for name, where in _COVERED_ELSEWHERE.items():
            covering = [g for g, body in guards.items() if "_physics_view_stale" in body]
            assert covering, f"{name} is exempted as covered by {where}, but no such guard consults the gate"
