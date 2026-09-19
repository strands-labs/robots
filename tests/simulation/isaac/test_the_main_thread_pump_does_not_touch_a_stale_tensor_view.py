"""The stale-view gate guards two harms, and the pump surfaces only have the second.

:mod:`tests.simulation.isaac.test_every_tick_refuses_a_scene_the_tensor_view_no_longer_covers`
grades the first harm - advancing ``_sim_time`` over a scene PhysX is not
simulating, which reports success for an action that never applied. Its sweep is
keyed on exactly that: a function whose body holds both a ``self._sim_time += ...``
and a ``self._world.step(...)``, with render-convergence helpers declared out of
scope because they "step for frames rather than for physics".

That predicate is right for what it measures and blind to the other harm the same
flag exists for: an articulation tensor **read or write** against an invalidated
view. ``remove_object`` measured it - "the next joint read HUNG - it did not return
empty, the process wedged until a 2-minute timeout killed it" - and
``get_observation`` records the non-hanging outcome, a bare ``Exception`` ("Failed
to get DOF positions from backend") that the narrow
``(RuntimeError, ValueError, AttributeError, TypeError)`` handlers around these
reads cannot catch and which AGENTS.md forbids widening to ``except Exception``.

Three surfaces performed such a read or write with no gate, and none of them
advances ``_sim_time``, so the sibling sweep could not see any of them:

* ``pump`` step 3 - refreshes the joint cache for every robot. This is the worst
  placement in the backend: it runs on the MAIN thread and ``run_pump_forever``
  wraps it in ``try`` / ``finally`` with **no** ``except``, so one escape ends the
  loop and takes the app down, and the hang wedges a live UI session. It now skips
  the refresh and keeps the last good cache, logging once rather than per ~50 ms
  tick.
* ``_converge_render`` - reads and then WRITES (``set_joint_positions`` plus
  ``set_joint_velocities``), ``max(1, n)`` times per call, reached from pump's step
  2 on the default idle preview path. It now drops the pose-hold and keeps
  rendering, so the preview stays live instead of freezing, and it re-reads the
  flag per iteration for the reason ``step`` re-checks per batch.
* ``set_joint_positions`` - reads the live vector before writing so a partial dict
  updates only the DOFs it names. It returns an envelope, so it refuses through the
  shared helper, placed after the robot resolves as ``send_action``'s gate is.

The trigger is ordinary shipped usage rather than an engineered race: a worker
thread's ``remove_object``, or ``load_scene``'s per-episode reload, invalidates the
view and the very next pump tick performs the read.

The sweep at the bottom is keyed on the articulation operation rather than on the
clock, so it answers for this second axis the way the sibling answers for the first.
Unit-level, as the sibling is: the Kit leaves are stood in, so what is graded is
which surfaces touch the view and what they do instead.
"""

from __future__ import annotations

import ast
import logging
import pathlib
import queue
import threading
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import (  # noqa: E402 - after importorskip
    IsaacConfig,
    IsaacSimulation,
    _RobotState,
)

#: What ``remove_object`` measured the backend raising when it does not hang. Not a
#: subclass of any handler tuple around these reads, which is the whole point.
_BACKEND_REFUSAL = "Failed to get DOF positions from backend"


class _Scene:
    def add(self, handle: Any) -> None:
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
    """Records every tensor touch, and can raise what the real backend raises.

    ``raises`` is the bare ``Exception`` case rather than the hang, because a test
    cannot assert a two-minute wedge: the two share a cause and a remedy, and the
    raise is the half that is observable in-process. A surface that does not call
    these methods at all is immune to both.
    """

    dof_names = ["j0", "j1"]

    def __init__(self, raises: bool = False) -> None:
        self.reads = 0
        self.position_writes: list[Any] = []
        self.velocity_writes = 0
        self._raises = raises

    def initialize(self, *a: Any, **k: Any) -> None:
        return None

    def get_joint_positions(self) -> Any:
        self.reads += 1
        if self._raises:
            raise Exception(_BACKEND_REFUSAL)  # noqa: TRY002 - the backend's own shape
        return [0.25, 0.5]

    def get_joint_velocities(self) -> Any:
        return [0.0, 0.0]

    def set_joint_positions(self, values: Any, *a: Any, **k: Any) -> None:
        self.position_writes.append(values)

    def set_joint_velocities(self, *a: Any, **k: Any) -> None:
        self.velocity_writes += 1

    def set_joint_position_targets(self, *a: Any, **k: Any) -> None:
        return None


def _engine(*, stale: bool, raises: bool = False) -> Any:
    """A skeleton engine with a live world, a pump, and a possibly-stale view."""
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
    engine._pump_stale_warned = False
    engine._action_q = queue.Queue()
    engine._main_jobs = queue.Queue()
    engine._joint_cache = {}
    engine._frame_cache = {}
    engine._pump_cameras = False
    engine._idle_converge = 3
    robot = _RobotState(name="arm", prim_path="/World/Robots/arm", joint_names=["j0", "j1"])
    robot.articulation = _Articulation(raises=raises)
    engine._robots = {"arm": robot}
    return engine


def _text(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result.get("content", []))


class TestPumpDoesNotReadAStaleView:
    """The headline. Pre-fix this read ran on every tick of the main-thread loop."""

    def test_the_joint_read_is_not_attempted(self) -> None:
        engine = _engine(stale=True)

        engine.pump(render=False)

        assert engine._robots["arm"].articulation.reads == 0

    def test_a_live_view_is_still_read(self) -> None:
        """Control: the cache refresh is the pump's job and still happens."""
        engine = _engine(stale=False)

        engine.pump(render=False)

        assert engine._robots["arm"].articulation.reads == 1
        assert engine._joint_cache["arm"] == {"j0": 0.25, "j1": 0.5}

    def test_the_last_good_cache_survives(self) -> None:
        """Skipping is not clearing: a consumer reading the cache gets the last
        answer that was true rather than an empty dict it cannot tell from a robot
        with no joints."""
        engine = _engine(stale=False)
        engine.pump(render=False)
        good = dict(engine._joint_cache["arm"])

        engine._physics_view_stale = True
        engine.pump(render=False)

        assert engine._joint_cache["arm"] == good

    def test_the_backend_refusal_does_not_escape_the_pump(self) -> None:
        """The failure mode, end to end. ``run_pump_forever`` wraps ``pump`` in
        ``try`` / ``finally`` with no ``except``, so an escape here ends the loop -
        and this exception is a bare ``Exception``, which the pump's own narrow
        handler cannot catch. Not attempting the read is what makes it unreachable.
        """
        engine = _engine(stale=True, raises=True)

        engine.pump(render=False)  # must not raise

        assert engine._robots["arm"].articulation.reads == 0

    def test_the_premise_that_no_handler_would_have_caught_it(self) -> None:
        """Why the gate is the fix rather than a wider ``except``: the class the
        backend raises is outside every handler tuple around these reads, and
        AGENTS.md forbids widening to ``except Exception``."""
        assert not issubclass(Exception, (RuntimeError, ValueError, AttributeError, TypeError, KeyError, IndexError))

    def test_it_is_reported_once_not_once_per_tick(self, caplog: pytest.LogCaptureFixture) -> None:
        """``run_pump_forever`` pumps every ~50 ms, so an unlatched warning writes
        thousands of identical lines while the view stays stale."""
        engine = _engine(stale=True)

        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                engine.pump(render=False)

        stale_records = [r for r in caplog.records if "tensor view no longer covers" in r.getMessage()]
        assert len(stale_records) == 1, f"expected one warning over five ticks, got {len(stale_records)}"

    def test_the_report_names_the_remedy(self, caplog: pytest.LogCaptureFixture) -> None:
        engine = _engine(stale=True)

        with caplog.at_level(logging.WARNING):
            engine.pump(render=False)

        text = " ".join(r.getMessage() for r in caplog.records)
        assert "reset()" in text
        assert "pump()" in text

    def test_the_latch_clears_when_the_view_is_rebuilt(self, caplog: pytest.LogCaptureFixture) -> None:
        """So a SECOND staleness is reported rather than swallowed by the first
        one's latch."""
        engine = _engine(stale=True)
        engine.pump(render=False)
        engine._physics_view_stale = False
        engine.pump(render=False)
        engine._physics_view_stale = True

        with caplog.at_level(logging.WARNING):
            engine.pump(render=False)

        assert [r for r in caplog.records if "tensor view no longer covers" in r.getMessage()]


class TestConvergeRenderDropsThePoseHoldAndKeepsRendering:
    def test_it_neither_reads_nor_writes(self) -> None:
        engine = _engine(stale=True)

        engine._converge_render(3)

        art = engine._robots["arm"].articulation
        assert (art.reads, art.position_writes, art.velocity_writes) == (0, [], 0)

    def test_it_still_renders_every_tick(self) -> None:
        """A frozen preview reads to an operator as a hung app, so the render is
        what must NOT be skipped. This is the half that distinguishes the gate from
        an early return."""
        engine = _engine(stale=True)

        engine._converge_render(3)

        assert engine._world.step_calls == 3

    def test_a_live_view_still_holds_the_pose(self) -> None:
        """Control: the DLSS-convergence behaviour is unchanged when the view is
        good."""
        engine = _engine(stale=False)

        engine._converge_render(2)

        art = engine._robots["arm"].articulation
        assert art.reads == 2
        assert len(art.position_writes) == 2
        assert art.velocity_writes == 2
        assert engine._world.step_calls == 2

    def test_the_flag_is_re_read_every_iteration(self) -> None:
        """A worker's dynamic remove lands BETWEEN iterations - the loop holds no
        lock - so reading the flag once above it would hold the pose for the rest
        of the call against a view that had gone stale under it. Same reason
        ``step`` re-checks per batch.
        """
        engine = _engine(stale=False)
        art = engine._robots["arm"].articulation
        original = engine._world.step

        def _go_stale_after_first(render: bool = False) -> None:
            original(render=render)
            engine._physics_view_stale = True

        engine._world.step = _go_stale_after_first  # type: ignore[method-assign]

        engine._converge_render(4)

        assert art.reads == 1, f"kept reading after the view went stale: {art.reads} reads"
        assert engine._world.step_calls == 4, "the render stopped instead of continuing"

    def test_the_backend_refusal_does_not_escape(self) -> None:
        engine = _engine(stale=True, raises=True)

        engine._converge_render(2)  # must not raise

        assert engine._world.step_calls == 2


class TestSetJointPositionsRefusesAStaleView:
    """It reads before it writes, so it carries the same hazard - and it is queued
    onto the pump when called off the main thread, which is how its escape reaches
    ``run_pump_forever``."""

    def test_it_returns_an_error(self) -> None:
        engine = _engine(stale=True)

        result = engine.set_joint_positions(positions={"j0": 0.1})

        assert result["status"] == "error"

    def test_the_refusal_names_the_verb_and_the_remedy(self) -> None:
        engine = _engine(stale=True)

        text = _text(engine.set_joint_positions(positions={"j0": 0.1}))

        assert "set_joint_positions" in text
        assert "reset()" in text

    def test_it_neither_reads_nor_writes(self) -> None:
        engine = _engine(stale=True)

        engine.set_joint_positions(positions={"j0": 0.1})

        art = engine._robots["arm"].articulation
        assert (art.reads, art.position_writes) == (0, [])

    def test_a_live_view_still_writes(self) -> None:
        engine = _engine(stale=False)

        result = engine.set_joint_positions(positions={"j0": 0.1})

        assert result["status"] == "success"
        assert engine._robots["arm"].articulation.position_writes

    def test_an_unknown_robot_reports_itself_rather_than_the_staleness(self) -> None:
        """Placement: after the robot resolves, as ``send_action``'s gate is, so the
        caller is told which of their own arguments to fix."""
        engine = _engine(stale=True)

        text = _text(engine.set_joint_positions(robot_name="nope", positions={"j0": 0.1}))

        assert "nope" in text
        assert "reset()" not in text


class TestEveryArticulationTouchConsultsTheGate:
    """The sweep on the second axis, derived from the source.

    Keyed on the tensor OPERATION rather than on the clock, because the sibling
    module's clock-keyed sweep is what let all three of the surfaces above ship
    ungated: none of them advances ``_sim_time``, and one of them is explicitly
    named out of scope there as a render helper.
    """

    #: Reads and writes that go through PhysX's tensor view, i.e. the ones an
    #: invalidated view answers by hanging or by raising a bare ``Exception``.
    _OPS = (
        "get_joint_positions",
        "set_joint_positions",
        "get_joint_velocities",
        "set_joint_velocities",
        "get_joint_efforts",
        "set_joint_efforts",
        "apply_action",
    )

    #: A surface that does not consult the gate itself must be covered by a named
    #: one, listed with WHERE. Explicit rather than a looser predicate, for the
    #: reason the sibling sweep gives for its own map: every predicate broad enough
    #: to accept these also accepts an ungated surface added later.
    _COVERED_ELSEWHERE = {
        # Reached only from move_to / set_gripper / rotate_wrist, which share
        # _primitive_resolve_robot (preflight) and _primitive_abort_reason (mid-loop).
        "_read_joint_positions": "_primitive_resolve_robot / _primitive_abort_reason",
        "_apply_position_targets": "_primitive_resolve_robot / _primitive_abort_reason",
        # Sole caller is _apply_all_and_step, which raises when the view is stale.
        "_apply_lockstep_action": "_apply_all_and_step",
        # Runs INSIDE reset(), between world.reset() rebuilding the view and the
        # flag being cleared. A gate here would refuse the very call that repairs
        # the view, so its exemption is structural rather than a concession.
        "_revive_articulations_after_reset": "reset() itself, which rebuilds the view",
    }

    @staticmethod
    def _sources() -> list[pathlib.Path]:
        """The whole isaac package, for the reason the sibling sweep walks it: two
        of the four exempt surfaces live in ``motion_primitives``, so a one-module
        scope would grade neither."""
        package_file = __import__("strands_robots.simulation.isaac", fromlist=["x"]).__file__
        assert package_file is not None, "the isaac package has no source file to read"
        sources = sorted(pathlib.Path(package_file).parent.glob("*.py"))
        assert len(sources) > 1, "the isaac package collapsed to one module; re-scope this sweep"
        return sources

    def _touching_scopes(self) -> dict[str, set[str]]:
        """``{innermost function name: ops it performs}`` for every tensor touch,
        paired with whether any ENCLOSING scope consults the gate.

        The enclosing chain matters and is easy to get wrong: ``set_joint_positions``
        performs its read inside a nested ``_apply`` closure, so a sweep that asked
        only about the innermost function would have called it ungated even after
        the outer method grew a gate.
        """
        touched: dict[str, set[str]] = {}
        gated: set[str] = set()
        for path in self._sources():
            source = path.read_text(encoding="utf-8")
            lines = source.splitlines()
            tree = ast.parse(source, filename=str(path))
            funcs = sorted(
                (node.lineno, node.end_lineno or node.lineno, node.name)
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            )
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if node.func.attr not in self._OPS:
                    continue
                chain = [f for f in funcs if f[0] <= node.lineno <= f[1]]
                if not chain:
                    continue
                name = chain[-1][2]
                touched.setdefault(name, set()).add(node.func.attr)
                if any("_physics_view_stale" in "\n".join(lines[s - 1 : e]) for s, e, _ in chain):
                    gated.add(name)
        self._gated = gated
        return touched

    def test_the_sweep_is_not_vacuous(self) -> None:
        touched = self._touching_scopes()

        assert len(touched) >= 5, f"found only {len(touched)} tensor-touching scopes; the sweep has drifted"

    def test_every_touch_is_gated_or_named_as_covered(self) -> None:
        touched = self._touching_scopes()

        offenders = sorted(
            f"{name} ({sorted(ops)})"
            for name, ops in touched.items()
            if name not in self._gated and name not in self._COVERED_ELSEWHERE
        )

        assert offenders == [], (
            "these read or write PhysX's tensor view without consulting the stale-view gate, "
            "so against an invalidated view they hang or raise a bare Exception no handler here "
            f"may catch: {offenders}"
        )

    def test_the_three_fixed_surfaces_are_gated_rather_than_exempted(self) -> None:
        """Named explicitly so a later change cannot quietly move one into the
        exemption map instead of keeping its gate.

        ``_apply`` rather than ``set_joint_positions`` because that is the scope the
        walk yields: the read lives in a nested closure, and it counts as gated only
        because the walk credits an ENCLOSING scope's check. That is the case this
        class's own docstring calls easy to get wrong, so it is the one pinned by
        name - asserting on ``set_joint_positions`` here passes vacuously if the
        closure is ever hoisted out, and fails spuriously today.
        """
        self._touching_scopes()

        for name in ("pump", "_converge_render", "_apply"):
            assert name in self._gated, f"{name} no longer consults the gate"
            assert name not in self._COVERED_ELSEWHERE, f"{name} was exempted rather than gated"

    def test_no_exemption_is_stale(self) -> None:
        """An entry in the map has to still be a surface that touches the view;
        otherwise it is a permanent licence for whatever later takes that name."""
        touched = self._touching_scopes()

        unused = sorted(name for name in self._COVERED_ELSEWHERE if name not in touched)

        assert unused == [], f"these exemptions no longer name a tensor-touching surface: {unused}"

    def test_each_covering_guard_really_consults_the_gate(self) -> None:
        """So an exemption cannot outlive the guard it cites."""
        bodies: dict[str, str] = {}
        for path in self._sources():
            source = path.read_text(encoding="utf-8")
            lines = source.splitlines()
            for node in ast.walk(ast.parse(source, filename=str(path))):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    bodies[node.name] = "\n".join(lines[node.lineno - 1 : node.end_lineno or node.lineno])

        for guard in ("_primitive_resolve_robot", "_primitive_abort_reason", "_apply_all_and_step"):
            assert guard in bodies, f"{guard} is cited as a covering guard but no longer exists"
            assert "_physics_view_stale" in bodies[guard], f"{guard} is cited as covering but does not consult the gate"
