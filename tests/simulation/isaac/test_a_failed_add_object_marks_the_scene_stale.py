"""A **failed** ``add_object`` marks the scene stale, as a successful one does.

``_physics_view_stale`` exists because adding or removing a physics body invalidates
PhysX's tensor simulation view, and ``step()`` refuses to advance until ``reset()``
rebuilds it - otherwise the clock and step count move over a scene PhysX is no longer
simulating and every robot's ``get_observation`` comes back empty.

Only the *success* path set it. And the failure path is stale by exactly the same
mechanism, which that clause's own comment already states:
``_construct_shape_prim`` stops the timeline - clearing the physics sim view -
**before** constructing a dynamic prim, so by the time the handler runs the view has
already been invalidated whether the construction went on to succeed or to raise.

So a failed ``add_object`` left ``step()`` willing to advance. To a caller that reads
as a transient add failure followed by a simulation that had quietly stopped
simulating: no exception, a plausible error envelope, and then empty observations
attributed to whatever ran next.

Nothing here needs Isaac Sim: ``_construct_shape_prim`` is stood in and made to raise
each of the exception types the handler is written for.
"""

from __future__ import annotations

import threading
import types
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.config import IsaacConfig  # noqa: E402
from strands_robots.simulation.isaac.simulation import IsaacSimulation  # noqa: E402

#: Every type the handler names. A guard that covered one would leave the rest.
_RAISES = [RuntimeError, ValueError, OSError, AttributeError, TypeError, ImportError]


def _engine(*, construct: Any) -> Any:
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(render_mode="headless")
    engine._world_created = True
    engine._world = types.SimpleNamespace(
        scene=types.SimpleNamespace(add=lambda h: None),
        step=lambda **k: None,
    )
    engine._objects = {}
    engine._robots = {}
    engine._prim_registry = []
    engine._applied_wrenches = {}
    engine._physics_view_stale = False
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._main_tid = threading.get_ident()
    engine._pump_running = False
    engine._construct_shape_prim = construct  # type: ignore[method-assign]
    return engine


def _raiser(exc: type[BaseException]) -> Any:
    def _construct(*a: Any, **k: Any) -> Any:
        raise exc("the prim could not be constructed")

    return _construct


def _succeeder() -> Any:
    def _construct(*a: Any, **k: Any) -> Any:
        return object(), [0.05, 0.05, 0.05]

    return _construct


class TestAFailedAddObjectMarksTheSceneStale:
    @pytest.mark.parametrize("exc", _RAISES, ids=[e.__name__ for e in _RAISES])
    def test_the_flag_is_set(self, exc: type[BaseException]) -> None:
        engine = _engine(construct=_raiser(exc))

        result = engine.add_object(name="cube", shape="cuboid", position=[0.0, 0.0, 0.5], size=[0.05] * 3)

        assert result["status"] == "error", result
        assert engine._physics_view_stale is True, "a failed add left step() willing to advance"

    @pytest.mark.parametrize("exc", _RAISES, ids=[e.__name__ for e in _RAISES])
    def test_step_then_refuses(self, exc: type[BaseException]) -> None:
        """The consequence the flag exists for, driven end to end."""
        engine = _engine(construct=_raiser(exc))
        engine.add_object(name="cube", shape="cuboid", position=[0.0, 0.0, 0.5], size=[0.05] * 3)

        stepped = engine.step(1)

        assert stepped["status"] == "error", "step advanced over a scene the view no longer covers"

    def test_the_clock_does_not_move_after_a_failed_add(self) -> None:
        """What the refusal protects: a clock that advanced over an unsimulated
        scene, which is what made the degradation look like something else."""
        engine = _engine(construct=_raiser(RuntimeError))
        engine.add_object(name="cube", shape="cuboid", position=[0.0, 0.0, 0.5], size=[0.05] * 3)

        engine.step(5)

        assert engine._step_count == 0
        assert engine._sim_time == 0.0

    def test_the_object_is_not_registered(self) -> None:
        """A failed add must not leave bookkeeping behind either."""
        engine = _engine(construct=_raiser(RuntimeError))

        engine.add_object(name="cube", shape="cuboid", position=[0.0, 0.0, 0.5], size=[0.05] * 3)

        assert "cube" not in engine._objects
        assert engine._prim_registry == []


class TestASuccessfulAddStillMarksItStale:
    """The control this fix must not disturb - the original behaviour."""

    def test_the_flag_is_set_on_success(self) -> None:
        engine = _engine(construct=_succeeder())

        result = engine.add_object(name="cube", shape="cuboid", position=[0.0, 0.0, 0.5], size=[0.05] * 3)

        assert result["status"] == "success", result
        assert engine._physics_view_stale is True

    def test_a_reset_clears_it_and_step_resumes(self) -> None:
        """Both paths lead to the same remedy, which is the point of the flag."""
        engine = _engine(construct=_raiser(RuntimeError))
        engine.add_object(name="cube", shape="cuboid", position=[0.0, 0.0, 0.5], size=[0.05] * 3)
        assert engine.step(1)["status"] == "error"

        engine._physics_view_stale = False

        assert engine.step(1)["status"] == "success"


class TestEveryReturnAfterTheViewIsInvalidatedSetsIt:
    """A drift guard derived from the source.

    ``_construct_shape_prim`` is the point after which the view is invalid. Any
    ``return`` below it that does not have the flag set on the way is a new instance
    of this bug, and there were two such returns when only one was covered.
    """

    def test_no_uncovered_return_below_the_construction(self) -> None:
        import ast
        import inspect
        import textwrap

        source = textwrap.dedent(inspect.getsource(IsaacSimulation.add_object))
        lines = source.split("\n")
        construct_at = next(i for i, line in enumerate(lines) if "_construct_shape_prim(" in line)

        uncovered = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Return):
                continue
            idx = node.lineno - 1
            if idx <= construct_at:
                continue
            if "_physics_view_stale = True" not in "\n".join(lines[construct_at:idx]):
                uncovered.append(node.lineno)

        assert uncovered == [], (
            f"return(s) at line offset {uncovered} sit below _construct_shape_prim - which has "
            f"already stopped the timeline - without the scene being marked stale first"
        )
