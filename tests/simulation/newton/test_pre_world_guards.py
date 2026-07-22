"""Pre-world guard contract for the Newton backend.

Every scene-mutating, actuation, and scene-query method on
:class:`~strands_robots.simulation.newton.simulation.NewtonSimEngine` must be
callable before :meth:`create_world` and return a structured tool error
(``{"status": "error", ...}`` carrying a ``"No world"`` hint) rather than
crashing on the ``None`` model/world it would otherwise dereference. This pins
that whole surface in one place so a new public method that forgets the guard
(and would raise ``AttributeError`` on ``self._world``/``self._model``) is
caught. Gated on Newton + Warp being importable.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

pytestmark = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


def _call(sim, method):
    """Invoke ``method`` on a world-less ``sim`` with minimal valid arguments.

    Each call passes only what the signature requires so it reaches the guard,
    which is the first statement of every method under test.
    """
    dispatch = {
        "add_robot": lambda: sim.add_robot("so100"),
        "add_object": lambda: sim.add_object("cube"),
        "move_object": lambda: sim.move_object("cube", position=[0.0, 0.0, 0.0]),
        "list_objects": sim.list_objects,
        "send_action": lambda: sim.send_action({"Rotation": 0.1}),
        "set_timestep": lambda: sim.set_timestep(0.01),
        "add_camera": lambda: sim.add_camera("cam"),
        "remove_camera": lambda: sim.remove_camera("cam"),
        "render": sim.render,
        "list_robots_info": sim.list_robots_info,
        "list_bodies": sim.list_bodies,
        "get_features": sim.get_features,
    }
    return dispatch[method]()


# The full public scene/actuation/query surface that dereferences the model.
PRE_WORLD_METHODS = [
    "add_robot",
    "add_object",
    "move_object",
    "list_objects",
    "send_action",
    "set_timestep",
    "add_camera",
    "remove_camera",
    "render",
    "list_robots_info",
    "list_bodies",
    "get_features",
]


@pytest.fixture
def worldless_engine():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    # No create_world() on purpose: exercise the pre-world guard path.
    return NewtonSimEngine(solver="mujoco")


@pytest.mark.parametrize("method", PRE_WORLD_METHODS)
def test_pre_world_call_returns_structured_error(worldless_engine, method):
    result = _call(worldless_engine, method)

    assert isinstance(result, dict), f"{method} must return a tool-result dict, got {type(result)!r}"
    assert result["status"] == "error", f"{method} must error before a world exists"
    assert "No world" in result["content"][0]["text"], f"{method} must hint that create_world is required"


def test_pre_world_methods_cover_the_declared_surface(worldless_engine):
    # Guards against silent drift: every method named here must exist on the
    # engine, so a rename cannot quietly leave this contract testing nothing.
    for method in PRE_WORLD_METHODS:
        assert callable(getattr(worldless_engine, method)), f"{method} is not a NewtonSimEngine method"
