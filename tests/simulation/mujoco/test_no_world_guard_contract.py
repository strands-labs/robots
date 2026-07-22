"""Unified "no world" guard contract for the MuJoCo Simulation facade.

Every world-touching facade/mixin method must, when called before
``create_world`` (or after a failed ``load_scene`` that leaves a partial
world), return the same structured error - never raise and never drift the
wording. The canonical text lives in a single shared constant,
``strands_robots.simulation.mujoco.backend._NO_WORLD_MSG`` (re-exported from
``...mujoco.simulation`` for the facade); this module pins that single string
across every guarded method - the high-level facade methods AND the dynamics,
randomization, rendering, and recording mixin methods - so an agent that learns
the error from one action recognises it from all of them. Because every method
sources the one constant, editing the message in one place can never silently
leave a hand-rolled copy behind in a mixin.

Two states are exercised:

* **No world** - a fresh ``Simulation`` with ``_world is None``.
* **Partial world** - ``_world`` set but ``_model``/``_data`` still ``None``.
  This is reachable in production: ``load_scene`` assigns ``self._world =
  SimWorld()`` before compiling the spec, so a compile failure leaves the
  partial handle behind. Methods that only checked ``_world is None`` would
  pass that guard and then dereference a ``None`` model.
"""

import pytest

from strands_robots.simulation.models import SimWorld
from strands_robots.simulation.mujoco.simulation import _NO_WORLD_MSG, Simulation

# (method_name, args) for every facade method that guards on a live world.
# Args are the minimum required positional/keyword params; the guard fires
# before any of them are used.
GUARDED_CALLS: list[tuple[str, tuple, dict]] = [
    ("send_action", ({},), {}),
    ("replace_scene_mjcf", ("<mujoco/>",), {}),
    ("patch_scene_mjcf", ([],), {}),
    ("add_robot", ("r",), {}),
    ("list_robots_info", (), {}),
    ("get_robot_state", (), {}),
    ("add_object", ("o",), {}),
    ("move_object", ("o",), {}),
    ("remove_object", ("o",), {}),
    ("list_objects", (), {}),
    ("add_camera", ("c",), {}),
    ("remove_camera", ("c",), {}),
    ("list_cameras_info", (), {}),
    ("step", (), {}),
    ("reset", (), {}),
    ("get_state", (), {}),
    ("set_gravity", ([0.0, 0.0, -1.0],), {}),
    ("set_timestep", (0.004,), {}),
    ("get_features", (), {}),
    ("start_policy", (), {}),
    ("run_policy", (), {}),
    ("run_multi_policy", ({},), {}),
    # PhysicsMixin dynamics/state methods - same unified guard.
    ("save_state", (), {}),
    ("load_state", (), {}),
    ("apply_force", ("base",), {}),
    ("raycast", ([0.0, 0.0, 1.0], [0.0, 0.0, -1.0]), {}),
    ("get_jacobian", (), {}),
    ("get_energy", (), {}),
    ("get_mass_matrix", (), {}),
    ("inverse_dynamics", (), {}),
    ("get_body_state", ("base",), {}),
    ("set_joint_positions", (), {}),
    ("set_joint_velocities", (), {}),
    ("get_sensor_data", (), {}),
    ("set_body_properties", ("base",), {}),
    ("set_geom_properties", (), {}),
    ("get_contact_forces", (), {}),
    ("multi_raycast", ([0.0, 0.0, 1.0], [[0.0, 0.0, -1.0]]), {}),
    ("forward_kinematics", (), {}),
    ("get_total_mass", (), {}),
    ("export_xml", (), {}),
    # ManipulationMixin - attach/actuate/zero primitives (GH #1533).
    ("attach_bodies", ("parent", "child"), {}),
    ("detach_bodies", ("parent", "child"), {}),
    ("actuate_robot", ("arm",), {}),
    ("zero_dynamics", (), {}),
    # RandomizationMixin.
    ("randomize", (), {}),
    # RenderingMixin - render + contact-query + camera-recording entry points.
    ("render", (), {}),
    ("render_depth", (), {}),
    ("get_contacts", (), {}),
    ("render_all", (), {}),
    ("start_cameras_recording", (), {}),
    ("start_cameras_recording_synchronous", (), {}),
    # RecordingMixin - LeRobotDataset recording (guard fires before the
    # lerobot extra is touched).
    ("start_recording", (), {}),
]


def _assert_no_world_error(result: dict, method: str) -> None:
    assert isinstance(result, dict), f"{method} returned non-dict {type(result)}"
    assert result.get("status") == "error", f"{method} should error with no world, got {result.get('status')}"
    text = result["content"][0]["text"]
    assert text == _NO_WORLD_MSG, f"{method} drifted from the unified guard message: {text!r}"


@pytest.fixture
def no_world_sim():
    s = Simulation(tool_name="no_world_contract", mesh=False)
    yield s
    s.cleanup()


@pytest.fixture
def partial_world_sim():
    """A sim whose _world is set but model/data are unbuilt (failed-load state)."""
    s = Simulation(tool_name="partial_world_contract", mesh=False)
    s._world = SimWorld()  # _model / _data left as None
    yield s
    s._world = None
    s.cleanup()


@pytest.mark.parametrize(("method", "args", "kwargs"), GUARDED_CALLS, ids=[c[0] for c in GUARDED_CALLS])
def test_no_world_returns_unified_error(no_world_sim, method, args, kwargs):
    """With no world at all, every guarded method returns the exact unified text."""
    result = getattr(no_world_sim, method)(*args, **kwargs)
    _assert_no_world_error(result, method)


@pytest.mark.parametrize(("method", "args", "kwargs"), GUARDED_CALLS, ids=[c[0] for c in GUARDED_CALLS])
def test_partial_world_returns_unified_error(partial_world_sim, method, args, kwargs):
    """A partial world (model/data still None) must NOT slip past the guard.

    Pre-fix, ``send_action``/``replace_scene_mjcf``/``patch_scene_mjcf``/
    ``add_robot`` only checked ``_world is None`` (or ``_model`` alone) and so
    accepted this state, then either crashed or silently recovered. They must
    now report the standard no-world error like every other guarded method.
    """
    result = getattr(partial_world_sim, method)(*args, **kwargs)
    _assert_no_world_error(result, method)


def test_require_world_helper_is_wired_to_the_constant(no_world_sim):
    """The canonical _require_world() helper returns the same unified text."""
    err = no_world_sim._require_world()
    assert err is not None
    assert err["content"][0]["text"] == _NO_WORLD_MSG


def test_require_world_passes_when_world_live():
    """Once a world exists, the guard helper returns None (live)."""
    s = Simulation(tool_name="live_world_contract", mesh=False)
    try:
        s.create_world()
        assert s._require_world() is None
    finally:
        s.cleanup()
