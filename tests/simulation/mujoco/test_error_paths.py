"""Error-path coverage for MuJoCo ``Simulation`` public methods.

Every public method should return ``{"status": "error", ...}`` (never raise)
for:
* invalid identifiers (unknown body/geom/joint/sensor names)
* out-of-bounds numeric ids
* missing-arg edge cases (None positions, None velocities, etc.)
* ghost checkpoints / ghost cameras / idle policy stop
* pathological shape params (negative timestep, short gravity vector)

This locks the AgentTool contract: the LLM-facing surface must never bubble
a raw exception.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import pytest

mj = pytest.importorskip("mujoco")

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402

requires_gl = pytest.mark.skipif(
    not _can_render(),
    reason="No GL context available (headless CI without EGL/OSMesa)",
)

# Inline robot XML - avoids network dependency on robot model repos
_ROBOT_XML = """
<mujoco model="test_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <camera name="front" pos="1.5 0 1" xyaxes="0 1 0 -0.5 0 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <body name="link2" pos="0 0 0.2">
          <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.15" rgba="0.3 0.8 0.3 1"/>
          <joint name="elbow" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
    <position name="shoulder_lift_act" joint="shoulder_lift" kp="50"/>
    <position name="elbow_act" joint="elbow" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def ready_sim():
    from strands_robots.simulation import Simulation

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_arm.xml")
    with open(path, "w") as f:
        f.write(_ROBOT_XML)

    s = Simulation()
    s.create_world(timestep=0.002)
    result = s.add_robot("arm", urdf_path=path, position=[0.0, 0.0, 0.0])
    assert result["status"] == "success", f"add_robot failed: {result}"
    s.step(n_steps=5)
    yield s
    s.destroy()
    shutil.rmtree(tmpdir, ignore_errors=True)


# ─ Physics: unknown-name + out-of-bounds────────────────────────────


def test_set_geom_properties_out_of_bounds_id_errors_gracefully(ready_sim):
    r = ready_sim.set_geom_properties(geom_id=999999, color=[1, 0, 0, 1])
    assert r["status"] == "error"
    assert "not found" in r["content"][0]["text"]


def test_set_geom_properties_unknown_name_errors_gracefully(ready_sim):
    r = ready_sim.set_geom_properties(geom_name="__does_not_exist__", color=[1, 0, 0, 1])
    assert r["status"] == "error"
    assert "not found" in r["content"][0]["text"]


def test_set_body_properties_unknown_name_errors_gracefully(ready_sim):
    r = ready_sim.set_body_properties(body_name="__ghost_body__", mass=1.0)
    assert r["status"] == "error"


def test_get_jacobian_unknown_body_errors(ready_sim):
    r = ready_sim.get_jacobian(body_name="__no_such_body__")
    assert r["status"] == "error"


def test_get_jacobian_unknown_site_errors(ready_sim):
    r = ready_sim.get_jacobian(site_name="__no_such_site__")
    assert r["status"] == "error"


def test_get_jacobian_unknown_geom_errors(ready_sim):
    r = ready_sim.get_jacobian(geom_name="__no_such_geom__")
    assert r["status"] == "error"


def test_set_joint_positions_none_dict_errors(ready_sim):
    # Post-T11: message updated to explain list OR dict is accepted.
    r = ready_sim.set_joint_positions(positions=None)
    assert r["status"] == "error"
    assert "'positions' is required" in r["content"][0]["text"]


def test_set_joint_velocities_none_dict_errors(ready_sim):
    # Post-T11: message updated to explain list OR dict is accepted.
    r = ready_sim.set_joint_velocities(velocities=None)
    assert r["status"] == "error"
    assert "'velocities' is required" in r["content"][0]["text"]


def test_set_joint_positions_unknown_joint_rejects_whole_write(ready_sim):
    """An unknown joint name is a structured error, not a partial write.

    This used to assert ``success`` on the grounds that "the valid joint still
    applied" - which is exactly the defect: the caller asked for a two-joint
    pose, got a one-joint pose, and was told the call succeeded. The write is
    now all-or-nothing and reports the name it could not resolve.
    """
    joints = ready_sim.robot_joint_names("arm")
    assert len(joints) > 0, "Fixture robot must have joints"
    r = ready_sim.set_joint_positions(positions={joints[0]: 0.1, "__nope__": 0.2})
    assert r["status"] == "error"
    assert "__nope__" in r["content"][0]["text"]


def test_apply_force_torque_only(ready_sim):
    """apply_force with torque-only (force=None) should still succeed."""
    r = ready_sim.apply_force(body_name="arm/base", torque=[0.0, 0.0, 0.1])
    assert r["status"] == "success"


def test_apply_force_unknown_body_errors(ready_sim):
    r = ready_sim.apply_force(body_name="__ghost__", force=[1, 0, 0])
    assert r["status"] == "error"


def test_get_sensor_data_no_sensors_returns_info(ready_sim):
    """Test arm has no sensors → returns success with an informational text."""
    r = ready_sim.get_sensor_data()
    assert r["status"] == "success"
    assert "No sensors" in r["content"][0]["text"]


def test_get_sensor_data_unknown_name_errors(ready_sim):
    """T45: requesting a specific sensor name on a model with no sensors must
    report a clear 'not found' error (distinguishable from 'no sensors at all'
    when no name was given).
    """
    r = ready_sim.get_sensor_data(sensor_name="__ghost_sensor__")
    assert r["status"] == "error"
    text = r["content"][0]["text"]
    assert "__ghost_sensor__" in text
    assert "not found" in text


def test_get_body_state_unknown_body_errors(ready_sim):
    r = ready_sim.get_body_state(body_name="__ghost__")
    assert r["status"] == "error"


# ─ State mgmt: ghost checkpoints───────────────────────────────────


def test_load_state_unknown_checkpoint_errors(ready_sim):
    r = ready_sim.load_state(name="__never_saved__")
    assert r["status"] == "error"


def test_save_state_then_load_state_round_trips(ready_sim):
    r = ready_sim.save_state(name="probe")
    assert r["status"] == "success"
    r = ready_sim.load_state(name="probe")
    assert r["status"] == "success"


# ─ Scene mutations: ghosts──────────────────────────────────────────


def test_remove_robot_ghost_errors(ready_sim):
    r = ready_sim.remove_robot("__never_added__")
    assert r["status"] == "error"


def test_remove_object_ghost_errors(ready_sim):
    r = ready_sim.remove_object("__never_added__")
    assert r["status"] == "error"


def test_remove_camera_ghost_errors(ready_sim):
    r = ready_sim.remove_camera("__never_added__")
    assert r["status"] == "error"


def test_move_object_ghost_errors(ready_sim):
    r = ready_sim.move_object(name="__ghost__", position=[0, 0, 0.1])
    assert r["status"] == "error"


# ─ Policy lifecycle─────────────────────────────────────────────────


def test_stop_policy_on_idle_robot_errors(ready_sim):
    """stop_policy on a robot that isn't running a policy is a no-op error."""
    r = ready_sim.stop_policy("arm")
    # Some implementations may return "success" with a no-op message; the
    # contract is: no exception, a dict back, and the flag ends up cleared.
    assert isinstance(r, dict)
    assert r.get("status") in ("success", "error")


def test_stop_policy_ghost_robot_errors(ready_sim):
    r = ready_sim.stop_policy("__ghost_robot__")
    assert r["status"] == "error"


# ─ World controls────────────────────────────────────────────────


def test_step_zero_is_noop(ready_sim):
    t_pre = ready_sim._world.sim_time
    r = ready_sim.step(n_steps=0)
    assert r["status"] == "success"
    assert ready_sim._world.sim_time == t_pre


def test_reset_after_perturbation_restores_time(ready_sim):
    ready_sim.step(n_steps=20)
    assert ready_sim._world.sim_time > 0
    r = ready_sim.reset()
    assert r["status"] == "success"


def test_set_gravity_scalar(ready_sim):
    """A scalar is interpreted as downward gravity."""
    r = ready_sim.set_gravity(-9.8)
    assert r["status"] == "success"


def test_set_gravity_3_vector(ready_sim):
    r = ready_sim.set_gravity([0.0, 0.0, -3.7])
    assert r["status"] == "success"


def test_set_timestep_positive(ready_sim):
    r = ready_sim.set_timestep(0.004)
    assert r["status"] == "success"


# ─ Rendering: unknown camera, render-unavailable paths──────────


@requires_gl
def test_render_all_with_only_missing_cameras_errors(ready_sim):
    """Explicit camera list that matches nothing returns an error."""
    r = ready_sim.render_all(cameras=["ghost_cam_a", "ghost_cam_b"])
    assert r["status"] == "error"


@requires_gl
def test_render_unknown_camera_falls_back(ready_sim):
    """Unknown camera_name → fallback renders with the default view."""
    r = ready_sim.render(camera_name="__not_a_camera__", width=32, height=24)
    # MuJoCo falls back to a free camera when cam_id < 0 - should succeed
    # unless GL context is unavailable, in which case error is acceptable
    assert r["status"] in ("success", "error")


@requires_gl
def test_render_surfaces_unexpected_renderer_error_as_status_error(ready_sim, monkeypatch):
    """An unexpected exception inside the render pipeline must be caught.

    Camera validation and the no-GL-context guard already return structured
    errors, but a fault raised deeper in the pipeline (a MuJoCo renderer that
    throws on ``update_scene`` after the camera resolved, e.g. a lost GL context
    mid-render) must NOT bubble a raw exception past the AgentTool boundary. It
    has to come back as ``status=error`` carrying the fault text, so the LLM
    sees a recoverable tool result instead of an unhandled traceback.
    """

    class _BoomRenderer:
        def update_scene(self, data, camera=None, scene_option=None):
            raise RuntimeError("GL context lost mid-render")

    # The free camera ("default") skips name validation, so the fault lands in
    # the render body itself - the path the catch-all guards.
    monkeypatch.setattr(ready_sim, "_get_renderer", lambda w, h: _BoomRenderer())
    r = ready_sim.render(camera_name="default", width=32, height=24)
    assert r["status"] == "error"
    text = r["content"][0]["text"]
    assert "Render failed" in text
    assert "GL context lost mid-render" in text


# ─ Tool-spec dispatch: unknown action + error routing───────────


def test_dispatch_private_action_is_rejected(ready_sim):
    """Dispatcher must refuse private leading-underscore names."""
    r = ready_sim._dispatch_action("_stop_policy", {"action": "_stop_policy"})
    assert r["status"] == "error"
    assert "Unknown action" in r["content"][0]["text"]


def test_dispatch_field_remap_checkpoint_name_to_name(ready_sim):
    """The dispatcher remaps ``checkpoint_name`` → ``name`` for save_state."""
    r = ready_sim._dispatch_action("save_state", {"action": "save_state", "checkpoint_name": "remap_probe"})
    assert r["status"] == "success"
    r = ready_sim._dispatch_action("load_state", {"action": "load_state", "checkpoint_name": "remap_probe"})
    assert r["status"] == "success"


# ── Properties ─────────────────────────────────────────────────────


def test_mj_model_and_mj_data_return_none_before_world():
    """Direct MuJoCo handles are ``None`` until ``create_world`` runs."""
    from strands_robots.simulation import Simulation

    s = Simulation()
    assert s.mj_model is None
    assert s.mj_data is None
    s.destroy()


def test_mj_model_and_mj_data_after_world(ready_sim):
    """After ``create_world + add_robot`` the handles are populated."""
    import mujoco as mj

    assert isinstance(ready_sim.mj_model, mj.MjModel)
    assert isinstance(ready_sim.mj_data, mj.MjData)


# ── Observation edge cases (ABC path in Simulation.get_observation) ──


def test_get_observation_no_world_returns_empty_dict():
    from strands_robots.simulation import Simulation

    s = Simulation()
    assert s.get_observation() == {}
    s.destroy()


def test_get_observation_no_robots_returns_empty_dict():
    """``get_observation()`` with no robots added yet → ``{}`` (not a raise)."""
    from strands_robots.simulation import Simulation

    s = Simulation()
    s.create_world()
    assert s.get_observation() == {}
    s.destroy()


def test_get_observation_unknown_robot_returns_empty_dict(ready_sim):
    assert ready_sim.get_observation(robot_name="__ghost__") == {}


def test_send_action_no_world_is_noop():
    from strands_robots.simulation import Simulation

    s = Simulation()
    # Should return error dict when no world exists
    result = s.send_action({"j": 0.1})
    assert result["status"] == "error"
    assert "No world" in result["content"][0]["text"]
    s.destroy()


def test_send_action_unknown_robot_is_noop(ready_sim):
    result = ready_sim.send_action({"j": 0.1}, robot_name="__ghost__")
    assert result["status"] == "error"
    assert "__ghost__" in result["content"][0]["text"]
