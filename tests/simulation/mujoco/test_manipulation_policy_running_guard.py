"""The attach/actuate/zero manipulation primitives refuse to mutate the model
while a policy is running.

``attach_bodies`` / ``detach_bodies`` / ``actuate_robot`` / ``zero_dynamics``
each mutate ``model``/``data`` - equality constraints, freejoint ``qpos``
writes, a spec recompile that swaps the model pointer, or a bulk ``qvel``/
``qacc`` clear. Doing any of that under a live ``PolicyRunner`` worker stepping
at control frequency is a data race; a recompile in particular swaps
``self._world._model``/``_data`` out from under the worker's cached arrays, so
its next ``mj_step`` reads freed memory. Every primitive therefore calls
``_require_no_running_policy`` before touching physics.

This pins that guard so a later refactor cannot silently drop it on one of the
four methods (mirrors the scene-mutation guard on
``add_robot``/``load_scene``/``replace_scene_mjcf`` and the dynamics guard on
``set_joint_velocities``/``set_geom_properties``). ``attach_bodies`` /
``detach_bodies`` use GLOBAL scope (any running policy blocks them);
``actuate_robot`` / ``zero_dynamics(robot_name=...)`` use PER-ROBOT scope, so
here they are scoped to the arm whose policy is running.
"""

import time

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# A one-joint arm that already carries a position actuator so a mock policy can
# actually drive it (the running-policy state we need is a live worker thread).
ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="50"/>
  </actuator>
</mujoco>
"""

# (method, args) - minimal args; the running-policy guard fires before any of
# them are resolved (before name lookup, mode/param validation, or the
# already-actuated check), so the values only need to be well-typed.
GUARDED_CALLS: list[tuple[str, tuple]] = [
    ("attach_bodies", ("parent_body", "child_body")),
    ("detach_bodies", ("parent_body", "child_body")),
    ("actuate_robot", ("arm",)),
    ("zero_dynamics", ("arm",)),
]


@pytest.fixture
def running_policy_sim(tmp_path):
    """A world with an arm whose mock policy is actively running."""
    arm_path = tmp_path / "arm.xml"
    arm_path.write_text(ARM_XML)
    sim = Simulation(tool_name="manip_policy_running_guard", mesh=False)
    sim.create_world()
    assert sim.add_robot(name="arm", urdf_path=str(arm_path))["status"] == "success"
    started = sim.start_policy("arm", policy_provider="mock", duration=2.0, fast_mode=False)
    assert started["status"] == "success", started
    # The future is registered synchronously; a brief settle keeps the worker
    # live across the assertion window (mirrors the replace_scene guard test).
    time.sleep(0.05)
    fut = sim._policy_threads.get("arm")
    assert fut is not None and not fut.done(), "the arm policy worker must be live for this contract"
    yield sim
    sim.cleanup(policy_stop_timeout=2.0)


@pytest.mark.parametrize(("method", "args"), GUARDED_CALLS, ids=[c[0] for c in GUARDED_CALLS])
def test_manipulation_primitive_blocked_while_policy_running(running_policy_sim, method, args):
    result = getattr(running_policy_sim, method)(*args)
    assert isinstance(result, dict), f"{method} returned non-dict {type(result)}"
    assert result["status"] == "error", f"{method} must refuse to mutate while a policy runs, got {result}"
    text = result["content"][0]["text"].lower()
    assert "policy is running" in text, f"{method} error must name the running-policy cause: {text!r}"


def test_primitive_allowed_again_after_policy_stops(running_policy_sim):
    """The guard is transient: once the policy is stopped, a primitive whose
    other preconditions hold succeeds - the guard is the only thing that was
    blocking it."""
    assert running_policy_sim.stop_policy("arm")["status"] == "success"
    # stop_policy signals the worker; wait for the future to actually finish
    # before the guard will clear.
    deadline = time.time() + 5.0
    fut = running_policy_sim._policy_threads.get("arm")
    while fut is not None and not fut.done() and time.time() < deadline:
        time.sleep(0.02)
    # zero_dynamics on the (now idle) arm has no other precondition to trip.
    result = running_policy_sim.zero_dynamics("arm")
    assert result["status"] == "success", result
