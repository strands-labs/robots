"""Physics-state mutations are refused while a policy is running.

The scene-topology mutations (``add_object`` / ``add_robot`` / ``load_scene`` /
...) already hard-fail while a ``PolicyRunner`` worker is live, because an XML
round-trip swaps ``self._world._model`` / ``_data`` under the worker's cached
pointers. The *in-place* physics-state mutations have the same hazard for a
different reason: ``set_joint_velocities`` writes ``data.qvel[...]`` and
``set_geom_properties`` mutates ``model.geom_*`` while the worker thread is
mid ``mj_step`` on the very same arrays. Both therefore route through the shared
``_require_no_running_policy`` guard (global scope) and must return the uniform
"policy is running" error rather than racing the worker.

The sibling state mutations (``set_joint_positions`` / ``set_body_properties`` /
``apply_force``) are pinned elsewhere; these two closed the remaining gap so the
whole state-mutation family is proven to observe the guard, and the guard cannot
regress on just one method.
"""

pytest = __import__("pytest")
pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Minimal single-DOF arm with a named worldbody geom so the post-stop
# set_geom_properties call has a concrete target to recolor.
ROBOT_XML = """
<mujoco model="guard_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def robot_path(tmp_path):
    path = tmp_path / "guard_arm.xml"
    path.write_text(ROBOT_XML)
    return str(path)


@pytest.fixture
def sim_with_running_policy(robot_path):
    """A live world + robot with a mock policy running (fast_mode)."""
    sim = Simulation(tool_name="test_state_guard", mesh=False)
    assert sim.create_world(gravity=[0, 0, -9.81])["status"] == "success"
    assert sim.add_robot("arm1", urdf_path=robot_path)["status"] == "success"
    assert sim.start_policy("arm1", policy_provider="mock", duration=2.0, fast_mode=True)["status"] == "success"
    yield sim
    sim.cleanup(policy_stop_timeout=2.0)


def _stop_and_await(sim) -> None:
    sim.stop_policy("arm1")
    fut = sim._policy_threads.get("arm1")
    if fut is not None:
        try:
            fut.result(timeout=10.0)
        except Exception:
            # Worker stop/timeout errors are irrelevant to the mutation guard
            # under test; swallow so teardown does not mask the assertion.
            pass


def test_set_joint_velocities_blocked_during_policy(sim_with_running_policy):
    """set_joint_velocities writes data.qvel and so is refused mid-policy;
    it succeeds once the worker has stopped."""
    sim = sim_with_running_policy

    result = sim.set_joint_velocities(velocities={"shoulder_pan": 0.5}, robot_name="arm1")
    assert result["status"] == "error"
    assert "policy is running" in result["content"][0]["text"].lower()

    _stop_and_await(sim)

    result = sim.set_joint_velocities(velocities={"shoulder_pan": 0.5}, robot_name="arm1")
    assert result["status"] == "success", result


def test_set_geom_properties_blocked_during_policy(sim_with_running_policy):
    """set_geom_properties mutates model.geom_* and so is refused mid-policy;
    it succeeds once the worker has stopped."""
    sim = sim_with_running_policy

    result = sim.set_geom_properties(geom_name="ground", color=[1.0, 0.0, 0.0, 1.0])
    assert result["status"] == "error"
    assert "policy is running" in result["content"][0]["text"].lower()

    _stop_and_await(sim)

    result = sim.set_geom_properties(geom_name="ground", color=[1.0, 0.0, 0.0, 1.0])
    assert result["status"] == "success", result


def test_state_mutation_guard_is_global_scope(sim_with_running_policy):
    """The guard is global scope: a policy on arm1 blocks a state mutation
    that names arm1 too, and the error identifies the live robot so the agent
    can stop it without guessing."""
    sim = sim_with_running_policy

    result = sim.set_joint_velocities(velocities=[0.5], robot_name="arm1")
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "arm1" in text
    assert "stop_policy" in text.lower()
