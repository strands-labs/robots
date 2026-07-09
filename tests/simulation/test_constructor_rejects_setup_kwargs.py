"""Regression: a simulation backend constructor must not silently swallow a
robot-setup keyword argument.

Backend constructors accept ``**kwargs`` purely as a cross-backend
forward-compatibility sink (so one call can carry GPU-backend options like
``num_envs`` / ``device`` that non-GPU backends drop). Before this contract was
pinned, passing ``robot_name`` there - a very natural mistake, since
``robot_name`` is the ubiquitous parameter of ``run_policy`` / ``eval_policy`` /
``get_observation`` / ``send_action`` - was silently dropped. The caller got a
robot-less engine that only failed much later, and far from the cause, with an
unrelated "No world" error. That is the same "success/failure contract, wrong
effect, no signal" footgun the project already fixed for ``add_object``.

These pin the corrected contract:

* the shared ``reject_setup_kwargs`` helper raises ``TypeError`` naming the
  offending argument and pointing at the ``Robot(name, mode="sim")`` factory and
  ``add_robot``;
* genuine forward-compatibility kwargs (``num_envs`` / ``device``) are still
  tolerated and dropped;
* both the direct MuJoCo constructor and the ``create_simulation`` factory that
  forwards to it fail loudly instead of returning a robot-less engine;
* the Newton backend shares the identical contract (gated on the extra).
"""

import pytest

from strands_robots.simulation.base import reject_setup_kwargs


def test_helper_rejects_robot_name() -> None:
    with pytest.raises(TypeError) as exc:
        reject_setup_kwargs({"robot_name": "so101"})
    msg = str(exc.value)
    assert "robot_name" in msg
    # Message must be actionable: point at the factory and add_robot.
    assert "Robot(" in msg
    assert "add_robot" in msg


def test_helper_rejects_robot() -> None:
    with pytest.raises(TypeError, match="robot"):
        reject_setup_kwargs({"robot": "so101"})


def test_helper_reports_all_offending_names() -> None:
    with pytest.raises(TypeError) as exc:
        reject_setup_kwargs({"robot_name": "a", "robot": "b"})
    msg = str(exc.value)
    assert "robot_name" in msg
    assert "'robot'" in msg


def test_helper_ignores_forward_compat_kwargs() -> None:
    """Genuine backend-specific kwargs must pass through untouched (no raise)."""
    reject_setup_kwargs({"num_envs": 4, "device": "cpu"})
    reject_setup_kwargs({})


def test_mujoco_constructor_rejects_robot_name() -> None:
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import Simulation

    with pytest.raises(TypeError) as exc:
        Simulation(robot_name="so101")
    assert "robot_name" in str(exc.value)


def test_create_simulation_factory_rejects_robot_name() -> None:
    pytest.importorskip("mujoco")
    from strands_robots.simulation import create_simulation

    with pytest.raises(TypeError, match="robot_name"):
        create_simulation("mujoco", robot_name="so101")


def test_mujoco_constructor_tolerates_forward_compat_kwargs() -> None:
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import Simulation

    sim = Simulation(num_envs=4, device="cpu", tool_name="fc_probe")
    try:
        assert sim is not None
    finally:
        sim.cleanup()


def test_newton_constructor_rejects_robot_name() -> None:
    pytest.importorskip("newton")
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    with pytest.raises(TypeError, match="robot_name"):
        NewtonSimEngine(robot_name="so101")
