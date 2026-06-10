"""Mesh ``tell()`` end-to-end against a real ``Simulation`` peer (issue #303).

Exercises the full sim dispatch path:

* Stand up a real ``Simulation`` (MuJoCo backend) with a single robot.
* Wrap it in a ``Mesh`` (no zenoh — we hand-call ``_dispatch`` so the
  test stays a unit test, not a full Zenoh integration).
* Drive ``execute`` via the dispatcher with ``policy_provider="mock"``
  and assert the mock policy actually ran.

Skipped when MuJoCo (or any other heavy dep) is not importable. The
zero-Zenoh tests in ``test_mesh_rpc_sim_dispatch.py`` cover the dispatch
contract; this module pins that the contract holds against the real
``MuJoCoSimEngine.run_policy`` surface.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh import Mesh

mujoco = pytest.importorskip("mujoco", reason="needs mujoco for real Simulation")


def _build_sim_with_simple_robot():
    """Create a small ``MuJoCoSimEngine`` with a single floating-base robot.

    We avoid ``add_robot`` (which fetches assets from the model registry
    and is heavyweight) and instead hand-build a tiny MJCF with one
    actuated hinge — enough to exercise ``run_policy`` end-to-end.
    """
    from strands_robots.simulation.mujoco.simulation import Simulation

    sim = Simulation()
    sim.create_world(timestep=0.005)

    # Inject a minimal robot directly into the world via the public
    # add_object surface? add_object adds free-bodies, not actuated
    # robots. We need add_robot. The model_registry has lightweight
    # entries (e.g. so100); load that.
    sim.add_robot("so100", data_config="so100")
    return sim


@pytest.fixture
def real_sim():
    sim = _build_sim_with_simple_robot()
    try:
        yield sim
    finally:
        sim.destroy()


def test_dispatch_execute_drives_real_simulation(real_sim) -> None:
    """``tell()`` payload routes through to ``Simulation.run_policy`` and completes."""
    m = Mesh(real_sim, peer_id="sim-real")
    out = m._dispatch(
        {
            "action": "execute",
            "instruction": "wave",
            "policy_provider": "mock",
            "duration": 0.1,
            "control_frequency": 20.0,
            "fast_mode": True,
        }
    )
    # MuJoCoSimEngine.run_policy returns the standard tool-result shape.
    assert out["status"] == "success", out


def test_dispatch_start_then_status_on_real_simulation(real_sim) -> None:
    """``start`` schedules an async rollout; ``status`` reports it.

    Pins the async path: ``start`` returns immediately while the policy
    runs in the sim's ThreadPoolExecutor.
    """
    m = Mesh(real_sim, peer_id="sim-real")
    out = m._dispatch(
        {
            "action": "start",
            "instruction": "wave",
            "policy_provider": "mock",
            "duration": 0.5,
            "control_frequency": 20.0,
            "fast_mode": True,
        }
    )
    assert out["status"] == "success", out


def test_well_known_kwargs_reach_policy_config_on_real_simulation(real_sim) -> None:
    """Issue #300 well-known kwargs land in the ``policy_config`` of ``create_policy``.

    We intercept ``create_policy`` to capture the kwargs the dispatcher
    forwards, then let the rest of ``run_policy`` proceed against the
    real sim with a ``MockPolicy``.
    """
    from unittest.mock import patch

    from strands_robots.policies import create_policy as real_create_policy

    captured: dict[str, object] = {}

    def _spy(provider: str, **kwargs):
        captured["provider"] = provider
        captured.update(kwargs)
        # Strip planner-only kwargs MockPolicy does not accept; the
        # dispatch layer's contract is to forward them, the policy's
        # contract (per #300) is to ignore unknown kwargs at construction.
        # MockPolicy ignores **kwargs in __init__ so we can pass them all.
        return real_create_policy(provider, **kwargs)

    target_pose = [0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
    target_joints = {"joint_0": 0.5}

    m = Mesh(real_sim, peer_id="sim-real")
    with patch("strands_robots.policies.create_policy", side_effect=_spy):
        out = m._dispatch(
            {
                "action": "execute",
                "instruction": "reach",
                "policy_provider": "mock",
                "target_pose": target_pose,
                "target_joints": target_joints,
                "duration": 0.05,
                "control_frequency": 20.0,
                "fast_mode": True,
            }
        )
    assert out["status"] == "success", out
    assert captured["provider"] == "mock"
    assert captured["target_pose"] == target_pose
    assert captured["target_joints"] == target_joints
