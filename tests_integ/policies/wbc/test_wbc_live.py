"""Integration test for :class:`WBCPolicy` against a real ONNX checkpoint.

Steps the Unitree G1 in MuJoCo under the GR00T Whole-Body-Control (SONIC)
policy for a short forward walk and asserts the base translates forward
without falling - the deploy-stage acceptance criterion from issue #466.

Gated behind the ``wbc`` pytest marker AND a ``WBC_LIVE=1`` env flag, and
skips cleanly if ``onnxruntime`` / ``mujoco`` are absent or no checkpoint is
configured. Enable with:

.. code-block:: bash

    pip install 'strands-robots[wbc,sim-mujoco]'
    # Download a GR00T-WBC (SONIC) checkpoint under the NVIDIA Open Model
    # License, e.g. from https://huggingface.co/nvidia/GEAR-SONIC, into a dir
    # containing policy.onnx (+ optional walk_policy.onnx + config.json).
    export WBC_CHECKPOINT=/path/to/GEAR-SONIC
    WBC_LIVE=1 hatch run test-integ tests_integ/policies/wbc/ -m wbc -v

No weights are bundled. The exact base-displacement threshold is configurable
via ``WBC_MIN_FORWARD_M`` (default 0.05 m over the rollout) so a checkpoint
with a different gait speed can still pass.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Skip cleanly when the optional deps are missing, rather than erroring at
# collection. ``importorskip`` emits a SKIPPED line with a clear reason.
pytest.importorskip("onnxruntime", reason="onnxruntime not installed - pip install 'strands-robots[wbc]'")
pytest.importorskip("mujoco", reason="mujoco not installed - pip install 'strands-robots[sim-mujoco]'")

LIVE = os.environ.get("WBC_LIVE", "").lower() in ("1", "true", "yes")
CHECKPOINT = os.environ.get("WBC_CHECKPOINT", "")
MIN_FORWARD_M = float(os.environ.get("WBC_MIN_FORWARD_M", "0.05"))

pytestmark = [
    pytest.mark.wbc,
    pytest.mark.skipif(
        not (LIVE and CHECKPOINT),
        reason=(
            "Requires onnxruntime + a downloaded GR00T-WBC (SONIC) checkpoint. "
            "Set WBC_LIVE=1 and WBC_CHECKPOINT=/path/to/GEAR-SONIC to enable."
        ),
    ),
]

# E402: importorskip must run before these imports so the skip is clean.
from strands_robots import Robot  # noqa: E402
from strands_robots.policies import create_policy  # noqa: E402
from strands_robots.policies.wbc import WBC_G1_LEG_WAIST_JOINTS, WBCPolicy  # noqa: E402


@pytest.fixture()
def g1_sim():  # type: ignore[no-untyped-def]
    """Spin up an MuJoCo Unitree G1 (sim mode, mesh off for a hermetic test)."""
    sim = Robot("unitree_g1", mesh=False)
    try:
        yield sim
    finally:
        sim.destroy()


def test_wbc_policy_loads_real_onnx() -> None:
    """The factory builds a WBCPolicy whose ONNX sessions load from the checkpoint."""
    policy = create_policy("wbc", checkpoint=CHECKPOINT, walk=True)
    assert isinstance(policy, WBCPolicy)
    assert policy.requires_images is False
    assert policy.policy_session is not None, "main ONNX session must load from the checkpoint"


def _load_harness():  # type: ignore[no-untyped-def]
    """Import the torque-deploy harness (examples/ is not an installed package)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "examples" / "wbc_g1_torque_deploy.py"
    spec = importlib.util.spec_from_file_location("wbc_g1_torque_deploy", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(
    os.environ.get("WBC_GAIT_CHECK", "").lower() not in ("1", "true", "yes"),
    reason=(
        "Gait-quality check runs the full torque-control deploy loop (PD->torque via "
        "compute_torques on a torque-actuator G1 at control_decimation=4) with the real "
        "SONIC weights - the deploy path, not sim.run_policy's position-servo path. It is "
        "compute-heavy and needs the real checkpoint, so it is opt-in. Set WBC_GAIT_CHECK=1 "
        "(with WBC_CHECKPOINT) to run it. The harness lives at examples/wbc_g1_torque_deploy.py."
    ),
)
def test_wbc_forward_walk_translates_base_without_falling() -> None:
    """Deploy-fidelity gait check via the torque-control harness: with the real
    weights the G1 must walk forward without falling.

    Drives the SAME ``simulate_rollout`` loop the example/CLI uses (torque PD on
    the 15 controlled joints, arms held, whole-body observation with real joint
    velocities + base IMU). Asserts the base advances >= WBC_MIN_FORWARD_M in +x
    while its height does not collapse.
    """
    harness = _load_harness()
    policy = create_policy("wbc", checkpoint=CHECKPOINT, walk=True)
    assert isinstance(policy, WBCPolicy)

    result = harness.simulate_rollout(policy, vx=0.5, vy=0.0, omega=0.0, duration=4.0)

    assert not result["fell"], f"robot fell (z {result['z0']:.3f} -> {result['z1']:.3f} m) during the rollout"
    assert result["z1"] > 0.5 * result["z0"], f"base height collapsed from {result['z0']:.3f} to {result['z1']:.3f} m"
    assert result["forward"] >= MIN_FORWARD_M, (
        f"base advanced only {result['forward']:.3f} m (< {MIN_FORWARD_M} m); gait may be unstable"
    )


def test_wbc_standing_balance_holds_in_place() -> None:
    """With a zero command the real policy should HOLD BALANCE (height steady,
    little drift) - a balance check distinct from the forward-walk one."""
    if os.environ.get("WBC_GAIT_CHECK", "").lower() not in ("1", "true", "yes"):
        pytest.skip("opt-in gait/balance check; set WBC_GAIT_CHECK=1")
    harness = _load_harness()
    policy = create_policy("wbc", checkpoint=CHECKPOINT, walk=True)
    result = harness.simulate_rollout(policy, vx=0.0, vy=0.0, omega=0.0, duration=3.0)
    assert not result["fell"], "robot fell while standing"
    assert result["z1"] > 0.7 * result["z0"], f"standing height collapsed: {result['z0']:.3f} -> {result['z1']:.3f}"


def test_wbc_action_shape_is_15dim_on_real_model(g1_sim) -> None:  # type: ignore[no-untyped-def]
    """One real inference step returns the 15 leg+waist targets by name."""
    sim = g1_sim
    policy = create_policy("wbc", checkpoint=CHECKPOINT, walk=True)
    policy.set_robot_state_keys(sim.robot_joint_names("unitree_g1"))

    obs = sim.get_observation("unitree_g1")
    actions = policy.get_actions_sync(obs, "", target_velocity=[0.3, 0.0, 0.0])

    assert len(actions) == 1
    assert set(actions[0].keys()) == set(WBC_G1_LEG_WAIST_JOINTS)
    assert all(np.isfinite(v) for v in actions[0].values())


def test_real_onnx_io_dims_match_config() -> None:
    """The real ONNX sessions' input/output dims must match the resolved config.

    Pins the single most important real-weights fact: the shipped GR00T-WBC
    ONNX (e.g. GR00T-WholeBodyControl-Balance.onnx) has input ``[batch, num_obs]``
    (516 for the default obs_history_len=6) and output ``[batch, num_actions]``
    (15). A config whose num_obs disagrees with the checkpoint would feed the
    model a wrong-width observation - this catches that mismatch up front."""
    policy = create_policy("wbc", checkpoint=CHECKPOINT, walk=True)
    assert isinstance(policy, WBCPolicy)
    sess = policy.policy_session
    in_shape = sess.get_inputs()[0].shape  # e.g. ['batch_size', 516]
    out_shape = sess.get_outputs()[0].shape  # e.g. ['batch_size', 15]
    assert in_shape[-1] == policy.config.num_obs, (
        f"ONNX input width {in_shape[-1]} != config.num_obs {policy.config.num_obs}; "
        "the checkpoint and config disagree on the observation dimension "
        "(check obs_history_len / single_obs_dim)."
    )
    assert out_shape[-1] == policy.config.num_actions, (
        f"ONNX output width {out_shape[-1]} != config.num_actions {policy.config.num_actions}."
    )
