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


def test_wbc_forward_walk_translates_base_without_falling(g1_sim) -> None:  # type: ignore[no-untyped-def]
    """Drive the G1 forward with WBC and assert the base moves forward + stays up.

    This is the deploy-stage end-to-end check: a real ONNX controller stepping
    the real MuJoCo G1. We command a modest forward velocity and verify the
    base translates in +x by at least ``WBC_MIN_FORWARD_M`` while its height
    does not collapse (a fall would drop z well below the standing height).
    """
    sim = g1_sim
    policy = create_policy("wbc", checkpoint=CHECKPOINT, walk=True)

    # Confirm the sim exposes every WBC leg+waist joint BY NAME before stepping
    # (set_robot_state_keys resolves by name, not position - the real sim list
    # leads with 'floating_base_joint' and interleaves arm joints, so a
    # positional [:15] check would be wrong). Asserting here gives a clearer
    # integration-failure message than a deep stack trace.
    joint_names = sim.robot_joint_names("unitree_g1")
    missing = [j for j in WBC_G1_LEG_WAIST_JOINTS if j not in joint_names]
    assert not missing, f"unitree_g1 sim is missing WBC leg+waist joints by name: {missing}"

    def _base_xz() -> tuple[float, float]:
        state = sim.get_body_state("unitree_g1/pelvis")
        # get_body_state returns {"status","content":[{text},{json:{position,...}}]}.
        # Find the json block (it is not necessarily first - a human-readable
        # text block precedes it).
        pos = None
        for blk in state.get("content", []):
            if "json" in blk and "position" in blk["json"]:
                pos = blk["json"]["position"]
                break
        assert pos is not None, f"get_body_state returned no position json block: {state}"
        return float(pos[0]), float(pos[2])

    x0, z0 = _base_xz()

    result = sim.run_policy(
        robot_name="unitree_g1",
        policy_object=policy,
        instruction="walk forward",
        policy_kwargs={"target_velocity": [0.5, 0.0, 0.0]},
        duration=4.0,
        control_frequency=50.0,
        action_horizon=1,  # WBC is closed-loop per tick
        fast_mode=True,
    )
    assert result["status"] == "success", result

    x1, z1 = _base_xz()
    forward = x1 - x0
    assert forward >= MIN_FORWARD_M, f"base advanced only {forward:.3f} m (< {MIN_FORWARD_M} m); gait may be unstable"
    # A fall drops the pelvis far below its standing height; allow a small dip.
    assert z1 > 0.5 * z0, f"base height collapsed from {z0:.3f} to {z1:.3f} m - robot likely fell"


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
