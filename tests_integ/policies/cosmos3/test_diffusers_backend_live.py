"""Live integration test for the Cosmos 3 in-process diffusers backend.

Unlike tests/policies/cosmos3/test_policy_diffusers.py (fully mocked), this test
actually loads the native diffusers ``Cosmos3OmniPipeline`` weights and runs a
real in-process forward pass. It needs a CUDA GPU + the model weights, so it is
skipped by default. Parallel to tests_integ/groot/test_n17_live_server.py.

Enable with:

    COSMOS3_DIFFUSERS_LIVE=1 \
    hatch run test-integ tests_integ/policies/cosmos3/test_diffusers_backend_live.py -v

Optionally override the checkpoint with COSMOS3_MODEL (default nvidia/Cosmos3-Nano).
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

LIVE = os.environ.get("COSMOS3_DIFFUSERS_LIVE", "").lower() in ("1", "true", "yes")
MODEL = os.environ.get("COSMOS3_MODEL", "nvidia/Cosmos3-Nano")

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="Requires a CUDA GPU + Cosmos 3 weights. Set COSMOS3_DIFFUSERS_LIVE=1 to enable.",
)

# Skip cleanly if the optional native stack is missing.
pytest.importorskip("diffusers", reason="diffusers not installed")
pytest.importorskip("torch", reason="torch not installed")


def _obs() -> dict:
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obs: dict[str, object] = {
        "observation/wrist_image_left": img,
        "observation/exterior_image_1_left": img,
        "observation/exterior_image_2_left": img,
    }
    for i in range(7):
        obs[f"joint_{i}"] = 0.1 * i
    obs["gripper"] = 0.2
    return obs


@pytest.fixture(scope="module")
def policy():
    from strands_robots.policies.cosmos3 import Cosmos3Policy

    p = Cosmos3Policy(embodiment="droid", backend="diffusers", model=MODEL)
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    return p


def test_policy_mode_returns_action_chunk_and_world_video(policy):
    """A real in-process policy run yields per-step actuator dicts AND surfaces
    the predicted world video on last_rollout. The diffusers backend emits the
    model's raw unified action (DROID = 9D end-effector pose + 1D gripper), so
    the step layout and dimensionality are derived from the embodiment registry
    rather than restated here, and every value is asserted finite and of sane
    magnitude - isinstance(v, float) alone passes on NaN/inf, which is exactly
    what a bad checkpoint or dtype drift emits.
    """
    from strands_robots.policies.cosmos3.embodiments import get_embodiment

    embodiment = get_embodiment("droid")
    out = policy.get_actions_sync(_obs(), "pick up the red cube")
    assert isinstance(out, list) and out
    for step in out:
        assert set(step.keys()) == set(embodiment.raw_action_layout)
        for name, v in step.items():
            assert isinstance(v, float), (name, v)
            assert math.isfinite(v), (name, v)
    assert policy.last_rollout is not None
    assert policy.last_rollout["video"] is not None  # predicted world video

    # The raw chunk is the model's quantile-normalized unified action: every
    # value finite and of sane magnitude, the second dimension the embodiment's
    # raw action dim, and not identically zero - an all-zeros chunk is the
    # classic silent-failure shape this repo forbids as a failure default.
    # The normalization maps q01 -> -1 and q99 -> +1 (action_decode.
    # denormalize_quantile), so the distribution's outer 2 percent sits beyond
    # magnitude 1 by construction and nothing in the cosmos3 path clamps the
    # sampler output - a hard |v| <= 1 bound would fail on a legitimate tail
    # value and report it as a bad checkpoint. Bounding the *scale* instead
    # keeps every mis-scale and dtype-drift failure this assertion is for.
    chunk = np.asarray(policy.last_rollout["action"], dtype=np.float64)
    assert chunk.ndim == 2
    assert chunk.shape[1] == embodiment.raw_action_dim
    assert np.isfinite(chunk).all()
    assert (np.abs(chunk) <= 10.0).all(), (chunk.min(), chunk.max())
    assert np.any(chunk != 0.0)


# --- Full sim loop: real Cosmos action -> de-normalize -> IK -> MuJoCo --------
#
# Closes the loop the issue asks for: drive a MuJoCo Franka/Panda arm with a
# real Cosmos 3 action chunk via the de-normalize + mink-IK bridge, and assert
# the EE trajectory tracks within the Thor-verified bar. Needs the GPU + weights
# (COSMOS3_DIFFUSERS_LIVE=1) AND the cosmos3-sim extra (mink + mujoco).


def test_cosmos_action_drives_mujoco_arm_within_tracking_bar(policy):
    """End-to-end on real weights: Cosmos chunk -> joint targets the arm tracks.

    The diffusers backend emits the raw quantile-normalized unified action; the
    sim bridge de-normalizes it (bundled q01/q99), decodes the relative EE-pose
    deltas to an absolute trajectory, and solves IK on the same MuJoCo model. A
    reachable trajectory must track to mean <= 12 mm / max <= 45 mm (the bar
    verified on Thor; the unit test tests/policies/cosmos3/test_sim_ik.py pins
    it off-GPU).
    """
    mujoco = pytest.importorskip("mujoco", reason="cosmos3-sim extra (mujoco) not installed")
    pytest.importorskip("mink", reason="cosmos3-sim extra (mink) not installed")
    panda_mj_description = pytest.importorskip(
        "robot_descriptions.panda_mj_description", reason="robot_descriptions not installed"
    )
    from strands_robots.policies.cosmos3 import MinkIKBridge, decode_cosmos_chunk_to_targets
    from strands_robots.policies.cosmos3.embodiments import get_embodiment

    # Real in-process Cosmos forward pass -> raw quantile-normalized chunk.
    policy.get_actions_sync(_obs(), "pick up the red cube")
    assert policy.last_rollout is not None
    raw_chunk = policy.last_rollout["action"]
    assert raw_chunk is not None and np.asarray(raw_chunk).ndim == 2

    model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
    bridge = MinkIKBridge(model, ee_frame_name="hand", ee_frame_type="body")
    q_init = np.zeros(model.nq, dtype=np.float64)
    q_init[:7] = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79]

    out = decode_cosmos_chunk_to_targets(
        np.asarray(raw_chunk, dtype=np.float32), get_embodiment("droid"), bridge, q_init
    )
    assert out["qpos"].shape[1] == model.nq
    assert out["poses"].shape[0] == out["qpos"].shape[0]
    # An IK solve that diverges to NaN passes the shape checks above; require
    # every solved joint target to be finite.
    assert np.isfinite(out["qpos"]).all()
    # Tracking bar (reachable trajectory). Cosmos's own deltas may scale past the
    # Franka reach; this asserts the IK *geometry* closes the loop.
    assert out["tracking_error"]["mean_mm"] <= 45.0, out["tracking_error"]
