"""MuJoCo rollout smoke path for the OpenArm Cosmos 3 embodiment (#2461).

End-to-end over the same seams the four original embodiments use, with a mock
policy server instead of GPU weights: a raw ``[T, 10]`` unified-action chunk is
served through :class:`~strands_robots.policies.cosmos3.policy.Cosmos3Policy`
(service mode, ``midtrain`` action space) into per-step column dicts, then the
same chunk is closed onto the OpenArm MuJoCo model through the de-normalize +
IK bridge
(:func:`~strands_robots.policies.cosmos3.sim_ik.decode_cosmos_chunk_to_targets`).

The ``openarm_lerobot`` domain ships **no bundled stats** (it is a
post-training-only domain - the entry maps a checkpoint post-trained on OpenArm
episodes via the ``cosmos3`` trainer, not a released zero-shot model), so the
decode here declares synthetic stats via ``stats=`` + ``stats_domain=``,
exactly the route the bundled-stats refusal advises. The stats are shaped so
the de-normalized chunk is physically reachable (millimetre translation deltas,
near-identity rotations), keeping this a geometry smoke test rather than a
model-quality claim.

``mujoco`` + ``mink`` (the ``cosmos3-sim`` extra) and ``robot_descriptions``
(for the OpenArm MJCF, the same ``openarm_v1_mj_description`` module the robot
registry's ``openarm`` entry declares) are imported via ``importorskip`` so the
module skips cleanly when the sim stack is absent, mirroring
``test_sim_ik.py``.
"""

import asyncio
from pathlib import Path

import numpy as np
import pytest

# Optional sim stack: skip cleanly when the cosmos3-sim extra is not installed.
mujoco = pytest.importorskip("mujoco", reason="mujoco not installed - pip install 'strands-robots[cosmos3-sim]'")
pytest.importorskip("mink", reason="mink not installed - pip install 'strands-robots[cosmos3-sim]'")
openarm_mj_description = pytest.importorskip(
    "robot_descriptions.openarm_v1_mj_description",
    reason="robot_descriptions not installed",
)

# E402: importorskip must run before these imports so the module skips cleanly.
from strands_robots.policies.cosmos3.embodiments import get_embodiment  # noqa: E402
from strands_robots.policies.cosmos3.policy import Cosmos3Policy  # noqa: E402
from strands_robots.policies.cosmos3.sim_ik import (  # noqa: E402
    MinkIKBridge,
    decode_cosmos_chunk_to_targets,
)

from .test_policy import FakeClient  # noqa: E402

# Tracking-error acceptance bar - same numbers test_sim_ik.py pins for DROID.
_MEAN_MM_BAR = 12.0
_MAX_MM_BAR = 45.0

# Elbow-bent OpenArm home (inside every joint range) so the EE starts
# mid-workspace and the synthetic targets stay reachable.
_Q_HOME = np.array([0.0, 0.6, 0.0, 1.2, 0.0, 0.5, 0.0], dtype=np.float64)


def _openarm_stats(dim: int = 10) -> dict[str, np.ndarray]:
    """Synthetic, physically-reachable quantile stats for ``openarm_lerobot``.

    Translation columns de-normalize to +-10 mm per step, the rot6d block to
    itself (so near-identity normalized rotations stay near-identity), and the
    grasp column to the OpenArm finger range [0, 0.044] m.
    """
    q01 = np.array([-0.01] * 3 + [-1.0] * 6 + [0.0], dtype=np.float32)
    q99 = np.array([0.01] * 3 + [1.0] * 6 + [0.044], dtype=np.float32)
    assert q01.shape[-1] == dim and q99.shape[-1] == dim
    return {"q01": q01, "q99": q99}


def _reachable_chunk(t: int, dim: int) -> np.ndarray:
    """A normalized [-1, 1] chunk whose de-normalized deltas stay reachable."""
    rng = np.random.default_rng(0)
    chunk = rng.uniform(-0.3, 0.3, (t, dim)).astype(np.float32)
    # Near-identity rot6d so the synthetic deltas do not wander out of the
    # workspace (a scaling concern, not an IK one - same note as test_sim_ik).
    chunk[:, 3:9] = np.tile([1, 0, 0, 0, 1, 0], (t, 1))
    return chunk


@pytest.fixture(scope="module")
def openarm_model():
    # The registry's ``openarm`` entry resolves the single-arm ``openarm.xml``
    # inside the ``openarm_v1_mj_description`` asset dir (the module's own
    # MJCF_PATH names the bimanual variant - the #2461 follow-up).
    xml = Path(openarm_mj_description.MJCF_PATH).with_name("openarm.xml")
    if not xml.exists():
        pytest.skip(f"single-arm openarm.xml not shipped next to {openarm_mj_description.MJCF_PATH}")
    return mujoco.MjModel.from_xml_path(str(xml))


@pytest.fixture
def q_init(openarm_model):
    q = np.zeros(openarm_model.nq, dtype=np.float64)
    q[:7] = _Q_HOME
    return q


@pytest.fixture
def bridge(openarm_model):
    # link7 is the last arm link (the single-arm MJCF has no hand_tcp body).
    return MinkIKBridge(openarm_model, ee_frame_name="openarm_link7", ee_frame_type="body")


def test_openarm_chunk_decodes_to_joint_targets_within_bar(bridge, q_init):
    """Raw normalized chunk -> declared stats -> EE poses -> OpenArm joints."""
    emb = get_embodiment("openarm")
    chunk = _reachable_chunk(16, emb.raw_action_dim)

    out = decode_cosmos_chunk_to_targets(
        chunk,
        emb,
        bridge,
        q_init,
        stats=_openarm_stats(emb.raw_action_dim),
        stats_domain=emb.domain_name,
    )

    assert out["qpos"].shape == (16, bridge.model.nq)
    assert out["poses"].shape == (16, 4, 4)
    # The openarm raw layout ends in "grasp" -> gripper column is split off.
    assert out["gripper"] is not None
    assert out["gripper"].shape == (16,)
    assert out["tracking_error"]["mean_mm"] <= _MEAN_MM_BAR, out["tracking_error"]
    assert out["tracking_error"]["max_mm"] <= _MAX_MM_BAR, out["tracking_error"]
    # Joint targets respect the OpenArm model's own limits (MinkIKBridge
    # enforces them - normalized columns fed raw would not).
    lo, hi = bridge.model.jnt_range[:, 0], bridge.model.jnt_range[:, 1]
    for q in out["qpos"]:
        assert np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6)


def test_openarm_decode_without_stats_fails_loudly(bridge, q_init):
    """No bundled openarm_lerobot stats: the default path refuses by name
    instead of silently substituting another domain's quantiles."""
    emb = get_embodiment("openarm")
    chunk = _reachable_chunk(4, emb.raw_action_dim)
    with pytest.raises(FileNotFoundError, match="openarm_lerobot"):
        decode_cosmos_chunk_to_targets(chunk, emb, bridge, q_init)


def test_openarm_mock_server_to_mujoco_end_to_end(bridge, q_init):
    """Mock policy server -> Cosmos3Policy column dicts -> IK joint targets.

    The full deploy-side loop for the OpenArm mapping, with the served chunk
    standing in for a post-trained checkpoint's output: the policy names every
    column of the ``midtrain`` layout per step, and reassembling those columns
    reproduces the exact chunk the IK bridge closes onto the MuJoCo arm.
    """
    emb = get_embodiment("openarm")
    chunk = _reachable_chunk(8, emb.raw_action_dim)
    policy = Cosmos3Policy(embodiment="openarm", client=FakeClient(chunk))

    img = np.zeros((240, 320, 3), dtype=np.uint8)
    obs = {"observation/image": img, "observation/wrist_image": img}
    steps = asyncio.run(policy.get_actions(obs, "hand over the cube"))

    layout = emb.action_layouts["midtrain"]
    assert len(steps) == 8
    assert all(sorted(s) == sorted(layout) for s in steps)

    # Round-trip: the named columns reassemble into the served chunk...
    reassembled = np.array([[s[c] for c in layout] for s in steps], dtype=np.float32)
    np.testing.assert_allclose(reassembled, chunk, atol=1e-6)

    # ...which the sim bridge closes onto the OpenArm model within the bar.
    out = decode_cosmos_chunk_to_targets(
        reassembled,
        emb,
        bridge,
        q_init,
        stats=_openarm_stats(emb.raw_action_dim),
        stats_domain=emb.domain_name,
    )
    assert out["qpos"].shape == (8, bridge.model.nq)
    assert out["tracking_error"]["mean_mm"] <= _MEAN_MM_BAR, out["tracking_error"]


def test_openarm_requires_its_full_camera_set():
    """The camera guard covers the openarm entry: a missing wrist view is
    refused client-side, naming the missing key."""
    emb = get_embodiment("openarm")
    policy = Cosmos3Policy(embodiment="openarm", client=FakeClient(_reachable_chunk(4, emb.raw_action_dim)))
    obs = {"observation/image": np.zeros((240, 320, 3), dtype=np.uint8)}
    with pytest.raises(ValueError, match="observation/wrist_image"):
        asyncio.run(policy.get_actions(obs, "hand over the cube"))
