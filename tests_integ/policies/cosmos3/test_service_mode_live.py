"""Live service-mode integration test for the Cosmos 3 policy provider.

Unlike tests/policies/cosmos3/test_policy.py (mocked client) and
tests_integ/policies/cosmos3/test_diffusers_backend_live.py (in-process
diffusers backend), this test exercises the WebSocket service path
(:mod:`strands_robots.policies.cosmos3.client`) with a real msgpack+NumPy
round-trip against a **pre-running** Cosmos Framework RoboLab policy server.
That makes it cheap: it can run from a non-GPU box pointed at a GPU host,
mirroring tests_integ/groot/test_n17_live_server.py.

Start the server first (holds the GPU) from a Cosmos Framework checkout:

    uv sync --all-extras --group=cu130-train --group=policy-server
    python -m cosmos_framework.scripts.action_policy_server_robolab \
        --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID --port 8000

Then enable with:

    COSMOS3_LIVE_SERVER=1 \
    COSMOS3_SERVER_HOST=192.168.1.151 \
    COSMOS3_SERVER_PORT=8000 \
    hatch run test-integ tests_integ/policies/cosmos3/test_service_mode_live.py -v

Spawning the server from a fixture is deliberately out of scope here (a
separate follow-up); this file documents the launch command instead so a
runner can start it by hand.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest

LIVE = os.environ.get("COSMOS3_LIVE_SERVER", "").lower() in ("1", "true", "yes")
HOST = os.environ.get("COSMOS3_SERVER_HOST", "localhost")
PORT = int(os.environ.get("COSMOS3_SERVER_PORT", "8000"))

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="Requires a pre-running Cosmos 3 RoboLab policy server. Set COSMOS3_LIVE_SERVER=1 to enable.",
)

# Skip cleanly if optional deps are missing.
pytest.importorskip("websockets")
pytest.importorskip("msgpack")

from strands_robots.policies import create_policy  # noqa: E402
from strands_robots.policies.cosmos3.embodiments import get_embodiment  # noqa: E402


@pytest.fixture(scope="module")
def policy():
    """One policy (and one lazily-connected WebSocket) for the whole module.

    Built through the public factory exactly as a user would; module scope
    means every test in this file shares the same underlying connection, so
    a second inference proves the wire loop (connect -> recv metadata ->
    send obs -> recv action) survives more than one round-trip.
    """
    p = create_policy("cosmos3", embodiment="droid", host=HOST, port=PORT)
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    return p


def _obs() -> dict:
    """Synthetic flat DROID observation: 3 RoboArena views + 7 joints + gripper.

    Same shape as _obs() in test_diffusers_backend_live.py - the flat robot
    observation strands_robots/policies/cosmos3/policy.py documents.
    """
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


def _assert_valid_joint_chunk(out: list) -> None:
    """Assert a chunk of per-step dicts matches the DROID joint_pos layout.

    The expected key set is derived from the embodiment registry rather than
    hard-coded, so an embodiment edit cannot silently desync this test.
    Finiteness is asserted explicitly: a NaN is an instance of ``float``, so
    ``isinstance`` alone passes on garbage.
    """
    embodiment = get_embodiment("droid")
    layout = embodiment.action_layouts["joint_pos"]
    joint_names = [c for c in layout if c != "gripper"]

    assert isinstance(out, list) and out
    for step in out:
        assert isinstance(step, dict)
        assert set(step.keys()) == set(layout)
        for name, v in step.items():
            assert isinstance(v, float), (name, v)
            assert math.isfinite(v), (name, v)

    # The server de-normalizes joint_pos to radians: every joint value must be
    # in a plausible joint-position range. A raw [-1, 1] normalized value would
    # also pass this bound, so additionally require the chunk is not
    # identically zero - an all-zeros chunk is the classic silent-failure
    # shape this repo forbids as a failure default.
    joints = np.asarray([[step[j] for j in joint_names] for step in out], dtype=np.float64)
    assert (np.abs(joints) < 2 * math.pi).all(), (joints.min(), joints.max())
    assert np.any(joints != 0.0)


def test_live_round_trip_returns_valid_droid_joint_chunk(policy):
    """One real round-trip: msgpack+NumPy over WebSocket, GPU inference, and a
    de-normalized DROID joint_pos chunk back."""
    t0 = time.perf_counter()
    out = policy.get_actions_sync(_obs(), "pick up the red cube")
    dt_ms = (time.perf_counter() - t0) * 1000

    _assert_valid_joint_chunk(out)

    # The raw [T, D] chunk surfaces on last_rollout (service-backend parity
    # with the diffusers backend).
    assert policy.last_rollout is not None
    chunk = np.asarray(policy.last_rollout["action"])
    assert chunk.ndim == 2, chunk.shape
    assert chunk.shape[0] >= 1
    assert chunk.shape[0] == len(out)
    assert np.isfinite(chunk).all()

    # Loose latency sanity - informational only (cold inference can be slow).
    print(f"\nCosmos 3 service inference latency: {dt_ms:.0f}ms  (informational)")


def test_connection_survives_a_second_round_trip(policy):
    """A second call with a different instruction on the SAME module-scoped
    policy also returns a valid chunk - the wire protocol is connect -> recv
    metadata -> send obs -> recv action (per
    strands_robots/policies/cosmos3/client.py), so this proves the connection
    serves more than one inference."""
    out = policy.get_actions_sync(_obs(), "place the cube in the bowl")
    _assert_valid_joint_chunk(out)
    assert policy.last_rollout is not None
    chunk = np.asarray(policy.last_rollout["action"])
    assert chunk.ndim == 2 and chunk.shape[0] >= 1
