"""GPU + docker integration test for the ``deterministic=True`` lifecycle flag (#1790).

Brings up a real GR00T N1.7 LIBERO server through
``gr00t_inference(action="lifecycle", lifecycle="full", deterministic=True, ...)``
and asserts the two acceptance criteria that only a live server can prove:

1. The container logs show the determinism-wrapper banner
   (``[srv_wrap] determinism: cudnn.deterministic=True ...``), i.e. the
   server really runs through the mounted wrapper, not the bare entrypoint.
2. Two evals at the same seed produce **bit-exact** per-episode action
   streams (short rollout of identical observations after
   ``reset(options={"seed": N})``), not merely equal success rates.

Heavy: needs docker, an NVIDIA GPU, the ``gr00t:latest`` image (built on
first run), the ``nvidia/GR00T-N1.7-LIBERO`` checkpoint (~10 GB download on
first run), and an HF token with access to the gated Cosmos backbone.
Opt in explicitly:

    GR00T_DETERMINISTIC_LIFECYCLE=1 \
    hatch run test-integ tests_integ/groot/test_deterministic_lifecycle.py -v
"""

from __future__ import annotations

import os
import subprocess
import time

import numpy as np
import pytest

ENABLED = os.environ.get("GR00T_DETERMINISTIC_LIFECYCLE", "").lower() in ("1", "true", "yes")
PORT = int(os.environ.get("GR00T_DETERMINISTIC_PORT", "18000"))
CONTAINER = "gr00t-deterministic-it"
CHECKPOINT_DIR = os.environ.get("STRANDS_ROBOTS_CHECKPOINT_DIR", "/tmp/strands_robots/checkpoints")
SEED = 42
ROLLOUT_STEPS = 3

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not ENABLED,
        reason="Full GR00T container lifecycle (docker + GPU + ~10GB checkpoint). Set GR00T_DETERMINISTIC_LIFECYCLE=1.",
    ),
]

pytest.importorskip("msgpack")
pytest.importorskip("zmq")

from strands_robots.policies.groot.policy import Gr00tPolicy  # noqa: E402
from strands_robots.tools.gr00t_inference import gr00t_inference  # noqa: E402


@pytest.fixture(scope="module")
def deterministic_server():
    """lifecycle="full" with deterministic=True; teardown removes the container."""
    result = gr00t_inference(
        action="lifecycle",
        lifecycle="full",
        hf_repo="nvidia/GR00T-N1.7-LIBERO",
        hf_subfolder="libero_10",
        hf_local_dir=CHECKPOINT_DIR,
        checkpoint_path="/data/checkpoints/libero_10",
        container_name=CONTAINER,
        embodiment_tag="libero_sim",
        protocol="n1.7",
        use_sim_policy_wrapper=True,
        deterministic=True,
        port=PORT,
        timeout=120,
    )
    assert result.get("status") == "success", f"lifecycle failed: {result}"

    # The port binds before the model finishes loading; poll the container
    # logs for the wrapper's reset-patch line (the last thing it prints
    # before handing off to the server entrypoint) so the rollout below
    # never races the model load.
    deadline = time.monotonic() + 300
    logs = ""
    while time.monotonic() < deadline:
        logs = subprocess.run(
            ["docker", "logs", CONTAINER],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        if "[srv_wrap] patched Gr00tPolicy.reset" in logs:
            break
        time.sleep(5)

    yield {"port": PORT, "logs": logs}

    gr00t_inference(action="lifecycle", lifecycle="teardown", container_name=CONTAINER)


def _libero_obs(rng: np.random.Generator) -> dict:
    """A fixed-seed LIBERO-panda robot observation (bare sensor keys).

    ``Gr00tPolicy`` translates these into the N1.7 wire layout exactly the
    way the LIBERO eval driver does.
    """
    obs: dict = {
        "image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        "wrist_image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
    }
    for key in ("x", "y", "z", "roll", "pitch", "yaw"):
        obs[key] = np.float32(rng.standard_normal())
    # LIBERO's gripper state is the two finger joints (see
    # strands_robots.benchmarks.libero.adapter - `merged["gripper"]`).
    obs["gripper"] = rng.standard_normal(2).astype(np.float32)
    return obs


def _rollout(port: int) -> list[list[dict]]:
    """reset(seed=SEED) then request actions for ROLLOUT_STEPS identical frames."""
    import asyncio

    policy = Gr00tPolicy(
        data_config="libero_panda",
        host="localhost",
        port=port,
        groot_version="n1.7",
    )
    policy.reset(seed=SEED)
    stream: list[list[dict]] = []
    rng = np.random.default_rng(SEED)
    for _ in range(ROLLOUT_STEPS):
        actions = asyncio.run(policy.get_actions(_libero_obs(rng), "pick up the object"))
        stream.append(actions)
    return stream


def test_wrapper_banner_in_container_logs(deterministic_server):
    """The server must be running through the mounted wrapper."""
    logs = deterministic_server["logs"]
    assert "[srv_wrap] determinism: cudnn.deterministic=True" in logs, (
        "wrapper banner missing from container logs - the deterministic lifecycle "
        f"did not exec the wrapper. Logs tail:\n{logs[-2000:]}"
    )
    assert "[srv_wrap] patched Gr00tPolicy.reset" in logs


def test_same_seed_gives_bit_exact_action_streams(deterministic_server):
    """Two rollouts at the same seed must agree bit-for-bit, not just on average."""
    port = deterministic_server["port"]
    first = _rollout(port)
    second = _rollout(port)

    assert len(first) == len(second) == ROLLOUT_STEPS
    for step, (chunk_a, chunk_b) in enumerate(zip(first, second)):
        assert len(chunk_a) == len(chunk_b), f"step {step}: action horizon diverged"
        for t, (a, b) in enumerate(zip(chunk_a, chunk_b)):
            assert a.keys() == b.keys(), f"step {step} t={t}: action keys diverged"
            for key in a:
                np.testing.assert_array_equal(
                    np.asarray(a[key]),
                    np.asarray(b[key]),
                    err_msg=f"step {step} t={t}, action {key!r}: streams are not bit-exact",
                )

    # And the wrapper actually observed the per-episode reseeds.
    logs = subprocess.run(["docker", "logs", CONTAINER], capture_output=True, text=True, check=False).stdout
    assert f"[srv_wrap] reset: re-seeded to {SEED}" in logs
