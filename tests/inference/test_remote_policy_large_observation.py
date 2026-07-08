"""Regression: the server must accept a realistically-sized image observation.

A real VLA observation carries camera frames. A single 640x480 RGB frame
base64-encodes to ~1.2 MiB and a multi-camera observation is several MiB, so
the WebSocket frame-size limit must be lifted on BOTH ends. The client already
passes ``max_size=None`` to ``connect()``; the server must pass it to ``serve()``
too, or every image-carrying ``get_actions`` request is rejected with a
``1009 (message too big)`` close before the wrapped policy ever runs.

The pre-existing round-trip tests use an image-free policy (``requires_images``
is ``False``), so their tiny state-only messages never approached the limit and
this gap was invisible. These tests stream an over-1-MiB observation through the
live loopback WebSocket and assert the wrapped policy receives it intact.
"""

from typing import Any

import numpy as np
import pytest

from strands_robots.inference import PolicyServer, RemotePolicy
from strands_robots.policies.base import Policy

# A single 640x480 RGB uint8 frame: 921,600 raw bytes -> ~1.23 MiB base64,
# comfortably over the websockets default 1 MiB (2**20) frame limit.
_IMG_H, _IMG_W = 480, 640
_LIMIT_BYTES = 1 << 20


class ImageObservingPolicy(Policy):
    """Chunk-emitting policy that READS (never mutates) the received image.

    Records the sum of the last image it was handed so a test can assert the
    full frame arrived intact server-side. It only reads the array, so this
    test isolates the frame-size contract from the array-writability contract.
    """

    def __init__(self) -> None:
        self.actions_per_step = 4
        self.supports_rtc = False
        self.robot_state_keys: list[str] = ["j0", "j1"]
        self.seen_image_sums: list[float] = []
        self.seen_image_shapes: list[tuple[int, ...]] = []

    @property
    def provider_name(self) -> str:
        return "image-observer"

    @property
    def requires_images(self) -> bool:
        return True

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.robot_state_keys = list(robot_state_keys)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        image = observation_dict["observation.images.cam"]
        arr = np.asarray(image)
        self.seen_image_shapes.append(arr.shape)
        self.seen_image_sums.append(float(arr.sum()))
        return [{key: 0.0 for key in self.robot_state_keys} for _ in range(self.actions_per_step)]


def _big_observation(seed: int = 0) -> tuple[dict[str, Any], float]:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(_IMG_H, _IMG_W, 3), dtype=np.uint8)
    obs = {
        "observation.images.cam": image,
        "observation.state": np.array([0.1, 0.2], dtype=np.float32),
    }
    return obs, float(image.sum())


def test_observation_frame_exceeds_default_websocket_limit():
    """Guard the premise: the encoded observation really is over 1 MiB."""
    from strands_robots.inference import protocol

    obs, _ = _big_observation()
    encoded = protocol.dumps({"type": protocol.MSG_GET_ACTIONS, "observation": obs})
    assert len(encoded.encode("utf-8")) > _LIMIT_BYTES, (
        "the test image must exceed the default 1 MiB frame limit to exercise the bug"
    )


def test_server_accepts_over_1mib_image_observation():
    """A >1 MiB image observation round-trips instead of a 1009 close."""
    policy = ImageObservingPolicy()
    server = PolicyServer(policy=policy, host="127.0.0.1", port=0).start()
    client = RemotePolicy(endpoint=f"ws://127.0.0.1:{server.port}")
    try:
        obs, expected_sum = _big_observation(seed=7)
        # Pre-fix this raises (server 1009-closes the oversize frame); post-fix
        # the wrapped policy runs and returns its chunk.
        chunk = client.get_actions_sync(obs, "pick up the red cube")
        assert len(chunk) == policy.actions_per_step
        assert set(chunk[0].keys()) == {"j0", "j1"}
        # The full frame reached the wrapped policy byte-intact.
        assert policy.seen_image_shapes[-1] == (_IMG_H, _IMG_W, 3)
        assert policy.seen_image_sums[-1] == pytest.approx(expected_sum)
    finally:
        client.close()
        server.stop()
