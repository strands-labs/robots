"""A ``rename_map`` camera entry is applied before the server resizes the image.

lerobot's ``PolicyServer`` runs ``prepare_raw_observation`` on every declared
``observation.images.<key>`` and looks the key up in the checkpoint's own image
features to pick the resize target - BEFORE the ``RenameObservationsProcessorStep``
that ``rename_map`` configures. A camera declared under the robot's name is a
``KeyError`` there (``Error in StreamActions: 'observation.images.front'`` on a
stock server), and the client raises "server returned no actions".

So the client applies an image rename itself: the handshake declares the
model's feature name and the raw observation carries the image under the
matching wire key. Fails on pre-fix code, which declared and sent ``front``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("lerobot")

import numpy as np

from strands_robots.policies.lerobot_async import LerobotAsyncPolicy

STATE_KEYS = ["j0", "j1"]


def _policy() -> LerobotAsyncPolicy:
    policy = LerobotAsyncPolicy(
        server_address="h:1",
        policy_type="act",
        pretrained_name_or_path="x/y",
        rename_map={"observation.images.front": "observation.images.laptop"},
    )
    policy.set_robot_state_keys(STATE_KEYS)
    return policy


def _observation() -> dict[str, object]:
    obs: dict[str, object] = {k: 0.0 for k in STATE_KEYS}
    obs["front"] = np.zeros((8, 8, 3), dtype=np.uint8)
    obs["wrist"] = np.zeros((8, 8, 3), dtype=np.uint8)
    return obs


def test_handshake_declares_the_model_camera_name() -> None:
    features = _policy()._build_lerobot_features(_observation())
    image_keys = sorted(k for k in features if k.startswith("observation.images."))
    assert image_keys == ["observation.images.laptop", "observation.images.wrist"]


def test_raw_observation_carries_the_image_under_the_model_name() -> None:
    raw = _policy()._to_raw_observation(_observation(), "")
    assert "laptop" in raw and "front" not in raw
    assert "wrist" in raw
