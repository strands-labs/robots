"""Real-pipeline injection tests for the declarative embodiment mapping.

Unlike the mock-heavy existing suite (which let B7/B12 slip past), these load an
ACTUAL LeRobot processor pipeline and verify the embodiment map is injected and
transforms raw strands-native observations correctly. Skips cleanly if the
model config isn't cached / lerobot processor unavailable.
"""

import numpy as np
import pytest

pytest.importorskip("lerobot")

from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap
from strands_robots.policies.lerobot_local.processor import ProcessorBridge

SMOLVLA = "lerobot/smolvla_base"


def _load_bridge():
    """Load SmolVLA's real preprocessor; skip if unavailable/uncached."""
    bridge = None
    try:
        bridge = ProcessorBridge.from_pretrained(
            SMOLVLA,
            device="cpu",
            policy_type="smolvla",
            overrides={"device_processor": {"device": "cpu"}},
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"SmolVLA pipeline unavailable: {exc}")
    if bridge is None or not bridge.has_preprocessor:
        pytest.skip("SmolVLA preprocessor not loaded")
    return bridge


def test_apply_embodiment_injects_rename_and_pack():
    bridge = _load_bridge()
    pre = bridge._preprocessor
    em = EmbodimentMap(
        name="t",
        obs_rename={"front": "observation.images.top", "wrist": "observation.images.wrist"},
        state_keys=["1", "2", "3", "4", "5", "6"],
        action_keys=["1", "2", "3", "4", "5", "6"],
        dim_policy="pad",
    )
    bridge.apply_embodiment(em, input_features={})
    names = [getattr(s, "_registry_name", type(s).__name__) for s in pre.steps]
    # rename first, pack-state immediately after
    assert names[0] == "rename_observations_processor"
    assert names[1] == "strands_pack_state"
    assert pre.steps[0].rename_map == em.obs_rename


def test_apply_embodiment_idempotent():
    bridge = _load_bridge()
    pre = bridge._preprocessor
    em = EmbodimentMap(name="t", obs_rename={}, state_keys=["1", "2"], dim_policy="pad")
    bridge.apply_embodiment(em, input_features={})
    bridge.apply_embodiment(em, input_features={})  # re-apply
    names = [getattr(s, "_registry_name", type(s).__name__) for s in pre.steps]
    assert names.count("strands_pack_state") == 1


def test_raw_obs_transforms_through_injected_steps():
    """RAW sim obs -> rename + pack steps -> LeRobot keys, no strands remap."""
    from lerobot.processor import TransitionKey
    from lerobot.processor.converters import create_transition

    bridge = _load_bridge()
    pre = bridge._preprocessor
    em = EmbodimentMap(
        name="t",
        obs_rename={"front": "observation.images.top", "wrist": "observation.images.wrist"},
        state_keys=["1", "2", "3", "4", "5", "6"],
        dim_policy="pad",
    )
    bridge.apply_embodiment(em, input_features={})

    raw = {
        "front": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist": np.zeros((4, 4, 3), dtype=np.uint8),
        "1": 0.1,
        "2": 0.2,
        "3": 0.3,
        "4": 0.4,
        "5": 0.5,
        "6": 0.6,
    }
    t = create_transition(observation=raw, complementary_data={"task": "pick"})
    t = pre.steps[0](t)  # rename
    t = pre.steps[1](t)  # pack-state
    obs = t[TransitionKey.OBSERVATION]

    assert "observation.images.top" in obs
    assert "observation.images.wrist" in obs
    assert "observation.state" in obs
    assert list(np.asarray(obs["observation.state"]).ravel()) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # raw strands keys are gone
    assert "front" not in obs and "1" not in obs
