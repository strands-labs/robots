"""WS-JSON wire protocol codec tests.

Verify that observations containing NumPy arrays round-trip byte-exact through
the JSON transport and that malformed frames are rejected loudly.
"""

import numpy as np
import pytest

from strands_robots.inference import protocol


def test_ndarray_roundtrip_is_byte_exact():
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    restored = protocol.decode_ndarray(protocol.encode_ndarray(arr))
    assert restored.dtype == arr.dtype
    assert restored.shape == arr.shape
    np.testing.assert_array_equal(restored, arr)


def test_uint8_image_roundtrip():
    image = np.random.randint(0, 256, size=(48, 64, 3), dtype=np.uint8)
    restored = protocol.decode_ndarray(protocol.encode_ndarray(image))
    assert restored.dtype == np.uint8
    np.testing.assert_array_equal(restored, image)


def test_observation_message_roundtrip_preserves_arrays_and_scalars():
    obs = {
        "observation.state": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "observation.images.top": np.zeros((8, 8, 3), dtype=np.uint8),
        "step": 5,
        "instruction": "pick the cube",
    }
    message = {"type": protocol.MSG_GET_ACTIONS, "observation": obs, "instruction": "pick"}
    decoded = protocol.loads(protocol.dumps(message))

    assert decoded["type"] == protocol.MSG_GET_ACTIONS
    assert decoded["instruction"] == "pick"
    decoded_obs = decoded["observation"]
    np.testing.assert_array_equal(decoded_obs["observation.state"], obs["observation.state"])
    np.testing.assert_array_equal(decoded_obs["observation.images.top"], obs["observation.images.top"])
    assert decoded_obs["step"] == 5
    assert decoded_obs["instruction"] == "pick the cube"


def test_decoded_array_is_writable_for_in_place_preprocessing():
    """A decoded observation array must be writable, matching the local path.

    A locally-run policy receives a freshly rendered (writable) observation and
    VLA preprocessors normalize it in place; a read-only array raises
    ``output array is read-only``.
    """
    image = np.random.randint(0, 256, size=(16, 16, 3), dtype=np.uint8)
    restored = protocol.decode_ndarray(protocol.encode_ndarray(image))
    assert restored.flags.writeable is True
    restored += 1  # in-place op must not raise

    state = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    restored_state = protocol.decode_ndarray(protocol.encode_ndarray(state))
    assert restored_state.flags.writeable is True
    restored_state *= 2.0


def test_observation_arrays_writable_after_full_message_roundtrip():
    """Arrays decoded from a full loads(dumps(...)) message are writable too."""
    obs = {
        "observation.state": np.array([0.1, 0.2], dtype=np.float64),
        "observation.images.top": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    decoded = protocol.loads(protocol.dumps({"type": protocol.MSG_GET_ACTIONS, "observation": obs}))
    decoded_obs = decoded["observation"]
    assert decoded_obs["observation.state"].flags.writeable is True
    assert decoded_obs["observation.images.top"].flags.writeable is True
    decoded_obs["observation.images.top"][0, 0, 0] = 42  # must not raise


def test_action_chunk_passes_through_untouched():
    # Action chunks are already JSON-native (dict[str, float | list[float]]).
    actions = [{"joint_0": 0.5, "gripper": [0.1, 0.2]}, {"joint_0": 0.6, "gripper": [0.15, 0.25]}]
    decoded = protocol.loads(protocol.dumps({"type": protocol.MSG_ACTIONS, "actions": actions}))
    assert decoded["actions"] == actions


def test_numpy_scalar_becomes_python_scalar():
    decoded = protocol.loads(protocol.dumps({"type": protocol.MSG_OK, "value": np.float32(1.5)}))
    assert decoded["value"] == 1.5
    assert isinstance(decoded["value"], float)


def test_loads_rejects_non_object_frame():
    with pytest.raises(ValueError, match="must be a JSON object"):
        protocol.loads("[1, 2, 3]")
