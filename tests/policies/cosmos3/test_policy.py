"""Unit tests for Cosmos3Policy — no GPU, no server (mocked client)."""

import asyncio

import numpy as np
import pytest

from strands_robots.policies.base import Policy
from strands_robots.policies.cosmos3 import Cosmos3Policy
from strands_robots.policies.cosmos3.policy import _to_image_uint8


class FakeClient:
    """Stand-in for Cosmos3WebsocketClient — records the obs, returns a chunk."""

    def __init__(self, action: np.ndarray):
        self._action = action
        self.last_obs = None
        self.reset_calls = 0

    def infer(self, observation):
        self.last_obs = observation
        return {"action": self._action, "server_timing": {"infer_ms": 1.0}}

    def reset(self):
        self.reset_calls += 1

    def get_server_metadata(self):
        return {}


def _droid_chunk(t=32, d=8):
    rng = np.random.default_rng(0)
    return rng.standard_normal((t, d)).astype(np.float32)


def _make_droid_policy(action=None, **kw):
    action = _droid_chunk() if action is None else action
    return Cosmos3Policy(embodiment="droid", client=FakeClient(action), **kw)


def test_is_a_policy():
    p = _make_droid_policy()
    assert isinstance(p, Policy)
    assert p.provider_name == "cosmos3"
    assert p.requires_images is True


def test_invalid_action_space_raises():
    with pytest.raises(ValueError, match="no action_space"):
        Cosmos3Policy(embodiment="droid", action_space="not_a_space", client=FakeClient(_droid_chunk()))


def test_default_action_space_from_embodiment():
    assert _make_droid_policy().action_space == "joint_pos"
    p = Cosmos3Policy(embodiment="av", client=FakeClient(_droid_chunk(60, 9)))
    assert p.action_space == "midtrain"


def _obs_with_state_and_images():
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    obs = {
        "observation/wrist_image_left": img,
        "observation/exterior_image_1_left": img,
        "observation/exterior_image_2_left": img,
    }
    # robot joint state (scalar floats) + gripper
    for i in range(7):
        obs[f"joint_{i}"] = float(i) * 0.1
    obs["gripper"] = 0.5
    return obs


def test_get_actions_returns_chunk_of_dicts():
    p = _make_droid_policy()
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    out = asyncio.run(p.get_actions(_obs_with_state_and_images(), "pick up the cube"))
    assert isinstance(out, list)
    assert len(out) == 32
    step = out[0]
    assert set(step.keys()) == {f"joint_{i}" for i in range(7)} | {"gripper"}
    assert all(isinstance(v, float) for v in step.values())


def test_get_actions_builds_correct_server_obs():
    client = FakeClient(_droid_chunk())
    p = Cosmos3Policy(embodiment="droid", client=client)
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    asyncio.run(p.get_actions(_obs_with_state_and_images(), "do it"))
    obs = client.last_obs
    assert obs["prompt"] == "do it"
    assert obs["observation/wrist_image_left"].shape == (360, 640, 3)
    assert obs["observation/wrist_image_left"].dtype == np.uint8
    assert obs["observation/joint_position"].shape == (1, 7)
    assert obs["observation/gripper_position"].shape == (1, 1)


def test_action_mapping_renames_columns():
    p = Cosmos3Policy(
        embodiment="droid",
        client=FakeClient(_droid_chunk()),
        action_mapping={"joint_0": "shoulder_pan", "gripper": "grip"},
    )
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    out = asyncio.run(p.get_actions(_obs_with_state_and_images(), "x"))
    assert "shoulder_pan" in out[0]
    assert "grip" in out[0]
    assert "joint_0" not in out[0]


def test_default_prompt_used_when_instruction_empty():
    client = FakeClient(_droid_chunk())
    p = Cosmos3Policy(embodiment="droid", client=client, prompt="default task")
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    asyncio.run(p.get_actions(_obs_with_state_and_images(), ""))
    assert client.last_obs["prompt"] == "default task"


def test_reset_forwards_to_client():
    client = FakeClient(_droid_chunk())
    p = Cosmos3Policy(embodiment="droid", client=client)
    p.reset(seed=7)
    assert client.reset_calls == 1


def test_get_actions_sync_wrapper():
    p = _make_droid_policy()
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    out = p.get_actions_sync(_obs_with_state_and_images(), "go")
    assert len(out) == 32


def test_unpack_1d_action_promoted():
    p = _make_droid_policy()
    steps = p._unpack_actions(np.zeros(8, dtype=np.float32))
    assert len(steps) == 1
    assert len(steps[0]) == 8


def test_unpack_unexpected_width_pads_names():
    p = _make_droid_policy()
    steps = p._unpack_actions(np.zeros((2, 10), dtype=np.float32))  # wider than 8
    assert "action_8" in steps[0] and "action_9" in steps[0]


def test_to_image_uint8_coerces_float():
    img = np.ones((4, 4, 3), dtype=np.float32) * 300.0
    out = _to_image_uint8(img)
    assert out.dtype == np.uint8
    assert out.max() == 255  # clipped


def test_to_image_uint8_rejects_bad_shape():
    with pytest.raises(ValueError, match="H, W, 3"):
        _to_image_uint8(np.zeros((4, 4), dtype=np.uint8))


def test_finger_joint_gripper_mapping():
    """Panda-style state keys (joint1..joint7 + finger_joint1) build (1,7)+(1,1)."""
    client = FakeClient(_droid_chunk())
    p = Cosmos3Policy(embodiment="droid", client=client)
    p.set_robot_state_keys([f"joint{i}" for i in range(1, 8)] + ["finger_joint1"])
    obs = {
        "observation/wrist_image_left": np.zeros((360, 640, 3), np.uint8),
        "observation/exterior_image_1_left": np.zeros((360, 640, 3), np.uint8),
        "observation/exterior_image_2_left": np.zeros((360, 640, 3), np.uint8),
    }
    for i in range(1, 8):
        obs[f"joint{i}"] = 0.1 * i
    obs["finger_joint1"] = 0.02
    asyncio.run(p.get_actions(obs, "go"))
    assert client.last_obs["observation/joint_position"].shape == (1, 7)
    assert client.last_obs["observation/gripper_position"].shape == (1, 1)
    assert abs(float(client.last_obs["observation/gripper_position"][0, 0]) - 0.02) < 1e-6


def test_missing_gripper_raises_no_silent_default():
    """joint_pos without a gripper state key must raise, not fabricate 0.0."""
    client = FakeClient(_droid_chunk())
    p = Cosmos3Policy(embodiment="droid", client=client)
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)])  # 7 joints, NO gripper
    obs = {"observation/wrist_image_left": np.zeros((360, 640, 3), np.uint8)}
    obs.update({f"joint_{i}": 0.1 * i for i in range(7)})
    obs["observation/wrist_image_left"] = np.zeros((360, 640, 3), np.uint8)
    with pytest.raises(ValueError, match="gripper"):
        asyncio.run(p.get_actions(obs, "go"))


def test_missing_joints_raises():
    client = FakeClient(_droid_chunk())
    p = Cosmos3Policy(embodiment="droid", client=client)
    p.set_robot_state_keys(["joint_0", "joint_1", "gripper"])  # only 2 joints
    obs = {
        "observation/wrist_image_left": np.zeros((360, 640, 3), np.uint8),
        "joint_0": 0.0,
        "joint_1": 0.1,
        "gripper": 0.5,
    }
    with pytest.raises(ValueError, match="7 joint state"):
        asyncio.run(p.get_actions(obs, "go"))


def test_invalid_action_mapping_key_raises_at_construction():
    with pytest.raises(ValueError, match="not in the .* action layout"):
        Cosmos3Policy(
            embodiment="droid",
            client=FakeClient(_droid_chunk()),
            action_mapping={"not_a_column": "whatever"},
        )


def test_requires_images_guard_raises_without_camera():
    client = FakeClient(_droid_chunk())
    p = Cosmos3Policy(embodiment="droid", client=client)
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    obs = {f"joint_{i}": 0.1 * i for i in range(7)}
    obs["gripper"] = 0.5  # state only, no camera
    with pytest.raises(ValueError, match="requires at least one camera"):
        asyncio.run(p.get_actions(obs, "go"))


def test_robot_panda_sugar_applies_builtin_mapping():
    """robot='panda' auto-maps DROID columns to Panda actuators."""
    client = FakeClient(_droid_chunk())
    p = Cosmos3Policy(embodiment="droid", client=client, robot="panda")
    p.set_robot_state_keys([f"joint{i}" for i in range(1, 8)] + ["finger_joint1"])
    obs = {
        "observation/wrist_image_left": np.zeros((360, 640, 3), np.uint8),
        "observation/exterior_image_1_left": np.zeros((360, 640, 3), np.uint8),
        "observation/exterior_image_2_left": np.zeros((360, 640, 3), np.uint8),
    }
    for i in range(1, 8):
        obs[f"joint{i}"] = 0.1 * i
    obs["finger_joint1"] = 0.02
    out = asyncio.run(p.get_actions(obs, "go"))
    assert set(out[0].keys()) == {f"joint{i}" for i in range(1, 8)} | {"finger_joint1"}


def test_explicit_action_mapping_overrides_robot_sugar():
    p = Cosmos3Policy(
        embodiment="droid",
        client=FakeClient(_droid_chunk()),
        robot="panda",
        action_mapping={"gripper": "grip"},  # explicit wins
    )
    assert p._action_mapping == {"gripper": "grip"}


def test_unknown_robot_raises_at_construction():
    """robot= with an unknown / typo'd name fails fast instead of silently
    keeping the raw DROID layout (which the user's robot would then ignore
    in send_action). Pins AGENTS.md key convention #6 "No silent defaults on
    error" and prevents a one-way-door regression in the public ``robot=``
    constructor kwarg.
    """
    client = FakeClient(_droid_chunk())
    with pytest.raises(ValueError, match="Unknown robot 'pannda'"):
        Cosmos3Policy(embodiment="droid", client=client, robot="pannda")
    # Also: a structurally-valid but unsupported robot name (e.g. so100) is
    # rejected, listing the available built-in mappings in the error message.
    with pytest.raises(ValueError, match="Available built-in mappings"):
        Cosmos3Policy(embodiment="droid", client=client, robot="so100")


def test_pretrained_name_or_path_stored_not_dropped():
    """create_policy("nvidia/Cosmos3-Nano-Policy-DROID") passes
    pretrained_name_or_path through the registry resolver; the policy must
    store it for introspection rather than silently dropping it via **kwargs.
    Pins AGENTS.md PR-#86 "Reject silently-dropped kwargs" and prevents the
    one-way-door regression where the model-id smart-string surface is
    advertised but the kwarg is silently discarded.
    """
    client = FakeClient(_droid_chunk())
    policy = Cosmos3Policy(
        embodiment="droid",
        client=client,
        pretrained_name_or_path="nvidia/Cosmos3-Nano-Policy-DROID",
    )
    assert policy.pretrained_name_or_path == "nvidia/Cosmos3-Nano-Policy-DROID"


def test_unexpected_kwargs_rejected():
    """Cosmos3Policy no longer accepts **kwargs — a typo'd kwarg like
    actoin_mapping (note the typo) must raise TypeError instead of being
    silently swallowed.
    """
    client = FakeClient(_droid_chunk())
    # Build the typo'd kwarg dynamically so static analysis doesn't flag the
    # intentionally-misspelled parameter name; the runtime behavior (TypeError
    # on an unsupported kwarg) is exactly what we're asserting.
    bad_kwargs = {"actoin_mapping": {"joint_0": "shoulder_pan"}}  # typo on purpose
    with pytest.raises(TypeError):
        Cosmos3Policy(embodiment="droid", client=client, **bad_kwargs)


def test_client_connection_error_has_actionable_hint():
    """When the server is down, infer() raises ConnectionError naming the
    server-start command (no cryptic Errno 111).

    The Cosmos3WebsocketClient now uses a self-contained raw transport (no
    ``openpi-client`` dependency), so this test only needs ``websockets``.
    """
    pytest.importorskip("websockets", reason="websockets needed for the raw transport")
    from strands_robots.policies.cosmos3.client import Cosmos3WebsocketClient

    # Port 1 is reserved/unused -> connection refused on first lazy connect.
    client = Cosmos3WebsocketClient(host="127.0.0.1", port=1)
    with pytest.raises(ConnectionError) as ei:
        client.infer({"prompt": "x"})
    msg = str(ei.value)
    assert "action_policy_server_robolab" in msg
    assert "ws://127.0.0.1:1" in msg
    assert "healthz" in msg


def test_raw_transport_is_only_supported_transport():
    """The vendored raw msgpack+websockets transport is the only supported
    transport. Constructing the client never requires ``openpi-client``;
    legacy ``transport='openpi'`` / ``'auto'`` values are accepted (with a
    deprecation warning) and silently treated as ``'raw'`` so existing call
    sites keep working through the cleanup window."""
    pytest.importorskip("websockets", reason="websockets needed for raw transport")
    from strands_robots.policies.cosmos3.client import Cosmos3WebsocketClient

    # Default: raw.
    c = Cosmos3WebsocketClient(host="127.0.0.1", port=8000)
    assert c.transport == "raw"

    # Explicit raw.
    c = Cosmos3WebsocketClient(host="127.0.0.1", port=8000, transport="raw")
    assert c.transport == "raw"

    # Legacy values are coerced to raw (back-compat shim).
    for legacy in ("openpi", "auto"):
        c = Cosmos3WebsocketClient(host="127.0.0.1", port=8000, transport=legacy)
        assert c.transport == "raw", f"transport={legacy!r} must be coerced to 'raw'"

    # The vendored packer round-trips numpy arrays (version-agnostic).
    import numpy as np

    from strands_robots.policies.cosmos3 import _msgpack_numpy as mnp

    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    back = mnp.unpackb(mnp.packb({"a": arr}))
    assert back["a"].shape == (2, 3)
    assert np.allclose(back["a"], arr)


def test_cosmos3_policy_transport_param():
    """The ``transport`` kwarg remains in the public signature as a
    deprecated back-compat shim (any value is treated as 'raw')."""
    import inspect

    from strands_robots.policies.cosmos3 import Cosmos3Policy

    assert "transport" in inspect.signature(Cosmos3Policy.__init__).parameters
