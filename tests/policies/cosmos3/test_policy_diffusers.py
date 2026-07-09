"""Unit tests for the Cosmos3Policy in-process diffusers backend.

No GPU, no model weights, no policy server: the native
:class:`diffusers.Cosmos3OmniPipeline` and :class:`diffusers.CosmosActionCondition`
are injected via the ``pipeline=`` / ``condition_cls=`` dependency-injection
seams on
:class:`~strands_robots.policies.cosmos3.policy_diffusers.Cosmos3DiffusersBackend`
(mirroring the ``client=`` injection the service-backend tests use).
"""

import asyncio
import types

import numpy as np
import pytest

from strands_robots.policies.base import Policy
from strands_robots.policies.cosmos3 import Cosmos3DiffusersBackend, Cosmos3Policy
from strands_robots.policies.cosmos3.embodiments import get_embodiment


class FakeCondition:
    """Stand-in for diffusers.CosmosActionCondition (records its kwargs)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _output(action=None, video="world", sound=None):
    """Mimic a Cosmos3OmniPipelineOutput (a simple attribute bag)."""
    return types.SimpleNamespace(action=action, video=video, sound=sound)


class FakePipeline:
    """Records pipeline calls; returns a canned Cosmos3OmniPipelineOutput.

    The native pipeline returns ``action`` as a ``list[torch.Tensor]`` (one
    ``[T, raw_action_dim]`` chunk). We emit a plain ``np.ndarray`` in a list -
    ``Cosmos3DiffusersBackend._to_numpy`` handles both tensors and arrays.
    """

    def __init__(self, t=32, d=10, video="world", sound=None, action_override="__default__"):
        rng = np.random.default_rng(0)
        if action_override == "__default__":
            self._action = [rng.uniform(-1.0, 1.0, (t, d)).astype(np.float32)]
        else:
            self._action = action_override
        self._video = video
        self._sound = sound
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _output(action=self._action, video=self._video, sound=self._sound)


def _obs_with_state_and_images():
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    obs = {
        "observation/wrist_image_left": img,
        "observation/exterior_image_1_left": img,
        "observation/exterior_image_2_left": img,
    }
    for i in range(7):
        obs[f"joint_{i}"] = float(i) * 0.1
    obs["gripper"] = 0.5
    return obs


def _make_diffusers_policy(pipeline=None, condition_cls=None, robot=None, mode="policy", **kw):
    """Build a Cosmos3Policy(backend='diffusers') with an injected fake backend."""
    pipeline = pipeline or FakePipeline()
    backend = Cosmos3DiffusersBackend(
        embodiment=get_embodiment("droid"),
        mode=mode,
        pipeline=pipeline,
        condition_cls=condition_cls or FakeCondition,
    )
    p = Cosmos3Policy(embodiment="droid", backend="diffusers", diffusers_backend=backend, robot=robot, mode=mode, **kw)
    return p, pipeline


def test_diffusers_backend_is_a_policy():
    p, _ = _make_diffusers_policy()
    assert isinstance(p, Policy)
    assert p.provider_name == "cosmos3"
    assert p.backend == "diffusers"


def test_diffusers_returns_raw_unified_action_columns():
    """backend='diffusers' returns the same list[dict] shape via the reused
    _unpack_actions, named by the embodiment's raw_action_layout (the model's
    native 10D unified action = 9D end-effector pose + 1D gripper)."""
    p, pipe = _make_diffusers_policy()
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    out = asyncio.run(p.get_actions(_obs_with_state_and_images(), "pick up the cube"))
    assert isinstance(out, list)
    assert len(out) == 32
    step = out[0]
    # Raw unified action layout: 3D translation + 6D rotation + grasp.
    assert set(step.keys()) == {"tx", "ty", "tz", "r0", "r1", "r2", "r3", "r4", "r5", "grasp"}
    assert all(isinstance(v, float) for v in step.values())
    # The native pipeline was driven once with the CosmosActionCondition.
    assert len(pipe.calls) == 1
    assert isinstance(pipe.calls[0]["action"], FakeCondition)


def test_last_rollout_carries_video_and_action():
    """The predicted world video/sound surface on last_rollout (non-breaking
    channel) - the get_actions return type stays list[dict]."""
    p, _ = _make_diffusers_policy(pipeline=FakePipeline(video="/tmp/world.mp4"))
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    assert p.last_rollout is None
    out = asyncio.run(p.get_actions(_obs_with_state_and_images(), "go"))
    assert isinstance(out, list) and len(out) == 32
    assert p.last_rollout is not None
    assert p.last_rollout["video"] == "/tmp/world.mp4"
    act = np.asarray(p.last_rollout["action"])
    assert act.shape == (32, 10)


def test_condition_params_use_embodiment_metadata():
    """The CosmosActionCondition is built with domain_name + chunk_size from the
    embodiment, and the pipeline is called with the condition + fps."""
    p, pipe = _make_diffusers_policy()
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    asyncio.run(p.get_actions(_obs_with_state_and_images(), "go"))
    call = pipe.calls[0]
    cond = call["action"]
    assert isinstance(cond, FakeCondition)
    assert cond.kwargs["mode"] == "policy"
    assert cond.kwargs["domain_name"] == "droid_lerobot"
    assert cond.kwargs["chunk_size"] == 32
    assert "image" in cond.kwargs  # policy mode conditions on the first frame
    assert call["fps"] == 15
    assert call["output_type"] == "np"


def test_service_backend_byte_identical_regression():
    """backend='service' (default) path is unchanged: it never touches the
    diffusers backend and returns the service action chunk verbatim."""

    class FakeClient:
        def __init__(self, action):
            self._action = action
            self.last_obs = None

        def infer(self, observation):
            self.last_obs = observation
            return {"action": self._action}

        def reset(self):
            pass

    action = np.arange(32 * 8, dtype=np.float32).reshape(32, 8)
    p = Cosmos3Policy(embodiment="droid", client=FakeClient(action.copy()), robot="panda")
    assert p.backend == "service"
    assert p.last_rollout is None  # service never populates the world channel
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    out = p.get_actions_sync(_obs_with_state_and_images(), "go")
    # Reconstruct the chunk from the per-step dicts and compare to the input.
    cols = [f"joint{i}" for i in range(1, 8)] + ["finger_joint1"]
    recon = np.asarray([[step[c] for c in cols] for step in out], dtype=np.float32)
    np.testing.assert_array_equal(recon, action)
    assert p.last_rollout is None


def test_missing_diffusers_raises_actionable_error(monkeypatch):
    """When native diffusers is not importable, constructing the diffusers
    backend (without injection) raises an actionable install error."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "diffusers" or name.startswith("diffusers."):
            raise ImportError("No module named 'diffusers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="diffusers"):
        Cosmos3DiffusersBackend(embodiment=get_embodiment("droid"))


def test_forward_dynamics_under_service_raises():
    """mode='forward_dynamics' under backend='service' raises a clear
    unsupported error (the RoboLab server serves only the policy surface)."""
    with pytest.raises(ValueError, match="only available with backend='diffusers'"):
        Cosmos3Policy(embodiment="droid", backend="service", mode="forward_dynamics")


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown Cosmos 3 backend"):
        Cosmos3Policy(embodiment="droid", backend="grpc")


def test_unknown_mode_raises_in_backend():
    with pytest.raises(ValueError, match="Unknown Cosmos 3 action mode"):
        Cosmos3DiffusersBackend(
            embodiment=get_embodiment("droid"),
            mode="teleport",
            pipeline=FakePipeline(),
            condition_cls=FakeCondition,
        )


def test_forward_dynamics_world_only_returns_empty_but_keeps_video():
    """forward_dynamics predicts world video only (no action chunk). get_actions
    returns [] but the world video is still captured on last_rollout."""
    pipe = FakePipeline(video="/tmp/fd.mp4", action_override=None)
    p, _ = _make_diffusers_policy(pipeline=pipe, mode="forward_dynamics")
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    raw = np.zeros((32, 10), dtype=np.float32)
    out = asyncio.run(p.get_actions(_obs_with_state_and_images(), "roll forward", raw_actions=raw))
    assert out == []
    assert p.last_rollout["video"] == "/tmp/fd.mp4"
    assert p.last_rollout["action"] is None


def test_forward_dynamics_requires_raw_actions():
    p, _ = _make_diffusers_policy(mode="forward_dynamics")
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    with pytest.raises(ValueError, match="raw_actions"):
        asyncio.run(p.get_actions(_obs_with_state_and_images(), "go"))


def test_inverse_dynamics_requires_video():
    p, _ = _make_diffusers_policy(mode="inverse_dynamics")
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    with pytest.raises(ValueError, match="observed video"):
        asyncio.run(p.get_actions(_obs_with_state_and_images(), "go"))


def test_inverse_dynamics_with_video_returns_action_chunk():
    """inverse_dynamics recovers the actions between frames of an observed video.

    Mirrors the live Thor run (a real Cosmos-predicted world video fed back
    yields a 32-step [tx,ty,tz,r0..r5,grasp] chunk). With a video supplied the
    mode returns the per-timestep actuator dicts, and the video the model was
    conditioned on is threaded into the CosmosActionCondition.
    """
    pipe = FakePipeline(t=32, d=10)
    p, _ = _make_diffusers_policy(pipeline=pipe, mode="inverse_dynamics")
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    observed = np.zeros((33, 360, 640, 3), dtype=np.uint8)
    out = asyncio.run(p.get_actions(_obs_with_state_and_images(), "recover actions", video=observed))
    assert len(out) == 32
    assert set(out[0].keys()) == {"tx", "ty", "tz", "r0", "r1", "r2", "r3", "r4", "r5", "grasp"}
    # The observed video must be threaded into the condition the pipeline saw.
    assert pipe.calls, "pipeline was not called"
    cond = pipe.calls[-1]["action"]
    assert cond.kwargs.get("video") is not None


def test_policy_mode_missing_action_raises():
    """A policy-mode run that returns no action field is surfaced, not silently
    swallowed (forward_dynamics is the only world-only mode)."""
    pipe = FakePipeline(action_override=None)
    p, _ = _make_diffusers_policy(pipeline=pipe)
    p.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    with pytest.raises(RuntimeError, match="no action field"):
        asyncio.run(p.get_actions(_obs_with_state_and_images(), "go"))


def test_reset_is_safe_in_process():
    """reset() is a no-op-safe hook for the in-process backend (no remote
    state, no exception)."""
    pipe = FakePipeline()
    p, _ = _make_diffusers_policy(pipeline=pipe)
    p.reset(seed=3)  # must not raise


def test_action_mapping_validates_against_raw_layout():
    """A diffusers-backend action_mapping is validated against raw_action_layout
    (not the service joint_pos layout); a service-only key is rejected."""
    backend = Cosmos3DiffusersBackend(
        embodiment=get_embodiment("droid"),
        pipeline=FakePipeline(),
        condition_cls=FakeCondition,
    )
    # "joint_0" is a service joint_pos column, not a raw unified-action column.
    with pytest.raises(ValueError, match="diffusers'-backend action layout"):
        Cosmos3Policy(
            embodiment="droid",
            backend="diffusers",
            diffusers_backend=backend,
            action_mapping={"joint_0": "j1"},
        )


# --- GPU-path regression tests ---------------------------------------------
# Bug A (safety checker) and Bug B (bf16 -> numpy) only surface when the real
# pipeline is loaded / when the pipeline returns the model's native bfloat16
# action tensors. The mocked happy-path tests above never exercise either, so
# these pin both against the unmocked code paths.


def test_to_numpy_upcasts_bfloat16_action_chunk():
    """Cosmos 3 runs in bfloat16; the output action tensor is bfloat16.

    ``np.asarray(bf16_tensor)`` raises ``TypeError: Got unsupported ScalarType
    BFloat16``. ``_to_numpy`` must up-cast half precision to float32 first so
    the shared ``_unpack_actions`` consumes the chunk.
    """
    import torch

    from strands_robots.policies.cosmos3.policy_diffusers import _to_numpy

    chunk = torch.tensor([[0.25, -0.5], [1.0, 2.0]], dtype=torch.bfloat16)
    arr = _to_numpy(chunk)
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, [[0.25, -0.5], [1.0, 2.0]], rtol=0, atol=1e-2)


def test_to_numpy_upcasts_float16_action_chunk():
    """float16 tensors are also unreadable by ``np.asarray`` and must up-cast."""
    import torch

    from strands_robots.policies.cosmos3.policy_diffusers import _to_numpy

    arr = _to_numpy(torch.tensor([1.5, -2.0], dtype=torch.float16))
    assert arr.dtype == np.float32
    np.testing.assert_allclose(arr, [1.5, -2.0])


def test_load_pipeline_disables_safety_checker_by_default(monkeypatch):
    """``Cosmos3OmniPipeline.__init__`` builds a ``CosmosSafetyChecker`` that
    hard-raises ``ImportError: cosmos_guardrail is not installed`` unless the
    heavy optional extra is present. The backend must pass
    ``enable_safety_checker=False`` to ``from_pretrained`` by default so the
    pipeline loads without ``cosmos_guardrail``.
    """
    import sys

    captured = {}

    class FakeOmniPipeline:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs
            return cls()

        def to(self, device):
            captured["device"] = device
            return self

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.Cosmos3OmniPipeline = FakeOmniPipeline
    fake_diffusers.CosmosActionCondition = FakeCondition
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    # No injected pipeline -> exercises the real _load_pipeline path.
    Cosmos3DiffusersBackend(embodiment=get_embodiment("droid"), device="cpu")
    assert captured["kwargs"].get("enable_safety_checker") is False


def test_load_pipeline_enables_safety_checker_when_requested(monkeypatch):
    """With ``enable_safety_checker=True`` the flag is NOT forced off, so a
    caller that installed ``cosmos_guardrail`` keeps the checker."""
    import sys

    captured = {}

    class FakeOmniPipeline:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            captured["kwargs"] = kwargs
            return cls()

        def to(self, device):
            return self

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.Cosmos3OmniPipeline = FakeOmniPipeline
    fake_diffusers.CosmosActionCondition = FakeCondition
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    Cosmos3DiffusersBackend(embodiment=get_embodiment("droid"), device="cpu", enable_safety_checker=True)
    assert "enable_safety_checker" not in captured["kwargs"]


# --- Raw-pipeline output contracts + dead-end errors -----------------------
# The FakePipeline above always emits a plain 2-D ``[T, D]`` action array, so
# the raw-pipeline paths that only surface with a real Cosmos3OmniPipeline are
# untested by the happy-path suite: the ``[num_chunks, T, D]`` squeeze in
# ``_extract_action`` (the model can emit a leading chunk dim), the tensor
# branch of ``_as_action_tensor`` (forward/inverse-dynamics driven with a
# ``torch.Tensor`` instead of a NumPy array), and the two ``ValueError``
# dead-end contracts (``_resolve_torch_dtype`` on an unknown dtype string,
# ``_first_frame`` on an observation carrying no ``(H, W, 3)`` frame). These
# pin those contracts, mirroring the bf16 / safety-checker GPU-path regression
# tests above.


def test_extract_action_squeezes_leading_num_chunks_dim():
    """A real ``Cosmos3OmniPipeline`` may emit ``action`` as a list holding a
    single ``[num_chunks, T, D]`` tensor; ``_extract_action`` must return the
    first chunk as ``[T, D]``. The mocked 2-D happy path never exercises this
    squeeze, so a regression here would silently mis-shape a real rollout chunk.
    """
    from strands_robots.policies.cosmos3.policy_diffusers import _extract_action

    chunk = np.arange(2 * 32 * 10, dtype=np.float32).reshape(2, 32, 10)
    arr = _extract_action([chunk], mode="policy")
    assert arr.shape == (32, 10)
    np.testing.assert_array_equal(arr, chunk[0])
    # A plain 2-D chunk passes through unchanged (the common case).
    flat = np.arange(32 * 10, dtype=np.float32).reshape(32, 10)
    np.testing.assert_array_equal(_extract_action([flat], mode="policy"), flat)


def test_extract_action_missing_field_is_mode_dependent():
    """``forward_dynamics`` is the only world-only mode: a missing/empty action
    field returns ``None`` there but is surfaced as a ``RuntimeError`` in the
    action-producing ``policy`` mode rather than silently swallowed."""
    from strands_robots.policies.cosmos3.policy_diffusers import _extract_action

    assert _extract_action(None, mode="forward_dynamics") is None
    assert _extract_action([], mode="forward_dynamics") is None
    with pytest.raises(RuntimeError, match="no action field"):
        _extract_action(None, mode="policy")
    with pytest.raises(RuntimeError, match="no action field"):
        _extract_action([], mode="policy")


def test_as_action_tensor_coerces_tensor_and_array_to_float32():
    """``raw_actions`` for forward/inverse-dynamics may arrive as a
    ``torch.Tensor`` (e.g. a chunk carried over from a prior rollout) or a NumPy
    array; both must coerce to a ``float32`` tensor. The forward-dynamics tests
    above only pass a NumPy array, leaving the tensor branch uncovered.
    """
    import torch

    from strands_robots.policies.cosmos3.policy_diffusers import Cosmos3DiffusersBackend

    from_tensor = Cosmos3DiffusersBackend._as_action_tensor(torch.tensor([[1.0, 2.0]], dtype=torch.float64))
    assert isinstance(from_tensor, torch.Tensor)
    assert from_tensor.dtype == torch.float32
    np.testing.assert_allclose(from_tensor.numpy(), [[1.0, 2.0]])

    from_array = Cosmos3DiffusersBackend._as_action_tensor(np.array([[3.0, 4.0]], dtype=np.float64))
    assert isinstance(from_array, torch.Tensor)
    assert from_array.dtype == torch.float32
    np.testing.assert_allclose(from_array.numpy(), [[3.0, 4.0]])


def test_resolve_torch_dtype_unknown_raises_actionable_error():
    """An unknown dtype string is a caller misconfiguration; it must raise a
    ``ValueError`` naming the offending value rather than falling through to a
    ``None`` dtype that would fail cryptically inside ``from_pretrained``."""
    import torch

    from strands_robots.policies.cosmos3.policy_diffusers import _resolve_torch_dtype

    assert _resolve_torch_dtype(torch, "bfloat16") is torch.bfloat16
    assert _resolve_torch_dtype(torch, "float32") is torch.float32
    with pytest.raises(ValueError, match="Unknown torch dtype 'not_a_dtype'"):
        _resolve_torch_dtype(torch, "not_a_dtype")


def test_first_frame_requires_an_hwc_camera_frame():
    """``_first_frame`` scans the embodiment camera keys and any image-like key
    for an ``(H, W, 3)`` array; an observation carrying only scalars or a
    wrong-shaped array must raise a ``ValueError`` that names both the keys it
    searched and the observation keys it saw (a dead-end error, not a crash
    deep inside the pipeline)."""
    backend = Cosmos3DiffusersBackend(
        embodiment=get_embodiment("droid"),
        pipeline=FakePipeline(),
        condition_cls=FakeCondition,
    )
    # No (H, W, 3) frame: a 2-D greyscale array and a scalar.
    bad = {
        "observation/wrist_image_left": np.zeros((360, 640), dtype=np.uint8),
        "observation/exterior_image_1_left": None,  # present-but-None key is skipped
        "joint_0": 0.1,
    }
    with pytest.raises(ValueError, match="requires at least one camera frame"):
        backend._first_frame(bad)
    # A valid HWC frame is returned as-is.
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert backend._first_frame({"observation/wrist_image_left": frame}).shape == (8, 8, 3)


def test_import_condition_cls_missing_diffusers_raises_install_hint(monkeypatch):
    """``forward_dynamics`` / ``inverse_dynamics`` import
    ``diffusers.CosmosActionCondition`` lazily via ``_import_condition_cls``;
    when native diffusers is absent it must raise the same actionable install
    error the pipeline import gives, not a bare ``ImportError`` from deep in the
    call. Mirrors ``test_missing_diffusers_raises_actionable_error`` for the
    condition-class import path."""
    import builtins

    # An injected pipeline + condition_cls lets __init__ skip the diffusers
    # import, so we exercise _import_condition_cls in isolation.
    backend = Cosmos3DiffusersBackend(
        embodiment=get_embodiment("droid"),
        pipeline=FakePipeline(),
        condition_cls=FakeCondition,
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "diffusers" or name.startswith("diffusers."):
            raise ImportError("No module named 'diffusers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="diffusers"):
        backend._import_condition_cls()
