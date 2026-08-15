# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``DiffusersKimodoAgent`` sampling and pipeline-output extraction.

Covers ``strands_robots.policies.kimodo._diffusers_agent``: how the agent
loads the diffusers pipeline, how it reads the sampled ``qpos`` back out of the
pipeline output, and how it refuses an output that carries no ``motion`` field.

The pipeline itself is stubbed through the module's own import seam -- the
agent does ``from diffusers import DiffusionPipeline`` inside ``__init__``, so
a stub registered under that name is what it loads. This deliberately avoids
touching the real ``DiffusionPipeline``, whose import pulls the whole diffusers
model graph that keeping this agent in its own module exists to defer.

The real ``BaseOutput`` is used as-is, because the refusal path turns on a
property of it: a diffusers pipeline output subclasses ``OrderedDict``, and
that mapping-ness is what decides whether the refusal is reachable at all.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from tests.mocks.torch_mock import real_torch_installed

pytest.importorskip("diffusers", reason="Kimodo sampling needs the [kimodo] extra")

from diffusers.utils import BaseOutput  # noqa: E402

from strands_robots.policies.kimodo._diffusers_agent import DiffusersKimodoAgent  # noqa: E402
from strands_robots.policies.kimodo.config import KimodoConfig  # noqa: E402

# The agent reads ``torch.float16``/``bfloat16`` and builds a ``torch.Generator``,
# which are outside the test-suite torch mock's surface.
pytestmark = pytest.mark.skipif(
    not real_torch_installed(),
    reason="needs real torch: the agent reads fp16/bf16 dtypes and seeds a torch.Generator",
)

_NUM_JOINTS = 29
_QPOS_WIDTH = 7 + _NUM_JOINTS


@dataclass
class _KimodoOutput(BaseOutput):
    """A pipeline output shaped like the Kimodo checkpoint's own."""

    motion: object = None


@dataclass
class _SampleOutput(BaseOutput):
    """A pipeline output using diffusers' conventional field name instead."""

    sample: object = None


class _BareOutput:
    """A non-mapping output object, for the ``__dict__`` refusal branch."""

    def __init__(self) -> None:
        self.frames = 3


class _StubPipe:
    """Stands in for a loaded ``DiffusionPipeline``."""

    def __init__(self, output: object) -> None:
        self._output = output
        self.moved_to: str | None = None
        self.load_kwargs: dict[str, object] = {}
        self.calls: list[dict[str, object]] = []

    def to(self, device: str) -> _StubPipe:
        self.moved_to = device
        return self

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self._output


def _motion_tensor(num_frames: int = 4, batched: bool = False):
    """A deterministic ``qpos`` tensor in the sampler's native layout."""
    import torch

    arr = np.zeros((num_frames, _QPOS_WIDTH), dtype=np.float64)
    arr[:, 6] = 1.0  # identity root quaternion
    for frame in range(num_frames):
        arr[frame, 7:] = np.linspace(0.0, 1.0, _NUM_JOINTS) * frame
    tensor = torch.from_numpy(arr)
    return tensor.unsqueeze(0) if batched else tensor


@pytest.fixture
def load_agent(monkeypatch):
    """Build an agent whose pipeline yields *output*, with no weights loaded.

    Registers a stub ``diffusers`` module carrying only ``DiffusionPipeline``,
    which is the single name the agent imports from it.

    Returns:
        A callable ``(output, **config_kwargs) -> (agent, stub_pipe)``. The stub
        pipe records the loader kwargs on ``load_kwargs`` and every sampling
        call on ``calls``.
    """

    def _load(output: object, **config_kwargs):
        config_kwargs.setdefault("device", "cpu")
        pipe = _StubPipe(output)

        class _StubDiffusionPipeline:
            @staticmethod
            def from_pretrained(model_id: str, **kwargs: object) -> _StubPipe:
                pipe.load_kwargs = {"model_id": model_id, **kwargs}
                return pipe

        stub_module = types.ModuleType("diffusers")
        stub_module.DiffusionPipeline = _StubDiffusionPipeline  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "diffusers", stub_module)

        return DiffusersKimodoAgent(KimodoConfig(**config_kwargs)), pipe

    return _load


def _sample(agent, **overrides):
    kwargs: dict[str, object] = {
        "prompt": "walking forward",
        "num_frames": 4,
        "diffusion_steps": 5,
        "guidance_scale": 7.5,
        "seed": None,
    }
    kwargs.update(overrides)
    return agent.sample(**kwargs)


# ----- The refusal path (regression) ----- #


def test_a_missing_motion_field_is_refused_with_the_model_id_and_the_fields_present(load_agent):
    """An output with no ``motion`` field raises the actionable refusal.

    This is the regression. A diffusers output is a ``dict`` subclass, so
    reading the field with a subscript raised a bare ``KeyError: 'motion'`` from
    inside the agent and never reached the refusal written for exactly this
    case, leaving the caller an exception that names neither the model nor the
    remedy.
    """
    agent, _pipe = load_agent(_SampleOutput(sample=_motion_tensor()), model_id="acme/not-kimodo")

    with pytest.raises(RuntimeError) as excinfo:
        _sample(agent)

    message = str(excinfo.value)
    assert "acme/not-kimodo" in message, message
    assert "'motion'" in message, message
    # It names what the output DID carry, so the caller can see the mismatch.
    assert "sample" in message, message
    # And it names a remedy, not just the symptom.
    assert "motion_agent=" in message, message


def test_a_diffusers_output_is_a_mapping_so_the_subscript_form_cannot_refuse():
    """Pins the third-party property the refusal path depends on.

    ``BaseOutput`` subclasses ``OrderedDict``, so ``output["motion"]`` raises
    ``KeyError`` for a field the checkpoint omits, while ``output.get("motion")``
    reports it as absent. If a diffusers output ever stops being a mapping, the
    reachability argument above changes and this fails.
    """
    output = _SampleOutput(sample=object())

    assert issubclass(BaseOutput, dict)
    assert isinstance(output, dict)
    assert getattr(output, "motion", None) is None
    assert output.get("motion") is None
    with pytest.raises(KeyError):
        output["motion"]


def test_a_non_mapping_output_without_a_motion_field_is_also_refused(load_agent):
    """The refusal also covers an output object that is not a mapping at all."""
    agent, _pipe = load_agent(_BareOutput())

    with pytest.raises(RuntimeError) as excinfo:
        _sample(agent)

    message = str(excinfo.value)
    assert "_BareOutput" in message, message
    assert "frames" in message, message


# ----- The sampling path ----- #


def test_the_sampled_qpos_is_returned_as_a_float32_array_from_the_motion_field(load_agent):
    """Happy path: the ``motion`` field becomes a float32 ``(T, 7+29)`` array."""
    expected = _motion_tensor(num_frames=4)
    agent, pipe = load_agent(_KimodoOutput(motion=expected))

    out = _sample(agent, num_frames=4)

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (4, _QPOS_WIDTH)
    np.testing.assert_allclose(out, expected.numpy().astype(np.float32))
    # The sampling knobs reach the pipeline under its own parameter names.
    call = pipe.calls[0]
    assert call["prompt"] == "walking forward"
    assert call["num_inference_steps"] == 5
    assert call["guidance_scale"] == 7.5
    assert call["num_frames"] == 4


def test_a_batched_output_is_squeezed_to_two_dimensions(load_agent):
    """A ``(1, T, D)`` output is reduced to the ``(T, D)`` the policy expects."""
    agent, _pipe = load_agent(_KimodoOutput(motion=_motion_tensor(num_frames=6, batched=True)))

    out = _sample(agent, num_frames=6)

    assert out.shape == (6, _QPOS_WIDTH)


def test_a_seed_is_the_only_thing_that_builds_a_generator(load_agent):
    """``seed=None`` passes no generator; an int seeds a torch generator."""
    import torch

    agent, pipe = load_agent(_KimodoOutput(motion=_motion_tensor()))

    _sample(agent, seed=None)
    assert pipe.calls[0]["generator"] is None

    _sample(agent, seed=1234)
    generator = pipe.calls[1]["generator"]
    assert isinstance(generator, torch.Generator)
    assert generator.initial_seed() == 1234


# ----- Pipeline loading ----- #


@pytest.mark.parametrize(
    ("dtype_name", "torch_attr"),
    [("fp16", "float16"), ("bf16", "bfloat16"), ("fp32", "float32")],
)
def test_each_dtype_string_loads_the_pipeline_in_its_torch_dtype(load_agent, dtype_name, torch_attr):
    """The config's dtype string selects the torch dtype passed to the loader."""
    import torch

    _agent, pipe = load_agent(_KimodoOutput(motion=_motion_tensor()), dtype=dtype_name)

    assert pipe.load_kwargs["torch_dtype"] is getattr(torch, torch_attr)


def test_the_configured_model_id_cache_dir_and_trust_flag_reach_from_pretrained(load_agent):
    """Loader arguments are forwarded verbatim, including the trust gate.

    ``trust_remote_code`` is security-relevant: Kimodo ships a custom pipeline
    class, so the caller's decision has to be the one that reaches diffusers
    rather than a value this agent picks.
    """
    _agent, pipe = load_agent(
        _KimodoOutput(motion=_motion_tensor()),
        model_id="nvidia/Kimodo-G1-RP-v1",
        cache_dir="/tmp/kimodo-cache",
        trust_remote_code=False,
    )

    assert pipe.load_kwargs["model_id"] == "nvidia/Kimodo-G1-RP-v1"
    assert pipe.load_kwargs["cache_dir"] == "/tmp/kimodo-cache"
    assert pipe.load_kwargs["trust_remote_code"] is False


def test_an_unset_device_auto_selects_cuda_only_when_it_is_available(load_agent):
    """``device=None`` resolves to cuda when torch reports it, else cpu."""
    import torch

    _agent, pipe = load_agent(_KimodoOutput(motion=_motion_tensor()), device=None)

    assert pipe.moved_to == ("cuda" if torch.cuda.is_available() else "cpu")


def test_an_explicit_device_is_honored_over_the_auto_selection(load_agent):
    """A configured device wins, so a caller can pin the sampler to the CPU."""
    _agent, pipe = load_agent(_KimodoOutput(motion=_motion_tensor()), device="cpu")

    assert pipe.moved_to == "cpu"
