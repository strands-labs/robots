"""The documented native-runtime adapter must apply the seed it is handed.

``docs/policies/kimodo.md`` documents a ``KimodoMotionAgent`` that drives
NVIDIA's own ``kimodo`` runtime, because that checkpoint is not published in
diffusers pipeline layout and so cannot be loaded by the built-in agent. The
documented adapter is the supported route to the real weights, which makes its
seeding behaviour part of the contract rather than an implementation detail.

Seeding is the one input such an adapter is most likely to drop. The runtime
draws its initial noise with ``torch.randn(shape, device=...)`` and accepts no
generator or seed argument, so the only way to make a sample reproducible is to
seed the global torch generator before the call. An adapter that accepts ``seed``
and ignores it still satisfies :class:`KimodoMotionAgent` - nothing raises, and
every request simply samples fresh noise.

That is not a cosmetic loss. ``eval_policy`` derives one seed per episode and
hands it to ``reset()``, and :class:`KimodoPolicy` re-samples whenever a sampler
input changes, so an unseeded adapter gives every episode an independent motion
that no seed can reproduce - a silently unreproducible evaluation reported as a
successful one.

So this executes the documented adapter itself, against a stub runtime that
draws from a global stream the way the real one does, and pins that the same seed
yields the same motion while different seeds diverge.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.kimodo.policy import KimodoMotionAgent

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_KIMODO_DOC = _REPO_ROOT / "docs" / "policies" / "kimodo.md"
_ADAPTER_HEADING = "## Driving the NVIDIA checkpoint"
_ADAPTER_CLASS = "class NativeKimodoAgent:"
_FRAME_WIDTH = 7 + 29


def documented_adapter_source() -> str:
    """Extract the documented native-runtime adapter from the Kimodo docs.

    Returns:
        The Python source of the documented example up to (but excluding) the
        ``run_policy`` call that follows it, so the adapter can be executed on
        its own.
    """
    doc = _KIMODO_DOC.read_text(encoding="utf-8")
    heading = doc.index(_ADAPTER_HEADING)
    block = re.search(r"```python\n(.*?)```", doc[heading:], re.DOTALL)
    assert block is not None, f"{_ADAPTER_HEADING} documents no python block"
    source = block.group(1)
    assert _ADAPTER_CLASS in source, f"{_ADAPTER_HEADING} documents no {_ADAPTER_CLASS}"
    return source[: source.index("sim.run_policy(")]


class _GlobalNoiseStream:
    """Stands in for the global torch generator the runtime draws noise from."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def draw(self, shape: tuple[int, ...]) -> np.ndarray:
        return self._rng.standard_normal(shape)


def _stub_module(name: str) -> Any:
    """Build an empty module object typed loosely enough to attach stubs to."""
    return types.ModuleType(name)


def _install_runtime_stubs(monkeypatch: pytest.MonkeyPatch, stream: _GlobalNoiseStream) -> None:
    """Install ``torch`` and ``kimodo`` stubs the documented adapter can import.

    The stubs reproduce the two properties that matter: ``manual_seed`` reseeds a
    process-global noise stream, and the runtime's sampler draws from that same
    stream without taking a seed of its own.

    Args:
        monkeypatch: Fixture used to install the stub modules.
        stream: The shared noise stream both stubs read and reseed.
    """
    torch_stub = _stub_module("torch")
    torch_stub.manual_seed = stream.reseed

    class _Converter:
        def __init__(self, skeleton: object) -> None:
            self._skeleton = skeleton

        def dict_to_qpos(self, output: dict, device: str | None = None) -> np.ndarray:
            # The real converter returns a batch, which is why the documented
            # adapter indexes a 3-D result.
            return output["motion"][None, ...]

    class _Model:
        skeleton = object()

        def __call__(
            self,
            prompts: list[str],
            num_frames: list[int],
            num_denoising_steps: int,
            num_samples: int | None = None,
            return_numpy: bool = False,
        ) -> dict:
            return {"motion": stream.draw((num_frames[0], _FRAME_WIDTH))}

    mujoco_stub = _stub_module("kimodo.exports.mujoco")
    mujoco_stub.MujocoQposConverter = _Converter
    load_model_stub = _stub_module("kimodo.model.load_model")
    load_model_stub.load_model = lambda modelname, device=None: _Model()

    for name, module in (
        ("torch", torch_stub),
        ("kimodo", _stub_module("kimodo")),
        ("kimodo.exports", _stub_module("kimodo.exports")),
        ("kimodo.exports.mujoco", mujoco_stub),
        ("kimodo.model", _stub_module("kimodo.model")),
        ("kimodo.model.load_model", load_model_stub),
    ):
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def documented_agent(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build the adapter exactly as ``docs/policies/kimodo.md`` writes it."""
    _install_runtime_stubs(monkeypatch, _GlobalNoiseStream())
    namespace: dict[str, Any] = {}
    exec(compile(documented_adapter_source(), str(_KIMODO_DOC), "exec"), namespace)  # noqa: S102
    return namespace["NativeKimodoAgent"](device="cpu")


def _sample(agent: Any, seed: int | None) -> np.ndarray:
    return agent.sample("a person walking forward", 12, 8, 2.0, seed)


class TestTheDocumentedAdapterHonoursItsSeed:
    """The documented adapter is the supported route to the real checkpoint."""

    def test_the_same_seed_reproduces_the_same_motion(self, documented_agent: Any) -> None:
        """Two requests carrying one seed must return one motion.

        Without this, a per-episode seed reaches the sampler and changes nothing.
        """
        first = _sample(documented_agent, 7)
        second = _sample(documented_agent, 7)
        assert np.array_equal(first, second), (
            "the documented adapter returned different motions for seed=7, so it accepts the "
            "seed and does not apply it to the generator the runtime draws from"
        )

    def test_a_different_seed_produces_a_different_motion(self, documented_agent: Any) -> None:
        """Seeding must select a sample, not pin every request to one motion."""
        assert not np.array_equal(_sample(documented_agent, 7), _sample(documented_agent, 99))

    def test_an_unseeded_request_samples_fresh_noise(self, documented_agent: Any) -> None:
        """``seed=None`` is documented as fresh each call, and must stay so."""
        assert not np.array_equal(_sample(documented_agent, None), _sample(documented_agent, None))

    def test_the_documented_adapter_satisfies_the_agent_protocol(self, documented_agent: Any) -> None:
        """A documented adapter that the policy would reject is not a recipe."""
        assert isinstance(documented_agent, KimodoMotionAgent)

    def test_the_documented_adapter_returns_one_qpos_frame_per_requested_frame(self, documented_agent: Any) -> None:
        """The protocol's return shape is ``(num_frames, 7+29)`` float32."""
        motion = _sample(documented_agent, 7)
        assert motion.shape == (12, _FRAME_WIDTH)
        assert motion.dtype == np.float32
