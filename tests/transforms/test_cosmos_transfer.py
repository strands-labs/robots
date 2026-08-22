"""Cosmos-Transfer-style backend: the pipeline seam, its refusals, its hand-off.

The backend is a vendor-neutral adapter: it binds any object satisfying the
``generate(video, prompt=..., seed=...)`` protocol - injected constructed, or
named by a dotted import path resolved lazily - and refuses cleanly (through
``validate``, never an ``AttributeError``) when none is bound. The tests
exercise the seam with a fake pipeline, the same dependency-injection shape
``Cosmos3Policy`` uses for its ``client=`` / ``diffusers_backend=`` kwargs, so
none of this needs a generation model installed.
"""

import numpy as np
import pytest

from strands_robots.transforms import TransformSpec, derive_variant_seed, load_provenance
from strands_robots.transforms.cosmos_transfer import CosmosTransferTransform


class FakePipeline:
    """Minimal video2video pipeline: records calls, shifts pixels."""

    version = "fake-2.5"

    def __init__(self):
        self.calls = []

    def generate(self, video, prompt="", seed=None):
        """Record the call and return a deterministic pixel shift."""
        self.calls.append({"shape": video.shape, "prompt": prompt, "seed": seed})
        return np.clip(video.astype(np.int16) + 30, 0, 255).astype(np.uint8)


#: Module-level instance so the dotted-import-path seam has a real target.
FAKE_PIPELINE = FakePipeline()


def _frames(t: int = 3) -> np.ndarray:
    return np.full((t, 8, 8, 3), 100, dtype=np.uint8)


class TestPipelineSeam:
    def test_no_pipeline_is_a_validate_problem_not_a_crash(self, tmp_path):
        problems = CosmosTransferTransform().validate(TransformSpec())
        assert any("no video2video pipeline is bound" in p for p in problems)
        assert any("Cosmos-Transfer" in p for p in problems)

    def test_object_without_generate_is_refused(self):
        problems = CosmosTransferTransform(pipeline=object())._pipeline_problems()
        assert any("no callable generate()" in p for p in problems)

    def test_injected_pipeline_passes(self):
        assert CosmosTransferTransform(pipeline=FakePipeline())._pipeline_problems() == []

    def test_dotted_path_resolves_lazily(self):
        t = CosmosTransferTransform(pipeline="tests.transforms.test_cosmos_transfer:FAKE_PIPELINE")
        assert t._pipeline_problems() == []
        out = t.transform_frames("cam", _frames(), TransformSpec(prompt="night"), source_episode=0, variant=0)
        assert out.shape == _frames().shape

    def test_dotted_path_to_a_factory_is_constructed(self):
        t = CosmosTransferTransform(pipeline="tests.transforms.test_cosmos_transfer.FakePipeline")
        assert t._pipeline_problems() == []
        assert t.transform_version == "fake-2.5"

    def test_unresolvable_dotted_path_is_a_problem_not_a_raise(self):
        problems = CosmosTransferTransform(pipeline="no.such.module:thing")._pipeline_problems()
        assert any("did not resolve" in p for p in problems)

    def test_malformed_dotted_path_is_a_problem(self):
        problems = CosmosTransferTransform(pipeline="justoneword")._pipeline_problems()
        assert any("is not 'pkg.mod:attr'" in p for p in problems)

    def test_direct_frames_call_without_pipeline_raises_with_remedy(self):
        with pytest.raises(RuntimeError, match="no video2video pipeline"):
            CosmosTransferTransform().transform_frames("cam", _frames(), TransformSpec(), source_episode=0, variant=0)


class TestHandOff:
    def test_prompt_and_derived_seed_reach_the_pipeline(self):
        pipeline = FakePipeline()
        t = CosmosTransferTransform(pipeline=pipeline)
        spec = TransformSpec(prompt="the same scene at night", seed=11)
        t.transform_frames("cam", _frames(), spec, source_episode=2, variant=1)
        assert pipeline.calls == [
            {
                "shape": (3, 8, 8, 3),
                "prompt": "the same scene at night",
                "seed": derive_variant_seed(11, 2, 1),
            }
        ]

    def test_no_seed_stays_none(self):
        pipeline = FakePipeline()
        CosmosTransferTransform(pipeline=pipeline).transform_frames(
            "cam", _frames(), TransformSpec(), source_episode=0, variant=0
        )
        assert pipeline.calls[0]["seed"] is None

    def test_transform_version_reads_the_pipeline_never_guesses(self):
        assert CosmosTransferTransform(pipeline=FakePipeline()).transform_version == "fake-2.5"
        assert CosmosTransferTransform().transform_version == "unversioned"


class TestEndToEnd:
    def test_pipeline_pixels_land_in_a_provenance_marked_dataset(self, record_source_dataset, tmp_path):
        """The generic orchestration carries the fake pipeline's output to disk."""
        pytest.importorskip("lerobot")
        source_root = record_source_dataset([100])
        output_root = str(tmp_path / "cosmos_out")
        pipeline = FakePipeline()
        result = CosmosTransferTransform(pipeline=pipeline).transform(
            TransformSpec(source_root=source_root, output_root=output_root, prompt="cluttered kitchen", seed=5)
        )
        assert result.status == "success", result.message
        assert result.episodes_written == 1
        assert pipeline.calls and pipeline.calls[0]["prompt"] == "cluttered kitchen"
        records = load_provenance(output_root)
        assert records[0]["transform"] == "cosmos_transfer"
        assert records[0]["transform_version"] == "fake-2.5"
        assert records[0]["prompt"] == "cluttered kitchen"

    def test_wrong_shape_return_is_refused_loudly(self, record_source_dataset, tmp_path):
        """A pipeline that changes the stream's schema is a bug, not a dataset."""
        pytest.importorskip("lerobot")

        class WrongShapePipeline:
            def generate(self, video, prompt="", seed=None):
                return video[:, ::2, ::2, :]  # downscales - schema breach

        source_root = record_source_dataset([100])
        spec = TransformSpec(source_root=source_root, output_root=str(tmp_path / "cosmos_bad"))
        with pytest.raises(ValueError, match="a transform changes pixels, never the stream's schema"):
            CosmosTransferTransform(pipeline=WrongShapePipeline()).transform(spec)


class TestCallShapedAdapterExample:
    """The documented ``__call__``-shaped adapter is a working remedy, not prose.

    Diffusers-family video pipelines (including the diffusers-hosted Cosmos
    Transfer checkpoints) expose ``__call__``, return an object carrying
    ``.frames``, seed via ``generator=`` and emit ``float`` frames in
    ``[0, 1]``. ``_PIPELINE_HELP`` documents a thin adapter closing those
    gaps; these tests pin that the guidance is present in every pipeline
    refusal and that an adapter written exactly as documented satisfies the
    seam, so the example cannot rot into the guesswork it exists to remove.
    """

    def test_pipeline_refusals_carry_the_adapter_guidance(self):
        problems = CosmosTransferTransform().validate(TransformSpec())
        assert any("__call__-shaped" in p for p in problems)
        assert any(".frames" in p for p in problems)

    def test_documented_adapter_bridges_a_call_shaped_pipeline(self):
        torch = pytest.importorskip("torch")

        class FakeDiffusersOutput:
            def __init__(self, frames):
                self.frames = frames

        class FakeDiffusersPipe:
            """``__call__``-shaped, ``.frames``-returning, ``generator=``-seeded."""

            def __init__(self):
                self.calls = []

            def __call__(self, *, video, prompt, output_type, generator):
                self.calls.append({"n_frames": len(video), "prompt": prompt, "generator": generator})
                assert output_type == "np"
                batch = np.stack([np.asarray(f, dtype=np.float64) / 255.0 for f in video])
                return FakeDiffusersOutput(frames=[batch])  # float in [0, 1], batched

        pipe = FakeDiffusersPipe()

        # The adapter exactly as _PIPELINE_HELP and the module docstring show it.
        class Adapter:
            def generate(self, video, prompt="", seed=None):
                g = None if seed is None else torch.Generator().manual_seed(seed)
                f = pipe(video=list(video), prompt=prompt, output_type="np", generator=g).frames[0]
                return np.clip(np.round(np.asarray(f) * 255.0), 0, 255).astype(np.uint8)

        transform = CosmosTransferTransform(pipeline=Adapter())
        assert transform._pipeline_problems() == []

        spec = TransformSpec(prompt="the same scene at night", seed=11)
        out = transform.transform_frames("cam", _frames(), spec, source_episode=2, variant=1)
        # The round-trip through the float [0, 1] convention lands back on the contract.
        assert out.shape == _frames().shape
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, _frames())
        # The derived seed reaches the pipeline as a generator, not a raw int.
        assert pipe.calls[0]["prompt"] == "the same scene at night"
        assert pipe.calls[0]["generator"].initial_seed() == derive_variant_seed(11, 2, 1)

        # An unseeded spec stays unseeded: no generator is fabricated for None.
        transform.transform_frames("cam", _frames(), TransformSpec(), source_episode=0, variant=0)
        assert pipe.calls[1]["generator"] is None
