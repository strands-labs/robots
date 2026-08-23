"""Real-weights Cosmos-Transfer run through the ``CosmosTransferTransform`` pipeline seam.

PR #2480 verified the transform surface end to end with a *fake* pipeline and
deliberately stopped at the injection seam; this suite (issue #2530) binds the
real diffusers-hosted Cosmos-Transfer2.5 checkpoints to that same seam once,
so the generative path has a real-inference test the way every policy backend
does (AGENTS.md convention 8). One source episode is recorded, transformed
through a real video2video generation, reopened, and held to the surface's
contract: schema parity, action/state byte-equality, provenance rows naming
the real pipeline version, pixels changed - plus the determinism claim the
provenance ``seed`` field makes, measured against a second identically-seeded
run.

Requirements (why this lives in ``tests_integ/``):

* a CUDA GPU with >= 20 GB free (measured peak ~19 GB on a 24 GB NVIDIA L4
  with ``enable_model_cpu_offload``), and ~40 GB of HF cache for the weights;
* ``diffusers >= 0.40`` (first release shipping
  ``Cosmos2_5_TransferPipeline``), ``transformers`` with Qwen2.5-VL,
  ``accelerate``, ``opencv-python``, and ``cosmos_guardrail`` (the pipeline
  refuses to run without the safety checker);
* Hugging Face access to ``nvidia/Cosmos-Transfer2.5-2B`` and
  ``nvidia/Cosmos-1.0-Guardrail``. Both are auto-gated behind the **NVIDIA
  Open Model License**; the account whose token runs this test must have
  accepted it, and the deployment must verify the license fits its use -
  the same caveat ``CosmosTransferTransform.validate()`` states.

Run with::

    hatch run test-integ tests_integ/transforms/ -m gpu

Determinism note (acceptance criterion 2 of #2530): on the stack this was
pinned against (L4, torch 2.11 + CUDA 13, diffusers 0.40.0, bf16,
``enable_model_cpu_offload``), two runs with the same spec ``seed`` produced
**byte-identical** uint8 frames - the pipeline seeds all noise from the
``generator=`` the adapter derives from the seed, and no nondeterministic
CUDA kernel surfaced at these shapes. So the provenance ``seed`` field's
reproducibility claim is honest for this real backend and the test asserts
byte-equality outright. If a future kernel/hardware change breaks that, this
test is the tripwire, and the honest fix is to document the divergence here
rather than to loosen the assertion silently.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("lerobot")
torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

if not hasattr(diffusers, "Cosmos2_5_TransferPipeline"):
    pytest.skip(
        f"diffusers {diffusers.__version__} has no Cosmos2_5_TransferPipeline (needs >= 0.40)",
        allow_module_level=True,
    )
if not torch.cuda.is_available():
    pytest.skip("Cosmos-Transfer2.5 inference requires a CUDA GPU", allow_module_level=True)

from strands_robots.transforms import TransformSpec, create_transform, load_provenance  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.timeout(1800)]

_MODEL_ID = "nvidia/Cosmos-Transfer2.5-2B"
_PIPELINE_REVISION = "diffusers/general"
_CONTROLNET_REVISION = "diffusers/controlnet/general/edge"

# The generation budget: enough steps for real denoising through every module,
# small enough that the whole suite is two ~1-minute generations on an L4.
# The transform contract under test (seam, provenance, pass-through,
# determinism) does not depend on sample quality.
_WORK_H, _WORK_W = 256, 256  # must be divisible by 16 (pipeline check_inputs)
_STEPS = 10

# Source clip: (T-1) % 4 == 0 keeps the frame count aligned with the Wan
# VAE's temporal downsample factor so no frames are padded then trimmed.
_FRAMES, _H, _W = 17, 64, 64

_PROMPT = "the same robot workspace re-rendered as a cluttered kitchen at night"


class CosmosTransfer25EdgeAdapter:
    """The adapter the pipeline seam documents, bound to real Transfer2.5 weights.

    ``CosmosTransferTransform`` hands over ``(T, H, W, 3) uint8`` pixels, a
    prompt and a derived seed; this adapter owns everything the vendor
    pipeline needs beyond that (exactly what #2530 asks to be measured):

    * **control derivation** - Transfer2.5 conditions on control maps, not on
      the source video directly; the edge-controlnet variant takes per-frame
      Canny maps derived from the source pixels (the upstream example's
      recipe), so the generated clip follows the source clip's structure;
    * **resolution round-trip** - the pipeline requires height/width divisible
      by 16 and is trained at video resolutions, so frames are upscaled to a
      work resolution and the generated frames resized back to the source
      shape (the transform contract requires shape/dtype parity);
    * **seeding** - the derived per-variant seed becomes a CPU
      ``torch.Generator``, the pipeline's only noise source;
    * **version** - recorded into every provenance row, naming the exact
      checkpoints + diffusers version that produced the pixels.
    """

    def __init__(self, pipe: Any) -> None:
        self._pipe = pipe
        self.version = f"{_MODEL_ID}@{_PIPELINE_REVISION}+{_CONTROLNET_REVISION} (diffusers {diffusers.__version__})"

    def generate(self, video: np.ndarray, prompt: str = "", seed: int | None = None) -> np.ndarray:
        import cv2
        from PIL import Image

        t, h, w, _ = video.shape
        resized = [cv2.resize(f, (_WORK_W, _WORK_H), interpolation=cv2.INTER_LINEAR) for f in video]
        controls = [np.repeat(cv2.Canny(f, 100, 200)[..., None], 3, axis=-1) for f in resized]
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(int(seed))
        out = self._pipe(
            controls=[Image.fromarray(c) for c in controls],
            prompt=prompt,
            height=_WORK_H,
            width=_WORK_W,
            num_frames=t,
            num_frames_per_chunk=t,
            num_inference_steps=_STEPS,
            generator=generator,
            output_type="np",
        ).frames[0]
        out8 = np.clip(np.round(np.asarray(out) * 255.0), 0, 255).astype(np.uint8)[:t]
        return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in out8])


def _load_real_pipeline_or_skip() -> Any:
    """Load the real Transfer2.5 pipeline (~23 GB of weights), or skip.

    A ``*_or_skip`` helper rather than a try-bind-then-use, so the caller's
    binding is unconditional on every analyzable path (see
    tests/test_optional_dependency_skips_bind_their_names.py).
    """
    from diffusers import AutoModel
    from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError

    # Attribute access on the importorskip'd module rather than a from-import:
    # the module-level gate above already proved the attribute exists, and a
    # static from-import would need diffusers >= 0.40 in the lint env too.
    transfer_pipeline_cls = diffusers.Cosmos2_5_TransferPipeline

    try:
        controlnet = AutoModel.from_pretrained(_MODEL_ID, revision=_CONTROLNET_REVISION, torch_dtype=torch.bfloat16)
        return transfer_pipeline_cls.from_pretrained(
            _MODEL_ID, controlnet=controlnet, revision=_PIPELINE_REVISION, torch_dtype=torch.bfloat16
        )
    except (GatedRepoError, LocalEntryNotFoundError, ImportError, OSError) as exc:
        raise pytest.skip.Exception(f"real Cosmos-Transfer2.5 weights unavailable here: {exc}") from exc


@pytest.fixture(scope="module")
def real_pipeline():
    """The real Transfer2.5 pipeline, loaded once for the module."""
    pipe = _load_real_pipeline_or_skip()
    pipe.enable_model_cpu_offload()
    return pipe


def _record_source_dataset(root: str) -> str:
    """Record one real LeRobot episode: a moving square, so Canny has real edges.

    Actions and states carry distinct non-zero values per frame so the
    byte-equality assertion cannot pass vacuously on zeros.
    """
    from strands_robots.dataset_recorder import DatasetRecorder

    recorder = DatasetRecorder.create(
        "local/source",
        fps=10,
        camera_keys=["cam"],
        camera_dims={"cam": (_H, _W)},
        joint_names=["j1", "j2"],
        root=root,
    )
    for t in range(_FRAMES):
        img = np.full((_H, _W, 3), 96, dtype=np.uint8)
        x = 8 + 2 * t
        img[20:44, x : x + 16] = 255
        observation = {"j1": 0.125 + t, "j2": -0.25 - t, "cam": img}
        action = {"j1": 0.5 + t, "j2": 1.5 + t}
        recorder.add_frame(observation, action, task="move the square")
    recorder.save_episode()
    recorder.finalize()
    return root


@pytest.fixture(scope="module")
def transformed_twice(real_pipeline, tmp_path_factory):
    """One recorded episode, transformed twice under the same seed.

    Two full ``transform()`` runs (one generation each) - the first carries
    every contract assertion, the pair carries the determinism claim.
    """
    tmp_path = tmp_path_factory.mktemp("cosmos_transfer_real")
    source_root = _record_source_dataset(str(tmp_path / "source"))

    outputs = []
    for run in ("a", "b"):
        transform = create_transform("cosmos_transfer", pipeline=CosmosTransfer25EdgeAdapter(real_pipeline))
        spec = TransformSpec(
            source_root=source_root,
            output_root=str(tmp_path / f"augmented_{run}"),
            variants_per_episode=1,
            seed=42,
            prompt=_PROMPT,
        )
        assert transform.validate(spec) == []
        result = transform.transform(spec)
        assert result.status == "success", result.message
        outputs.append(result)
    return source_root, outputs


def _episode_column(ds, episode: int, key: str) -> np.ndarray:
    info = ds.meta.episodes[episode]
    start, stop = int(info["dataset_from_index"]), int(info["dataset_to_index"])
    return np.stack([ds[i][key].numpy() for i in range(start, stop)])


def _open(repo_id: str, root: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id=repo_id, root=root)


class TestRealWeightsRoundTrip:
    """Acceptance criterion 1 of #2530: the fake-pipeline round trip, on real weights."""

    def test_counts(self, transformed_twice):
        _, (result, _) = transformed_twice
        assert result.episodes_read == 1
        assert result.episodes_written == 1
        assert result.episodes_discarded == 0

    def test_schema_parity(self, transformed_twice):
        source_root, (result, _) = transformed_twice
        source = _open("local/source", source_root)
        output = _open("local/augmented", result.output_root)
        src_features, out_features = dict(source.meta.features), dict(output.meta.features)
        assert set(src_features) == set(out_features)
        for key in ("observation.state", "action", "observation.images.cam"):
            assert tuple(src_features[key]["shape"]) == tuple(out_features[key]["shape"]), key
            assert src_features[key]["dtype"] == out_features[key]["dtype"], key
        assert source.meta.fps == output.meta.fps

    def test_action_and_state_columns_pass_through_byte_identical(self, transformed_twice):
        source_root, (result, _) = transformed_twice
        source = _open("local/source", source_root)
        output = _open("local/augmented", result.output_root)
        for key in ("action", "observation.state"):
            src_col = _episode_column(source, 0, key)
            out_col = _episode_column(output, 0, key)
            assert src_col.tobytes() == out_col.tobytes(), key
            assert np.any(src_col != 0.0), key  # not vacuous on zeros

    def test_pixels_changed(self, transformed_twice):
        source_root, (result, _) = transformed_twice
        source = _open("local/source", source_root)
        output = _open("local/augmented", result.output_root)
        src_img = _episode_column(source, 0, "observation.images.cam")
        out_img = _episode_column(output, 0, "observation.images.cam")
        # A real generation replaces the rendering wholesale; the measured
        # mean shift dwarfs codec noise by orders of magnitude.
        assert abs(float(src_img.mean()) - float(out_img.mean())) > 0.01

    def test_provenance_names_the_real_pipeline(self, transformed_twice):
        _, (result, _) = transformed_twice
        records = load_provenance(result.output_root)
        assert len(records) == 1
        record = records[0]
        assert record["synthetic"] is True
        assert record["transform"] == "cosmos_transfer"
        assert record["source_episode_index"] == 0
        assert record["prompt"] == _PROMPT
        assert record["seed"] is not None
        # The version pins the real generator, never "unversioned".
        assert _MODEL_ID in record["transform_version"]
        assert _CONTROLNET_REVISION in record["transform_version"]


class TestRealWeightsDeterminism:
    """Acceptance criterion 2 of #2530: the provenance ``seed`` field is honest."""

    def test_same_seed_twice_is_byte_identical(self, transformed_twice):
        _, (result_a, result_b) = transformed_twice
        out_a = _open("local/augmented", result_a.output_root)
        out_b = _open("local/augmented", result_b.output_root)
        img_a = _episode_column(out_a, 0, "observation.images.cam")
        img_b = _episode_column(out_b, 0, "observation.images.cam")
        # Byte-identical as measured on this stack (see the module docstring
        # for the exact stack and for what to do if this ever breaks).
        assert img_a.tobytes() == img_b.tobytes()

    def test_provenance_seeds_match_across_runs(self, transformed_twice):
        _, (result_a, result_b) = transformed_twice
        seed_a = load_provenance(result_a.output_root)[0]["seed"]
        seed_b = load_provenance(result_b.output_root)[0]["seed"]
        assert seed_a == seed_b
