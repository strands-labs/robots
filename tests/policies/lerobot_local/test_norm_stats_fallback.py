"""Regression tests for the ``norm_stats.json`` processor fallback.

Covers :mod:`strands_robots.policies.lerobot_local.norm_stats` and its wiring
into :class:`~strands_robots.policies.lerobot_local.processor.ProcessorBridge`.

The bug these guard against: a checkpoint that ships only ``norm_stats.json``
(no ``policy_preprocessor.json`` / ``policy_postprocessor.json``, e.g. the
MolmoAct2 SO-100/101 family) used to produce a passthrough bridge -- state
reached the policy un-normalized and predicted actions reached the motors
un-unnormalized, the single biggest cause of off-policy arm motion.

The numeric transform is validated two ways: against a local q01_q99 reference
formula, and -- as a per-formula audit -- bit-for-bit against the *installed*
upstream lerobot ``NormalizerProcessorStep`` / ``UnnormalizerProcessorStep`` in
``NormalizationMode.QUANTILES`` (see ``TestUpstreamLerobotParity``, max-abs-diff
< 1e-6). The latter guards against silent drift between this port and lerobot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from strands_robots.policies.lerobot_local import norm_stats as ns


def _lerobot_pipeline_importable() -> bool:
    """Return True if LeRobot's real processor pipeline can be imported.

    Used to gate only the tests that build real LeRobot ``ProcessorStep`` /
    ``DataProcessorPipeline`` objects. The pure-numpy ``FeatureNormalizer``
    tests below must NOT depend on this: the normalizer math is the single
    most safety-critical path (un-normalized actions reach the motors), so it
    has to stay covered even on installs without the optional processor
    framework. The import can fail with ImportError (framework absent) or, on
    some torch/coverage ABI combinations, AttributeError while resolving the
    lazy ``torch`` surface -- both mean "pipeline unavailable here".
    """
    try:
        import lerobot.processor.pipeline  # noqa: F401
    except (ImportError, AttributeError):
        return False
    return True


_requires_lerobot_pipeline = pytest.mark.skipif(
    not _lerobot_pipeline_importable(),
    reason="requires LeRobot's processor pipeline (real ProcessorStep framework)",
)

FIXTURE = Path(__file__).parent / "fixtures" / "molmoact2_norm_stats.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _ref_q01_q99_normalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Reference q01_q99 normalize (matches modeling_molmoact2._FeatureNormalizer)."""
    normed = 2.0 * (x - q01) / np.maximum(q99 - q01, 1e-6) - 1.0
    return np.clip(normed, -1.0, 1.0)


def _ref_q01_q99_unnormalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    arr = np.clip(x, -1.0, 1.0)
    return (arr + 1.0) * (q99 - q01) / 2.0 + q01


class TestFeatureNormalizer:
    """The ported per-feature normalizer."""

    def test_q01_q99_normalize_matches_reference(self):
        fix = _load_fixture()
        stats = fix["metadata_by_tag"]["so100_so101_molmoact2"]["state_stats"]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        fn = ns.FeatureNormalizer.from_stats(stats, "q01_q99")

        x = np.array([0.0, 100.0, 90.0, 40.0, -10.0, 20.0], dtype=np.float32)
        got = fn.normalize(x)
        want = _ref_q01_q99_normalize(x, q01, q99)
        assert np.allclose(got, want, atol=1e-4)

    def test_q01_q99_unnormalize_matches_reference(self):
        fix = _load_fixture()
        stats = fix["metadata_by_tag"]["so100_so101_molmoact2"]["action_stats"]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        fn = ns.FeatureNormalizer.from_stats(stats, "q01_q99")

        a = np.array([0.1, -0.5, 0.3, 0.9, -1.0, 0.0], dtype=np.float32)
        got = fn.unnormalize(a)
        want = _ref_q01_q99_unnormalize(a, q01, q99)
        assert np.allclose(got, want, atol=1e-4)

    def test_round_trip_within_quantile_range_is_identity(self):
        # Values strictly inside [q01, q99] survive normalize->unnormalize.
        fix = _load_fixture()
        stats = fix["metadata_by_tag"]["so100_so101_molmoact2"]["state_stats"]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        fn = ns.FeatureNormalizer.from_stats(stats, "q01_q99")

        x = (q01 + q99) / 2.0  # midpoint, well inside range
        round_tripped = fn.unnormalize(fn.normalize(x))
        assert np.allclose(round_tripped, x, atol=1e-3)

    def test_clip_saturates_out_of_range(self):
        stats = {"q01": [0.0, 0.0], "q99": [10.0, 10.0]}
        fn = ns.FeatureNormalizer.from_stats(stats, "q01_q99")
        # 100 >> q99 -> normalized clips to +1.0
        got = fn.normalize(np.array([100.0, -100.0], dtype=np.float32))
        assert np.allclose(got, [1.0, -1.0])

    def test_mean_std_mode(self):
        stats = {"mean": [1.0, 2.0], "std": [2.0, 4.0]}
        fn = ns.FeatureNormalizer.from_stats(stats, "mean_std")
        got = fn.normalize(np.array([3.0, 6.0], dtype=np.float32))
        assert np.allclose(got, [1.0, 1.0])
        assert np.allclose(fn.unnormalize(got), [3.0, 6.0])

    def test_min_max_mode(self):
        stats = {"min": [0.0, 0.0], "max": [10.0, 10.0]}
        fn = ns.FeatureNormalizer.from_stats(stats, "min_max")
        got = fn.normalize(np.array([5.0, 0.0], dtype=np.float32))
        assert np.allclose(got, [0.0, -1.0])

    def test_unsupported_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported robot normalization mode"):
            ns.FeatureNormalizer.from_stats({"q01": [0.0]}, "bogus_mode")

    def test_missing_required_stats_raises(self):
        with pytest.raises(ValueError, match="requires q01 and q99"):
            ns.FeatureNormalizer.from_stats({"min": [0.0]}, "q01_q99")

    def test_none_input_returns_none(self):
        fn = ns.FeatureNormalizer.from_stats({"q01": [0.0], "q99": [1.0]}, "q01_q99")
        assert fn.normalize(None) is None
        assert fn.unnormalize(None) is None

    def test_q10_q90_mode_round_trip_matches_analytic_form(self):
        # q10_q90 is a documented mode (see class docstring) that shares the
        # quantile-scaling math of q01_q99 but keys off q10/q90 stats. Verify
        # both the forward map and that a value inside [q10, q90] round-trips.
        stats = {"q10": [0.0, -5.0], "q90": [10.0, 5.0]}
        fn = ns.FeatureNormalizer.from_stats(stats, "q10_q90")
        q10 = np.asarray(stats["q10"], dtype=np.float32)
        q90 = np.asarray(stats["q90"], dtype=np.float32)

        x = np.array([2.5, 0.0], dtype=np.float32)  # strictly inside [q10, q90]
        want = 2.0 * (x - q10) / np.maximum(q90 - q10, 1e-6) - 1.0
        got = fn.normalize(x)
        assert np.allclose(got, want, atol=1e-6)
        assert np.allclose(fn.unnormalize(got), x, atol=1e-4)

    def test_normalizes_without_torch_installed(self, monkeypatch):
        # torch is an optional dep: the normalizer must still work numpy-in /
        # numpy-out when torch cannot be imported. Masking sys.modules["torch"]
        # with None makes ``import torch`` raise ImportError, exercising the
        # documented torch-absent fallback in _to_array and _restore_like.
        import sys

        monkeypatch.setitem(sys.modules, "torch", None)
        stats = {"min": [0.0, 0.0], "max": [10.0, 10.0]}
        fn = ns.FeatureNormalizer.from_stats(stats, "min_max")

        got = fn.normalize(np.array([5.0, 0.0], dtype=np.float32))
        assert isinstance(got, np.ndarray)
        assert np.allclose(got, [0.0, -1.0])
        assert np.allclose(fn.unnormalize(got), [5.0, 0.0])


class TestSchemaDetection:
    """Payload schema detection and tag selection."""

    def test_recognizes_molmoact2_schema(self):
        assert ns.is_norm_stats_payload(_load_fixture()) is True

    def test_rejects_unrelated_json(self):
        assert ns.is_norm_stats_payload({"foo": "bar"}) is False
        assert ns.is_norm_stats_payload({"format": "molmoact2_norm_stats.v1"}) is False
        assert ns.is_norm_stats_payload(None) is False

    def test_select_sole_tag(self):
        assert ns.select_norm_tag(_load_fixture()) == "so100_so101_molmoact2"

    def test_explicit_tag_wins(self):
        payload = {"metadata_by_tag": {"a": {}, "b": {}}}
        assert ns.select_norm_tag(payload, "b") == "b"

    def test_unknown_explicit_tag_returns_none(self):
        payload = {"metadata_by_tag": {"a": {}}}
        assert ns.select_norm_tag(payload, "missing") is None

    def test_default_tag_used_on_ambiguity(self):
        payload = {"metadata_by_tag": {ns.DEFAULT_SO_NORM_TAG: {}, "other": {}}}
        assert ns.select_norm_tag(payload) == ns.DEFAULT_SO_NORM_TAG

    def test_ambiguous_no_default_returns_none(self):
        payload = {"metadata_by_tag": {"x": {}, "y": {}}}
        assert ns.select_norm_tag(payload) is None


class TestLoadNormStats:
    """Loading norm_stats.json from a local checkpoint directory."""

    def test_loads_local_norm_stats(self, tmp_path):
        (tmp_path / "norm_stats.json").write_text(FIXTURE.read_text())
        payload = ns.load_norm_stats(str(tmp_path))
        assert ns.is_norm_stats_payload(payload)

    def test_honors_config_filename_override(self, tmp_path):
        (tmp_path / "custom_stats.json").write_text(FIXTURE.read_text())
        (tmp_path / "config.json").write_text(json.dumps({"norm_stats_filename": "custom_stats.json"}))
        payload = ns.load_norm_stats(str(tmp_path))
        assert ns.is_norm_stats_payload(payload)

    def test_missing_file_returns_none(self, tmp_path):
        assert ns.load_norm_stats(str(tmp_path)) is None

    def test_empty_path_returns_none(self):
        assert ns.load_norm_stats("") is None


@_requires_lerobot_pipeline
class TestBuildProcessors:
    """build_norm_stats_processors against the real LeRobot pipeline."""

    def test_builds_active_pre_and_post(self):
        pre, post = ns.build_norm_stats_processors(_load_fixture())
        assert pre is not None and post is not None
        assert len(pre) == 1 and len(post) == 1

    def test_preprocessor_normalizes_state_through_pipeline(self):
        from lerobot.processor import TransitionKey
        from lerobot.processor.converters import create_transition

        fix = _load_fixture()
        stats = fix["metadata_by_tag"]["so100_so101_molmoact2"]["state_stats"]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)

        pre, _ = ns.build_norm_stats_processors(fix)
        x = np.array([0.0, 100.0, 90.0, 40.0, -10.0, 20.0], dtype=np.float32)
        out = pre._forward(create_transition(observation={"observation.state": x.copy()}))
        normed = out[TransitionKey.OBSERVATION]["observation.state"]
        assert np.allclose(normed, _ref_q01_q99_normalize(x, q01, q99), atol=1e-4)

    def test_postprocessor_unnormalizes_action_through_pipeline(self):
        fix = _load_fixture()
        stats = fix["metadata_by_tag"]["so100_so101_molmoact2"]["action_stats"]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)

        _, post = ns.build_norm_stats_processors(fix)
        a = np.array([0.1, -0.5, 0.3, 0.9, -1.0, 0.0], dtype=np.float32)
        out = post.process_action(a)
        assert np.allclose(out, _ref_q01_q99_unnormalize(a, q01, q99), atol=1e-4)

    def test_unresolved_tag_returns_none_pair(self):
        payload = {"format": ns.MOLMOACT2_NORM_STATS_FORMAT, "metadata_by_tag": {"x": {}, "y": {}}}
        assert ns.build_norm_stats_processors(payload) == (None, None)


@_requires_lerobot_pipeline
class TestProcessorBridgeFallback:
    """ProcessorBridge.from_pretrained wires the norm_stats fallback.

    Regression: pre-fix, a checkpoint dir with ONLY norm_stats.json yielded an
    inactive (passthrough) bridge. Post-fix it builds working normalizers.
    """

    def test_fallback_activates_bridge(self, tmp_path):
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        (tmp_path / "norm_stats.json").write_text(FIXTURE.read_text())
        bridge = ProcessorBridge.from_pretrained(str(tmp_path), device="cpu")
        assert bridge.is_active
        assert bridge.has_preprocessor and bridge.has_postprocessor

    def test_fallback_normalizes_and_unnormalizes(self, tmp_path):
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        fix = _load_fixture()
        (tmp_path / "norm_stats.json").write_text(json.dumps(fix))
        bridge = ProcessorBridge.from_pretrained(str(tmp_path), device="cpu")

        sstats = fix["metadata_by_tag"]["so100_so101_molmoact2"]["state_stats"]
        q01 = np.asarray(sstats["q01"], dtype=np.float32)
        q99 = np.asarray(sstats["q99"], dtype=np.float32)
        x = np.array([0.0, 100.0, 90.0, 40.0, -10.0, 20.0], dtype=np.float32)
        out = bridge.preprocess({"observation.state": x.copy()})
        assert np.allclose(out["observation.state"], _ref_q01_q99_normalize(x, q01, q99), atol=1e-4)

    def test_empty_checkpoint_stays_passthrough(self, tmp_path):
        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        bridge = ProcessorBridge.from_pretrained(str(tmp_path), device="cpu")
        assert not bridge.is_active

    def test_migration_error_is_treated_as_missing_config(self):
        # LeRobot 0.5.2 raises ProcessorMigrationError for missing configs; the
        # bridge must classify it as "no standard config" so the fallback runs.
        from strands_robots.policies.lerobot_local.processor import _missing_config_errors

        pipeline = pytest.importorskip("lerobot.processor.pipeline")
        migration_error = getattr(pipeline, "ProcessorMigrationError", None)
        if migration_error is None:
            pytest.skip("installed lerobot has no ProcessorMigrationError")
        assert migration_error in _missing_config_errors()

    def test_missing_migration_error_falls_back_to_builtin_errors(self, monkeypatch):
        # On an older lerobot that predates ProcessorMigrationError the set must
        # degrade to the built-in (FileNotFoundError, ValueError) rather than
        # raise ImportError while classifying a missing processor config.
        from lerobot.processor import pipeline as lr_pipeline

        from strands_robots.policies.lerobot_local.processor import _missing_config_errors

        monkeypatch.delattr(lr_pipeline, "ProcessorMigrationError", raising=False)
        assert _missing_config_errors() == (FileNotFoundError, ValueError)


@_requires_lerobot_pipeline
class TestUpstreamLerobotParity:
    """Bit-equality audit vs the real upstream lerobot normalizer.

    The other ``TestFeatureNormalizer`` cases compare against a local reference
    reimplementation of the formula, which cannot catch drift between this port
    and lerobot itself. These tests run the SAME data through the installed
    ``lerobot.processor.NormalizerProcessorStep`` /
    ``UnnormalizerProcessorStep`` with ``NormalizationMode.QUANTILES`` (q01/q99)
    and assert max-abs-diff < 1e-6 against ``FeatureNormalizer``.

    Upstream's bare QUANTILES step does NOT clamp to [-1, 1] (the MolmoAct2
    pipeline clamps in a separate ``MolmoAct2ClampNormalizedProcessorStep``),
    so for the forward direction we feed values strictly inside [q01, q99]
    where ``FeatureNormalizer``'s built-in clip is a no-op and the cores match.
    """

    @staticmethod
    def _require_upstream():
        pytest.importorskip("lerobot")
        processor = pytest.importorskip("lerobot.processor")
        configs = pytest.importorskip("lerobot.configs")
        types = pytest.importorskip("lerobot.types")
        for name in ("NormalizerProcessorStep", "UnnormalizerProcessorStep"):
            if not hasattr(processor, name):
                pytest.skip(f"installed lerobot lacks {name}")
        for name in ("FeatureType", "NormalizationMode", "PolicyFeature"):
            if not hasattr(configs, name):
                pytest.skip("installed lerobot.configs lacks typed-feature API (needs >= 0.5.2)")
        if not hasattr(configs.NormalizationMode, "QUANTILES"):
            pytest.skip("installed lerobot has no NormalizationMode.QUANTILES")
        if not hasattr(types, "TransitionKey"):
            pytest.skip("installed lerobot.types lacks TransitionKey")
        return processor, configs, types

    def _stats(self):
        fix = _load_fixture()
        stats = fix["metadata_by_tag"]["so100_so101_molmoact2"]["state_stats"]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        return stats, q01, q99

    def test_normalize_bit_equal_to_upstream_quantiles(self):
        processor, configs, types = self._require_upstream()
        import torch

        stats, q01, q99 = self._stats()
        key = "observation.state"
        fn = ns.FeatureNormalizer.from_stats(stats, "q01_q99")

        up = processor.NormalizerProcessorStep(
            features={key: configs.PolicyFeature(type=configs.FeatureType.STATE, shape=q01.shape)},
            norm_map={configs.FeatureType.STATE: configs.NormalizationMode.QUANTILES},
            stats={key: {"q01": torch.from_numpy(q01), "q99": torch.from_numpy(q99)}},
            eps=ns._EPS,
        )

        rng = np.random.default_rng(58)
        # Strictly inside [q01, q99] so FeatureNormalizer's [-1, 1] clip is inert
        # and we exercise the raw affine core both implementations share.
        x = rng.uniform(q01, q99, (8, q01.shape[0])).astype(np.float32)
        got = fn.normalize(x)
        want = up({types.TransitionKey.OBSERVATION: {key: torch.from_numpy(x)}})[types.TransitionKey.OBSERVATION][
            key
        ].numpy()
        assert np.max(np.abs(got - want)) < 1e-6

    def test_unnormalize_bit_equal_to_upstream_quantiles(self):
        processor, configs, types = self._require_upstream()
        import torch

        stats, q01, q99 = self._stats()
        key = "action"
        fn = ns.FeatureNormalizer.from_stats(stats, "q01_q99")

        up = processor.UnnormalizerProcessorStep(
            features={key: configs.PolicyFeature(type=configs.FeatureType.ACTION, shape=q01.shape)},
            norm_map={configs.FeatureType.ACTION: configs.NormalizationMode.QUANTILES},
            stats={key: {"q01": torch.from_numpy(q01), "q99": torch.from_numpy(q99)}},
            eps=ns._EPS,
        )

        rng = np.random.default_rng(99)
        # Normalized actions live in [-1, 1]; both sides clamp there identically.
        a = rng.uniform(-1.0, 1.0, (8, q01.shape[0])).astype(np.float32)
        got = fn.unnormalize(a)
        want = up({types.TransitionKey.ACTION: torch.from_numpy(a)})[types.TransitionKey.ACTION].numpy()
        assert np.max(np.abs(got - want)) < 1e-6


class TestContainerTypePreservation:
    """``FeatureNormalizer`` preserves the input container (numpy vs torch).

    The normalizer is wired into LeRobot processor steps that hand it whatever
    the policy/runtime uses -- numpy arrays in sim, torch tensors on the GPU
    path. Returning the wrong container (e.g. a numpy array where a tensor with a
    specific device/dtype was expected) breaks the downstream pipeline silently.
    """

    def test_torch_tensor_in_tensor_out_same_dtype_device(self):
        torch = pytest.importorskip("torch")
        fn = ns.FeatureNormalizer.from_stats({"q01": [0.0, 0.0], "q99": [10.0, 10.0]}, "q01_q99")

        t = torch.tensor([5.0, 5.0], dtype=torch.bfloat16)
        out = fn.normalize(t)
        assert torch.is_tensor(out)
        assert out.dtype == torch.bfloat16
        assert out.device == t.device
        # 5.0 maps to the midpoint of [-1, 1] -> 0.0.
        assert torch.allclose(out.float(), torch.zeros(2), atol=1e-2)

    def test_torch_round_trip_returns_tensor(self):
        torch = pytest.importorskip("torch")
        fn = ns.FeatureNormalizer.from_stats({"min": [0.0, 0.0], "max": [10.0, 10.0]}, "min_max")
        a = torch.tensor([0.2, -0.4], dtype=torch.float32)
        out = fn.unnormalize(a)
        assert torch.is_tensor(out)
        assert out.dtype == torch.float32


class TestNoneMode:
    """``norm_mode='none'`` is an explicit identity transform (no passthrough bug)."""

    def test_none_mode_is_identity(self):
        fn = ns.FeatureNormalizer.from_stats({"mean": [1.0, 2.0]}, "none")
        x = np.array([9.0, -3.0], dtype=np.float32)
        assert np.allclose(fn.normalize(x), x)
        assert np.allclose(fn.unnormalize(x), x)

    def test_none_mode_from_empty_stats_still_builds(self):
        # No usable stat keys: fallback stays None, mask is None, identity holds.
        fn = ns.FeatureNormalizer.from_stats({}, "none")
        assert fn is not None
        assert fn.mask is None
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert np.allclose(fn.normalize(x), x)

    def test_from_stats_none_payload_returns_none(self):
        assert ns.FeatureNormalizer.from_stats(None, "q01_q99") is None


class TestMaskAndZeroMask:
    """Selective per-feature mask and degenerate (min==max) handling."""

    def test_mask_keeps_unmasked_features_unchanged(self):
        # mask=[True, False]: feature 0 normalized, feature 1 passes through raw.
        fn = ns.FeatureNormalizer.from_stats({"min": [0.0, 0.0], "max": [10.0, 10.0], "mask": [True, False]}, "min_max")
        out = fn.normalize(np.array([5.0, 5.0], dtype=np.float32))
        # 5.0 over [0,10] -> 0.0 where masked-in; raw 5.0 where masked-out.
        assert np.allclose(out, [0.0, 5.0])

    def test_mask_applies_to_unnormalize_too(self):
        fn = ns.FeatureNormalizer.from_stats({"min": [0.0, 0.0], "max": [10.0, 10.0], "mask": [True, False]}, "min_max")
        out = fn.unnormalize(np.array([0.0, 0.0], dtype=np.float32))
        # 0.0 normalized -> 5.0 where masked-in; raw 0.0 where masked-out.
        assert np.allclose(out, [5.0, 0.0])

    def test_zero_mask_when_min_equals_max(self):
        # Degenerate feature (min==max) must map to 0.0, never divide-by-zero.
        fn = ns.FeatureNormalizer.from_stats({"min": [0.0, 5.0], "max": [10.0, 5.0]}, "min_max")
        out = fn.normalize(np.array([5.0, 5.0], dtype=np.float32))
        assert np.allclose(out, [0.0, 0.0])

    def test_scalar_mask_broadcasts_to_feature_shape(self):
        # A scalar mask must broadcast to the per-feature stat shape.
        fn = ns.FeatureNormalizer.from_stats(
            {"q01": [0.0, 0.0, 0.0], "q99": [10.0, 10.0, 10.0], "mask": True}, "q01_q99"
        )
        assert fn.mask is not None
        assert fn.mask.shape == (3,)
        assert bool(fn.mask.all())


class TestMinMaxUnnormalize:
    """min_max unnormalize is the exact inverse of normalize in-range."""

    def test_min_max_round_trip(self):
        fn = ns.FeatureNormalizer.from_stats({"min": [-2.0, 0.0], "max": [2.0, 10.0]}, "min_max")
        x = np.array([1.0, 7.5], dtype=np.float32)
        assert np.allclose(fn.unnormalize(fn.normalize(x)), x, atol=1e-5)


class TestBuildProcessorsGuards:
    """build_norm_stats_processors refuses malformed metadata (no silent build)."""

    def test_non_dict_metadata_returns_none_pair(self):
        payload = {"format": ns.MOLMOACT2_NORM_STATS_FORMAT, "metadata_by_tag": {"t": "not-a-dict"}}
        assert ns.build_norm_stats_processors(payload, "t") == (None, None)

    def test_missing_state_or_action_stats_returns_none_pair(self):
        payload = {
            "format": ns.MOLMOACT2_NORM_STATS_FORMAT,
            "metadata_by_tag": {"t": {"state_stats": {"q01": [0.0], "q99": [1.0]}, "action_stats": "bad"}},
        }
        assert ns.build_norm_stats_processors(payload, "t") == (None, None)


class TestOptionalDepDegradation:
    """``build_norm_stats_processors`` degrades to ``(None, None)`` -- never
    raises -- when LeRobot's processor framework is unavailable.

    The norm-stats fallback runs during policy load. LeRobot's processor
    pipeline is an optional surface: an install that ships the pure-numpy
    ``FeatureNormalizer`` math but not the ``ProcessorStep`` /
    ``DataProcessorPipeline`` classes (older/partial lerobot) must still load
    the policy and simply skip the pipeline wiring, rather than crashing the
    whole ``ProcessorBridge.from_pretrained`` path. These lock that contract
    at each import seam the builder crosses.
    """

    def test_make_step_classes_none_when_processor_framework_absent(self, monkeypatch):
        # ``_make_step_classes`` imports its ProcessorStep bases lazily. When
        # that import fails (framework absent) it returns None so the caller can
        # degrade -- masking the submodule in sys.modules forces the ImportError.
        import sys

        monkeypatch.setitem(sys.modules, "lerobot.processor.pipeline", None)
        assert ns._make_step_classes() is None

    def test_build_degrades_when_step_classes_unavailable(self, monkeypatch):
        # Valid, resolvable payload, but the ProcessorStep framework cannot be
        # built: the builder must return the (None, None) pair, not raise.
        monkeypatch.setattr(ns, "_make_step_classes", lambda: None)
        assert ns.build_norm_stats_processors(_load_fixture()) == (None, None)

    @_requires_lerobot_pipeline
    def test_build_degrades_when_pipeline_class_unimportable(self, monkeypatch):
        # The ProcessorStep bases resolve, but ``DataProcessorPipeline`` itself
        # is missing from the installed lerobot: the deferred import inside the
        # builder raises ImportError, which must fall through to (None, None).
        import lerobot.processor.pipeline as pipeline_mod

        monkeypatch.delattr(pipeline_mod, "DataProcessorPipeline", raising=False)
        assert ns.build_norm_stats_processors(_load_fixture()) == (None, None)

    def test_build_degrades_when_normalizer_fails_to_build(self, monkeypatch):
        # Defensive contract at the normalizer seam: the stats dicts are present
        # and well-formed, but ``FeatureNormalizer.from_stats`` yields None (an
        # unusable normalizer). The builder must return the (None, None) pair
        # rather than wire a None normalizer into a ProcessorStep -- guarding the
        # degradation path if from_stats ever gains a dict-input None return.
        monkeypatch.setattr(ns.FeatureNormalizer, "from_stats", staticmethod(lambda *a, **k: None))
        assert ns.build_norm_stats_processors(_load_fixture()) == (None, None)


class TestHubLoader:
    """load_norm_stats Hub path: config.json filename override + fetch."""

    def test_hub_load_honors_config_filename_override(self, tmp_path, monkeypatch):
        # A non-local repo id forces the Hub branch. hf_hub_download is patched to
        # return local fixture files: config.json points at a custom stats name,
        # and that custom file carries the real payload.
        (tmp_path / "config.json").write_text(json.dumps({"norm_stats_filename": "custom.json"}))
        (tmp_path / "custom.json").write_text(FIXTURE.read_text())

        def fake_download(repo_id, filename, *args, **kwargs):
            return str(tmp_path / filename)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        payload = ns.load_norm_stats("acme/molmoact2-so101")
        assert ns.is_norm_stats_payload(payload)

    def test_hub_load_without_config_uses_default_filename(self, tmp_path, monkeypatch):
        # No config.json on the Hub: the config fetch raises, the default
        # norm_stats.json is fetched instead (the except-and-continue path).
        (tmp_path / "norm_stats.json").write_text(FIXTURE.read_text())

        def fake_download(repo_id, filename, *args, **kwargs):
            if filename == "config.json":
                raise FileNotFoundError("no config on hub")
            return str(tmp_path / filename)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        payload = ns.load_norm_stats("acme/molmoact2-so101")
        assert ns.is_norm_stats_payload(payload)

    def test_hub_fetch_failure_returns_none(self, monkeypatch):
        # Network/repo errors fetching the stats file are non-fatal: None, no raise.
        def fake_download(repo_id, filename, *args, **kwargs):
            raise OSError("connection reset")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        assert ns.load_norm_stats("acme/does-not-exist") is None


class TestFromStatsModeGuards:
    """from_stats raises (never silently passes through) on missing mode stats."""

    def test_mean_std_missing_std_raises(self):
        with pytest.raises(ValueError, match="requires mean and std"):
            ns.FeatureNormalizer.from_stats({"mean": [0.0]}, "mean_std")

    def test_min_max_missing_max_raises(self):
        with pytest.raises(ValueError, match="requires min and max"):
            ns.FeatureNormalizer.from_stats({"min": [0.0]}, "min_max")


class TestSelectTagMalformedMetadata:
    """select_norm_tag rejects payloads with no usable metadata_by_tag mapping."""

    def test_empty_tag_mapping_returns_none(self):
        assert ns.select_norm_tag({"metadata_by_tag": {}}) is None

    def test_non_dict_tag_mapping_returns_none(self):
        assert ns.select_norm_tag({"metadata_by_tag": "not-a-dict"}) is None

    def test_missing_tag_key_returns_none(self):
        assert ns.select_norm_tag({}) is None


@_requires_lerobot_pipeline
class TestProcessorStepEdgeCases:
    """Pre/post ProcessorStep helpers: identity transform_features + None action."""

    def test_transform_features_is_identity_on_both_steps(self):
        pre, post = ns.build_norm_stats_processors(_load_fixture())
        sentinel = object()
        assert pre.steps[0].transform_features(sentinel) is sentinel
        assert post.steps[0].transform_features(sentinel) is sentinel

    def test_postprocessor_passes_through_none_action(self):
        _, post = ns.build_norm_stats_processors(_load_fixture())
        assert post.steps[0].action(None) is None

    def test_preprocessor_leaves_observation_without_state_unchanged(self):
        pre, _ = ns.build_norm_stats_processors(_load_fixture())
        obs = {"observation.images.top": "frame"}
        assert pre.steps[0].observation(obs) == obs


class TestUnknownModeRaises:
    """An unrecognized mode must raise, never silently pass values through.

    Silent passthrough is the exact failure this module exists to prevent:
    un-normalized state reaching the policy and un-unnormalized actions reaching
    the motors. ``from_stats`` already rejects unknown modes; these guard the
    public transform hot-path (a directly-constructed or future-mode normalizer)
    so it fails loudly instead of degrading to identity.
    """

    def test_normalize_raises_on_unknown_mode(self):
        fn = ns.FeatureNormalizer(mode="totally-bogus")
        with pytest.raises(ValueError, match="unsupported mode"):
            fn.normalize(np.array([1.0, 2.0], dtype=np.float32))

    def test_unnormalize_raises_on_unknown_mode(self):
        fn = ns.FeatureNormalizer(mode="totally-bogus")
        with pytest.raises(ValueError, match="unsupported mode"):
            fn.unnormalize(np.array([0.0, 0.5], dtype=np.float32))

    def test_none_and_recognized_modes_still_transform(self):
        # Regression fence: the raise must not leak into the five valid modes.
        none_fn = ns.FeatureNormalizer.from_stats({"mean": [1.0]}, "none")
        x = np.array([7.0], dtype=np.float32)
        assert np.allclose(none_fn.normalize(x), x)
        assert np.allclose(none_fn.unnormalize(x), x)
        mm = ns.FeatureNormalizer.from_stats({"min": [0.0], "max": [10.0]}, "min_max")
        assert np.allclose(mm.unnormalize(mm.normalize(np.array([5.0], dtype=np.float32))), [5.0], atol=1e-5)
