"""Regression tests: route ``processor_overrides`` to the pipeline that owns the step.

A checkpoint ships two processor pipelines whose step keys are DISJOINT: state
normalization lives in the preprocessor's ``normalizer_processor`` step and
action un-normalization in the postprocessor's ``unnormalizer_processor`` step.
LeRobot raises ``KeyError`` for an override key that matches no step in the
pipeline it is loading (``DataProcessorPipeline._validate_overrides_used``, a
typo guard).

Pre-fix, :meth:`ProcessorBridge.from_pretrained` handed the caller's whole
``overrides`` dict to BOTH pipelines, so every pipeline-specific step was
rejected by whichever pipeline lacked it -- ``normalizer_processor`` by the
postprocessor, ``unnormalizer_processor`` by the preprocessor, and naming both
by each in turn. Only ``device_processor``, the one step present in both
pipelines, ever got through. Supplying normalizer ``stats`` was therefore
impossible, even though that is the documented remedy for a pretraining base
checkpoint whose stats are dataset-prefixed (``so100.buffer.action``) and whose
declared normalization is consequently inert -- state reaches the model raw and
predicted actions reach the robot un-unnormalized.

These tests build real pipeline configs on disk with LeRobot's own
``save_pretrained`` and load them through the real LeRobot pipeline, so they
verify behavior against actual LeRobot internals rather than mocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("lerobot.processor.pipeline")
torch = pytest.importorskip("torch")

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.processor import (  # noqa: E402
    DataProcessorPipeline,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    UnnormalizerProcessorStep,
)

from strands_robots.policies.lerobot_local.processor import ProcessorBridge  # noqa: E402

_FEATURES = {
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
}
_NORM_MAP = {
    FeatureType.STATE: NormalizationMode.MEAN_STD,
    FeatureType.ACTION: NormalizationMode.MEAN_STD,
}


def _stats(*keys: str) -> dict[str, dict[str, Any]]:
    """Mean/std stats under the given lookup keys."""
    return {key: {"mean": torch.zeros(6), "std": torch.ones(6)} for key in keys}


def _canonical_stats() -> dict[str, dict[str, Any]]:
    """Stats a caller computes from their own dataset, under canonical keys."""
    return _stats("observation.state", "action")


def _write_checkpoint(directory: Path) -> None:
    """Write both pipelines of a base checkpoint whose declared norm is inert.

    Mirrors ``lerobot/smolvla_base``: a preprocessor carrying only
    ``normalizer_processor``, a postprocessor carrying only
    ``unnormalizer_processor``, and stats keyed by the training dataset rather
    than the canonical ``observation.state`` / ``action`` keys -- so both
    declared normalizations silently pass through.
    """
    dataset_prefixed = _stats("ds.buffer.action")
    DataProcessorPipeline(
        steps=[
            NormalizerProcessorStep(features=dict(_FEATURES), norm_map=dict(_NORM_MAP), stats=dict(dataset_prefixed))
        ],
        name="policy_preprocessor",
    ).save_pretrained(str(directory))
    DataProcessorPipeline(
        steps=[
            UnnormalizerProcessorStep(
                features={"action": _FEATURES["action"]},
                norm_map=dict(_NORM_MAP),
                stats=dict(dataset_prefixed),
            )
        ],
        name="policy_postprocessor",
    ).save_pretrained(str(directory))


def test_the_documented_remedy_makes_the_inert_pipeline_normalize_again(tmp_path: Path) -> None:
    """Supplying canonical stats for both steps leaves nothing inert.

    This is the remedy the inert-normalization diagnostic points callers at.
    Fails pre-fix with ``KeyError``: each key is rejected by the pipeline that
    does not declare it, so no stats could be supplied at all.
    """
    _write_checkpoint(tmp_path)
    stats = _canonical_stats()

    bridge = ProcessorBridge.from_pretrained(
        str(tmp_path),
        device="cpu",
        overrides={
            "normalizer_processor": {"stats": stats},
            "unnormalizer_processor": {"stats": stats},
        },
    )

    assert bridge.inert_normalization_features() == []


def test_a_preprocessor_only_step_override_is_applied(tmp_path: Path) -> None:
    """``normalizer_processor`` stats reach the preprocessor and normalize state.

    The action half stays inert, because only the preprocessor step was given
    stats -- the diagnostic keeps reporting exactly what remains unnormalized.
    """
    _write_checkpoint(tmp_path)

    bridge = ProcessorBridge.from_pretrained(
        str(tmp_path), device="cpu", overrides={"normalizer_processor": {"stats": _canonical_stats()}}
    )

    assert bridge.inert_normalization_features() == ["action (ACTION/MEAN_STD)"]


def test_a_postprocessor_only_step_override_is_applied(tmp_path: Path) -> None:
    """``unnormalizer_processor`` stats reach the postprocessor and unnormalize actions."""
    _write_checkpoint(tmp_path)

    bridge = ProcessorBridge.from_pretrained(
        str(tmp_path), device="cpu", overrides={"unnormalizer_processor": {"stats": _canonical_stats()}}
    )

    assert bridge.inert_normalization_features() == ["observation.state (STATE/MEAN_STD)"]


def test_both_pipelines_load_when_a_pipeline_specific_override_is_given(tmp_path: Path) -> None:
    """A pipeline-specific override does not cost the caller the other pipeline.

    Pre-fix the rejected key raised out of the load, so neither pipeline was
    reached; the bridge was never even constructed.
    """
    _write_checkpoint(tmp_path)

    bridge = ProcessorBridge.from_pretrained(
        str(tmp_path), device="cpu", overrides={"normalizer_processor": {"stats": _canonical_stats()}}
    )

    assert bridge.has_preprocessor is True
    assert bridge.has_postprocessor is True


def test_an_unknown_override_key_names_both_pipelines_steps(tmp_path: Path) -> None:
    """A key matching neither pipeline is refused, and the refusal names both.

    Pre-fix a typo was also refused (LeRobot's guard), but the message listed
    only the steps of whichever pipeline happened to be loaded first, so a
    caller reaching for a step of the OTHER pipeline was shown a key list that
    did not contain it.
    """
    _write_checkpoint(tmp_path)

    with pytest.raises(KeyError) as excinfo:
        ProcessorBridge.from_pretrained(
            str(tmp_path), device="cpu", overrides={"nomalizer_processor": {"stats": _canonical_stats()}}
        )

    message = str(excinfo.value)
    assert "nomalizer_processor" in message
    assert "normalizer_processor" in message
    assert "unnormalizer_processor" in message


def test_a_typo_is_still_refused(tmp_path: Path) -> None:
    """LeRobot's typo protection survives routing (passes pre-fix and post-fix)."""
    _write_checkpoint(tmp_path)

    with pytest.raises(KeyError):
        ProcessorBridge.from_pretrained(
            str(tmp_path), device="cpu", overrides={"not_a_step": {"stats": _canonical_stats()}}
        )


def test_a_step_declared_by_both_pipelines_still_reaches_both(tmp_path: Path) -> None:
    """``device_processor`` is in both pipelines and is still applied to both.

    The one override that worked pre-fix keeps working: routing must not narrow
    a key that both pipelines declare. Passes pre-fix and post-fix.
    """
    DataProcessorPipeline(
        steps=[
            NormalizerProcessorStep(features=dict(_FEATURES), norm_map=dict(_NORM_MAP), stats=_canonical_stats()),
            DeviceProcessorStep(device="cpu"),
        ],
        name="policy_preprocessor",
    ).save_pretrained(str(tmp_path))
    DataProcessorPipeline(
        steps=[
            UnnormalizerProcessorStep(
                features={"action": _FEATURES["action"]}, norm_map=dict(_NORM_MAP), stats=_canonical_stats()
            ),
            DeviceProcessorStep(device="cpu"),
        ],
        name="policy_postprocessor",
    ).save_pretrained(str(tmp_path))

    bridge = ProcessorBridge.from_pretrained(
        str(tmp_path), device="cpu", overrides={"device_processor": {"device": "cpu"}}
    )

    assert bridge.has_preprocessor is True
    assert bridge.has_postprocessor is True


def test_no_overrides_loads_both_pipelines_unchanged(tmp_path: Path) -> None:
    """The default path is untouched: no overrides, both pipelines, still inert."""
    _write_checkpoint(tmp_path)

    bridge = ProcessorBridge.from_pretrained(str(tmp_path), device="cpu")

    assert bridge.has_preprocessor is True
    assert bridge.has_postprocessor is True
    assert bridge.inert_normalization_features() == [
        "observation.state (STATE/MEAN_STD)",
        "action (ACTION/MEAN_STD)",
    ]
