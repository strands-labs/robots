"""A ``norm_tag`` the checkpoint's stats do not declare is reported, not absorbed.

A checkpoint that ships only ``norm_stats.json`` gets its normalization from
:mod:`strands_robots.policies.lerobot_local.norm_stats`, keyed by an embodiment
*tag*. ``norm_tag`` is a documented ``config_keys`` knob, so the tag is a
free-form caller string and a misspelling is the ordinary mistake.

That misspelling used to be absorbed. ``build_norm_stats_processors`` answered a
requested-but-undeclared tag with the same ``(None, None)`` it uses for "this
checkpoint ships no stats", so the bridge became a full passthrough while usable
stats sat in the payload: ``observation.state`` reached the policy un-normalized
and predicted actions reached the motors un-unnormalized - the exact failure the
norm-stats fallback exists to prevent (a mid-range SO-101 action collapses from
~150 degrees to the raw 0.5 the model emitted). The load report then blamed a
missing ``policy_postprocessor.json`` and told the caller to supply a
postprocessor the checkpoint was never going to ship, so the one-character
remedy appeared nowhere.

The reference implementation refuses the same input:
``_RobotStats.validate_tag`` in
``lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2`` raises
``ValueError`` naming the allowed tags rather than proceeding un-normalized.

These tests pin the two halves and the boundary:

* a requested tag the payload does not declare raises ``UnknownNormTagError``
  naming the tag, the declared tags and the consequence, and that reason reaches
  the load report instead of the missing-postprocessor message;
* a benign absence (no stats file at all) keeps the old ``(None, None)`` verdict,
  a ``None`` reason and the existing missing-postprocessor message, and the
  ambiguous multi-tag case is untouched.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from strands_robots.policies.lerobot_local import norm_stats as ns
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.policies.lerobot_local.processor import ProcessorBridge

_FIXTURE = Path(__file__).parent / "fixtures" / "molmoact2_norm_stats.json"


def _payload() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text())


def _real_tag(payload: dict[str, Any]) -> str:
    return next(iter(payload["metadata_by_tag"]))


def _checkpoint(tmp_path: Path, payload: dict[str, Any]) -> str:
    """A checkpoint dir carrying only ``norm_stats.json`` (the MolmoAct2 shape)."""
    (tmp_path / "norm_stats.json").write_text(json.dumps(payload))
    return str(tmp_path)


def _lerobot_pipeline_importable() -> bool:
    try:
        from lerobot.processor.pipeline import DataProcessorPipeline  # noqa: F401
    except ImportError:
        return False
    return True


_requires_pipeline = pytest.mark.skipif(
    not _lerobot_pipeline_importable(),
    reason="lerobot processor pipeline framework not installed",
)


class TestAnUndeclaredTagIsRefused:
    """The resolver refuses a tag the payload does not carry."""

    def test_requested_tag_absent_from_the_payload_raises(self):
        payload = _payload()
        with pytest.raises(ns.UnknownNormTagError) as exc:
            ns.build_norm_stats_processors(payload, norm_tag="so101")
        message = str(exc.value)
        # Names what was asked for, what exists, and what would otherwise happen.
        assert "so101" in message
        assert _real_tag(payload) in message
        assert "un-normalized" in message

    def test_the_refusal_is_a_value_error(self):
        """Callers narrowing on the loader's existing exception contract still catch it."""
        assert issubclass(ns.UnknownNormTagError, ValueError)

    def test_a_declared_tag_named_by_the_refusal_actually_works(self):
        """The remedy the message offers is not a second dead end."""
        payload = _payload()
        with pytest.raises(ns.UnknownNormTagError) as exc:
            ns.build_norm_stats_processors(payload, norm_tag="so101")
        offered = re.search(r"Declared tags: \[([^\]]*)\]", str(exc.value))
        assert offered is not None, str(exc.value)
        tags = [t.strip().strip("'\"") for t in offered.group(1).split(",") if t.strip()]
        assert tags, str(exc.value)
        # Applying the offered tag resolves; no exception, a real tag back.
        assert ns.select_norm_tag(payload, tags[0]) == tags[0]


@_requires_pipeline
class TestTheBridgeRecordsWhyItIsInert:
    """A caller error puts the reason on the bridge; a benign absence does not.

    The three control cases read the reason defensively (``getattr``), so "no
    reason is recorded" is an assertion about behaviour rather than about the
    attribute existing - they hold on a tree without it too.
    """

    def test_undeclared_tag_leaves_the_reason_on_the_bridge(self, tmp_path):
        ckpt = _checkpoint(tmp_path, _payload())
        bridge = ProcessorBridge.from_pretrained(ckpt, norm_tag="so101")
        assert bridge.is_active is False
        assert bridge.inert_reason is not None
        assert "so101" in bridge.inert_reason
        assert bridge.get_info()["inert_reason"] == bridge.inert_reason

    def test_a_checkpoint_with_no_stats_has_no_reason(self, tmp_path):
        """The benign absence keeps its quiet verdict - nothing was within reach."""
        bridge = ProcessorBridge.from_pretrained(str(tmp_path), norm_tag="so101")
        assert bridge.is_active is False
        assert getattr(bridge, "inert_reason", None) is None

    def test_a_declared_tag_still_builds_both_pipelines(self, tmp_path):
        """Control: the path that already worked is untouched."""
        payload = _payload()
        ckpt = _checkpoint(tmp_path, payload)
        bridge = ProcessorBridge.from_pretrained(ckpt, norm_tag=_real_tag(payload))
        assert bridge.has_preprocessor and bridge.has_postprocessor
        assert getattr(bridge, "inert_reason", None) is None

    def test_an_omitted_tag_still_auto_resolves(self, tmp_path):
        """Control: single-tag auto-detection is untouched."""
        bridge = ProcessorBridge.from_pretrained(_checkpoint(tmp_path, _payload()))
        assert bridge.has_preprocessor and bridge.has_postprocessor
        assert getattr(bridge, "inert_reason", None) is None

    def test_the_undeclared_tag_is_what_costs_the_unnormalization(self, tmp_path):
        """The measured consequence: the action stays in the model's raw space."""
        payload = _payload()
        ckpt = _checkpoint(tmp_path, payload)
        stats = payload["metadata_by_tag"][_real_tag(payload)]["action_stats"]
        emitted = np.full(len(stats["q01"]), 0.5, dtype=np.float32)

        good = ProcessorBridge.from_pretrained(ckpt, norm_tag=_real_tag(payload))
        applied = np.asarray(good.postprocess(emitted.copy()), dtype=np.float32)
        # Unnormalized into robot units, far from the raw 0.5 the model emitted.
        assert np.max(np.abs(applied)) > 10.0 * float(np.max(np.abs(emitted)))

        typo = ProcessorBridge.from_pretrained(ckpt, norm_tag="so101")
        assert typo.has_postprocessor is False  # nothing would unnormalize it


class TestTheLoadReportNamesTheAccurateCause:
    """The report blames the tag, not a postprocessor the checkpoint never had."""

    @staticmethod
    def _policy(**kwargs) -> LerobotLocalPolicy:
        with patch.object(LerobotLocalPolicy, "_load_model"):
            return LerobotLocalPolicy(pretrained_name_or_path="fake/ckpt", **kwargs)

    @staticmethod
    def _bridge(inert_reason: str | None) -> MagicMock:
        bridge = MagicMock(name="ProcessorBridge")
        bridge.is_active = False
        bridge.has_postprocessor = False
        bridge.inert_reason = inert_reason
        bridge.inert_normalization_features.return_value = []
        return bridge

    def _load(self, monkeypatch, bridge) -> None:
        monkeypatch.setattr(
            "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
            classmethod(lambda cls, *a, **k: bridge),
        )

    def test_an_inert_reason_replaces_the_missing_postprocessor_message(self, monkeypatch, caplog):
        reason = "norm_tag='so101' is not declared by this checkpoint's norm stats."
        self._load(monkeypatch, self._bridge(reason))
        pol = self._policy()

        with caplog.at_level(logging.WARNING):
            pol._load_processor_bridge()

        msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(reason in m for m in msgs), msgs
        # NOT the misleading message: the checkpoint was never going to ship one,
        # and supplying a postprocessor does not fix a misspelled tag.
        assert not any("policy_postprocessor.json" in m for m in msgs), msgs

    def test_a_reasonless_inert_bridge_keeps_the_existing_message(self, monkeypatch, caplog):
        """Control: the benign absence still reports exactly what it used to."""
        self._load(monkeypatch, self._bridge(None))
        pol = self._policy()

        with caplog.at_level(logging.WARNING):
            pol._load_processor_bridge()

        msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("policy_postprocessor.json" in m for m in msgs), msgs


class TestTheBoundaryIsUnchanged:
    """Verdicts that are not caller errors keep their existing behaviour."""

    def test_an_ambiguous_multi_tag_payload_still_degrades_quietly(self):
        """Nothing to blame the caller for: no tag was requested."""
        payload = {"format": ns.MOLMOACT2_NORM_STATS_FORMAT, "metadata_by_tag": {"x": {}, "y": {}}}
        assert ns.build_norm_stats_processors(payload) == (None, None)

    def test_the_public_tag_resolver_still_returns_none(self):
        """``select_norm_tag`` keeps its documented "resolve or None" contract."""
        assert ns.select_norm_tag({"metadata_by_tag": {"a": {}}}, "missing") is None

    def test_an_empty_tag_mapping_does_not_raise(self):
        """Nothing is declared, so there is no declared set to have missed."""
        payload = {"format": ns.MOLMOACT2_NORM_STATS_FORMAT, "metadata_by_tag": {}}
        assert ns.build_norm_stats_processors(payload, norm_tag="so101") == (None, None)
