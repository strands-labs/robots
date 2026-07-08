"""Load-time diagnostics when an ACTIVE processor pipeline cannot be configured.

``LerobotLocalPolicy._load_processor_bridge`` loads the processor pipeline and
wires the declarative embodiment into it. Two failure modes exist and must be
reported differently:

* ``ProcessorBridge.from_pretrained`` fails -> the checkpoint ships no pipeline;
  fall back to the raw obs/action flow (debug), and the generic
  missing-postprocessor warning still fires.
* ``_configure_embodiment`` raises ``ValueError`` -> the pipeline loaded and was
  ACTIVE, but the caller's embodiment / ``image_keys`` are incompatible with the
  model's declared features. The (working) normalization pipeline is discarded,
  which is a silent behaviour change. Previously this was swallowed at debug and
  the downstream warning then falsely blamed a missing ``policy_postprocessor.json``.
  It must now surface the real cause as a warning (or raise under
  ``processor_overrides``), and must NOT emit the misleading missing-postprocessor
  message.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy


def _feature(dim: int) -> MagicMock:
    feat = MagicMock()
    feat.shape = (dim,)
    return feat


def _make_policy(**kwargs) -> LerobotLocalPolicy:
    """Construct a policy without touching the heavy ``_load_model`` path."""
    with patch.object(LerobotLocalPolicy, "_load_model"):
        pol = LerobotLocalPolicy(pretrained_name_or_path="fake/ckpt", **kwargs)
    pol._device = None
    pol._input_features = {
        "observation.images.top": _feature(3),
        "observation.state": _feature(6),
    }
    pol._output_features = {"action": _feature(6)}
    return pol


def _fake_bridge(*, active: bool, has_postprocessor: bool) -> MagicMock:
    bridge = MagicMock(name="ProcessorBridge")
    bridge.is_active = active
    bridge.has_postprocessor = has_postprocessor
    bridge.inert_normalization_features.return_value = []
    return bridge


def _patch_from_pretrained(monkeypatch, bridge) -> None:
    monkeypatch.setattr(
        "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
        classmethod(lambda cls, *a, **k: bridge),
    )


# An embodiment whose action_keys length (3) disagrees with the model's action
# dim (6) -> EmbodimentMap.validate raises ValueError inside _configure_embodiment.
def _incompatible_embodiment() -> EmbodimentMap:
    return EmbodimentMap(
        name="wrong_dims",
        obs_rename={},
        state_keys=[],
        action_keys=["a", "b", "c"],
        dim_policy="pad",
    )


def test_embodiment_config_failure_warns_with_real_cause_and_discards(monkeypatch, caplog):
    """An active pipeline + incompatible embodiment -> accurate warning, bridge discarded."""
    bridge = _fake_bridge(active=True, has_postprocessor=True)
    _patch_from_pretrained(monkeypatch, bridge)
    pol = _make_policy(embodiment=_incompatible_embodiment())

    with caplog.at_level(logging.WARNING):
        pol._load_processor_bridge()

    # The working pipeline was discarded (falls back to raw flow) ...
    assert pol._processor_bridge is None
    assert pol._embodiment_config_failed is True
    # ... with a warning that names the real cause (embodiment misconfiguration) ...
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("embodiment could not be configured" in m for m in msgs), msgs
    # ... and NOT the misleading "no policy_postprocessor.json" message (the
    # checkpoint shipped one; it was discarded here, not absent).
    assert not any("policy_postprocessor.json" in m for m in msgs), msgs


def test_embodiment_config_failure_raises_when_overrides_requested(monkeypatch):
    """With processor_overrides the caller opted into fail-fast -> RuntimeError."""
    bridge = _fake_bridge(active=True, has_postprocessor=True)
    _patch_from_pretrained(monkeypatch, bridge)
    pol = _make_policy(
        embodiment=_incompatible_embodiment(),
        processor_overrides={"normalizer_processor": {"stats": {}}},
    )
    with pytest.raises(RuntimeError, match="Embodiment configuration failed"):
        pol._load_processor_bridge()


def test_active_bridge_without_postprocessor_still_warns(monkeypatch, caplog):
    """No embodiment failure -> the generic missing-postprocessor warning still fires."""
    bridge = _fake_bridge(active=True, has_postprocessor=False)
    _patch_from_pretrained(monkeypatch, bridge)
    # No embodiment spec and no robot_state_keys -> _configure_embodiment is a no-op.
    pol = _make_policy(embodiment=None)

    with caplog.at_level(logging.WARNING):
        pol._load_processor_bridge()

    assert pol._embodiment_config_failed is False
    assert pol._processor_bridge is bridge
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("policy_postprocessor.json" in m for m in msgs), msgs


def _patch_from_pretrained_raises(monkeypatch, exc: Exception) -> None:
    """Make ``ProcessorBridge.from_pretrained`` raise -- the checkpoint ships
    no loadable processor pipeline (missing configs / optional import)."""

    def _boom(cls, *a, **k):
        raise exc

    monkeypatch.setattr(
        "strands_robots.policies.lerobot_local.policy.ProcessorBridge.from_pretrained",
        classmethod(_boom),
    )


def test_from_pretrained_failure_falls_back_to_raw_flow(monkeypatch, caplog):
    """When ``ProcessorBridge.from_pretrained`` raises and no overrides were
    requested, the checkpoint legitimately has no pipeline: fall back to the
    raw obs/action flow (bridge is None, no embodiment-config failure) and
    still emit the missing-postprocessor warning so a raw-action checkpoint
    is not mistaken for a frozen policy."""
    _patch_from_pretrained_raises(monkeypatch, FileNotFoundError("no processor configs"))
    pol = _make_policy(embodiment=None)

    with caplog.at_level(logging.WARNING):
        pol._load_processor_bridge()

    assert pol._processor_bridge is None
    assert pol._embodiment_config_failed is False
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    # The generic missing-postprocessor diagnostic still fires for the raw flow.
    assert any("policy_postprocessor.json" in m for m in msgs), msgs


def test_from_pretrained_failure_raises_when_overrides_requested(monkeypatch):
    """A caller that passes ``processor_overrides`` has opted into the
    processor pipeline; if ``from_pretrained`` cannot load it, silently
    dropping the overrides would be a hidden behaviour change. The load must
    fail-fast with a RuntimeError naming the real cause instead."""
    _patch_from_pretrained_raises(monkeypatch, ValueError("incompatible processor config"))
    pol = _make_policy(
        embodiment=None,
        processor_overrides={"normalizer_processor": {"stats": {}}},
    )

    with pytest.raises(RuntimeError, match="Processor bridge failed to load"):
        pol._load_processor_bridge()
