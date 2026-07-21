"""Relative-action RTC prefix re-anchoring (LeRobot parity).

A relative-action flow policy (pi0 / pi0.5 / pi0-FAST with an enabled
``RelativeActionsProcessorStep``) trains on actions expressed as offsets from
the current robot state. The unexecuted tail of the previous chunk
(``prev_chunk_left_over``) is therefore only valid in the coordinate frame of
the observation that produced it. When the robot state moves between chunks,
the leftover must be re-expressed against the NEW state before it is fed back
to the policy, otherwise the model blends a stale-frame prefix into the next
chunk and the seam is corrupted.

These tests pin that the LeRobot RTC consumer in ``LerobotLocalPolicy``
re-anchors the leftover via LeRobot's ``reanchor_relative_rtc_prefix`` for
relative-action policies, and carries it verbatim for absolute-action policies
(whose frame does not move). Skips cleanly when LeRobot is unavailable.
"""

import importlib
import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("lerobot.processor")

from lerobot.processor import (  # noqa: E402
    AbsoluteActionsProcessorStep,
    IdentityProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.processor.converters import create_transition  # noqa: E402
from lerobot.processor.pipeline import DataProcessorPipeline  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402

from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy  # noqa: E402
from strands_robots.policies.lerobot_local.processor import ProcessorBridge  # noqa: E402


def _lerobot_has_reanchor_helper() -> bool:
    """True when the installed lerobot ships ``reanchor_relative_rtc_prefix``.

    The helper (and its ``lerobot.policies.rtc`` module) landed after lerobot
    0.5.1. Detection imports the module by string and probes for the symbol
    rather than ``from ... import`` (which CodeQL flags as an unused import and
    which only narrows to ImportError). It tolerates TypeError because importing
    ``lerobot.policies.rtc`` on lerobot 0.5.1 executes a module whose dataclass
    fails to build, raising at load time - exactly the failure the policy's own
    fallback guard must absorb.
    """
    try:
        module = importlib.import_module("lerobot.policies.rtc")
    except (ImportError, TypeError):
        return False
    return hasattr(module, "reanchor_relative_rtc_prefix")


_HAS_RTC_REANCHOR = _lerobot_has_reanchor_helper()

# The relative-action RTC re-anchoring path only engages when the installed
# lerobot ships reanchor_relative_rtc_prefix (added after 0.5.1). On an older
# lerobot the feature deliberately falls back to carrying the leftover verbatim
# (see test_relative_action_falls_back_when_reanchor_helper_unavailable), so
# the re-anchoring assertions below are meaningful only when the helper exists.
_requires_reanchor = pytest.mark.skipif(
    not _HAS_RTC_REANCHOR,
    reason="lerobot.policies.rtc.reanchor_relative_rtc_prefix unavailable (added after lerobot 0.5.1)",
)

_ACTION_DIM = 4
_CHUNK_LEN = 6
_EXEC_HORIZON = 2


def _model_chunk() -> torch.Tensor:
    """A deterministic (1, T, A) action chunk in model (normalized-relative) space."""
    return torch.arange(_CHUNK_LEN * _ACTION_DIM, dtype=torch.float32).reshape(1, _CHUNK_LEN, _ACTION_DIM)


def _model_leftover() -> torch.Tensor:
    """The tail of the chunk consumers do NOT execute this step: chunk[exec_horizon:]."""
    return _model_chunk().squeeze(0)[_EXEC_HORIZON:]


def _make_rtc_policy(preprocessor: DataProcessorPipeline, postprocessor: DataProcessorPipeline):
    """Build an RTC-enabled LerobotLocalPolicy wired to a real processor pipeline.

    The inner LeRobot policy is a mock whose ``predict_action_chunk`` records the
    ``prev_chunk_left_over`` it receives so the test can assert on the prefix that
    was actually fed to the model.
    """
    with patch.object(LerobotLocalPolicy, "_load_model"):
        policy = LerobotLocalPolicy(pretrained_name_or_path="test/model")

    policy._loaded = True
    policy._device = torch.device("cpu")

    inner = MagicMock()
    inner.config = MagicMock()
    inner.config.action_feature_names = [f"j{i}.pos" for i in range(_ACTION_DIM)]

    captured: list[torch.Tensor | None] = []

    def _predict(_batch, **kwargs):
        captured.append(kwargs.get("prev_chunk_left_over"))
        return _model_chunk()

    inner.predict_action_chunk.side_effect = _predict
    policy._policy = inner

    policy._rtc_enabled = True
    policy._rtc_execution_horizon = _EXEC_HORIZON
    # Deterministic zero inference delay: world is paused, exactly 0 steps elapse.
    policy.rtc_observed_delay_steps = 0

    policy._processor_bridge = ProcessorBridge(preprocessor=preprocessor, postprocessor=postprocessor)
    return policy, captured


def _prime_state(relative_step: RelativeActionsProcessorStep, state: torch.Tensor) -> None:
    """Cache ``state`` as the relative step's reference (mimics a preprocess pass)."""
    relative_step(create_transition(observation={OBS_STATE: state}))


@_requires_reanchor
def test_relative_action_rtc_prefix_is_reanchored_to_current_state():
    names = [f"j{i}.pos" for i in range(_ACTION_DIM)]
    relative_step = RelativeActionsProcessorStep(enabled=True, action_names=names)
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)
    policy, captured = _make_rtc_policy(
        preprocessor=DataProcessorPipeline(steps=[relative_step]),
        postprocessor=DataProcessorPipeline(steps=[absolute_step]),
    )

    state1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    _prime_state(relative_step, state1)
    with torch.inference_mode():
        policy._predict_with_rtc({})

    # First call has no prior chunk to blend.
    assert captured[0] is None
    leftover_model = _model_leftover()
    assert torch.allclose(policy._rtc_prev_chunk, leftover_model)
    # Absolute leftover = unnormalize (identity here) + add the current state.
    assert torch.allclose(policy._rtc_prev_chunk_abs, leftover_model + state1)

    # State moves: the leftover must be re-expressed against the new frame.
    state2 = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    _prime_state(relative_step, state2)
    with torch.inference_mode():
        policy._predict_with_rtc({})

    prev = captured[1]
    assert prev is not None
    # Re-anchored relative prefix = absolute - new_state = leftover + state1 - state2.
    expected = leftover_model + state1 - state2
    assert torch.allclose(prev, expected, atol=1e-5)
    # Crucially NOT the stale model-space leftover that pre-fix code carried.
    assert not torch.allclose(prev, leftover_model)


def test_absolute_action_policy_carries_leftover_verbatim():
    # No RelativeActionsProcessorStep -> the prefix frame never moves, so the
    # leftover is fed back unchanged and no absolute copy is kept.
    policy, captured = _make_rtc_policy(
        preprocessor=DataProcessorPipeline(steps=[IdentityProcessorStep()]),
        postprocessor=DataProcessorPipeline(steps=[IdentityProcessorStep()]),
    )

    with torch.inference_mode():
        policy._predict_with_rtc({})
    assert captured[0] is None
    leftover_model = _model_leftover()
    assert torch.allclose(policy._rtc_prev_chunk, leftover_model)
    assert policy._rtc_prev_chunk_abs is None

    with torch.inference_mode():
        policy._predict_with_rtc({})
    prev = captured[1]
    assert prev is not None
    assert torch.allclose(prev, leftover_model)


@_requires_reanchor
def test_resolve_rtc_rebase_steps_is_idempotent_and_detects_relative():
    names = [f"j{i}.pos" for i in range(_ACTION_DIM)]
    relative_step = RelativeActionsProcessorStep(enabled=True, action_names=names)
    policy, _ = _make_rtc_policy(
        preprocessor=DataProcessorPipeline(steps=[relative_step]),
        postprocessor=DataProcessorPipeline(steps=[IdentityProcessorStep()]),
    )

    policy._resolve_rtc_rebase_steps()
    assert policy._rtc_rebase_resolved is True
    assert policy._rtc_relative_step is relative_step
    assert policy._rtc_reanchor_fn is not None

    # A disabled relative step must NOT trigger re-anchoring.
    disabled = RelativeActionsProcessorStep(enabled=False, action_names=names)
    policy2, _ = _make_rtc_policy(
        preprocessor=DataProcessorPipeline(steps=[disabled]),
        postprocessor=DataProcessorPipeline(steps=[IdentityProcessorStep()]),
    )
    policy2._resolve_rtc_rebase_steps()
    assert policy2._rtc_relative_step is None


def _stub_rtc_missing_helper() -> types.ModuleType:
    """lerobot.policies.rtc present but without the helper symbol.

    ``from lerobot.policies.rtc import reanchor_relative_rtc_prefix`` then raises
    ImportError - the <= 0.5.1 case where the module ships no such symbol.
    """
    return types.ModuleType("lerobot.policies.rtc")


def _stub_rtc_broken_import() -> types.ModuleType:
    """lerobot.policies.rtc whose symbol access raises TypeError.

    Mirrors lerobot 0.5.1, where importing lerobot.policies.rtc executes a
    module that builds a broken dataclass, so the import raises TypeError at
    load time instead of cleanly missing the symbol.
    """

    class _RaisingModule(types.ModuleType):
        def __getattr__(self, name: str):
            if name == "reanchor_relative_rtc_prefix":
                raise TypeError("non-default argument 'backbone_cfg' follows default argument")
            raise AttributeError(name)

    return _RaisingModule("lerobot.policies.rtc")


@pytest.mark.parametrize(
    "make_stub",
    [_stub_rtc_missing_helper, _stub_rtc_broken_import],
    ids=["missing_helper", "broken_import"],
)
def test_relative_action_falls_back_when_reanchor_helper_unavailable(make_stub, monkeypatch, caplog):
    # A lerobot without a usable reanchor_relative_rtc_prefix cannot re-express
    # the leftover against the moved state. The relative-action policy must then
    # DISABLE re-anchoring, keep no absolute copy, warn once, and carry the
    # model-space leftover verbatim - never crash or silently drop the prefix.
    # Both unavailability modes are covered: the symbol is simply absent
    # (ImportError, <= 0.5.1) or importing lerobot.policies.rtc raises at load
    # time (TypeError, lerobot 0.5.1's broken rtc import chain).
    monkeypatch.setitem(sys.modules, "lerobot.policies.rtc", make_stub())

    names = [f"j{i}.pos" for i in range(_ACTION_DIM)]
    relative_step = RelativeActionsProcessorStep(enabled=True, action_names=names)
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)
    policy, captured = _make_rtc_policy(
        preprocessor=DataProcessorPipeline(steps=[relative_step]),
        postprocessor=DataProcessorPipeline(steps=[absolute_step]),
    )

    state1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    _prime_state(relative_step, state1)
    with caplog.at_level(logging.WARNING):
        with torch.inference_mode():
            policy._predict_with_rtc({})

    # Helper absent -> re-anchoring disabled, no absolute leftover retained.
    assert policy._rtc_relative_step is None
    assert policy._rtc_reanchor_fn is None
    assert policy._rtc_prev_chunk_abs is None
    leftover_model = _model_leftover()
    assert torch.allclose(policy._rtc_prev_chunk, leftover_model)
    assert any("reanchor_relative_rtc_prefix" in record.message for record in caplog.records)

    # State moves, but with no helper the leftover is fed back unchanged.
    state2 = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    _prime_state(relative_step, state2)
    with torch.inference_mode():
        policy._predict_with_rtc({})
    prev = captured[1]
    assert prev is not None
    assert torch.allclose(prev, leftover_model)


@_requires_reanchor
def test_resolve_rtc_rebase_backfills_action_names_from_policy_config():
    # A RelativeActionsProcessorStep that was serialized without action_names
    # (action_names is None) cannot build the per-joint relative mask, so the
    # re-anchor would convert the wrong action dimensions. _resolve_rtc_rebase_steps
    # must backfill the layout from the policy config's action_feature_names -
    # the same fallback LeRobot's RTCInferenceEngine applies - before wiring the
    # step in as the re-anchor source.
    names = [f"j{i}.pos" for i in range(_ACTION_DIM)]
    relative_step = RelativeActionsProcessorStep(enabled=True, action_names=None)
    policy, _ = _make_rtc_policy(
        preprocessor=DataProcessorPipeline(steps=[relative_step]),
        postprocessor=DataProcessorPipeline(steps=[IdentityProcessorStep()]),
    )
    # The fixture's inner policy exposes config.action_feature_names == names.
    assert relative_step.action_names is None

    policy._resolve_rtc_rebase_steps()

    assert policy._rtc_relative_step is relative_step
    assert policy._rtc_reanchor_fn is not None
    # Backfilled from config so the joint mask can be built for re-anchoring.
    assert relative_step.action_names == names


@_requires_reanchor
def test_resolve_rtc_rebase_leaves_action_names_unset_without_config_layout():
    # When the policy config carries no action_feature_names either, there is
    # nothing to backfill from: the relative step keeps action_names is None
    # (LeRobot builds an all-joints mask in that case) yet the step is still
    # resolved as the re-anchor source rather than being silently dropped.
    relative_step = RelativeActionsProcessorStep(enabled=True, action_names=None)
    policy, _ = _make_rtc_policy(
        preprocessor=DataProcessorPipeline(steps=[relative_step]),
        postprocessor=DataProcessorPipeline(steps=[IdentityProcessorStep()]),
    )
    policy._policy.config.action_feature_names = None

    policy._resolve_rtc_rebase_steps()

    assert policy._rtc_relative_step is relative_step
    assert policy._rtc_reanchor_fn is not None
    assert relative_step.action_names is None
