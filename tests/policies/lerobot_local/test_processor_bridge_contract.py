"""ProcessorBridge degrades gracefully and delegates its lifecycle correctly.

:class:`~strands_robots.policies.lerobot_local.processor.ProcessorBridge` wraps a
pair of (optional) LeRobot pre/post-processor pipelines. Callers construct it in
several partially-configured states -- no pipeline at all (mock policies, ACT
checkpoints that ship no processor), preprocessor-only, or both -- and rely on it
being a safe no-op wherever a pipeline is absent rather than raising.

These tests pin that contract on the bridge's own surface, using duck-typed fake
pipelines so they exercise the wrapper logic without loading real model weights:

* ``apply_embodiment`` / ``preprocessor_steps`` / ``postprocess`` / ``reset`` are
  no-ops (not errors) when the corresponding pipeline is absent;
* a raw ``postprocess`` action passes through unchanged with no postprocessor;
* ``reset`` delegates to whichever pipelines are present and skips absent ones;
* ``apply_embodiment`` falls back to inserting a fresh rename step when the
  existing rename step refuses a ``rename_map`` assignment (frozen/odd step);
* ``get_info`` / ``__repr__`` report the configured state accurately.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap
from strands_robots.policies.lerobot_local.processor import ProcessorBridge


class _FakePipeline:
    """Minimal stand-in for a LeRobot DataProcessorPipeline.

    Exposes the small surface ProcessorBridge touches: a mutable ``steps`` list,
    a ``reset`` hook, and ``__len__`` (used by ``ProcessorBridge.__repr__``).
    """

    def __init__(self, steps: list | None = None) -> None:
        self.steps = steps if steps is not None else []
        self.reset = MagicMock(name="reset")

    def __len__(self) -> int:
        return len(self.steps)


class _FrozenRenameStep:
    """A rename step whose ``rename_map`` cannot be assigned.

    Named ``RenameObservationsProcessorStep`` so ProcessorBridge recognises it as
    the pipeline's rename step, but any attempt to set ``rename_map`` raises --
    emulating a frozen dataclass step. This drives the insert-a-fresh-step
    fallback path.
    """

    _registry_name = "rename_observations_processor"

    @property
    def rename_map(self) -> dict:
        return {}

    @rename_map.setter
    def rename_map(self, value: dict) -> None:
        raise AttributeError("cannot assign field 'rename_map' (frozen step)")


# Give the frozen step the class name ProcessorBridge matches on by identity.
_FrozenRenameStep.__name__ = "RenameObservationsProcessorStep"


def test_apply_embodiment_is_noop_without_preprocessor() -> None:
    """A bridge with no preprocessor swallows apply_embodiment silently."""
    bridge = ProcessorBridge()
    embodiment = EmbodimentMap(name="e", obs_rename={"front": "observation.images.image"})

    bridge.apply_embodiment(embodiment)  # must not raise

    assert bridge.has_preprocessor is False
    assert bridge.preprocessor_steps == []


def test_preprocessor_steps_empty_without_pipeline() -> None:
    """preprocessor_steps returns an empty list (not None / not an error)."""
    assert ProcessorBridge().preprocessor_steps == []


class _FreshRenameStep:
    """Controllable rename step the fallback inserts (stands in for LeRobot's)."""

    def __init__(self, rename_map: dict) -> None:
        self.rename_map = dict(rename_map)


def test_apply_embodiment_inserts_fresh_rename_when_existing_step_is_frozen(monkeypatch) -> None:
    """A frozen rename step triggers the insert-a-new-rename-step fallback.

    The bridge cannot mutate the frozen step's rename_map, so it must fall back
    to prepending a fresh rename step carrying the embodiment's obs_rename. A
    fake ``lerobot.processor.rename_processor`` module is injected so the test
    exercises the bridge's fallback logic hermetically (no real LeRobot import).
    """
    fake_mod = types.ModuleType("lerobot.processor.rename_processor")
    setattr(fake_mod, "RenameObservationsProcessorStep", _FreshRenameStep)
    monkeypatch.setitem(sys.modules, "lerobot.processor.rename_processor", fake_mod)
    # No pack-state step for an empty state_keys embodiment: keep the fallback
    # path hermetic (older LeRobot without a registered pack-state also returns
    # None here), isolating the rename-fallback behaviour under test.
    monkeypatch.setattr(
        "strands_robots.policies.lerobot_local.embodiment.register_pack_state_step",
        lambda: None,
    )

    frozen = _FrozenRenameStep()
    pipeline = _FakePipeline(steps=[frozen])
    bridge = ProcessorBridge(preprocessor=pipeline)
    obs_rename = {"front": "observation.images.image"}
    embodiment = EmbodimentMap(name="e", obs_rename=obs_rename, state_keys=[])

    bridge.apply_embodiment(embodiment)

    # Fallback prepends a usable rename step whose map carries obs_rename; the
    # frozen original is kept after it (order preserved, nothing dropped).
    steps = bridge.preprocessor_steps
    assert isinstance(steps[0], _FreshRenameStep)
    assert steps[0].rename_map == obs_rename
    assert frozen in steps


def test_postprocess_passes_action_through_without_postprocessor() -> None:
    """With no postprocessor, postprocess returns the exact action object."""
    sentinel = object()
    assert ProcessorBridge().postprocess(sentinel) is sentinel


def test_postprocess_wraps_pipeline_failure_in_runtime_error() -> None:
    """A raising postprocessor surfaces as a RuntimeError, not the raw error."""
    post = MagicMock()
    post.process_action.side_effect = ValueError("bad action")
    bridge = ProcessorBridge(postprocessor=post)

    try:
        bridge.postprocess([0.0, 1.0])
    except RuntimeError as exc:
        assert "Postprocessor pipeline failed" in str(exc)
        assert isinstance(exc.__cause__, ValueError)
    else:  # pragma: no cover - failure to raise is the bug under test
        raise AssertionError("expected RuntimeError from failing postprocessor")


def test_reset_delegates_to_both_present_pipelines() -> None:
    """reset forwards to whichever pipelines exist."""
    pre = _FakePipeline()
    post = _FakePipeline()
    bridge = ProcessorBridge(preprocessor=pre, postprocessor=post)

    bridge.reset()

    pre.reset.assert_called_once_with()
    post.reset.assert_called_once_with()


def test_reset_is_noop_on_empty_bridge() -> None:
    """reset on a fully-empty bridge is a harmless no-op."""
    ProcessorBridge().reset()  # must not raise


def test_get_info_and_repr_report_configured_state() -> None:
    """get_info / __repr__ reflect which pipelines are loaded."""
    empty = ProcessorBridge()
    info = empty.get_info()
    assert info["has_preprocessor"] is False
    assert info["has_postprocessor"] is False
    assert info["is_active"] is False
    assert "pre=None" in info["repr"] and "post=None" in info["repr"]

    both = ProcessorBridge(
        preprocessor=_FakePipeline(steps=[object(), object()]),
        postprocessor=_FakePipeline(steps=[object()]),
    )
    info = both.get_info()
    assert info["has_preprocessor"] is True
    assert info["has_postprocessor"] is True
    assert info["is_active"] is True
    assert "pre=2steps" in info["repr"] and "post=1steps" in info["repr"]
