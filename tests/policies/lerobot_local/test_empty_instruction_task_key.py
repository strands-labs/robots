"""Empty instruction must not crash a language-VLA preprocessor.

``run_policy`` documents its instruction default as ``instruction=""``. A
language-conditioned VLA (SmolVLA, pi0, ...) routes that instruction through
LeRobot's ``TokenizerProcessorStep``, which reads ``complementary_data["task"]``.
If :class:`ProcessorBridge` omits the ``task`` key when the instruction is
empty, that indexing raises a cryptic ``KeyError: 'task'`` -- the exact failure
the bridge's transition-based path exists to avoid (see ``preprocess`` docstring).

These tests build a real (network-free) LeRobot ``DataProcessorPipeline`` whose
single step mimics the tokenizer's ``["task"]`` access, so the regression is
verified against real LeRobot pipeline internals, not a mock of the bridge.
"""

from typing import Any

import pytest

pytest.importorskip("lerobot.processor.pipeline")

from lerobot.configs.types import PipelineFeatureType, PolicyFeature  # noqa: E402
from lerobot.processor.pipeline import (  # noqa: E402
    ComplementaryDataProcessorStep,
    DataProcessorPipeline,
)

from strands_robots.policies.lerobot_local.processor import ProcessorBridge  # noqa: E402


class _RequiresTaskStep(ComplementaryDataProcessorStep):
    """Minimal stand-in for LeRobot's TokenizerProcessorStep.

    Indexes ``complementary_data["task"]`` exactly like the real tokenizer's
    ``get_task``, so a missing key reproduces the production ``KeyError: 'task'``.
    Records the task it saw so tests can assert the empty-instruction value.
    """

    def __init__(self) -> None:
        self.seen_task: Any = "<unset>"

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        self.seen_task = complementary_data["task"]  # KeyError: 'task' if absent
        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def _bridge_with_task_step() -> tuple[ProcessorBridge, _RequiresTaskStep]:
    step = _RequiresTaskStep()
    pipe = DataProcessorPipeline(steps=[step])
    return ProcessorBridge(preprocessor=pipe, device="cpu"), step


_OBS = {"observation.state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}


def test_empty_instruction_does_not_raise_key_error():
    """The documented ``instruction=""`` default must reach the tokenizer step."""
    bridge, step = _bridge_with_task_step()
    # Pre-fix this raised RuntimeError("Preprocessor pipeline failed: 'task'").
    bridge.preprocess(dict(_OBS), instruction="")
    assert step.seen_task == ""


def test_none_instruction_does_not_raise_key_error():
    """A ``None`` instruction is also a valid empty task, not a crash."""
    bridge, step = _bridge_with_task_step()
    bridge.preprocess(dict(_OBS), instruction=None)
    assert step.seen_task == ""


def test_non_empty_instruction_still_forwarded_verbatim():
    """A real instruction must still reach the tokenizer unchanged."""
    bridge, step = _bridge_with_task_step()
    bridge.preprocess(dict(_OBS), instruction="pick up the red cube")
    assert step.seen_task == "pick up the red cube"
