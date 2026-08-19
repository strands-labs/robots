"""A dataset frame the recorder could not write must fail the rollout.

``PolicyRunner`` tolerates a bounded number of *consecutive* ``on_frame``
failures because that hook is caller telemetry (see
``TestOnFrameFailureCounter`` in the behaviour suite, whose
``test_consecutive_counter_resets_on_success`` pins the reset). The library's
own dataset recorder is fed through that same hook, and ``DatasetRecorder``
defaults to ``strict=True`` - fail-fast - so a failed write raises. Absorbed by
the telemetry tolerance, whose counter resets on every success, an intermittent
write failure never reaches the limit: the rollout reports success while the
episode on disk is short and every surviving frame has been re-timestamped from
the declared ``fps``.

These tests pin the split: a lost recording frame is fatal, a caller's telemetry
failure keeps its tolerance.

They also pin the split's *documentation*. The exemption is invisible in the
signature, so the only place a caller can learn it is the docstring of the
surface they call, and every such docstring named ``CooperativeStop`` alone -
telling a caller who followed the hook's own documented use ("use it to record
frames") that a failed dataset write is logged and tolerated when it is neither.
:class:`TestTheDocumentedPostureNamesEveryExemptClass` derives the exempt set
from the handlers themselves, so a third exempt class cannot be added without
the surfaces that describe the posture being updated with it.
"""

from __future__ import annotations

import ast
import inspect
import logging
import re
import sys
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.dataset_recorder import DatasetRecorder, RecordingFrameError
from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.policy_runner import PolicyRunner

_FEATURES: dict[str, Any] = {
    "observation.state": {"dtype": "float32", "names": ["1", "2", "3", "4", "5", "6"]},
    "action": {"dtype": "float32", "names": ["1", "2", "3", "4", "5", "6"]},
}


class _FlakyDataset:
    """Fake LeRobotDataset whose write fails on a chosen subset of frames."""

    def __init__(self, *, fail_every: int) -> None:
        self.repo_id = "local/flaky"
        self.root = "/tmp/local-flaky"
        self.features = _FEATURES
        self.fail_every = fail_every
        self.attempts = 0
        self.written = 0

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.attempts += 1
        if self.fail_every and self.attempts % self.fail_every == 0:
            raise RuntimeError(f"transient dataset write failure at frame {self.attempts}")
        self.written += 1

    def save_episode(self) -> None:
        return None


def _recording_sim(recorder: Any) -> Simulation:
    """A one-robot sim with ``recorder`` attached as the live session's writer.

    Opens a recording session without a real ``LeRobotDataset`` - the same
    injection seam the chunk-alignment and episode-boundary regressions use.
    """
    sim = Simulation(tool_name="frame_loss", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm", data_config="so100")
    assert sim._world is not None
    sim._world._backend_state["recording"] = True
    sim._world._backend_state["trajectory"] = []
    sim._world._backend_state["dataset_recorder"] = recorder
    return sim


def _policy(sim: Simulation) -> MockPolicy:
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    return policy


class TestALostRecordingFrameIsFatal:
    def test_an_intermittent_write_failure_aborts_the_rollout(self) -> None:
        """Every other frame failing must fail the rollout, not truncate it.

        The telemetry counter resets on each success, so alternating failures
        never reach the consecutive limit. Pre-fix this rollout reported success
        with half of its frames missing from the dataset.
        """
        ds = _FlakyDataset(fail_every=2)
        sim = _recording_sim(DatasetRecorder(dataset=ds, task="t"))
        try:
            result = sim.run_policy(
                robot_name="arm",
                policy_object=_policy(sim),
                n_steps=20,
                control_frequency=50.0,
                fast_mode=True,
            )
        finally:
            sim.cleanup()

        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "transient dataset write failure" in text
        assert "the recording is incomplete" in text
        # Stopped at the first lost frame rather than writing 10 of 20.
        assert ds.attempts == 2
        assert ds.written == 1

    def test_a_persistent_write_failure_aborts_on_the_first_frame(self) -> None:
        """A recorder that can never write must not burn the tolerance window.

        The generic tolerance would spend ``max_onframe_failures`` steps before
        aborting; a lost dataset frame is already data loss on the first one.
        """
        ds = _FlakyDataset(fail_every=1)
        sim = _recording_sim(DatasetRecorder(dataset=ds, task="t"))
        try:
            result = sim.run_policy(
                robot_name="arm",
                policy_object=_policy(sim),
                n_steps=20,
                control_frequency=50.0,
                fast_mode=True,
            )
        finally:
            sim.cleanup()

        assert result["status"] == "error", result
        assert ds.attempts == 1
        assert ds.written == 0

    def test_the_eval_loop_does_not_swallow_it_either(self) -> None:
        """``eval_policy`` must surface it, not log it as best-effort telemetry.

        Its sibling pin ``test_on_frame_exception_is_logged_not_fatal`` asserts a
        generic hook failure leaves the eval successful; this is the exception to
        that rule. It surfaces as a raise rather than an error envelope because
        that is how ``eval_policy`` already surfaces every rollout failure - a
        policy raising from ``get_actions`` propagates the same way, unchanged
        here.
        """
        sim = Simulation(tool_name="frame_loss_eval", mesh=False)
        sim.create_world()
        sim.add_robot(name="arm", data_config="so100")
        steps: list[int] = []

        def lose_a_frame(step: int, obs: dict[str, Any], action: dict[str, Any]) -> None:
            steps.append(step)
            raise RecordingFrameError("dataset add_frame failed after 0 frame(s) written")

        try:
            with pytest.raises(RecordingFrameError, match="add_frame failed"):
                sim.eval_policy(
                    robot_name="arm",
                    policy_object=_policy(sim),
                    n_episodes=1,
                    max_steps=4,
                    control_frequency=50.0,
                    on_frame=lose_a_frame,
                )
        finally:
            sim.cleanup()

        # Aborted on the first lost frame instead of running the episode out.
        assert steps == [0]


class TestCallerTelemetryKeepsItsTolerance:
    def test_an_alternating_telemetry_failure_still_completes(self) -> None:
        """The exclusion is scoped to the recorder, not to every hook failure."""
        sim = Simulation(tool_name="frame_loss_telemetry", mesh=False)
        sim.create_world()
        sim.add_robot(name="arm", data_config="so100")
        calls = {"n": 0}

        def flaky_telemetry(step: int, obs: dict[str, Any], action: dict[str, Any]) -> None:
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise ValueError(f"telemetry hiccup {calls['n']}")

        try:
            runner = PolicyRunner(sim)
            result = runner.run(
                "arm",
                _policy(sim),
                n_steps=20,
                control_frequency=50,
                fast_mode=True,
                on_frame=flaky_telemetry,
            )
        finally:
            sim.cleanup()

        assert result["status"] == "success", result
        assert calls["n"] == 20

    def test_a_non_strict_recorder_failure_remains_tolerated(self, caplog: pytest.LogCaptureFixture) -> None:
        """``strict=False`` still drops, counts and completes - unchanged."""
        ds = _FlakyDataset(fail_every=2)
        recorder = DatasetRecorder(dataset=ds, task="t", strict=False)
        sim = _recording_sim(recorder)
        try:
            with caplog.at_level(logging.WARNING, logger="strands_robots.dataset_recorder"):
                result = sim.run_policy(
                    robot_name="arm",
                    policy_object=_policy(sim),
                    n_steps=20,
                    control_frequency=50.0,
                    fast_mode=True,
                )
        finally:
            sim.cleanup()

        assert result["status"] == "success", result
        assert ds.attempts == 20
        assert recorder.dropped_frame_count == 10


class TestAnExplicitToleranceDoesNotResizeTheExemption:
    """``max_onframe_failures`` sizes the tolerance, not the exempt set.

    The knob's own documentation is what this pins: a caller who raises the
    tolerance is buying patience for flaky telemetry, and a lost dataset frame is
    outside what that patience covers however high the number goes.
    """

    @pytest.mark.parametrize(
        ("exception_cls", "expected_calls"),
        [
            pytest.param(RuntimeError, 5, id="telemetry-failure-spends-the-tolerance"),
            pytest.param(RecordingFrameError, 1, id="lost-frame-ignores-the-tolerance"),
        ],
    )
    def test_only_a_telemetry_failure_spends_the_configured_tolerance(
        self, exception_cls: type[Exception], expected_calls: int
    ) -> None:
        tolerance = 5
        sim = Simulation(tool_name="frame_loss_tolerance", mesh=False)
        sim.create_world()
        sim.add_robot(name="arm", data_config="so100")
        calls: list[int] = []

        def failing_hook(step: int, obs: dict[str, Any], action: dict[str, Any]) -> None:
            calls.append(step)
            raise exception_cls("synthetic hook failure")

        try:
            result = PolicyRunner(sim).run(
                robot_name="arm",
                policy=_policy(sim),
                n_steps=20,
                control_frequency=50.0,
                on_frame=failing_hook,
                max_onframe_failures=tolerance,
            )
        finally:
            sim.cleanup()

        assert result["status"] == "error", result
        assert len(calls) == expected_calls, (
            f"with max_onframe_failures={tolerance}, a {exception_cls.__name__} hook was called "
            f"{len(calls)} time(s); expected {expected_calls}"
        )


class TestRecorderRaisesADistinguishableError:
    def test_strict_add_frame_raises_recording_frame_error_chaining_the_cause(self) -> None:
        """The type is what the runner keys on; the cause must stay readable."""
        ds = _FlakyDataset(fail_every=1)
        recorder = DatasetRecorder(dataset=ds, task="t")

        with pytest.raises(RecordingFrameError) as excinfo:
            recorder.add_frame(observation={"1": 1.0}, action={"1": 0.1}, task="t")

        assert "transient dataset write failure at frame 1" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        # The strict path re-raises rather than dropping, so the drop counter -
        # which only describes swallowed frames - must stay at zero.
        assert recorder.dropped_frame_count == 0
        assert recorder.frame_count == 0


def _on_frame_guards() -> list[ast.Try]:
    """Return the innermost ``try`` guarding each ``on_frame`` call site.

    An enclosing rollout ``try`` also contains the call and is not the handler
    that decides tolerance, so only the innermost one is returned. Shared by the
    handler guard and the docstring guard below, which must agree about which
    handlers define the posture.
    """
    tree = ast.parse(inspect.getsource(sys.modules[PolicyRunner.__module__]))

    def calls_on_frame(node: ast.Try) -> bool:
        return any(
            isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "on_frame"
            for stmt in node.body
            for child in ast.walk(stmt)
        )

    containing = [t for t in ast.walk(tree) if isinstance(t, ast.Try) and calls_on_frame(t)]
    containing_ids = {id(t) for t in containing}
    return [
        t
        for t in containing
        if not any(isinstance(o, ast.Try) and o is not t and id(o) in containing_ids for o in ast.walk(t))
    ]


def test_every_on_frame_call_site_excludes_a_lost_recording_frame() -> None:
    """No rollout loop may absorb a lost dataset frame into hook tolerance.

    Walks every ``try`` that invokes the ``on_frame`` hook - including the
    benchmark-spec eval loop, which needs a registered benchmark to reach
    behaviourally - and requires each to handle ``RecordingFrameError``.
    """
    guarded = _on_frame_guards()
    assert len(guarded) >= 3, f"expected every on_frame call site to be guarded, found {len(guarded)}"

    for node in guarded:
        names = {h.type.id for h in node.handlers if isinstance(h.type, ast.Name)}
        assert "RecordingFrameError" in names, (
            f"the on_frame try at line {node.lineno} does not exclude RecordingFrameError "
            f"from its tolerance (handlers: {sorted(names)})"
        )


# The posture is stated on four caller-facing surfaces plus ``run_policy``'s
# tolerance knob. Fewer graded blocks than this means the scan stopped reaching
# the docstrings, not that they agree with the handlers.
_MINIMUM_GRADED_DOC_BLOCKS = 5

# An Args entry begins a new block: a claim about ``on_frame`` and a claim about
# ``max_onframe_failures`` are separate promises and each has to stand alone.
_ARG_ENTRY = re.compile(r"^\s{8,}[a-z_][a-z0-9_]*:\s")

# A block is graded only when it quantifies over the hook's exceptions - the
# claim being checked is which of them the posture covers. A block that sizes the
# tolerance numerically makes no claim about exception classes and is left alone.
_QUANTIFIERS = ("logged at WARN", "consecutive")


def _exempt_hook_exception_classes() -> frozenset[str]:
    """Return the exception classes exempt from the ``on_frame`` posture.

    Read from the handlers themselves rather than listed here: a handler whose
    whole body is a bare ``raise`` re-raises instead of tolerating, which is what
    makes its class exempt. Deriving it means a third exempt class is graded
    against the docstrings the moment it is added.
    """
    exempt: set[str] = set()
    for guard in _on_frame_guards():
        for handler in guard.handlers:
            body = handler.body
            reraises = len(body) == 1 and isinstance(body[0], ast.Raise) and body[0].exc is None
            if isinstance(handler.type, ast.Name) and reraises:
                exempt.add(handler.type.id)
    return frozenset(exempt)


def _doc_blocks(docstring: str) -> list[str]:
    """Split ``docstring`` into blocks at blank lines and at Args entries."""
    blocks: list[str] = []
    current: list[str] = []
    for line in docstring.splitlines():
        if not line.strip() or _ARG_ENTRY.match(line):
            if current:
                blocks.append(" ".join(" ".join(current).split()))
            current = [line] if line.strip() else []
        else:
            current.append(line)
    if current:
        blocks.append(" ".join(" ".join(current).split()))
    return blocks


def _posture_blocks(source: str) -> list[tuple[str, str]]:
    """Return ``(owner, text)`` for each doc block describing the hook posture."""
    graded: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if not docstring:
            continue
        owner = getattr(node, "name", "<module>")
        for block in _doc_blocks(docstring):
            if not any(marker in block for marker in _QUANTIFIERS):
                continue
            if "exception" not in block:
                continue
            if "on_frame" not in block and "hook" not in block:
                continue
            graded.append((owner, block))
    return graded


def _documented_surfaces() -> list[tuple[str, str, str]]:
    """Return ``(module, owner, block)`` for every graded block on both modules."""
    graded: list[tuple[str, str, str]] = []
    for label, module in (("base", SimEngine.__module__), ("policy_runner", PolicyRunner.__module__)):
        source = inspect.getsource(sys.modules[module])
        graded.extend((label, owner, block) for owner, block in _posture_blocks(source))
    return graded


class TestTheDocumentedPostureNamesEveryExemptClass:
    """A surface that describes the hook posture names every exempt class.

    The tolerance is invisible in the signature, so a caller learns it only from
    the docstring of the surface they call. Naming one of the two exempt classes
    describes a posture the runner does not have.
    """

    def test_every_graded_surface_names_every_exempt_class(self) -> None:
        """Pre-fix, five surfaces promised a posture ``RecordingFrameError`` never had."""
        exempt = _exempt_hook_exception_classes()
        graded = _documented_surfaces()
        offenders = [
            (module, owner, sorted(cls for cls in exempt if cls not in block), block)
            for module, owner, block in graded
            if any(cls not in block for cls in exempt)
        ]
        assert not offenders, "\n".join(
            f"{module}.{owner} describes the on_frame posture without naming {missing}: {block[:150]}"
            for module, owner, missing, block in offenders
        )

    def test_the_exempt_set_is_read_from_the_handlers(self) -> None:
        """The graded set comes from the code, so a third class cannot slip past.

        Pins the two classes the handlers exempt today. Adding a third is a
        change to the posture every surface above describes, so it should fail
        here until those surfaces say so too.
        """
        assert _exempt_hook_exception_classes() == frozenset({"CooperativeStop", "RecordingFrameError"})

    def test_the_scan_reaches_the_surfaces_it_grades(self) -> None:
        """A reformat that hides the blocks must fail rather than report clean."""
        graded = _documented_surfaces()
        assert len(graded) >= _MINIMUM_GRADED_DOC_BLOCKS, (
            f"only {len(graded)} doc blocks describe the on_frame posture; "
            f"expected at least {_MINIMUM_GRADED_DOC_BLOCKS}"
        )
        owners = {owner for _module, owner, _block in graded}
        assert {"run_policy", "eval_policy", "evaluate_benchmark", "run", "evaluate"} <= owners, sorted(owners)

    def test_a_surface_naming_one_class_is_reported(self) -> None:
        """Non-vacuity: the grader must reject the wording this change replaced."""
        planted = '''
def f():
    """One line.

    Args:
        on_frame: A non-``CooperativeStop`` hook exception is logged at WARN.
    """
'''
        graded = _posture_blocks(planted)
        assert len(graded) == 1, graded
        assert "RecordingFrameError" not in graded[0][1]
