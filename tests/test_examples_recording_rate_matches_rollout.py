"""Every shipped example records at the rate its own rollout captures.

The dataset recorder writes one frame per control step with **no decimation**,
so ``start_recording(fps=N)`` is only honoured by a ``run_policy`` that captures
at ``control_frequency=N``. A mismatch is refused outright rather than written
at a distorted timestamp rate, which means the episode lands with zero frames
and the example demonstrates nothing it claims to.

That refusal arrived with the recorder-rate guard and three shipped examples
were left declaring 30 fps while their rollouts ran at the 50 Hz default, so
each recorded an empty dataset: ``03_record_dataset.py`` exited on
``stop_recording`` reporting ``0 frames`` against a docstring promising
``100 frames``, ``07_post_tune_any_policy.py`` printed
``Recorded LeRobotDataset ->`` and then failed two steps later inside lerobot
with an unrelated-looking Hub 404 for the empty local set, and
``locomotion/vla_g1_workflow.py`` printed ``Episode N/N recorded.`` per episode
for a dataset with no frames in it.

The rates are compared as the example states them, and the default is read from
``run_policy``'s own signature rather than restated here, so a change to that
default cannot leave this test asserting a stale number.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation.base import SimEngine

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

#: The rate a rollout captures at when the caller does not say. Read from the
#: signature so this file cannot disagree with the code it grades.
_DEFAULT_CONTROL_FREQUENCY = inspect.signature(SimEngine.run_policy).parameters["control_frequency"].default

#: A refactor that stops reaching the examples must fail loudly rather than
#: report a clean sweep over nothing.
_MINIMUM_GRADED_EXAMPLES = 3

_DYNAMIC = object()


def _literal(node: ast.AST) -> Any:
    """The literal value of ``node``, or ``_DYNAMIC`` when it is computed."""
    try:
        return ast.literal_eval(node)
    except Exception:
        return _DYNAMIC


def _keyword(call: ast.Call, name: str) -> Any:
    for kw in call.keywords:
        if kw.arg == name:
            return _literal(kw.value)
    return None


def _splatted_names(call: ast.Call) -> list[str]:
    """Names of ``**kwargs`` splats in ``call`` (``kw.arg is None``)."""
    return [kw.value.id for kw in call.keywords if kw.arg is None and isinstance(kw.value, ast.Name)]


def _dict_assignments(tree: ast.Module, name: str) -> list[dict[str, Any]]:
    """Every literal dict assigned to ``name`` anywhere in ``tree``."""
    out: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        if isinstance(node.value, ast.Dict):
            keys = [_literal(k) if k is not None else _DYNAMIC for k in node.value.keys]
            vals = [_literal(v) for v in node.value.values]
            out.append({k: v for k, v in zip(keys, vals, strict=True) if isinstance(k, str)})
    return out


def _capture_rates(tree: ast.Module, call: ast.Call) -> set[Any]:
    """Every rate ``call`` can capture at, given the module's assignments.

    A caller may pass the rate directly, omit it (taking the signature
    default), or splat a dict that carries it - a locomotion example builds one
    dict per data source, so every branch it can take is graded.
    """
    direct = _keyword(call, "control_frequency")
    if direct is not None:
        return {direct}
    splats = _splatted_names(call)
    if not splats:
        return {_DEFAULT_CONTROL_FREQUENCY}
    rates: set[Any] = set()
    for name in splats:
        assignments = _dict_assignments(tree, name)
        if not assignments:
            rates.add(_DYNAMIC)
            continue
        for mapping in assignments:
            rates.add(mapping.get("control_frequency", _DEFAULT_CONTROL_FREQUENCY))
    return rates


def _graded_examples() -> list[tuple[Path, float, set[Any]]]:
    """``(path, declared fps, capture rates)`` per example that records."""
    graded: list[tuple[Path, float, set[Any]]] = []
    for path in sorted(_EXAMPLES.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        recordings, rollouts = [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "start_recording":
                    recordings.append(_keyword(node, "fps"))
                elif node.func.attr == "run_policy":
                    rollouts.append(node)
        declared = {f for f in recordings if isinstance(f, (int, float))}
        # One rate per example is what a static reading can attribute; an
        # example declaring several is left ungraded rather than guessed at.
        if len(declared) != 1 or not rollouts:
            continue
        fps = float(declared.pop())
        rates: set[Any] = set()
        for call in rollouts:
            rates |= _capture_rates(tree, call)
        graded.append((path, fps, rates))
    return graded


def test_every_recording_example_captures_at_the_rate_it_declares() -> None:
    """No example declares one rate and captures at another."""
    graded = _graded_examples()
    assert len(graded) >= _MINIMUM_GRADED_EXAMPLES, (
        f"only {len(graded)} example(s) were graded, expected at least "
        f"{_MINIMUM_GRADED_EXAMPLES}: the scan is no longer reaching "
        f"{_EXAMPLES}, so a clean result would prove nothing"
    )
    mismatched = []
    for path, fps, rates in graded:
        wrong = sorted(r for r in rates if r is not _DYNAMIC and isinstance(r, (int, float)) and float(r) != fps)
        if wrong:
            mismatched.append(
                f"{path.relative_to(_EXAMPLES.parent)}: records at {fps:g} fps but its "
                f"rollout captures at {', '.join(f'{w:g}' for w in wrong)} Hz - the "
                f"recorder refuses the mismatch and the episode lands with zero frames"
            )
    assert not mismatched, "example(s) record at a rate their rollout does not capture at:\n  " + "\n  ".join(
        mismatched
    )


def test_the_default_capture_rate_is_read_from_the_signature() -> None:
    """The default is the one ``run_policy`` declares, not a copy of it."""
    assert isinstance(_DEFAULT_CONTROL_FREQUENCY, (int, float))
    assert _DEFAULT_CONTROL_FREQUENCY > 0


@pytest.mark.parametrize(
    ("source", "should_flag"),
    [
        # Omitting the rate takes the default, which only matches a recording
        # that declares that same rate.
        ("sim.start_recording(fps=30)\nsim.run_policy(robot_name='r')", True),
        (f"sim.start_recording(fps={_DEFAULT_CONTROL_FREQUENCY:g})\nsim.run_policy(robot_name='r')", False),
        # Stating it wrongly is flagged; stating it correctly is not.
        ("sim.start_recording(fps=30)\nsim.run_policy(control_frequency=50.0)", True),
        ("sim.start_recording(fps=30)\nsim.run_policy(control_frequency=30.0)", False),
        # A splatted dict is resolved through its assignment.
        ("kw = {'control_frequency': 50.0}\nsim.start_recording(fps=30)\nsim.run_policy(**kw)", True),
        ("kw = {'control_frequency': 30.0}\nsim.start_recording(fps=30)\nsim.run_policy(**kw)", False),
    ],
)
def test_the_grader_flags_a_mismatch_and_leaves_a_match_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, should_flag: bool
) -> None:
    """A clean sweep means the examples agree, not that the grader is blind."""
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "planted.py").write_text(source, encoding="utf-8")
    monkeypatch.setattr(f"{__name__}._EXAMPLES", examples)
    (path, fps, rates) = _graded_examples()[0]
    flagged = any(r is not _DYNAMIC and isinstance(r, (int, float)) and float(r) != fps for r in rates)
    assert flagged is should_flag, f"grader returned rates={rates!r} against fps={fps} for:\n{source}"
