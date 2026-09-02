# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""No module in the package lets a steppable clock decide whether to keep waiting.

``time.time()`` is not a clock: it is the current opinion about the date, and an
NTP correction, a ``date -s`` or a resume from suspend moves it by an arbitrary
amount. A gate built on it therefore moves with the correction -- a forward step
ends a wait early with the work still in flight, a backward step runs past the
budget by the size of the step -- and the tree settled that boundary several
times over, each time the same way: a *duration* this process decides on its own
belongs on ``time.monotonic()``, while an *absolute stamp* that something off
this machine correlates stays on the wall clock.

This scan is the package-wide half of that contract. It used to live beside the
tool wait-budget tests and walk ``strands_robots/tools`` alone, which is a root
narrower than the shape it grades: it read clean while the one offender in the
tree sat in ``strands_robots/dashboard/auth.py``, where a WebAuthn challenge's
five-minute TTL was measured on the wall clock. Running the same predicate over
the whole package found it immediately, so the predicate was never the problem
and the walk root was. It now walks the package, with no exemption list, so a
new *subsystem* cannot reintroduce the idiom either -- not just a new tool.

What the shape does and does not carry, stated so it is not mistaken for a
complete guard. A wall-clock read inside the test of a ``while`` or of an ``if``
that leaves the wait is a decision and is reported. Two things are deliberately
not: a read that only *reports* (a record's timestamp, a logged duration), and a
read hoisted into a local before the comparison (``now = time.time()`` compared
further down). The subsystem grader owning each surface pins the rest on
behaviour, which is stronger than any source shape -- see
``tests/test_dashboard_challenge_expiry_survives_a_clock_step.py``,
``tests/mesh/test_mesh_durations_survive_a_clock_step.py``,
``tests/test_control_loop_budgets_survive_a_clock_step.py`` and the rendering,
rollout and RTC members of the same family.

Two wall-clock comparisons in ``mesh/core.py`` are correct and must stay: the
estop and resume freshness windows measure ``now - envelope_t`` against a
timestamp a *peer* put on the wire, which is the one place a stamp crosses a
machine boundary (``docs/mesh.md``). They are not reported here because the read
is hoisted, but a future widening of this scan must keep clearing them.
"""

from __future__ import annotations

import ast
import pathlib

import strands_robots

_PACKAGE_ROOT = pathlib.Path(strands_robots.__file__).parent


def _reads_wall_clock(node: ast.AST) -> bool:
    """True if ``node`` contains a ``time.time()`` / ``time.time_ns()`` read."""
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in ("time", "time_ns")
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == "time"
        ):
            return True
    return False


def _wall_clock_wait_decisions(source: str) -> list[tuple[int, str]]:
    """Sites in ``source`` where a wall-clock read decides whether to keep waiting.

    Two shapes carry that decision: a ``while`` whose test reads the wall clock,
    and an ``if`` that reads it to leave a wait (``raise`` / ``break`` /
    ``return`` / ``continue``). A wall-clock read that only *reports* -- a
    record's timestamp, a logged duration -- is not a decision and is not
    reported.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.While) and _reads_wall_clock(node.test):
            found.append((node.lineno, "while-gate"))
        elif isinstance(node, ast.If) and _reads_wall_clock(node.test):
            exits = {type(sub).__name__ for sub in ast.walk(ast.Module(body=node.body, type_ignores=[]))}
            if exits & {"Raise", "Break", "Return", "Continue"}:
                found.append((node.lineno, "wait-abort"))
    return found


def test_no_module_lets_a_steppable_clock_decide_whether_to_keep_waiting() -> None:
    """No module under ``strands_robots`` gates a wait or an expiry on ``time.time()``."""
    offenders: list[str] = []
    scanned = 0
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        scanned += 1
        for lineno, shape in _wall_clock_wait_decisions(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(_PACKAGE_ROOT.parent)}:{lineno} ({shape})")
    assert scanned >= 100, f"premise: the scan reached only {scanned} modules, so the walk root is wrong"
    assert not offenders, (
        "these gates let the wall clock decide whether to keep waiting, so an NTP "
        "step moves the deadline by the size of the step: "
        f"{offenders}. Measure a duration on time.monotonic(); the wall clock is "
        "for absolute stamps something off this machine correlates."
    )


def test_the_scan_detects_a_planted_wall_clock_wait() -> None:
    """A clean sweep means the package is right, not that the scan sees nothing."""
    planted = (
        "import time\n"
        "def wait(timeout):\n"
        "    deadline = time.time() + timeout\n"
        "    while time.time() < deadline:\n"
        "        pass\n"
        "def abort(deadline):\n"
        "    if time.time() >= deadline:\n"
        "        raise TimeoutError\n"
    )
    shapes = {shape for _, shape in _wall_clock_wait_decisions(planted)}
    assert shapes == {"while-gate", "wait-abort"}, f"the scan missed a planted wait: {shapes}"


def test_a_reported_wall_clock_stamp_is_not_read_as_a_wait_decision() -> None:
    """Reporting the date is not deciding how long to wait, and stays allowed."""
    reporting_only = (
        "import time\n"
        "def record(chunk):\n"
        "    return {'timestamp': time.time(), 'data': chunk}\n"
        "def log(start):\n"
        "    print(time.time() - start)\n"
    )
    assert _wall_clock_wait_decisions(reporting_only) == []
