"""No example computes an audit-integrity verdict over the developer's whole log.

An example that prints a scoped audit trail and an integrity verdict is
printing one document, so both halves have to describe the same records.
:func:`~strands_robots.mesh.audit.verify_audit_integrity` called with no
argument does not: it re-reads the entire log, which unlike a test's log is the
developer's real ``~/.strands_robots/mesh_audit.jsonl`` -- examples do not
redirect ``STRANDS_MESH_AUDIT_DIR``, and are not supposed to, because their
point is to write where the mesh really writes.

Measured on ``e4fe2f9``, before the call site this module guards was paired.
``examples/fleet/04_emergency_evacuation.py`` scoped its read to the run
(``read_audit_log(since=run_start - 1.0)``) and then attested everything::

    report = build_incident_report(records, verify_audit_integrity())

With 4000 records of prior history in the log and 5 events from the run, the
rendered incident report read::

    Audit integrity: ok=False (signed=5/4005)

    | t | peer | event | detail |
    |---|---|---|---|
    | +0.00s | evac-coordinator | evacuation_alarm | ... |     <- 5 rows
    ...

A header counting 4005 records above a table showing 5, in a forensic artifact
whose whole purpose is to say what happened during one incident. The ``ok``
value is the worse half: the planted history is unsigned, which is what a log
written before ``STRANDS_MESH_AUDIT_PSK`` was configured looks like, and with a
PSK set at verification time an unsigned record is a forgery by definition. So
the verdict is not merely wide -- a completely successful evacuation reports
``ok=False``, which reads as tamper evidence. The same run with the verdict
scoped to the records shown reports ``ok=True (signed=5/5)``.

Why the rule is keyed on the *call site* rather than on the resolved path:

- The hazard is not that the shared log is read. It is that a verdict about one
  record set is printed as a verdict about another, and only the call site knows
  which records the surrounding code is reporting. A path-level rule would have
  to model that, and would also have to forbid the read outright -- which would
  break a forensic example whose subject genuinely *is* the whole log.
- ``tests/`` is deliberately out of scope, and the difference is mechanical
  rather than a matter of trust: a test redirects ``STRANDS_MESH_AUDIT_DIR`` to
  ``tmp_path`` (see ``tests/test_fleet_emergency_evacuation.py``), so a
  no-argument call there reads a log containing only what that test wrote, which
  is exactly the whole log it means. The same call in an example reads whatever
  the developer's machine has accumulated. The two are the same expression with
  different meanings, so the scope is the rule.
- ``records`` is accepted positionally or by keyword and the two are the same
  pairing, so both forms satisfy this module. What it refuses is the argument
  being absent.

The behavioural half of this pair lives in
``tests/test_fleet_emergency_evacuation.py::test_the_report_attests_the_records_it_shows_and_not_the_whole_log``,
which pins the verdict a correctly paired report renders. That test cannot pin
this: it calls ``build_incident_report`` directly, so a call site that went back
to computing its own unscoped verdict would leave it green. This module is the
half that reads the call sites.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"

_GUARDED_FUNCTION = "verify_audit_integrity"

# The whole-log read is the point of these, so a no-argument call is correct.
# Empty today; an entry here needs a comment saying why the whole log is the
# record set the example reports on.
_WHOLE_LOG_IS_THE_SUBJECT: frozenset[str] = frozenset()


def _called_function_name(node: ast.Call) -> str | None:
    """The bare name of a call target, through an attribute access if needed.

    Both ``verify_audit_integrity(...)`` and ``audit.verify_audit_integrity(...)``
    reach the same function and are the same exposure.
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _integrity_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _called_function_name(node) == _GUARDED_FUNCTION
    ]


def _attests_a_record_set(call: ast.Call) -> bool:
    """True when the call names the records it is a verdict about."""
    if call.args:
        return True
    return any(keyword.arg == "records" for keyword in call.keywords)


def _example_sources() -> list[tuple[Path, ast.AST]]:
    # Annotated because ``ast.parse`` returns ``Module``: an inferred
    # ``list[tuple[Path, Module]]`` is not a ``list[tuple[Path, AST]]``, since
    # ``list`` is invariant.
    sources: list[tuple[Path, ast.AST]] = []
    for path in sorted(_EXAMPLES_DIR.rglob("*.py")):
        if path.relative_to(_REPO_ROOT).as_posix() in _WHOLE_LOG_IS_THE_SUBJECT:
            continue
        sources.append((path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))))
    return sources


def test_no_example_attests_a_record_set_it_does_not_name() -> None:
    """Every ``verify_audit_integrity`` call in ``examples/`` names its records."""
    offenders = [
        f"{path.relative_to(_REPO_ROOT).as_posix()}:{call.lineno}"
        for path, tree in _example_sources()
        for call in _integrity_calls(tree)
        if not _attests_a_record_set(call)
    ]
    assert not offenders, (
        "verify_audit_integrity() with no argument re-reads the developer's whole "
        "audit log, so its verdict describes records the example is not showing. "
        "Pass the records being reported on: " + ", ".join(offenders)
    )


def test_the_scan_reaches_the_calls_it_is_meant_to_check() -> None:
    """Non-vacuity: an example really does call the guarded function.

    Without this, a scan that resolved no call sites -- a renamed function, a
    moved ``examples/`` directory, a parse that silently yielded nothing --
    would satisfy the assertion above by finding nothing to reject.
    """
    calls = [(path, call) for path, tree in _example_sources() for call in _integrity_calls(tree)]
    assert calls, f"no {_GUARDED_FUNCTION} call found under {_EXAMPLES_DIR}; the scan is not reaching the examples"


def test_the_guard_detects_an_unscoped_call() -> None:
    """Planted positive: the predicate rejects what it claims to reject.

    Both spellings of the correct form are accepted, so the rule cannot be
    satisfied merely by importing the function differently.
    """
    tree = ast.parse(
        "\n".join(
            (
                "verify_audit_integrity()",  # offender: no record set
                "audit.verify_audit_integrity()",  # offender through an attribute
                "verify_audit_integrity(records)",  # positional
                "audit.verify_audit_integrity(records=records)",  # keyword
            )
        )
    )
    calls = _integrity_calls(tree)
    assert len(calls) == 4, "the walker missed a call spelling"
    assert [_attests_a_record_set(call) for call in calls] == [False, False, True, True]
