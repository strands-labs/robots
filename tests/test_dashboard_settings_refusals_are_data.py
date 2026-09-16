"""A refused settings value is reported as data, never as a rendered exception.

:func:`~strands_robots.dashboard.settings.update_strict` hands its refusal text to
whoever asked for the write, and ``POST /api/settings`` puts that text in an HTTP
response body. Text that arrives there off an ``except ... as exc`` clause is a
rendering of an exception object - the shape by which a stack trace, a filesystem
path or any other internal reaches a client that asked for none, and the shape
CodeQL's ``py/stack-trace-exposure`` reads. The store composed each reason itself
and then carried it out on an exception the write path caught and interpolated, so
the text on the wire was benign only because every ``raise`` site happened to be
authored; one wrapping a third-party error would have shipped it.

So the grader RETURNS its reason. ``_graded`` answers ``(value, None)`` or
``(None, reason)``, the write path appends what it was handed, and there is no
handler on that path to name an exception - the flow does not exist rather than
being harmless today. Each reason still names the dotted key and the caller's own
value, unchanged: the cells below are the reason text itself, not a substring.

Returning the text moves it under the package rule that returned refusals render
the value they refuse through :func:`~strands_robots.utils.refusal_repr`, and that
rule earns its place here: the store rendered a bare ``{value!r}``, so a value
whose own ``__repr__`` raises made ``update_strict`` raise - the refusal path
failing on the one input it exists to answer. It reports now.

``_read_file`` keeps its ``except Exception as exc`` and is deliberately outside
the roster here: it renders that exception into a log line, which is the operator
reading their own machine, not a response body.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from strands_robots.dashboard import settings

# Every function a settings write passes through. Spelled out rather than derived
# so a rename cannot empty the roster and pass the pin vacuously; the cell below
# fails if this module stops having one of them.
_WRITE_PATH = (
    "_finite_float",
    "_finite_int",
    "_graded",
    "_degraded",
    "_coerce",
    "_update",
    "update",
    "update_strict",
)

# One row per refusal mechanism, with the reason in full. The value domain itself
# is crossed exhaustively in ``test_dashboard_settings_value_domain.py``; what is
# pinned here is that the reason comes back as a return value and reads the same.
_REFUSALS = [
    ("agent", "temperature", 3.0, "temperature: 3.0 is outside 0..2"),
    ("agent", "temperature", "abc", "temperature: 'abc' is not a number"),
    ("agent", "temperature", float("inf"), "temperature: inf is not a finite number"),
    ("agent", "max_tokens", 0, "max_tokens: 0 must be at least 1"),
    ("agent", "max_tokens", "x", "max_tokens: 'x' is not an integer"),
    ("mesh", "port", 70000, "port: 70000 is outside 1..65535"),
    ("mesh", "connect", 5, "connect: expected a list or comma-separated string, got int"),
    ("mesh", "connect", [None], "connect: entry 0 expected a string, got NoneType"),
    ("runtime", "trust_remote_code", "maybe", "trust_remote_code: 'maybe' is not a boolean (use true/false)"),
    ("agent", "model_id", {}, "model_id: expected a string, got dict"),
]


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Point the module at a scratch settings file so a write under test touches nothing else."""
    path = tmp_path / "settings.json"
    path.write_text("{}")
    monkeypatch.setattr(settings, "SETTINGS_FILE", path)
    settings.clear_overrides()
    settings.load(refresh=True)
    yield path
    settings.clear_overrides()
    settings.load(refresh=True)


@pytest.mark.parametrize(("section", "key", "value", "reason"), _REFUSALS)
def test_a_refused_value_comes_back_as_a_reason(store, section, key, value, reason):
    """The grader returns the reason. Nothing raises, so nothing has to be caught."""
    graded, reported = settings._graded(section, key, value)
    assert (graded, reported) == (None, reason)


def test_the_write_path_reports_what_the_grader_returned_and_stores_nothing(store):
    """The dotted key the caller wrote, the grader's reason, and an unchanged file."""
    changed, errors = settings.update_strict({"agent": {"temperature": 9.0, "max_tokens": -1}})
    assert (changed, errors) == (
        [],
        ["agent.temperature: 9.0 is outside 0..2", "agent.max_tokens: -1 must be at least 1"],
    )
    assert store.read_text() == "{}"


def test_a_value_whose_own_repr_raises_is_still_reported(store):
    """A reason is built through the shared renderer, so building one cannot raise.

    ``numbers.Real`` is a registration rather than an inheritance, so a scalar that
    satisfies the store's type test owes it no working ``__repr__``. Such a value
    arrives through the module's Python API - ``override``, ``update_strict`` - which
    the CLI and the agent console both call.
    """

    class Unprintable:
        def __repr__(self) -> str:
            raise RuntimeError("this repr raises")

    changed, errors = settings.update_strict({"agent": {"temperature": Unprintable()}})
    assert (changed, errors) == ([], ["agent.temperature: <unrepresentable Unprintable> is not a number"])


def test_no_function_on_the_write_path_names_the_exception_it_caught(store):
    """A reason that reaches a response body is composed by this module, not by a handler."""
    module = ast.parse(Path(settings.__file__).read_text(encoding="utf-8"))
    functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
    present = [name for name in _WRITE_PATH if name in functions]
    # Graded before the roster is checked for completeness, so a tree that has not
    # got the split yet fails on the handler it still has rather than on the two
    # function names it is missing.
    named = [
        f"{name} binds {handler.name!r} at line {handler.lineno}"
        for name in present
        for handler in ast.walk(functions[name])
        if isinstance(handler, ast.ExceptHandler) and handler.name
    ]
    assert named == []
    assert present == list(_WRITE_PATH), "the roster names a function this module lost"
