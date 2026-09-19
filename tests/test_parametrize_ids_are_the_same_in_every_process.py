"""A ``parametrize`` table renders the same test IDs in every process.

pytest-xdist runs the suite as several worker processes. Each one imports the
whole test tree and collects it independently, then the workers compare their
collections: one test ID that differs between two of them aborts the entire
session with "Different tests were collected between gw0 and gw1" before a
single test runs. So the suite is only runnable in parallel if every ID is a
function of the source and of nothing else.

A ``parametrize`` table is built once per process, at import time, which makes
two kinds of expression inside one unsafe:

* a read of the clock or of entropy - ``time.time()`` in
  ``tests/mesh/test_wire_timestamp_must_be_finite.py`` gave each worker a
  different float, and therefore a different ID, for the same row;
* a value that renders its own address - ``object()`` and ``memoryview(...)``
  in ``tests/drivers/test_telemetry_coercion_refuses_the_same_non_readings.py``
  inherit the default ``__repr__``, which prints ``<object object at 0x...>``,
  and that table labels its rows with ``ids=repr``.

Both are invisible in a serial run: the ID is used once and never compared.
Neither site was special, so the rule is derived over the tree rather than
pinned at the two files that broke it. ``ids=repr`` itself is fine and is the
package's usual spelling for a domain table - what has to hold is that the
values it renders are literals.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent

#: Calls that answer differently in each process. Matched on the trailing dotted
#: name, so ``time.time()``, ``real_time.time()`` and a ``from time import time``
#: spelling are all covered.
_PER_PROCESS_CALLS = frozenset(
    {
        "time",
        "time_ns",
        "monotonic",
        "monotonic_ns",
        "perf_counter",
        "perf_counter_ns",
        "now",
        "utcnow",
        "uuid1",
        "uuid4",
        "urandom",
        "random",
        "randint",
        "randrange",
        "getpid",
        "mkdtemp",
        "mkstemp",
        "token_hex",
        "id",
    }
)

#: Constructors whose instances inherit ``object.__repr__`` and so render the
#: address they happen to live at. Only a problem where the IDs are derived from
#: the values: with no ``ids=`` pytest numbers such a row ``value3``, which is
#: stable.
_ADDRESS_IN_REPR = frozenset({"object", "memoryview"})

_REPR_IDS = frozenset({"repr", "str", "ascii"})


def _dotted_tail(node: ast.expr) -> str:
    """The last component of a call target, or "" when it is not a name."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _renders_one_of(names: set[str], body: ast.expr) -> bool:
    """Whether ``body`` puts the rendering of a value bound to one of ``names`` in the ID.

    ``type(v).__name__`` - the spelling two suites already use for a table of
    stand-in objects - is stable; ``repr(v)``, ``str(v)``, ``f"{v}"`` and a bare
    ``v`` are not, and slicing the text (``repr(v)[:24]``) does not make them so:
    ``<object object at 0x7f5c`` is 24 characters, so the truncation keeps
    exactly the address prefix that differs between two processes.
    """
    if isinstance(body, ast.Name) and body.id in names:
        return True
    for inner in ast.walk(body):
        if isinstance(inner, ast.Call) and _dotted_tail(inner.func) in _REPR_IDS:
            return True
        if isinstance(inner, ast.FormattedValue) and isinstance(inner.value, ast.Name) and inner.value.id in names:
            return True
    return False


def _lambda_renders_the_value(node: ast.Lambda) -> bool:
    """Whether an ``ids=`` lambda puts the value's own rendering in the ID."""
    return _renders_one_of({argument.arg for argument in node.args.args}, node.body)


def _comprehension_renders_the_values(node: ast.ListComp) -> bool:
    """Whether an ``ids=[... for v in table]`` list renders the rows it walks.

    A comprehension is a lambda spelled inline: the element expression plays
    the body and the ``for`` targets play the parameters. It is graded the same
    way, because the ID it produces is the same string.
    """
    names = {
        target.id
        for generator in node.generators
        for target in ast.walk(generator.target)
        if isinstance(target, ast.Name)
    }
    return _renders_one_of(names, node.elt)


def _ids_render_the_values(keywords: list[ast.keyword]) -> bool:
    """Whether this ``parametrize`` derives its IDs from the values themselves."""
    for keyword in keywords:
        if keyword.arg == "ids":
            if isinstance(keyword.value, ast.Lambda):
                return _lambda_renders_the_value(keyword.value)
            if isinstance(keyword.value, ast.ListComp):
                return _comprehension_renders_the_values(keyword.value)
            return _dotted_tail(keyword.value) in _REPR_IDS
    return False


def _parametrize_decorators(tree: ast.Module) -> list[ast.Call]:
    """Every ``@pytest.mark.parametrize(...)`` decorator in a module."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and _dotted_tail(decorator.func) == "parametrize":
                found.append(decorator)
    return found


def _tables(tree: ast.Module) -> dict[str, ast.expr]:
    """Module-level assignments, so ``parametrize("v", TABLE)`` can be followed."""
    tables: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.value is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    tables[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            tables[node.target.id] = node.value
    return tables


def offences(source: str, label: str) -> list[str]:
    """Every per-process expression reachable from a ``parametrize`` table.

    Args:
        source: Python source of a test module.
        label: How to name the module in a finding (a path, usually).

    Returns:
        One line per offence, naming the file, the line and the expression.
    """
    tree = ast.parse(source)
    tables = _tables(tree)
    found: list[str] = []
    for decorator in _parametrize_decorators(tree):
        argvalues = decorator.args[1] if len(decorator.args) > 1 else None
        for keyword in decorator.keywords:
            if keyword.arg == "argvalues":
                argvalues = keyword.value
        if argvalues is None:
            continue
        # Follow the tables the argvalues expression names, one level: a domain
        # table is nearly always a module-level constant referenced by name.
        reachable: list[ast.expr] = [argvalues]
        for node in ast.walk(argvalues):
            if isinstance(node, ast.Name) and node.id in tables:
                reachable.append(tables[node.id])
        renders_values = _ids_render_the_values(decorator.keywords)
        for root in reachable:
            for node in ast.walk(root):
                if not isinstance(node, ast.Call):
                    continue
                tail = _dotted_tail(node.func)
                if tail in _PER_PROCESS_CALLS:
                    found.append(f"{label}:{node.lineno}: {tail}() in a parametrize table")
                elif tail in _ADDRESS_IN_REPR and renders_values:
                    found.append(f"{label}:{node.lineno}: {tail}() labelled by its repr, which is its address")
    return sorted(set(found))


def _test_sources() -> list[Path]:
    """Every test module in the repository."""
    paths = sorted(TESTS_ROOT.rglob("test_*.py"))
    paths += sorted((REPO_ROOT / "tests_integ").rglob("test_*.py"))
    return paths


def test_no_parametrize_table_is_built_from_the_clock_or_an_address() -> None:
    """The invariant xdist enforces at collection, checked at the source."""
    found: list[str] = []
    for path in _test_sources():
        found += offences(path.read_text(encoding="utf-8"), str(path.relative_to(REPO_ROOT)))
    assert found == [], "a parametrize ID would differ between xdist workers:\n" + "\n".join(found)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('@pytest.mark.parametrize("v", [1, time.time()])\ndef test_x(v): pass\n', "time() in a parametrize table"),
        ('@pytest.mark.parametrize("v", [uuid.uuid4()])\ndef test_x(v): pass\n', "uuid4() in a parametrize table"),
        (
            'T = [(object(), 1)]\n@pytest.mark.parametrize(("v", "n"), T, ids=repr)\ndef test_x(v, n): pass\n',
            "object() labelled by its repr",
        ),
        (
            '@pytest.mark.parametrize("v", [memoryview(b"ab")], ids=lambda v: repr(v))\ndef test_x(v): pass\n',
            "memoryview() labelled by its repr",
        ),
        (
            # The shape that shipped in tests/mesh/test_iot_provisioning_flag_domain.py:
            # a comprehension over the table, and a slice that keeps the address.
            'T = [1, object()]\n@pytest.mark.parametrize("v", T, ids=[repr(v)[:24] for v in T])\ndef test_x(v): pass\n',
            "object() labelled by its repr",
        ),
        (
            'T = [object()]\n@pytest.mark.parametrize("v", T, ids=[f"{v}" for v in T])\ndef test_x(v): pass\n',
            "object() labelled by its repr",
        ),
    ],
    ids=[
        "clock",
        "entropy",
        "address-via-table",
        "address-via-lambda",
        "address-via-comprehension",
        "address-via-f-string",
    ],
)
def test_the_check_reports_each_way_an_id_can_drift(source: str, expected: str) -> None:
    """A grader that reports nothing on a clean tree has to be shown to report."""
    found = offences(source, "m.py")
    assert any(expected in line for line in found), found


@pytest.mark.parametrize(
    "source",
    [
        # ids=repr over literals is the package's usual spelling and is stable.
        '@pytest.mark.parametrize("v", [1, float("nan"), "x"], ids=repr)\ndef test_x(v): pass\n',
        # An address-bearing value is fine when pytest numbers the row instead.
        '@pytest.mark.parametrize("v", [object()])\ndef test_x(v): pass\n',
        # A clock read inside the test body is not part of any ID.
        '@pytest.mark.parametrize("v", [1])\ndef test_x(v): assert time.time() > v\n',
        # The spelling two suites already use for a table of stand-in objects.
        '@pytest.mark.parametrize("v", [object()], ids=lambda v: type(v).__name__)\ndef test_x(v): pass\n',
        # A comprehension over literals is the same stable spelling as ids=repr.
        'T = [1, "x", None]\n@pytest.mark.parametrize("v", T, ids=[repr(v)[:24] for v in T])\ndef test_x(v): pass\n',
        # A comprehension that labels the row by type, as the lambda above does.
        'T = [object()]\n@pytest.mark.parametrize("v", T, ids=[type(v).__name__ for v in T])\ndef test_x(v): pass\n',
    ],
    ids=[
        "literals-by-repr",
        "address-numbered-by-pytest",
        "clock-in-the-body",
        "address-labelled-by-type",
        "comprehension-over-literals",
        "comprehension-labelled-by-type",
    ],
)
def test_a_stable_table_is_not_reported(source: str) -> None:
    """The check has to leave the 230 tables that already spell IDs by repr alone."""
    assert offences(source, "m.py") == []
