"""An optional-dependency skip must bind its names on every path to their use.

The suite gates on optional dependencies constantly, and the shortest way to
write that gate is a ``try`` whose handler calls :func:`pytest.skip`::

    try:
        from libero.libero.envs.robots.mounted_panda import MountedPanda
    except ImportError:
        pytest.skip("libero does not expose MountedPanda")
    qpos = MountedPanda().init_qpos          # <- bound only on the try path

That runs correctly, because ``pytest.skip`` raises. But the binding's
liveness is then a property of pytest's control flow rather than of the
enclosing function, so ``py/uninitialized-local-variable`` reports the use --
and every code-scanning alert opens a review thread that has to be resolved
before a merge, on whichever pull request happens to touch the file next.

Three sites of this shape carried five of those alerts on ``main`` for long
enough to become the lowest-numbered open alerts in the repository, and the
shape was reintroduced once after being fixed, which is what this check is
for: the same mistake now fails in a fraction of a second locally instead of
arriving as an alert after the merge.

The remedy is the suite's own idiom, and it is shorter than what it replaces:

* a missing **module** -- :func:`pytest.importorskip`, already used at ~890
  call sites here;
* a missing **attribute** on a module that does import -- ``getattr(module,
  name, None)`` plus an explicit skip, which keeps an upstream rename a skip
  rather than converting it into an ``AttributeError``;
* a **value** that has to be built (a model load, a decode) -- a module-level
  ``*_or_skip`` helper that returns it, so the caller's binding is
  unconditional. Raise ``pytest.skip.Exception`` inside such a helper rather
  than calling :func:`pytest.skip`, so that every path out of it is explicit;
  calling it moves the same unanalyzable branch into the helper, where it is
  reported as ``py/mixed-returns`` instead.

Scope is deliberately the skip-handler shape and nothing wider. A handler
that ends in ``return``, ``raise`` or :func:`pytest.fail` leaves the same
name conditionally bound, and none of those is reported: measured against
``main``, the tree carries 20 ``return`` sites, 4 ``pytest.fail`` sites and 1
``raise`` site of the same structure with zero alerts between them. Those
forms are idiomatic here -- a ``pytest.fail`` handler asserts the ``try`` body
succeeds, rather than gating on an environment -- so widening this check to
them would trade a bounded rule for a large mechanical rewrite that no gate
asks for.

A guard has a second obligation, checked here too because it is the same
construct and the same mistake: **the reason handed to the guard must not be
the first thing to read what the guard is unsure about.**::

    libero = pytest.importorskip("libero")
    mounted_panda = pytest.importorskip(
        "libero.libero.envs.robots.mounted_panda",
        reason=f"libero {libero.__version__} ...",   # <- evaluated eagerly
    )

An f-string argument is built before the call it is passed to, so a module
attribute read there runs on every host where the module imports -- that is,
on exactly the hosts the test exists for. ``libero`` is the case that makes it
concrete: the distribution the ``benchmark-libero`` extra resolves to ships an
empty top-level ``libero/__init__.py``, so that read raises ``AttributeError``
and a passing test becomes an error. No behavioural gate can catch it, because
a host without the dependency skips at the first ``importorskip`` and never
reaches the reason, which is why the check is on the source. Resolve the value
into a local first (see ``_installed_version`` in
``tests/benchmarks/libero/test_libero_adapter.py``) and quote that.

A read that already happened on the way to the diagnostic is not reported: the
attribute is then a proven input to the test's own decision rather than a bet
the message is making. ``tests/benchmarks/libero/test_numba_coverage_clash_remedy.py``
is that shape and is deliberately left alone.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEST_ROOTS = ("tests", "tests_integ")

# A sweep that reaches nothing would pass silently, so require that it saw a
# plausible amount of the tree. These are floors, not measurements.
_MINIMUM_FILES_SCANNED = 400
_MINIMUM_TRY_STATEMENTS_SCANNED = 200
_MINIMUM_GATED_FUNCTIONS_SCANNED = 80

# ``importorskip`` is deliberately absent: called from a handler it raises only
# when its own module is missing, so such a handler can fall through.
_SKIPPING_CALLS = frozenset({"skip"})

# Calls whose arguments are pure diagnostics - built to explain an outcome, so
# building one must not be able to raise. ``importorskip`` belongs here (unlike
# in ``_SKIPPING_CALLS``): the question is what its ``reason`` reads, not where
# the call sits.
_DIAGNOSTIC_CALLS = frozenset({"skip", "importorskip", "fail", "xfail"})


def _leaves_via_skip(handler: ast.ExceptHandler) -> bool:
    """Whether this handler's only exit is a :func:`pytest.skip` call.

    Args:
        handler: Exception handler to classify.

    Returns:
        ``True`` when the handler reaches a skip before any ``return`` or
        ``raise``. A handler that returns or raises is out of scope (see the
        module docstring).
    """
    for statement in handler.body:
        if isinstance(statement, (ast.Return, ast.Raise)):
            return False
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            func = statement.value.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in _SKIPPING_CALLS:
                return True
            if name in {"fail", "xfail", "exit"}:
                return False
    return False


def _statements_excluding_nested_scopes(node: ast.AST) -> list[ast.AST]:
    """Walk ``node`` without descending into a nested function or class.

    A name assigned inside a nested definition is that scope's local, not the
    one under test, so counting it would report bindings that were never in
    question.

    Args:
        node: Statement to walk.

    Returns:
        Every descendant node in the same scope, including ``node`` itself.
    """
    collected: list[ast.AST] = []
    queue: list[ast.AST] = [node]
    while queue:
        current = queue.pop()
        collected.append(current)
        for child in ast.iter_child_nodes(current):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            queue.append(child)
    return collected


def _names_bound_by(statements: list[ast.stmt]) -> set[str]:
    """Collect the plain names ``statements`` bind in their own scope.

    Args:
        statements: Statements to inspect.

    Returns:
        Every name bound by an import, assignment or ``with ... as``.
    """
    bound: set[str] = set()
    for statement in statements:
        for node in _statements_excluding_nested_scopes(statement):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bound.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                bound.add(node.target.id)
            elif isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
                bound.add(node.optional_vars.id)
    return bound


def _conditionally_bound_names(source: str) -> list[tuple[int, str]]:
    """Report every name a skipping ``try`` binds and later code reads.

    Args:
        source: Python source text.

    Returns:
        ``(line, name)`` pairs, sorted, one per offending name. A name already
        bound earlier in the same statement list is excluded: pre-binding is a
        valid fix for this shape and is in use in the tree.
    """
    tree = ast.parse(source)
    findings: set[tuple[int, str]] = set()
    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(parent, field, None)
            if not isinstance(statements, list):
                continue
            for index, statement in enumerate(statements):
                if not isinstance(statement, ast.Try) or not statement.handlers:
                    continue
                if not all(_leaves_via_skip(handler) for handler in statement.handlers):
                    continue
                candidates = _names_bound_by(statement.body) - _names_bound_by(list(statements[:index]))
                if not candidates:
                    continue
                for later in statements[index + 1 :]:
                    for node in ast.walk(later):
                        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in candidates:
                            findings.add((statement.lineno, node.id))
    return sorted(findings)


def _count_try_statements(source: str) -> int:
    """Count ``try`` statements in ``source``.

    Args:
        source: Python source text.

    Returns:
        The number of ``try`` statements, used as a non-vacuity floor.
    """
    return sum(1 for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Try))


def _gated_module_names(function: ast.AST) -> set[str]:
    """Names this function binds from a :func:`pytest.importorskip` result.

    Those are the modules the function has declared it is unsure about, so an
    attribute of one is exactly what a diagnostic must not depend on.

    Args:
        function: Function definition to scan.

    Returns:
        The bound names, empty when the function gates on nothing.
    """
    names: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if isinstance(func, ast.Attribute) and func.attr == "importorskip":
            names |= {target.id for target in node.targets if isinstance(target, ast.Name)}
    return names


def _diagnostic_calls(function: ast.AST) -> list[ast.Call]:
    """Every diagnostic call in ``function``.

    Args:
        function: Function definition to scan.

    Returns:
        The matching calls, in walk order.
    """
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _DIAGNOSTIC_CALLS
    ]


def _reasons_reading_an_ungated_attribute(source: str) -> list[tuple[int, str]]:
    """Find diagnostics that are the first thing to read a gated attribute.

    An attribute already read outside a diagnostic is proven present by the
    time the message quotes it, so only a read with no such predecessor is
    reported. Nested scopes are walked, which can only widen the set of reads
    treated as proven, so the rule under-reports rather than over-reports.

    Args:
        source: Python source of one test module.

    Returns:
        ``(line, "module.attribute")`` pairs, sorted, for each read that a
        diagnostic is the first to make.
    """
    offenders: set[tuple[int, str]] = set()
    for function in ast.walk(ast.parse(source)):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        gated = _gated_module_names(function)
        if not gated:
            continue
        calls = _diagnostic_calls(function)
        arguments = [
            argument for call in calls for argument in list(call.args) + [keyword.value for keyword in call.keywords]
        ]
        in_a_diagnostic = {id(node) for argument in arguments for node in ast.walk(argument)}
        proven = {
            (node.value.id, node.attr)
            for node in ast.walk(function)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in gated
            and id(node) not in in_a_diagnostic
        }
        for argument in arguments:
            for node in ast.walk(argument):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in gated
                    and (node.value.id, node.attr) not in proven
                ):
                    offenders.add((node.lineno, f"{node.value.id}.{node.attr}"))
    return sorted(offenders)


def _test_sources() -> list[tuple[Path, str]]:
    """Read every Python file under the test roots.

    Returns:
        ``(path, source)`` pairs, sorted by path.
    """
    sources: list[tuple[Path, str]] = []
    for root in _TEST_ROOTS:
        for path in sorted((_REPO_ROOT / root).rglob("*.py")):
            sources.append((path, path.read_text(encoding="utf-8")))
    return sources


class TestASkippingGuardBindsItsNames:
    """No test binds a name only inside a ``try`` whose handler skips."""

    def test_no_name_is_bound_only_on_the_non_skipping_path(self) -> None:
        """Every name a skipping guard binds is bound on all paths to its use."""
        sources = _test_sources()
        assert len(sources) >= _MINIMUM_FILES_SCANNED, (
            f"scan reached only {len(sources)} files under {_TEST_ROOTS}; "
            f"expected at least {_MINIMUM_FILES_SCANNED}, so a clean result here proves nothing"
        )
        tries = sum(_count_try_statements(source) for _, source in sources)
        assert tries >= _MINIMUM_TRY_STATEMENTS_SCANNED, (
            f"scan reached only {tries} try statements; expected at least "
            f"{_MINIMUM_TRY_STATEMENTS_SCANNED}, so a clean result here proves nothing"
        )

        offenders = [
            f"{path.relative_to(_REPO_ROOT)}:{line} binds {name!r}"
            for path, source in sources
            for line, name in _conditionally_bound_names(source)
        ]
        assert not offenders, (
            "a name bound inside a try whose handler calls pytest.skip is read after the block, so it is "
            "bound only on the success path as far as any analysis of the enclosing function can tell "
            "(py/uninitialized-local-variable):\n  "
            + "\n  ".join(offenders)
            + "\nUse pytest.importorskip for a missing module, getattr(module, name, None) plus an explicit "
            "skip for a missing attribute, or a module-level *_or_skip helper that returns the value."
        )

    def test_the_detector_reports_a_planted_skipping_guard(self) -> None:
        """A clean sweep means the tree is clean, not that the rule is inert."""
        planted = (
            "import pytest\n"
            "def test_x():\n"
            "    try:\n"
            "        from somewhere import Thing\n"
            "    except ImportError:\n"
            "        pytest.skip('absent')\n"
            "    assert Thing\n"
        )
        assert _conditionally_bound_names(planted) == [(3, "Thing")]

    def test_the_detector_reports_a_planted_value_built_under_a_skip(self) -> None:
        """The shape covers a built value, not only an import."""
        planted = (
            "import pytest\n"
            "def test_x():\n"
            "    try:\n"
            "        policy = load()\n"
            "    except Exception as exc:\n"
            "        pytest.skip(f'no model: {exc}')\n"
            "    return wrap(policy)\n"
        )
        assert _conditionally_bound_names(planted) == [(3, "policy")]

    @pytest.mark.parametrize(
        ("label", "handler_body"),
        [
            ("fail", "        pytest.fail('must not happen')"),
            ("return", "        return"),
            ("raise", "        raise AssertionError('leaked')"),
        ],
    )
    def test_a_non_skipping_handler_is_out_of_scope(self, label: str, handler_body: str) -> None:
        """Only the reported shape is in scope; see the module docstring."""
        source = (
            "import pytest\n"
            "def test_x():\n"
            "    try:\n"
            "        value = compute()\n"
            "    except ValueError:\n"
            f"{handler_body}\n"
            "    assert value\n"
        )
        assert _conditionally_bound_names(source) == [], f"{label} handler must not be reported"

    def test_pre_binding_before_the_try_resolves_the_shape(self) -> None:
        """Binding the name first is a valid fix and must not be reported."""
        source = (
            "import pytest\n"
            "def test_x():\n"
            "    cls = None\n"
            "    try:\n"
            "        cls = resolve()\n"
            "    except ImportError:\n"
            "        pytest.skip('absent')\n"
            "    assert cls\n"
        )
        assert _conditionally_bound_names(source) == []

    def test_a_name_the_try_never_binds_is_not_reported(self) -> None:
        """A guard that binds nothing has nothing to report."""
        source = (
            "import pytest\n"
            "def test_x(module):\n"
            "    try:\n"
            "        module.check()\n"
            "    except ImportError:\n"
            "        pytest.skip('absent')\n"
            "    assert module\n"
        )
        assert _conditionally_bound_names(source) == []


class TestASkipReasonCannotBeWhatFails:
    """No guard's reason is the first thing to read what the guard gates on."""

    def test_no_reason_reads_an_attribute_the_guard_has_not_proven(self) -> None:
        """A diagnostic must explain an outcome, never produce one."""
        sources = _test_sources()
        assert len(sources) >= _MINIMUM_FILES_SCANNED, (
            f"scan reached only {len(sources)} files under {_TEST_ROOTS}; "
            f"expected at least {_MINIMUM_FILES_SCANNED}, so a clean result here proves nothing"
        )
        gated = sum(
            1
            for _, source in sources
            for function in ast.walk(ast.parse(source))
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) and _gated_module_names(function)
        )
        assert gated >= _MINIMUM_GATED_FUNCTIONS_SCANNED, (
            f"scan reached only {gated} functions that gate on importorskip; expected at least "
            f"{_MINIMUM_GATED_FUNCTIONS_SCANNED}, so a clean result here proves nothing"
        )

        offenders = [
            f"{path.relative_to(_REPO_ROOT)}:{line} reads {read}"
            for path, source in sources
            for line, read in _reasons_reading_an_ungated_attribute(source)
        ]
        assert not offenders, (
            "a diagnostic reads an attribute of a module the same function gates on with "
            "importorskip, and nothing on the way there has read it. An f-string argument is "
            "built before the call receives it, so a module that lacks the attribute turns the "
            "test into an AttributeError instead of the skip the guard was written to produce - "
            "on precisely the hosts that have the dependency:\n  "
            + "\n  ".join(offenders)
            + "\nResolve the value into a local first and quote that."
        )

    def test_the_detector_reports_a_planted_eager_reason(self) -> None:
        """A clean sweep means the tree is clean, not that the rule is inert."""
        planted = (
            "import pytest\n"
            "def test_x():\n"
            "    dep = pytest.importorskip('dep')\n"
            "    sub = pytest.importorskip('dep.sub', reason=f'dep {dep.__version__} lacks sub')\n"
            "    assert sub\n"
        )
        assert _reasons_reading_an_ungated_attribute(planted) == [(4, "dep.__version__")]

    def test_the_detector_reports_a_planted_eager_skip_reason(self) -> None:
        """The shape covers a plain skip, not only ``importorskip``."""
        planted = (
            "import pytest\n"
            "def test_x():\n"
            "    dep = pytest.importorskip('dep')\n"
            "    if not hasattr(dep, 'Thing'):\n"
            "        pytest.skip(f'dep {dep.__version__} lacks Thing')\n"
            "    assert dep.Thing\n"
        )
        assert _reasons_reading_an_ungated_attribute(planted) == [(5, "dep.__version__")]

    def test_an_attribute_read_before_the_diagnostic_is_not_reported(self) -> None:
        """A read the test already depends on is proven, not a bet.

        ``tests/benchmarks/libero/test_numba_coverage_clash_remedy.py`` is this
        shape: it reads ``coverage.__version__`` to decide whether to skip, so
        quoting it in the reason cannot introduce a new failure.
        """
        source = (
            "import pytest\n"
            "def test_x():\n"
            "    dep = pytest.importorskip('dep')\n"
            "    installed = parse(dep.__version__)\n"
            "    if installed < FLOOR:\n"
            "        pytest.skip(f'dep {dep.__version__} predates the floor')\n"
            "    assert dep\n"
        )
        assert _reasons_reading_an_ungated_attribute(source) == []

    def test_a_constant_reason_is_not_reported(self) -> None:
        """A reason that reads nothing cannot fail."""
        source = (
            "import pytest\n"
            "def test_x():\n"
            "    dep = pytest.importorskip('dep')\n"
            "    sub = pytest.importorskip('dep.sub', reason='dep does not expose sub')\n"
            "    assert sub\n"
        )
        assert _reasons_reading_an_ungated_attribute(source) == []

    def test_a_local_resolved_first_is_the_remedy(self) -> None:
        """The fix this check asks for must itself pass."""
        source = (
            "import pytest\n"
            "def test_x():\n"
            "    dep = pytest.importorskip('dep')\n"
            "    dep_version = _installed_version('dep', dep)\n"
            "    sub = pytest.importorskip('dep.sub', reason=f'dep {dep_version} lacks sub')\n"
            "    assert sub\n"
        )
        assert _reasons_reading_an_ungated_attribute(source) == []

    def test_a_function_that_gates_on_nothing_is_not_reported(self) -> None:
        """Without an importorskip there is no gated module to bet on."""
        source = (
            "import pytest\n"
            "def test_x(dep):\n"
            "    if not dep.ready:\n"
            "        pytest.skip(f'dep {dep.__version__} not ready')\n"
            "    assert dep\n"
        )
        assert _reasons_reading_an_ungated_attribute(source) == []
