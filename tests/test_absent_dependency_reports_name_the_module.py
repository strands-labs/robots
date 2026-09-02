"""An ``ImportError`` reporting an absent optional dependency must name it.

``ImportError`` carries a keyword-only ``name`` field for exactly one purpose:
telling a caller *which* module could not be imported, without parsing prose.
The interpreter always populates it. Code that constructs the exception itself
only populates it if it says ``name=``, and no site in this package did::

    grep -rn "raise ImportError(" strands_robots/ | grep -c "name="
    -> 0   (of 28 sites)

Both raise sites of :func:`strands_robots.utils.require_optional` and
:func:`strands_robots.utils.require_optionals` were among them -- the mechanism
AGENTS.md convention 7 makes mandatory for an optional dependency, and the path
48 call sites in the package take. So the project's own sanctioned way of
reporting an absent extra reported it in a form only a human could read.

What that costs, concretely: ``tests/test_docstring_xref_roles_resolve.py`` has
to decide, for an ``ImportError`` raised out of a module body it imported,
whether an extra is absent (this environment cannot grade the target) or a
``strands_robots`` path is dead (the rot it exists to catch). With ``name``
empty it fell back to the chained exception, and then to matching the reported
*message* -- and ``require_optionals`` raises after its ``except`` block has
exited, so ``__context__`` is ``None`` there and the message was the only thing
left. A reader reduced to reading prose is one wording change from
misclassifying, and misclassifying that direction aborted the sweep (#2963).

The fix is at the source, so the distinction does not depend on any reader's
fallback: every site that reports an absent dependency names it. This module
pins that as a property of the package rather than of the sites it was written
against, because the hazard arrives with a *new* raise site: nothing else
refuses the shape. ``ruff`` has no rule for it, ``mypy`` cannot know which
argument carries a module name, and code scanning has no query for it.

Two sites do not report an absent dependency at all and are exempt by name
below, with the reason each is not one. The check is a subset assertion rather
than an equality: a site that stops being blind -- including by its module being
deleted -- must not turn the gate red for an unrelated change.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

import pytest

import strands_robots
from strands_robots.utils import require_optional, require_optionals

# The consumer this change was made for. Imported rather than restated so the
# cell below grades the guard's real classification, not a copy of it.
from tests.test_docstring_xref_roles_resolve import _undecidable_import

#: The shipped package. Reached through the imported module so a layout change
#: cannot silently narrow the scan to nothing.
_PACKAGE_ROOT = Path(strands_robots.__file__).resolve().parent

#: Extras group the probes below ask for. Synthetic: these cells exercise the
#: message formatter, not the project's extras table, and the remedy is asserted
#: through this name rather than spelled inline so the "every written
#: ``strands-robots[...]`` names a declared extra" audit reads a template hole
#: instead of an instruction.
_PROBE_EXTRA = "probe"

#: Lower bound on ``raise ImportError(...)`` sites found. Pinned so a scan
#: rooted somewhere unexpected fails loudly instead of reporting a clean sweep
#: over nothing. Measured at 28 when written.
_MINIMUM_CONSTRUCTED_IMPORT_ERRORS = 20

#: Package-relative modules whose constructed ``ImportError`` is not a report
#: that a dependency is absent, so there is no module name for it to carry.
_NOT_AN_ABSENT_DEPENDENCY_REPORT = {
    "policies/groot/policy.py": (
        "raised when the configured GR00T version string matches none of the "
        "supported loaders - a rejected argument, not a probe of an install"
    ),
    "policies/lerobot_local/resolution.py": (
        "raised when no lerobot policy class matches the requested policy_type; "
        "with lerobot present that is a typo'd argument, and naming a module "
        "would report an absence that did not happen"
    ),
}


class _Site(NamedTuple):
    """One ``raise ImportError(...)`` and what a reader could recover from it."""

    label: str
    lineno: int
    names_the_module: bool
    chains_a_cause: bool
    inside_a_handler: bool

    @property
    def recoverable(self) -> bool:
        """True when the absent module is reachable without reading the message.

        ``name=`` is the direct report. Failing that, ``raise ... from exc`` sets
        ``__cause__``, and raising lexically inside an ``except`` block sets
        ``__context__`` to the exception being handled -- ``from None`` suppresses
        the *rendering* of that chain but does not clear the attribute.
        """
        return self.names_the_module or self.chains_a_cause or self.inside_a_handler


def _import_error_sites(source: str, label: str) -> list[_Site]:
    """Every ``raise ImportError(...)`` in *source*, graded for recoverability."""
    sites: list[_Site] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.handlers: list[ast.ExceptHandler] = []

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            self.handlers.append(node)
            self.generic_visit(node)
            self.handlers.pop()

        def visit_Raise(self, node: ast.Raise) -> None:
            exc = node.exc
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "ImportError":
                from_none = isinstance(node.cause, ast.Constant) and node.cause.value is None
                sites.append(
                    _Site(
                        label=label,
                        lineno=node.lineno,
                        names_the_module=any(kw.arg == "name" for kw in exc.keywords),
                        chains_a_cause=node.cause is not None and not from_none,
                        inside_a_handler=bool(self.handlers),
                    )
                )
            self.generic_visit(node)

    Visitor().visit(ast.parse(source))
    return sites


def _package_sites() -> list[_Site]:
    """Grade every ``raise ImportError(...)`` in the shipped package."""
    sites: list[_Site] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        label = path.relative_to(_PACKAGE_ROOT).as_posix()
        sites.extend(_import_error_sites(path.read_text(encoding="utf-8"), label))
    return sites


class TestTheSanctionedMechanismNamesTheModule:
    """The two raise sites 48 call sites in the package go through."""

    def test_require_optional_names_the_module_it_could_not_import(self) -> None:
        with pytest.raises(ImportError) as caught:
            require_optional("a_module_this_environment_does_not_have", extra=_PROBE_EXTRA)
        assert caught.value.name == "a_module_this_environment_does_not_have"

    def test_require_optional_keeps_the_interpreter_report_on_the_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``name`` is what was asked for; ``__context__`` is what was missing.

        The two differ when the requested module is present but something *it*
        imports is not, so overwriting one with the other would lose a fact.
        """
        (tmp_path / "a_module_needing_something_absent.py").write_text(
            "import a_transitive_module_that_is_absent\n", encoding="utf-8"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(ImportError) as caught:
            require_optional("a_module_needing_something_absent", extra=_PROBE_EXTRA)

        assert caught.value.name == "a_module_needing_something_absent"
        assert getattr(caught.value.__context__, "name", None) == "a_transitive_module_that_is_absent"

    def test_require_optionals_names_the_first_missing_module_in_the_order_given(self) -> None:
        """The plural form has nothing but ``name``: it raises outside the handler."""
        with pytest.raises(ImportError) as caught:
            require_optionals(("absent_module_one", "absent_module_two"), extra=_PROBE_EXTRA)

        assert caught.value.name == "absent_module_one"
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None

    @pytest.mark.parametrize(
        ("call", "module"),
        [
            (
                lambda: require_optional("absent_singular_probe", extra=_PROBE_EXTRA, purpose="a probe"),
                "absent_singular_probe",
            ),
            (
                lambda: require_optionals(("absent_plural_probe",), extra=_PROBE_EXTRA, purpose="a probe"),
                "absent_plural_probe",
            ),
        ],
        ids=["require_optional", "require_optionals"],
    )
    def test_naming_the_module_leaves_the_remedy_message_intact(self, call, module: str) -> None:
        """``name=`` is metadata, not text: the install instruction is unchanged."""
        with pytest.raises(ImportError) as caught:
            call()
        reported = str(caught.value)
        assert module in reported
        assert "a probe" in reported
        assert f"pip install 'strands-robots[{_PROBE_EXTRA}]'" in reported
        assert "name=" not in reported


class TestTheGuardClassifiesOnTheName:
    """The xref guard now reads the field, not the chain and not the prose."""

    @pytest.mark.parametrize(
        ("call", "module"),
        [
            (lambda: require_optional("absent_read_probe", extra=_PROBE_EXTRA), "absent_read_probe"),
            (lambda: require_optionals(("absent_read_plural",), extra=_PROBE_EXTRA), "absent_read_plural"),
        ],
        ids=["require_optional", "require_optionals"],
    )
    def test_an_absent_extra_is_reported_as_the_module_name(self, call, module: str) -> None:
        with pytest.raises(ImportError) as caught:
            call()
        # Pre-fix this returned the first line of the install message for the
        # singular form and, for the plural one, could return nothing else.
        assert _undecidable_import(caught.value) == module

    def test_a_strands_robots_path_is_still_read_as_the_rot_being_graded(self) -> None:
        """The opposite verdict is unchanged: a package path is not an absent extra."""
        assert _undecidable_import(ImportError("gone", name="strands_robots.no_such_module")) is None


class TestTheRuleHoldsAcrossThePackage:
    def test_every_absent_dependency_report_names_its_module(self) -> None:
        sites = _package_sites()
        assert len(sites) >= _MINIMUM_CONSTRUCTED_IMPORT_ERRORS, (
            f"found only {len(sites)} constructed ImportError sites under {_PACKAGE_ROOT}; "
            "the scan is rooted wrong or the pattern stopped matching"
        )
        blind = [s for s in sites if not s.recoverable and s.label not in _NOT_AN_ABSENT_DEPENDENCY_REPORT]
        assert not blind, (
            "these ImportErrors report an absent module in prose only - pass name= so a "
            "caller does not have to parse the message: " + ", ".join(f"{s.label}:{s.lineno}" for s in blind)
        )


class TestTheRuleIsNotVacuous:
    @pytest.mark.parametrize(
        ("source", "recoverable"),
        [
            ('raise ImportError("zmq is required")', False),
            ('raise ImportError("zmq is required", name="zmq")', True),
            ('try:\n    import zmq\nexcept ImportError as e:\n    raise ImportError("no zmq") from e', True),
            ('try:\n    import zmq\nexcept ImportError:\n    raise ImportError("no zmq")', True),
            ('try:\n    import zmq\nexcept ImportError:\n    pass\nraise ImportError("no zmq")', False),
            ('try:\n    import zmq\nexcept ImportError:\n    raise ImportError("no zmq") from None', True),
        ],
        ids=["bare", "named", "from-cause", "in-handler", "after-handler", "from-none"],
    )
    def test_recoverability_is_graded_per_raise_shape(self, source: str, recoverable: bool) -> None:
        sites = _import_error_sites(source, "planted.py")
        assert len(sites) == 1
        assert sites[0].recoverable is recoverable

    def test_a_planted_blind_site_is_reported(self) -> None:
        """The exemptions are by module path, so an unlisted module is refused."""
        sites = _import_error_sites('raise ImportError("torch is required")', "policies/somewhere_new.py")
        blind = [s for s in sites if not s.recoverable and s.label not in _NOT_AN_ABSENT_DEPENDENCY_REPORT]
        assert [s.lineno for s in blind] == [1]

    def test_a_raise_of_another_exception_type_is_not_graded(self) -> None:
        assert _import_error_sites('raise ValueError("not an import problem")', "planted.py") == []
