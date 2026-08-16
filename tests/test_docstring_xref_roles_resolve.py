"""Guard: fully-qualified ``strands_robots`` Sphinx cross-references must resolve.

A docstring role such as :func:`~strands_robots.simulation.predicates.base_below_z`
promises the reader an importable API object at that exact dotted path. When the
path is wrong - a private implementation renamed public, a symbol moved or split,
or, as happened here, a *registered predicate name* dressed up as an importable
function that never existed - the pointer is a silent dead end: Sphinx renders a
plain-text token, IDEs cannot jump to it, and the reader chases a path that
does not import.

The sibling filename guards (for example
:mod:`tests.mesh.test_docstring_module_xrefs`) already forbid citing a source
*file* by name; this guard closes the complementary gap for the *recommended*
form - the ``:mod:`` / ``:class:`` / ``:func:`` / ``:meth:`` roles that name a
dotted API path - by verifying that every fully-qualified ``strands_robots.*``
target actually resolves to a real module or attribute.

A target only counts as named if it is a CONTIGUOUS dotted path. A role whose
path is long enough to wrap over a line break carries a newline plus the next
line's indentation, so the token a reader copies out of the source does not
import - and neither does the whitespace-collapsed form a renderer would see.
The pattern here therefore admits whitespace inside the target and reports such
a role, rather than failing to match it and exempting it from the guard.

Scope is deliberately conservative: only targets that start with
``strands_robots.`` are checked. Unqualified roles (``:func:`reset```) and roles
into third-party packages resolve against Sphinx's current-module context or an
intersphinx inventory that is not available here, so flagging them would produce
false positives without catching the rot this guard exists to prevent.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import strands_robots

_PKG_ROOT = Path(strands_robots.__file__).resolve().parent

# Sphinx cross-reference roles naming a dotted Python target, optionally with a
# leading ``~`` (display-shortening) tilde. Whitespace is admitted INSIDE the
# target so a role whose long dotted path was wrapped over a line break is
# still extracted and checked: a pattern that stopped at the newline skipped
# such a role outright, which exempted it from this guard entirely.
_ROLE_RE = re.compile(r":(?:mod|class|func|meth|attr|data|obj|exc):`~?([A-Za-z_][\w.\s]*)`")


def _resolves(target: str) -> bool:
    """True if ``target`` names a real ``strands_robots`` module or attribute.

    Imports the longest importable module prefix, then walks the remaining
    dotted components as attributes (so ``pkg.mod.Class.method`` resolves).
    """
    parts = target.split(".")
    module = None
    consumed = 0
    for i in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:i]))
            consumed = i
            break
        except Exception:
            continue
    if module is None:
        return False
    obj = module
    for attr in parts[consumed:]:
        if not hasattr(obj, attr):
            return False
        obj = getattr(obj, attr)
    return True


def _offending_roles_in(doc: str) -> list[str]:
    """Report every ``strands_robots.*`` role target in ``doc`` a reader cannot follow.

    Two distinct ways a target fails to name an object, both reported here:

    * It is not a contiguous dotted path, because the role was wrapped over a
      line break. The token as written carries a newline and the following
      indentation, so it does not import, and no renderer reassembles it - the
      whitespace collapses to a space, which does not import either. Rewrap the
      line so the dotted path stays intact.
    * It is a contiguous path that does not resolve - the dead pointer this
      guard was written for.

    A wrapped target is reported even when joining its segments WOULD resolve:
    the path a reader copies out of the source is the one that has to work.

    Args:
        doc: One raw docstring.

    Returns:
        One entry per offending target, each naming the target and, for a
        wrapped one, the contiguous path it was meant to be.
    """
    offenders: list[str] = []
    for target in _ROLE_RE.findall(doc):
        contiguous = "".join(target.split())
        if not contiguous.startswith("strands_robots."):
            continue
        if target != contiguous:
            offenders.append(f"{target!r} is wrapped over a line break; write it as {contiguous!r}")
        elif not _resolves(target):
            offenders.append(target)
    return offenders


def _unresolved_xref_roles() -> dict[str, list[str]]:
    """Map ``relpath::qualname`` -> list of unfollowable ``strands_robots.*`` roles."""
    offenders: dict[str, list[str]] = {}
    for source_file in sorted(_PKG_ROOT.rglob("*.py")):
        if "__pycache__" in source_file.parts:
            continue
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            bad = _offending_roles_in(doc)
            if bad:
                qualname = getattr(node, "name", "<module>")
                rel = source_file.relative_to(_PKG_ROOT)
                offenders[f"{rel}::{qualname}"] = bad
    return offenders


def test_qualified_strands_robots_xref_roles_resolve() -> None:
    offenders = _unresolved_xref_roles()
    assert not offenders, (
        "docstring cross-reference roles must name a real importable object. "
        "Cite a registered predicate/backend by its literal name (``base_below_z``) "
        "rather than a :func:`...` role, and reference actual API objects with "
        ":mod:/:class:/:func:/:meth:. A path long enough to wrap must be moved "
        "onto its own line so it stays contiguous. Offending docstrings: " + repr(offenders)
    )


def test_guard_resolver_accepts_real_symbol_and_rejects_bogus() -> None:
    """The resolver walks module.attr chains and rejects nonexistent paths."""
    assert _resolves("strands_robots.simulation.base.SimEngine.get_observation")
    assert not _resolves("strands_robots.simulation.predicates.base_below_z")


# A real object, and a path that has never existed, used below to separate
# "the guard cannot see this role" from "the target does not resolve".
_REAL = "strands_robots.simulation.base.SimEngine.get_observation"
_BOGUS = "strands_robots.simulation.predicates.base_below_z"


def test_the_role_pattern_extracts_a_target_wrapped_over_a_line_break() -> None:
    """A wrapped dotted path is still a role target, so the pattern must match it.

    A pattern that stops at the newline finds nothing here, which is how a
    wrapped role escapes every check below it.
    """
    doc = "See :class:`~strands_robots.policies.protomotions.motion_utils.\n    MotionPlayer` for the window."
    assert _ROLE_RE.findall(doc) == ["strands_robots.policies.protomotions.motion_utils.\n    MotionPlayer"]


def test_a_wrapped_target_is_reported_even_when_its_segments_would_resolve() -> None:
    """The path as written is what a reader copies, so a wrap is refused on its own.

    Joining the segments of this target reaches a real class; the token in the
    source does not, and that is the pointer being graded.
    """
    head, tail = _REAL.rsplit(".", 1)
    doc = f"Delegates to :meth:`~{head}.\n    {tail}` on every backend."

    offenders = _offending_roles_in(doc)

    assert len(offenders) == 1, offenders
    assert "wrapped over a line break" in offenders[0]
    assert _REAL in offenders[0], "the remedy must spell out the contiguous path"
    assert _resolves(_REAL), "premise: joining the wrapped segments reaches a real object"


def test_a_wrapped_target_that_does_not_resolve_is_reported() -> None:
    """A wrap must not hide a dead pointer - the case the widened pattern recovers."""
    head, tail = _BOGUS.rsplit(".", 1)
    doc = f"Registered as :func:`~{head}.\n    {tail}`."

    assert _offending_roles_in(doc), "a wrapped role naming nothing must not pass unseen"


def test_a_contiguous_resolving_target_is_accepted() -> None:
    """Control: the ordinary, correct form stays silent."""
    assert _offending_roles_in(f"Delegates to :meth:`~{_REAL}`.") == []


def test_a_contiguous_target_that_does_not_resolve_is_reported() -> None:
    """Control: the original dead-pointer check is unchanged by the widening."""
    assert _offending_roles_in(f"See :func:`~{_BOGUS}`.") == [_BOGUS]


def test_a_wrapped_target_outside_the_package_is_left_alone() -> None:
    """Control: third-party targets stay out of scope, wrapped or not."""
    assert _offending_roles_in("See :class:`~mujoco.\n    MjSpec` for the spec API.") == []
