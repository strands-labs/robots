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

Two spellings of the same promise are graded, because a target only has to
resolve - it does not have to be fully qualified to be checkable:

* A fully-qualified target (``strands_robots.a.b.C.d``) resolves on its own.
* A SHORT-FORM target (``C.d``) resolves once its head is pinned. The head is
  looked up in the citing module first, and then, if the module does not import
  it, against a package-wide index of class names - used only when exactly one
  class in the package answers to that name, so an ambiguous head is skipped
  rather than guessed. This is the spelling most method cross-references
  actually use, and the earlier scope excluded it, which left the rot it exists
  to prevent free to accumulate in the majority form.

A member is resolved permissively: an attribute, a dataclass field, an annotated
class attribute anywhere in the MRO, a ``__slots__`` entry, or a name the class
assigns to ``self``. Being strict about what counts as a *claim* and permissive
about what counts as *evidence* means this guard can only ever under-report; a
class-level ``hasattr`` alone reads a dataclass field with no default, and an
attribute only ever assigned in ``__init__``, as missing.

Still out of scope, because these have no decidable target here: a bare
unqualified role (``:func:`reset```), a short-form role whose head names nothing
the package defines, and roles into third-party packages - all of which resolve
against Sphinx's current-module context or an intersphinx inventory that is not
available, so flagging them would produce false positives.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

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


# --------------------------------------------------------------------------- #
# Short-form roles (``C.member``), the spelling most method cross-references use
# --------------------------------------------------------------------------- #


def _package_modules() -> dict[str, ModuleType]:
    """Map dotted name -> imported module for every importable package module.

    A module whose optional dependency is absent simply does not appear, so this
    guard grades the roles it can reach and never turns a missing extra into a
    dead pointer.
    """
    modules: dict[str, ModuleType] = {}
    for source_file in sorted(_PKG_ROOT.rglob("*.py")):
        if "__pycache__" in source_file.parts:
            continue
        rel = source_file.relative_to(_PKG_ROOT.parent).with_suffix("")
        try:
            modules[".".join(rel.parts)] = importlib.import_module(".".join(rel.parts))
        except Exception:
            continue
    return modules


def _class_index(modules: dict[str, ModuleType]) -> dict[str, set[type]]:
    """Map a bare class name -> every package class answering to it.

    A name with more than one class is ambiguous and is skipped by the caller:
    guessing which one a role meant would invent the very pointer being graded.
    """
    index: dict[str, set[type]] = {}
    for module in modules.values():
        for name, obj in vars(module).items():
            if inspect.isclass(obj) and getattr(obj, "__module__", "").startswith("strands_robots"):
                index.setdefault(name, set()).add(obj)
    return index


def _self_assigned_names(cls: type) -> set[str]:
    """Names ``cls`` or a package base assigns to ``self``.

    An attribute that only ever exists because ``__init__`` assigns it is a real
    member a role may cite, and no class-level lookup can see it.
    """
    names: set[str] = set()
    for klass in inspect.getmro(cls):
        if not getattr(klass, "__module__", "").startswith("strands_robots"):
            continue
        try:
            source = inspect.getsource(klass)
        except (OSError, TypeError):
            continue
        names |= set(re.findall(r"self\.([A-Za-z_]\w*)\s*(?::[^=\n]+)?=", source))
    return names


def _has_member(owner: object, attr: str) -> bool:
    """True if ``attr`` is a member of ``owner`` under any declaration style."""
    if hasattr(owner, attr):
        return True
    if attr in getattr(owner, "__dataclass_fields__", {}):
        return True
    if inspect.isclass(owner):
        for klass in inspect.getmro(owner):
            if attr in getattr(klass, "__annotations__", {}):
                return True
            if attr in tuple(getattr(klass, "__slots__", ())):
                return True
        if attr in _self_assigned_names(owner):
            return True
    return False


def _short_form_resolves(citing_module: ModuleType, index: dict[str, set[type]], target: str) -> bool | None:
    """Grade one short-form ``target`` cited from ``citing_module``.

    Returns:
        ``True`` when every component resolves, ``False`` when one does not, and
        ``None`` when the target has no decidable head here (a bare name, an
        ambiguous class name, or a head the package does not define) and is
        therefore out of scope rather than an offender.
    """
    head, *members = target.split(".")
    if not members:
        return None
    owner: object | None = getattr(citing_module, head, None)
    if owner is None:
        candidates = index.get(head, set())
        if len(candidates) != 1:
            return None
        owner = next(iter(candidates))
    for attr in members:
        if not _has_member(owner, attr):
            return False
        owner = getattr(owner, attr, owner)
    return True


def _short_form_targets(doc: str) -> list[str]:
    """Every contiguous short-form (non ``strands_robots.``-qualified) dotted target."""
    targets = []
    for raw in _ROLE_RE.findall(doc):
        target = " ".join(raw.split())
        if target != raw or target.startswith("strands_robots.") or "." not in target:
            # A wrapped target and a qualified one are both already graded by
            # the checks above; a bare name has no decidable target.
            continue
        targets.append(target)
    return targets


def _graded_short_form_roles() -> tuple[dict[str, list[str]], int]:
    """Report unresolvable short-form roles, plus how many were graded at all."""
    modules = _package_modules()
    index = _class_index(modules)
    offenders: dict[str, list[str]] = {}
    graded = 0
    for dotted, module in modules.items():
        source_file = getattr(module, "__file__", None)
        if source_file is None:
            continue
        tree = ast.parse(Path(source_file).read_text(encoding="utf-8"), filename=source_file)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            for target in _short_form_targets(doc):
                verdict = _short_form_resolves(module, index, target)
                if verdict is None:
                    continue
                graded += 1
                if not verdict:
                    qualname = getattr(node, "name", "<module>")
                    offenders.setdefault(f"{dotted}::{qualname}", []).append(target)
    return offenders, graded


# The short-form spelling carries the bulk of this package's method
# cross-references, so a sweep that grades only a handful has stopped reaching
# them - a reformat or an import change must fail loudly here rather than report
# a clean tree it never inspected.
_MINIMUM_GRADED_SHORT_FORM_ROLES = 100


def test_short_form_xref_roles_resolve() -> None:
    offenders, graded = _graded_short_form_roles()
    assert graded >= _MINIMUM_GRADED_SHORT_FORM_ROLES, (
        f"only {graded} short-form roles were graded; a clean result would prove nothing"
    )
    assert not offenders, (
        "a short-form docstring cross-reference must name a real member of its head. "
        "Cite the member that exists - a private spelling of a public method is a dead "
        "pointer twice over - or drop the role. Offending docstrings: " + repr(offenders)
    )


def test_the_sweep_reports_a_planted_short_form_dead_pointer() -> None:
    """Non-vacuity: a clean sweep must mean the docstrings are right.

    Grades a target against a real package class so the only thing wrong with it
    is the member name, which is exactly the shape the sweep exists to catch.
    """
    from strands_robots.simulation.base import SimEngine

    module = importlib.import_module("strands_robots.simulation.base")
    index = _class_index({"strands_robots.simulation.base": module})

    assert _short_form_resolves(module, index, "SimEngine.get_observation") is True
    assert _short_form_resolves(module, index, "SimEngine.no_such_method") is False
    assert not hasattr(SimEngine, "no_such_method"), "premise: the planted member does not exist"


def test_a_dataclass_field_is_a_member() -> None:
    """Control: a field with no class-level default must not read as missing.

    ``hasattr(cls, field)`` is False for such a field, so a class-level lookup
    alone would report every ``:attr:`Config.field``` role in the package.
    """
    module = ModuleType("stub_dataclass_module")

    @dataclass
    class Embodiment:
        camera_keys: list[str]

    module.Embodiment = Embodiment  # type: ignore[attr-defined]

    assert not hasattr(Embodiment, "camera_keys"), "premise: no class-level attribute exists"
    assert _short_form_resolves(module, {}, "Embodiment.camera_keys") is True


def test_an_attribute_only_assigned_in_init_is_a_member() -> None:
    """Control: ``self.port = port`` is a real member a role may cite."""
    module = importlib.import_module("strands_robots.inference.server")
    from strands_robots.inference.server import PolicyServer

    assert not hasattr(PolicyServer, "port"), "premise: no class-level attribute exists"
    assert _short_form_resolves(module, {}, "PolicyServer.port") is True


def test_an_ambiguous_or_unknown_head_is_out_of_scope() -> None:
    """Control: an undecidable head is skipped, never reported.

    Reporting one would flag every role into a third-party package, which is the
    false-positive class the qualified-only scope was written to avoid.
    """
    module = ModuleType("stub_empty_module")

    class First:
        pass

    class Second:
        pass

    assert _short_form_resolves(module, {}, "SomeThirdPartyThing.method") is None
    assert _short_form_resolves(module, {"Shared": {First, Second}}, "Shared.method") is None
    assert _short_form_resolves(module, {}, "reset") is None, "a bare name has no decidable target"
