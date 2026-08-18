"""Guard: fully-qualified ``strands_robots`` Sphinx cross-references must resolve.

A docstring role naming ``strands_robots.simulation.predicates.base_below_z``
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

Every tree a reader reads is graded: the shipped package, ``tests/`` and
``tests_integ/``. A role is a pointer a human follows, and the roles in a test
module's docstring are the first thing a maintainer working on that subsystem
reads. Scoping the scan to the package left the test trees' several hundred
qualified targets ungraded, and a module that had been folded into a package
went on being cited at its old top-level path long after it stopped existing.

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

Both spellings resolve a member through that one rule. A qualified target and a
short-form target that name the SAME member have to get the same verdict, or the
guard reports a dead pointer for an attribute that exists and the remedy is to
delete a correct cross-reference.

Still out of scope, because these have no decidable target here: a bare
unqualified role (``:func:`reset```), a short-form role whose head names nothing
the package defines, and roles into third-party packages - all of which resolve
against Sphinx's current-module context or an intersphinx inventory that is not
available, so flagging them would produce false positives.

One further target is undecidable for a reason that lives in the environment
rather than in the docstring: a cited module whose own optional dependency is not
installed. The module exists in the tree, so the citation is sound, but nothing
here can confirm the object it names, and reporting it accuses a correct pointer
whose only remedy would be deletion. Such a target is skipped - the rule the
short-form half has always applied to the modules it indexes.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

import strands_robots

_PKG_ROOT = Path(strands_robots.__file__).resolve().parent
_REPO_ROOT = _PKG_ROOT.parent

# Sphinx cross-reference roles naming a dotted Python target, optionally with a
# leading ``~`` (display-shortening) tilde. Whitespace is admitted INSIDE the
# target so a role whose long dotted path was wrapped over a line break is
# still extracted and checked: a pattern that stopped at the newline skipped
# such a role outright, which exempted it from this guard entirely.
_ROLE_RE = re.compile(r":(?:mod|class|func|meth|attr|data|obj|exc):`~?([A-Za-z_][\w.\s]*)`")


def _absent_dependency(exc: ImportError) -> str | None:
    """The absent third-party module ``exc`` names, if that is what it reports.

    Both causes of a failed import here raise the same exception type and need
    opposite verdicts, and the name it carries is what separates them. A
    ``strands_robots`` path means no such module exists - the rot being graded. A
    name outside the package means the cited module does exist and could not be
    imported because an extra is not installed, which is a fact about this
    environment and not about the docstring.
    """
    name = getattr(exc, "name", None)
    if not name or name == "strands_robots" or name.startswith("strands_robots."):
        return None
    return name


def _resolves(target: str) -> bool | None:
    """Grade one fully-qualified ``target`` against the real API.

    Imports the longest importable module prefix, then walks the remaining
    dotted components as members (so ``pkg.mod.Class.method`` resolves), using
    the same :func:`_has_member` rule the short-form half applies. A bare
    ``hasattr`` walk here would report a dataclass field with no class-level
    default, and an attribute only ever assigned in ``__init__``, as a dead
    pointer - while the identical member cited in the short form resolved.

    The member walk is guarded for the same reason the imports are: a package
    that imports its submodules from a module-level ``__getattr__`` raises
    ``ModuleNotFoundError`` from inside ``hasattr``, which swallows only
    ``AttributeError``. Left to propagate, one absent extra ends the sweep and
    nothing is graded at all.

    Returns:
        ``True`` when every component resolves, ``False`` when the path names
        nothing, and ``None`` when an optional dependency is absent, so this
        environment cannot decide the target either way. The tri-state is the one
        :func:`_short_form_resolves` already returns for an undecidable target.
    """
    parts = target.split(".")
    module = None
    consumed = 0
    for i in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:i]))
        except ImportError as exc:
            if _absent_dependency(exc):
                return None
            continue
        except Exception:
            continue
        consumed = i
        break
    if module is None:
        return False
    obj = module
    for attr in parts[consumed:]:
        try:
            if not _has_member(obj, attr):
                return False
            obj = getattr(obj, attr, obj)
        except ImportError as exc:
            if _absent_dependency(exc):
                return None
            raise
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
        elif _resolves(target) is False:
            offenders.append(target)
    return offenders


def _graded_source_files() -> list[Path]:
    """Every Python file whose docstrings carry pointers a reader follows.

    The shipped package plus both test trees. A missing tree is skipped rather
    than assumed, so the sweep still runs in a checkout that ships only one.
    """
    files: list[Path] = []
    for root in (_PKG_ROOT, _REPO_ROOT / "tests", _REPO_ROOT / "tests_integ"):
        if not root.is_dir():
            continue
        files.extend(f for f in sorted(root.rglob("*.py")) if "__pycache__" not in f.parts)
    return files


def _unresolved_xref_roles() -> tuple[dict[str, list[str]], int]:
    """Report unfollowable qualified roles, plus how many were graded at all.

    Returns:
        ``({relpath::qualname: [offending target, ...]}, graded_count)``.
    """
    offenders: dict[str, list[str]] = {}
    graded = 0
    for source_file in _graded_source_files():
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            graded += sum(1 for t in _ROLE_RE.findall(doc) if "".join(t.split()).startswith("strands_robots."))
            bad = _offending_roles_in(doc)
            if bad:
                qualname = getattr(node, "name", "<module>")
                rel = source_file.relative_to(_REPO_ROOT)
                offenders[f"{rel}::{qualname}"] = bad
    return offenders, graded


# The qualified spelling carries several hundred targets across the package and
# both test trees, so a sweep that grades only a handful has stopped reaching
# them - a reformat or a scope change must fail loudly here rather than report a
# clean tree it never inspected.
_MINIMUM_GRADED_QUALIFIED_ROLES = 600


def test_qualified_strands_robots_xref_roles_resolve() -> None:
    offenders, graded = _unresolved_xref_roles()
    assert graded >= _MINIMUM_GRADED_QUALIFIED_ROLES, (
        f"only {graded} qualified roles were graded; a clean result would prove nothing"
    )
    assert not offenders, (
        "docstring cross-reference roles must name a real importable object. "
        "Cite a registered predicate/backend by its literal name (``base_below_z``) "
        "rather than a :func:`...` role, and reference actual API objects with "
        ":mod:/:class:/:func:/:meth:. A path long enough to wrap must be moved "
        "onto its own line so it stays contiguous. Offending docstrings: " + repr(offenders)
    )


def test_guard_resolver_accepts_real_symbol_and_rejects_bogus() -> None:
    """The resolver walks module.attr chains and rejects nonexistent paths.

    Asserted against the exact verdict rather than its truthiness: the resolver
    also returns ``None`` for a target it cannot decide, and ``not None`` is true,
    so a truthiness check would read an undecided target as a rejected one.
    """
    assert _resolves("strands_robots.simulation.base.SimEngine.get_observation") is True
    assert _resolves("strands_robots.simulation.predicates.base_below_z") is False


# A real object, and a path that has never existed, used below to separate
# "the guard cannot see this role" from "the target does not resolve".
_REAL = "strands_robots.simulation.base.SimEngine.get_observation"
_BOGUS = "strands_robots.simulation.predicates.base_below_z"

# The module prefix of ``_REAL``, whose import is staged as failing below.
_REAL_MODULE = "strands_robots.simulation.base"

# Named so a traceback carrying it cannot be mistaken for a real absent package.
_ABSENT_EXTRA = "an_extra_this_environment_does_not_have"


def _import_module_lacking(prefix: str):
    """Stand in for ``import_module``, with ``prefix`` needing an absent extra.

    Which extras are actually missing varies with the developer's install and is
    nothing in a full CI environment, so the failure is staged rather than
    sampled: the exception a missing extra really raises, named on a module this
    tree really does ship. Every other name is imported for real, so the walk
    under test is the production one.
    """
    real = importlib.import_module

    def _import(name: str, package: str | None = None) -> ModuleType:
        if name == prefix:
            raise ModuleNotFoundError(f"No module named {_ABSENT_EXTRA!r}", name=_ABSENT_EXTRA)
        return real(name, package) if package else real(name)

    return _import


def test_a_target_whose_module_needs_an_absent_extra_is_not_a_dead_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unimportable module is an unknown here, and an unknown is not an offender.

    The resolver falls back to the longest prefix that does import, so a module
    waiting on an extra was looked for as a *member* of its own parent package,
    was not found, and the citation was reported as rot. Measured on a
    ``[dev]``-only install of this tree: 39 such targets, every one of them
    correct, and the only remedy the report offers is deleting them.
    """
    assert _resolves(_REAL) is True, "premise: the target resolves when the module imports"

    monkeypatch.setattr(importlib, "import_module", _import_module_lacking(_REAL_MODULE))

    assert _resolves(_REAL) is None
    assert _offending_roles_in(f"Delegates to :meth:`~{_REAL}`.") == []


def test_a_lazy_import_failure_in_the_member_walk_does_not_end_the_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A module-level ``__getattr__`` raising must not escape through ``hasattr``.

    ``strands_robots.tools`` imports its submodules on attribute access, so
    grading ``strands_robots.tools.pose_tool`` without the ``serial`` extra raised
    out of the sweep: the guard reported no docstring at all, and the traceback
    named a dependency rather than a cross-reference. Four targets in this tree
    reach it, which is the worse half of the same cause - an over-report is at
    least legible as a list of docstrings.
    """
    lazy = ModuleType(_REAL_MODULE.rsplit(".", 1)[0])

    def _raise_absent(attr: str) -> object:
        raise ModuleNotFoundError(f"No module named {_ABSENT_EXTRA!r}", name=_ABSENT_EXTRA)

    lazy.__getattr__ = _raise_absent  # type: ignore[method-assign]
    with pytest.raises(ModuleNotFoundError):
        hasattr(lazy, "base")  # premise: hasattr swallows only AttributeError

    def _import(name: str, package: str | None = None) -> ModuleType:
        if name == lazy.__name__:
            return lazy
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(importlib, "import_module", _import)

    assert _resolves(_REAL_MODULE) is None
    assert _offending_roles_in(f"See :mod:`~{_REAL_MODULE}`.") == []


def test_a_module_the_package_does_not_have_is_still_a_dead_pointer() -> None:
    """Control: an absent extra and absent rot must not become the same verdict.

    Both raise ``ModuleNotFoundError`` and only the name it carries separates
    them, so this is the assertion that stops the skip from laundering real rot
    into an unknown. ``strands_robots.mesh_session`` is the module folded into the
    ``mesh`` package whose citations #2427 found still being followed.
    """
    assert _resolves("strands_robots.mesh_session.get_session") is False
    assert _resolves(_BOGUS) is False

    assert _absent_dependency(ModuleNotFoundError("x", name="strands_robots.mesh_session")) is None
    assert _absent_dependency(ModuleNotFoundError("x", name="strands_robots")) is None
    assert _absent_dependency(ModuleNotFoundError("x", name="serial")) == "serial"


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


def test_the_sweep_reaches_the_test_trees() -> None:
    """Non-vacuity: the scan must actually open files outside the package.

    A role is a pointer a reader follows, and a test module's docstring is where
    a maintainer starts. A sweep that silently covered only the package would
    report a clean tree while the majority of docstrings went unread.
    """
    scanned = {str(f.relative_to(_REPO_ROOT)).split("/", 1)[0] for f in _graded_source_files()}

    assert {"strands_robots", "tests"} <= scanned, f"graded trees: {sorted(scanned)}"


@pytest.mark.parametrize(
    ("qualified", "short_form"),
    [
        # An attribute that exists only because __init__ assigns it.
        ("strands_robots.inference.server.PolicyServer.port", "PolicyServer.port"),
        ("strands_robots.rendering.HybridCompositor.default_width", "HybridCompositor.default_width"),
    ],
)
def test_both_spellings_agree_that_a_member_is_a_member(qualified: str, short_form: str) -> None:
    """The same member must resolve whichever way a role names it.

    Two resolvers grading one concept is how a guard starts contradicting
    itself: the short form accepted ``self``-assigned attributes while the
    qualified form, walking with a bare ``hasattr``, called the identical
    member a dead pointer. The only remedy such a report offers is deleting a
    cross-reference that was correct.
    """
    head, member = qualified.rsplit(".", 1)
    module = importlib.import_module(head.rsplit(".", 1)[0])
    owner = getattr(module, head.rsplit(".", 1)[1])

    assert not hasattr(owner, member), "premise: no class-level attribute exists"
    assert _short_form_resolves(module, {}, short_form) is True, "premise: the short form resolves it"
    assert _resolves(qualified), f"the qualified spelling of {short_form} must resolve too"


def test_a_qualified_target_that_names_nothing_is_still_reported() -> None:
    """Control: sharing the permissive member rule must not blunt the guard.

    ``_has_member`` widens what counts as evidence, not what counts as a claim,
    so a path naming no object at all stays an offender.
    """
    assert not _resolves(_BOGUS)
    assert _offending_roles_in(f"See :func:`~{_BOGUS}`.") == [_BOGUS]
    assert not _resolves("strands_robots.simulation.base.SimEngine.no_such_method")
