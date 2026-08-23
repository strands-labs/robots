"""A ``Performed checks:`` census names the checks the function performs.

Two functions in :mod:`strands_robots.mesh.security` open with a
``Performed checks:`` enumeration. Both are validators on a wire boundary, and
for a reader the census *is* the contract: a publisher author decides what to
put on the wire from it, and an operator tuning the teleop safety envelope
reads it to learn what the bound is. Nothing graded it.

Two properties are checked, both derived from the function's own body so that a
census added later is held to them with no list to update:

* **A census may only cite a domain the function reads.** Citing a module
  constant the body never loads names a value that is not the one enforced. That
  is how :func:`~strands_robots.mesh.security.validate_input_frame` came to cite
  ``MAX_INPUT_VALUE_ABS`` -- a snapshot frozen at import -- while applying
  ``_input_value_abs()``, which re-reads the env var per call so an operator can
  narrow the envelope without a restart. The two agree until the var is set
  after import, at which point the census reports one number and another
  refuses the frame.

* **Every type a positive ``isinstance`` gate refuses is named.** A gate of the
  shape ``if isinstance(v, T): raise`` says "T is refused", so a census that
  does not name ``T`` describes a wider admission than the code performs.

Both rules are permissive by construction -- they can only under-report. A
constant named in prose *outside* the bullets is not graded (the bullets are the
enumeration; the surrounding paragraphs are commentary), and no wording is
required beyond the type name itself.

``validate_command``'s census satisfies both rules unchanged, so it is the
control: the properties are this module's own convention rather than a new one.
"""

from __future__ import annotations

import ast
import inspect
import math
import re
from typing import Any

import pytest

from strands_robots.mesh import security

#: Fewest ``Performed checks:`` censuses the scan must reach. Below this the
#: rules would hold vacuously over an empty population.
_MINIMUM_CENSUSES = 2

_CONSTANT_RE = re.compile(r"[`\s(]([A-Z][A-Z0-9_]{3,})[`\s,.)]")
_DATA_ROLE_RE = re.compile(r":data:`([A-Z][A-Z0-9_]*)`")


def _census_bullets(doc: str | None) -> str | None:
    """Return the bullet block of *doc*'s census, or ``None`` if it has none.

    The census is its bullets. *doc* arrives from :func:`ast.get_docstring`,
    which dedents, so a bullet sits at column 0 and its continuations and
    sub-bullets are indented. Extraction stops at the first non-empty line
    back at column 0, which is where a clarifying paragraph after the
    enumeration begins -- deliberately out of scope, so prose may name a
    constant the bullets must not claim.
    """
    if not doc or "Performed checks:" not in doc:
        return None
    kept: list[str] = []
    started = False
    for line in doc.split("Performed checks:", 1)[1].splitlines():
        stripped = line.strip()
        if stripped.startswith("* "):
            started = True
            kept.append(line)
        elif started and stripped and line[:1].isspace():
            kept.append(line)
        elif started and stripped:
            break
    return "\n".join(kept)


def _module_constants(tree: ast.Module) -> set[str]:
    """Module-level upper-case names: the domains a census can cite."""
    found: set[str] = set()
    for node in tree.body:
        targets = [node.target] if isinstance(node, ast.AnnAssign) else getattr(node, "targets", [])
        for target in targets:
            if isinstance(target, ast.Name) and target.id.lstrip("_").isupper():
                found.add(target.id)
    return found


def _loaded_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    return {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _positively_refused_types(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Types refused by an ``if isinstance(v, T): raise`` gate inside *fn*."""
    found: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Call) and isinstance(test.func, ast.Name) and test.func.id == "isinstance"):
            continue
        if len(test.args) < 2:
            continue
        # The branch must *be* the refusal: a raise as a direct statement of the
        # body. A raise nested inside a try/if is a check performed within the
        # accept path (``if isinstance(v, dict): try: json.dumps(v) ...``), which
        # says nothing about whether T is admitted.
        if not any(isinstance(stmt, ast.Raise) for stmt in node.body):
            continue
        declared = test.args[1]
        elements = declared.elts if isinstance(declared, ast.Tuple) else [declared]
        for element in elements:
            found.add(ast.unparse(element))
    return found


def _censuses() -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, str, set[str]]]:
    """Every census in :mod:`strands_robots.mesh.security`, with its bullets."""
    source = inspect.getsource(security)
    tree = ast.parse(source)
    constants = _module_constants(tree)
    found: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, str, set[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bullets = _census_bullets(ast.get_docstring(node))
        if bullets is None:
            continue
        found.append((node.name, node, bullets, constants))
    return found


def _cited_constants(bullets: str, constants: set[str]) -> set[str]:
    cited = set(_CONSTANT_RE.findall(bullets)) | set(_DATA_ROLE_RE.findall(bullets))
    return cited & constants


def _refused(value: Any) -> str | None:
    """Drive the real validator with *value*; return its refusal, or ``None``."""
    try:
        security.validate_input_frame({"j1": value})
    except security.ValidationError as refusal:
        return str(refusal)
    return None


class TestThePerformedChecksCensusNamesWhatTheCodeDoes:
    """The census and the body must agree about the checks and the bounds."""

    def test_a_census_only_cites_a_domain_the_function_reads(self) -> None:
        """A cited constant the body never loads is not the value enforced."""
        offenders: dict[str, list[str]] = {}
        for name, node, bullets, constants in _censuses():
            loaded = _loaded_names(node)
            unread = sorted(c for c in _cited_constants(bullets, constants) if c not in loaded)
            if unread:
                offenders[name] = unread
        assert not offenders, (
            f"a Performed-checks census cites a domain its own body never reads: {offenders}. "
            "The value a reader is sent to is then not the value that refuses; cite the "
            "resolver or the constant the function actually loads."
        )

    def test_the_cited_envelope_is_the_one_that_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A census naming an import-time snapshot reports the wrong bound.

        Measured rather than argued: the operator-facing env var is narrowed
        after import, which is the whole reason the hot path re-reads it.
        """
        monkeypatch.setenv("STRANDS_MESH_INPUT_VALUE_ABS", "10.0")
        snapshot = security.MAX_INPUT_VALUE_ABS
        enforced = security._input_value_abs()
        assert enforced != snapshot, (
            "premise: with the env var narrowed after import the resolver must "
            f"disagree with the snapshot (both {enforced})"
        )
        probe = (snapshot + enforced) / 2.0
        refusal = _refused(probe)
        assert refusal is not None, f"premise: {probe} must exceed the enforced envelope {enforced}"

        for name, node, bullets, constants in _censuses():
            if name != "validate_input_frame":
                continue
            cited = _cited_constants(bullets, constants)
            assert "MAX_INPUT_VALUE_ABS" not in cited, (
                f"the census cites MAX_INPUT_VALUE_ABS ({snapshot}) as the envelope, but the "
                f"envelope that refuses is _input_value_abs() ({enforced}): a value of {probe} "
                f"is inside the cited bound and was refused with {refusal!r}. An operator who "
                "narrows the teleop envelope reads the snapshot and is told the wrong number."
            )

    def test_every_type_the_function_refuses_is_named_in_its_census(self) -> None:
        """``if isinstance(v, T): raise`` means T is refused, so T is named."""
        offenders: dict[str, list[str]] = {}
        for name, node, bullets, _constants in _censuses():
            refused = _positively_refused_types(node)
            missing = sorted(t for t in refused if not re.search(rf"\b{re.escape(t)}\b", bullets))
            if missing:
                offenders[name] = missing
        assert not offenders, (
            f"a Performed-checks census does not name a type its own body refuses outright: "
            f"{offenders}. The census then describes a wider admission than the function "
            "performs, and a caller reads the refused value as accepted."
        )

    def test_a_value_the_census_admits_is_not_refused(self) -> None:
        """The census's own value rule must not admit what the code refuses.

        A ``bool`` is coercible to ``float``, finite, and inside the envelope,
        so a census whose value rule is stated as coercibility admits it. The
        validator refuses it -- deliberately, because ``bool`` is an ``int``
        subclass and ``True`` would otherwise reach an actuator as ``1.0``.
        """
        bullets = next(b for n, _f, b, _c in _censuses() if n == "validate_input_frame")
        census_names_bool = bool(re.search(r"\bbool\b", bullets))
        refusal = _refused(True)
        assert refusal is not None, "premise: the validator must refuse a bool"
        assert census_names_bool, (
            "the census states its value rule without naming bool, so a reader concludes "
            f"True is admitted (it is coercible to float, finite and in range); the validator "
            f"refused it with {refusal!r}. A publisher built from the census sends a bool as a "
            "1.0 actuator command and its stream is refused."
        )


class TestTheCensusRulesHoldOnTheSiblingUnchanged:
    """``validate_command`` satisfies both rules as shipped: the control."""

    def test_the_sibling_census_cites_only_domains_it_reads(self) -> None:
        name, node, bullets, constants = next(c for c in _censuses() if c[0] == "validate_command")
        loaded = _loaded_names(node)
        cited = _cited_constants(bullets, constants)
        assert cited, "premise: the sibling census must cite at least one domain"
        assert not [c for c in cited if c not in loaded], f"{name} cites {sorted(cited)}"

    def test_the_sibling_census_names_every_type_it_refuses(self) -> None:
        name, node, bullets, _constants = next(c for c in _censuses() if c[0] == "validate_command")
        refused = _positively_refused_types(node)
        assert not [t for t in refused if not re.search(rf"\b{re.escape(t)}\b", bullets)], name


class TestTheScanIsNotVacuous:
    """A rule over an empty population, or a mis-parsed census, proves nothing."""

    def test_the_scan_reaches_every_census_in_the_module(self) -> None:
        names = sorted(name for name, _f, _b, _c in _censuses())
        assert len(names) >= _MINIMUM_CENSUSES, f"only {names} scanned"
        assert "validate_input_frame" in names
        assert "validate_command" in names

    def test_every_scanned_census_has_bullets(self) -> None:
        for name, _node, bullets, _constants in _censuses():
            assert bullets.strip().startswith("* "), f"{name} census parsed as {bullets!r}"

    def test_a_gate_that_does_not_raise_is_not_a_refusal(self) -> None:
        """An ``isinstance`` branch with no raise says nothing about admission."""
        fn = ast.parse("def f(v):\n    if isinstance(v, bytes):\n        v = v.decode()\n").body[0]
        assert isinstance(fn, ast.FunctionDef)
        assert _positively_refused_types(fn) == set()

    def test_a_nested_raise_inside_an_accept_branch_is_not_a_refusal(self) -> None:
        """``if isinstance(v, dict): try: ... except: raise`` admits the dict.

        The raise bounds something *about* an accepted value. Counting it as a
        refusal of ``dict`` would make ``validate_command`` -- which takes that
        exact shape for ``world_update`` -- look like it refuses the type its
        census documents as accepted.
        """
        fn = ast.parse(
            "def f(v):\n"
            "    if isinstance(v, dict):\n"
            "        try:\n"
            "            json.dumps(v)\n"
            "        except TypeError as exc:\n"
            "            raise ValueError('nope') from exc\n"
        ).body[0]
        assert isinstance(fn, ast.FunctionDef)
        assert _positively_refused_types(fn) == set()


class TestTheRulesDoNotOverReach:
    """The graded region is the bullets, and no wording beyond the type name."""

    def test_a_constant_named_outside_the_bullets_is_not_graded(self) -> None:
        # Shaped as ``ast.get_docstring`` delivers it: dedented, so bullets sit
        # at column 0, continuations are indented, and a following paragraph
        # returns to column 0 and ends the enumeration.
        doc = (
            "Performed checks:\n\n"
            "* At most ``MAX_KEYS`` keys, counted\n"
            "  after the frame is unwrapped.\n\n"
            "The envelope is the resolver, not the ``MAX_SNAPSHOT`` snapshot of it.\n"
        )
        bullets = _census_bullets(doc)
        assert bullets is not None
        cited = _cited_constants(bullets, {"MAX_KEYS", "MAX_SNAPSHOT"})
        assert cited == {"MAX_KEYS"}, cited

    def test_naming_the_type_is_enough(self) -> None:
        """No explanation is required -- only that the refused type appears."""
        fn = ast.parse("def f(v):\n    if isinstance(v, bool):\n        raise ValueError('no')\n").body[0]
        assert isinstance(fn, ast.FunctionDef)
        refused = _positively_refused_types(fn)
        assert refused == {"bool"}
        assert not [t for t in refused if not re.search(rf"\b{re.escape(t)}\b", "* never a ``bool``.")]

    def test_a_planted_unread_citation_is_reported(self) -> None:
        """Non-vacuity for rule A: the check must be able to fail."""
        fn = ast.parse("def f(v):\n    return v > MAX_READ\n").body[0]
        assert isinstance(fn, ast.FunctionDef)
        bullets = "    * within ``MAX_READ`` and ``MAX_UNREAD``.\n"
        cited = _cited_constants(bullets, {"MAX_READ", "MAX_UNREAD"})
        unread = sorted(c for c in cited if c not in _loaded_names(fn))
        assert unread == ["MAX_UNREAD"], unread


class TestTheValueRuleTheCensusNowStates:
    """What the corrected census claims, pinned against the validator."""

    @pytest.mark.parametrize("value", [True, False])
    def test_a_python_bool_is_refused_rather_than_coerced(self, value: bool) -> None:
        refusal = _refused(value)
        assert refusal is not None and "not bool" in refusal, refusal

    def test_a_numpy_bool_is_refused_too(self) -> None:
        numpy = pytest.importorskip("numpy")
        refusal = _refused(numpy.bool_(True))
        assert refusal is not None and "not bool" in refusal, refusal

    def test_a_numeric_looking_string_is_refused(self) -> None:
        """The type is checked, not the value's coercibility."""
        assert float("1.0") == 1.0, "premise: the string is coercible to float"
        refusal = _refused("1.0")
        assert refusal is not None and "must be numeric" in refusal, refusal

    @pytest.mark.parametrize("value", [1, 1.5, -1.5, 0])
    def test_an_int_or_float_inside_the_envelope_is_admitted(self, value: float) -> None:
        assert _refused(value) is None

    def test_a_numpy_float_is_unwrapped_and_admitted(self) -> None:
        numpy = pytest.importorskip("numpy")
        assert _refused(numpy.float32(1.5)) is None

    def test_a_non_finite_value_is_refused(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            refusal = _refused(value)
            assert refusal is not None and "finite" in refusal, (value, refusal)

    def test_an_empty_key_is_refused(self) -> None:
        with pytest.raises(security.ValidationError, match="key length out of range"):
            security.validate_input_frame({"": 1.0})

    def test_the_envelope_is_re_read_per_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The narrowed envelope refuses a value the wider one admitted."""
        assert _refused(50.0) is None, "premise: 50.0 is inside the default envelope"
        monkeypatch.setenv("STRANDS_MESH_INPUT_VALUE_ABS", "10.0")
        refusal = _refused(50.0)
        assert refusal is not None and "out of range" in refusal, refusal
