"""Grade the mesh policy-type-allowlist environment reference against the module's env surface.

``STRANDS_MESH_POLICY_TYPE_ALLOW`` is the operator-facing extension knob for
:func:`~strands_robots.mesh.security.validate_command`'s ``policy_type`` /
``policy_provider`` gate: comma-separated extras appended to the built-in
:data:`~strands_robots.mesh.security._DEFAULT_POLICY_TYPES` set. The two
vocabularies share one allowlist by design (the built-in set is the union of
:data:`~strands_robots.mesh.security._LEROBOT_POLICY_FAMILIES` and
:data:`~strands_robots.mesh.security._REGISTRY_POLICY_PROVIDERS`), so a single
env-var widening admits both a new ``policy_type`` family and a new
``policy_provider`` spelling in one step. A payload whose ``policy_type`` or
``policy_provider`` names a value not in the widened union is refused on the
mesh ``execute`` / ``start`` path with :class:`refusal_codes.POLICY_TYPE_NOT_ALLOWED`;
the refusal message names this variable as the recourse.

Until this file's companion change, ``STRANDS_MESH_POLICY_TYPE_ALLOW`` was
named on neither the README environment-variable matrix (which carried 35+
other ``STRANDS_MESH_*`` rows) nor ``docs/security.md``. The variable is
referenced 10 times inside ``mesh/security.py`` itself -- one refusal code, one
regex-charset comment, two class docstrings on the built-in list, one loader,
one cache key and two ``ValidationError`` messages that name it as the
recourse -- so an operator who reads the module source finds it, but an
operator who reads the two documentation surfaces the module points them at
(the README matrix and ``docs/security.md``) does not. The refusal message
names a variable the two operator-facing pages do not, which is the drift.

The rules below read the module's own ``os.getenv`` / ``os.environ`` literals
so the graded population tracks the code rather than a list kept beside it. A
future ``STRANDS_MESH_POLICY_TYPE_ALLOW_*`` sibling (a per-fleet override, a
scoped grant) is held to the same rule the hour it lands. Five properties,
plus one keep-the-derivation-honest premise and three behavioural pins:

- **Every policy-type-allowlist variable the module reads has a README matrix
  row.** This is what makes the matrix a single index for the family; a
  reader scanning the matrix for the widening knob should find it.
- **Every one of them is named on the security page.** The prose surface that
  describes the posture -- and this variable's whole point is a security
  posture (the allowlist is the gate ``validate_command`` enforces) -- must
  name the variables that shape it, or the posture is unconfigurable.
- **The security page names the shared-allowlist invariant.** The single
  most surprising property is that widening admits both vocabularies at once.
  An operator adding a new ``policy_provider`` needs to know they are also
  admitting the same value as a ``policy_type`` (and vice versa), because the
  charset is disjoint from neither the LeRobot family names nor the registry
  provider spellings.
- **The security page names the charset rule.** ``^[a-z][a-z0-9_]*$`` on each
  entry is a refusal condition, so an operator whose entry silently drops
  needs the rule to be discoverable from the documentation rather than by
  reading the module.
- **The security page warns against using the variable to route around a
  registry omission.** ``_REGISTRY_POLICY_PROVIDERS`` and
  ``registry/policies.json`` must stay in sync (a guard test in the tree
  enforces the bijection); this variable widens the *union* of the two, so
  using it to admit a missing registry spelling is precisely the anti-pattern
  the sync-guard exists to catch, and the documented warning is what steers
  operators away from patching the wrong side.
- **The derivation is not empty.** If the scan ever goes blind (module moved,
  literal inlined into a helper, a refactor breaks the AST walk) every rule
  above would pass vacuously; a positive premise refuses that state.

The behavioural facts the documentation exists to make discoverable are
asserted too, so the prose cannot drift away from what the loader does: the
default with nothing set resolves to :data:`_DEFAULT_POLICY_TYPES` (the
built-in union, no extras), an unset / empty variable does not widen anything
(a caller who forgets to set the variable does not accidentally admit a
malformed provider), and a well-formed extra is admitted verbatim under the
lowercasing normaliser.
"""

from __future__ import annotations

import ast
import pathlib
import re

from strands_robots.mesh import security as _security

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MODULE = _ROOT / "strands_robots" / "mesh" / "security.py"
_PAGE = _ROOT / "docs" / "security.md"
_README = _ROOT / "README.md"

_HEADING = "### Policy vocabulary allowlist (policy_type / policy_provider)"
_PREFIX = "STRANDS_MESH_POLICY_TYPE_ALLOW"
_KNOWN = frozenset({"STRANDS_MESH_POLICY_TYPE_ALLOW"})


def _policy_type_allow_env_reads() -> frozenset[str]:
    """Return every ``STRANDS_MESH_POLICY_TYPE_ALLOW*`` literal the security module reads.

    Derived from the module's own source so a variant added later is held to
    the same documentation rule without editing a list here.
    """
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        literal = None
        if isinstance(node, ast.Call) and ast.unparse(node.func) in (
            "os.getenv",
            "os.environ.get",
        ):
            if node.args and isinstance(node.args[0], ast.Constant):
                literal = node.args[0].value
        elif isinstance(node, ast.Subscript) and ast.unparse(node.value) == "os.environ":
            if isinstance(node.slice, ast.Constant):
                literal = node.slice.value
        if isinstance(literal, str) and literal.startswith(_PREFIX):
            names.add(literal)
    return frozenset(names)


def test_the_derivation_finds_the_policy_type_allow_variable_the_module_reads() -> None:
    """Positive premise: the scan is not vacuous.

    The population every downstream rule quantifies over is the set of
    ``STRANDS_MESH_POLICY_TYPE_ALLOW*`` literals the security module names on
    an ``os.getenv`` / ``os.environ`` read. If a refactor inlines the literal
    into a helper or moves the read to a different module the AST walk misses,
    every rule below would pass vacuously and the guard would silently drop
    into 'the module reads nothing so all reads are documented'. This
    premise refuses that state, and pins the known population against a
    literal so a rename is loud rather than silent.
    """
    reads = _policy_type_allow_env_reads()
    assert reads, (
        "The AST scan of mesh/security.py found no STRANDS_MESH_POLICY_TYPE_ALLOW* "
        "literal on any os.getenv / os.environ read. Either the module stopped "
        "reading the variable (in which case the security page section above "
        "should be removed too) or the walk is silently missing the read (which "
        "would make every rule below pass vacuously). "
        f"Expected the set {sorted(_KNOWN)!r}."
    )
    assert reads == _KNOWN, (
        "The set of STRANDS_MESH_POLICY_TYPE_ALLOW* literals the security module "
        f"reads has drifted from what this guard tracks: reads={sorted(reads)!r}, "
        f"known={sorted(_KNOWN)!r}. A new sibling means the README matrix, the "
        "security page and the behavioural pins below all need to grow to cover it."
    )


def test_every_policy_type_allow_variable_the_module_reads_has_a_readme_matrix_row() -> None:
    """Every ``STRANDS_MESH_POLICY_TYPE_ALLOW*`` env-var read has a README matrix row.

    The README carries a single ``STRANDS_MESH_*`` matrix of ~35 rows. Every
    variable the module reads should be findable from it, or the matrix stops
    being an index and starts being a subset of the module's env surface.
    """
    text = _README.read_text(encoding="utf-8")
    reads = _policy_type_allow_env_reads()
    missing = [name for name in sorted(reads) if f"`{name}`" not in text]
    assert not missing, (
        f"README.md does not carry a matrix row naming {missing!r} even though "
        f"mesh/security.py reads it. The row should sit in the policy family, "
        "beside `STRANDS_MESH_POLICY_HOST_ALLOW`."
    )


def test_every_policy_type_allow_variable_the_module_reads_is_named_on_the_security_page() -> None:
    """Every ``STRANDS_MESH_POLICY_TYPE_ALLOW*`` env-var read is on ``docs/security.md``.

    The security page describes the posture and this variable's whole point is
    a security posture (it is the extension knob for the ``validate_command``
    allowlist). The page must name the variables that shape it, or the posture
    is unconfigurable from the documentation.
    """
    text = _PAGE.read_text(encoding="utf-8")
    reads = _policy_type_allow_env_reads()
    missing = [name for name in sorted(reads) if f"`{name}`" not in text]
    assert not missing, (
        f"docs/security.md does not name {missing!r} even though "
        f"mesh/security.py reads it. Add it under the '{_HEADING}' section."
    )


def test_the_security_page_names_the_shared_allowlist_invariant() -> None:
    """The security page names the fact that widening admits both vocabularies at once.

    ``_DEFAULT_POLICY_TYPES`` is the union of ``_LEROBOT_POLICY_FAMILIES`` and
    ``_REGISTRY_POLICY_PROVIDERS``, and ``STRANDS_MESH_POLICY_TYPE_ALLOW``
    widens the union -- so a single entry admits both a new ``policy_type``
    family and a new ``policy_provider`` spelling. An operator who does not
    know this invariant reads the variable name as narrower than it is.
    """
    text = _PAGE.read_text(encoding="utf-8")
    section = _extract_section(text, _HEADING)
    assert "policy_provider" in section and "policy_type" in section, (
        f"The '{_HEADING}' section on docs/security.md does not name both "
        "`policy_type` and `policy_provider` in the same block. The two share "
        "one allowlist; an operator widening the variable needs to know they "
        "are widening both vocabularies at once."
    )
    assert re.search(r"share\s+one\s+allowlist", section, re.IGNORECASE), (
        f"The '{_HEADING}' section on docs/security.md does not state that "
        "`policy_type` and `policy_provider` share one allowlist. That is the "
        "surprising invariant the variable makes visible; without it the "
        "variable name reads as narrower than the gate it widens."
    )


def test_the_security_page_names_the_charset_rule() -> None:
    """The security page names the charset an entry is validated against.

    ``_POLICY_TYPE_ENTRY_RE`` is ``^[a-z][a-z0-9_]*$`` on each parsed entry: a
    malformed entry (embedded punctuation, whitespace, digit-leading) drops
    with a WARNING rather than widening the allowlist. An operator whose
    entry is silently rejected has no signal that the entry was refused unless
    they check the logs, so the rule needs to be on the page.
    """
    text = _PAGE.read_text(encoding="utf-8")
    section = _extract_section(text, _HEADING)
    assert re.search(r"\[a-z\]\[a-z0-9_\]\*|lowercase[- ]identifier", section, re.IGNORECASE), (
        f"The '{_HEADING}' section on docs/security.md does not name the "
        "`^[a-z][a-z0-9_]*$` charset rule the loader validates each entry "
        "against. Without it, an operator whose malformed entry drops has no "
        "signal from the documentation that the drop was the loader's charset "
        "check rather than a bug."
    )


def test_the_security_page_warns_against_routing_around_a_registry_omission() -> None:
    """The section warns against using the variable to patch a registry omission.

    ``_REGISTRY_POLICY_PROVIDERS`` and ``registry/policies.json`` must stay in
    sync (a separate guard test in ``tests/mesh/`` enforces the bijection).
    Using ``STRANDS_MESH_POLICY_TYPE_ALLOW`` to admit a spelling that should
    have been added to ``_REGISTRY_POLICY_PROVIDERS`` is the anti-pattern the
    sync-guard exists to catch; the documentation has to steer the operator
    to the right side, or the sync-guard's protection is bypassed at runtime
    by an env var.
    """
    text = _PAGE.read_text(encoding="utf-8")
    section = _extract_section(text, _HEADING)
    assert "registry" in section.lower(), (
        f"The '{_HEADING}' section on docs/security.md does not mention the "
        "registry. Widening this variable to admit a spelling that belongs in "
        "`_REGISTRY_POLICY_PROVIDERS` / `registry/policies.json` is the "
        "anti-pattern the sync-guard exists to catch; the section has to "
        "steer the operator to the registry, not this env var."
    )


def test_the_default_the_documentation_names_is_what_the_loader_returns() -> None:
    """Behavioural pin: nothing set resolves to ``_DEFAULT_POLICY_TYPES`` verbatim.

    The security page states the variable is 'optional' and that widening it
    'appends' to the built-in list. Both claims collapse if a caller who has
    not set the variable gets anything other than the built-in union.
    """
    import os

    # Snapshot and drop, then clear the lru_cache so the loader re-reads env.
    previous = os.environ.pop("STRANDS_MESH_POLICY_TYPE_ALLOW", None)
    try:
        _security._clear_security_caches_for_tests()
        assert _security._policy_type_allowlist() == _security._DEFAULT_POLICY_TYPES, (
            "With STRANDS_MESH_POLICY_TYPE_ALLOW unset the loader must return "
            "_DEFAULT_POLICY_TYPES verbatim. The documentation calls the "
            "variable 'optional' and its widening 'appends'; both statements "
            "collapse if the caller who has not set it gets a different set."
        )
    finally:
        if previous is not None:
            os.environ["STRANDS_MESH_POLICY_TYPE_ALLOW"] = previous
        _security._clear_security_caches_for_tests()


def test_an_empty_value_does_not_widen_anything(monkeypatch) -> None:
    """Behavioural pin: empty / whitespace value falls back to ``_DEFAULT_POLICY_TYPES``.

    A caller who sets ``STRANDS_MESH_POLICY_TYPE_ALLOW=""`` or
    ``STRANDS_MESH_POLICY_TYPE_ALLOW="   "`` must get the built-in list back,
    not an empty allowlist. The alternative would be a mesh that refuses every
    ``execute`` / ``start`` payload once the operator set the variable in a
    shell script and forgot the value, which would ship as a support burden
    the documented posture ('optional; comma-separated extras') has to hold up.
    """
    for value in ("", "   ", "\t\n"):
        monkeypatch.setenv("STRANDS_MESH_POLICY_TYPE_ALLOW", value)
        _security._clear_security_caches_for_tests()
        result = _security._policy_type_allowlist()
        assert result == _security._DEFAULT_POLICY_TYPES, (
            f"STRANDS_MESH_POLICY_TYPE_ALLOW={value!r} must resolve to the "
            "built-in list verbatim. An empty / whitespace value falls back to "
            f"the default; got {sorted(result - _security._DEFAULT_POLICY_TYPES)!r} "
            f"extra, {sorted(_security._DEFAULT_POLICY_TYPES - result)!r} missing."
        )
    _security._clear_security_caches_for_tests()


def test_a_well_formed_extra_is_admitted_verbatim(monkeypatch) -> None:
    """Behavioural pin: a well-formed extra is admitted under lowercase normalisation.

    The documentation states extras are 'appended to the built-in list' and
    that spellings are 'normalised through .lower() before the compare'.
    ``foo_provider`` (well-formed) is admitted; ``FOO_PROVIDER`` normalises to
    the same and is admitted; ``bar`` beside a comma is admitted; a malformed
    entry (embedded punctuation) is not admitted (dropped with a WARNING).
    """
    monkeypatch.setenv("STRANDS_MESH_POLICY_TYPE_ALLOW", "foo_provider, BAR, evil;rm")
    _security._clear_security_caches_for_tests()
    result = _security._policy_type_allowlist()
    assert "foo_provider" in result, (
        "A well-formed lowercase extra must be admitted. The loader lowercases "
        "the raw string before splitting; a well-formed entry survives the "
        f"charset gate; got extras {sorted(result - _security._DEFAULT_POLICY_TYPES)!r}."
    )
    assert "bar" in result, (
        "A case-variant well-formed extra must normalise to the lowercased "
        "spelling and be admitted. `BAR` in the env var admits payloads naming "
        "`bar`; got extras "
        f"{sorted(result - _security._DEFAULT_POLICY_TYPES)!r}."
    )
    assert "evil;rm" not in result, (
        "A malformed entry (embedded punctuation the charset regex rejects) "
        "must not widen the allowlist. `evil;rm` fails ^[a-z][a-z0-9_]*$ and "
        "must drop with a WARNING; got extras "
        f"{sorted(result - _security._DEFAULT_POLICY_TYPES)!r}."
    )
    _security._clear_security_caches_for_tests()


def _extract_section(text: str, heading: str) -> str:
    """Return the text between ``heading`` and the next ``###`` heading or end.

    Lets the per-property assertions above read a scoped slice of the security
    page, so a claim about the mTLS section cannot pass because the phrase
    happens to appear in the AWS IoT section further down.
    """
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        return ""
    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("### ") or stripped.startswith("## "):
            end = i
            break
    return "\n".join(lines[start:end])
