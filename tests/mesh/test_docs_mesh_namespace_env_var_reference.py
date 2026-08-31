"""Grade the mesh namespace environment reference against the module's env surface.

``STRANDS_MESH_NAMESPACE`` is the Zenoh ``namespace`` field on every peer of a
fleet. It prefixes every mesh key-expression the tree emits, and Zenoh only
routes application traffic between peers whose namespaces match -- so it is the
knob that keeps a test rig from receiving a production fleet's commands on a
shared LAN. The failure mode of a mismatch is deliberately silent (peers connect
at the transport layer and exchange no application traffic), which puts a
particular weight on the variable being discoverable at all: an operator who
never sees the peers of the other fleet cannot deduce the knob from the
symptom.

Until this file's companion change, ``STRANDS_MESH_NAMESPACE`` was named on
neither the README environment-variable matrix nor ``docs/security.md``. The
default value (``"strands"``) is a documented default an operator overriding
fleet isolation needs to know before they change it, because a rolling change
across a fleet leaves the two halves unable to see each other for the duration
of the roll.

The rules below read the module's own ``os.getenv`` / ``os.environ`` literals
so the graded population tracks the code rather than a list kept beside it. A
future ``STRANDS_MESH_NAMESPACE_*`` sibling (a per-topic namespace override, a
compat spelling) is held to the same rule the hour it lands. Four properties,
plus one keep-the-derivation-honest premise:

- **Every namespace variable the module reads has a README matrix row.** This
  is what makes the matrix a single index for the family; a reader scanning
  the matrix for the routing-isolation knob should find it.
- **Every one of them is named on the security page.** The prose surface that
  describes the posture -- and this variable's whole point is a security
  posture -- must name the variables that shape it, or the posture is
  unconfigurable.
- **The documentation names the default value.** ``STRANDS_MESH_NAMESPACE``
  defaults to ``"strands"`` and the default is what the built-in ACL
  key_exprs and the hardcoded topic prefixes all track, so an operator
  overriding it needs the default named to know what they are diverging from.
- **The documentation names the silent-mismatch failure mode.** The variable
  is worth documenting precisely *because* its failure mode is silent -- a
  peer that appears absent from the fleet rather than one that raises. A
  documentation change that named the variable and did not name that
  property would leave an operator scanning for a peer they cannot see.
- **The derivation is not empty.** If the scan ever goes blind (module moved,
  literals inlined, a refactor breaks the AST walk) every rule above would
  pass vacuously; a positive premise refuses that state.

The behavioural facts the documentation exists to make discoverable are
asserted too, so the prose cannot drift away from what the loader does: the
default with nothing set resolves to ``"strands"``, an empty / whitespace value
falls back to the default rather than producing an empty prefix, and a
non-empty value is honoured verbatim.
"""

from __future__ import annotations

import ast
import pathlib
import re

from strands_robots.mesh import _zenoh_config

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MODULE = _ROOT / "strands_robots" / "mesh" / "_zenoh_config.py"
_PAGE = _ROOT / "docs" / "security.md"
_README = _ROOT / "README.md"

_HEADING = "### Fleet routing isolation (namespace)"
_PREFIX = "STRANDS_MESH_NAMESPACE"
_KNOWN = frozenset({"STRANDS_MESH_NAMESPACE"})
_DEFAULT = "strands"


def _namespace_env_reads() -> frozenset[str]:
    """Return every ``STRANDS_MESH_NAMESPACE*`` name the config module reads.

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


def test_the_derivation_finds_the_namespace_variable_the_module_reads() -> None:
    """Positive premise: the scan is not vacuous.

    The population every downstream rule quantifies over is the set of
    ``STRANDS_MESH_NAMESPACE*`` literals the config module names on an
    ``os.getenv`` / ``os.environ`` read. If a refactor makes that scan empty
    every rule below would pass vacuously; this cell refuses that state and
    pins the one variable we currently know about.
    """
    reads = _namespace_env_reads()
    assert _KNOWN <= reads, (
        f"the config module no longer reads {sorted(_KNOWN - reads)}; "
        "either the read moved to another module (in which case the scan target "
        "in this file needs updating) or the variable was renamed / removed "
        "(in which case both documentation surfaces need updating too)"
    )
    assert reads, "the AST scan of _zenoh_config.py found no STRANDS_MESH_NAMESPACE* reads at all"


def test_every_namespace_variable_the_module_reads_has_a_readme_matrix_row() -> None:
    """Every read is a row on the README env-var matrix.

    The matrix is the single index the module's own code cites for fleet
    configuration, so a variable the module reads without a row is a knob
    with no discoverable entry.
    """
    reads = _namespace_env_reads()
    readme_text = _README.read_text(encoding="utf-8")
    missing = sorted(name for name in reads if f"`{name}`" not in readme_text)
    assert not missing, f"README matrix is missing rows for: {missing}"


def test_every_namespace_variable_the_module_reads_is_named_on_the_security_page() -> None:
    """Every read is named on the security page.

    The variable configures a security posture (fleet routing isolation), so
    the prose surface that describes that posture must name the variables
    that shape it.
    """
    reads = _namespace_env_reads()
    page_text = _PAGE.read_text(encoding="utf-8")
    missing = sorted(name for name in reads if name not in page_text)
    assert not missing, f"docs/security.md is missing names for: {missing}"


def test_the_security_page_names_the_default_namespace() -> None:
    """The documentation names the default value.

    ``STRANDS_MESH_NAMESPACE`` defaults to ``"strands"`` and the default is
    what the built-in ACL key_exprs and the hardcoded topic prefixes all
    track. An operator overriding it needs the default named to know what
    they are diverging from.
    """
    page_text = _PAGE.read_text(encoding="utf-8")
    # Find the namespace section and grade its content specifically.
    section_match = re.search(
        rf"{re.escape(_HEADING)}\n\n(.*?)(?=\n### |\n## |\Z)",
        page_text,
        re.DOTALL,
    )
    assert section_match, f"heading '{_HEADING}' is missing from docs/security.md"
    section = section_match.group(1)
    assert _DEFAULT in section, (
        f"the namespace section names the variable but not its default {_DEFAULT!r}; "
        "operators overriding fleet isolation need the default named to know what "
        "they are diverging from"
    )


def test_the_security_page_names_the_silent_mismatch_failure_mode() -> None:
    """The documentation names the silent-mismatch failure mode.

    A namespace mismatch does not raise: peers connect at the transport
    layer and exchange no application traffic. That is the whole reason the
    variable is worth documenting -- an operator scanning for a peer they
    cannot see cannot deduce a namespace mismatch from any error message.
    A documentation change that named the variable and did not name that
    property would leave the operator without the diagnostic hook.
    """
    page_text = _PAGE.read_text(encoding="utf-8")
    section_match = re.search(
        rf"{re.escape(_HEADING)}\n\n(.*?)(?=\n### |\n## |\Z)",
        page_text,
        re.DOTALL,
    )
    assert section_match, f"heading '{_HEADING}' is missing from docs/security.md"
    section = section_match.group(1).lower()
    # The section must name the fact that a mismatch is silent / absent /
    # not-loud. Any of these phrasings satisfies the rule; the point is
    # that the operator reading the section learns the diagnostic hook.
    silent_phrases = ("silent", "absent", "appears absent", "no application traffic")
    matched = [phrase for phrase in silent_phrases if phrase in section]
    assert matched, (
        "the namespace section does not name the silent-mismatch failure mode; "
        f"none of {silent_phrases} appear in the section prose. An operator "
        "scanning for a peer they cannot see cannot deduce a namespace "
        "mismatch from any error message, so the documentation is where the "
        "diagnostic hook lives"
    )


def test_the_readme_row_names_the_default_and_the_silent_mismatch() -> None:
    """The README matrix row itself names the two facts that make the knob usable.

    A caller who never opens docs/security.md and only reads the matrix must
    still learn (a) the default value, and (b) that a mismatch is silent --
    the two facts that make the variable actionable rather than a name.
    """
    readme_text = _README.read_text(encoding="utf-8")
    # Row starts at the pipe-and-backtick line; regex captures through the
    # closing pipe of the default column (which ends with a bare newline).
    row_match = re.search(
        r"^\| `STRANDS_MESH_NAMESPACE` \| ([^|]+) \| ([^|]+) \|",
        readme_text,
        re.MULTILINE,
    )
    assert row_match, "README matrix has no row for STRANDS_MESH_NAMESPACE"
    description, default_column = row_match.group(1), row_match.group(2)
    assert _DEFAULT in default_column, (
        f"the STRANDS_MESH_NAMESPACE row's default column is {default_column!r}, does not name the {_DEFAULT!r} default"
    )
    description_lower = description.lower()
    silent_phrases = ("silent", "cannot exchange", "no application traffic")
    matched = [phrase for phrase in silent_phrases if phrase in description_lower]
    assert matched, (
        "the STRANDS_MESH_NAMESPACE row does not name the silent-mismatch "
        f"failure mode; none of {silent_phrases} appear in the description column"
    )


def test_the_default_the_documentation_names_is_what_the_loader_returns(monkeypatch) -> None:
    """Behavioural: with the variable unset, ``resolve_namespace()`` returns the documented default.

    Pins the documentation to the code: a future change to
    :data:`_zenoh_config.DEFAULT_NAMESPACE` that did not update both surfaces
    would leave the documentation naming a default the loader no longer uses.
    """
    monkeypatch.delenv("STRANDS_MESH_NAMESPACE", raising=False)
    assert _zenoh_config.resolve_namespace() == _DEFAULT


def test_an_empty_value_falls_back_to_the_default_as_documented(monkeypatch) -> None:
    """Behavioural: an empty / whitespace value falls back to the default.

    The security page states this behaviour explicitly (empty is treated
    identically to unset, to prevent keys like ``"//presence"``). Pin it so
    a future refactor that treated empty as an override would surface here.
    """
    for value in ("", "   ", "\t"):
        monkeypatch.setenv("STRANDS_MESH_NAMESPACE", value)
        assert _zenoh_config.resolve_namespace() == _DEFAULT, (
            f"empty/whitespace value {value!r} did not fall back to the default; "
            "the security page documents this behaviour, so a divergence here "
            "would silently break the documented contract"
        )


def test_a_non_empty_value_is_honoured_verbatim(monkeypatch) -> None:
    """Behavioural: a non-empty value overrides the default.

    Pins the documented override path -- a test fleet on a shared LAN uses
    this to isolate itself from production, so the override has to be honoured
    exactly as set (not lowercased, not stripped of internal characters).
    """
    monkeypatch.setenv("STRANDS_MESH_NAMESPACE", "prod-fleet-42")
    assert _zenoh_config.resolve_namespace() == "prod-fleet-42"
