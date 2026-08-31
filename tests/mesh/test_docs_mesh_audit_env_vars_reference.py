"""Grade the mesh audit-log environment reference against the module's env surface.

The mesh audit log is one channel with four sibling environment knobs. Each
one configures the same file: where it is written (``STRANDS_MESH_AUDIT_DIR``),
whether records carry a per-record HMAC that lets a verifier reject a forged
entry (``STRANDS_MESH_AUDIT_PSK``), and the per-file / rotation bounds that
keep disk use finite (``STRANDS_MESH_AUDIT_MAX_BYTES`` and
``STRANDS_MESH_AUDIT_MAX_FILES``). ``docs/security.md`` documents them together
under ``## Audit log``. Until this file's companion change, the README
environment-variable matrix carried a row for ``_DIR`` alone -- so a reader
scanning the matrix for the audit family found one variable of four and not
the tamper-evidence gate the security page calls "a deliberate posture" or
the rotation bounds that same page pairs it with.

The rules below read the module's own ``os.getenv`` literals so the graded
population tracks the code rather than a list maintained beside it. A fifth
``STRANDS_MESH_AUDIT_*`` path added later is held to the same rule the hour it
lands. Four properties, plus one keep-the-derivation-honest premise:

- **Every audit variable the module reads has a README matrix row.** This is
  what makes the matrix a single index for the family; a reader scanning the
  matrix section for the audit family should find every knob that configures
  the same file.
- **Every one of them is named on the security page.** The prose surface that
  describes the posture must name the variables that shape it, or the posture
  is unconfigurable.
- **The matrix rows sit contiguously.** One channel, one place to look: an
  operator sizing storage or hardening tamper-evidence should not have to
  hunt for siblings scattered through 32 unrelated rows.
- **The derivation is not empty.** If the scan ever goes blind (module moved,
  literals inlined, a refactor breaks the AST walk) every rule above would
  pass vacuously; a positive premise refuses that state.
"""

import ast
import pathlib
import re

from strands_robots.mesh import audit as _audit_module

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MODULE = _ROOT / "strands_robots" / "mesh" / "audit.py"
_PAGE = _ROOT / "docs" / "security.md"
_README = _ROOT / "README.md"

_PREFIX = "STRANDS_MESH_AUDIT_"
_KNOWN = frozenset(
    {
        "STRANDS_MESH_AUDIT_DIR",
        "STRANDS_MESH_AUDIT_PSK",
        "STRANDS_MESH_AUDIT_MAX_BYTES",
        "STRANDS_MESH_AUDIT_MAX_FILES",
    }
)


def _audit_env_reads() -> frozenset[str]:
    """Return every ``STRANDS_MESH_AUDIT_*`` name the audit module reads.

    Derived from the module's own source so a variable added later is held to
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


def _documented(text: str) -> frozenset[str]:
    """Return the ``STRANDS_MESH_AUDIT_*`` names a document names."""
    return frozenset(re.findall(rf"{_PREFIX}[A-Z_]+", text))


def _readme_matrix_rows() -> frozenset[str]:
    """Return the ``STRANDS_MESH_AUDIT_*`` names carrying a README matrix row.

    A mention in prose is not a matrix row: the matrix is a scan index for
    the family, so the rule reads the table.
    """
    rows = re.findall(
        rf"^\|\s*`({_PREFIX}[A-Z_]+)`\s*\|",
        _README.read_text(encoding="utf-8"),
        re.M,
    )
    return frozenset(rows)


def _readme_matrix_row_order() -> list[str]:
    """Return the ``STRANDS_MESH_AUDIT_*`` names in the matrix, in order of appearance."""
    return re.findall(
        rf"^\|\s*`({_PREFIX}[A-Z_]+)`\s*\|",
        _README.read_text(encoding="utf-8"),
        re.M,
    )


def _audit_headings_naming(name: str) -> list[str]:
    """Return every ``##``/``###`` heading whose section names ``name``."""
    text = _PAGE.read_text(encoding="utf-8")
    parts = re.split(r"^(#{2,3} .+)$", text, flags=re.M)
    out = []
    for index in range(1, len(parts), 2):
        if name in parts[index + 1]:
            out.append(parts[index].lstrip("# ").strip())
    return out


class TestThePopulationIsDerived:
    """The rules read the module, not a list maintained beside them."""

    def test_the_module_reads_the_four_known_audit_variables(self) -> None:
        reads = _audit_env_reads()
        assert _KNOWN <= reads, (
            f"expected the four documented audit knobs among the module's reads, got {sorted(reads)}"
        )

    def test_the_derivation_is_not_empty(self) -> None:
        assert _audit_env_reads(), (
            "the audit env scan found nothing; the derivation has gone blind and every rule below would pass vacuously"
        )

    def test_the_audit_module_still_lives_where_the_scan_expects(self) -> None:
        # If audit.py moves, this rule points the failing test at the reason.
        assert _MODULE.exists(), (
            "strands_robots/mesh/audit.py is gone; move the scan target to wherever the audit env-reads live now"
        )
        # Behavioural premise: the constant table also names its knobs.  Keeps
        # the AST literal walk honest against a rewrite that pulls names from
        # a dict, since :func:`_read_env_max_bytes` and friends would still
        # call ``os.getenv``.
        _module_source = _MODULE.read_text(encoding="utf-8")
        assert 'os.getenv("STRANDS_MESH_AUDIT_PSK")' in _module_source, (
            "the AST walk finds only literal os.getenv calls; a refactor to a "
            "dict-driven read must update this test's scan to match"
        )


class TestEveryAuditVariableIsDocumented:
    """A knob the audit log honours and no page names cannot be configured."""

    def test_the_readme_matrix_names_every_audit_variable(self) -> None:
        missing = sorted(_audit_env_reads() - _readme_matrix_rows())
        assert not missing, (
            f"the README env-var matrix has no row for {missing}; the audit "
            "family is documented as one channel and the matrix is the scan "
            "index a reader uses to discover the family's knobs"
        )

    def test_the_security_page_names_every_audit_variable(self) -> None:
        missing = sorted(_audit_env_reads() - _documented(_PAGE.read_text(encoding="utf-8")))
        assert not missing, (
            f"docs/security.md names none of {missing}; the audit posture is "
            "described there and a variable that shapes it must be nameable"
        )


class TestTheyAreDocumentedTogether:
    """One channel, one place to look."""

    def test_the_matrix_rows_are_contiguous(self) -> None:
        rows = _readme_matrix_row_order()
        reads = _audit_env_reads()
        indices = [i for i, name in enumerate(rows) if name in reads]
        if not indices:
            return
        span = indices[-1] - indices[0] + 1
        assert span == len(indices), (
            f"the audit matrix rows are not contiguous: {rows[indices[0] : indices[-1] + 1]}; "
            "an operator sizing storage or hardening tamper-evidence should "
            "find every knob together, not scattered through unrelated rows"
        )

    def test_every_audit_variable_is_named_under_one_security_heading(self) -> None:
        headings = {name: _audit_headings_naming(name) for name in sorted(_audit_env_reads())}
        assert all(headings.values()), f"an audit variable is named under no security-page heading: {headings}"
        shared = set.intersection(*(set(v) for v in headings.values()))
        assert shared, f"the audit variables are split across headings with none in common: {headings}"


class TestTheBehaviourThePagesExistToMakeDiscoverable:
    """The prose cannot drift away from what the audit module honours."""

    def test_the_psk_read_lives_where_the_page_says_it_does(self) -> None:
        # docs/security.md says: "when set, per-record HMAC is on".
        # The module read is the wire between them.  If the read moves or its
        # env name changes, the page's promise no longer resolves.
        assert 'os.getenv("STRANDS_MESH_AUDIT_PSK")' in _MODULE.read_text(encoding="utf-8"), (
            "the audit module no longer reads STRANDS_MESH_AUDIT_PSK by that "
            "name; docs/security.md promises HMAC when the variable is set, "
            "and the promise must reach the loader"
        )

    def test_the_size_and_file_caps_are_read_by_the_declared_names(self) -> None:
        source = _MODULE.read_text(encoding="utf-8")
        for name in ("STRANDS_MESH_AUDIT_MAX_BYTES", "STRANDS_MESH_AUDIT_MAX_FILES"):
            assert f'os.getenv("{name}")' in source, (
                f"{name} is no longer read at that name; the README matrix "
                "and the security page both promise it, and the promise must "
                "reach the rotator"
            )

    def test_the_module_is_reachable_via_its_public_import_path(self) -> None:
        # Grounds the behavioural cells above in the same import surface docs
        # readers use.  If the module is renamed, the scan needs to move with
        # it, and this cell names the reason at the failure line.
        assert getattr(_audit_module, "__file__", None), (
            "strands_robots.mesh.audit could not resolve to a file; the "
            "scan target and the documented posture are diverging"
        )
