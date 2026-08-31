"""Grade the mesh transport TLS environment reference against the code's env surface.

``STRANDS_MESH_AUTH_MODE=mtls`` is the default wire posture, and three
filesystem paths are the whole of that posture's configuration: the CA bundle,
this peer's certificate, and its private key.  With any one of them unset
``_resolve_tls_paths`` raises ``ValueError`` naming all three and the session
never opens - so on a fleet that sets no dev flag, those three names are the
difference between a mesh that comes up and one that does not.

Until this file's companion change, none of the three was named on any
documentation surface.  Two things made that worse than an ordinary omission:

- The same security page documents the *optional* AWS IoT transport's
  credential family exhaustively - four variables, each with its
  required/optional status, its default, and the failure posture - while naming
  none of the three the *default* transport requires.  An operator reading the
  production-posture section found the ACL half fully named by variable
  (``STRANDS_MESH_ACL_FILE``, ``STRANDS_MESH_ACCEPT_PERMISSIVE_ACL``) and the
  mTLS half, which that section calls the thing the ACL is paired with, named
  by none.
- ``mesh/_zenoh_config.py`` cites the README environment-variable matrix three
  times as the surface that promises the key-mode contract, once inside a
  WARNING an operator reads on Windows.  That matrix names 32 other
  ``STRANDS_MESH_*`` variables and named no TLS path, so the warning pointed a
  reader at a table where the variable it was warning about did not appear.

Four properties are graded, all derived from the package rather than from a
list kept beside it, so a fourth ``STRANDS_MESH_TLS_*`` variable is graded on
arrival:

- **Every TLS variable the module reads is documented on the security page.**
- **Every one of them has a row in the README matrix the module cites.** This
  is what makes the module's own pointer resolve; a variable whose key-mode
  contract the code attributes to that table must be in it.
- **They are documented together under one named heading.** The three
  configure one channel - the transport's TLS identity - so an operator
  bringing up a production fleet should find them together.
- **The section describes the refusal rather than a downgrade.** The absence of
  the material is a refusal at session-open, not a quiet fall back to plain
  TCP, and an operator planning monitoring needs that distinction.

The behavioural facts the documentation exists to make discoverable are
asserted too, so the prose cannot drift away from what the loader does: the
default posture with nothing set resolves to ``mtls`` and refuses naming all
three, and the same call succeeds once the material is supplied.
"""

import ast
import pathlib
import re

import pytest

from strands_robots.mesh import _zenoh_config

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MODULE = _ROOT / "strands_robots" / "mesh" / "_zenoh_config.py"
_SESSION = _ROOT / "strands_robots" / "mesh" / "session.py"
_PAGE = _ROOT / "docs" / "security.md"
_README = _ROOT / "README.md"

_HEADING = "### Transport credentials (mTLS material)"
_PREFIX = "STRANDS_MESH_TLS_"
_KNOWN = frozenset({"STRANDS_MESH_TLS_CA", "STRANDS_MESH_TLS_CERT", "STRANDS_MESH_TLS_KEY"})


def _tls_env_reads() -> frozenset[str]:
    """Return every ``STRANDS_MESH_TLS_*`` name the config module reads.

    Derived from the module's own source so a path added later is held to the
    same documentation rule without editing a list here.
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
    """Return the ``STRANDS_MESH_TLS_*`` names a document names."""
    return frozenset(re.findall(rf"{_PREFIX}[A-Z_]+", text))


def _readme_matrix_rows() -> frozenset[str]:
    """Return the ``STRANDS_MESH_TLS_*`` names carrying a README matrix row.

    A mention in prose is not a matrix row: the module points a reader at a
    table, so the rule reads the table.
    """
    rows = re.findall(
        rf"^\|\s*`({_PREFIX}[A-Z_]+)`\s*\|",
        _README.read_text(encoding="utf-8"),
        re.M,
    )
    return frozenset(rows)


def _section() -> str:
    """Return the named section's body, or fail naming the missing heading."""
    text = _PAGE.read_text(encoding="utf-8")
    assert _HEADING in text, (
        f"docs/security.md no longer carries {_HEADING!r}; the TLS material "
        "rules below read that section, so a rename must move them with it"
    )
    after = text.split(_HEADING, 1)[1]
    return after.split("\n## ", 1)[0].split("\n### ", 1)[0]


def _headings_naming(name: str) -> list[str]:
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

    def test_the_module_reads_the_three_known_tls_paths(self) -> None:
        reads = _tls_env_reads()
        assert _KNOWN <= reads, f"expected the CA/cert/key paths among the module's TLS reads, got {sorted(reads)}"

    def test_the_derivation_is_not_empty(self) -> None:
        assert _tls_env_reads(), (
            "the TLS env scan found nothing; the derivation has gone blind and every rule below would pass vacuously"
        )


class TestEveryTlsPathIsDocumented:
    """A path the transport requires and no page names cannot be configured."""

    def test_the_security_page_names_every_tls_path(self) -> None:
        missing = sorted(_tls_env_reads() - _documented(_PAGE.read_text(encoding="utf-8")))
        assert not missing, (
            f"docs/security.md names none of {missing}; AUTH_MODE=mtls is the "
            "default, so these are required on a fleet that sets no dev flag"
        )

    def test_the_readme_matrix_the_module_cites_names_every_tls_path(self) -> None:
        missing = sorted(_tls_env_reads() - _readme_matrix_rows())
        assert not missing, (
            f"the README env-var matrix has no row for {missing}, and "
            "mesh/_zenoh_config.py cites that matrix as the surface promising "
            "the key-mode contract - so its own pointer does not resolve"
        )

    def test_the_module_does_cite_the_readme_matrix(self) -> None:
        source = _MODULE.read_text(encoding="utf-8")
        assert "README env-var matrix" in source, (
            "the rule above exists because the module points operators at the "
            "README matrix; if that citation is gone, re-scope the rule"
        )


class TestTheyAreDocumentedTogether:
    """One channel, one place to look."""

    def test_every_tls_path_is_named_under_one_heading(self) -> None:
        headings = {name: _headings_naming(name) for name in sorted(_tls_env_reads())}
        assert all(headings.values()), f"a TLS path is named under no heading: {headings}"
        shared = set.intersection(*(set(v) for v in headings.values()))
        assert shared, f"the TLS paths are split across headings with none in common: {headings}"

    def test_they_sit_under_the_named_heading(self) -> None:
        section = _section()
        missing = sorted(name for name in _tls_env_reads() if name not in section)
        assert not missing, f"{missing} are documented outside {_HEADING!r}"


class TestTheSectionDescribesTheRefusal:
    """The absence of the material refuses; it does not downgrade."""

    def test_the_section_calls_the_absence_a_refusal(self) -> None:
        section = _section().lower()
        assert "required" in section and "together" in section, (
            "the section must say the three are required together - that is the contract _resolve_tls_paths enforces"
        )

    def test_the_section_denies_a_silent_downgrade(self) -> None:
        section = _section().lower()
        assert "downgrad" in section, (
            "the section must say the loader does not downgrade to plain TCP; "
            "an operator planning monitoring needs the refusal named"
        )

    def test_the_section_names_the_key_mode_contract_and_its_windows_gap(self) -> None:
        section = _section()
        assert "0600" in section or "0o600" in section, (
            "the section must name the key-mode contract the loader enforces"
        )
        assert "Windows" in section, (
            "the section must name the platform where the mode gate is skipped; "
            "that is exactly what the loader's one-shot WARNING points at"
        )


class TestTheBehaviourTheDocumentationNames:
    """Drive the loader, so the prose cannot drift from what it does."""

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            *sorted(_KNOWN),
            "STRANDS_MESH_AUTH_MODE",
            "STRANDS_MESH_LOCAL_DEV",
            "STRANDS_MESH_I_KNOW_THIS_IS_INSECURE",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_the_default_posture_is_mtls(self) -> None:
        assert _zenoh_config.resolve_auth_mode() == "mtls", (
            "the security page calls mtls the default when neither dev flag is "
            "set; the documented requirement rests on that"
        )

    def test_the_default_posture_refuses_and_names_all_three(self) -> None:
        with pytest.raises(ValueError) as caught:
            _zenoh_config.tls_block()
        message = str(caught.value)
        missing = sorted(name for name in _KNOWN if name not in message)
        assert not missing, f"the refusal does not name {missing}: {message}"

    def test_supplying_the_material_brings_the_block_up(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        for name, filename in (
            ("STRANDS_MESH_TLS_CA", "ca.pem"),
            ("STRANDS_MESH_TLS_CERT", "cert.pem"),
            ("STRANDS_MESH_TLS_KEY", "key.pem"),
        ):
            path = tmp_path / filename
            path.write_text("x", encoding="utf-8")
            monkeypatch.setenv(name, str(path))
        (tmp_path / "key.pem").chmod(0o600)
        block = _zenoh_config.tls_block()
        assert block, "supplying the documented material must build the TLS block"


class TestTheRefusalBelongsToTheProductionPosture:
    """The caller reaches it exactly under mtls, which is what the page says."""

    def test_the_tls_block_is_built_only_under_mtls(self) -> None:
        tree = ast.parse(_SESSION.read_text(encoding="utf-8"))
        guarded = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if "mtls" not in ast.unparse(node.test):
                continue
            if "tls_block()" in "".join(ast.unparse(s) for s in node.body):
                guarded = True
        assert guarded, (
            "session.py no longer gates tls_block on the mtls posture, so the "
            "refusal the section documents is reached somewhere else too"
        )
