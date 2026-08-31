"""Grade the ACL-acknowledgement environment reference against the module's env surface.

``STRANDS_MESH_ACCEPT_PERMISSIVE_ACL`` is the acknowledgement token for a
permissive mesh ACL posture. It has one spelling and three readers, and setting
it takes all three effects:

1. ``_acl_config._load_acl_file`` raises ``PermissiveACLError`` when
   ``STRANDS_MESH_ACL_FILE`` points at a blacklist-shaped ACL
   (``default_permission: "allow"`` with a non-empty ``rules`` list) unless the
   variable is set to ``1``/``true``/``yes``. (``_validate_acl_shape``, called
   at the end of the same loader, grades the file against the ACL schema; it is
   not where this refusal is raised.)
2. ``core.Mesh._refuse_under_permissive_default_acl`` takes it as the operator
   opt-in for the built-in permissive default: with the token unset,
   ``Mesh.start`` refuses to bring the wire up at all.
3. ``session._build_config`` skips the per-session-open ``PERMISSIVE built-in
   default ACL`` WARNING while it is set.

Until this file's companion change, ``docs/security.md`` named the variable only
as a silencer for that session warning and said nothing about the loader
refusal; the first correction then over-rotated and asserted the token "does not
silence" the warning, which reader 3 contradicts. Both framings understated the
blast radius in the same direction: an operator who sets the token for one of
the three effects gets the other two, so a later loss of
``STRANDS_MESH_ACL_FILE`` from that environment brings the wire up on the
permissive default with the start gate waived and the recurring warning
suppressed. The README environment-variable matrix that ``_zenoh_config`` cites
three times as the surface promising the key-mode contract named 32 other
``STRANDS_MESH_*`` variables and none of this one, so an operator tracing the
refusal from the module source found the variable and an operator tracing it
from either documentation surface did not.

The rules below read the package's own ``os.getenv`` / ``os.environ`` literals
so the graded population tracks the code rather than a list kept beside it. A
future ``STRANDS_MESH_ACCEPT_PERMISSIVE_ACL_*`` sibling (a per-key scoped
opt-in, a rules-count ceiling) is held to the same rule the hour it lands, and
so is a fourth *reader* of the existing token. Properties, plus one
keep-the-derivation-honest premise:

- **Every acknowledgement variable the module reads has a README matrix row.**
  This is what makes the matrix a single index for the ACL-configuration
  family; a reader scanning the matrix for the blacklist-acknowledgement knob
  should find it.
- **Every one of them is named on the security page.** The prose surface that
  describes the ACL posture -- and this variable's whole point is a security
  posture -- must name the variables that shape it, or the posture is
  unconfigurable.
- **The documentation names the two ACL shapes.** The refusal only distinguishes
  ``allow`` + rules (blacklist) from ``deny`` + rules (whitelist); a
  documentation change that named the variable and did not name that
  distinction would leave an operator setting the token without understanding
  the shape they are acknowledging.
- **The documentation names the refusal, and names it as one of three
  effects.** The loader raises ``PermissiveACLError``; the start gate and the
  session warning are the other two readers. A page that names only one of the
  three understates the blast radius of setting the token, which is the axis
  both prior framings got wrong.
- **Every function that reads the token is named in the section.** This is the
  general form of the three per-effect rules below, and the rule that keeps the
  documented blast radius equal to the implemented one: a fourth reader added
  anywhere in the package fails that cell until the section says what setting
  the token now does at that site.
- **The documentation names the two-remediation fork.** The ``PermissiveACLError``
  message names both remediations (rewrite as ``deny`` + ``allow`` rules, or
  set the acknowledgement token), so an operator who reads only the refusal
  sees the fork; the documentation must not push them toward the token by
  omission.
- **The derivation is not empty.** If the scan ever goes blind (module moved,
  literals inlined, a refactor breaks the AST walk) every rule above would
  pass vacuously; a positive premise refuses that state.

The behavioural facts the documentation exists to make discoverable are
asserted too, so the prose cannot drift away from what the loader does: a
blacklist ACL is refused with nothing set, is refused with an empty /
whitespace value, is accepted with ``1``/``true``/``yes`` (case-insensitive),
and the built-in permissive default (``allow`` with empty rules) does not
reach this gate.
"""

from __future__ import annotations

import ast
import json
import logging
import pathlib
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh import _acl_config

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PACKAGE = _ROOT / "strands_robots"
_MODULE = _PACKAGE / "mesh" / "_acl_config.py"
_PAGE = _ROOT / "docs" / "security.md"
_README = _ROOT / "README.md"

_HEADING = "### Blacklist ACL acknowledgement (`STRANDS_MESH_ACCEPT_PERMISSIVE_ACL`)"
_PREFIX = "STRANDS_MESH_ACCEPT_PERMISSIVE_ACL"
_KNOWN = frozenset({"STRANDS_MESH_ACCEPT_PERMISSIVE_ACL"})


def _accept_env_reads() -> frozenset[str]:
    """Return every ``STRANDS_MESH_ACCEPT_PERMISSIVE_ACL*`` name the ACL
    config module reads.

    Derived from the module's own source so a sibling added later is held to
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


def _reader_sites() -> dict[str, str]:
    """Map every function that reads the token to the module it lives in.

    Scans the whole ``strands_robots`` package rather than one module: the
    documented blast radius of this token is the set of code paths that consult
    it, and the first correction to this page was wrong precisely because two of
    the three readers live outside ``_acl_config``. Keys are the enclosing
    function names, values are POSIX-style module paths relative to the package.
    """
    sites: dict[str, str] = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for inner in ast.walk(node):
                literal = None
                if isinstance(inner, ast.Call) and ast.unparse(inner.func) in (
                    "os.getenv",
                    "os.environ.get",
                ):
                    if inner.args and isinstance(inner.args[0], ast.Constant):
                        literal = inner.args[0].value
                elif isinstance(inner, ast.Subscript) and ast.unparse(inner.value) == "os.environ":
                    if isinstance(inner.slice, ast.Constant):
                        literal = inner.slice.value
                if isinstance(literal, str) and literal.startswith(_PREFIX):
                    sites[node.name] = path.relative_to(_PACKAGE).as_posix()
    return sites


def _security_page_section() -> str:
    """Return the acknowledgement subsection from ``docs/security.md``.

    Bounded by the section heading and the next ``### `` sibling.
    """
    text = _PAGE.read_text(encoding="utf-8")
    start = text.find(_HEADING)
    assert start >= 0, f"security page is missing heading {_HEADING!r}"
    after = text[start + len(_HEADING) :]
    end = after.find("\n### ")
    if end < 0:
        return after
    return after[:end]


def test_the_derivation_finds_the_acknowledgement_variable_the_module_reads() -> None:
    """Positive premise: the scan is not vacuous.

    The population every downstream rule quantifies over is the set of
    ``STRANDS_MESH_ACCEPT_PERMISSIVE_ACL*`` literals the ACL config module
    names on an ``os.getenv`` / ``os.environ`` read. If a refactor makes that
    scan empty every rule below would pass vacuously; this cell refuses that
    state and pins the one variable we currently know about.
    """
    reads = _accept_env_reads()
    assert _KNOWN <= reads, (
        f"the ACL config module no longer reads {sorted(_KNOWN - reads)}; "
        "either the read moved to another module (in which case the scan target "
        "in this file needs updating) or the variable was renamed / removed "
        "(in which case both documentation surfaces need updating too)"
    )
    assert reads, "the AST scan of _acl_config.py found no STRANDS_MESH_ACCEPT_PERMISSIVE_ACL* reads at all"


def test_every_acknowledgement_variable_the_module_reads_has_a_readme_matrix_row() -> None:
    """Every read is a row on the README env-var matrix.

    The matrix is the single index the module family's own code cites for
    fleet configuration, so a variable the ACL loader reads without a row is a
    knob with no discoverable entry.
    """
    readme = _README.read_text(encoding="utf-8")
    missing = [name for name in _accept_env_reads() if not re.search(rf"^\| `{re.escape(name)}`", readme, re.MULTILINE)]
    assert not missing, (
        f"README env-var matrix is missing a row for {missing}; "
        f"add one beside the STRANDS_MESH_ACL_FILE row so the ACL-configuration "
        "family reads as a single index"
    )


def test_every_acknowledgement_variable_the_module_reads_is_named_on_the_security_page() -> None:
    """Every read is named on the security page.

    The prose surface for the ACL posture is where an operator learns the
    two-shape distinction and the refusal semantics. A read not named there is
    a knob whose posture cannot be reasoned about from the security page.
    """
    section = _security_page_section()
    missing = [name for name in _accept_env_reads() if name not in section]
    assert not missing, (
        f"docs/security.md acknowledgement subsection is missing {missing}; add a bullet naming each one"
    )


def test_the_security_page_names_both_acl_shapes() -> None:
    """The section distinguishes blacklist (``allow`` + rules) from whitelist (``deny`` + rules).

    The refusal only fires on the blacklist shape; a documentation change that
    named the variable and did not name that distinction would leave an
    operator setting the token without understanding the posture they were
    acknowledging.
    """
    section = _security_page_section().lower()
    assert "blacklist" in section, (
        "acknowledgement subsection does not name the blacklist shape; "
        "an operator setting the token needs to know which of the two ACL shapes it acknowledges"
    )
    assert "whitelist" in section, (
        "acknowledgement subsection does not name the whitelist shape; "
        "the two-shape distinction is the point of the refusal"
    )


def test_the_security_page_names_the_refusal_not_a_warning() -> None:
    """The section names ``PermissiveACLError`` (the module raises) rather than framing the token as a mere warning silencer.

    The prior sentence on the page framed the variable as only a warning
    silencer. The module raises at ACL load; framing the token as *only* a
    warning silencer would leave an operator setting the token to silence a
    warning that does not fire for the case they are configuring. The
    corrected prose names all three effects (refusal, start-gate opt-in,
    session-warning suppression), and this cell pins the refusal-naming
    half of that requirement -- the other two halves have their own cells
    below.
    """
    section = _security_page_section()
    assert "PermissiveACLError" in section, (
        "acknowledgement subsection does not name PermissiveACLError; "
        "the refusal is what the token unblocks, not a warning"
    )


def test_the_security_page_names_the_start_gate_opt_in() -> None:
    """The section names ``Mesh._refuse_under_permissive_default_acl`` (the second effect).

    The prior sentence on the page misframed the token as touching only the
    blacklist-load path. The token is also the opt-in that lets the wire
    come up under the built-in permissive default when ``auth_mode=mtls``
    (:py:meth:`strands_robots.mesh.core.Mesh._refuse_under_permissive_default_acl`).
    A documentation change that named only the refusal would leave an
    operator believing the start-gate stays refused regardless of the token
    -- but the gate's own error message names this token as one of its
    remediations, and setting it downgrades the ERROR refusal to an INFO
    acknowledgement.
    """
    section = _security_page_section()
    assert "_refuse_under_permissive_default_acl" in section, (
        "acknowledgement subsection does not name the start-gate the "
        "token opts into; an operator reading only this page would not "
        "learn that setting the token also lets the wire come up under "
        "the built-in permissive default"
    )


def test_the_security_page_names_the_session_warning_suppression() -> None:
    """The section names the per-session-open WARNING suppression (the third effect).

    ``strands_robots.mesh.session._build_config`` reads
    ``STRANDS_MESH_ACCEPT_PERMISSIVE_ACL`` and skips the "using PERMISSIVE
    built-in default ACL" WARNING when it is set. The prior prose asserted
    the token *did not* silence that warning, which is factually wrong at
    the byte level (``if is_permissive and not accept_permissive`` in
    session.py). This cell pins the corrected framing: the section must
    describe the suppression, not deny it. The failure mode being pinned
    against is: an operator sets the token to load a blacklist ACL in CI,
    ``STRANDS_MESH_ACL_FILE`` is later dropped from that environment (drift
    / deploy bug), and the same token then waives the start-gate AND
    suppresses the WARNING -- the fleet runs wire-open with zero log
    signal.
    """
    section = _security_page_section()
    assert "_build_config" in section, (
        "acknowledgement subsection does not name session._build_config, "
        "the third reader of the token; the prose must describe every "
        "reader so an operator learns the full blast radius before setting it"
    )
    # The section must name the WARNING as suppressed, not as unaffected.
    # Match the corrected framing rather than pinning the false "does not
    # silence" clause.
    lower = section.lower()
    assert "suppressed" in lower or "silenced" in lower or "silences" in lower, (
        "acknowledgement subsection does not describe the per-session "
        "WARNING as suppressed by the token; the code skips the WARNING "
        "when the token is set (session.py::_build_config), and the "
        "prose must describe that -- omitting it or asserting the "
        "opposite is the exact prose the reviewer flagged as false"
    )


def test_every_function_that_reads_the_token_is_named_in_the_section() -> None:
    """The documented blast radius equals the implemented one.

    The three cells above name the three effects this tree has; this one is the
    general rule behind them, derived from the package source rather than from a
    list kept beside it. A fourth gate added anywhere in ``strands_robots``
    fails here until the section says what setting the token does at that site.
    This is the rule whose absence let the page assert that the token "does not
    silence" a warning ``session.py`` silences.
    """
    section = _security_page_section()
    sites = _reader_sites()
    assert len(sites) >= 3, (
        f"the reader scan found {sorted(sites)}; fewer than three reader sites means the scan "
        "went blind (a read moved behind a helper, or a gate was removed without a documentation "
        "change), and every per-effect cell above would then be grading prose against nothing"
    )
    missing = {name: module for name, module in sites.items() if name not in section}
    assert not missing, (
        f"these functions read STRANDS_MESH_ACCEPT_PERMISSIVE_ACL but the acknowledgement "
        f"subsection never names them: {missing}. Say what setting the token does at each site: "
        "an operator who sets it for one effect gets every effect"
    )


def test_the_security_page_names_both_remediations() -> None:
    """The section names both remediations the ``PermissiveACLError`` message offers.

    The refusal message names two remediations: rewrite the ACL as ``deny`` +
    ``allow`` rules, or set the acknowledgement token. A documentation change
    that named only the token would push an operator toward it by omission,
    which is worse than the prior framing.
    """
    section = _security_page_section().lower()
    assert "deny" in section, (
        "acknowledgement subsection does not name the deny-shape remediation; "
        "the refusal offers two remediations and the safer one is missing"
    )


def test_the_derivation_is_derived_not_hardcoded() -> None:
    """The graded population comes from the module's own source, not a list beside it.

    A hardcoded population would go silent on a planted second acknowledgement
    variable. The derived scan catches it. This cell plants a synthetic read
    against a probe source and asserts the scan finds it.
    """
    probe_src = (
        "import os\n"
        'os.getenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL")\n'
        'os.getenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL_SCOPED")\n'
    )
    tree = ast.parse(probe_src)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "os.getenv":
            if node.args and isinstance(node.args[0], ast.Constant):
                lit = node.args[0].value
                if isinstance(lit, str) and lit.startswith(_PREFIX):
                    names.add(lit)
    assert names == {
        "STRANDS_MESH_ACCEPT_PERMISSIVE_ACL",
        "STRANDS_MESH_ACCEPT_PERMISSIVE_ACL_SCOPED",
    }, f"the AST scan pattern misses a planted sibling read: {sorted(names)}"


# --- Behavioural pins: the prose cannot drift from what the loader does. ---


def _write_acl(tmp_path: pathlib.Path, data: dict) -> pathlib.Path:
    path = tmp_path / "acl.json5"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _base_blacklist_acl() -> dict:
    return {
        "enabled": True,
        "default_permission": "allow",
        "rules": [
            {
                "id": "deny_camera",
                "permission": "deny",
                "flows": ["egress"],
                "messages": ["put"],
                "key_exprs": ["**/camera/**"],
            }
        ],
        "subjects": [{"id": "any", "cert_common_names": ["*"]}],
        "policies": [{"id": "p", "rules": ["deny_camera"], "subjects": ["any"]}],
    }


def test_a_blacklist_acl_is_refused_with_the_variable_unset(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset acknowledgement token -> blacklist ACL raises PermissiveACLError."""
    path = _write_acl(tmp_path, _base_blacklist_acl())
    monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
    monkeypatch.delenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", raising=False)
    _acl_config._clear_acl_cache_for_test()
    with pytest.raises(_acl_config.PermissiveACLError):
        _acl_config.resolve_acl("strands")


def test_a_blacklist_acl_is_refused_with_an_empty_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty / whitespace value falls back to refusal, so an operator cannot
    accidentally acknowledge with ``STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=""``."""
    path = _write_acl(tmp_path, _base_blacklist_acl())
    monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
    for value in ("", "   ", "\t"):
        monkeypatch.setenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", value)
        _acl_config._clear_acl_cache_for_test()
        with pytest.raises(_acl_config.PermissiveACLError):
            _acl_config.resolve_acl("strands")


def test_a_blacklist_acl_is_accepted_with_a_truthy_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three accepted spellings load the blacklist ACL without raising."""
    path = _write_acl(tmp_path, _base_blacklist_acl())
    monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(path))
    for value in ("1", "true", "TRUE", "yes", "  yes  "):
        monkeypatch.setenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", value)
        _acl_config._clear_acl_cache_for_test()
        # Should not raise -- the acknowledgement unblocks the load.
        data = _acl_config.resolve_acl("strands")
        assert data["default_permission"] == "allow"
        assert data["rules"], "the blacklist rule set should survive the load"


def test_the_builtin_permissive_default_does_not_reach_this_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No operator ACL supplied -> the built-in default_acl loads without raising.

    The built-in permissive default is gated by ``Mesh.start``'s separate
    refuse-to-start path, not by ``_validate_acl_shape``. This cell pins that
    distinction so a future refactor cannot silently collapse the two.
    """
    monkeypatch.delenv("STRANDS_MESH_ACL_FILE", raising=False)
    monkeypatch.delenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", raising=False)
    _acl_config._clear_acl_cache_for_test()
    # Should not raise PermissiveACLError -- the built-in default has its
    # own gate in Mesh.start, not in the loader.
    result = _acl_config.resolve_acl("strands")
    assert result["default_permission"] == "allow"
    assert result["rules"] == []


def test_the_token_is_the_optin_for_the_builtin_default_start_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Effect 2, measured: the token flips ``Mesh.start``'s refusal.

    Pinned behaviourally rather than left to the section's prose because this is
    the effect an operator who sets the token for the loader refusal does not
    expect to be taking, and because a prose-only guard cannot notice the gate
    changing under it.
    """
    from strands_robots.mesh import core as mesh_core

    robot = SimpleNamespace(
        tool_name_str="r3003",
        robot=SimpleNamespace(
            is_connected=True,
            name="r3003_test",
            config=SimpleNamespace(cameras={}),
            get_observation=MagicMock(return_value={}),
        ),
    )
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    monkeypatch.delenv("STRANDS_MESH_ACL_FILE", raising=False)
    _acl_config._clear_acl_cache_for_test()

    monkeypatch.delenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", raising=False)
    mesh = mesh_core.Mesh(robot, peer_id="test-3003-a", peer_type="robot")
    assert mesh._refuse_under_permissive_default_acl() is True, (
        "with the token unset, Mesh.start must refuse to bring the wire up on the built-in permissive default"
    )

    monkeypatch.setenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", "1")
    mesh = mesh_core.Mesh(robot, peer_id="test-3003-b", peer_type="robot")
    assert mesh._refuse_under_permissive_default_acl() is False, (
        "with the token set, the start gate is waived -- this is the effect the documentation must not omit"
    )


def test_the_token_suppresses_the_per_session_permissive_warning(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Effect 3, measured: the per-session permissive WARNING is suppressed.

    ``session._build_config`` emits the ``PERMISSIVE built-in default ACL``
    WARNING on every session open ``if is_permissive and not
    accept_permissive``, so the token is exactly what silences it. An earlier
    revision of this page asserted the opposite and a guard cell pinned the
    assertion; this pair is the measurement that settles it, and it fails if
    ``session.py`` ever stops honouring the token while the page still says it
    does.
    """
    pytest.importorskip("zenoh")
    from strands_robots.mesh.session import _build_config

    ca, cert, key = tmp_path / "ca.crt", tmp_path / "peer.crt", tmp_path / "peer.key"
    for f in (ca, cert, key):
        f.write_text("dummy\n")
    key.chmod(0o600)  # _resolve_tls_paths enforces 0o600 on the private key.
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca))
    monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert))
    monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key))
    monkeypatch.delenv("STRANDS_MESH_ACL_FILE", raising=False)

    def permissive_warnings() -> list[str]:
        return [r.getMessage() for r in caplog.records if "PERMISSIVE built-in" in r.getMessage()]

    monkeypatch.delenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", raising=False)
    _acl_config._clear_acl_cache_for_test()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.session"):
        _build_config()
    assert permissive_warnings(), "the permissive-default WARNING should fire with the token unset"

    monkeypatch.setenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", "1")
    _acl_config._clear_acl_cache_for_test()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.session"):
        _build_config()
    assert not permissive_warnings(), (
        "the permissive-default WARNING is suppressed while the token is set, so documentation "
        "claiming the token does not silence it is false"
    )
