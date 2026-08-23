"""A continuable refusal is recognised by its code, never by its prose.

A refusal an operator can answer -- an untrusted provider, a repo or host or
policy type outside an allowlist, a teleop frame past the value envelope -- has
to be recognised by whoever offers that choice. Before the codes the only thing
on offer was the message text, so recognition meant prose matching: an env-var
name appearing somewhere in the sentence, the subject pulled back out with an
anchored regex.

That made every wording improvement a silent breaking change, and this package
could not detect it: rewording the two teleop refusals leaves the whole mesh
suite green (measured), because nothing here asserts on those phrases at all.

These tests pin the contract in three parts.

* Each continuable refusal carries its ``code`` and its ``subject``
  structurally, so a consumer switches on identity.
* The message is *unchanged* by that -- the codes are additive, and the prose
  stays free to improve. ``test_the_code_does_not_come_from_the_message``
  states the half that matters: an unrecognisable message keeps its code.
* ``test_every_refusal_naming_an_operator_grant_carries_a_code`` finds the sites
  it must grade by the *exception type* they raise -- the types whose
  constructor accepts a ``code`` at all -- and then asks whether the message
  names an environment variable. Scoping it by the grants already registered
  would only ever find a refusal naming a grant that is already coded, so the
  first refusal to offer a *new* grant would be invisible; that is the one case
  this rule exists for, and
  ``test_a_refusal_offering_an_unregistered_grant_is_reported`` is it.

A rejection with no operator grant behind it stays code-less on purpose: a
schema or bounds failure gives a consumer nothing to offer, so there is nothing
to recognise. ``test_a_rejection_with_no_grant_behind_it_stays_code_less`` is
the boundary.
"""

from __future__ import annotations

import ast
import pathlib
import re
from typing import Any

import pytest

from strands_robots.mesh.security import ValidationError, validate_command, validate_input_frame
from strands_robots.policies.factory import UntrustedRemoteCodeError, _check_trust_remote_code

_PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "strands_robots"

#: Any ``STRANDS_*`` variable a refusal message names. A refusal of a
#: code-carrying type that names one is treated as an offer to the operator.
_ENV_VAR = re.compile(r"\bSTRANDS_[A-Z0-9_]+\b")

#: Grant env vars are cleared per test so the refusals under test actually fire.
_GRANT_ENV = (
    "STRANDS_TRUST_REMOTE_CODE",
    "STRANDS_MESH_HF_REPO_ALLOW",
    "STRANDS_MESH_POLICY_TYPE_ALLOW",
    "STRANDS_MESH_POLICY_HOST_ALLOW",
    "STRANDS_MESH_INPUT_VALUE_ABS",
)


@pytest.fixture(autouse=True)
def _no_grants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse by default: an inherited grant would silence the refusal."""
    for name in _GRANT_ENV:
        monkeypatch.delenv(name, raising=False)


def _command(**overrides: Any) -> dict[str, Any]:
    cmd: dict[str, Any] = {
        "action": "execute",
        "instruction": "pick up the cube",
        "sender": "operator",
        "policy_provider": "mock",
    }
    cmd.update(overrides)
    return cmd


def _refused(fn: Any) -> ValidationError | UntrustedRemoteCodeError:
    """Return the refusal ``fn`` raises, or fail naming what it did instead."""
    try:
        result = fn()
    except (ValidationError, UntrustedRemoteCodeError) as refusal:
        return refusal
    raise AssertionError(f"expected a refusal, got {result!r}")


class TestEveryContinuableRefusalCarriesItsCodeAndSubject:
    """The five codes, at the six sites that raise them."""

    def test_an_untrusted_provider_names_itself(self) -> None:
        refusal = _refused(lambda: _check_trust_remote_code("lerobot_local"))
        assert refusal.code == "TRUST_REMOTE_CODE_REQUIRED"
        assert refusal.subject == "lerobot_local"

    def test_a_repo_outside_the_allowlist_names_the_repo(self) -> None:
        refusal = _refused(lambda: validate_command(_command(pretrained_name_or_path="sketchy-org/model")))
        assert refusal.code == "HF_REPO_NOT_ALLOWED"
        assert refusal.subject == "sketchy-org/model"

    def test_a_policy_type_outside_the_allowlist_names_the_type(self) -> None:
        refusal = _refused(lambda: validate_command(_command(policy_type="mystery_net")))
        assert refusal.code == "POLICY_TYPE_NOT_ALLOWED"
        assert refusal.subject == "mystery_net"

    def test_a_provider_outside_the_allowlist_shares_the_policy_type_code(self) -> None:
        """Provider and policy_type share one allowlist, so they share one code."""
        refusal = _refused(lambda: validate_command(_command(policy_provider="mystery_prov")))
        assert refusal.code == "POLICY_TYPE_NOT_ALLOWED"
        assert refusal.subject == "mystery_prov"

    def test_a_host_outside_the_allowlist_names_the_host(self) -> None:
        refusal = _refused(lambda: validate_command(_command(policy_host="10.9.9.9")))
        assert refusal.code == "POLICY_HOST_NOT_ALLOWED"
        assert refusal.subject == "10.9.9.9"

    def test_a_server_address_outside_the_allowlist_names_the_address(self) -> None:
        """The subject is the whole address: the host was derived from it."""
        refusal = _refused(lambda: validate_command(_command(server_address="tcp://10.9.9.9:5555")))
        assert refusal.code == "POLICY_HOST_NOT_ALLOWED"
        assert refusal.subject == "tcp://10.9.9.9:5555"

    def test_a_teleop_frame_past_the_envelope_names_the_joint(self) -> None:
        """This refusal names no env var, so its code is the only handle on it."""
        refusal = _refused(lambda: validate_input_frame({"shoulder.pos": 900.0}))
        assert refusal.code == "TELEOP_VALUE_OUT_OF_RANGE"
        assert refusal.subject == "shoulder.pos"


class TestTheCodesAreAdditive:
    """Adding a code changed no message and refused nothing new."""

    def test_the_refusal_messages_are_unchanged(self) -> None:
        host = str(_refused(lambda: validate_command(_command(policy_host="10.9.9.9"))))
        teleop = str(_refused(lambda: validate_input_frame({"shoulder.pos": 900.0})))
        assert host == "policy_host='10.9.9.9' not in allowlist. Set STRANDS_MESH_POLICY_HOST_ALLOW to extend."
        assert teleop == "input frame value for 'shoulder.pos' out of range: |900.0| > 720.0"

    def test_a_well_formed_command_is_still_accepted(self) -> None:
        assert validate_command(_command())["instruction"] == "pick up the cube"

    def test_a_message_only_rejection_still_constructs(self) -> None:
        """The ~70 rejections that pass only a message keep working, code-less."""
        refusal = ValidationError("something the operator cannot grant")
        assert refusal.code is None
        assert refusal.subject is None
        assert str(refusal) == "something the operator cannot grant"

    def test_the_code_does_not_come_from_the_message(self) -> None:
        """The contract survives a message a prose matcher cannot recognise."""
        refusal = ValidationError("", code="TELEOP_VALUE_OUT_OF_RANGE", subject="shoulder.pos")
        assert refusal.code == "TELEOP_VALUE_OUT_OF_RANGE"
        assert refusal.subject == "shoulder.pos"


class TestTheBoundaryIsWhetherAnOperatorCanGrantSomething:
    """A refusal with nothing to grant has nothing for a consumer to offer."""

    def test_a_rejection_with_no_grant_behind_it_stays_code_less(self) -> None:
        refusal = _refused(lambda: validate_command(_command(instruction="x" * 99999)))
        assert refusal.code is None
        assert refusal.subject is None

    def test_a_lockout_is_a_security_error_and_is_not_code_less_by_accident(self) -> None:
        """LockoutError inherits the attributes; nothing here claims a code."""
        from strands_robots.mesh.security import LockoutError

        refusal = LockoutError("estop engaged")
        assert refusal.code is None


class TestTheRegistryIsAClosedVocabulary:
    """A consumer may switch on the codes, so they must be unique and complete."""

    def test_the_codes_are_unique(self) -> None:
        from strands_robots import refusal_codes

        assert len(set(refusal_codes.REFUSAL_CODES)) == len(refusal_codes.REFUSAL_CODES)

    def test_every_code_names_the_grant_that_lifts_it(self) -> None:
        from strands_robots import refusal_codes

        assert set(refusal_codes.REFUSAL_GRANTS) == set(refusal_codes.REFUSAL_CODES)
        assert all(g.startswith("STRANDS_") for g in refusal_codes.REFUSAL_GRANTS.values())

    def test_every_code_is_exported_under_its_own_name(self) -> None:
        from strands_robots import refusal_codes

        for code in refusal_codes.REFUSAL_CODES:
            assert getattr(refusal_codes, code) == code, code
            assert code in refusal_codes.__all__

    def test_the_literals_the_site_tests_assert_are_the_declared_codes(self) -> None:
        """The per-site tests spell their codes out; this is what ties them here."""
        from strands_robots import refusal_codes

        assert {
            "TRUST_REMOTE_CODE_REQUIRED",
            "HF_REPO_NOT_ALLOWED",
            "POLICY_TYPE_NOT_ALLOWED",
            "POLICY_HOST_NOT_ALLOWED",
            "TELEOP_VALUE_OUT_OF_RANGE",
        } == set(refusal_codes.REFUSAL_CODES)

    def test_every_declared_code_is_actually_raised_somewhere(self) -> None:
        """A code no site raises is a vocabulary entry a consumer waits for forever."""
        from strands_robots import refusal_codes

        raised = {code for _, _, code in _coded_raise_sites()}
        assert raised == set(refusal_codes.REFUSAL_CODES)


def _string_literals(node: ast.AST) -> str:
    """Every string literal under ``node``, including f-string parts."""
    return " ".join(n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str))


def _raise_sites() -> list[tuple[str, int, ast.expr]]:
    """Every ``raise <expr>`` in the package, as (path, line, the expression)."""
    sites: list[tuple[str, int, ast.expr]] = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = str(path.relative_to(_PACKAGE.parent))
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and node.exc is not None:
                sites.append((rel, node.lineno, node.exc))
    return sites


def _imported_code_names(source: str) -> dict[str, str]:
    """Names a module binds with ``from ...refusal_codes import X``, to their code.

    A code reached that way is a bare :class:`ast.Name` at the raise site, so
    resolving it needs the module's own import list. The mapping is
    ``bound name -> declared name`` so an ``as`` alias resolves to the code it
    aliases rather than to a name this package does not declare.
    """
    bound: dict[str, str] = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("refusal_codes"):
            for alias in node.names:
                bound[alias.asname or alias.name] = alias.name
    return bound


def _code_identifier(value: ast.expr, imported: dict[str, str]) -> str | None:
    """The declared-code name a ``code=`` argument states, or ``None`` if unreadable.

    Three spellings reach a declared code, and
    ``test_every_code_is_exported_under_its_own_name`` is what makes them one
    channel: every code is exported under its own name, so the module attribute,
    the imported name and the string literal are all that same name.

    ``None`` means the value is not statically readable -- a parameter forwarded
    by a shared helper, a table lookup -- so nothing here can say whether the
    code that reaches a consumer is declared.
    """
    if isinstance(value, ast.Attribute):
        return value.attr
    if isinstance(value, ast.Name):
        return imported.get(value.id)
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _code_keyword_sites(sources: list[tuple[str, str]]) -> list[tuple[str, int, str | None, str]]:
    """Every ``code=`` at a raise site: (path, line, declared name or ``None``, as written).

    Read spelling-independently, the way :func:`_refusals_naming_an_env_var`
    already reads the *presence* of the keyword. That rule accepts a code this
    package does not declare on purpose, because grading the value belongs
    here -- so a spelling this scan cannot read is a code nothing checks.
    """
    found: list[tuple[str, int, str | None, str]] = []
    for rel, source in sources:
        imported = _imported_code_names(source)
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
                continue
            for keyword in node.exc.keywords:
                if keyword.arg == "code":
                    found.append(
                        (rel, node.lineno, _code_identifier(keyword.value, imported), ast.unparse(keyword.value))
                    )
    return found


def _coded_raise_sites() -> list[tuple[str, int, str]]:
    """Every package raise passing a readable ``code=``, with the code it names."""
    return [(rel, lineno, name) for rel, lineno, name, _ in _code_keyword_sites(_package_sources()) if name]


def _undeclared_codes_in(sources: list[tuple[str, str]]) -> list[str]:
    """Sites passing a readable ``code=`` that is not a declared code, as messages."""
    from strands_robots import refusal_codes

    return [
        f"{rel}:{lineno} passes code={written} -> {name!r}"
        for rel, lineno, name, written in _code_keyword_sites(sources)
        if name is not None and getattr(refusal_codes, name, None) not in refusal_codes.REFUSAL_CODES
    ]


def _unreadable_codes_in(sources: list[tuple[str, str]]) -> list[str]:
    """Sites whose ``code=`` value this scan cannot resolve, as messages."""
    return [
        f"{rel}:{lineno} passes code={written}"
        for rel, lineno, name, written in _code_keyword_sites(sources)
        if name is None
    ]


def _package_sources() -> list[tuple[str, str]]:
    """Every shipped module, as (path-for-messages, source)."""
    return [
        (str(path.relative_to(_PACKAGE.parent)), path.read_text(encoding="utf-8"))
        for path in sorted(_PACKAGE.rglob("*.py"))
    ]


def _code_carrying_exception_types() -> frozenset[str]:
    """Exception class names that can carry a refusal code, derived from the package.

    A class whose ``__init__`` accepts a ``code`` keyword defines the contract;
    anything inheriting one carries it. Scoping the rule below to these is what
    keeps it honest in both directions: a builtin ``ValueError`` has nowhere to
    put a code, so demanding one would be a ``TypeError`` rather than a finding,
    and a class added later that does accept one is graded without an edit here.
    """
    defines: set[str] = set()
    bases: dict[str, set[str]] = {}
    for _rel, source in _package_sources():
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef):
                continue
            named_bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
            named_bases |= {b.attr for b in node.bases if isinstance(b, ast.Attribute)}
            bases.setdefault(node.name, set()).update(named_bases)
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "__init__":
                    params = [a.arg for a in member.args.args] + [a.arg for a in member.args.kwonlyargs]
                    if "code" in params:
                        defines.add(node.name)
    carriers = set(defines)
    while True:
        inheriting = {name for name, parents in bases.items() if parents & carriers} - carriers
        if not inheriting:
            return frozenset(carriers)
        carriers |= inheriting


def _refusals_naming_an_env_var(sources: list[tuple[str, str]]) -> list[tuple[str, int, list[str], bool]]:
    """Raise sites of a code-carrying type whose message names a ``STRANDS_*`` variable.

    Naming one reads as an offer -- *set this and the identical request
    succeeds* -- which is exactly what makes a refusal continuable. The match is
    on any such variable rather than on a phrasing like ``"Set X"``, so a
    reworded offer is still graded; a variable that is merely the *source* of a
    bad value is caught too, and the remedy there is to reword rather than to
    invent a code.
    """
    carriers = _code_carrying_exception_types()
    found: list[tuple[str, int, list[str], bool]] = []
    for rel, source in sources:
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            exc = node.exc
            if not isinstance(exc, ast.Call):
                continue
            raised = exc.func.id if isinstance(exc.func, ast.Name) else getattr(exc.func, "attr", None)
            if raised not in carriers:
                continue
            named = sorted(set(_ENV_VAR.findall(_string_literals(exc))))
            if not named:
                continue
            found.append((rel, node.lineno, named, any(k.arg == "code" for k in exc.keywords)))
    return found


def _uncoded_refusals_naming_an_env_var(sources: list[tuple[str, str]]) -> list[str]:
    """The offending subset of :func:`_refusals_naming_an_env_var`, as messages."""
    return [
        f"{rel}:{lineno} names {named} but carries no code"
        for rel, lineno, named, has_code in _refusals_naming_an_env_var(sources)
        if not has_code
    ]


class TestTheContractReachesEveryGrantBearingRefusal:
    """Scoped by what can carry a code, so a refusal offering a new grant is graded."""

    def test_every_refusal_naming_an_operator_grant_carries_a_code(self) -> None:
        uncoded = _uncoded_refusals_naming_an_env_var(_package_sources())
        assert not uncoded, (
            "a refusal that tells an operator which environment variable lifts it is "
            "continuable, so a consumer must be able to recognise it without reading "
            "the prose. Either pass a code from strands_robots.refusal_codes, or -- if "
            "that variable is not something an operator can set to make this identical "
            "request succeed -- reword the message so it does not read as an offer: " + "; ".join(uncoded)
        )

    def test_a_refusal_offering_an_unregistered_grant_is_reported(self) -> None:
        """The case the rule exists for: a grant this package does not yet know.

        Scoping the scan by :data:`~strands_robots.refusal_codes.REFUSAL_GRANTS`
        cannot see this, because the variable is absent from that table until
        someone adds the code -- which is the very thing being forgotten.
        """
        planted = (
            "def _sink_gate(value: str) -> None:\n"
            "    raise ValidationError(\n"
            '        f"telemetry_sink={value!r} not in allowlist. '
            'Set STRANDS_MESH_SINK_ALLOW to extend."\n'
            "    )\n"
        )
        reported = _uncoded_refusals_naming_an_env_var([("planted.py", planted)])
        assert len(reported) == 1, reported
        assert "STRANDS_MESH_SINK_ALLOW" in reported[0]

    def test_a_coded_refusal_offering_a_new_grant_is_accepted(self) -> None:
        """The rule asks for a code, not for membership of the grant table."""
        planted = (
            "def _sink_gate(value: str) -> None:\n"
            "    raise ValidationError(\n"
            '        f"telemetry_sink={value!r} not in allowlist. '
            'Set STRANDS_MESH_SINK_ALLOW to extend.",\n'
            "        code=refusal_codes.SINK_NOT_ALLOWED,\n"
            "        subject=value,\n"
            "    )\n"
        )
        assert _uncoded_refusals_naming_an_env_var([("planted.py", planted)]) == []

    def test_a_refusal_that_cannot_carry_a_code_is_out_of_scope(self) -> None:
        """A builtin has no ``code`` parameter, so requiring one would be a TypeError.

        The package refuses a permissive ACL, an unacknowledged ``AUTH_MODE=none``
        and a CA-pin timeout with a ``RuntimeError``/``ValueError``/``OSError``,
        each naming the variable that lifts it. Giving those a code means giving
        them a code-carrying type first, which is a separate change.
        """
        planted = (
            "def _mode_gate(mode: str) -> None:\n"
            '    raise ValueError("set STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1 to confirm")\n'
        )
        assert _uncoded_refusals_naming_an_env_var([("planted.py", planted)]) == []

    def test_the_scan_reaches_the_package_and_the_shipped_refusals(self) -> None:
        """Non-vacuity: a scan reaching nothing would pass the rule above."""
        naming = _refusals_naming_an_env_var(_package_sources())
        assert len(naming) >= 6, naming
        assert len(_raise_sites()) >= 200, "the package-wide raise scan reached almost nothing"

    def test_the_graded_types_are_derived_from_the_package(self) -> None:
        """Non-vacuity for the scope: an empty carrier set grades nothing."""
        carriers = _code_carrying_exception_types()
        assert {"SecurityError", "ValidationError", "UntrustedRemoteCodeError"} <= carriers, carriers
        assert "LockoutError" in carriers, "a subclass inherits the constructor, so it carries the contract"
        assert "ValueError" not in carriers and "RuntimeError" not in carriers, carriers

    def test_every_code_passed_at_a_raise_site_is_a_declared_code(self) -> None:
        undeclared = _undeclared_codes_in(_package_sources())
        assert not undeclared, (
            "a code outside REFUSAL_CODES reaches a consumer that switches on the closed "
            "vocabulary, which has no branch for it and no grant to offer: " + "; ".join(undeclared)
        )


class TestTheCodeValueIsReadInEverySpellingThatReachesIt:
    """A ``code=`` this scan cannot read is a code nothing checks.

    :func:`_refusals_naming_an_env_var` reads the *presence* of the keyword
    spelling-independently, and
    ``test_a_coded_refusal_offering_a_new_grant_is_accepted`` accepts a code this
    package does not declare on purpose -- grading the value belongs to
    ``test_every_code_passed_at_a_raise_site_is_a_declared_code``. So that rule
    has to read the value in every spelling that reaches a declared code, and
    ``test_every_code_is_exported_under_its_own_name`` is what makes them one
    channel: each code is exported under its own name and listed in ``__all__``,
    so importing it directly is a first-class way to reach it.

    Nothing validates the value at runtime, and nothing should: ``code`` is
    stored as given (``SecurityError.__init__`` assigns it), and raising from an
    exception constructor would replace a security refusal with a constructor
    error. This scan is the only thing standing between a mistyped code and a
    consumer, so its reach is the contract.
    """

    _MODULE_ATTRIBUTE = (
        "from strands_robots import refusal_codes\n"
        "def _gate(value: str) -> None:\n"
        "    raise ValidationError(\n"
        '        f"sink={value!r} not in allowlist. Set STRANDS_MESH_SINK_ALLOW to extend.",\n'
        "        code=refusal_codes.<CODE>,\n"
        "    )\n"
    )
    _IMPORTED_NAME = (
        "from strands_robots.refusal_codes import <CODE>\n"
        "def _gate(value: str) -> None:\n"
        "    raise ValidationError(\n"
        '        f"sink={value!r} not in allowlist. Set STRANDS_MESH_SINK_ALLOW to extend.",\n'
        "        code=<CODE>,\n"
        "    )\n"
    )
    _STRING_LITERAL = (
        "def _gate(value: str) -> None:\n"
        "    raise ValidationError(\n"
        '        f"sink={value!r} not in allowlist. Set STRANDS_MESH_SINK_ALLOW to extend.",\n'
        '        code="<CODE>",\n'
        "    )\n"
    )

    @pytest.mark.parametrize(
        "spelling",
        ["_MODULE_ATTRIBUTE", "_IMPORTED_NAME", "_STRING_LITERAL"],
    )
    def test_a_declared_code_is_read_however_the_site_spells_it(self, spelling: str) -> None:
        planted = getattr(self, spelling).replace("<CODE>", "HF_REPO_NOT_ALLOWED")
        read = _code_keyword_sites([("planted.py", planted)])
        assert [name for _rel, _line, name, _written in read] == ["HF_REPO_NOT_ALLOWED"], read
        assert _undeclared_codes_in([("planted.py", planted)]) == []

    @pytest.mark.parametrize(
        "spelling",
        ["_MODULE_ATTRIBUTE", "_IMPORTED_NAME", "_STRING_LITERAL"],
    )
    def test_a_mistyped_code_is_reported_however_the_site_spells_it(self, spelling: str) -> None:
        """The regression: a typo in any spelling ships an undeclared code."""
        planted = getattr(self, spelling).replace("<CODE>", "HF_REPO_NOT_ALOWED")
        reported = _undeclared_codes_in([("planted.py", planted)])
        assert len(reported) == 1, reported
        assert "HF_REPO_NOT_ALOWED" in reported[0]

    def test_an_aliased_import_resolves_to_the_code_it_aliases(self) -> None:
        """``import X as Y`` must not read as a code named ``Y``, which is undeclared."""
        planted = (
            "from strands_robots.refusal_codes import HF_REPO_NOT_ALLOWED as REPO_CODE\n"
            "def _gate(value: str) -> None:\n"
            '    raise ValidationError("nope", code=REPO_CODE)\n'
        )
        read = _code_keyword_sites([("planted.py", planted)])
        assert [name for _rel, _line, name, _written in read] == ["HF_REPO_NOT_ALLOWED"], read
        assert _undeclared_codes_in([("planted.py", planted)]) == []

    def test_a_code_this_scan_cannot_read_is_reported_not_skipped(self) -> None:
        """A forwarded parameter is not gradeable here, so silence would be a hole."""
        planted = "def _refuse(message: str, code: str) -> None:\n    raise ValidationError(message, code=code)\n"
        assert _undeclared_codes_in([("planted.py", planted)]) == []
        unreadable = _unreadable_codes_in([("planted.py", planted)])
        assert len(unreadable) == 1, unreadable
        assert "code=code" in unreadable[0]

    def test_every_code_keyword_in_the_package_is_read_or_reported(self) -> None:
        """Root cause: a keyword neither graded nor reported is one nothing checks."""
        sources = _package_sources()
        every = _code_keyword_sites(sources)
        graded = [row for row in every if row[2] is not None]
        assert len(graded) + len(_unreadable_codes_in(sources)) == len(every)
        assert len(every) >= 7, every

    def test_the_shipped_sites_are_all_read(self) -> None:
        """Nothing in the package is unreadable today, so the scan grades all of it."""
        assert _unreadable_codes_in(_package_sources()) == []
        assert {name for _rel, _line, name in _coded_raise_sites()} == {
            "TRUST_REMOTE_CODE_REQUIRED",
            "HF_REPO_NOT_ALLOWED",
            "POLICY_TYPE_NOT_ALLOWED",
            "POLICY_HOST_NOT_ALLOWED",
            "TELEOP_VALUE_OUT_OF_RANGE",
        }
