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
* ``test_every_refusal_naming_an_operator_grant_carries_a_code`` derives the
  sites it must find from :data:`~strands_robots.refusal_codes.REFUSAL_GRANTS`
  rather than a hand-written list, so a seventh grant-bearing refusal added
  later is graded on arrival.

A rejection with no operator grant behind it stays code-less on purpose: a
schema or bounds failure gives a consumer nothing to offer, so there is nothing
to recognise. ``test_a_rejection_with_no_grant_behind_it_stays_code_less`` is
the boundary.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

import pytest

from strands_robots.mesh.security import ValidationError, validate_command, validate_input_frame
from strands_robots.policies.factory import UntrustedRemoteCodeError, _check_trust_remote_code

_PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "strands_robots"

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


def _coded_raise_sites() -> list[tuple[str, int, str]]:
    """Every raise passing ``code=<refusal_codes.NAME>``, with that name."""
    found: list[tuple[str, int, str]] = []
    for rel, lineno, exc in _raise_sites():
        if not isinstance(exc, ast.Call):
            continue
        for keyword in exc.keywords:
            if keyword.arg == "code" and isinstance(keyword.value, ast.Attribute):
                found.append((rel, lineno, keyword.value.attr))
    return found


class TestTheContractReachesEveryGrantBearingRefusal:
    """Derived from the registry, so a new grant-bearing refusal is graded."""

    def test_every_refusal_naming_an_operator_grant_carries_a_code(self) -> None:
        from strands_robots import refusal_codes

        grants = set(refusal_codes.REFUSAL_GRANTS.values())
        uncoded = []
        for rel, lineno, exc in _raise_sites():
            text = _string_literals(exc)
            named = sorted(g for g in grants if g in text)
            if not named:
                continue
            has_code = isinstance(exc, ast.Call) and any(k.arg == "code" for k in exc.keywords)
            if not has_code:
                uncoded.append(f"{rel}:{lineno} names {named} but carries no code")
        assert not uncoded, (
            "a refusal that tells an operator which grant lifts it is continuable, "
            "so a consumer must be able to recognise it without reading the prose: " + "; ".join(uncoded)
        )

    def test_the_scan_found_the_grant_bearing_refusals(self) -> None:
        """Non-vacuity: a scan reaching nothing would pass the rule above."""
        from strands_robots import refusal_codes

        grants = set(refusal_codes.REFUSAL_GRANTS.values())
        naming = [
            (rel, lineno) for rel, lineno, exc in _raise_sites() if any(g in _string_literals(exc) for g in grants)
        ]
        assert len(naming) >= 6, naming
        assert len(_raise_sites()) >= 200, "the package-wide raise scan reached almost nothing"

    def test_every_code_passed_at_a_raise_site_is_a_declared_code(self) -> None:
        from strands_robots import refusal_codes

        for rel, lineno, name in _coded_raise_sites():
            assert getattr(refusal_codes, name, None) in refusal_codes.REFUSAL_CODES, f"{rel}:{lineno} {name}"
