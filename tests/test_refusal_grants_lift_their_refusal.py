"""A grant this package names is one that lifts the refusal it is mapped to.

:data:`~strands_robots.refusal_codes.REFUSAL_GRANTS` makes a behavioural claim
about every entry: setting that environment variable is what lets an operator
who accepts the risk make the identical request succeed. A consumer offering
that choice reads the variable from there rather than hard-coding it, precisely
so the two cannot drift apart.

Nothing drove that claim. The test named after it asserted that the map's keys
are the declared codes and that each value starts with ``STRANDS_`` -- the
spelling of the variable, not its effect. So the mapping was graded for
membership and prefix, never for whether each code is paired with the variable
that lifts *that* refusal.

Measured on the mapping as shipped, swapping the ``POLICY_TYPE_NOT_ALLOWED``
and ``POLICY_HOST_NOT_ALLOWED`` entries with each other -- so a consumer
recognising a refused policy type is told to extend the *host* allowlist --
leaves the existing suite at 21 passed. Applying the swapped grant is a no-op,
and the refusal returns the identical message, so neither the package nor the
consumer can tell. The prose the codes exist to replace stays correct in that
state while the structured contract that replaced it is wrong.

These tests drive the claim instead:

* every grant lifts the refusal it is mapped to;
* no *other* code's grant lifts it, which is what a permutation trips;
* the graded set is :data:`REFUSAL_GRANTS` itself, so a sixth code is driven on
  arrival rather than added to a list here.

``test_the_subject_is_the_grant_value_only_for_an_allowlist`` is the boundary
that explains why each code has to state its own operation: three grants are
allowlists the refusal's ``subject`` is appended to, and two are not. Applying
the subject to those two is silent -- the identical message comes back -- so a
consumer holding only ``code`` and ``subject`` cannot act on them correctly.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from strands_robots import refusal_codes
from strands_robots.mesh.security import ValidationError, validate_command, validate_input_frame
from strands_robots.policies.factory import UntrustedRemoteCodeError, _check_trust_remote_code

_MODULE = pathlib.Path(refusal_codes.__file__)

#: A complete command envelope, so a refusal under test is the only one that can fire.
_ENVELOPE = {"action": "execute", "instruction": "go", "policy_provider": "mock"}

#: The magnitude the teleop frame commands, which is what its grant is raised past.
_REFUSED_MAGNITUDE = 1500.0

_REFUSALS = (ValidationError, UntrustedRemoteCodeError)


@dataclass(frozen=True)
class _Case:
    """One continuable refusal: how to provoke it, and how to grant past it."""

    drive: Callable[[], object]
    #: The value its grant must be set to, given the refusal's own ``subject``.
    grant_value: Callable[[str], str]


_CASES: dict[str, _Case] = {
    refusal_codes.TRUST_REMOTE_CODE_REQUIRED: _Case(
        drive=lambda: _check_trust_remote_code("lerobot_local"),
        grant_value=lambda subject: "1",
    ),
    refusal_codes.HF_REPO_NOT_ALLOWED: _Case(
        drive=lambda: validate_command({**_ENVELOPE, "pretrained_name_or_path": "evilorg/backdoor"}),
        grant_value=lambda subject: subject,
    ),
    refusal_codes.POLICY_TYPE_NOT_ALLOWED: _Case(
        drive=lambda: validate_command({**_ENVELOPE, "policy_type": "sketchy_net"}),
        grant_value=lambda subject: subject,
    ),
    refusal_codes.POLICY_HOST_NOT_ALLOWED: _Case(
        drive=lambda: validate_command({**_ENVELOPE, "policy_host": "10.9.9.9"}),
        grant_value=lambda subject: subject,
    ),
    refusal_codes.TELEOP_VALUE_OUT_OF_RANGE: _Case(
        drive=lambda: validate_input_frame({"shoulder_pan.pos": _REFUSED_MAGNITUDE}),
        grant_value=lambda subject: str(_REFUSED_MAGNITUDE + 1.0),
    ),
}


@pytest.fixture(autouse=True)
def _no_grants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every grant cleared: an inherited one would silence the refusal under test."""
    for name in refusal_codes.REFUSAL_GRANTS.values():
        monkeypatch.delenv(name, raising=False)


def _refuse(code: str) -> tuple[str, str]:
    """Provoke *code*'s refusal, returning its ``subject`` and message."""
    try:
        _CASES[code].drive()
    except _REFUSALS as exc:
        assert exc.code == code, f"{code}: refused as {exc.code!r}"
        assert exc.subject is not None, f"{code}: carries no subject"
        return str(exc.subject), str(exc)
    raise AssertionError(f"{code}: no grant is set, so the request must be refused")


def _lifted(code: str) -> bool:
    """Whether *code*'s request now succeeds."""
    try:
        _CASES[code].drive()
    except _REFUSALS:
        return False
    return True


def _grant_block(code: str) -> str:
    """The ``#:`` comment block documenting *code*, from the module source."""
    src = _MODULE.read_text(encoding="utf-8")
    match = re.search(rf"((?:^#:.*\n)+)^{re.escape(code)} = ", src, re.MULTILINE)
    assert match is not None, f"{code}: no '#:' comment block above its assignment"
    return match.group(1)


class TestAGrantLiftsTheRefusalItIsMappedTo:
    @pytest.mark.parametrize("code", sorted(refusal_codes.REFUSAL_GRANTS))
    def test_every_grant_lifts_the_refusal_it_is_mapped_to(self, code: str, monkeypatch: pytest.MonkeyPatch) -> None:
        subject, _ = _refuse(code)
        monkeypatch.setenv(refusal_codes.REFUSAL_GRANTS[code], _CASES[code].grant_value(subject))
        assert _lifted(code), (
            f"{code}: its grant {refusal_codes.REFUSAL_GRANTS[code]} did not lift it, so the map "
            f"names a variable a consumer cannot use to offer the choice"
        )

    @pytest.mark.parametrize("code", sorted(refusal_codes.REFUSAL_GRANTS))
    def test_another_codes_grant_does_not_lift_it(self, code: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """A permutation of the mapping is what this catches."""
        subject, _ = _refuse(code)
        for other, grant in sorted(refusal_codes.REFUSAL_GRANTS.items()):
            if other == code:
                continue
            monkeypatch.setenv(grant, _CASES[other].grant_value(subject))
            assert not _lifted(code), f"{code}: lifted by {other}'s grant {grant}"
            monkeypatch.delenv(grant, raising=False)

    def test_every_code_in_the_map_is_driven_here(self) -> None:
        """Derived from the map, so a sixth code is driven rather than assumed."""
        assert set(_CASES) == set(refusal_codes.REFUSAL_GRANTS)
        assert len(_CASES) >= 5, "the graded set collapsed; a clean run would prove nothing"


class TestWhyEachCodeStatesItsOwnOperation:
    def test_the_subject_is_the_grant_value_only_for_an_allowlist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Three grants take the subject; two are silent when handed it."""
        took_subject, was_silent = [], []
        for code in sorted(refusal_codes.REFUSAL_GRANTS):
            subject, message = _refuse(code)
            monkeypatch.setenv(refusal_codes.REFUSAL_GRANTS[code], subject)
            if _lifted(code):
                took_subject.append(code)
            else:
                _, retry = _refuse(code)
                assert retry == message, f"{code}: the retry reported a different refusal"
                was_silent.append(code)
            monkeypatch.delenv(refusal_codes.REFUSAL_GRANTS[code], raising=False)
        assert took_subject == sorted(
            [
                refusal_codes.HF_REPO_NOT_ALLOWED,
                refusal_codes.POLICY_TYPE_NOT_ALLOWED,
                refusal_codes.POLICY_HOST_NOT_ALLOWED,
            ]
        )
        assert was_silent == sorted([refusal_codes.TRUST_REMOTE_CODE_REQUIRED, refusal_codes.TELEOP_VALUE_OUT_OF_RANGE])

    @pytest.mark.parametrize("code", sorted(refusal_codes.REFUSAL_GRANTS))
    def test_every_code_states_what_to_set_its_grant_to(self, code: str) -> None:
        """The variable alone is not actionable: two codes do not take the subject."""
        assert "Grant:" in _grant_block(code), (
            f"{code}: its comment names a subject but not what to set {refusal_codes.REFUSAL_GRANTS[code]} to"
        )
