"""The trust-remote-code gate must cover every spelling ``create_policy`` accepts.

``_check_trust_remote_code`` membership-tests its argument against
``_HF_REMOTE_CODE_PROVIDERS``, a set of *canonical* provider names, and
``create_policy`` hands it whatever ``_resolve_policy_class`` returned. That
function documents its first return value as ``canonical_provider_name``, but
its third stage returned the caller's spelling unchanged -- so a provider's
declared alias or shorthand reached the gate as a name the set does not hold,
the membership test missed, and the provider was constructed with no opt-in.

Three of the five spellings the registry accepted for a gated provider skipped
the gate entirely: ``lerobot`` (for ``lerobot_local``), and ``kimodo_g1`` and
``text2motion`` (for ``kimodo``). None is obscure -- ``text2motion`` is the
natural-language shorthand the registry advertises for Kimodo, and ``lerobot``
is the obvious short name for ``lerobot_local``. The gate decides, in the words
of ``tests/policies/kimodo`` where the flag it guards is pinned, "whether the
provider may be constructed at all", and these providers load HuggingFace
models with ``trust_remote_code=True``: arbitrary code execution from the model
repository.

The contract violation was wider than the gate. Fourteen accepted spellings
returned a non-canonical name; only three happened to name a gated provider, so
adding an alias to a gated provider -- or gating a provider that already has one
-- would have silently inherited the same bypass. Stage 1 (the runtime alias
map) and stage 2 (``resolve_policy``'s shorthand stage) both canonicalise
already; stage 3 is the one that did not, which is why the tests below pin the
returned name for every accepted spelling rather than only for the gated ones.

The controls pass before and after the fix. They pin that the gate stays an
opt-in gate rather than becoming a ban, that an ungated provider's aliases are
never refused, and that an unknown spelling is still reported rather than
silently canonicalised into something else.
"""

from __future__ import annotations

import pytest

from strands_robots.policies.factory import (
    _HF_REMOTE_CODE_PROVIDERS,
    UntrustedRemoteCodeError,
    _resolve_policy_class,
    create_policy,
)
from strands_robots.registry.policies import _build_alias_map, list_policy_providers

_TRUST_ENV = "STRANDS_TRUST_REMOTE_CODE"


def _canonical(provider: str) -> str:
    """Canonicalise ``provider`` the way ``policies.json`` declares it.

    Spelled out here rather than imported so this file pins the *behaviour*
    ``create_policy`` must have, not the helper that happens to implement it.
    """
    return _build_alias_map().get(provider, provider)


def _accepted_spellings(canonical: str) -> list[str]:
    """Return every spelling the registry accepts for ``canonical``."""
    aliases = sorted(a for a, c in _build_alias_map().items() if c == canonical and a != canonical)
    return [canonical, *aliases]


def _gated_spellings() -> list[tuple[str, str]]:
    """Return ``(spelling, canonical)`` for every accepted spelling of a gated provider."""
    return [(s, g) for g in sorted(_HF_REMOTE_CODE_PROVIDERS) for s in _accepted_spellings(g)]


def _every_accepted_spelling() -> list[str]:
    """Return every spelling the JSON registry accepts, canonical names included."""
    return sorted(set(_build_alias_map()) | set(list_policy_providers()))


def _reported_provider_or_skip(spelling: str) -> str:
    """Return the provider name resolution reports for ``spelling``, or skip.

    Resolution imports the provider's module, so a spelling whose optional
    dependency is absent cannot be resolved in every environment. Skipping
    reports that the spelling went unchecked, where passing quietly would report
    a checked name the run never read.

    The handler raises ``pytest.skip.Exception`` rather than calling
    ``pytest.skip``, which is the same skip by a route a reader of this function
    alone can follow. ``pytest.skip`` terminates by raising that class, but that
    is a property of pytest rather than of this body, so a liveness analysis
    scoped to one function reads the handler as falling through -- reporting
    ``reported`` as possibly-uninitialized and this function as mixing an
    explicit return with an implicit one. Both readings are wrong about the
    control flow and right about what the body states, and an explicit ``raise``
    is what makes the two agree.

    Args:
        spelling: Any spelling the registry accepts.

    Returns:
        The provider name ``_resolve_policy_class`` reports for ``spelling``.

    Raises:
        pytest.skip.Exception: If ``spelling`` cannot be resolved here.
    """
    try:
        reported, _cls, _kwargs = _resolve_policy_class(spelling)
    except (ImportError, ValueError) as exc:
        raise pytest.skip.Exception(f"{spelling!r} is not resolvable in this environment: {exc}") from exc
    return reported


def test_a_gated_provider_really_has_a_non_canonical_spelling() -> None:
    """Premise: without an alias on a gated provider these tests prove nothing."""
    extra = [s for s, canonical in _gated_spellings() if s != canonical]
    assert extra, (
        "premise: no gated provider declares an alias or shorthand, so the gate "
        "can only ever be reached by its canonical name and the tests below are "
        f"vacuous (gated providers: {sorted(_HF_REMOTE_CODE_PROVIDERS)})"
    )


@pytest.mark.parametrize(("spelling", "canonical"), _gated_spellings())
def test_every_accepted_spelling_of_a_gated_provider_is_refused_without_opt_in(
    spelling: str, canonical: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gated provider is gated under every name the registry answers to.

    The gate is what stands between a caller and arbitrary code execution from a
    model repository, so reaching it must not depend on which of the provider's
    advertised spellings the caller happened to type.
    """
    monkeypatch.delenv(_TRUST_ENV, raising=False)
    with pytest.raises(UntrustedRemoteCodeError) as excinfo:
        create_policy(spelling)
    # The refusal names the remedy, so a caller who typed the alias can act on it.
    assert _TRUST_ENV in str(excinfo.value)


@pytest.mark.parametrize("spelling", _every_accepted_spelling())
def test_resolution_reports_the_canonical_name_for_every_accepted_spelling(spelling: str) -> None:
    """``_resolve_policy_class`` returns the canonical name it documents.

    This is the root cause rather than the symptom: any decision keyed on the
    returned name -- the trust gate today, anything added later -- is only
    correct if the name is the one the registry is keyed on.
    """
    reported = _reported_provider_or_skip(spelling)
    assert reported in set(list_policy_providers()), (
        f"_resolve_policy_class({spelling!r}) reported {reported!r}, which is not a "
        f"canonical provider name; its docstring promises canonical_provider_name"
    )
    assert reported == _canonical(spelling)


@pytest.mark.parametrize(("spelling", "canonical"), _gated_spellings())
def test_opting_in_still_admits_every_gated_spelling(
    spelling: str, canonical: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: the gate is an opt-in gate, not a ban.

    Passes before and after the fix. Fails if the gate is ever made
    unconditional, which is the obvious wrong way to close the bypass. Only the
    gate's own refusal is asserted on: a provider may still decline to construct
    for its own reasons (a missing checkpoint, an absent optional dependency),
    and that is not this gate talking.
    """
    monkeypatch.setenv(_TRUST_ENV, "1")
    try:
        create_policy(spelling)
    except UntrustedRemoteCodeError as exc:  # pragma: no cover - the failure this pins
        pytest.fail(f"opting in did not admit {spelling!r}: {exc}")
    except Exception:  # noqa: BLE001 - provider-specific construction failure is fine
        pass


@pytest.mark.parametrize(
    "spelling",
    [s for s in _every_accepted_spelling() if _canonical(s) not in _HF_REMOTE_CODE_PROVIDERS],
)
def test_an_ungated_provider_is_never_refused_by_the_trust_gate(spelling: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: canonicalising the name must not pull ungated providers into the gate.

    Passes before and after the fix. Fails if the gate starts matching on
    something broader than the canonical name of a gated provider.
    """
    monkeypatch.delenv(_TRUST_ENV, raising=False)
    try:
        create_policy(spelling)
    except UntrustedRemoteCodeError as exc:  # pragma: no cover - the failure this pins
        pytest.fail(f"{spelling!r} resolves to an ungated provider but was gated: {exc}")
    except Exception:  # noqa: BLE001 - provider-specific construction failure is fine
        pass


def test_an_unknown_spelling_is_reported_rather_than_canonicalised() -> None:
    """Control: canonicalisation leaves a name no provider declares alone.

    Passes before and after the fix. A spelling the registry does not hold has
    to survive to the refusal so that refusal can quote what the caller typed.
    """
    assert _canonical("no_such_provider") == "no_such_provider"
    with pytest.raises(ValueError, match="no_such_provider"):
        _resolve_policy_class("no_such_provider")
