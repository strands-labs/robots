"""The documented resolution ladder names the URL forms the registry ships.

``strands_robots.registry.policies.resolve_policy`` opens with a five-stage
``Resolution order:`` block whose first rung enumerates the URL forms stage 1
recognises. That stage has no vocabulary of its own: it matches the
``url_patterns`` entries providers declare in ``policies.json``, so the forms it
recognises are exactly the ones the shipped registry carries. A rung that names
a form no shipped pattern matches sends the reader after a spelling that reaches
stage 5 instead, where it is forwarded to ``lerobot_local`` as a checkpoint id;
a rung that omits a shipped pattern hides a transport a caller could have used.

Both directions are graded here against one relation - does some shipped
``url_patterns`` entry match a probe of this form - so the rung and the registry
cannot drift apart in either direction. The same relation grades
:func:`strands_robots.policies.factory.policy_provider_error`, whose docstring
enumerates the spellings it treats as resolvable in a ``--`` aside.

In both cases only the enumerating construct is graded - the numbered rung, and
the aside - never the prose around it. That prose exists to explain the forms
stage 1 does *not* match, so a form named there is a qualification rather than a
claim, and grading it would make an accurate docstring fail.
"""

import inspect
import re

import pytest

import strands_robots.registry.policies as policies_mod
from strands_robots.policies.factory import policy_provider_error
from strands_robots.registry.policies import resolve_policy

# A URL form as the prose spells one: a scheme with its separator, or the
# scheme-less ``host:port`` shape.
_FORM = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://|host:port")

# The rung-1 text: everything between the ``1.`` marker and the ``2.`` marker of
# the ``Resolution order:`` block.
_RUNG_ONE = re.compile(r"^Resolution order:\s*$(.*?)\n\s*2\.", re.M | re.S)

# The enumerating aside in ``policy_provider_error``'s docstring: the spellings
# it lists between the ``--`` pair as ones that resolve.
_PREFLIGHT_ASIDE = re.compile(r"every spelling that provider accepts\s*--(.*?)--", re.S)


def _inject_registry(monkeypatch, providers: dict) -> None:
    """Force resolve_policy to see a synthetic provider registry.

    Mirrors the shim the URL-scheme coverage tests use; kept local so this
    module reads standalone.
    """
    real_load = policies_mod._load

    def fake_load(name: str):
        if name == "policies":
            return {"providers": providers}
        return real_load(name)

    monkeypatch.setattr(policies_mod, "_load", fake_load)


def _rung_one_text() -> str:
    """The first rung of ``resolve_policy``'s documented resolution order."""
    doc = inspect.getdoc(resolve_policy) or ""
    match = _RUNG_ONE.search(doc)
    assert match, "resolve_policy no longer opens with a numbered 'Resolution order:' block"
    return match.group(1)


def _preflight_enumerated_text() -> str:
    """The aside in ``policy_provider_error``'s docstring that enumerates spellings."""
    doc = inspect.getdoc(policy_provider_error) or ""
    match = _PREFLIGHT_ASIDE.search(doc)
    assert match, "policy_provider_error no longer enumerates the spellings that resolve"
    return match.group(1)


def _url_forms(text: str) -> list[str]:
    """The URL forms ``text`` names, in order, deduplicated."""
    return list(dict.fromkeys(_FORM.findall(text)))


def _shipped_url_patterns() -> list[tuple[str, str]]:
    """Every ``(provider, url_pattern)`` pair the shipped registry declares."""
    providers = policies_mod._load("policies").get("providers", {})
    return [(name, pattern) for name, info in providers.items() for pattern in info.get("url_patterns", [])]


def _probe(form: str) -> str:
    """A minimal address of ``form``, for matching against a declared pattern."""
    return "h:1" if form == "host:port" else f"{form}h:1"


def _unmatched(forms: list[str]) -> list[str]:
    """The named forms no shipped ``url_patterns`` entry matches."""
    patterns = [pattern for _name, pattern in _shipped_url_patterns()]
    return [f for f in forms if not any(re.match(p, _probe(f)) for p in patterns)]


class TestTheLadderAndTheRegistryNameTheSameUrlForms:
    """Stage 1's documented vocabulary is the registry's, in both directions."""

    def test_every_url_form_the_ladder_names_is_one_the_registry_matches(self):
        """A named form no pattern matches reaches stage 5, not stage 1."""
        forms = _url_forms(_rung_one_text())
        unsound = _unmatched(forms)
        assert not unsound, (
            f"resolve_policy's stage-1 rung names {unsound}, which no shipped url_patterns entry "
            f"matches; such a string falls through to stage 5 and is forwarded to lerobot_local as "
            f"a checkpoint id. Declared patterns: {_shipped_url_patterns()}"
        )

    def test_every_url_pattern_the_registry_ships_is_named_in_the_ladder(self):
        """A shipped pattern the rung omits is a transport the reader cannot find."""
        probes = [_probe(f) for f in _url_forms(_rung_one_text())]
        unnamed = [
            f"{name} ({pattern})"
            for name, pattern in _shipped_url_patterns()
            if not any(re.match(pattern, p) for p in probes)
        ]
        assert not unnamed, (
            f"the shipped registry declares url_patterns that resolve_policy's stage-1 rung does not name: {unnamed}"
        )

    def test_the_preflight_docstring_names_no_form_the_registry_cannot_match(self):
        """``policy_provider_error`` enumerates the spellings it calls resolvable."""
        forms = _url_forms(_preflight_enumerated_text())
        unsound = _unmatched(forms)
        assert not unsound, (
            f"policy_provider_error's docstring names {unsound} among the spellings that resolve, "
            f"but no shipped url_patterns entry matches it"
        )

    def test_the_ladder_and_the_registry_are_both_non_empty(self):
        """A parse that reached nothing would make the two rules above vacuous."""
        forms = _url_forms(_rung_one_text())
        assert len(forms) >= 3, f"only parsed {forms} out of the stage-1 rung"
        assert len(_shipped_url_patterns()) >= 3, "the shipped registry declares almost no url_patterns"
        assert _url_forms(_preflight_enumerated_text()), "parsed no form out of the preflight's aside"


class TestTheSchemeLessAddressIsAnExtensionPointNotAShippedForm:
    """A bare ``host:port`` is matched only by a declared scheme-less pattern."""

    @pytest.mark.parametrize("address", ["gpu-box:8080", "localhost:5555"])
    def test_a_scheme_less_address_reaches_the_fallback_with_the_shipped_registry(self, address):
        """No shipped pattern matches it, so stage 5 forwards it as a checkpoint id."""
        provider, kwargs = resolve_policy(address)
        assert provider == "lerobot_local"
        assert kwargs["pretrained_name_or_path"] == address
        assert "server_address" not in kwargs
        assert "host" not in kwargs

    def test_the_preflight_reports_no_reason_for_a_scheme_less_address(self):
        """It resolves - as a checkpoint id - so the preflight has nothing to report."""
        assert policy_provider_error("gpu-box:8080") is None

    def test_a_declared_scheme_less_pattern_still_maps_to_server_address(self, monkeypatch):
        """The generic branch is a live extension point, just not a shipped form."""
        _inject_registry(monkeypatch, {"hostport": {"url_patterns": [r"^[^/]+:[0-9]+$"]}})
        provider, kwargs = resolve_policy("myserver:8080")
        assert provider == "hostport"
        assert kwargs["server_address"] == "myserver:8080"
