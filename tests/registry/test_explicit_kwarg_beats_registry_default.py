"""``build_policy_kwargs`` must not discard a value the caller supplied.

The function merges three sources into one kwargs dict: the generic parameters
(``policy_host``, ``policy_port``, ...), the provider's registry ``defaults``,
and the provider-specific keys the caller passes through ``**extra``.  Only one
merge order is honest, and the suite already states it -- ``test_registry.py``'s
``test_explicit_value_overrides_default`` asserts "An explicit param must win
over the JSON default for the same key", and the sibling ``resolve_policy``
implements exactly that with its trailing ``kwargs.update(extra_kwargs)``.

That contract used to hold only for the generic parameters, because they were
merged before the defaults.  A key reachable only through ``extra`` was merged
*after* the defaults while skipping keys already present, so the default the
previous loop had just inserted won and the caller's value was dropped with no
error anywhere -- the silent fallback to a default that #317 fixed for
URL-parsed host/port.

The pairs below are derived from ``policies.json`` rather than listed, so a
provider that gains a default is covered the moment it is declared.  Reading
the registry as JSON keeps every case here independent of the optional
dependencies the provider classes need.
"""

import ast
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from strands_robots.registry.policies import build_policy_kwargs

# ─── registry-derived cases ───────────────────────────────────────────


def _registry_path() -> Path:
    """Locate policies.json from the module under test, never a path literal."""
    return Path(inspect.getfile(build_policy_kwargs)).parent / "policies.json"


def _providers() -> dict[str, Any]:
    return json.loads(_registry_path().read_text())["providers"]


def _distinct_from(default: Any, key: str) -> Any:
    """A caller value unmistakably different from ``default``.

    A bool is inverted so the discarded value would flip the meaning of the
    knob rather than merely change it; a number is displaced far enough that no
    rounding could explain the result.
    """
    if isinstance(default, bool):
        return not default
    if isinstance(default, (int, float)):
        return type(default)(default + 41)
    return f"CALLER-{key}"


def _shadowable_pairs() -> list[tuple[str, str, Any]]:
    """Every (provider, key, default) a caller can also supply through extra."""
    pairs = []
    for name, cfg in sorted(_providers().items()):
        allowed = set(cfg.get("config_keys") or [])
        for key, default in (cfg.get("defaults") or {}).items():
            if key in allowed:
                pairs.append((name, key, default))
    return pairs


PAIRS = _shadowable_pairs()
PAIR_IDS = [f"{p}-{k}" for p, k, _ in PAIRS]


class TestTheRegistryStillDeclaresShadowableDefaults:
    """Non-vacuity: the cases below are only meaningful if there are any."""

    def test_several_providers_declare_a_default_for_a_forwardable_key(self):
        """An empty case list would make every parametrised test vacuous."""
        assert len(PAIRS) >= 8, f"only {len(PAIRS)} shadowable (provider, key) pairs"

    def test_at_least_one_shadowable_key_is_a_boolean_mode_switch(self):
        """A bool default is the case where a discarded value inverts intent."""
        assert any(isinstance(d, bool) for _, _, d in PAIRS)


class TestAnExplicitExtraValueBeatsTheRegistryDefault:
    """The caller named the provider's own key; the default must yield."""

    @pytest.mark.parametrize(("provider", "key", "default"), PAIRS, ids=PAIR_IDS)
    def test_the_caller_value_survives(self, provider, key, default):
        """A default may fill an unset key, never replace a supplied one."""
        asked = _distinct_from(default, key)
        kwargs = build_policy_kwargs(provider, **{key: asked})
        assert kwargs[key] == asked, (
            f"{provider}: asked for {key}={asked!r}, got {kwargs.get(key)!r} (registry default {default!r} won)"
        )

    def test_wbc_walk_false_is_not_turned_back_into_walk_true(self):
        """The headline case: the discarded value inverted a locomotion mode.

        ``wbc`` declares ``walk: true``, so a caller asking for the balance
        controller used to receive the walking one -- a policy doing the
        opposite of what was requested, reported as success.
        """
        kwargs = build_policy_kwargs("wbc", walk=False)
        assert kwargs["walk"] is False

    def test_a_falsy_caller_value_is_not_read_as_absent(self):
        """0 / False / "" are values, not omissions."""
        assert build_policy_kwargs("curobo", action_horizon=0)["action_horizon"] == 0
        assert build_policy_kwargs("cosmos3", prompt="")["prompt"] == ""


class TestTheProviderSpecificSpellingBeatsTheGenericParameter:
    """``host`` and ``policy_host`` name one key; the explicit one wins."""

    def test_extra_host_beats_the_generic_policy_host(self):
        """The provider's own spelling wins over the generic one."""
        kwargs = build_policy_kwargs("groot", host="gpu-box")
        assert kwargs["host"] == "gpu-box"

    def test_extra_port_beats_the_registry_default(self):
        """``policy_port`` defaults to None, so only the registry shadowed it."""
        kwargs = build_policy_kwargs("cosmos3", port=9100)
        assert kwargs["port"] == 9100

    def test_extra_wins_when_both_spellings_are_supplied(self):
        """Both are explicit; the provider's own key is the more specific."""
        kwargs = build_policy_kwargs("groot", policy_host="generic", host="specific")
        assert kwargs["host"] == "specific"


class TestTheGenericParameterStillBeatsTheDefault:
    """The middle tier of the chain is unchanged."""

    def test_policy_host_still_overrides_the_json_default(self):
        """Restates test_registry.py's existing contract for the middle tier."""
        assert build_policy_kwargs("cosmos3", policy_host="gpu-box")["host"] == "gpu-box"

    def test_policy_port_still_reaches_a_provider_that_declares_port(self):
        assert build_policy_kwargs("groot", policy_port=5555)["port"] == 5555


class TestADefaultStillFillsAnUnsetKey:
    """Over-reach control: the defaults must still apply when nothing else does."""

    @pytest.mark.parametrize(("provider", "key", "default"), PAIRS, ids=PAIR_IDS)
    def test_an_omitted_key_takes_the_registry_default(self, provider, key, default):
        """Honouring an explicit value must not stop a default from filling in.

        ``host`` used to be carved out of this contract: ``policy_host``
        carried the literal default ``"localhost"``, so it filled the key
        before the registry could.  Every generic parameter now defaults to
        ``None``, so the rule is uniform.
        """
        kwargs = build_policy_kwargs(provider)
        assert kwargs[key] == default, (
            f"{provider}: {key} declares {default!r} but build_policy_kwargs returned {kwargs.get(key)!r}"
        )


class TestUnknownExtraKeysAreStillDropped:
    """Scope control: the ``config_keys`` filter itself is unchanged."""

    def test_a_key_outside_config_keys_is_not_forwarded(self):
        """Deciding what is forwardable is what ``config_keys`` is for."""
        assert "not_a_real_key" not in build_policy_kwargs("groot", not_a_real_key="x")

    def test_an_unknown_provider_still_returns_no_kwargs(self):
        assert build_policy_kwargs("nonexistent_xyz", host="x") == {}


# ─── structural guard ─────────────────────────────────────────────────


def _merge_loop_order(source: str) -> list[str]:
    """The iterables of the three merge loops, in source order."""
    fn = next(
        n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "build_policy_kwargs"
    )
    return [str(ast.get_source_segment(source, lp.iter)) for lp in fn.body if isinstance(lp, ast.For)]


def _loop_bodies(source: str) -> dict[str, str]:
    fn = next(
        n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "build_policy_kwargs"
    )
    out = {}
    for lp in fn.body:
        if isinstance(lp, ast.For):
            out[str(ast.get_source_segment(source, lp.iter))] = str(ast.get_source_segment(source, lp))
    return out


class TestTheMergeOrderIsTheContract:
    """The precedence is expressed by the loop order, so pin the loop order.

    A behavioural test catches the values that are wrong today.  This catches
    the edit that would make them wrong again -- moving the ``extra`` loop back
    below the defaults, or reinstating the ``key not in kwargs`` guard on it.
    """

    @staticmethod
    def _source() -> str:
        return Path(inspect.getfile(build_policy_kwargs)).read_text()

    def test_extra_is_merged_before_the_generic_parameters_and_defaults(self):
        assert _merge_loop_order(self._source()) == [
            "extra.items()",
            "param_map.items()",
            "defaults.items()",
        ]

    def test_the_extra_loop_does_not_skip_keys_already_present(self):
        """That guard is what let an already-inserted default win."""
        body = _loop_bodies(self._source())["extra.items()"]
        assert "key not in kwargs" not in body

    def test_the_later_loops_do_skip_keys_already_present(self):
        """They are the ones that must yield to an explicit value."""
        bodies = _loop_bodies(self._source())
        assert "key not in kwargs" in bodies["param_map.items()"]
        assert "key not in kwargs" in bodies["defaults.items()"]

    def test_the_scan_finds_the_real_function(self):
        """Non-vacuity: a scan that resolved elsewhere would report nothing."""
        assert len(_merge_loop_order(self._source())) == 3

    def test_the_order_check_rejects_the_pre_fix_arrangement(self):
        """Planted defect: the arrangement this change replaced must fail."""
        planted = (
            "def build_policy_kwargs(provider, **extra):\n"
            "    for key, value in param_map.items():\n"
            "        kwargs[key] = value\n"
            "    for key, default_val in defaults.items():\n"
            "        if key not in kwargs:\n"
            "            kwargs[key] = default_val\n"
            "    for key, value in extra.items():\n"
            "        if key not in kwargs:\n"
            "            kwargs[key] = value\n"
        )
        assert _merge_loop_order(planted) != [
            "extra.items()",
            "param_map.items()",
            "defaults.items()",
        ]
        assert "key not in kwargs" in _loop_bodies(planted)["extra.items()"]
