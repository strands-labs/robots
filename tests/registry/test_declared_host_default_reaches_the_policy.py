"""A provider's declared ``host`` default must reach the policy it configures.

``build_policy_kwargs`` merges three sources and decides "the caller left this
key unset" by testing the generic parameter against ``None``.  Five of its six
generic parameters default to ``None``, so that test means something for them.
``policy_host`` defaulted to the literal ``"localhost"``, which is
indistinguishable from a caller who asked for ``localhost`` -- so ``host`` was
injected on every call and no declared default for it was ever reachable.

Four of the six providers that accept a ``host`` were affected.  ``moveit2``
and ``vera`` declare ``127.0.0.1`` in ``policies.json``; ``lerobot_async`` and
``remote`` declare it as their constructor default.  All four were handed
``localhost`` -- a value none of them declares anywhere -- while this
function's own docstring promised "A default only ever fills a key the caller
left unset", and the sibling ``resolve_policy`` already left ``host`` unset
unless a URL supplied one.

The cases are derived from ``policies.json`` and the provider constructors
rather than listed, so a provider that gains or changes a host default is
covered the moment it is declared.
"""

import ast
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from strands_robots.registry.policies import build_policy_kwargs

# ─── registry-derived cases ───────────────────────────────────────────


def _registry() -> dict[str, Any]:
    """Locate policies.json from the module under test, never a path literal."""
    path = Path(inspect.getfile(build_policy_kwargs)).parent / "policies.json"
    return json.loads(path.read_text())["providers"]


def _accepts_host() -> dict[str, Any]:
    return {n: c for n, c in sorted(_registry().items()) if "host" in (c.get("config_keys") or [])}


DECLARED = [
    (n, (c.get("defaults") or {})["host"]) for n, c in _accepts_host().items() if "host" in (c.get("defaults") or {})
]
DECLARED_IDS = [n for n, _ in DECLARED]

UNDECLARED = [n for n, c in _accepts_host().items() if "host" not in (c.get("defaults") or {})]


class TestTheCasesAreStillReal:
    """Non-vacuity: an empty case list would make the tests below silent."""

    def test_some_providers_declare_a_registry_host_default(self):
        assert len(DECLARED) >= 2, f"only {DECLARED_IDS} declare a host default"

    def test_some_providers_leave_the_host_default_to_their_constructor(self):
        assert len(UNDECLARED) >= 2, f"only {UNDECLARED} omit a registry host default"

    def test_the_two_groups_do_not_overlap(self):
        assert not set(DECLARED_IDS) & set(UNDECLARED)


class TestTheRegistryHostDefaultIsReachable:
    """The headline: a declared default must not be pre-empted."""

    @pytest.mark.parametrize(("provider", "declared"), DECLARED, ids=DECLARED_IDS)
    def test_an_omitted_host_takes_the_declared_default(self, provider, declared):
        """``moveit2``/``vera`` declare ``127.0.0.1`` and used to get ``localhost``."""
        got = build_policy_kwargs(provider).get("host")
        assert got == declared, (
            f"{provider} declares host={declared!r} in policies.json but "
            f"build_policy_kwargs returned {got!r}, which no provider declares"
        )


class TestTheConstructorHostDefaultIsReachable:
    """A provider with no registry default must be left to its own."""

    @pytest.mark.parametrize("provider", UNDECLARED)
    def test_no_host_is_invented_for_a_provider_that_declares_none(self, provider):
        """Injecting a key is what stopped the constructor default applying."""
        kwargs = build_policy_kwargs(provider)
        assert "host" not in kwargs, (
            f"{provider} declares no registry host default, yet build_policy_kwargs "
            f"supplied host={kwargs['host']!r}, overriding its constructor default"
        )

    @pytest.mark.parametrize("provider", UNDECLARED)
    def test_the_constructor_it_falls_back_to_does_define_a_host(self, provider):
        """Omitting the key is only safe because the constructor answers."""
        cfg = _registry()[provider]
        module = pytest.importorskip(cfg["module"], reason=f"{provider} needs an optional dependency")
        cls = getattr(module, cfg["class"])
        param = inspect.signature(cls.__init__).parameters.get("host")
        assert param is not None and param.default is not inspect.Parameter.empty, (
            f"{cfg['class']} must default host= for an omitted key to be safe"
        )


class TestEveryGenericParameterCanMeanUnset:
    """The root cause, pinned structurally so it cannot return on a sibling.

    The merge tests "did the caller supply this?" as ``value is not None``.
    A generic parameter carrying any other default silently exempts its key
    from the precedence chain, which is exactly what ``policy_host`` did.
    """

    @staticmethod
    def _signature() -> inspect.Signature:
        return inspect.signature(build_policy_kwargs)

    def test_no_generic_parameter_carries_a_value_as_its_default(self):
        offenders = {
            name: p.default
            for name, p in self._signature().parameters.items()
            if name != "provider"
            and p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
            and p.default is not None
        }
        assert not offenders, (
            f"these generic parameters cannot express 'unset', so the registry "
            f"default for their key is unreachable: {offenders}"
        )

    def test_the_scan_reaches_the_real_parameters(self):
        """Non-vacuity: a signature that resolved elsewhere would report clean."""
        names = set(self._signature().parameters)
        assert {"provider", "policy_port", "policy_host", "extra"} <= names


class TestTheDerivedServerAddressNeverCarriesTheSentinel:
    """Scope control on the fix itself: ``None`` must not be interpolated."""

    def test_a_port_alone_does_not_produce_a_none_host_address(self):
        """``f"{policy_host}:{port}"`` would read ``"None:9000"``."""
        for provider in _accepts_host():
            address = build_policy_kwargs(provider, policy_port=9000).get("server_address")
            assert address is None or "None" not in address, f"{provider}: server_address={address!r}"

    def test_an_explicit_host_still_derives_the_address(self):
        kwargs = build_policy_kwargs("lerobot_async", policy_host="gpu-box", policy_port=9000)
        assert kwargs["server_address"] == "gpu-box:9000"


class TestAnExplicitHostStillWins:
    """Over-reach control: the upper tiers of the chain are unchanged."""

    @pytest.mark.parametrize(("provider", "declared"), DECLARED, ids=DECLARED_IDS)
    def test_the_generic_parameter_beats_the_declared_default(self, provider, declared):
        assert build_policy_kwargs(provider, policy_host="gpu-box")["host"] == "gpu-box"

    @pytest.mark.parametrize(("provider", "declared"), DECLARED, ids=DECLARED_IDS)
    def test_the_provider_spelling_beats_the_generic_parameter(self, provider, declared):
        kwargs = build_policy_kwargs(provider, policy_host="generic", host="specific")
        assert kwargs["host"] == "specific"

    def test_an_unknown_provider_still_returns_no_kwargs(self):
        assert build_policy_kwargs("nonexistent_xyz", policy_host="gpu-box") == {}


# ─── structural guard ─────────────────────────────────────────────────


class TestTheMergeStillTreatsNoneAsUnset:
    """The behavioural tests above are only sound while the merge reads ``None``.

    If the ``param_map`` loop stopped testing ``is not None``, a ``None``
    default would start writing ``host=None`` into the kwargs instead of
    leaving the key for the registry, and every test here would still pass.
    """

    def test_the_param_map_loop_skips_none_values(self):
        source = Path(inspect.getfile(build_policy_kwargs)).read_text()
        fn = next(
            n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "build_policy_kwargs"
        )
        loops = {
            str(ast.get_source_segment(source, lp.iter)): str(ast.get_source_segment(source, lp))
            for lp in fn.body
            if isinstance(lp, ast.For)
        }
        assert "value is not None" in loops["param_map.items()"]
