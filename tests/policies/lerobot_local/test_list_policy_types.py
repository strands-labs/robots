"""Tests for the ``lerobot_local`` policy-type discovery surface.

``list_policy_types()`` lets a caller enumerate the ``policy_type`` strings the
installed lerobot can resolve, instead of reading lerobot internals to guess.
The same list also turns ``resolve_policy_class_by_name``'s previously
dead-end "could not resolve" error into an actionable one that names the valid
choices.
"""

from __future__ import annotations

import sys

import pytest

# Skip the whole module unless lerobot is importable (the policy-type registry
# is sourced from lerobot's own draccus choice registry).
pytest.importorskip("lerobot")

from strands_robots.policies.lerobot_local import list_policy_types  # noqa: E402
from strands_robots.policies.lerobot_local.resolution import (  # noqa: E402
    resolve_policy_class_by_name,
)


def test_list_policy_types_is_sorted_and_includes_core_families() -> None:
    """The discovery surface returns the resolvable types, sorted and deduped."""
    types = list_policy_types()
    assert types, "expected a non-empty list of policy types with lerobot installed"
    assert types == sorted(types), "policy types must be returned sorted"
    assert len(types) == len(set(types)), "policy types must be deduplicated"
    # ACT and Diffusion ship in every lerobot >= 0.4; assert the stable core.
    for core in ("act", "diffusion"):
        assert core in types, f"expected core policy type {core!r} in {types}"


def test_every_registered_type_resolves_or_raises_clean_importerror() -> None:
    """Resolution is well-behaved for EVERY registered policy type.

    ``resolve_policy_class_by_name`` deliberately catches every internal
    failure mode of the underlying lerobot import (``ImportError`` from a
    module absent in this install, plus the ``TypeError`` / ``AttributeError``
    / ``RuntimeError`` / ``ValueError`` that a ``modeling_*`` / ``factory``
    import can raise when an optional VLA dependency is missing or a config
    dataclass is malformed under the installed transformers) and re-surfaces
    the dead end as a single, actionable ``ImportError`` that names the valid
    choices. That "never leak lerobot's internal exception type" contract is
    the whole point of the module's hardening -- so this guard pins it: for
    every type the installed lerobot registers, resolution must EITHER return
    a concrete class OR raise ``ImportError``, and never propagate a raw
    internal exception.

    This is the forward-compat guard against lerobot drift: if a future
    lerobot layout change makes strands' resolution leak a raw ``TypeError``
    (etc.) for a registered type, this fails -- whereas a narrower test that
    tolerated the leak (``except (ImportError, TypeError)``) would silently
    pass and defeat the hardening. It couples to neither a specific lerobot
    release nor which types happen to resolve in a given install (only that
    resolution is well-behaved for all of them).
    """
    resolved_count = 0
    for policy_type in list_policy_types():
        try:
            cls = resolve_policy_class_by_name(policy_type)
        except ImportError:
            # The one sanctioned failure: the type is registered in the
            # draccus choice registry but its concrete class is not importable
            # in this install (missing optional dep / trimmed install).
            continue
        except BaseException as exc:  # noqa: BLE001 - the point is to catch a leak
            raise AssertionError(
                f"resolve_policy_class_by_name({policy_type!r}) leaked a "
                f"{type(exc).__name__} ({exc}); the contract is to return a "
                f"class or raise a clean ImportError, never a raw internal error."
            ) from exc
        assert isinstance(cls, type), f"{policy_type} resolved to {cls!r} which is not a type"
        resolved_count += 1
    # At minimum the stable core (act, diffusion, shipped in every lerobot
    # >= 0.4) must resolve to a concrete class.
    assert resolved_count >= 2, f"expected at least 2 types to resolve, got {resolved_count}"
    for core in ("act", "diffusion"):
        cls = resolve_policy_class_by_name(core)
        assert isinstance(cls, type), f"core type {core!r} must resolve"


def test_resolution_surfaces_clean_importerror_when_a_strategy_raises_internally(monkeypatch) -> None:
    """An internal (non-ImportError) failure is re-surfaced as ImportError.

    Regression teeth for the contract pinned above: the ``modeling_*`` and
    package-level import strategies catch ``TypeError`` (a missing VLA dep
    makes ``class Foo(None)`` raise ``TypeError`` at import) and fall through
    to the clean, enumerated ``ImportError`` rather than leaking it. Forcing
    ``importlib.import_module`` to raise ``TypeError`` inside those strategies
    must therefore still produce an ``ImportError`` from the public function.
    If the except-tuple that reclassifies internal errors is ever narrowed,
    the raw ``TypeError`` leaks and this fails.
    """
    from strands_robots.policies.lerobot_local import resolution

    def _boom(name: str, *args: object, **kwargs: object) -> object:
        raise TypeError(f"simulated internal import failure for {name}")

    # Only the module-import strategies route through importlib.import_module;
    # the factory strategy uses a `from ... import`, so it still runs for real
    # and cleanly rejects the unknown type. The final list-of-choices lookup
    # is exception-guarded and unaffected.
    monkeypatch.setattr(resolution.importlib, "import_module", _boom)
    with pytest.raises(ImportError):
        resolution.resolve_policy_class_by_name("totally_made_up_policy_xyz")


def test_unknown_type_error_enumerates_available_types() -> None:
    """The unresolvable-type error names the valid choices (actionable error).

    Pre-fix the message ended with a bare "Ensure lerobot is installed" hint
    and gave a user with a typo'd ``policy_type`` no way to discover the right
    spelling; this regression pins the enumerated remedy.
    """
    types = list_policy_types()
    with pytest.raises(ImportError) as excinfo:
        resolve_policy_class_by_name("definitely_not_a_real_policy_type")
    message = str(excinfo.value)
    assert "definitely_not_a_real_policy_type" in message
    # The actionable part: the error enumerates the resolvable policy types.
    assert all(t in message for t in types), f"error message should list every available type {types}, got: {message}"


def test_list_policy_types_empty_when_lerobot_config_unimportable(monkeypatch) -> None:
    """A missing dependency yields an empty list, never an exception.

    ``list_policy_types`` is a discovery surface, so it degrades gracefully:
    setting the config module to ``None`` in ``sys.modules`` makes the internal
    ``from lerobot.configs.policies import PreTrainedConfig`` raise ImportError,
    and the function must swallow it and return ``[]``.
    """
    monkeypatch.setitem(sys.modules, "lerobot.configs.policies", None)
    assert list_policy_types() == []


def test_list_policy_types_falls_back_to_choice_registry_on_old_draccus(monkeypatch) -> None:
    """Older draccus lacks the public ``get_known_choices()`` accessor.

    The discovery surface must degrade to the private ``_choice_registry`` dict
    that accessor wraps so enumeration still works on those installs, instead of
    raising ``AttributeError``. A fake ``PreTrainedConfig`` (no
    ``get_known_choices``, only ``_choice_registry``) is injected so the
    fallback runs without depending on the installed draccus version.
    """
    import types

    from strands_robots.policies.lerobot_local import resolution

    class _OldDraccusConfig:
        _choice_registry = {"diffusion": object, "act": object}

    fake_module = types.ModuleType("lerobot.configs.policies")
    setattr(fake_module, "PreTrainedConfig", _OldDraccusConfig)
    monkeypatch.setitem(sys.modules, "lerobot.configs.policies", fake_module)

    assert resolution.list_policy_types() == ["act", "diffusion"]
