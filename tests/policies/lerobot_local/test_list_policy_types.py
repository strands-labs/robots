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


def test_core_types_actually_resolve() -> None:
    """The stable core types resolve to concrete classes.

    The draccus choice registry may contain types from newer lerobot modules
    that are not yet resolvable in the current install (e.g. ``wall_x`` is
    registered but has no ``modeling_wall_x`` module on lerobot 0.5.x), or
    types whose config triggers a ``TypeError`` at import time (GR00T N1.5
    dataclass issue under transformers 5.x). Rather than asserting ALL listed
    types resolve (which would couple this test to the full lerobot release
    matrix), we assert the stable core always resolves and verify the
    resolution function does not crash on any listed type.
    """
    resolved_count = 0
    for policy_type in list_policy_types():
        try:
            cls = resolve_policy_class_by_name(policy_type)
            assert isinstance(cls, type), f"{policy_type} resolved to {cls!r} which is not a type"
            resolved_count += 1
        except (ImportError, TypeError):
            # ImportError: module not present in this lerobot version.
            # TypeError: dataclass field-ordering issue (GR00T N1.5).
            continue
    # At minimum the stable core (act, diffusion) must resolve.
    assert resolved_count >= 2, f"expected at least 2 types to resolve, got {resolved_count}"
    # Verify the stable core specifically.
    for core in ("act", "diffusion"):
        cls = resolve_policy_class_by_name(core)
        assert isinstance(cls, type), f"core type {core!r} must resolve"


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
