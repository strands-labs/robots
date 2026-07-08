"""The dynamic policy resolver enumerates every policy family lerobot ships.

``strands_robots.policies.lerobot_local.resolution`` does not hard-code a table
of known lerobot policy types. It walks ``lerobot.policies`` at runtime,
imports each subpackage's ``configuration_*`` module, and lets every
``@PreTrainedConfig.register_subclass("<type>")`` decorator populate lerobot's
own draccus choice registry -- which ``list_policy_types()`` then reports. That
design means a new policy family a future lerobot release ships is picked up
automatically, with no strands-side edit.

This module pins that completeness contract against the *installed* lerobot so
the guarantee cannot silently regress: derive the ground-truth set of policy
families directly from lerobot's source (the ``register_subclass`` decorator
arguments under ``lerobot/policies/*/configuration_*.py``) and assert the
resolver enumerates every one of them. If a future lerobot layout change breaks
the dynamic walk -- e.g. a subpackage stops being importable through the stub,
or a policy moves such that ``_ensure_policy_configs_registered`` no longer
reaches it -- ``list_policy_types()`` becomes a strict subset of what lerobot
registers and this fails, surfacing the drift as a resolver gap rather than a
runtime "could not resolve policy type" miss in a user's ``create_policy`` call.

The test is version-agnostic on purpose: it never asserts a fixed policy count
or a fixed name list, only that the resolver's coverage equals lerobot's own.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# The registry is sourced from lerobot's own draccus choice registry, so the
# whole module is meaningless without lerobot installed.
pytest.importorskip("lerobot")

from strands_robots.policies.lerobot_local import list_policy_types  # noqa: E402

# Matches ``@PreTrainedConfig.register_subclass("act")`` and single-quoted /
# whitespace variants. The captured group is the lerobot policy ``type`` string
# -- exactly the token accepted as ``create_policy("lerobot_local",
# policy_type=...)`` and as the ``policy_type`` argument to
# ``resolve_policy_class_by_name``.
_REGISTER_SUBCLASS = re.compile(
    r"""@PreTrainedConfig\.register_subclass\(\s*["']([\w-]+)["']""",
)


def _registered_policy_families() -> set[str]:
    """Ground-truth policy families the installed lerobot registers.

    Scans every ``configuration_*.py`` under the installed lerobot's
    ``policies/`` tree for the ``@PreTrainedConfig.register_subclass("<type>")``
    decorator and collects the decorator argument. This reads lerobot's source
    on disk rather than importing it, so it stays cheap and does not depend on
    optional heavy VLA dependencies being importable.

    Returns:
        The set of policy ``type`` strings lerobot registers in this install.
    """
    import lerobot

    policies_dir = Path(lerobot.__path__[0]) / "policies"
    families: set[str] = set()
    for config_file in policies_dir.glob("*/configuration_*.py"):
        match = _REGISTER_SUBCLASS.search(config_file.read_text(encoding="utf-8"))
        if match:
            families.add(match.group(1))
    return families


def test_resolver_enumerates_every_registered_policy_family() -> None:
    """``list_policy_types()`` covers every policy family lerobot registers.

    The dynamic walk must never silently drop a registered policy: every type
    lerobot exposes via ``@PreTrainedConfig.register_subclass`` has to appear in
    the resolver's discovery surface. A missing entry would present to a user as
    a ``create_policy(policy_type=...)`` resolver miss for a policy their
    installed lerobot actually ships.
    """
    registered = _registered_policy_families()
    assert registered, (
        "expected to find at least one @PreTrainedConfig.register_subclass "
        "decorator in the installed lerobot; the ground-truth scan found none, "
        "which means the lerobot policies layout changed and this guard needs "
        "updating"
    )

    listed = set(list_policy_types())
    missing = registered - listed
    assert not missing, (
        f"the dynamic resolver dropped policy families that lerobot registers: "
        f"{sorted(missing)}. list_policy_types() reported {sorted(listed)} but "
        f"lerobot registers {sorted(registered)}. _ensure_policy_configs_registered "
        f"is no longer reaching every configuration_* module."
    )


def test_ground_truth_scan_is_a_subset_of_the_resolver_surface() -> None:
    """The resolver surface is a superset of the on-disk registration scan.

    ``list_policy_types()`` sources from lerobot's live draccus registry, which
    can legitimately hold entries the coarse source scan misses (e.g. a policy
    registered from a module the regex does not match). The contract is
    one-directional: the resolver must cover at least every family the scan
    finds. This guards the direction that matters -- resolver completeness --
    without over-constraining the resolver to the scan's exact spelling.
    """
    registered = _registered_policy_families()
    listed = set(list_policy_types())
    assert registered.issubset(listed), (
        f"registered families {sorted(registered - listed)} are absent from the "
        f"resolver discovery surface {sorted(listed)}"
    )
