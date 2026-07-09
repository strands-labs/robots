"""``resolve_name`` tolerates stray separators in a known alias.

``resolve_name`` normalizes a robot name/alias to its canonical form. Beyond the
direct alias and canonical lookups, it has a fallback: strip every underscore
(dashes are first folded to underscores) and retry the lookup, but only return
the stripped match when it actually resolves to something the registry knows.
This lets an agent that types a separator into an otherwise separator-free alias
- ``"g-1"`` / ``"g_1"`` for the alias ``"g1"`` - still reach the canonical robot
instead of getting an unresolved name back.

These pin that fallback for aliases that carry no internal separator, using only
the public ``resolve_name`` API (the canonical target is discovered by resolving
the bare alias, so the test does not hardcode registry canonicals).
"""

import pytest

from strands_robots.registry import resolve_name

# Aliases known to carry no internal separator; a stray dash/underscore inside
# them must not defeat resolution. Skipped automatically if the registry stops
# shipping one of them, so the test tracks the alias set without hardcoding it.
_NO_SEPARATOR_ALIASES = ["g1", "gr1", "cf2", "franka", "reachy"]


@pytest.mark.parametrize("alias", _NO_SEPARATOR_ALIASES)
def test_stray_separator_in_alias_still_resolves(alias: str) -> None:
    """A dash/underscore injected into a separator-free alias resolves the same.

    Folds to the stripped-alias fallback: neither ``"g-1"`` nor ``"g_1"`` is a
    registered key, but stripping separators recovers the alias ``"g1"``.
    """
    canonical = resolve_name(alias)
    if canonical == alias:
        pytest.skip(f"{alias!r} is not a live alias in this registry build")

    with_dash = f"{alias[:1]}-{alias[1:]}"
    with_underscore = f"{alias[:1]}_{alias[1:]}"

    # The injected separator is not itself a registered name...
    assert with_dash != canonical
    # ...yet resolution recovers the same canonical robot as the bare alias.
    assert resolve_name(with_dash) == canonical
    assert resolve_name(with_underscore) == canonical


def test_stripped_form_matching_nothing_is_returned_unchanged() -> None:
    """An unknown name whose stripped form is also unknown is returned as-is.

    The fallback must not invent a resolution: a made-up name normalizes
    (lowercase, dashes to underscores) and is handed back verbatim when neither
    it nor its stripped form matches any alias or canonical.
    """
    assert resolve_name("Totally-Unknown-Bot") == "totally_unknown_bot"
