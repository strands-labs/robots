"""``invalidate_cache`` scoping contract for the JSON registry loader.

The loader caches each registry (``robots``, ``policies``) independently and
exposes ``invalidate_cache(name=None)`` to drop cached data so the next access
re-reads from disk. The scoping matters on the hot path: every
``register_robot`` / ``unregister_robot`` calls ``invalidate_cache("robots")``
(via ``user_registry._invalidate_cache``) so a freshly registered robot is
immediately visible - but it must evict *only* the ``robots`` cache and leave
the unrelated ``policies`` cache intact, rather than nuking every registry on
each registration. ``invalidate_cache(None)`` is the explicit "drop everything"
escape hatch.

These pin both modes by their observable effect: the loader re-reads an
invalidated registry (a fresh dict object) while returning the same cached
object for one that was left alone.
"""

from __future__ import annotations

from strands_robots.registry import get_robot, loader


def _make_robot(parent, name="cache_bot", xml_name="bot.xml"):
    """Write a minimal MJCF under a per-robot dir and return that dir."""
    robot_dir = parent / name
    robot_dir.mkdir(parents=True, exist_ok=True)
    (robot_dir / xml_name).write_text('<mujoco model="cache_bot"><worldbody/></mujoco>')
    return robot_dir


def test_invalidate_named_registry_reloads_only_that_registry():
    """``invalidate_cache("robots")`` forces a re-read of the robots registry on
    next access while the policies cache is left untouched."""
    loader.invalidate_cache()  # start from a cold, known state
    robots_before = loader._load("robots")
    policies_before = loader._load("policies")
    # Warm reads return the very same cached objects (no re-read).
    assert loader._load("robots") is robots_before
    assert loader._load("policies") is policies_before

    loader.invalidate_cache("robots")

    assert loader._load("robots") is not robots_before, "named invalidation must force a robots re-read"
    assert loader._load("policies") is policies_before, "named invalidation must not disturb the policies cache"


def test_invalidate_all_reloads_every_registry():
    """``invalidate_cache(None)`` drops every registry, so both re-read."""
    loader.invalidate_cache()
    robots_before = loader._load("robots")
    policies_before = loader._load("policies")

    loader.invalidate_cache(None)

    assert loader._load("robots") is not robots_before, "invalidate_cache(None) must re-read robots"
    assert loader._load("policies") is not policies_before, "invalidate_cache(None) must re-read policies"


def test_register_robot_makes_robot_visible_without_disturbing_policies(tmp_path):
    """The documented register_robot workflow: the new robot is immediately
    discoverable (robots cache evicted) yet the policies cache is preserved,
    because registration scopes its invalidation to the robots registry."""
    from strands_robots.registry import register_robot

    policies_before = loader._load("policies")  # warm the policies cache
    assert get_robot("cache_bot") is None  # not registered yet

    robot_dir = _make_robot(tmp_path)
    register_robot(name="cache_bot", model_xml="bot.xml", asset_dir=str(robot_dir), joints=6)

    got = get_robot("cache_bot")
    assert got is not None, "registered robot must be immediately visible (robots cache was evicted)"
    assert loader._load("policies") is policies_before, "registering a robot must not evict the policies cache"
