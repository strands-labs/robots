"""The mtime hot-reload must honor the user overlay, not just the package JSON.

The ``robots`` registry the read API serves is the package ``robots.json``
merged with the user-local overlay ``$STRANDS_BASE_DIR/user_robots.json``
(:func:`loader._merge_user_robots`).  ``register_robot`` / ``unregister_robot``
invalidate the cache explicitly, but any *other* writer - a second process, a
manual edit, or a tool that writes the file directly - relies on the loader's
documented "re-read when the source changes" behavior.  These tests pin that
the cache signature tracks the overlay file's mtime, so create / modify / delete
of ``user_robots.json`` are all observed without a manual ``invalidate_cache``.

Each test warms the cache first, then mutates the overlay file *directly*
(never via ``register_robot``, which would invalidate the cache and mask the
bug).  The autouse fixture in ``conftest`` isolates ``STRANDS_BASE_DIR`` per
test and clears the cache on entry/exit.
"""

from __future__ import annotations

import json
import os

from strands_robots.registry import get_robot, list_robots, loader
from strands_robots.utils import get_base_dir


def _overlay_path():
    return get_base_dir() / "user_robots.json"


def _write_overlay(robots: dict) -> None:
    """Write user_robots.json directly, bypassing register_robot()."""
    path = _overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"robots": robots}))


def _entry(description: str) -> dict:
    return {
        "description": description,
        "category": "arm",
        "joints": 6,
        "asset": {"dir": "x", "model_xml": "x.xml", "scene_xml": "x.xml"},
    }


def test_external_overlay_creation_is_picked_up():
    """A user robot written to the overlay by an external writer becomes visible
    on the next read, without a manual cache invalidation."""
    # Warm the cache while the overlay file does not exist yet.
    assert get_robot("ext_created") is None

    _write_overlay({"ext_created": _entry("added out of process")})

    got = get_robot("ext_created")
    assert got is not None, "external overlay creation was not picked up (stale cache)"
    assert got["description"] == "added out of process"


def test_external_overlay_modification_is_picked_up():
    """Rewriting an existing overlay entry is observed on the next read."""
    _write_overlay({"ext_mod": _entry("first")})
    assert get_robot("ext_mod")["description"] == "first"  # warm cache

    _write_overlay({"ext_mod": _entry("second")})
    # Force a strictly-newer mtime so the change is unambiguous regardless of
    # filesystem timestamp granularity.
    path = _overlay_path()
    now = path.stat().st_mtime
    os.utime(path, (now + 10, now + 10))

    assert get_robot("ext_mod")["description"] == "second", (
        "external overlay modification was not picked up (stale cache)"
    )


def test_external_overlay_deletion_is_picked_up():
    """Deleting the overlay file drops its robots on the next read."""
    _write_overlay({"ext_deleted": _entry("temporary")})
    assert get_robot("ext_deleted") is not None  # warm cache

    _overlay_path().unlink()

    assert get_robot("ext_deleted") is None, "external overlay deletion was not picked up (stale cache)"
    # Package robots are unaffected by the overlay churn.
    assert any(r["name"] == "so101" for r in list_robots())


class TestAnOverlayEditInsideOneTimestampTickIsSeen:
    """The tests above step the clock; a real writer does not.

    ``test_external_overlay_modification_is_picked_up`` advances the overlay's
    mtime by ten seconds "so the change is unambiguous regardless of filesystem
    timestamp granularity" - which grades the reload while stepping around the
    only case where a timestamp cannot express it.  The kernel stamps mtime from
    a coarse clock (4 ms on ext4 here), so two writes inside one tick carry the
    same timestamp, and that timestamp never changes again: a signature that
    cannot see the second write serves the first one for the life of the
    process.  Two processes registering robots a millisecond apart is exactly
    that shape.

    The tick is pinned with ``os.utime`` rather than raced for, so the cells are
    deterministic on any filesystem: a real writer produces this state whenever
    its two writes land in one tick, which is the common case, not the corner.
    """

    @staticmethod
    def _rewrite_without_advancing_the_clock(robots: dict) -> None:
        """Rewrite the overlay and restore the timestamp it had beforehand."""
        path = _overlay_path()
        before = path.stat()
        _write_overlay(robots)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = path.stat()
        assert after.st_mtime_ns == before.st_mtime_ns, "the timestamp must be unchanged"

    def test_a_same_size_rewrite_in_one_tick_is_picked_up(self):
        """The overlay is re-read even when neither timestamp nor size moved.

        Rotating one entry's value for another of equal length is what a
        rename or a re-registration looks like on disk, and it leaves every
        stat field identical.
        """
        _write_overlay({"tick_same": _entry("aaaa")})
        assert get_robot("tick_same")["description"] == "aaaa"  # warm the cache

        before = _overlay_path().stat()
        self._rewrite_without_advancing_the_clock({"tick_same": _entry("bbbb")})
        assert _overlay_path().stat().st_size == before.st_size, "the size must be unchanged too"

        assert get_robot("tick_same")["description"] == "bbbb", (
            "a same-tick overlay edit was not picked up (stale cache)"
        )

    def test_a_new_entry_added_in_one_tick_is_picked_up(self):
        """A longer overlay is re-read too, so size is no substitute for it."""
        _write_overlay({"tick_first": _entry("first")})
        assert get_robot("tick_first") is not None  # warm the cache

        self._rewrite_without_advancing_the_clock(
            {"tick_first": _entry("first"), "tick_second": _entry("added in the same tick")}
        )

        got = get_robot("tick_second")
        assert got is not None, "a robot registered in the same tick was not picked up (stale cache)"
        assert got["description"] == "added in the same tick"

    def test_an_unchanged_overlay_is_neither_reparsed_nor_revalidated(self, monkeypatch):
        """Reading the source is what licenses the hit; the parse is what it saves.

        Keying on contents costs one read per lookup and must still skip the
        JSON parse, the overlay merge and the uniqueness validation - otherwise
        the fix would have traded a stale cache for no cache.
        """
        _write_overlay({"tick_cached": _entry("once")})
        assert get_robot("tick_cached") is not None  # warm the cache

        validated: list[str] = []
        real_validate = loader._validate

        def counting_validate(name: str, data: dict) -> None:
            validated.append(name)
            real_validate(name, data)

        monkeypatch.setattr(loader, "_validate", counting_validate)

        for _ in range(5):
            assert get_robot("tick_cached")["description"] == "once"

        assert validated == [], "an unchanged registry must be served from cache, not reparsed"
