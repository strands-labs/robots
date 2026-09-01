"""The resolved settings tree is cached under the file it was resolved from.

The cache was a value global (``_cache: dict | None``) rebound through ``global``
at two sites and through ``globals()["_cache"]`` at two more -- the same
rebinding, spelled so it does not need the declaration. That made "the cached
tree describes the current ``SETTINGS_FILE``" an invariant maintained by hand at
every write, and the two could disagree: nothing in ``load()`` compared the
cached tree against the path it came from, so a process that repointed
``SETTINGS_FILE`` was served the previous file's tree, and a stale hit is
indistinguishable from a fresh one.

Repointing is not hypothetical. ``SETTINGS_FILE`` is module state, the dashboard
CLI's settings flag rewrites it, and every fixture in ``tests/test_dashboard_*``
that wants a scratch file monkeypatches it -- those pass only because they also
call ``clear_overrides()`` and ``load(refresh=True)`` by hand, which is the
by-hand part.

Keyed on the path, they cannot disagree: a tree is reachable only through the
file it came from, so a repoint is a miss rather than a wrong hit.
``auth.py::_cache`` keys its store on a file identity for the same reason and
carries the same reasoning in a comment beside it -- the two modules are siblings
in one package and should not disagree about this idiom.

``test_repointing_the_file_is_not_served_the_previous_tree`` is the cell that
discriminates: it fails against the value-global implementation. The rest exist
because the cheapest way to pass that one cell is to delete the cache, so they
pin that a cache is still there (a repeat ``load()`` touches no disk), that it
stays bounded at one entry, and that both override paths still drop it.
"""

from __future__ import annotations

import json

import pytest

from strands_robots.dashboard import settings


def _write(path, token: str) -> None:
    path.write_text(json.dumps({"security": {"auth_token": token}}))


@pytest.fixture()
def files(tmp_path, monkeypatch):
    """Two settings files on disk, with the module pointed at the first."""
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write(first, "FIRST")
    _write(second, "SECOND")
    monkeypatch.setattr(settings, "SETTINGS_FILE", first)
    settings.clear_overrides()
    settings.load(refresh=True)
    yield first, second
    settings.clear_overrides()
    settings.load(refresh=True)


class TestTheCacheCannotOutliveTheFileItDescribes:
    def test_repointing_the_file_is_not_served_the_previous_tree(self, files, monkeypatch):
        first, second = files
        assert settings.load()["security"]["auth_token"] == "FIRST"

        # No refresh, no clear_overrides -- exactly what a caller that repoints
        # the path and then reads does. Under the value global this returned
        # FIRST: the tree was cached, and nothing tied it to `first`.
        monkeypatch.setattr(settings, "SETTINGS_FILE", second)

        assert settings.load()["security"]["auth_token"] == "SECOND", (
            "a tree resolved from a different settings file was served as current"
        )

    def test_returning_to_the_first_file_re_reads_it(self, files, monkeypatch):
        first, second = files
        assert settings.load()["security"]["auth_token"] == "FIRST"
        monkeypatch.setattr(settings, "SETTINGS_FILE", second)
        assert settings.load()["security"]["auth_token"] == "SECOND"

        # Coming back must read `first` again rather than serve a tree that was
        # cached under it earlier -- the file may have changed meanwhile, which
        # is why the cache is cleared before each insert instead of accumulating.
        _write(first, "FIRST-EDITED")
        monkeypatch.setattr(settings, "SETTINGS_FILE", first)

        assert settings.load()["security"]["auth_token"] == "FIRST-EDITED"


class TestThereIsStillACache:
    def test_a_repeated_load_touches_no_disk(self, files, monkeypatch):
        # The cheapest way to pass the cell above is to cache nothing, so pin the
        # hit: the second load must not reach _read_file.
        settings.load(refresh=True)

        # Counted rather than refused: a stub that raises would also fire in this
        # fixture's teardown, which reloads on purpose, and report a product
        # defect where there is a finaliser.
        reads: list[str] = []
        real = settings._read_file

        def counted():
            reads.append("read")
            return real()

        monkeypatch.setattr(settings, "_read_file", counted)

        assert settings.load()["security"]["auth_token"] == "FIRST"
        assert reads == [], "load() re-read the settings file instead of using the cache"

    def test_it_holds_at_most_one_entry(self, files, monkeypatch):
        first, second = files
        settings.load()
        monkeypatch.setattr(settings, "SETTINGS_FILE", second)
        settings.load()
        monkeypatch.setattr(settings, "SETTINGS_FILE", first)
        settings.load()

        # One settings path per process, so a keyed cache that grew per path
        # would be an unbounded map an operator could feed.
        assert len(settings._cache) == 1, f"the cache grew past one entry: {sorted(settings._cache)}"


class TestBothOverridePathsStillDropIt:
    def test_setting_an_override_drops_the_cached_tree(self, files):
        assert settings.load()["security"]["auth_token"] == "FIRST"

        settings.override("security", "auth_token", "OVERRIDDEN")

        assert settings.load()["security"]["auth_token"] == "OVERRIDDEN"

    def test_clearing_overrides_drops_the_cached_tree(self, files):
        settings.override("security", "auth_token", "OVERRIDDEN")
        assert settings.load()["security"]["auth_token"] == "OVERRIDDEN"

        settings.clear_overrides()

        assert settings.load()["security"]["auth_token"] == "FIRST", "the overridden tree survived clear_overrides()"
