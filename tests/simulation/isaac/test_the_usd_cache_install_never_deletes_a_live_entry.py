"""Installing a converted USD never deletes an entry another process is reading.

The USD cache root is shared cross-process -
``~/.strands_robots/asset_cache/usd_robots`` - and the pid-suffixed staging
directory says concurrent converters are an intended case. The install step
defeated it:

    _remove_tree(target_root)
    os.replace(staging, target_root)

When two processes miss the marker for one key and both convert - a parallel test
session on a cold cache, or several sims launching on one host - the loser's
``_remove_tree(target_root)`` deletes the winner's finished entry. Not before the
winner uses it: *after* the winner wrote its marker, returned ``final``, and its
``add_robot`` referenced that USD into a live stage. USD composes payloads
lazily, so a read landing in the delete-then-rename window fails, or composes the
robot without its meshes, in a process whose own conversion was entirely correct.
That is close to undebuggable in the field, because the error surfaces in a
different process from the one that caused it and names neither.

A second shape of the same race needed no delete at all: the marker used to be
written *after* the rename, so a reader could observe ``target_root`` with no
marker and convert again for nothing, and a crash in that window left a torn
entry behind for good.

The fix is two changes that answer both:

* the marker is written INSIDE staging, so the rename publishes a complete entry;
* nothing deletes a completed entry - the install renames onto an ABSENT target
  and reads a failure as "another process published this key first", which is
  sound because both converters produce identical content for a given key.

A torn ``target_root`` would otherwise become permanently uninstallable once the
delete is gone, so it is moved aside with a single ``os.rename`` - atomic, so the
process that loses that race re-reads the marker rather than fighting.

These pins drive the real ``convert_mjcf_to_usd`` with the vendor importer stood
in, and the concurrency cell uses a ``threading.Barrier`` to put two conversions
in the window on purpose rather than hoping to land in it.
"""

from __future__ import annotations

import concurrent.futures
import os
import pathlib
import threading
import types
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac import mjcf_assets  # noqa: E402 - after importorskip
from strands_robots.simulation.isaac.mjcf_assets import convert_mjcf_to_usd  # noqa: E402

_MJCF = '<mujoco model="probe"><worldbody><body name="b"/></worldbody></mujoco>'


class _Importer:
    """Writes the layout the real importer writes. ``gate`` lets a test hold a
    conversion inside the install window."""

    gate: threading.Barrier | None = None
    #: Set by the importer once it has been entered, so a test can know a
    #: conversion is past its marker check and parked inside the window.
    entered: threading.Event | None = None
    #: Cleared by a test to hold the conversion there until it says otherwise.
    release: threading.Event | None = None

    def __init__(self, config: Any) -> None:
        self._config = config

    def import_mjcf(self) -> str:
        # Bound to locals before the checks: ``type(self).X is not None`` does not
        # narrow the following ``type(self).X.wait()``, and a test clears these
        # mid-call on purpose, so re-reading the attribute is also a real race.
        entered = type(self).entered
        if entered is not None:
            entered.set()
        gate = type(self).gate
        if gate is not None:
            gate.wait(timeout=10)
        release = type(self).release
        if release is not None:
            release.wait(timeout=10)
        stem = os.path.splitext(os.path.basename(self._config.mjcf_path))[0]
        out_dir = os.path.join(self._config.usd_path, stem)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{stem}.usda")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("#usda 1.0\n")
        return out


class _Config:
    def __init__(self) -> None:
        self.mjcf_path: str | None = None
        self.usd_path: str | None = None
        self.fix_base: bool | None = None
        self.import_scene: bool = True


@pytest.fixture
def importer(monkeypatch: pytest.MonkeyPatch) -> type[_Importer]:
    """Install a fake ``isaacsim.asset.importer.mjcf`` for the duration."""
    _Importer.gate = None
    _Importer.entered = None
    _Importer.release = None
    import sys

    for name in ("isaacsim", "isaacsim.asset", "isaacsim.asset.importer", "isaacsim.asset.importer.mjcf"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    module = sys.modules["isaacsim.asset.importer.mjcf"]
    module.MJCFImporter = _Importer  # type: ignore[attr-defined]
    module.MJCFImporterConfig = _Config  # type: ignore[attr-defined]
    sys.modules["isaacsim"].asset = sys.modules["isaacsim.asset"]  # type: ignore[attr-defined]
    sys.modules["isaacsim.asset"].importer = sys.modules["isaacsim.asset.importer"]  # type: ignore[attr-defined]
    sys.modules["isaacsim.asset.importer"].mjcf = module  # type: ignore[attr-defined]
    return _Importer


@pytest.fixture
def mjcf(tmp_path: pathlib.Path) -> str:
    path = tmp_path / "src" / "probe.xml"
    path.parent.mkdir(parents=True)
    path.write_text(_MJCF, encoding="utf-8")
    return str(path)


class TestTwoConcurrentConversionsBothEndUpWithAUsablePath:
    def test_neither_caller_is_handed_a_path_that_disappears(
        self, importer: type[_Importer], mjcf: str, tmp_path: pathlib.Path
    ) -> None:
        """The headline. Pre-fix the loser's install deleted the winner's entry,
        so one of these two paths stopped existing while its caller held it."""
        cache = str(tmp_path / "cache")
        importer.gate = threading.Barrier(2, timeout=10)

        # ``concurrent.futures`` rather than a hand-rolled exception-marshal box.
        # These threads are created here, so CPython's own ``_WorkItem.run``
        # already holds the ``except BaseException`` this needs and
        # ``Future.result()`` re-raises with object identity preserved - which
        # AGENTS.md records as strictly better than the box, not merely quieter,
        # and is why the box belongs only to a marshal onto an ALREADY-RUNNING
        # foreign thread (``IsaacSimulation.run_on_main``).
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(convert_mjcf_to_usd, mjcf, cache) for _ in range(2)]
            # ``result()`` re-raises, so a conversion that failed fails this cell
            # naming its own exception rather than an empty-list assertion.
            paths = [future.result(timeout=20) for future in futures]

        assert len(paths) == 2
        for index, path in enumerate(paths):
            assert os.path.isfile(path), f"caller {index} was handed {path!r}, which does not exist"

    def test_they_agree_on_one_entry(self, importer: type[_Importer], mjcf: str, tmp_path: pathlib.Path) -> None:
        """Both converters produce identical content for a key, so the winner's
        entry is the right answer for both - which is what makes deferring sound."""
        cache = str(tmp_path / "cache")
        importer.gate = threading.Barrier(2, timeout=10)
        results: list[str] = []

        def _convert() -> None:
            results.append(convert_mjcf_to_usd(mjcf, cache))

        threads = [threading.Thread(target=_convert) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        assert len(set(results)) == 1, f"two callers were handed different entries: {results}"

    def test_no_staging_directory_is_left_behind(
        self, importer: type[_Importer], mjcf: str, tmp_path: pathlib.Path
    ) -> None:
        """The loser discards its own staging rather than leaking it."""
        cache = tmp_path / "cache"
        importer.gate = threading.Barrier(2, timeout=10)

        threads = [threading.Thread(target=lambda: convert_mjcf_to_usd(mjcf, str(cache))) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

        leftovers = [entry.name for entry in cache.iterdir() if entry.name.startswith(".")]
        assert leftovers == [], f"staging directories survived: {leftovers}"


class TestTheInstallNeverDeletesACompletedEntry:
    """Measured across concurrent conversions, because that is the only way two
    installs for one key ever happen.

    A *sequential* second call reads the published marker and returns without
    reaching the install at all, so it cannot observe this property - it is a
    cache hit, which :class:`TestTheEntryIsPublishedCompleteOrNotAtAll` covers.
    Stated because the sequential version of this test looks like it measures the
    race and does not: it passes on the racing code too.
    """

    def _loser_parked_inside_the_window(
        self, mjcf: str, cache: str
    ) -> tuple[threading.Thread, threading.Event, list[str]]:
        """Start a conversion and hold it *after* its marker check.

        This is what makes the race deterministic rather than hopeful. The loser
        has already decided to convert - it passed the marker check while the
        cache was cold - so releasing it later drives the exact install that used
        to delete the winner's entry, with the winner's work fully complete and
        published in between. Racing two threads on a barrier cannot order them
        that way, and a plain second call cannot reach the install at all.
        """
        entered = threading.Event()
        release = threading.Event()
        _Importer.entered = entered
        _Importer.release = release
        results: list[str] = []

        def _convert() -> None:
            results.append(convert_mjcf_to_usd(mjcf, cache))

        loser = threading.Thread(target=_convert)
        loser.start()
        assert entered.wait(timeout=10), "the held conversion never reached the importer"

        # The loser is now blocked on the event object it already fetched, so
        # clearing these class attributes releases the WINNER from the same gate
        # without freeing the loser. Leaving them set made the winner wait on the
        # loser's event too, which timed out and left the two installs unordered -
        # the flake this comment exists to prevent recurring.
        _Importer.entered = None
        _Importer.release = None
        return loser, release, results

    def test_the_winners_file_keeps_its_inode(
        self, importer: type[_Importer], mjcf: str, tmp_path: pathlib.Path
    ) -> None:
        """The inode rather than the bytes: a delete-and-replace produces
        identical content at a NEW inode, which is exactly the failure a content
        check cannot see and exactly what breaks a USD stage holding the old one
        open while it composes payloads lazily.
        """
        cache = str(tmp_path / "cache")
        loser, release, _ = self._loser_parked_inside_the_window(mjcf, cache)

        # The winner converts and publishes while the loser is held.
        winner = convert_mjcf_to_usd(mjcf, cache)
        before = os.stat(winner).st_ino

        release.set()
        loser.join(timeout=20)

        assert os.path.isfile(winner), "the loser's install deleted the winner's entry"
        assert os.stat(winner).st_ino == before, "the entry was relinked, so a live stage reading it would break"

    def test_an_open_handle_survives_the_losers_install(
        self, importer: type[_Importer], mjcf: str, tmp_path: pathlib.Path
    ) -> None:
        """The field symptom, as close as a unit test reaches: the winner is
        holding the USD open - as a composing stage does - when the loser installs."""
        cache = str(tmp_path / "cache")
        loser, release, _ = self._loser_parked_inside_the_window(mjcf, cache)
        winner = convert_mjcf_to_usd(mjcf, cache)

        with open(winner, encoding="utf-8") as live:
            release.set()
            loser.join(timeout=20)
            assert live.read() == "#usda 1.0\n"

        assert os.path.isfile(winner)

    def test_the_loser_is_handed_the_winners_entry(
        self, importer: type[_Importer], mjcf: str, tmp_path: pathlib.Path
    ) -> None:
        """And it defers rather than failing: the winner's entry is the right
        answer for both, since identical input produces identical content."""
        cache = str(tmp_path / "cache")
        loser, release, results = self._loser_parked_inside_the_window(mjcf, cache)
        winner = convert_mjcf_to_usd(mjcf, cache)

        release.set()
        loser.join(timeout=20)

        assert results == [winner]


class TestTheEntryIsPublishedCompleteOrNotAtAll:
    def test_the_marker_is_written_before_the_rename(
        self, importer: type[_Importer], mjcf: str, tmp_path: pathlib.Path
    ) -> None:
        """So no reader can observe the directory without one. Checked by reading
        the installed entry: the marker sits inside it, which can only be true if
        it was written into staging."""
        cache = tmp_path / "cache"
        convert_mjcf_to_usd(mjcf, str(cache))

        entries = [entry for entry in cache.iterdir() if entry.is_dir() and not entry.name.startswith(".")]
        assert len(entries) == 1
        assert (entries[0] / ".converted").is_file()

    def test_the_marker_names_the_installed_path_not_the_staging_one(
        self, importer: type[_Importer], mjcf: str, tmp_path: pathlib.Path
    ) -> None:
        """It is written before the rename, so it has to record the path the file
        will have afterwards - a staging path would be a dead pointer."""
        cache = tmp_path / "cache"
        final = convert_mjcf_to_usd(mjcf, str(cache))

        entry = next(e for e in cache.iterdir() if e.is_dir() and not e.name.startswith("."))
        recorded = (entry / ".converted").read_text(encoding="utf-8").strip()

        assert recorded == final
        assert os.path.isfile(recorded)
        assert ".tmp" not in recorded

    def test_a_cache_hit_needs_no_importer(self, importer: type[_Importer], mjcf: str, tmp_path: pathlib.Path) -> None:
        """The published marker is what makes the second call a hit rather than a
        second conversion."""
        cache = str(tmp_path / "cache")
        first = convert_mjcf_to_usd(mjcf, cache)

        def _explode(config: Any) -> Any:
            raise AssertionError("the importer ran on a cache hit")

        import sys

        sys.modules["isaacsim.asset.importer.mjcf"].MJCFImporter = _explode  # type: ignore[attr-defined]

        assert convert_mjcf_to_usd(mjcf, cache) == first


class TestATornEntryIsRecoveredRatherThanFatal:
    """Nothing deletes a completed entry any more, so a directory with no usable
    marker must not become permanently uninstallable."""

    def test_a_directory_with_no_marker_is_replaced(
        self, importer: type[_Importer], mjcf: str, tmp_path: pathlib.Path
    ) -> None:
        cache = tmp_path / "cache"
        final = convert_mjcf_to_usd(mjcf, str(cache))
        entry = next(e for e in cache.iterdir() if e.is_dir() and not e.name.startswith("."))

        # Tear it the way a crash between rename and marker-write used to.
        (entry / ".converted").unlink()

        again = convert_mjcf_to_usd(mjcf, str(cache))

        assert os.path.isfile(again)
        assert again == final

    def test_a_marker_naming_a_missing_file_is_replaced(
        self, importer: type[_Importer], mjcf: str, tmp_path: pathlib.Path
    ) -> None:
        """A marker is only usable while the file it names exists."""
        cache = tmp_path / "cache"
        final = convert_mjcf_to_usd(mjcf, str(cache))
        os.unlink(final)

        again = convert_mjcf_to_usd(mjcf, str(cache))

        assert os.path.isfile(again)

    def test_no_quarantine_directory_survives(
        self, importer: type[_Importer], mjcf: str, tmp_path: pathlib.Path
    ) -> None:
        cache = tmp_path / "cache"
        convert_mjcf_to_usd(mjcf, str(cache))
        entry = next(e for e in cache.iterdir() if e.is_dir() and not e.name.startswith("."))
        (entry / ".converted").unlink()

        convert_mjcf_to_usd(mjcf, str(cache))

        torn = [e.name for e in cache.iterdir() if ".torn." in e.name]
        assert torn == [], f"a quarantined directory was left behind: {torn}"


class TestTheMarkerReaderTreatsEveryUnusableStateAlike:
    """``_read_marker`` answers ``None`` for all of them, because the caller's
    action is the same - convert and try to install."""

    def test_an_absent_marker(self, tmp_path: pathlib.Path) -> None:
        assert mjcf_assets._read_marker(str(tmp_path / "nope")) is None

    def test_an_empty_marker(self, tmp_path: pathlib.Path) -> None:
        marker = tmp_path / ".converted"
        marker.write_text("   \n", encoding="utf-8")

        assert mjcf_assets._read_marker(str(marker)) is None

    def test_a_marker_naming_a_missing_file(self, tmp_path: pathlib.Path) -> None:
        marker = tmp_path / ".converted"
        marker.write_text(str(tmp_path / "gone.usda"), encoding="utf-8")

        assert mjcf_assets._read_marker(str(marker)) is None

    def test_a_marker_that_is_a_directory(self, tmp_path: pathlib.Path) -> None:
        """An ``OSError`` on the read, rather than a raise out of the caller."""
        marker = tmp_path / ".converted"
        marker.mkdir()

        assert mjcf_assets._read_marker(str(marker)) is None

    def test_a_usable_marker_is_returned(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "robot.usda"
        target.write_text("#usda 1.0\n", encoding="utf-8")
        marker = tmp_path / ".converted"
        marker.write_text(f"{target}\n", encoding="utf-8")

        assert mjcf_assets._read_marker(str(marker)) == str(target)
