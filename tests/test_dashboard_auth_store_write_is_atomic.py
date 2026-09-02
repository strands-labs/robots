"""The credential store is replaced, never rewritten in place.

`tests/test_dashboard_auth_corrupt_store.py` pins what happens *after* the store
is found unparseable: keep the bytes, say so, and let the person at the machine
re-enroll. These cells pin the other half -- that this module does not produce
that artifact itself.

The store is the deployment's only credential record and `_save_locked` is its
highest-frequency writer: every successful `finish_authentication` persists
`sign_count` through it, as does every enrollment and the corruption re-seed. A
truncating write made a routine login the thing that could destroy the passkey
records and the `jwt_secret` permanently -- the recovery posture bounds the
security damage, not the loss. So the payload lands in a sibling temp file and
is `os.replace`d into position, and these cells grade that behaviour: the
previous store survives every failure, no debris outlives it, and owner-only
permissions are a property of creation rather than a chmod that may not land.
"""

import json
import os

import pytest

from strands_robots.dashboard import auth

_PREVIOUS = {"credentials": [{"id": "AAA", "name": "the-operator-passkey"}], "jwt_secret": "keep-me"}


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    for key in ("STRANDS_DASH_AUTH_ENABLED", "STRANDS_DASH_AUTH_RP_ID", "STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    auth._cache = {}
    auth._corrupt = None
    yield
    auth._corrupt = None


def _seed_previous_store(tmp_path):
    """Put a good store on disk the way an enrolled deployment would have one."""
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(_PREVIOUS, indent=2))
    return path, path.read_bytes()


def _mode(path) -> int:
    return path.stat().st_mode & 0o777


class TestAnInterruptedSaveCannotDestroyTheStore:
    """The failure modes that used to leave a truncated store leave the old one.

    Each cell kills the save at a different point and asserts the *bytes* on
    disk are still the previous store's -- not merely parseable, byte-identical,
    because a partially-updated credential record is as unrecoverable as a
    truncated one.
    """

    def test_a_write_that_dies_midway_leaves_the_previous_store_byte_identical(self, tmp_path, monkeypatch):
        """A kill inside the write window: the old store is what a reader still finds."""
        path, before = _seed_previous_store(tmp_path)
        real_fdopen = os.fdopen

        class DyingHandle:
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def write(self, _data):
                raise OSError(28, "No space left on device")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self._wrapped.close()
                return False

        monkeypatch.setattr(auth.os, "fdopen", lambda fd, *a, **k: DyingHandle(real_fdopen(fd, *a, **k)))

        with pytest.raises(OSError):
            auth._save({"credentials": [], "jwt_secret": "the-replacement"})

        assert path.read_bytes() == before, "an interrupted save must not touch the store on disk"
        assert json.loads(path.read_text()) == _PREVIOUS, "the operator's passkey record survives"

    def test_a_replace_that_fails_leaves_the_previous_store_byte_identical(self, tmp_path, monkeypatch):
        """The final step failing is the same guarantee: nothing was ever overwritten."""
        path, before = _seed_previous_store(tmp_path)
        monkeypatch.setattr(
            auth.os,
            "replace",
            lambda *a, **k: (_ for _ in ()).throw(OSError(30, "Read-only file system")),
        )

        with pytest.raises(OSError):
            auth._save({"credentials": [], "jwt_secret": "the-replacement"})

        assert path.read_bytes() == before, "a failed replace must not disturb the store"

    def test_a_failed_save_is_reported_rather_than_swallowed(self, tmp_path, monkeypatch):
        """Losing the write silently would leave the caller's sign_count un-persisted.

        `finish_authentication` persists through here; a save that failed and said
        nothing would let the ceremony report success over a store that never
        recorded it, which is the replay window the counter exists to close.
        """
        _seed_previous_store(tmp_path)
        monkeypatch.setattr(
            auth.os,
            "replace",
            lambda *a, **k: (_ for _ in ()).throw(OSError(30, "Read-only file system")),
        )

        with pytest.raises(OSError):
            auth._save({"credentials": []})

    @pytest.mark.parametrize(
        ("failing", "boom"),
        [
            ("replace", OSError(30, "Read-only file system")),
            ("replace", KeyboardInterrupt()),
        ],
        ids=["an-os-error", "an-interrupt"],
    )
    def test_no_temp_file_outlives_a_failed_save(self, tmp_path, monkeypatch, failing, boom):
        """Debris beside the store is read by nothing and cleaned up by nobody.

        `KeyboardInterrupt` is in here deliberately: it is not an `Exception`, so a
        cleanup written as `except Exception` would leak on exactly the signal an
        operator sends by hand.
        """
        _seed_previous_store(tmp_path)

        def explode(*_a, **_k):
            raise boom

        monkeypatch.setattr(auth.os, failing, explode)

        with pytest.raises(type(boom)):
            auth._save({"credentials": []})

        strays = [p.name for p in tmp_path.iterdir() if p.name != "auth.json"]
        assert strays == [], f"a failed save left debris beside the store: {strays}"


class TestOwnerOnlyIsAPropertyOfCreation:
    """0600 comes from `mkstemp`, so there is no window and no chmod to fail.

    The store used to be created at the umask default and chmod-ed afterwards.
    That left a freshly generated `jwt_secret` briefly world-readable, and made
    owner-only a wish on any volume with no chmod concept.
    """

    def test_a_new_store_is_never_group_or_world_readable(self, tmp_path):
        auth._save({"credentials": [], "jwt_secret": "secret"})
        assert _mode(tmp_path / "auth.json") == 0o600

    def test_a_permissive_umask_does_not_widen_it(self, tmp_path):
        previous = os.umask(0o000)
        try:
            auth._save({"credentials": [], "jwt_secret": "secret"})
        finally:
            os.umask(previous)
        assert _mode(tmp_path / "auth.json") == 0o600, "the mode must not be inherited from the umask"

    def test_a_store_left_wide_open_by_an_older_build_is_tightened_on_the_next_write(self, tmp_path):
        """`os.replace` carries the temp file's bits onto the store, so this self-heals.

        The wide-open store is produced the way the older build produced one -- created
        at the umask default with no chmod afterwards -- rather than chmod-ed into place.
        The premise is then the defect's own mechanism rather than a mode this tree never
        writes, and the cell states no literal mode it would have to keep in step with
        the umask it runs under.
        """
        path = tmp_path / "auth.json"
        previous = os.umask(0o000)
        try:
            path.write_text(json.dumps(_PREVIOUS, indent=2))
        finally:
            os.umask(previous)
        assert _mode(path) & 0o077, "the seeded store must be group- or world-reachable to grade anything"

        auth._save({"credentials": [], "jwt_secret": "secret"})

        assert _mode(path) == 0o600


class TestWhatTheAtomicWriteDoesNotChange:
    """The controls. These held before the change and must still hold."""

    def test_a_successful_save_is_what_load_reads_back(self, tmp_path):
        auth._save({"credentials": [], "jwt_secret": "round-trip"})
        auth._cache = {}
        assert auth._load()["jwt_secret"] == "round-trip"

    def test_the_store_is_still_indented_json(self, tmp_path):
        auth._save({"credentials": [], "jwt_secret": "s"})
        assert "\n  " in (tmp_path / "auth.json").read_text(), "kept human-repairable"

    def test_a_missing_parent_directory_is_still_created(self, tmp_path, monkeypatch):
        nested = tmp_path / "a" / "b" / "auth.json"
        monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(nested))
        auth._save({"credentials": [], "jwt_secret": "s"})
        assert json.loads(nested.read_text())["jwt_secret"] == "s"

    def test_the_cache_is_updated_so_the_next_read_needs_no_parse(self, tmp_path, monkeypatch):
        """A save leaves the store readable without parsing the file again.

        Asserted by making the parse fail: on a hit the store comes from memory. The body
        is still read, and that read is what licenses skipping the parse - a stat tuple
        cannot, because it names the file rather than a version of it (two writes inside
        one coarse-clock mtime tick that keep the byte count share one). This cell asked
        for no read at all while the tuple was trusted to have changed; what it grades
        now is the work the cache actually saves.
        """
        auth._save({"credentials": [], "jwt_secret": "cached"})

        def no_parse(*a, **k):
            raise AssertionError("parsed the store body on a cache hit")

        monkeypatch.setattr(auth.json, "loads", no_parse)
        assert auth._load()["jwt_secret"] == "cached"
