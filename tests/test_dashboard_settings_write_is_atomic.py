"""The settings file holds the dashboard's bearer token, so how it is written matters.

``security.auth_token`` is the credential every ``/api`` and ``/ws`` request must
present. Two properties of the write are therefore security properties rather than
housekeeping, and neither was graded before:

* the file is **owner-only**, and is so by construction rather than by a ``chmod``
  that may not land -- measured at ``umask 022`` the token previously sat in a
  ``0o644`` file, and permanently so wherever the ``chmod`` failed, under a silent
  ``except OSError``;
* an interrupted write **cannot destroy** the settings already on disk, and leaves
  no ``.tmp`` sibling holding the token behind.

Two cells discriminate against the previous ``write_text`` + ``chmod``
implementation and are the reason this file exists:
``test_no_chmod_is_needed_for_the_file_to_end_owner_only`` (``0o644``, token
readable by everyone, permanently) and
``test_a_keyboard_interrupt_leaves_the_previous_file_intact`` (a ``.tmp`` holding
the token outlives the write). The other five pass against the old code too and
are kept deliberately: they grade invariants the old sequence happened to satisfy
by a different route, so they are what stops a future rewrite from losing them
quietly.

One property is stated in ``_write_file``'s docstring but is **not** pinned here,
because it is a window rather than an end state: under the old code the token was
readable by everyone *during* the write, in the ``.tmp`` sibling and then in the
destination until the ``chmod`` landed. It was measured directly rather than
graded.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from strands_robots.dashboard import settings

TOKEN = "SUPER-SECRET-BEARER"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    f = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "SETTINGS_FILE", f)
    settings.clear_overrides()
    settings.load(refresh=True)
    yield f
    settings.clear_overrides()
    settings.load(refresh=True)


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def _siblings(path) -> list[str]:
    return [p.name for p in path.parent.iterdir() if p.name != path.name]


class TestTheTokenIsNeverWorldReadable:
    def test_a_permissive_umask_does_not_widen_the_file(self, store, monkeypatch):
        old = os.umask(0o000)
        try:
            settings._write_file({"security": {"auth_token": TOKEN}})
        finally:
            os.umask(old)

        assert _mode(store) == 0o600, "the token must not be readable by anyone but the owner"

    def test_no_chmod_is_needed_for_the_file_to_end_owner_only(self, store, monkeypatch):
        # THE POINT: the old code reached this invariant through a best-effort
        # chmod under `except OSError: pass`, so on any filesystem where the call
        # failed the token stayed world-readable for good, with no log line. Now
        # the mode comes from how the file is created, so a chmod that cannot run
        # costs nothing.
        def refuse(*_a, **_k):
            raise OSError(1, "Operation not permitted")

        monkeypatch.setattr(settings.os, "chmod", refuse)
        old = os.umask(0o022)
        try:
            settings._write_file({"security": {"auth_token": TOKEN}})
        finally:
            os.umask(old)

        m = _mode(store)
        assert not m & (stat.S_IRGRP | stat.S_IROTH), f"token left readable beyond the owner: {oct(m)}"
        assert m == 0o600

    def test_a_file_left_wide_open_by_an_older_build_is_tightened(self, store):
        # A deployment that already wrote settings.json under the old code has a
        # 0o644 file on disk. os.replace carries the new 0o600 bits over it, so the
        # next write repairs it rather than preserving the mode.
        #
        # The wide-open state is reached the way the old build reached it -- a
        # plain create under the umask default -- rather than by a chmod to an
        # explicit 0o644. Two reasons, and the second is the load-bearing one:
        # no code path in this repository ever chmods a file world-readable, so a
        # literal permissive mask would be the only one in the tree; and the
        # defect never took that route. The old `write_text` + best-effort
        # `chmod` left 0o644 because of how the file was *created*, which is
        # precisely what the fix changes, so creating it that way here grades the
        # repair against the state a real deployment is in.
        old = os.umask(0o022)
        try:
            store.write_text(json.dumps({"security": {"auth_token": "OLD"}}))
        finally:
            os.umask(old)
        assert _mode(store) == 0o644, (
            "the pre-condition this cell grades was not reached: the file must start "
            f"readable beyond its owner for the repair to be observable, got {oct(_mode(store))}"
        )

        settings._write_file({"security": {"auth_token": TOKEN}})

        assert _mode(store) == 0o600
        assert json.loads(store.read_text())["security"]["auth_token"] == TOKEN

    def test_no_sibling_holding_the_token_is_left_behind(self, store):
        settings._write_file({"security": {"auth_token": TOKEN}})

        assert _siblings(store) == [], "a .tmp holding the bearer token outlived the write"


class TestAnInterruptedWriteCannotDestroyTheSettings:
    def test_a_keyboard_interrupt_leaves_the_previous_file_intact(self, store, monkeypatch):
        # KeyboardInterrupt is named because it is NOT an Exception: a cleanup
        # written as `except Exception` would leak the temp file on exactly the
        # signal an operator sends by hand.
        settings._write_file({"security": {"auth_token": "FIRST"}})

        def interrupt(*_a, **_k):
            raise KeyboardInterrupt

        monkeypatch.setattr(settings.os, "replace", interrupt)
        with pytest.raises(KeyboardInterrupt):
            settings._write_file({"security": {"auth_token": "SECOND"}})

        assert json.loads(store.read_text())["security"]["auth_token"] == "FIRST", (
            "the previous settings must survive an interrupted write"
        )
        assert _siblings(store) == [], "the abandoned temp file was not cleaned up"

    def test_an_unserialisable_payload_writes_nothing_and_leaves_nothing(self, store):
        # allow_nan=False makes this raise. The old code raised before creating the
        # temp file, so this cell passes there too; it is here because the new
        # order creates the file first and must still clean up.
        settings._write_file({"security": {"auth_token": "FIRST"}})

        with pytest.raises(ValueError):
            settings._write_file({"agent": {"temperature": float("nan")}})

        assert json.loads(store.read_text())["security"]["auth_token"] == "FIRST"
        assert _siblings(store) == []

    def test_a_reader_never_sees_a_partial_file(self, store, monkeypatch):
        # The rename is what buys this: the payload is complete in the sibling
        # before it is moved into place, so there is no window in which the path
        # holds a prefix of the new contents.
        settings._write_file({"security": {"auth_token": "FIRST"}})
        seen: list[str] = []
        real_replace = os.replace

        def observe(src, dst):
            seen.append(store.read_text())
            return real_replace(src, dst)

        monkeypatch.setattr(settings.os, "replace", observe)
        settings._write_file({"security": {"auth_token": TOKEN}})

        assert seen == [json.dumps({"security": {"auth_token": "FIRST"}}, indent=2, sort_keys=True)], (
            "the destination changed before the rename, so a reader could see a prefix"
        )
        assert json.loads(store.read_text())["security"]["auth_token"] == TOKEN
