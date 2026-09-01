"""The cached store is reachable only under the identity of the file it came from.

`_load` caches so that a login does not re-read and re-parse the credential store
on every request. What it caches under decides whether a hit can ever be wrong.

The cache was a value global (`_cache`) beside a parallel key global
(`_cache_key`), which made "these two describe the same file" an invariant kept
by hand at both writers. Two globals can disagree, and when they do a stale hit
is indistinguishable from a fresh one -- on the file that decides whether this
dashboard is sealed. Keying the store *by* its file identity removes the class
rather than the instance: a value is only reachable through the identity it was
stored under, so there is no second variable to fall out of step.

`strands_robots/mesh/_acl_config.py` keys its ACL cache on a file identity tuple
for the same reason, and it also names the cost that comes with the shape: every
rewrite of the file mints a *new* key, so an identity-keyed cache grows unless it
is bounded. That file caps at four entries. Here exactly one store path is live
per process, so the bound is one entry -- and these cells grade both halves of
that: the entry is replaced rather than accumulated, and two different store
paths never answer for each other.

The re-read-on-change and read-from-memory-on-hit behaviours are graded by
`test_dashboard_auth_module.py::test_store_hot_reloads_on_file_change` and
`test_dashboard_auth_store_write_is_atomic.py::TestWhatTheAtomicWriteDoesNotChange`
and are deliberately not repeated here.
"""

import json

import pytest

from strands_robots.dashboard import auth


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    for key in ("STRANDS_DASH_AUTH_ENABLED", "STRANDS_DASH_AUTH_RP_ID", "STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    auth._cache = {}
    auth._corrupt = None
    yield
    auth._corrupt = None


class TestTheCacheIsBounded:
    """One store path per process, so one entry -- however often the file is written."""

    def test_rewriting_the_store_does_not_accumulate_entries(self, tmp_path):
        """Every save changes mtime/size, so every save mints a fresh key.

        Without replacement, the deployment's most frequent write -- the `sign_count`
        persisted by every successful authentication -- would be a memory-growth lever
        for anyone who can drive logins.
        """
        for i in range(8):
            store = auth._load()
            store["note"] = f"write-{i}"
            auth._save(store)
            assert auth._load()["note"] == f"write-{i}", "the newest write is what is served"

        assert len(auth._cache) == 1, f"8 saves left {len(auth._cache)} cached entries, want 1"

    def test_a_re_read_after_an_outside_change_does_not_accumulate_entries(self, tmp_path):
        """The other writer is `_load` itself, on the miss that follows an outside edit."""
        path = tmp_path / "auth.json"
        auth._save({"credentials": [], "jwt_secret": "s", "note": "first"})

        for i in range(8):
            path.write_text(json.dumps({"credentials": [], "jwt_secret": "s", "note": f"outside-{i}"}))
            assert auth._load()["note"] == f"outside-{i}", "an outside change is re-read"

        assert len(auth._cache) == 1, f"8 outside changes left {len(auth._cache)} cached entries, want 1"


class TestOneStoreNeverAnswersForAnother:
    """The path is part of the identity, so a store move cannot serve the old contents."""

    def test_repointing_the_store_does_not_serve_the_previous_file(self, tmp_path, monkeypatch):
        """An operator moving `STRANDS_DASH_AUTH_STORE` is repointing at other credentials.

        Serving the first file's store for the second would carry a passkey set, and the
        `jwt_secret` that signs sessions, across a boundary the operator drew on purpose.
        """
        auth._save({"credentials": [{"id": "FIRST"}], "jwt_secret": "first-secret"})
        assert auth._load()["jwt_secret"] == "first-secret"

        second = tmp_path / "elsewhere" / "auth.json"
        second.parent.mkdir()
        second.write_text(json.dumps({"credentials": [{"id": "SECOND"}], "jwt_secret": "second-secret"}))
        monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(second))

        loaded = auth._load()
        assert loaded["jwt_secret"] == "second-secret", "the new path's own store is what is read"
        assert [c["id"] for c in loaded["credentials"]] == ["SECOND"]

    def test_a_store_that_does_not_exist_yet_is_seeded_rather_than_inherited(self, tmp_path, monkeypatch):
        """A fresh path has no file, so it has no identity: it must not inherit a cached store."""
        auth._save({"credentials": [{"id": "FIRST"}], "jwt_secret": "first-secret"})

        monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "fresh" / "auth.json"))
        loaded = auth._load()

        assert loaded["credentials"] == [], "a fresh store starts with no passkeys"
        assert loaded["jwt_secret"] != "first-secret", "and with a secret of its own"
