"""A credential store that cannot be parsed must not quietly unseal the dashboard.

`auth_enabled()` IS `has_credentials()`, and `_load()` used to answer an unreadable store by
writing a fresh default one over it. So one truncated write -- a crash mid-save, a full disk --
did two things nobody would see: it dropped auth on every /api and /ws route (through the
tunnel: the public internet), and it destroyed the only record of the operator's passkey, so
even repairing the JSON by hand could not bring it back.

These tests pin the recovery posture instead: keep the bytes, say so, and let the person AT the
machine re-enroll while a stranger who merely benefited from a disk error cannot.
"""

import json
import logging
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from strands_robots.dashboard import auth


class FakeRequest:
    """A request as it arrived: over a SCHEME, carrying headers.

    The scheme is a property of the connection, not of a header, so a stand-in
    that answers headers alone cannot represent one -- and an expectation
    derived from it would look right here while being caller-controlled in
    production.
    """

    def __init__(self, headers=None, client_host="127.0.0.1", scheme="http"):
        self.headers = headers or {"host": "localhost:8090"}
        self.client = type("C", (), {"host": client_host})()
        self.url = SimpleNamespace(scheme=scheme)


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    for k in ("STRANDS_DASH_AUTH_ENABLED", "STRANDS_DASH_AUTH_RP_ID", "STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    auth._cache = {}
    auth._corrupt = None
    yield
    auth._corrupt = None


def _corrupt_store(tmp_path, body: str = '{"credentials": [{"id": "AAA'):
    path = tmp_path / "auth.json"
    path.write_text(body)  # truncated JSON: exactly what a killed process leaves
    return path


def test_the_unreadable_bytes_are_kept_not_clobbered(tmp_path):
    path = _corrupt_store(tmp_path)
    original = path.read_text()

    auth._load()

    backups = list(tmp_path.glob("auth.json.corrupt-*"))
    assert len(backups) == 1, "the operator's only credential record must survive"
    assert backups[0].read_text() == original, "kept verbatim -- it may hold the credential id"
    # And the working store is valid again, so the dashboard still comes up.
    assert json.loads(path.read_text())["credentials"] == []


def test_the_corruption_is_reported_not_swallowed(tmp_path):
    _corrupt_store(tmp_path)
    auth._load()

    damage = auth.store_corruption()
    assert damage, "a silent recovery is how this stayed invisible"
    assert "corrupt-" in damage["backup"]
    assert "Error" in damage["reason"] or "Decode" in damage["reason"]


def test_a_healthy_store_reports_no_damage(tmp_path):
    auth._load()
    assert auth.store_corruption() is None
    assert not list(tmp_path.glob("auth.json.corrupt-*"))


def test_a_stranger_cannot_seize_the_dashboard_through_a_disk_error(tmp_path):
    _corrupt_store(tmp_path)
    auth._load()

    with pytest.raises(HTTPException) as e:
        auth.begin_registration(FakeRequest(client_host="203.0.113.9"), label="attacker")
    assert e.value.status_code == 403
    # The refusal must say what happened and where the bytes went -- an operator reading only
    # this message has to be able to recover.
    assert "unreadable" in e.value.detail and "corrupt-" in e.value.detail
    assert "BOOTSTRAP_TOKEN" in e.value.detail


def test_the_person_at_the_machine_can_still_recover(tmp_path):
    _corrupt_store(tmp_path)
    auth._load()

    opts = auth.begin_registration(FakeRequest(client_host="127.0.0.1"), label="recovery")
    assert opts.get("challenge_id"), "recovery must not be a dead end for the owner"


def test_bootstrap_token_still_works_from_anywhere(tmp_path, monkeypatch):
    monkeypatch.setenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", "let-me-in")
    _corrupt_store(tmp_path)
    auth._load()

    with pytest.raises(HTTPException):
        auth.begin_registration(FakeRequest(client_host="203.0.113.9"), bootstrap="wrong")
    opts = auth.begin_registration(FakeRequest(client_host="203.0.113.9"), bootstrap="let-me-in")
    assert opts.get("challenge_id")


def test_a_genuinely_new_dashboard_is_gated_too_and_says_so_differently(tmp_path):
    """The fresh-install case is gated on the same predicate, with its own wording.

    This cell used to assert the opposite -- that a first enrollment from anywhere kept
    working -- which was this file's scope discipline rather than a finding that the normal
    path was safe. It is not: a fresh store is the case an attacker most wants, because
    nothing has to break first. Both routes into a first enrollment now refuse a caller who
    is not at the machine; only the diagnosis differs, and neither mentions the other's cause.
    """
    with pytest.raises(HTTPException) as e:
        auth.begin_registration(FakeRequest(client_host="203.0.113.9"), label="fresh")
    assert e.value.status_code == 403
    # The wording must fit the cause: no disk error happened here, so it must not claim one.
    assert "unreadable" not in e.value.detail and "corrupt-" not in e.value.detail
    assert "BOOTSTRAP_TOKEN" in e.value.detail
    assert auth.store_corruption() is None


class TestTheLastNineLines:
    """The defensive branches of this module, which had no coverage at all (2026-08-21).

    Each of these is a filesystem or request failing in a way that is rare and consequential: the
    module's whole promise is that auth cannot be dropped by an accident, and an untested `except`
    is exactly where that promise is usually broken.
    """

    def test_a_quarantine_that_CANNOT_move_the_file_still_records_the_damage(self, tmp_path, monkeypatch):
        """The flag, not the backup, is what re-seals enrollment.

        On a read-only or full volume the rename fails - and if the code gave up there, the
        credential-less window would open with nobody recorded as responsible: a stranger through the
        tunnel could seize a dashboard whose owner still has a passkey on disk. So the move is
        best-effort and the FLAG is mandatory.
        """
        _corrupt_store(tmp_path)
        # Scoped to the quarantine rename. `_save_locked` replaces the store through
        # `os.replace` too, and failing that one as well would be a different scenario
        # from the one this cell is about - a volume where the re-seed cannot land is a
        # volume where there is no working store to come up on, which the enclosing
        # `_load` is right to raise about rather than paper over.
        real_replace = auth.os.replace

        def replace_unless_it_is_the_quarantine(src, dst, *a, **k):
            if ".corrupt-" in str(dst):
                raise OSError(30, "Read-only file system")
            return real_replace(src, dst, *a, **k)

        monkeypatch.setattr(auth.os, "replace", replace_unless_it_is_the_quarantine)
        auth._cache = {}
        auth._load()
        damage = auth.store_corruption()
        assert damage, "an unmovable corrupt store must still be REPORTED as corrupt"
        assert damage["backup"] == "", "and honest that no backup file exists"
        assert "JSONDecodeError" in damage["reason"] or "Expecting" in damage["reason"]
        # ... and the narrowing it exists for still applies, naming the missing backup gracefully.
        with pytest.raises(HTTPException) as err:
            auth.begin_registration(FakeRequest(client_host="203.0.113.9"), label="stranger")
        assert err.value.status_code == 403
        assert "a backup" in str(err.value.detail), (
            "the refusal must read sensibly when there is no backup path to name"
        )

    def test_a_store_on_a_volume_that_cannot_chmod_is_saved_and_still_owner_only(self, tmp_path, monkeypatch):
        """Refusing to save would leave such a volume unable to enroll a passkey at all.

        The store used to be written at the umask default and chmod-ed afterwards, so
        owner-only was a wish that a volume with no chmod concept could not grant. It is
        now a property of creation - `mkstemp` opens at 0600 and `os.replace` carries
        those bits onto the store - so this cell asserts both halves: the save still
        succeeds with `chmod` unavailable, and the result is *still* 0600 because
        nothing on the path needed the call that failed.
        """
        monkeypatch.setattr(
            auth.os,
            "chmod",
            lambda *a, **k: (_ for _ in ()).throw(OSError(45, "Operation not supported")),
        )
        auth._save({"credentials": [], "note": "kept"})
        path = tmp_path / "auth.json"
        assert json.loads(path.read_text())["note"] == "kept"
        assert path.stat().st_mode & 0o777 == 0o600, "owner-only comes from mkstemp, not from a chmod"

    def test_a_store_that_vanishes_between_write_and_stat_is_not_cached(self, tmp_path, monkeypatch):
        """The cache key is (path, mtime, size). If stat fails there is no key to cache under.

        Caching nothing means the next _load reads the file again - the safe direction. Serving a
        remembered store would answer for a file somebody else has since replaced, so the
        observable is exactly that: replace it behind the process's back and the new contents win.
        """
        real_stat = auth.Path.stat
        failed: list[bool] = []

        def stat_fails_once(self, *a, **k):
            if self.name == "auth.json" and not failed:
                failed.append(True)
                raise OSError(2, "No such file or directory")
            return real_stat(self, *a, **k)

        monkeypatch.setattr(auth.Path, "stat", stat_fails_once)
        auth._save({"credentials": [], "note": "written"})
        path = tmp_path / "auth.json"
        assert failed, "the post-write stat is the one that failed"
        assert json.loads(path.read_text())["note"] == "written", "the write itself still happened"

        path.write_text(json.dumps({"credentials": [], "note": "replaced"}))
        assert auth._load()["note"] == "replaced", "an un-keyed store must not be served from memory"

    def test_a_request_whose_headers_explode_still_returns_a_status(self, tmp_path):
        """status() feeds the login screen. A diagnostic that raises takes the screen with it."""

        class Hostile:
            @property
            def headers(self):
                raise RuntimeError("no headers on this transport")

        out = auth.status(Hostile())
        assert out["setup_required"] is True and out["enabled"] is False
        assert "rp_id" not in out, "an undiscoverable rp_id is absent, never guessed"

    def test_an_advisory_that_fails_halfway_contributes_nothing(self, tmp_path, monkeypatch, caplog):
        """The rp_id hints arrive together or not at all.

        `rp_id` says which relying-party id to use and `rpid_usable` says whether it
        can be used. Emitting the first without the second would tell the login
        screen to attempt an id that this module never approved, which is the one
        combination worse than having no hints - so the block is assembled aside and
        merged only once complete. Here the host resolves and the verdict then
        explodes, which is exactly the halfway point.
        """
        monkeypatch.setattr(
            auth,
            "rpid_is_usable",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("verdict unavailable")),
        )
        caplog.set_level(logging.DEBUG, logger="strands_robots.dashboard.auth")
        out = auth.status(FakeRequest())
        assert "rp_id" not in out, "a half-derived advisory must not be published"
        assert "rpid_usable" not in out and "warning" not in out
        # ... and the skip is attributable rather than silent: an operator staring at a
        # login screen with no hints has no other way to learn that deriving them failed.
        skipped = [r for r in caplog.records if "rp_id advisory" in r.getMessage()]
        assert len(skipped) == 1, "the skipped advisory must be logged once"
        assert skipped[0].exc_info is not None, "and carry the cause, not just the fact"

    def test_a_healthy_request_still_gets_its_advisory(self, tmp_path):
        """The control: the merge must not have cost the working path its hints."""
        out = auth.status(FakeRequest())
        assert out["rp_id"] == "localhost"
        assert out["rpid_usable"] is True
