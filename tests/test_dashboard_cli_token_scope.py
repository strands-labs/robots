"""A token handed to ONE dashboard must not become the machine's default (Q49).

Measured 2026-08-20: rehearsing a restart on port 8099 with --auth-token rewrote
`security.auth_token` in the shared ~/.strands_robots/dashboard/settings.json - twice -
so the LIVE dashboard on 8090 would have come up at its next restart demanding a token
nobody had. Every CLI flag was persisted by design ("flags behave like the same field set
in the UI"); that is right for a mesh port and wrong for a secret, because a second
instance is a normal thing to run and losing the credential is not a normal cost.
"""

from __future__ import annotations

import json

import pytest

from strands_robots.dashboard import settings


@pytest.fixture()
def store(tmp_path, monkeypatch):
    f = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "SETTINGS_FILE", f)
    settings.clear_overrides()
    settings.load(refresh=True)
    yield f
    settings.clear_overrides()
    settings.load(refresh=True)


def _stored(f) -> dict:
    return json.loads(f.read_text()) if f.exists() else {}


def test_override_wins_over_the_file_but_never_reaches_it(store):
    settings.update({"security": {"auth_token": "STORED"}})
    settings.override("security", "auth_token", "RUNONLY")

    assert settings.load(refresh=True)["security"]["auth_token"] == "RUNONLY", (
        "the override must beat the file - relying on the env layer would lose to a "
        "stored value and leave the flag with no effect at all"
    )
    assert _stored(store)["security"]["auth_token"] == "STORED", "the file must be untouched"

    settings.clear_overrides()
    assert settings.load(refresh=True)["security"]["auth_token"] == "STORED"


def test_override_is_visible_without_an_explicit_refresh(store):
    settings.update({"security": {"auth_token": "STORED"}})
    settings.load()  # warm the cache the way a real process does
    settings.override("security", "auth_token", "RUNONLY")
    assert settings.get("security", "auth_token") == "RUNONLY", "override must invalidate the cache"


def test_override_ignores_keys_that_are_not_in_the_schema(store):
    settings.override("security", "not_a_real_key", "x")
    settings.override("nosuchsection", "auth_token", "x")
    tree = settings.load(refresh=True)
    assert "not_a_real_key" not in tree["security"]
    assert "nosuchsection" not in tree


def test_the_first_token_on_a_fresh_machine_is_still_saved(store):
    """No stored secret means nothing to destroy, and persisting is pure gain."""
    assert not _stored(store).get("security", {}).get("auth_token")
    settings.update({"security": {"auth_token": "FIRST"}})
    assert _stored(store)["security"]["auth_token"] == "FIRST"
