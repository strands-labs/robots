# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin: an unrecognized ``STRANDS_DASH_AUTH_ENABLED`` cannot drop the gate.

``auth_enabled()`` used to read the variable as ``raw in ("1", "true", "yes",
"on")`` behind a bare ``if raw:``, so every non-empty value outside that tuple
resolved to auth-OFF -- and it did so *in preference to* the credential store.
``STRANDS_DASH_AUTH_ENABLED=enabled`` or ``=y``, the spellings an operator
reaches for, therefore removed passkey auth from every route this function
guards, on a dashboard that commands real hardware, with nothing raised and
nothing logged.

The misparse direction is what makes this a safety bug rather than an
inconvenience: an unrecognized value did not fall back to the store's verdict,
it disabled the gate. The store is the source of truth, so the only safe
reading of a value nobody can interpret is to ignore it and let an enrolled
passkey keep guarding the API.

Both vocabularies are now explicit and anything else is reported through
``logger.warning`` and ignored. These cells grade all three arms: the true
spellings, the false spellings, and the unrecognized value that must fall
through to the store while saying so.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pytest

from strands_robots.dashboard import auth


@pytest.fixture(autouse=True)
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A store of this test's own, and no inherited override."""
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("STRANDS_DASH_AUTH_RP_ID", raising=False)
    monkeypatch.delenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", raising=False)
    auth._cache = {}


def _enroll(tmp_path: Path) -> None:
    """Put one passkey in the store, so the store's own verdict is ON."""
    auth.has_credentials()  # creates the store
    path = tmp_path / "auth.json"
    data = json.loads(path.read_text())
    data["credentials"] = [{"id": "abc", "public_key": "cGs", "sign_count": 0, "name": "phone"}]
    path.write_text(json.dumps(data))
    os.utime(path, (time.time() + 2, time.time() + 2))


class TestAnUnrecognizedValueCannotDisableAnEnrolledGate:
    """The regression: a value nobody can interpret must not be read as
    "off" when a passkey is enrolled."""

    @pytest.mark.parametrize(
        "raw",
        ["enabled", "y", "n", "disabled", "True ", "sure", "2", "-1", "none", "null", "OFF!"],
    )
    def test_the_store_still_decides(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        _enroll(tmp_path)
        monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", raw)
        assert auth.auth_enabled() is True

    def test_the_unrecognized_value_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _enroll(tmp_path)
        monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "enabled")
        with caplog.at_level(logging.WARNING, logger="strands_robots.dashboard.auth"):
            assert auth.auth_enabled() is True
        assert "STRANDS_DASH_AUTH_ENABLED" in caplog.text
        assert "enabled" in caplog.text

    def test_a_recognized_value_is_not_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _enroll(tmp_path)
        monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "false")
        with caplog.at_level(logging.WARNING, logger="strands_robots.dashboard.auth"):
            assert auth.auth_enabled() is False
        assert caplog.text == ""


class TestTheRecognizedVocabulariesStillOverride:
    """The override is the point of the variable, so both directions keep
    working for every spelling the function declares."""

    @pytest.mark.parametrize("raw", ["1", "true", "yes", "on", "TRUE", "  On  "])
    def test_a_true_spelling_enables_an_empty_store(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", raw)
        assert auth.auth_enabled() is True

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", "FALSE", "  Off  "])
    def test_a_false_spelling_disables_an_enrolled_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        _enroll(tmp_path)
        monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", raw)
        assert auth.auth_enabled() is False

    def test_the_two_vocabularies_do_not_overlap(self) -> None:
        assert not set(auth._ENABLED_TRUE) & set(auth._ENABLED_FALSE)


class TestAnAbsentOrBlankValueReadsTheStore:
    """Unset and whitespace-only were already store-deferring; that half
    must not move."""

    @pytest.mark.parametrize("raw", ["", "   ", "\t"])
    def test_a_blank_value_defers_to_the_store(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", raw)
        assert auth.auth_enabled() is False
        _enroll(tmp_path)
        assert auth.auth_enabled() is True

    def test_an_unset_variable_defers_to_the_store(self, tmp_path: Path) -> None:
        assert auth.auth_enabled() is False
        _enroll(tmp_path)
        assert auth.auth_enabled() is True
