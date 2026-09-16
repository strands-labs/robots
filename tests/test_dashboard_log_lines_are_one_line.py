"""A request header is chosen by whoever sends the request, and a log file is
parsed by line. `log_redaction.one_line` is the step between the two: every
dashboard log statement that quotes a header value goes through it, so a CRLF
inside Host, Origin or X-Forwarded-For cannot forge a second log entry.
"""

from __future__ import annotations

import json
import logging
import os
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException  # noqa: E402
from webauthn.helpers import bytes_to_base64url  # noqa: E402

from strands_robots.dashboard import auth, settings  # noqa: E402
from strands_robots.dashboard.log_redaction import one_line  # noqa: E402

FORGED_SECOND_LINE = "WARNING strands_robots.dashboard.auth: passkey enrolled for root"
FORGED = "evil.example\r\n" + FORGED_SECOND_LINE


class TestOneLine:
    def test_a_crlf_becomes_its_escape_and_the_forged_line_stays_inside_the_value(self) -> None:
        line = one_line(FORGED)
        assert "\r" not in line and "\n" not in line
        assert line == "evil.example\\r\\nWARNING strands_robots.dashboard.auth: passkey enrolled for root"

    def test_other_control_characters_are_escaped_not_dropped(self) -> None:
        assert one_line("a\x1b[31mred\x00") == "a\\x1b[31mred\\x00"

    def test_a_plain_value_is_untouched(self) -> None:
        assert one_line("203.0.113.9") == "203.0.113.9"
        assert (
            one_line("agent: expected a mapping of agent keys, got str")
            == "agent: expected a mapping of agent keys, got str"
        )

    def test_an_endless_value_is_cut(self) -> None:
        assert len(one_line("h" * 10_000)) == 200
        assert one_line("h" * 10_000).endswith("…")

    def test_any_object_is_accepted(self) -> None:
        assert one_line(None) == "None"
        assert one_line(42) == "42"


class _Url:
    scheme = "http"


class _Req:
    def __init__(self, **headers: str) -> None:
        self.headers = headers
        self.client = None
        self.cookies: dict[str, str] = {}
        self.url = _Url()


class TestTheDashboardLogsThroughIt:
    def test_a_refused_origin_is_logged_on_one_line(self, caplog, monkeypatch) -> None:
        monkeypatch.delenv("STRANDS_DASH_AUTH_ORIGIN", raising=False)
        monkeypatch.delenv("STRANDS_DASH_AUTH_RP_ID", raising=False)
        request = _Req(host="localhost:8090", origin="http://" + FORGED)
        with caplog.at_level(logging.WARNING, logger="strands_robots.dashboard.auth"), pytest.raises(HTTPException):
            auth._derive_origin(request)
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "\n" not in message and "\r" not in message
        assert "evil.example" in message

    def test_a_lenient_settings_patch_that_goes_nowhere_is_logged_on_one_line(
        self, caplog, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
        settings.clear_overrides()
        settings.load(refresh=True)
        with caplog.at_level(logging.WARNING, logger="strands_robots.dashboard.settings"):
            settings.update({"agent": "not a mapping\r\nforged"})
        assert len(caplog.records) == 1
        assert "\n" not in caplog.records[0].getMessage()

    def test_a_patched_sections_type_name_cannot_forge_a_second_entry(self, caplog, tmp_path, monkeypatch) -> None:
        """The message above quotes one caller-derived token: `type(values).__name__`.

        The cell above puts the CRLF in the value, which this message never renders, so
        it passes whether or not the statement goes through the step. The type name is
        the half that is rendered - and this is the lenient path, whose callers are the
        settings file, the environment and the CLI, so the object is any Python object
        the caller hands the store, and a class made by `type()` names itself whatever
        it likes. Over HTTP the name can only be one the JSON decoder chose.
        """
        monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
        settings.clear_overrides()
        settings.load(refresh=True)
        forger = type("str\r\n" + FORGED_SECOND_LINE, (), {})
        with caplog.at_level(logging.WARNING, logger="strands_robots.dashboard.settings"):
            changed = settings.update({"agent": forger()})
        assert changed == []
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert len(message.splitlines()) == 1
        assert FORGED_SECOND_LINE not in message.splitlines()

    def test_a_forwarded_address_cannot_forge_the_challenge_cap_entry(self, caplog, monkeypatch) -> None:
        """X-Forwarded-For is quoted with %s, so its bytes reach the line as they arrived.

        The refusal reasons above are built with `!r`, which escapes its own control
        characters; this is the header site where nothing did, and the entry a flooding
        client provokes is the one an operator reads while being flooded.
        """
        monkeypatch.setattr(auth, "_challenges", {})
        monkeypatch.setattr(auth, "_CHAL_MAX_PER_IP", 1)
        ip = "203.0.113.9\r\n" + FORGED_SECOND_LINE
        with caplog.at_level(logging.WARNING, logger="strands_robots.dashboard.auth"):
            auth._stash_challenge("reg", b"c", {}, ip=ip)
            auth._stash_challenge("reg", b"c", {}, ip=ip)
        (message,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("challenge cap")]
        assert message.splitlines() == [message]
        assert "203.0.113.9\\r\\n" + FORGED_SECOND_LINE in message

    @pytest.mark.parametrize(("header", "door"), [("host", "_derive_rp_id"), ("origin", "_derive_origin")])
    def test_a_refused_ceremony_cannot_fill_the_log_file(self, header, door, caplog, monkeypatch) -> None:
        """A header is bounded by the cut, not by whatever the caller decided to send."""
        monkeypatch.delenv("STRANDS_DASH_AUTH_ORIGIN", raising=False)
        monkeypatch.delenv("STRANDS_DASH_AUTH_RP_ID", raising=False)
        monkeypatch.setattr(auth, "known_rp_ids", lambda store=None: {"good.example"})
        # The pad precedes the CRLF because _host_only cuts the Host at its first colon.
        value = "evil.example" + "x" * 500 + "\r\n" + FORGED_SECOND_LINE
        request = _Req(host=value) if header == "host" else _Req(host="localhost:8090", origin="http://" + value)
        with caplog.at_level(logging.WARNING, logger="strands_robots.dashboard.auth"), pytest.raises(HTTPException):
            getattr(auth, door)(request)
        (message,) = [r.getMessage() for r in caplog.records]
        assert message.splitlines() == [message]
        assert len(message) <= len("refused WebAuthn ceremony: ") + 200


CRED_ID = bytes_to_base64url(b"\x01" * 16)


def _enroll_with_label(monkeypatch, request, label: str) -> None:
    """Enrol one passkey whose display name is `label`, verifier stubbed."""
    begun = auth.begin_registration(request, label=label, bootstrap=auth._local_enroll_token())
    monkeypatch.setattr(
        auth,
        "verify_registration_response",
        lambda **kw: SimpleNamespace(credential_id=b"\x01" * 16, credential_public_key=b"\x02" * 32, sign_count=7),
    )
    auth.finish_registration(request, begun["challenge_id"], {"id": CRED_ID})


class TestAValuePersistedInTheStoreIsStillCallerSupplied:
    """The label an enrolling request chose is written to the credential store and
    read back at the next login. A taint tracker loses it at the file - CodeQL
    raised no alert for this line - but the value still arrived from outside, and
    the self-heal log statement is the one place it is quoted.
    """

    def test_a_passkey_label_cannot_forge_the_rp_id_selfheal_entry(self, monkeypatch, tmp_path, caplog) -> None:
        monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
        monkeypatch.delenv("STRANDS_DASH_AUTH_RP_ID", raising=False)
        monkeypatch.delenv("STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN", raising=False)
        auth._cache = {}
        request = _Req(host="localhost:8090")
        _enroll_with_label(monkeypatch, request, "owner\r\n" + FORGED_SECOND_LINE)

        # Erase the binding, as a credential enrolled before rp_ids were recorded
        # has none - that is the only path to the self-heal statement.
        store = tmp_path / "auth.json"
        data = json.loads(store.read_text())
        del data["credentials"][0]["rp_id"]
        store.write_text(json.dumps(data))
        os.utime(store, (time.time() + 2, time.time() + 2))
        auth._cache = {}

        begun = auth.begin_authentication(request)
        monkeypatch.setattr(auth, "verify_authentication_response", lambda **kw: SimpleNamespace(new_sign_count=8))
        with caplog.at_level(logging.INFO, logger="strands_robots.dashboard.auth"):
            auth.finish_authentication(request, begun["challenge_id"], {"id": CRED_ID})

        recorded = [r.getMessage() for r in caplog.records if r.getMessage().startswith("recorded rp_id")]
        assert len(recorded) == 1
        message = recorded[0]
        assert message.splitlines() == [message]
        assert FORGED_SECOND_LINE in message  # escaped, not dropped: the bytes stay legible


class TestTheNameOnTheEStopEntryIsCallerSupplied:
    """`/api/safety/estop` records whoever pressed it, and for a passkey caller that
    is the label the enrolling request chose - the same stored-then-read-back value
    the class above covers, at the one dashboard statement that quotes it.
    """

    def test_a_passkey_label_cannot_forge_the_estop_entry(self, caplog) -> None:
        pytest.importorskip("fastapi")
        from strands_robots.dashboard.routes_sim import Safety
        from strands_robots.dashboard.sim_session import SessionStore

        safety = Safety(SessionStore())
        with caplog.at_level(logging.WARNING, logger="strands_robots.dashboard.routes_sim"):
            safety.estop(by="owner\r\n" + FORGED_SECOND_LINE)

        (message,) = [r.getMessage() for r in caplog.records]
        assert message.splitlines() == [message]
        assert FORGED_SECOND_LINE in message  # escaped, not dropped: the bytes stay legible
