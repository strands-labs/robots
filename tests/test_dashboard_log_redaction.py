"""A log that leaks a working credential is a vulnerability, not a log.

Measured on the live dashboard: 63,000 access-log lines, each carrying a complete valid
JWT in a WebSocket query string, in a 21 MB world-readable file in /tmp - for a
dashboard published through a public tunnel that can drive real arms.
"""

from __future__ import annotations

import logging

import pytest

from strands_robots.dashboard.log_redaction import (
    RedactingFilter,
    fingerprint,
    forget_secrets,
    install_redaction,
    redact_secrets,
    register_secret,
)

REAL_LINE = (
    '2600:4041:4256:7e00:0 - "WebSocket /ws/camera/so101-arm-1/top?'
    "token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJyLVFOMmNiVDlIRXEt."
    'W1L2czgAYPYQO-zzCc-0C-HEDGwliiChAZR9jZuScQE" [accepted]'
)


class TestTheRealLine:
    def test_the_token_is_gone(self) -> None:
        out = redact_secrets(REAL_LINE)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in out
        assert "HEDGwliiChAZR9jZuScQE" not in out

    def test_the_line_is_still_a_useful_log_line(self) -> None:
        # the whole point of the access log is answering "which socket, from whom,
        # how many times" - redaction must not take that away
        out = redact_secrets(REAL_LINE)
        assert "/ws/camera/so101-arm-1/top" in out
        assert "2600:4041:4256:7e00:0" in out
        assert "[accepted]" in out
        assert "token=" in out, "the parameter NAME is not a secret and its absence hides the shape"

    def test_two_different_tokens_stay_distinguishable(self) -> None:
        a = redact_secrets("?token=" + "a" * 40 + "wxyz")
        b = redact_secrets("?token=" + "b" * 40 + "abcd")
        assert a != b


class TestWhatCountsAsASecret:
    def test_every_credential_query_key(self) -> None:
        for key in ("token", "access_code", "api_key", "password", "secret"):
            out = redact_secrets(f"GET /x?{key}=supersecretvalue123 HTTP/1.1")
            assert "supersecretvalue123" not in out, key

    def test_a_bearer_header(self) -> None:
        assert "kDD9toTMVDwOXYn5XfDI0v9nKGC6tSM8xPKcNaco" not in redact_secrets(
            "Authorization: Bearer kDD9toTMVDwOXYn5XfDI0v9nKGC6tSM8xPKcNaco"
        )

    def test_a_loose_jwt_with_no_key_to_hang_on(self) -> None:
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjYWdhdGF5In0.QWxsWW91ck1lc2g"
        assert jwt not in redact_secrets(f"auth failed for {jwt} from 10.0.0.5")

    def test_case_does_not_help_a_leak(self) -> None:
        assert "LEAKYVALUE" not in redact_secrets("?TOKEN=LEAKYVALUE")

    def test_ordinary_query_parameters_are_untouched(self) -> None:
        line = 'GET /api/training/datasets?q=so101&limit=12 HTTP/1.1" 200 OK'
        assert redact_secrets(line) == line

    def test_a_path_that_merely_contains_the_word_token_is_untouched(self) -> None:
        line = 'GET /api/auth/bootstrap_token HTTP/1.1" 200 OK'
        assert redact_secrets(line) == line

    def test_empty_and_none_ish_input_is_safe(self) -> None:
        assert redact_secrets("") == ""


class TestFingerprint:
    def test_it_reports_length_and_a_tail(self) -> None:
        fp = fingerprint("abcdefghijkl")
        assert "12" in fp and "ijkl" in fp

    def test_a_short_secret_gets_no_tail(self) -> None:
        # 4 of 200 JWT characters is not a credential; 4 of 6 might be
        assert fingerprint("abc123") == "<redacted:6>"


class TestFilterInstallation:
    def test_a_record_is_redacted_in_place(self, caplog) -> None:
        logger = logging.getLogger("test.redaction.inplace")
        logger.addFilter(RedactingFilter())
        with caplog.at_level(logging.INFO, logger="test.redaction.inplace"):
            logger.info("opened ?token=%s", "eyJa.bcdefgh.ijklmnop")
        assert "eyJa.bcdefgh.ijklmnop" not in caplog.text
        assert "token=" in caplog.text

    def test_install_is_idempotent(self) -> None:
        install_redaction(("test.redaction.twice",))
        install_redaction(("test.redaction.twice",))
        filters = logging.getLogger("test.redaction.twice").filters
        assert sum(isinstance(f, RedactingFilter) for f in filters) == 1

    def test_a_broken_record_does_not_break_logging(self) -> None:
        class _Bad:
            def __str__(self) -> str:
                raise RuntimeError("nope")

        record = logging.LogRecord("x", logging.INFO, __file__, 1, "%s", (_Bad(),), None)
        assert RedactingFilter().filter(record) is True


# --- Q117: the shapes the SHAPE-BASED rules missed ---------------------------------------------
#
# MEASURED against this machine's live token in nine realistic log lines: FIVE printed it verbatim.
# Every fixture above shares one incidental property - the secret sits after `key=` or `Bearer ` -
# so these rules were being tested against that property rather than against "a credential must not
# reach a log". That is the Q116 law a second time, and here it costs a live token instead of a
# cache header.
FAKE = "kDD6toTMVDwOXYn51XfDI0vNnKGC4tSM5xP5c858aco"


@pytest.mark.parametrize(
    "line",
    [
        f"env STRANDS_DASHBOARD_TOKEN={FAKE} inherited by child",  # prefixed key, `=`
        f"curl -H 'X-Auth-Token: {FAKE}' localhost:8090",  # prefixed key, `:`
        f'{{"token": "{FAKE}"}}',  # JSON body echoed into a log
        f'headers={{"authorization": "{FAKE}"}}',  # no "Bearer " to hang it on
        f"api_key = {FAKE}",  # spaces around the separator
    ],
)
def test_a_credential_is_redacted_whatever_holds_it(line: str) -> None:
    assert FAKE not in redact_secrets(line)


def test_a_registered_literal_is_redacted_in_a_shape_nobody_predicted() -> None:
    """The rail that cannot be out-guessed: argv and prose have no key at all."""
    argv = f"spawn argv: ['python', '-m', 'strands_robots.dashboard', '--token', '{FAKE}']"
    prose = f"wrote the token to ~/.strands_dashboard/local_api_token.txt ({FAKE})"
    try:
        assert FAKE in redact_secrets(prose)  # no key, no pattern: unredactable by shape alone
        register_secret(FAKE)
        for line in (argv, prose):
            out = redact_secrets(line)
            assert FAKE not in out
            assert "<redacted:43:8aco>" in out  # the fingerprint keeps the log useful
    finally:
        forget_secrets()


def test_a_registered_literal_does_not_corrupt_the_fingerprint_of_a_keyed_value() -> None:
    """Order matters: literals LAST.

    A fingerprint's own text matches the value pattern, so replacing literals first made the keyed
    rail redact the LABEL and print a wrong length: `?token=<redacted:18:aco>>`.
    """
    try:
        register_secret(FAKE)
        assert redact_secrets(f"?token={FAKE}&x=1") == "?token=<redacted:43:8aco>&x=1"
    finally:
        forget_secrets()


def test_a_short_value_is_never_registered() -> None:
    """Redacting a 4-character string would scribble over ordinary words in every line."""
    try:
        register_secret("abc")
        register_secret("")
        register_secret(None)
        assert redact_secrets("abc is a note about abcdef") == "abc is a note about abcdef"
    finally:
        forget_secrets()


def test_an_http_status_code_survives_but_an_oauth_code_does_not() -> None:
    """`code` is a credential key AND the commonest word in an HTTP log.

    Measured before this split: `response code=404` was logged as `code=<redacted:3>`, which hides
    the one thing that line exists to say.
    """
    assert redact_secrets("response code=404 detail='no endpoint'") == "response code=404 detail='no endpoint'"
    assert redact_secrets("HTTP status code: 200") == "HTTP status code: 200"
    assert "4/0AY0e-g7xQoauthgrant" not in redact_secrets("code=4/0AY0e-g7xQoauthgrant")
