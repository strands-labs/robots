"""Keep credentials out of the log file. A browser cannot set headers on a WebSocket handshake, so
every camera and mesh socket carries its JWT in the QUERY STRING - and uvicorn's access log
writes the request line verbatim.
"""

from __future__ import annotations

import logging
import re

#: query parameters whose VALUE is a credential
_SECRET_QUERY_KEYS = ("token", "access_code", "api_key", "apikey", "password", "secret")
# : `code` is an oauth credential AND the commonest word in an HTTP log.
_LONG_ONLY_QUERY_KEYS = ("code",)

_QUERY_RE = re.compile(
    r"(?P<key>\b(?:" + "|".join(_SECRET_QUERY_KEYS) + r")=)(?P<val>[^\s&\"'#]+)",
    re.IGNORECASE,
)
_LONG_QUERY_RE = re.compile(
    r"(?P<key>\b(?:" + "|".join(_LONG_ONLY_QUERY_KEYS) + r")=)(?P<val>[^\s&\"'#]{8,})",
    re.IGNORECASE,
)
#: `Authorization: Bearer xyz`, and the bare `Bearer xyz` some clients log
_BEARER_RE = re.compile(r"(?i)(?P<key>bearer\s+)(?P<val>[A-Za-z0-9._\-~+/]{8,}=*)")
#: a JWT sitting loose in a message, with no key to hang the redaction on
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\b")

_KEYED_WORDS = (
    "token",
    "secret",
    "password",
    "passwd",
    "passphrase",
    "api_key",
    "apikey",
    "access_code",
    "auth",
    "authorization",
    "credential",
)
_KEYED_RE = re.compile(
    r"(?i)(?P<key>[\w.\-]*(?:" + "|".join(_KEYED_WORDS) + r")[\w.\-]*\"?'?\s*[:=]\s*\"?'?)"
    r"(?P<val>[A-Za-z0-9._\-~+/]{8,}=*)"
)

# : Rail 2, and the one that cannot be out-guessed: the process's OWN credentials, registered
# as : literals when they are loaded.
_known: set[str] = set()


def register_secret(value: str | None) -> None:
    """Redact ``value`` from every future log line, whatever shape it appears in."""
    if value and len(value.strip()) >= 12:
        _known.add(value.strip())


def forget_secrets() -> None:
    """Test-only: drop the registered literals (a leaked registration outlives one test)."""
    _known.clear()


def fingerprint(secret: str) -> str:
    """A stable, non-usable label for a credential: its length and last 4 characters."""
    tail = secret[-4:] if len(secret) >= 8 else ""
    return f"<redacted:{len(secret)}{':' + tail if tail else ''}>"


def redact_secrets(message: str) -> str:
    """Return ``message`` with every credential-shaped value replaced by a fingerprint."""
    if not message:
        return message

    def _q(m: re.Match[str]) -> str:
        return m.group("key") + fingerprint(m.group("val"))

    out = _QUERY_RE.sub(_q, message)
    out = _LONG_QUERY_RE.sub(_q, out)
    # The keyed rail runs AFTER the query rail so `?token=x` keeps its narrower, well-tested
    # handling (it must stop at & and #, which a generic value pattern does not know about).
    out = _KEYED_RE.sub(_q, out)
    out = _BEARER_RE.sub(lambda m: m.group("key") + fingerprint(m.group("val")), out)
    out = _JWT_RE.sub(lambda m: fingerprint(m.group(0)), out)
    # Rail 2 LAST, deliberately: a fingerprint's own text ("<redacted:43:8aco>") matches the value
    # pattern, so replacing literals FIRST let the keyed rail redact the LABEL and print a wrong
    # length ("?token=<redacted:18:aco>>").
    for secret in sorted(_known, key=len, reverse=True):
        if secret in out:
            out = out.replace(secret, fingerprint(secret))
    return out


class RedactingFilter(logging.Filter):
    """Redacts the FORMATTED message of every record that passes through."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            original = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken record must not break logging
            return True
        cleaned = redact_secrets(original)
        if cleaned != original:
            record.msg = cleaned
            record.args = ()
        return True


#: loggers that carry request lines; the root catches everything else
_TARGETS = ("", "uvicorn", "uvicorn.access", "uvicorn.error", "fastapi", "strands_robots")


def install_redaction(logger_names: tuple[str, ...] = _TARGETS) -> None:
    """Attach the filter to the loggers that can carry a URL. Idempotent."""
    for name in logger_names:
        logger = logging.getLogger(name)
        if not any(isinstance(f, RedactingFilter) for f in logger.filters):
            logger.addFilter(RedactingFilter())
        # uvicorn attaches its own handlers with their own filter chain; a handler-level
        # filter is what actually catches records logged directly to those handlers
        for handler in logger.handlers:
            if not any(isinstance(f, RedactingFilter) for f in handler.filters):
                handler.addFilter(RedactingFilter())
