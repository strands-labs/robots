"""Persistent dashboard settings - the store behind ``/api/config``."""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from strands_robots.dashboard import log_redaction
from strands_robots.utils import refusal_repr

logger = logging.getLogger(__name__)

SETTINGS_FILE = Path(
    os.getenv(
        "DASHBOARD_SETTINGS_FILE",
        os.path.join(Path.home(), ".strands_robots", "dashboard", "settings.json"),
    )
).expanduser()

#: Section -> key -> (env var fallback, built-in default). ``None`` for the env
#: var means "no env fallback".
_SCHEMA: dict[str, dict[str, tuple[str | None, Any]]] = {
    "agent": {
        "model_id": ("DASHBOARD_MODEL_ID", None),
        "system_prompt": ("DASHBOARD_SYSTEM_PROMPT", None),  # None -> DEFAULT_SYSTEM_PROMPT
        "temperature": (None, None),
        "max_tokens": (None, None),
    },
    "voice": {
        "provider": ("VOICE_PROVIDER", "openai"),
        "voice_name": ("VOICE_NAME", None),
    },
    "mesh": {
        # Mesh endpoints. Empty list/None means "leave the env alone".
        "connect": ("ZENOH_CONNECT", []),
        "listen": ("ZENOH_LISTEN", []),
        "port": ("STRANDS_MESH_PORT", None),
        "backend": ("STRANDS_MESH_BACKEND", None),
        "camera_hz": ("STRANDS_MESH_CAMERA_HZ", None),
        "policy_type_allow": ("STRANDS_MESH_POLICY_TYPE_ALLOW", []),
    },
    "runtime": {
        "trust_remote_code": ("STRANDS_TRUST_REMOTE_CODE", False),
    },
    "security": {
        # When set, every /api and /ws request must present this token
        # (Authorization: Bearer <t>, X-Dashboard-Token, or ?token=).
        "auth_token": ("DASHBOARD_AUTH_TOKEN", None),
        # Comma-separated origins allowed to make BROWSER cross-origin calls. Default: none -
        # same-origin only.
        "cors_origins": ("DASHBOARD_CORS_ORIGINS", []),
    },
}

_LIST_KEYS = {
    ("mesh", "connect"),
    ("mesh", "listen"),
    ("mesh", "policy_type_allow"),
    ("security", "cors_origins"),
}

#: Keys whose value is a boolean. One owner, because the strict path and the
#: lenient degrade below both have to agree about which keys these are: a key
#: the strict path validates as a boolean and the lenient path degrades as a
#: string is the disagreement this set exists to make impossible.
_BOOL_KEYS: set[tuple[str, str]] = {
    ("runtime", "trust_remote_code"),
}

#: Keys whose value is a number, split by the shape it takes. One owner for the
#: same reason ``_BOOL_KEYS`` is one: the strict path and the lenient degrade
#: both have to agree about which keys these are, and the degrade spelled their
#: union as a second literal tuple -- so the two halves of the numeric domain
#: could drift apart, and they had, on the value neither literal names.
#: Keyed by name rather than ``(section, key)``, which is how both paths already
#: match them.
_FLOAT_KEYS: tuple[str, ...] = ("temperature", "camera_hz")
_INT_KEYS: tuple[str, ...] = ("max_tokens", "port")
_NUMERIC_KEYS: tuple[str, ...] = _FLOAT_KEYS + _INT_KEYS

_lock = threading.RLock()
# The resolved tree, cached under the path it was resolved from. A `dict | None`
# rebound through `global` -- and at two of the four write sites through
# `globals()["_cache"]`, which is the same rebinding spelled so it does not need
# the declaration -- makes "the cache describes the current SETTINGS_FILE" an
# invariant maintained by hand at every write, and the two can disagree: a
# process that repoints SETTINGS_FILE is served the tree resolved from the
# previous file, and a stale hit is indistinguishable from a fresh one. Keyed
# this way they cannot, because the path is the dict's key, so a tree is only
# reachable through the file it came from. ``strands_robots.dashboard.auth``
# keys its store cache on a
# file identity for the same reason. Cleared before each insert, so it holds at
# most one entry.
_cache: dict[str, dict[str, dict[str, Any]]] = {}
# Process-scoped values that must never reach settings.json - see override().
_overrides: dict[str, dict[str, Any]] = {}

# ----------------------------------------------------------------------
# Coercion
# ----------------------------------------------------------------------


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    if isinstance(value, (list, tuple)):
        # Entries are strings by the time they reach here: _graded grades
        # each one, so nothing is spelled into a string on its way to the store.
        return [p.strip() for p in value if p.strip()]
    return []


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


#: Public alias - other dashboard modules parse comma-separated endpoint
#: strings with the same rules the settings store uses.
def as_list(value: Any) -> list[str]:
    """Split a comma-separated string - or a sequence - into non-empty entries."""
    return _as_list(value)


def _finite_float(key: str, value: Any) -> tuple[float | None, str | None]:
    """A float key's value, or the reason this store refuses it."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None, f"{key}: {refusal_repr(value)} is not a number"
    if not math.isfinite(out):
        # json.dumps would emit bare NaN/Infinity - not JSON (RFC 8259); one
        # such write bricks the config screen for every browser forever.
        return None, f"{key}: {refusal_repr(value)} is not a finite number"
    return out, None


def _finite_int(key: str, value: Any) -> tuple[int | None, str | None]:
    """An integer key's value, refusing the non-finite numbers ``int()`` cannot convert.

    ``int(float("inf"))`` raises ``OverflowError``, which is neither of the two
    errors the conversion below catches, so a non-finite value in an integer
    slot used to escape the refusal altogether: the strict path raised out of
    :func:`update_strict` instead of reporting the key, and the lenient path
    raised out of :func:`load` -- where a process-scoped
    :func:`override` left every later read of the WHOLE tree raising for as
    long as it was set, since there is no value for the degrade to fall back
    to if the coercion never returns. ``json.loads`` accepts the bare
    ``Infinity`` token, so a request body is one of the ways such a value
    arrives; :func:`_read_file` refuses it for the file for the neighbouring
    reason, and :func:`_finite_float` refuses it for the other two numeric
    keys. This is that same refusal for the two whose shape is an integer.

    Non-finite is one class and gets one message, the wording
    :func:`_finite_float` already uses. ``int(float("nan"))`` happens to raise
    ``ValueError`` and so was reported -- but as "is not an integer", which
    describes how the value was spelled rather than what it is.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None, f"{key}: {refusal_repr(value)} is not a finite number"
    try:
        return int(value), None
    except OverflowError:
        # A non-finite that is not a ``float`` instance, so the check above does
        # not see it: ``numpy.float32("inf")`` and ``Decimal("Infinity")`` both
        # convert with an OverflowError rather than one of the two below.
        return None, f"{key}: {refusal_repr(value)} is not a finite number"
    except (TypeError, ValueError):
        return None, f"{key}: {refusal_repr(value)} is not an integer"


def _coerce(section: str, key: str, value: Any) -> Any:
    """What a lenient path (env, CLI, file) holds for ``section.key``: the graded value, or the degrade."""
    graded, reason = _graded(section, key, value)
    return graded if reason is None else _degraded(section, key, value)


def _degraded(section: str, key: str, value: Any) -> Any:
    """What a lenient path holds for a value this store refuses.

    Lenient degrade (env/CLI/file paths) must still degrade to the key's own SHAPE: a list key
    that fell back to a scalar poisons every comma-split consumer, which is worse than the empty
    default.
    """
    if (section, key) in _LIST_KEYS:
        return []
    # A boolean key degrades to False, not to the value that failed to be
    # one. Returning the raw value left a non-boolean in a boolean slot, and
    # `apply_mesh_env` publishes such a key by TRUTHINESS - so an
    # unparseable spelling was published as the literal "1", the exact
    # spelling `policies.factory._check_trust_remote_code` accepts, while
    # the same spelling reaching that gate directly is refused. A gate that
    # cannot read its own setting must stay shut, so the degrade is
    # fail-closed rather than a passthrough.
    if (section, key) in _BOOL_KEYS:
        return False
    return None if key in _NUMERIC_KEYS else value


# : What "true" and "false" may be spelled like.
_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off", "")


def _graded(section: str, key: str, value: Any) -> tuple[Any, str | None]:
    """The value this store would hold for ``section.key``, or the reason it refuses it.

    ``(value, None)`` when the store accepts *value*; ``(None, reason)`` when it does
    not. The reason is composed here and RETURNED. :func:`update_strict` hands it to
    whoever asked for the write - an HTTP client included - so it must name the key and
    the caller's own value and nothing else; text carried out on an exception is a
    rendering of an exception, which is not a thing this store puts on a wire.
    """
    if (section, key) in _LIST_KEYS:
        if value is not None and not isinstance(value, (str, list, tuple)):
            return None, f"{key}: expected a list or comma-separated string, got {type(value).__name__}"
        if isinstance(value, (list, tuple)):
            # The container's shape was graded above; the entries' shape was
            # not, and ``str(p)`` spelled whatever arrived: a ``null`` in a
            # JSON body became the endpoint ``"None"``, stored, and published
            # by apply_mesh_env() as ``ZENOH_CONNECT=None``. An entry is a
            # string or the list is refused, naming the entry and its type.
            for index, entry in enumerate(value):
                if not isinstance(entry, str):
                    return None, f"{key}: entry {index} expected a string, got {type(entry).__name__}"
        return _as_list(value), None
    if (section, key) in _BOOL_KEYS:
        if not isinstance(value, bool):
            spelled = str(value).strip().lower()
            if spelled not in _TRUTHY and spelled not in _FALSY:
                return None, f"{key}: {refusal_repr(value)} is not a boolean (use true/false)"
        return _as_bool(value), None
    if key in _FLOAT_KEYS:
        if value in (None, ""):
            return None, None
        out, reason = _finite_float(key, value)
        if out is None:
            return None, reason
        if key == "temperature" and not 0.0 <= out <= 2.0:
            return None, f"temperature: {refusal_repr(out)} is outside 0..2"
        if key == "camera_hz" and not 0.0 < out <= 240.0:
            # a publisher sleeps 1/hz between frames: 0 divides by zero,
            # negative sleeps never, huge busy-loops the camera thread.
            return None, f"camera_hz: {refusal_repr(out)} is outside (0, 240]"
        return out, None
    if key in _INT_KEYS:
        if value in (None, ""):
            return None, None
        whole, reason = _finite_int(key, value)
        if whole is None:
            return None, reason
        if key == "port" and not 1 <= whole <= 65535:
            return None, f"port: {refusal_repr(whole)} is outside 1..65535"
        if key == "max_tokens" and whole < 1:
            return None, f"max_tokens: {refusal_repr(whole)} must be at least 1"
        return whole, None
    if value is None:
        return None, None
    if isinstance(value, (dict, list, tuple, set)):
        return None, f"{key}: expected a string, got {type(value).__name__}"
    return str(value), None


# ----------------------------------------------------------------------
# Load / save
# ----------------------------------------------------------------------


def _defaults() -> dict[str, dict[str, Any]]:
    """Built-in defaults resolved through the environment."""
    out: dict[str, dict[str, Any]] = {}
    for section, keys in _SCHEMA.items():
        out[section] = {}
        for key, (env_name, default) in keys.items():
            raw = os.getenv(env_name) if env_name else None
            out[section][key] = _coerce(section, key, raw if raw not in (None, "") else default)
    return out


def _read_file() -> dict[str, Any]:
    try:
        if SETTINGS_FILE.exists():
            # parse_constant fires only for NaN/Infinity, which are not JSON; a file poisoned by an old
            # write must count as corrupt (browsers already refuse it), so it heals to defaults instead of
            # being handed back to JSON.parse forever.
            data = json.loads(
                SETTINGS_FILE.read_text(encoding="utf-8"),
                parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-finite {c}")),
            )
            if isinstance(data, dict):
                return data
    except Exception as exc:  # noqa: BLE001 - a corrupt file must not kill startup
        logger.warning("could not read %s: %s (using defaults)", SETTINGS_FILE, exc)
    return {}


def override(section: str, key: str, value: Any) -> None:
    """Set a value for THIS PROCESS ONLY - never written to settings.json. For a value the caller means
    for one run rather than forever.
    """
    with _lock:
        _overrides.setdefault(section, {})[key] = value
        _cache.clear()


def clear_overrides() -> None:
    """Drop every process-scoped override (tests, and re-reading from scratch)."""
    with _lock:
        _overrides.clear()
        _cache.clear()


def load(refresh: bool = False) -> dict[str, dict[str, Any]]:
    """Full settings tree: overrides over file values over env/defaults."""
    with _lock:
        # Not `key`: the merge loops below bind that name to schema keys, so a
        # cache key called `key` is cached under whichever setting was merged last.
        origin = str(SETTINGS_FILE)
        if not refresh:
            cached = _cache.get(origin)
            if cached is not None:
                return copy.deepcopy(cached)
        merged = _defaults()
        stored = _read_file()
        for section, values in stored.items():
            if section not in merged or not isinstance(values, dict):
                continue
            for key, value in values.items():
                if key in _SCHEMA[section]:
                    merged[section][key] = _coerce(section, key, value)
        for section, values in _overrides.items():
            for key, value in values.items():
                if section in merged and key in _SCHEMA[section]:
                    merged[section][key] = _coerce(section, key, value)
        _cache.clear()
        _cache[origin] = merged
        return copy.deepcopy(merged)


def get(section: str, key: str | None = None, default: Any = None) -> Any:
    """One settings value, or a whole section, with a default for what is unset.

    Args:
        section: Settings section name.
        key: A key within that section; omit it for the whole section.
        default: Returned when the section, the key or its value is absent - an
            empty string and an empty list count as absent.

    Returns:
        The stored value, the section mapping, or *default*.
    """
    tree = load()
    if section not in tree:
        return default
    if key is None:
        return tree[section]
    value = tree[section].get(key)
    return default if value in (None, "", []) else value


def update(patch: dict[str, Any]) -> list[str]:
    """Merge *patch* into the settings file. Returns the changed dotted keys."""
    changed, _ = _update(patch, strict=False)
    return changed


def unknown_keys(patch: dict[str, Any]) -> list[str]:
    """Dotted names in ``patch`` that this schema does not know.

    NAMES only. A name this schema does know, carrying a value it cannot use, is
    not reported here - that is :func:`update_strict`'s errors, which is where a
    flattened section (a known section given a scalar rather than a mapping of
    its keys) is reported too. The two doors together account for every part of
    a patch that cannot be applied, and the split is by name-versus-value so
    that neither has to guess what the other reported.
    """
    out: list[str] = []
    for section, values in (patch or {}).items():
        if section not in _SCHEMA:
            out.append(f"{section}.*" if isinstance(values, dict) else str(section))
            continue
        if not isinstance(values, dict):
            # Known name, unusable value: update_strict's error, not this list.
            continue
        for key in values:
            if key not in _SCHEMA[section]:
                out.append(f"{section}.{key}")
    return sorted(out)


def update_strict(patch: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Like :func:`update`, but invalid VALUES are reported, never stored. Returns ``(changed,
    errors)`` where each error names the dotted key and the reason.

    A section given something other than a mapping of its keys is such a value,
    and is reported under the section's own name; :func:`unknown_keys` does not
    report it, because the name is one this schema knows.
    """
    return _update(patch, strict=True)


def _update(patch: dict[str, Any], strict: bool) -> tuple[list[str], list[str]]:
    changed: list[str] = []
    errors: list[str] = []
    with _lock:
        current = load()
        stored = _read_file()
        for section, values in (patch or {}).items():
            if section not in _SCHEMA:
                continue
            if not isinstance(values, dict):
                # A section holds keys, so a scalar - or a list, or None - in its
                # place names none of them: there is nothing to coerce, and no
                # key whose shape a lenient degrade could fall back to. Every
                # other part of a patch this store cannot apply is accounted for
                # to the caller: an unknown NAME by unknown_keys(), an unusable
                # VALUE by the errors below. A known section flattened to a
                # scalar was the one spelling accounted for by neither, so
                # ``{"security": "<token>"}`` - the section holding the bearer
                # every /api and /ws request must present - returned no unknown
                # name, no error and no changed key, and stored nothing. The
                # caller's only honest report of that is a success.
                #
                # WARNING rather than DEBUG on the lenient path for the same
                # reason: there the report the caller can make IS a success (an
                # empty changed list), so this log line is the only signal the
                # patch went nowhere.
                shape = f"{section}: expected a mapping of {section} keys, got {type(values).__name__}"
                if strict:
                    errors.append(shape)
                else:
                    logger.warning("ignoring settings patch: %s", log_redaction.one_line(shape))
                continue
            for key, raw in values.items():
                if key not in _SCHEMA[section]:
                    continue
                value, reason = _graded(section, key, raw)
                if reason is not None:
                    if strict:
                        errors.append(f"{section}.{reason}")
                        continue
                    value = _degraded(section, key, raw)
                if value == current[section].get(key):
                    continue
                stored.setdefault(section, {})[key] = value
                changed.append(f"{section}.{key}")
        if changed:
            _write_file(stored)
            # Cleared before the reload, not left to it: a reload that raises
            # must not leave the pre-write tree cached as if it were current.
            _cache.clear()
            load(refresh=True)
    return changed, errors


def _write_file(data: dict[str, Any]) -> None:
    """Replace the settings file atomically, and never let it be world-readable.

    This file holds ``security.auth_token`` -- the bearer every ``/api`` and
    ``/ws`` request must present -- so its mode is part of the deployment's
    security posture rather than tidiness. Writing the path and tightening it
    afterwards got that wrong in two ways, both measured:

    * The payload was written at the umask default and chmod-ed only *after* the
      rename, so at ``umask 022`` the token sat in a ``0o644`` file for the
      length of the write, and in a ``0o644`` sibling ``.tmp`` before that.
    * The ``chmod`` was best-effort under a silent ``except OSError``, so where it
      could not be applied the token stayed world-readable **permanently**, with
      no log line to say so.

    ``mkstemp`` opens at ``0o600`` and ``os.replace`` carries those bits onto the
    destination, so the mode is a property of how the file is created rather than
    a call that may or may not land -- which also tightens a settings file left
    at ``0o644`` by an earlier build the next time it is written. The rename is
    atomic within a directory, so a concurrent reader sees either the whole
    previous file or the whole new one, never a prefix; the old fixed-name
    ``.tmp`` sibling was also shared by every writer that reached it.

    This is the sequence :func:`strands_robots.dashboard.auth._save_locked` uses
    for the credential store, for the same reasons and deliberately not imported
    from it: ``settings`` imports no dashboard sibling, which is what lets it be
    reviewed and used on its own.
    """
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False)
    fd, tmp = tempfile.mkstemp(prefix=f"{SETTINGS_FILE.name}.", suffix=".tmp", dir=SETTINGS_FILE.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, SETTINGS_FILE)
    except BaseException:
        # Leave no debris behind a failed write: the file on disk is still the
        # previous good one, and a stray .tmp beside it holds the auth token while
        # being read by nothing. BaseException rather than Exception because
        # KeyboardInterrupt is the one signal an operator sends by hand, and it is
        # not an Exception - narrowing this clause would leak the token file on
        # exactly that.
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# ----------------------------------------------------------------------
# Mesh env application
# ----------------------------------------------------------------------

#: Settings key -> env var read by ``strands_robots.mesh.session`` / the
#: transport factory.
MESH_ENV = {
    "connect": "ZENOH_CONNECT",
    "listen": "ZENOH_LISTEN",
    "port": "STRANDS_MESH_PORT",
    "backend": "STRANDS_MESH_BACKEND",
    "camera_hz": "STRANDS_MESH_CAMERA_HZ",
    "policy_type_allow": "STRANDS_MESH_POLICY_TYPE_ALLOW",
}


def apply_mesh_env() -> dict[str, str]:
    """Push mesh settings into ``os.environ``."""
    mesh = load()["mesh"]
    applied: dict[str, str] = {}
    for key, env_name in MESH_ENV.items():
        value = mesh.get(key)
        if isinstance(value, list):
            value = ",".join(value)
        if value in (None, ""):
            continue
        os.environ[env_name] = str(value)
        applied[env_name] = str(value)
    if applied:
        logger.info("mesh env from settings: %s", applied)
    also = {
        "runtime.trust_remote_code": ("STRANDS_TRUST_REMOTE_CODE", load()["runtime"]["trust_remote_code"]),
    }
    for _, (env_name, value) in also.items():
        if value:
            os.environ[env_name] = "1"
            applied[env_name] = "1"
    return applied
