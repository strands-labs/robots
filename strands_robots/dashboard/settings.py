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

_lock = threading.RLock()
# The resolved tree, cached under the path it was resolved from. A `dict | None`
# rebound through `global` -- and at two of the four write sites through
# `globals()["_cache"]`, which is the same rebinding spelled so it does not need
# the declaration -- makes "the cache describes the current SETTINGS_FILE" an
# invariant maintained by hand at every write, and the two can disagree: a
# process that repoints SETTINGS_FILE is served the tree resolved from the
# previous file, and a stale hit is indistinguishable from a fresh one. Keyed
# this way they cannot, because the path is the dict's key, so a tree is only
# reachable through the file it came from. `auth.py` keys its store cache on a
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
        return [str(p).strip() for p in value if str(p).strip()]
    return []


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


#: Public alias - other dashboard modules parse comma-separated endpoint
#: strings with the same rules the settings store uses.
def as_list(value: Any) -> list[str]:
    return _as_list(value)


class CoercionError(ValueError):
    """A settings value that must be REPORTED, not silently defaulted. Raised only on the strict path
    (UI/API writes).
    """


def _finite_float(key: str, value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise CoercionError(f"{key}: {value!r} is not a number")
    if not math.isfinite(out):
        # json.dumps would emit bare NaN/Infinity - not JSON (RFC 8259); one
        # such write bricks the config screen for every browser forever.
        raise CoercionError(f"{key}: {value!r} is not a finite number")
    return out


def _coerce(section: str, key: str, value: Any, strict: bool = False) -> Any:
    try:
        return _coerce_strict(section, key, value)
    except CoercionError:
        if strict:
            raise
        # Lenient degrade (env/CLI/file paths) must still degrade to the key's own SHAPE: a list key
        # that fell back to a scalar poisons every comma-split consumer, which is worse than the empty
        # default.
        if (section, key) in _LIST_KEYS:
            return []
        return None if key in ("temperature", "camera_hz", "max_tokens", "port") else value


# : What "true" and "false" may be spelled like.
_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off", "")


def _coerce_strict(section: str, key: str, value: Any) -> Any:
    if (section, key) in _LIST_KEYS:
        if value is not None and not isinstance(value, (str, list, tuple)):
            raise CoercionError(f"{key}: expected a list or comma-separated string, got {type(value).__name__}")
        return _as_list(value)
    if key in ("trust_remote_code",):
        if not isinstance(value, bool):
            spelled = str(value).strip().lower()
            if spelled not in _TRUTHY and spelled not in _FALSY:
                raise CoercionError(f"{key}: {value!r} is not a boolean (use true/false)")
        return _as_bool(value)
    if key == "temperature":
        if value in (None, ""):
            return None
        out = _finite_float(key, value)
        if not 0.0 <= out <= 2.0:
            raise CoercionError(f"temperature: {out} is outside 0..2")
        return out
    if key == "camera_hz":
        if value in (None, ""):
            return None
        out = _finite_float(key, value)
        if not 0.0 < out <= 240.0:
            # a publisher sleeps 1/hz between frames: 0 divides by zero,
            # negative sleeps never, huge busy-loops the camera thread.
            raise CoercionError(f"camera_hz: {out} is outside (0, 240]")
        return out
    if key in ("max_tokens", "port"):
        if value in (None, ""):
            return None
        try:
            out = int(value)
        except (TypeError, ValueError):
            raise CoercionError(f"{key}: {value!r} is not an integer")
        if key == "port" and not 1 <= out <= 65535:
            raise CoercionError(f"port: {out} is outside 1..65535")
        if key == "max_tokens" and out < 1:
            raise CoercionError(f"max_tokens: {out} must be at least 1")
        return out
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, set)):
        raise CoercionError(f"{key}: expected a string, got {type(value).__name__}")
    return str(value)


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
                SETTINGS_FILE.read_text(),
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
    """Dotted names in ``patch`` that this schema does not know."""
    out: list[str] = []
    for section, values in (patch or {}).items():
        if section not in _SCHEMA:
            out.append(f"{section}.*" if isinstance(values, dict) else str(section))
            continue
        if not isinstance(values, dict):
            continue
        for key in values:
            if key not in _SCHEMA[section]:
                out.append(f"{section}.{key}")
    return sorted(out)


def update_strict(patch: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Like :func:`update`, but invalid VALUES are reported, never stored. Returns ``(changed,
    errors)`` where each error names the dotted key and the reason.
    """
    return _update(patch, strict=True)


def _update(patch: dict[str, Any], strict: bool) -> tuple[list[str], list[str]]:
    changed: list[str] = []
    errors: list[str] = []
    with _lock:
        current = load()
        stored = _read_file()
        for section, values in (patch or {}).items():
            if section not in _SCHEMA or not isinstance(values, dict):
                continue
            for key, raw in values.items():
                if key not in _SCHEMA[section]:
                    continue
                try:
                    value = _coerce(section, key, raw, strict=strict)
                except CoercionError as exc:
                    errors.append(f"{section}.{exc}")
                    continue
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
        with os.fdopen(fd, "w") as handle:
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

#: Settings key -> env var read by ``mesh/session.py`` / the transport factory.
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
