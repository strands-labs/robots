"""Configuration surface for the dashboard - the ``/api/config`` payload."""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from strands_robots.dashboard import settings
from strands_robots.dashboard.argv_exposure import argv_token_notice

logger = logging.getLogger(__name__)

ENV_FILE = Path(os.getenv("DASHBOARD_ENV_FILE", ".env")).expanduser()

#: One writer at a time for the file that holds the operator's credentials. ``apply`` runs
#: in a worker thread per request, so two tabs saving at once would otherwise each
#: read-modify-write their own copy and the loser's keys would silently revert.
_ENV_FILE_LOCK = threading.Lock()

SECRET_RX = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|BEARER|API_?KEY)", re.I)

#: Characters our masks are made of. A submitted value containing one of these
#: is assumed to be an untouched mask and is not written back.
_MASK_MARKERS = ("•", "…")

#: Models offered as chips in the UI. Free text is always allowed too - this is
#: a convenience list, not an allowlist.
KNOWN_MODELS = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
]

VOICE_PROVIDERS = ["openai", "nova_sonic"]

#: Env vars worth surfacing even when absent from the .env file, so an operator
#: can discover what the dashboard actually reads.
INTERESTING_ENV = [
    "OPENAI_API_KEY",
    "HF_TOKEN",
    "AWS_REGION",
    "AWS_PROFILE",
    "VOICE_MODEL",
    "STRANDS_MESH_LOCAL_DEV",
    "STRANDS_MESH_MULTICAST",
    "STRANDS_ROBOTS_VIDEO_ROOT",
    "STRANDS_ROBOTS_NO_DYLD_SHIM",
    "STRANDS_DASH_TASK_REQUIRES_CONFIRM",
]


def is_secret(key: str) -> bool:
    """Whether an env key names a credential and so is masked in every view."""
    return bool(SECRET_RX.search(key))


# .env is read by every process the dashboard spawns, so an unrestricted upsert is
# configuration -> code execution (PATH=/tmp/evil hijacks python/ffmpeg for every child).
# The set is CLOSED, not a prefix: a prefix (``STRANDS_*``) admitted the very keys the
# other defenses stand on - the containment homes, the auth switch, the standing motion
# grant, the remote-code opt-in - so holding a session became holding every gate.
#: Keys the page may show and edit. Add a key here only if a client changing it
#: cannot move a containment home, weaken auth or consent, or opt code execution in.
ALLOWED_ENV_KEYS: frozenset[str] = frozenset(
    (
        *INTERESTING_ENV,
        "STRANDS_MODEL_ID",
        "AWS_DEFAULT_REGION",
        "OPENAI_BASE_URL",
        "VOICE_PROVIDER",
        "VOICE_NAME",
        "DASHBOARD_VOICE_PROMPT",
        "STRANDS_DASH_RECORD_CRUMB",
    )
)
#: Never dashboard-managed, whatever the allowlist says later: each of these is a gate
#: some other route reads live from ``os.environ``. Kept as a second fence so that adding
#: a key above by mistake still cannot open one of them.
GATE_BEARING_ENV_PREFIXES: tuple[str, ...] = (
    "STRANDS_DASH_AUTH_",
    "STRANDS_MESH_AUTH",
    "STRANDS_MESH_MTLS",
    "STRANDS_MESH_INSECURE",
)
GATE_BEARING_ENV_KEYS: frozenset[str] = frozenset(
    {
        "STRANDS_DASH_AGENT_PHYSICAL_MOTION",
        "STRANDS_DASH_TASK_REQUIRES_CONFIRM",
        "STRANDS_TRUST_REMOTE_CODE",
        "HF_LEROBOT_HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "STRANDS_TRAIN_OUTPUT_DIR",
        "STRANDS_ROBOTS_DATA_DIRS",
        "DASHBOARD_ENV_FILE",
        "DASHBOARD_AUTH_TOKEN",
        "BYPASS_TOOL_CONSENT",
        "PATH",
        "PYTHONPATH",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
    }
)
ENV_VALUE_MAX_LEN = 4096


def env_key_gate_bearing(key: str) -> bool:
    """Whether a key is one of the gates this process reads live - never page-writable."""
    return key in GATE_BEARING_ENV_KEYS or key.startswith(GATE_BEARING_ENV_PREFIXES)


def env_key_allowed(key: str) -> bool:
    """Whether the page may read or write this env key at all."""
    return key in ALLOWED_ENV_KEYS and not env_key_gate_bearing(key)


def env_entry_error(key: str, value: str, *, allowed_keys: frozenset[str] | None = None) -> str | None:
    """Why this key/value pair must not reach the env file, or None if fine.

    ``allowed_keys`` replaces the page allowlist for a caller that owns its own closed
    set - the consent routes, whose grant variables are gate-bearing on purpose and so
    are refused on the general ``/api/config`` surface. It is a replacement, not an
    extension: a key must be in the given set, exactly.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key or ""):
        return f"invalid env key {key!r}"
    if allowed_keys is not None:
        if key not in allowed_keys:
            return f"env key {key!r} is not one this route may write - allowed: {', '.join(sorted(allowed_keys))}"
    elif not env_key_allowed(key):
        return f"env key {key!r} is not dashboard-managed - allowed: {', '.join(sorted(ALLOWED_ENV_KEYS))}"
    if any(ord(ch) < 0x20 for ch in value) or value.splitlines() != [value]:
        # a newline in a VALUE writes a second variable on its own line,
        # defeating any key allow-list - so control chars are refused outright.
        # The read-back path parses with str.splitlines(), which also breaks on
        # U+0085, U+2028 and U+2029 (none below 0x20), so the gate is the same
        # predicate the parser uses: the value must survive splitlines() whole.
        return f"env value for {key} contains control characters"
    if len(value) > ENV_VALUE_MAX_LEN:
        return f"env value for {key} exceeds {ENV_VALUE_MAX_LEN} characters"
    return None


def mask(value: str) -> str:
    """``sk-abc...xyz`` -> ``sk-••••••yz``. Short values are fully hidden."""
    if not value:
        return ""
    if len(value) <= 6:
        return "•" * 6
    return f"{value[:3]}{'•' * 6}{value[-2:]}"


def looks_masked(value: Any) -> bool:
    """Whether *value* is the masked echo of a secret rather than a new value."""
    return isinstance(value, str) and any(m in value for m in _MASK_MARKERS)


# ----------------------------------------------------------------------
# .env read / write
# ----------------------------------------------------------------------


def read_env_file() -> dict[str, str]:
    """The ``.env`` file as key/value pairs; missing file means empty."""
    out: dict[str, str] = {}
    try:
        if not ENV_FILE.exists():
            return out
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key:
                out[key] = value
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read %s: %s", ENV_FILE, exc)
    return out


def _write_env_durably(lines: Sequence[str]) -> None:
    """Replace the env file in one step: a ``0600`` tempfile beside it, fsync, ``os.replace``.

    ``write_text`` truncates in place, so a crash mid-write left a well-formed but shorter
    file and the credentials were simply gone; and the file was world-readable between the
    write and the ``chmod``. The tempfile is created ``0600`` so no reader ever sees more.
    Caller holds ``_ENV_FILE_LOCK``.
    """
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ENV_FILE.with_name(f".{ENV_FILE.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass  # best effort: a filesystem that refuses modes (some mounts) still holds the file
        os.replace(tmp, ENV_FILE)
    finally:
        # a failed write must not leave a half file next to the real one
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass  # the tempfile is already gone or unremovable; nothing else to do


def upsert_env_file(updates: dict[str, str], *, allowed_keys: frozenset[str] | None = None) -> list[str]:
    """Order-preserving upsert into the env file. Returns the keys written.

    ``allowed_keys`` is the caller's own closed set (see :func:`env_entry_error`); without
    it the page allowlist applies.
    """
    if not updates:
        return []
    for key, value in updates.items():
        problem = env_entry_error(str(key), str(value), allowed_keys=allowed_keys)
        if problem:
            raise ValueError(problem)
    with _ENV_FILE_LOCK:
        lines: list[str] = []
        if ENV_FILE.exists():
            lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
        remaining = dict(updates)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key = stripped.partition("=")[0].strip()
            if key in remaining:
                lines[i] = f"{key}={remaining.pop(key)}"
        for key, value in remaining.items():
            lines.append(f"{key}={value}")
        _write_env_durably(lines)
    return list(updates)


def split_env_patch(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Split an env patch into writes and deletions, refusing keys the page may not touch."""
    updates: dict[str, Any] = {}
    deletions: list[str] = []
    for key, value in raw.items():
        name = str(key).strip()
        if value is None:
            deletions.append(name)
        else:
            updates[name] = value
    return updates, deletions


def delete_env_keys(keys: Sequence[str]) -> list[str]:
    """Remove keys from the env file, order-preserving. Returns those removed."""
    wanted = [str(k).strip() for k in keys]
    allowed = [k for k in wanted if not env_entry_error(k, "")]
    if not allowed:
        return []
    with _ENV_FILE_LOCK:
        if not ENV_FILE.exists():
            return []
        kept: list[str] = []
        removed: list[str] = []
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.partition("=")[0].strip()
                if key in allowed:
                    removed.append(key)
                    continue
            kept.append(line)
        if removed:
            _write_env_durably(kept)
    return removed


def bootstrap_env(from_file: Mapping[str, str], environ: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """(values to export, keys the process environment already decides)."""
    to_set: dict[str, str] = {}
    shadowed: list[str] = []
    for key, value in from_file.items():
        live = environ.get(key)
        if live is None:
            to_set[key] = value
        elif live != value:
            shadowed.append(key)
    return to_set, sorted(shadowed)


def load_env_file() -> tuple[list[str], list[str]]:
    """Apply .env to this process (once, at startup). Returns (exported, shadowed)."""
    to_set, shadowed = bootstrap_env(read_env_file(), os.environ)
    for key, value in to_set.items():
        os.environ[key] = value
    return sorted(to_set), shadowed


def env_view() -> list[dict[str, Any]]:
    """Masked env listing for the UI: .env contents + interesting live vars."""
    from_file = read_env_file()
    keys = list(from_file)
    for key in INTERESTING_ENV:
        if key not in keys:
            keys.append(key)
    rows: list[dict[str, Any]] = []
    for key in keys:
        live = os.environ.get(key, "")
        in_file = key in from_file
        raw = live if live else from_file.get(key, "")
        shadowed = bool(in_file and live and live != from_file.get(key))
        secret = is_secret(key)
        rows.append(
            {
                "key": key,
                "value": mask(raw) if secret and raw else raw,
                "secret": secret,
                "set": bool(raw),
                "in_file": in_file,
                "shadowed": shadowed,
                "editable": env_key_allowed(key),
            }
        )
    return rows


# ----------------------------------------------------------------------
# Combined document
# ----------------------------------------------------------------------

# : Keys ``mesh.security.validate_command()`` admits for execute/start, minus : the ones the
# dashboard sets itself (action/instruction/policy_provider/ : duration).
WIRE_CMD_KEYS: tuple[str, ...] = (
    "policy_host",
    "policy_port",
    "policy_type",
    "server_address",
    "model_path",
    "pretrained_name_or_path",
    "robot_name",
    "target_pose",
    "target_joints",
    "world_update",
    "control_frequency",
    "action_horizon",
    "fast_mode",
    "n_steps",
)

#: How each wire key should be rendered / parsed by the run form.
WIRE_KEY_TYPES: dict[str, str] = {
    "policy_host": "string",
    "policy_port": "int",
    "policy_type": "string",
    "server_address": "string",
    "model_path": "string",
    "pretrained_name_or_path": "string",
    "robot_name": "string",
    "target_pose": "json",
    "target_joints": "json",
    "world_update": "json",
    "control_frequency": "float",
    "action_horizon": "int",
    "fast_mode": "bool",
    "n_steps": "int",
}

#: Registry key -> the wire key that actually carries it. The registry names a
#: provider's constructor kwargs; the wire schema names its own fields, and the
#: two only partly overlap.
_WIRE_ALIASES = {
    "port": "policy_port",
    "host": "policy_host",
    "checkpoint": "model_path",
    "policy_path": "model_path",
    "repo_id": "pretrained_name_or_path",
    "server_address": "server_address",
}


def _policy_catalog() -> list[dict[str, Any]]:
    """Full provider objects from ``registry/policies.json``."""
    try:
        from strands_robots.registry.policies import get_policy_provider, list_policy_providers
    except Exception as exc:  # noqa: BLE001
        logger.warning("policy registry unavailable: %s", exc)
        return []

    try:
        from strands_robots.mesh.security import is_safe_policy_provider
    except Exception:  # noqa: BLE001

        def is_safe_policy_provider(_name: str) -> bool:  # type: ignore[misc]
            return True

    out: list[dict[str, Any]] = []
    for name in list_policy_providers():
        spec = get_policy_provider(name) or {}
        requires = list(spec.get("requires") or [])
        config_keys = list(spec.get("config_keys") or [])
        # Split the provider's inputs by what the wire will actually carry, so
        # the form can render the deliverable fields and *say* that the rest
        # only work when the policy is built locally.
        wire_fields: list[dict[str, Any]] = []
        unsettable: list[str] = []
        for key in dict.fromkeys(requires + config_keys):
            wire_key = _WIRE_ALIASES.get(key, key)
            if wire_key in WIRE_CMD_KEYS:
                wire_fields.append(
                    {
                        "key": key,
                        "wire_key": wire_key,
                        "type": WIRE_KEY_TYPES.get(wire_key, "string"),
                        "required": key in requires,
                        "default": (spec.get("defaults") or {}).get(key),
                    }
                )
            else:
                unsettable.append(key)
        out.append(
            {
                "name": name,
                "description": spec.get("description", ""),
                "requires": requires,
                "config_keys": config_keys,
                "defaults": dict(spec.get("defaults") or {}),
                "shorthands": list(spec.get("shorthands") or []),
                "url_patterns": list(spec.get("url_patterns") or []),
                "extra": spec.get("extra"),
                "trainable": bool(spec.get("trainer")),
                "wire_fields": wire_fields,
                "unsettable_over_mesh": unsettable,
                # False -> the mesh security gate rejects it; the card shows a lock
                # and points at STRANDS_MESH_POLICY_TYPE_ALLOW rather than letting
                # the operator discover it as a wire rejection.
                "wire_safe": bool(is_safe_policy_provider(name)),
                # Hardware peers cannot build checkpoint policies over the wire
                # (they only accept {port, host, data_config}).
                "server_based": bool(
                    {"port", "policy_port", "server_address", "host"} & set(spec.get("requires") or [])
                ),
            }
        )
    return out


def snapshot(*, bridge: Any = None, agent_status: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ``GET /api/config`` document."""
    tree = settings.load(refresh=True)
    agent = dict(tree["agent"])
    from strands_robots.dashboard.agent_console import SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT

    prompt = agent.get("system_prompt")
    return {
        "agent": {
            "model_id": agent.get("model_id"),
            "known_models": KNOWN_MODELS,
            "system_prompt": prompt or DEFAULT_SYSTEM_PROMPT,
            "is_default_prompt": not prompt,
            "temperature": agent.get("temperature"),
            "max_tokens": agent.get("max_tokens"),
            **(agent_status or {}),
        },
        "voice": {
            "provider": tree["voice"].get("provider") or "openai",
            "voice_name": tree["voice"].get("voice_name"),
            "providers": VOICE_PROVIDERS,
        },
        "mesh": bridge.mesh_info() if bridge is not None else {},
        "runtime": dict(tree["runtime"]),
        "security": {
            # Never echo the token back - only whether one is configured.
            "auth_enabled": bool(tree["security"].get("auth_token")),
            "cors_origins": tree["security"].get("cors_origins") or ["*"],
            # This process's own posture: a token passed as --auth-token is readable by every local user
            # via `ps`.
            **({"notice": notice} if (notice := argv_token_notice(sys.argv)) else {}),
        },
        "policies": _policy_catalog(),
        "env": env_view(),
        "env_file": str(ENV_FILE),
        "settings_file": str(settings.SETTINGS_FILE),
    }


#: Settings keys that only take effect on a new mesh session. Everything else
#: is hot-applied, and the response says which is which per field rather than
#: making the operator guess.
_RESTART_KEYS = {"mesh.connect", "mesh.listen", "mesh.port", "mesh.backend"}

_RESPAWN_KEYS = {"mesh.camera_hz"}

_STARTUP_KEYS = {"security.cors_origins"}

#: Body fields of ``POST /api/config`` that are NOT settings sections: the
#: caller's own vocabulary, so they must never be reported as unknown settings.
_BODY_NON_SECTION_KEYS = frozenset({"env", "reset_prompt", "reset_agent", "clear_history", "restart_mesh", "force"})

#: Changing these rebuilds the agent on the next turn.
_AGENT_KEYS = {"agent.model_id", "agent.system_prompt", "agent.temperature", "agent.max_tokens"}


def apply(body: dict[str, Any]) -> dict[str, Any]:
    """Apply a ``POST /api/config`` body.

    Returns ``{applied, restart_required, env_written, skipped_masked,
    agent_reset, errors}``.
    """
    body = body or {}
    errors: list[str] = []
    patch: dict[str, dict[str, Any]] = {}

    for section in ("agent", "voice", "mesh", "runtime", "security"):
        values = body.get(section)
        if isinstance(values, dict):
            patch[section] = dict(values)

    # "Reset to default prompt" is an explicit action, not an empty string -
    # an empty prompt field should not silently wipe a customised prompt.
    if body.get("reset_prompt"):
        patch.setdefault("agent", {})["system_prompt"] = None

    # Never let the UI persist the resolved default prompt as an override:
    # is_default_prompt would then be wrong forever.
    if isinstance(patch.get("agent"), dict):
        from strands_robots.dashboard.agent_console import SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT

        prompt = patch["agent"].get("system_prompt")
        if isinstance(prompt, str) and prompt.strip() == DEFAULT_SYSTEM_PROMPT.strip():
            patch["agent"]["system_prompt"] = None

    # Endpoint schemes: mtls refuses non-TLS endpoints loudly at session open
    # (``strands_robots.mesh.session._validate_endpoint_schemes``). Catch it at the form.
    mesh_patch = patch.get("mesh") or {}
    if mesh_patch.get("connect") or mesh_patch.get("listen"):
        local_dev = os.getenv("STRANDS_MESH_LOCAL_DEV", "") not in ("", "0", "false")
        if not local_dev:
            bad = [
                ep
                for key in ("connect", "listen")
                for ep in settings.as_list(mesh_patch.get(key))
                if ep.split("/", 1)[0] in ("tcp", "udp", "ws")
            ]
            if bad:
                errors.append(
                    f"non-TLS endpoints {bad} are rejected when mesh auth is mtls - "
                    "use tls/... or quic/..., or run with --local-dev"
                )
                for key in ("connect", "listen"):
                    mesh_patch.pop(key, None)

    # Names the schema does not know are dropped without an error, so without this the drawer says
    # "nothing changed" for a patch that changed nothing BECAUSE IT WAS NOT UNDERSTOOD.
    ignored = settings.unknown_keys(
        {key: value for key, value in body.items() if isinstance(value, dict) and key not in _BODY_NON_SECTION_KEYS}
    )

    if patch:
        changed, coercion_errors = settings.update_strict(patch)
        errors.extend(coercion_errors)
    else:
        changed = []

    # --- env upsert -------------------------------------------------
    env_written: list[str] = []
    skipped_masked: list[str] = []
    env_removed: list[str] = []
    raw_env = body.get("env")
    if isinstance(raw_env, dict):
        raw_updates, deletions = split_env_patch(raw_env)
        updates: dict[str, str] = {}
        for key, value in raw_updates.items():
            if looks_masked(value):
                skipped_masked.append(key)
                continue
            value_str = str(value)
            problem = env_entry_error(key, value_str)
            if problem:
                errors.append(problem)
                continue
            updates[key] = value_str
        for key in deletions:
            problem = env_entry_error(key, "")
            if problem:
                errors.append(problem)
        env_written = upsert_env_file(updates)
        for key, value in updates.items():
            os.environ[key] = value
        env_removed = delete_env_keys(deletions)
        for key in env_removed:
            # The live process inherited it at startup; leaving it exported would make the
            # dashboard itself the one place where the deleted variable is still in effect.
            os.environ.pop(key, None)

    # --- hot apply --------------------------------------------------
    agent_reset = False
    if any(k in _AGENT_KEYS for k in changed) or body.get("reset_agent"):
        # Consoles on main are per-socket (routes_agent builds one per /ws/agent), so
        # a settings change reaches the next conversation by itself; the flag tells
        # the page to reconnect its dock.
        agent_reset = True

    restart_required = sorted(k for k in changed if k in _RESTART_KEYS)
    respawn_required = sorted(k for k in changed if k in _RESPAWN_KEYS)
    startup_required = sorted(k for k in changed if k in _STARTUP_KEYS)
    return {
        "ignored": ignored,
        # Stored, and inherited by the NEXT child - but not applied to anything running, so it
        # does not belong in `applied` either.
        "applied": sorted(
            k for k in changed if k not in _RESTART_KEYS and k not in _RESPAWN_KEYS and k not in _STARTUP_KEYS
        ),
        "restart_required": restart_required,
        "respawn_required": respawn_required,
        "startup_required": startup_required,
        "env_written": env_written,
        "env_removed": env_removed,
        "skipped_masked": skipped_masked,
        "agent_reset": agent_reset,
        "errors": errors,
    }
