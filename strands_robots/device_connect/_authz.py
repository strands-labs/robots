"""Caller-authorization helpers for Device Connect robot/sim drivers.

Security hardening: Device Connect RPC handlers run on the device side with no
built-in per-call authorization. State-mutating RPCs (execute / stop / step /
reset) and lifecycle events (emergencyStop) must therefore verify the calling
device against an operator-controlled allowlist before acting on physical (or
simulated) hardware.

Allowlists are sourced from environment variables so deployments opt in without
code changes:

* ``DEVICE_CONNECT_RPC_ALLOW`` — comma-separated device ids permitted to call
  state-mutating RPCs. ``*`` (or unset) means "allow all" but logs a warning so
  the permissive posture is visible. An explicit empty value (``""`` after
  stripping) is treated as unset.
* ``DEVICE_CONNECT_ESTOP_ALLOW`` — comma-separated device ids permitted to
  trigger emergency-stop handling. Falls back to ``DEVICE_CONNECT_RPC_ALLOW``
  when unset.

Matching supports trailing ``*`` glob prefixes (e.g. ``safety-*``).

Caller-identity semantics (READ THIS before relying on the allowlist):

* The caller id is whatever the messaging layer reported as the RPC's
  ``source_device``. A device-to-device caller (another ``DeviceRuntime``) and
  an agent that sets ``STRANDS_ROBOT_MESH_AGENT_ID`` both carry an id; an
  anonymous client carries **none** (``caller=None``).
* When an allowlist IS set, a missing/None caller cannot be authorized and is
  denied (fail-closed). So setting ``DEVICE_CONNECT_RPC_ALLOW`` will reject
  every anonymous caller — configure an id on the caller side to allow it.
* The id is only as trustworthy as the transport. Under authenticated
  transport (mTLS) it is bound to the sender's certificate. Under insecure
  transport (``DEVICE_CONNECT_ALLOW_INSECURE``) it is **self-asserted** — any
  peer can claim any id — so the allowlist is advisory there, not a
  cryptographic boundary. A one-time warning is logged in that case.
"""

from __future__ import annotations

import fnmatch
import logging
import os

logger = logging.getLogger(__name__)

_RPC_ALLOW_ENV = "DEVICE_CONNECT_RPC_ALLOW"
_ESTOP_ALLOW_ENV = "DEVICE_CONNECT_ESTOP_ALLOW"

_warned_permissive: set[str] = set()
_warned_insecure_acl: set[str] = set()

_INSECURE_ENV = "DEVICE_CONNECT_ALLOW_INSECURE"


def _insecure_transport_active() -> bool:
    return os.environ.get(_INSECURE_ENV, "").lower() in ("true", "1", "yes")


def _warn_insecure_acl_once(scope: str) -> None:
    """Warn (once per scope) that an allowlist is being enforced against a
    self-asserted caller id because the transport is insecure."""
    if scope in _warned_insecure_acl:
        return
    _warned_insecure_acl.add(scope)
    logger.warning(
        "Device Connect %s allowlist is enforced against a SELF-ASSERTED caller "
        "identity: %s is set, so any peer can claim an allowed id. Treat the "
        "allowlist as advisory here; use authenticated transport (mTLS) for a "
        "cryptographic authorization boundary.",
        scope,
        _INSECURE_ENV,
    )


def _parse_allowlist(raw: str | None) -> list[str] | None:
    """Parse a comma-separated allowlist. Returns None when unset/empty."""
    if raw is None:
        return None
    entries = [e.strip() for e in raw.split(",") if e.strip()]
    return entries or None


def _matches(caller: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if pat == "*" or fnmatch.fnmatchcase(caller, pat):
            return True
    return False


def _warn_permissive_once(scope: str) -> None:
    if scope not in _warned_permissive:
        _warned_permissive.add(scope)
        logger.warning(
            "Device Connect %s authorization is permissive (no %s allowlist set). "
            "Any device that can reach the network may invoke state-mutating "
            "operations. Set the allowlist to restrict callers.",
            scope,
            _RPC_ALLOW_ENV if scope == "rpc" else _ESTOP_ALLOW_ENV,
        )


def is_authorized_caller(caller: str | None, *, scope: str = "rpc") -> bool:
    """Return True iff *caller* is authorized for the given *scope*.

    scope="rpc"   -> state-mutating RPCs (execute/stop/step/reset)
    scope="estop" -> emergency-stop event handling
    """
    if scope == "estop":
        raw = os.environ.get(_ESTOP_ALLOW_ENV) or os.environ.get(_RPC_ALLOW_ENV)
        env_scope = "estop"
    else:
        raw = os.environ.get(_RPC_ALLOW_ENV)
        env_scope = "rpc"

    patterns = _parse_allowlist(raw)
    if patterns is None:
        # No allowlist configured — preserve out-of-the-box dev usability but
        # make the permissive posture loud so operators notice.
        _warn_permissive_once(env_scope)
        return True

    # An allowlist is configured. If the transport is insecure the caller id is
    # self-asserted, so the allowlist is advisory — say so once, loudly.
    if _insecure_transport_active():
        _warn_insecure_acl_once(env_scope)

    # Allowlist configured: a missing caller identity cannot be authorized.
    if not caller:
        return False
    return _matches(caller, patterns)


def authz_error(caller: str | None, function: str) -> dict:
    """Standard structured rejection for an unauthorized RPC call."""
    logger.warning("Rejected unauthorized Device Connect RPC %s from caller=%r", function, caller)
    return {
        "status": "error",
        "reason": f"caller not authorized for {function!r}",
        "caller": caller or "unknown",
    }
