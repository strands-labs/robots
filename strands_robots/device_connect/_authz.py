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
"""

from __future__ import annotations

import fnmatch
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_RPC_ALLOW_ENV = "DEVICE_CONNECT_RPC_ALLOW"
_ESTOP_ALLOW_ENV = "DEVICE_CONNECT_ESTOP_ALLOW"

_warned_permissive: set[str] = set()


def _parse_allowlist(raw: Optional[str]) -> Optional[list[str]]:
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


def is_authorized_caller(caller: Optional[str], *, scope: str = "rpc") -> bool:
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

    # Allowlist configured: a missing caller identity cannot be authorized.
    if not caller:
        return False
    return _matches(caller, patterns)


def authz_error(caller: Optional[str], function: str) -> dict:
    """Standard structured rejection for an unauthorized RPC call."""
    logger.warning(
        "Rejected unauthorized Device Connect RPC %s from caller=%r", function, caller
    )
    return {
        "status": "error",
        "reason": f"caller not authorized for {function!r}",
        "caller": caller or "unknown",
    }
