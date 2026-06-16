"""Payload validation for the strands-robots mesh.

This module owns the *payload-semantic* security boundary -- the rules
that depend on what is inside a mesh command, not who sent it. Wire-
level authentication (peer identity, replay protection, rate limiting,
fleet membership) is delegated entirely to Zenoh: mTLS at
``transport/link/tls``, role policy at ``access_control``, frequency
caps at ``downsampling``, byte caps at ``low_pass_filter``. See
:mod:`strands_robots.mesh._zenoh_config` and
:mod:`strands_robots.mesh._acl_config` for the transport-side
configuration.

What this module covers:

* :func:`validate_command` -- action allowlist plus per-action bounds
  (instruction length, duration, step count,...).
* :func:`is_safe_policy_host` -- VLA inference target host / CIDR
  allowlist.
* :func:`is_safe_model_path` -- HuggingFace repo / local model path
  validation with optional ``<org>/<repo>`` prefix gating.
* :func:`is_safe_policy_type` / :func:`is_safe_policy_provider` --
  policy registry allowlist.
* :func:`is_safe_server_address` -- composite host[:port] check.

Everything here defends against an *authenticated* peer that has
already cleared mTLS + ACL but whose payload contents we still need
to bound. Without these checks an authorised operator could steer a
robot at an attacker-controlled inference server, request a 24-hour
``execute`` action, or drive the robot to download an arbitrary
HuggingFace model.

Configuration env vars
----------------------
``STRANDS_MESH_POLICY_HOST_ALLOW``
    Comma-separated host / CIDR list extending the loopback-only
    default ``policy_host`` allowlist.
``STRANDS_MESH_HF_REPO_ALLOW``
    Comma-separated HF org prefixes (or full ``<org>/<repo>`` prefixes)
    accepted in ``pretrained_name_or_path``. Defaults to
    ``nvidia,huggingface,lerobot``.
``STRANDS_MESH_POLICY_TYPE_ALLOW``
    Comma-separated extra policy_type / policy_provider values.
    Note: this single env var extends both
    :func:`is_safe_policy_type` and :func:`is_safe_policy_provider`
    (they share one allowlist by design, see #239 bucket C).
"""

from __future__ import annotations

import functools
import ipaddress
import json
import logging
import math
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# --- Constants -----------------------------------------------------------

#: Maximum duration (seconds) accepted for ``execute`` / ``start`` commands.
MAX_DURATION_S: float = 3600.0

#: Maximum length (characters) of a natural-language ``instruction`` payload.
MAX_INSTRUCTION_LEN: int = 2000

#: Maximum length of a HuggingFace repo id / local model path. Real-world
#: HF ids are well under 200 chars; 512 leaves headroom for nested local
#: paths without becoming a DoS vector.
MAX_MODEL_PATH_LEN: int = 512

#: Maximum length of a remote policy ``server_address`` (host[:port] or URL).
#: Bounded independently of :data:`MAX_MODEL_PATH_LEN`: a fully-qualified
#: hostname is RFC-1035-bounded at 253 chars, IPv6-bracketed-host + port +
#: scheme caps under 80, so 256 is roomy for any legitimate input while
#: keeping the address-cap semantics owned by the address validator.
MAX_SERVER_ADDRESS_LEN: int = 256

#: Allowed characters for HF repo ids and local model paths. We reject
#: ``..`` traversal, shell metacharacters, NUL bytes, whitespace, and any
#: byte outside the printable ASCII subset below.
_MODEL_PATH_RE = re.compile(r"^[A-Za-z0-9_./\-]+$")

#: Maximum length of a mesh ``peer_id`` as carried on wire-side ``cmd``
#: payloads (e.g. ``teleop_receive.source_peer_id``). The local
#: :class:`~strands_robots.mesh.core.Mesh` only generates short
#: ``"<host>-<pid>-<short>"`` ids well under 64 bytes; 128 leaves
#: headroom for operator-supplied namespaces without becoming a
#: log-flood / dispatch-arg DoS vector.
MAX_PEER_ID_LEN: int = 128

#: Allowed characters for a peer_id appearing on the ``cmd`` wire surface.
#: Mirrors :data:`_MODEL_PATH_RE` discipline (printable ASCII only, no
#: shell metacharacters, no whitespace, no NULs, no unicode controls)
#: but additionally forbids ``/`` because peer_ids are not paths and any
#: ``/`` in one is a wire-side red flag.
_PEER_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")

#: Charset gate for wire-routing passthrough fields (``turn_id``,
#: ``sender_id``, ``override_code``) and host/address strings stored in
#: ``out``. Rejects NUL, CRLF, and all C0/C1 control characters while
#: allowing the full printable ASCII range (0x20-0x7E). Applied *after*
#: type-check + length-bound so a malicious payload with embedded control
#: bytes is rejected cleanly rather than passing through to audit logs or
#: downstream string ops.
_SAFE_PASSTHROUGH_RE = re.compile(r"^[\x20-\x7E]+$")

#: Maximum length of wire-routing passthrough fields (``turn_id``,
#: ``sender_id``). These are ULID/UUID-shaped correlation tokens; 128
#: chars is generous for any legitimate usage.
MAX_PASSTHROUGH_LEN: int = 128

#: Maximum length of an operator-supplied second-factor override code
#: used to clear an estop lockout (``resume.override_code``). Bounded
#: so a malformed payload cannot DoS the comparison via a multi-MB
#: string.
MAX_OVERRIDE_CODE_LEN: int = 256

#: Maximum number of joint/motor keys accepted in a single teleop input
#: frame. A leader arm streams a fixed small set of motor positions
#: (typically 6-16); 64 leaves generous headroom while bounding a peer
#: that floods ``_on_input`` with a giant dict to exhaust CPU/memory in
#: the apply path.
MAX_INPUT_FRAME_KEYS: int = 64


def _env_pos_float(env_var: str, default: float) -> float:
    """Parse a positive float from *env_var*, falling back to *default*.

    Non-numeric / non-positive / NaN / inf values fall back to the default.
    Local to security.py (the analogous helper in core.py is not importable
    here without a cycle). Used for the teleop input safety bound (H-2).
    """
    raw = os.getenv(env_var)
    if raw is None:
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(val) or val <= 0:
        return default
    return val


# Operators with degree-valued
#: actuators or multi-turn joints can widen via
#: ``STRANDS_MESH_INPUT_VALUE_ABS``. The module-level constant is captured
#: at import for backward compat; the hot path calls
#: :func:`_input_value_abs` so an operator-set env var takes effect
#: without a process restart.
DEFAULT_INPUT_VALUE_ABS: float = 12.566370614359172  # 4 * pi
MAX_INPUT_VALUE_ABS: float = _env_pos_float("STRANDS_MESH_INPUT_VALUE_ABS", DEFAULT_INPUT_VALUE_ABS)


def _input_value_abs() -> float:
    """Lazy resolver for ``STRANDS_MESH_INPUT_VALUE_ABS`` (see
    :data:`MAX_INPUT_VALUE_ABS`). Re-reads env on every call so operators
    can tune the teleop safety envelope without a restart."""
    return _env_pos_float("STRANDS_MESH_INPUT_VALUE_ABS", DEFAULT_INPUT_VALUE_ABS)


#: Charset for teleop input-frame keys (motor/joint names like
#: ``"motor.pos"``, ``"shoulder_pan"``, ``"j0"``). Printable, no
#: whitespace, no shell metacharacters, no path separators.
_INPUT_KEY_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")

#: Maximum length of a single input-frame key name.
MAX_INPUT_KEY_LEN: int = 64

#: Bound on the number of joints accepted in a sim ``execute`` /
#: ``start`` payload's ``target_joints`` dict (issue #300 well-known
#: kwarg). 256 is well above any real humanoid (Asimov-V0 has ~30) and
#: keeps a malicious payload from forcing an unbounded dict walk.
MAX_TARGET_JOINTS: int = 256

#: Bound on a sim ``execute`` / ``start`` payload's ``world_update``
#: nested dict size, in JSON-encoded bytes. Mesh does not interpret the
#: per-call collision-world refresh payload; it forwards it to the
#: planner provider via ``policy_config``. The cap is purely DoS
#: defence - 64 KiB fits any realistic obstacle list.
MAX_WORLD_UPDATE_BYTES: int = 65536

#: Charset for entries in ``STRANDS_MESH_HF_REPO_ALLOW``. Operator-supplied
#: HuggingFace org / ``<org>/<repo>`` prefixes; rejects shell metacharacters,
#: whitespace, NUL bytes, and any byte outside the printable ASCII subset.
#: Symmetric with :data:`_POLICY_HOST_ENTRY_RE` so a reviewer reading this
#: module sees a uniform fail-loud-on-misconfig posture across every
#: operator-extensible env-var allowlist.
_HF_REPO_ENTRY_RE = re.compile(r"^[A-Za-z0-9_./\-]+$")

#: Charset for entries in ``STRANDS_MESH_POLICY_TYPE_ALLOW``. Lowercase
#: identifier shape -- matches the spirit of the built-in
#: :data:`_DEFAULT_POLICY_TYPES` set.
_POLICY_TYPE_ENTRY_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: Built-in policy_type allowlist. Mirrors the LeRobot policy registry
#: families plus the providers registered in registry/policies.json. Keep this
#: in sync with the registry so a provider that ``create_policy`` can build can
#: also be driven over the mesh ``tell()`` path and Device Connect. Operators
#: extend via ``STRANDS_MESH_POLICY_TYPE_ALLOW`` (comma-separated).
_DEFAULT_POLICY_TYPES: frozenset[str] = frozenset(
    {
        "mock",
        "groot",
        "lerobot",
        "lerobot_local",
        "act",
        "diffusion",
        "tdmpc",
        "vqbet",
        "pi0",
        "pi0fast",
        "smolvla",
        "sac",
        # GR00T Whole-Body-Control (SONIC) locomotion provider (registry: wbc,
        # shorthand sonic). Without these, tell(..., policy_provider="wbc") and
        # the Device Connect drivers reject WBC at the security gate.
        "wbc",
        "sonic",
    }
)

#: Action vocabulary accepted by :func:`validate_command`. Mirrors the
#: dispatch table in :meth:`Mesh._dispatch`. Keep these two sets in sync
#: when adding a new action.
ALLOWED_ACTIONS: frozenset[str] = frozenset(
    {
        "status",
        "stop",
        "features",
        "state",
        "execute",
        "start",
        "step",
        "reset",
        "teleop_status",
        "teleop_receive",
        "teleop_stop",
        # ``resume`` clears the emergency-stop lockout; the only action
        # other than ``status`` permitted while the lockout is engaged.
        "resume",
    }
)

#: Device Connect native-RPC function names (e.g. the Reachy's ``nod`` /
#: ``look`` / ``playMove``). These are device-defined, NOT members of
#: :data:`ALLOWED_ACTIONS` -- the policy-robot action allowlist does not
#: apply to a device's own advertised function surface. We still bound the
#: name to a conservative identifier charset so a function name cannot carry
#: control bytes / shell metacharacters into the device runtime or audit log.
_DC_RPC_FUNC_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Max length of a Device Connect RPC function name.
MAX_DC_RPC_FUNC_LEN: int = 64

#: Max JSON-encoded byte size of a Device Connect RPC params object. Keeps a
#: native-function call from becoming a DoS vector, mirroring
#: :data:`MAX_WORLD_UPDATE_BYTES`.
MAX_DC_RPC_PARAMS_BYTES: int = 64 * 1024

#: Default allowlist for VLA policy server targets (loopback only).
_DEFAULT_POLICY_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})


# --- Exception hierarchy -------------------------------------------------


class SecurityError(Exception):
    """Base class for payload-validation rejections."""


class ValidationError(SecurityError):
    """Command payload failed schema or bounds checks."""


class LockoutError(SecurityError):
    """Command rejected because emergency-stop lockout is engaged.

    Raised by :class:`Mesh._dispatch` for every action other than ``status``
    and ``resume`` while ``self._estop_lockout`` is set. Caught by
    :meth:`Mesh._exec_cmd` to emit a structured ``command_rejected_lockout``
    audit event and a ``type="error"`` wire response.
    """


# --- Policy-host allowlist -----------------------------------------------


#: Charset for valid hostname / CIDR / IP-literal entries in
#: ``STRANDS_MESH_POLICY_HOST_ALLOW``. Rejects shell metacharacters,
#: whitespace, NUL bytes, and any byte outside the printable ASCII
#: subset typical of DNS labels and CIDR ranges.
_POLICY_HOST_ENTRY_RE = re.compile(r"^[A-Za-z0-9.:/_\-]+$")


def _validate_env_allowlist_entries(env_var: str, raw: str, regex: re.Pattern[str]) -> list[str]:
    """Split *raw* on comma, strip, lowercase-strip, and reject malformed entries.

    Single helper used by every operator-extensible allowlist parser
    (``_policy_host_allowlist``, ``_hf_repo_allowlist``,
    ``_policy_type_allowlist``). Comma-splits *raw*, strips, drops
    empties, and rejects entries that fail *regex* with a WARNING that
    names the env var, the offending entry, and the expected charset.

    The malformed entries are not exploitable downstream (none of the
    values are subprocess-interpolated; the downstream consumers do
    literal-equality / CIDR / set-membership comparisons), but a typo
    like ``STRANDS_MESH_HF_REPO_ALLOW="nvidia,;rm -rf /"`` would
    otherwise silently produce an allowlist containing ``';rm -rf '``
    -- the operator-visible signal that something is wrong is the
    WARNING; without it, the typo is invisible until the next time
    someone reads the env var.

    Caching at the call sites (``functools.lru_cache`` keyed on the
    raw env-var string) means the WARNING fires once per distinct
    misconfig value, not once per ``validate_command`` call.
    """
    entries: list[str] = []
    for raw_entry in raw.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if not regex.fullmatch(entry):
            logger.warning(
                "[security] %s: dropping malformed entry %r "
                "(charset must match %s); fix the env var to include this entry",
                env_var,
                entry,
                regex.pattern,
            )
            continue
        entries.append(entry)
    return entries


@functools.lru_cache(maxsize=1)
def _policy_host_allowlist_cached(raw: str) -> tuple[str, ...]:
    """Cached parse of ``STRANDS_MESH_POLICY_HOST_ALLOW``.

    Cached parse keyed on the raw env-var string. Without the cache
    every ``is_safe_policy_host`` call re-parses the env var and
    re-emits the malformed-entry WARNING; on a busy mesh
    ``validate_command`` runs per inbound cmd, and the WARNING flood
    can drown the audit log on a typo'd env var. A
    ``monkeypatch.setenv`` change naturally re-parses on the next
    call (different cache key); tests that mutate the env in-place
    can call :func:`_clear_security_caches_for_tests` to force a
    refresh.
    """
    extras = _validate_env_allowlist_entries("STRANDS_MESH_POLICY_HOST_ALLOW", raw, _POLICY_HOST_ENTRY_RE)
    return tuple(_DEFAULT_POLICY_HOSTS) + tuple(extras)


def _policy_host_allowlist() -> list[str]:
    """Return the configured policy-host allowlist (defaults + env extras)."""
    return list(_policy_host_allowlist_cached(os.getenv("STRANDS_MESH_POLICY_HOST_ALLOW", "")))


def is_safe_policy_host(host: str) -> bool:
    """Return True when *host* is permitted as a VLA policy server target.

    The default allowlist is loopback only (``localhost``, ``127.0.0.1``,
    ``::1``). Operators extend it via ``STRANDS_MESH_POLICY_HOST_ALLOW``,
    a comma-separated list of hostnames or CIDR ranges
    (``"vla.internal,10.0.0.0/24"``).

    Hostnames are matched literally (case-insensitive); IP literals are
    additionally matched against any CIDR entries in the operator list.

    .. warning::
       Hostname entries are matched LITERALLY against the caller's input string;
       no DNS resolution is performed at allowlist time. Adding ``vla.internal``
       to the allowlist therefore implicitly trusts whatever resolver the
       inference call uses at runtime. Deployments on a hostile or weak DNS path
       should prefer IP literals or CIDR ranges (``10.0.0.0/24``) over hostnames
       so the trust boundary stays under operator control.
    """
    if not isinstance(host, str) or not host:
        return False
    # Charset gate before strip so external callers (PR-7 tools, PR-8
    # iot) that import this function directly via ``__all__`` are also
    # protected from CRLF / NUL / C0 control bytes. ``str.strip()``
    # below otherwise silently drops ``\r\n\t\v\f``, letting
    # ``"localhost\r\n"`` pass membership while preserving the
    # injection-shaped bytes for any caller that bypasses
    # :func:`validate_command`. AGENTS.md > Review Learnings (#92).
    if not _SAFE_PASSTHROUGH_RE.fullmatch(host):
        return False
    host_lc = host.strip().lower()
    allowlist = _policy_host_allowlist()

    for entry in allowlist:
        if host_lc == entry.strip().lower():
            return True

    try:
        ip = ipaddress.ip_address(host_lc)
    except ValueError:
        return False
    for entry in allowlist:
        try:
            net = ipaddress.ip_network(entry, strict=False)
        except ValueError:
            continue
        if ip in net:
            return True
    return False


# --- HuggingFace repo / local model path / policy_type allowlists -------


@functools.lru_cache(maxsize=1)
def _hf_repo_allowlist_cached(raw: str) -> tuple[str, ...]:
    """Cached parse of ``STRANDS_MESH_HF_REPO_ALLOW``.

    Cached parse keyed on the raw env-var string. Charset-validates each
    operator-supplied entry via :func:`_validate_env_allowlist_entries`.
    Strips trailing ``/`` so an operator who pastes ``"nvidia/"``-style
    prefixes still matches the bare-org pattern, and lowercases each
    entry for case-insensitive comparison against ``parts[0].lower()``
    in :func:`is_safe_model_path`.
    """
    builtin = ("nvidia", "huggingface", "lerobot")
    validated = _validate_env_allowlist_entries("STRANDS_MESH_HF_REPO_ALLOW", raw, _HF_REPO_ENTRY_RE)
    extras = tuple(e.strip("/").lower() for e in validated)
    return builtin + extras


def _hf_repo_allowlist() -> list[str]:
    """Return operator-extensible HF repo prefix allowlist.

    Defaults to ``["nvidia", "huggingface", "lerobot"]`` covering GR00T
    and LeRobot models. Operators extend via
    ``STRANDS_MESH_HF_REPO_ALLOW`` (comma-separated ``<org>`` or
    ``<org>/<repo>`` prefixes; charset enforced via
    :data:`_HF_REPO_ENTRY_RE`).
    """
    return list(_hf_repo_allowlist_cached(os.getenv("STRANDS_MESH_HF_REPO_ALLOW", "")))


def is_safe_model_path(path: str, *, hf_only: bool = False) -> bool:
    """Return True when *path* is a permitted HF repo id or local path.

    Checks performed:

    * Type and length: ``str``, non-empty, ``<= MAX_MODEL_PATH_LEN``.
    * Charset: ``[A-Za-z0-9_./-]+`` only (rejects shell metacharacters,
      whitespace, NUL bytes, non-ASCII).
    * No path traversal: rejects any segment equal to ``..``.
    * If *hf_only* is True (recommended for cross-mesh kwargs): the path
      MUST resemble ``<org>/<repo>`` and the org prefix MUST be in
      :func:`_hf_repo_allowlist`. This prevents an authenticated peer
      from steering a robot at an attacker-controlled HF repo.
    """
    if not isinstance(path, str) or not path:
        return False
    if len(path) > MAX_MODEL_PATH_LEN:
        return False
    if not _MODEL_PATH_RE.fullmatch(path):
        return False
    parts = path.replace("\\", "/").split("/")
    # ``..`` is always a traversal red flag, in both hf_only and
    # local-path modes. ``.`` and empty segments are legal in relative
    # local paths (e.g. ``./local/checkpoint``, doubled separators
    # collapse on disk) but illegal under the strict ``<org>/<repo>``
    # contract -- those checks live in the ``hf_only`` branch below.
    if any(seg == ".." for seg in parts):
        return False

    if hf_only:
        # Require exactly two non-empty path segments under ``hf_only``.
        # Real HuggingFace repo IDs are exactly ``<org>/<repo>``; deeper
        # paths would be branch/revision pinning (e.g. ``org/repo/blob/sha``)
        # which today's loaders do not accept. Without this check
        # ``nvidia/etc/passwd`` would pass (3 segments, traversal-shaped,
        # ``nvidia`` matches the org allowlist) and reach the HF loader,
        # which would then 404 -- but the validator's job is to enforce
        # the wire contract, not rely on downstream rejection. If a future
        # loader adds revision-pin support, relax this gate then; the
        # current gate is the conservative default for the documented
        # ``<org>/<repo>`` shape.
        # Reject any path-traversal, current-directory, or empty segment
        # under the strict ``<org>/<repo>`` wire contract. Without these,
        # degenerate inputs slipped through the prior ``non_empty_parts``
        # length-2 gate: ``nvidia//repo`` (parts=['nvidia','','repo'],
        # non_empty=2), ``nvidia/.`` (regex passes, ``.`` is not ``..``),
        # and ``nvidia/repo/`` (trailing slash). HF would 404 on these,
        # but the validator's job is to enforce the wire contract at the
        # boundary -- not to rely on downstream rejection. Same posture
        # as R1's reject of ``nvidia/etc/passwd``. (R3 review fix.)
        if any(seg in ("", ".") for seg in parts):
            return False
        if len(parts) != 2:
            return False
        org = parts[0].lower()
        allow = _hf_repo_allowlist()
        for entry in allow:
            entry_low = entry.lower()
            if "/" in entry_low:
                if path.lower().startswith(entry_low + "/") or path.lower() == entry_low:
                    return True
            elif org == entry_low:
                return True
        return False

    return True


@functools.lru_cache(maxsize=1)
def _policy_type_allowlist_cached(raw: str) -> frozenset[str]:
    """Cached parse of ``STRANDS_MESH_POLICY_TYPE_ALLOW``.

    Cached parse keyed on the raw env-var string. Charset-validates each
    entry against :data:`_POLICY_TYPE_ENTRY_RE` (lowercase identifier,
    matching the shape of the built-in :data:`_DEFAULT_POLICY_TYPES`
    set). Lowercases the raw input first so case-variant typos like
    ``"FOO,BAR"`` are normalised before the regex compare.
    """
    extras = _validate_env_allowlist_entries("STRANDS_MESH_POLICY_TYPE_ALLOW", raw.lower(), _POLICY_TYPE_ENTRY_RE)
    return frozenset(_DEFAULT_POLICY_TYPES | set(extras))


def _policy_type_allowlist() -> frozenset[str]:
    """Return the configured policy_type allowlist (defaults + env extras)."""
    return _policy_type_allowlist_cached(os.getenv("STRANDS_MESH_POLICY_TYPE_ALLOW", ""))


def _clear_security_caches_for_tests() -> None:
    """Clear the per-allowlist ``lru_cache`` instances. For test use only.

    The caches are keyed on the raw env-var string, so the next call
    after a ``monkeypatch.setenv`` change naturally re-parses. This
    helper exists for fixtures that want a deterministic WARNING emit
    (e.g. tests asserting the malformed-entry log path) -- without
    explicitly clearing, a previously-cached call from another test
    would suppress the WARNING on the second invocation.
    """
    _policy_host_allowlist_cached.cache_clear()
    _hf_repo_allowlist_cached.cache_clear()
    _policy_type_allowlist_cached.cache_clear()


def is_safe_policy_type(policy_type: str) -> bool:
    """Return True iff *policy_type* is in the allowlist."""
    if not isinstance(policy_type, str) or not policy_type:
        return False
    return policy_type.strip().lower() in _policy_type_allowlist()


def is_safe_policy_provider(provider: str) -> bool:
    """Return True iff *provider* is in the allowlist.

    ``policy_provider`` is the registry key the dispatcher passes to
    ``r._execute_task_sync`` / ``r.start_task`` to choose the policy
    class. Without this gate an authenticated peer could steer a robot
    to any registered provider, bypassing the spirit of the other
    allowlists. Shares the allowlist with :func:`is_safe_policy_type`.
    """
    if not isinstance(provider, str) or not provider:
        return False
    return provider.strip().lower() in _policy_type_allowlist()


def is_safe_server_address(addr: str) -> bool:
    """Validate a remote policy ``server_address`` (host[:port] or URL).

    Strips any scheme + port; the host portion is then checked against
    :func:`is_safe_policy_host`. Reuses the operator-controlled
    ``STRANDS_MESH_POLICY_HOST_ALLOW`` rather than introducing a
    parallel one.
    """
    if not isinstance(addr, str) or not addr:
        return False
    if len(addr) > MAX_SERVER_ADDRESS_LEN:
        return False
    s = addr.strip()

    # 1. Strip optional scheme prefix (single ://)
    if "://" in s:
        s = s.split("://", 1)[1]

    # 2. Strip optional path (everything from first /)
    s = s.split("/", 1)[0]

    # 3. Detect bracketed IPv6: [host]:port or [host]
    if s.startswith("["):
        if "]" not in s:
            return False  # Malformed bracketed address
        bracket_end = s.index("]")
        host = s[1:bracket_end]  # Extract host without brackets
        remainder = s[bracket_end + 1 :]  # Everything after ]

        if remainder:
            # Must be :port
            if not remainder.startswith(":"):
                return False  # Malformed
            port_str = remainder[1:]
            if not port_str:
                return False  # Empty port
            # Validate port is digits in [1, 65535]
            if not port_str.isdigit():
                return False
            port = int(port_str)
            if port < 1 or port > 65535:
                return False

        return is_safe_policy_host(host)

    # 4. For unbracketed: count colons
    colon_count = s.count(":")

    if colon_count == 0:
        # No colons: treat whole string as host
        return is_safe_policy_host(s)

    elif colon_count == 1:
        # One colon: treat as host:port
        host, port_str = s.rsplit(":", 1)
        # Validate port is digits in [1, 65535]
        if not port_str.isdigit():
            return False
        port = int(port_str)
        if port < 1 or port > 65535:
            return False
        return is_safe_policy_host(host)

    else:
        # Two or more colons: MUST be an unbracketed IPv6 literal
        # Try to parse as IPv6 address
        try:
            ipaddress.ip_address(s)
            # Valid IPv6, treat whole string as host
            return is_safe_policy_host(s)
        except ValueError:
            # Not a valid IPv6 address
            return False


# --- Command schema and bounds -------------------------------------------


def _coerce_float(name: str, value: Any, *, lo: float, hi: float, default: float | None) -> float:
    """Coerce *value* to a float in ``[lo, hi]`` or raise :class:`ValidationError`.

    Two defences against numeric edge cases that a naive bounds check
    misses:

    1. **NaN / inf rejection.** IEEE-754 ``NaN`` compares False against
       any bound (``nan < lo`` and ``nan > hi`` are both False), so a
       payload like ``{"duration": NaN}`` would silently pass a clamp
       and reach the robot adapter where ``time.sleep(nan)`` raises and
       ``time.monotonic() + nan`` produces a never-terminating deadline.
       Python's ``json.loads`` accepts the JSON-non-standard literal
       ``NaN`` by default so this IS reachable from the wire.
    2. **Coercion-error wrapping.** ``float(...)`` can raise
       ``OverflowError`` on a huge int or ``ValueError`` on an exotic
       Decimal. Without the ``try/except``, those would bubble out as
       bare exceptions and bypass the structured ``command_rejected``
       audit + wire response in ``_exec_cmd`` (which catches
       :class:`ValidationError` only).
    """
    if value is None:
        if default is None:
            raise ValidationError(f"{name} is required")
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value).__name__}")
    try:
        coerced = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValidationError(f"{name}: cannot coerce {value!r} to float ({exc})") from exc
    if not math.isfinite(coerced):
        raise ValidationError(f"{name} must be finite, got {coerced}")
    if coerced < lo or coerced > hi:
        raise ValidationError(f"{name}={coerced} out of bounds [{lo}, {hi}]")
    return coerced


def _coerce_int(name: str, value: Any, *, lo: int, hi: int, default: int | None) -> int:
    """Coerce *value* to an int in ``[lo, hi]`` or raise :class:`ValidationError`.

    Mirrors the defences in :func:`_coerce_float`: explicit
    finite-check on float inputs (``int(nan)`` raises a bare
    ``ValueError`` that ``_exec_cmd`` would not audit as a structured
    ``command_rejected``) and a ``try/except`` wrap so coercion errors
    surface as :class:`ValidationError` instead of bypassing the
    audit shape. Without these, ``{"action": "step", "steps": NaN}``
    would log as a generic "dispatch error" and be invisible to the
    validation-rejection forensics path.
    """
    if value is None:
        if default is None:
            raise ValidationError(f"{name} is required")
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value}")
    try:
        coerced = int(value)
    except (ValueError, OverflowError) as exc:
        raise ValidationError(f"{name}: cannot coerce {value!r} to int ({exc})") from exc
    if coerced < lo or coerced > hi:
        raise ValidationError(f"{name}={coerced} out of bounds [{lo}, {hi}]")
    return coerced


def validate_command(cmd: dict[str, Any]) -> dict[str, Any]:
    """Validate a mesh command and return a sanitised copy.

    Performed checks:

    * ``action`` must be a string and a member of :data:`ALLOWED_ACTIONS`.
    * ``execute`` and ``start`` actions require:
        - ``instruction``: non-empty str up to :data:`MAX_INSTRUCTION_LEN`.
        - ``policy_host``: in the allowlist (defaults to ``"localhost"``).
        - ``duration``: ``[0, MAX_DURATION_S]``, defaults to 30.
        - ``policy_port`` (optional): integer in ``[1, 65535]``.
        - ``pretrained_name_or_path`` (optional): HF repo, allowlist-gated.
        - ``model_path`` (optional): HF id or local path, no traversal.
        - ``policy_type`` (optional): in :func:`is_safe_policy_type`.
        - ``policy_provider``: REQUIRED, in :func:`is_safe_policy_provider`.
          No silent default -- a peer that omits this is rejected so it is
          never ambiguous whether ``mock`` was an explicit choice or a bug.
        - ``server_address`` (optional): in :func:`is_safe_server_address`.
        - ``robot_name`` (optional): peer-id charset, used by sim peers
          to disambiguate which robot in the world the policy targets.
        - ``target_pose`` (optional): list of 7 floats
          ``[x, y, z, qw, qx, qy, qz]`` for planner-style providers
          (issue #300 well-known kwarg).
        - ``target_joints`` (optional): dict of joint-name to float
          (issue #300 well-known kwarg). Bounded by
          :data:`MAX_TARGET_JOINTS`.
        - ``world_update`` (optional): opaque dict forwarded to the
          policy via ``policy_config``. Bounded by
          :data:`MAX_WORLD_UPDATE_BYTES` JSON-encoded bytes.
        - ``control_frequency`` (optional): float in ``[0.1, 2000]`` Hz.
        - ``action_horizon`` (optional): integer in ``[1, 10_000]``.
        - ``fast_mode`` (optional): boolean.
        - ``n_steps`` (optional): integer in ``[1, 10_000_000]``.
    * ``step``: ``steps`` integer in ``[1, 10_000]``, defaults to 1.
    * ``teleop_receive``: ``source_peer_id`` non-empty str.

    Raises :class:`ValidationError` on any rule violation.
    """
    if not isinstance(cmd, dict):
        raise ValidationError("command must be a dict")

    action = cmd.get("action", "status")
    if not isinstance(action, str):
        raise ValidationError("action must be a string")
    if action not in ALLOWED_ACTIONS:
        raise ValidationError(f"unknown action: {action!r} (allowed: {sorted(ALLOWED_ACTIONS)})")

    # strict per-action key allowlist.
    #
    # Earlier the validator did ``out = dict(cmd)`` and overlaid the
    # validated fields, preserving every unknown key the caller sent.
    # Today's ``Mesh._dispatch`` only reads a known whitelist of keys,
    # so this was not exploitable -- but the contract was fragile: any
    # future action handler that did ``**cmd`` or pulled a not-yet-
    # validated key would silently pick up an attacker-controlled
    # value. Defence-in-depth: build ``out`` from the *validated*
    # subset only.
    out: dict[str, Any] = {"action": action}
    # turn_id and sender_id are wire-routing fields (RPC turn correlation,
    # not action payload). Type-check, length-bound, and charset-validate
    # them so control bytes / non-string types cannot reach audit logs or
    # downstream string ops. Anything else gets dropped silently.
    for passthrough in ("turn_id", "sender_id"):
        if passthrough in cmd:
            value = cmd[passthrough]
            if not isinstance(value, str):
                raise ValidationError(f"{passthrough} must be a string (got {type(value).__name__})")
            if len(value) > MAX_PASSTHROUGH_LEN:
                raise ValidationError(f"{passthrough} exceeds {MAX_PASSTHROUGH_LEN} chars (got {len(value)})")
            if not _SAFE_PASSTHROUGH_RE.fullmatch(value):
                raise ValidationError(f"{passthrough} contains control characters, NUL, or non-printable bytes")
            out[passthrough] = value

    if action in ("execute", "start"):
        instruction = cmd.get("instruction", "")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValidationError("execute/start requires non-empty `instruction`")
        if len(instruction) > MAX_INSTRUCTION_LEN:
            raise ValidationError(f"instruction exceeds {MAX_INSTRUCTION_LEN} chars (got {len(instruction)})")
        out["instruction"] = instruction

        policy_host = cmd.get("policy_host", "localhost")
        if not is_safe_policy_host(str(policy_host)):
            raise ValidationError(
                f"policy_host={policy_host!r} not in allowlist. Set STRANDS_MESH_POLICY_HOST_ALLOW to extend."
            )
        # R7 defence-in-depth. ``is_safe_policy_host`` now applies the
        # same charset gate before its internal strip, so this
        # post-check is redundant for in-process call sites. Kept
        # because we Reject at the validator boundary regardless of
        # how the membership compare is implemented; a future refactor
        # of the allowlist must not silently drop the wire rejection.
        host_str = str(policy_host)
        if not _SAFE_PASSTHROUGH_RE.fullmatch(host_str):
            raise ValidationError(
                f"policy_host={policy_host!r} contains control characters (CRLF/NUL/C0). Use printable ASCII only."
            )
        out["policy_host"] = policy_host

        out["duration"] = _coerce_float(
            "duration",
            cmd.get("duration", 30.0),
            lo=0.0,
            hi=MAX_DURATION_S,
            default=30.0,
        )

        if "policy_port" in cmd and cmd["policy_port"] is not None:
            out["policy_port"] = _coerce_int("policy_port", cmd["policy_port"], lo=1, hi=65535, default=None)

        if "pretrained_name_or_path" in cmd:
            value = cmd["pretrained_name_or_path"]
            if not isinstance(value, str) or not is_safe_model_path(value, hf_only=True):
                raise ValidationError(
                    f"pretrained_name_or_path={value!r} not in allowlist. Set "
                    "STRANDS_MESH_HF_REPO_ALLOW to add an org/repo prefix."
                )
            out["pretrained_name_or_path"] = value

        if "model_path" in cmd:
            value = cmd["model_path"]
            if not isinstance(value, str) or not is_safe_model_path(value, hf_only=False):
                raise ValidationError(
                    f"model_path={value!r} contains disallowed characters or path-traversal segments."
                )
            out["model_path"] = value

        if "policy_type" in cmd:
            value = cmd["policy_type"]
            if not isinstance(value, str) or not is_safe_policy_type(value):
                raise ValidationError(
                    f"policy_type={value!r} not in allowlist. Set STRANDS_MESH_POLICY_TYPE_ALLOW to extend."
                )
            out["policy_type"] = value.strip().lower()

        if "policy_provider" in cmd:
            value = cmd["policy_provider"]
            if not isinstance(value, str) or not is_safe_policy_provider(value):
                raise ValidationError(
                    f"policy_provider={value!r} not in allowlist. "
                    "Set STRANDS_MESH_POLICY_TYPE_ALLOW to extend "
                    "(provider and policy_type share one allowlist)."
                )
            out["policy_provider"] = value.strip().lower()
        else:
            raise ValidationError(
                "policy_provider is required for execute/start actions; "
                "set it explicitly (e.g. 'mock' for the noop policy). "
                "Silent defaults are not honoured on the security boundary."
            )

        if "server_address" in cmd:
            value = cmd["server_address"]
            if not isinstance(value, str) or not is_safe_server_address(value):
                raise ValidationError(
                    f"server_address={value!r} host not in allowlist. Set STRANDS_MESH_POLICY_HOST_ALLOW to extend."
                )
            # Same CRLF/NUL/control-byte gate as policy_host.
            if not _SAFE_PASSTHROUGH_RE.fullmatch(value):
                raise ValidationError(
                    f"server_address={value!r} contains control characters (CRLF/NUL/C0). Use printable ASCII only."
                )
            out["server_address"] = value

        # Sim-targeted execute/start fields. These are admitted only for
        # the ``execute`` / ``start`` actions and are inert when the
        # receiving peer is a HardwareRobot - ``Mesh._dispatch`` ignores
        # them on the hardware path. Validating them here keeps the wire
        # schema honest end-to-end so a malicious peer cannot smuggle
        # control-byte instruction strings in via ``robot_name`` or
        # exhaust the dispatcher with a multi-MB ``world_update`` blob.
        if "robot_name" in cmd:
            value = cmd["robot_name"]
            if not isinstance(value, str) or not value:
                raise ValidationError("robot_name must be a non-empty string")
            if len(value) > MAX_PEER_ID_LEN:
                raise ValidationError(f"robot_name length {len(value)} > MAX_PEER_ID_LEN ({MAX_PEER_ID_LEN}).")
            if not _PEER_ID_RE.fullmatch(value):
                raise ValidationError(
                    "robot_name must match [A-Za-z0-9_.-]+ (no whitespace, NULs, "
                    "control chars, shell metacharacters, or '/')."
                )
            out["robot_name"] = value

        # Issue #300 well-known per-call policy kwargs. Forwarded into
        # ``policy_config`` by the dispatcher; planner-style providers
        # (cuRobo, MoveIt2, MPC) consume them, VLA providers ignore them.
        if "target_pose" in cmd:
            value = cmd["target_pose"]
            if not isinstance(value, list) or len(value) != 7:
                raise ValidationError("target_pose must be a list of 7 floats [x, y, z, qw, qx, qy, qz]")
            coerced_pose: list[float] = []
            for i, component in enumerate(value):
                coerced_pose.append(_coerce_float(f"target_pose[{i}]", component, lo=-1e6, hi=1e6, default=None))
            out["target_pose"] = coerced_pose

        if "target_joints" in cmd:
            value = cmd["target_joints"]
            if not isinstance(value, dict):
                raise ValidationError("target_joints must be a dict mapping joint name -> float")
            if len(value) > MAX_TARGET_JOINTS:
                raise ValidationError(
                    f"target_joints has {len(value)} entries > MAX_TARGET_JOINTS ({MAX_TARGET_JOINTS})."
                )
            coerced_joints: dict[str, float] = {}
            for joint_name, joint_value in value.items():
                if not isinstance(joint_name, str) or not joint_name:
                    raise ValidationError("target_joints keys must be non-empty strings")
                if len(joint_name) > MAX_PEER_ID_LEN:
                    raise ValidationError(
                        f"target_joints key length {len(joint_name)} > MAX_PEER_ID_LEN ({MAX_PEER_ID_LEN})."
                    )
                if not _PEER_ID_RE.fullmatch(joint_name):
                    raise ValidationError(
                        f"target_joints key {joint_name!r} must match [A-Za-z0-9_.-]+ "
                        "(no whitespace, NULs, control chars, shell metacharacters, or '/')."
                    )
                coerced_joints[joint_name] = _coerce_float(
                    f"target_joints[{joint_name}]", joint_value, lo=-1e6, hi=1e6, default=None
                )
            out["target_joints"] = coerced_joints

        if "world_update" in cmd:
            value = cmd["world_update"]
            if value is not None and not isinstance(value, dict):
                raise ValidationError("world_update must be a dict or null")
            if isinstance(value, dict):
                # Bound the encoded size - mesh treats world_update as
                # opaque and forwards it to the planner provider; we
                # only need to keep it from becoming a DoS vector.
                try:
                    encoded = json.dumps(value)
                except (TypeError, ValueError) as exc:
                    raise ValidationError(f"world_update is not JSON-serialisable: {exc}") from exc
                if len(encoded.encode("utf-8")) > MAX_WORLD_UPDATE_BYTES:
                    raise ValidationError(
                        f"world_update encoded size > MAX_WORLD_UPDATE_BYTES ({MAX_WORLD_UPDATE_BYTES})."
                    )
            out["world_update"] = value

        # Optional sim-side controls. Bounds match the SimEngine.run_policy
        # surface so a wire-side ``tell()`` cannot drive the runner to
        # absurd frequencies / step counts.
        if "control_frequency" in cmd:
            out["control_frequency"] = _coerce_float(
                "control_frequency", cmd["control_frequency"], lo=0.1, hi=2000.0, default=None
            )
        if "action_horizon" in cmd:
            out["action_horizon"] = _coerce_int("action_horizon", cmd["action_horizon"], lo=1, hi=10_000, default=None)
        if "fast_mode" in cmd:
            value = cmd["fast_mode"]
            if not isinstance(value, bool):
                raise ValidationError("fast_mode must be a boolean")
            out["fast_mode"] = value
        if "n_steps" in cmd:
            out["n_steps"] = _coerce_int("n_steps", cmd["n_steps"], lo=1, hi=10_000_000, default=None)

    elif action == "step":
        out["steps"] = _coerce_int("steps", cmd.get("steps", 1), lo=1, hi=10_000, default=1)

    elif action == "teleop_receive":
        source = cmd.get("source_peer_id", "")
        if not isinstance(source, str) or not source:
            raise ValidationError("teleop_receive requires non-empty source_peer_id")
        # ``source_peer_id`` flows into ``r.start_teleop_receive(source, dev)``
        # and into log messages, and is concatenated into device-key state. An
        # authenticated peer publishing a ``teleop_receive`` cmd whose
        # ``source_peer_id`` carries arbitrary unicode / control characters /
        # NUL bytes / shell metacharacters has no business reaching downstream
        # code, regardless of whether today's downstream consumers happen to
        # be safe. The validator's job is to enforce the contract at the wire.
        if len(source) > MAX_PEER_ID_LEN:
            raise ValidationError(
                f"teleop_receive.source_peer_id length {len(source)} > MAX_PEER_ID_LEN ({MAX_PEER_ID_LEN})."
            )
        if not _PEER_ID_RE.fullmatch(source):
            raise ValidationError(
                "teleop_receive.source_peer_id must match [A-Za-z0-9_.-]+ "
                "(no whitespace, NULs, control chars, shell metacharacters, or '/')."
            )
        out["source_peer_id"] = source
        # device_name is optional and defaults to "leader" in _dispatch.
        if "device_name" in cmd:
            device = cmd["device_name"]
            if not isinstance(device, str):
                raise ValidationError("teleop_receive.device_name must be a string")
            # Same charset + length discipline for device_name -- it is
            # concatenated into log messages and used as a key in internal
            # state mappings (e.g. the per-device state dict).
            if len(device) > MAX_PEER_ID_LEN:
                raise ValidationError(
                    f"teleop_receive.device_name length {len(device)} > MAX_PEER_ID_LEN ({MAX_PEER_ID_LEN})."
                )
            if not _PEER_ID_RE.fullmatch(device):
                raise ValidationError(
                    "teleop_receive.device_name must match [A-Za-z0-9_.-]+ "
                    "(no whitespace, NULs, control chars, shell metacharacters, or '/')."
                )
            out["device_name"] = device

    elif action == "teleop_stop":
        # device_name optional; if present, must be a string.
        if "device_name" in cmd:
            device = cmd["device_name"]
            if device is not None and not isinstance(device, str):
                raise ValidationError("teleop_stop.device_name must be a string or null")
            out["device_name"] = device

    elif action == "resume":
        # override_code is the operator-supplied second factor for
        # clearing an estop lockout. Bound the type and length defensively
        # so a non-string or oversized value cannot reach
        # Mesh._resume_lockout (which calls.strip() and would
        # raise AttributeError on a list/dict, surfacing as a generic
        # dispatch error rather than a clean ValidationError).
        override_code = cmd.get("override_code", "")
        if not isinstance(override_code, str):
            raise ValidationError("resume.override_code must be a string")
        if len(override_code) > MAX_OVERRIDE_CODE_LEN:
            raise ValidationError(f"resume.override_code too long (>{MAX_OVERRIDE_CODE_LEN} chars)")
        if override_code and not _SAFE_PASSTHROUGH_RE.fullmatch(override_code):
            raise ValidationError(
                "resume.override_code contains control characters (CRLF/NUL/C0). Use printable ASCII only."
            )
        out["override_code"] = override_code

    return out


def validate_device_rpc(function: str, params: Any = None) -> tuple[str, dict[str, Any]]:
    """Validate a Device Connect *native* RPC call and return a sanitised copy.

    This is the validation path for device-defined functions invoked directly
    over Device Connect (``conn.invoke(target, function, params)``) -- e.g. the
    Reachy's ``nod`` / ``look`` / ``playMove``. Unlike :func:`validate_command`,
    it deliberately does NOT enforce :data:`ALLOWED_ACTIONS`: that allowlist
    describes the SO-100/SO-101 policy-robot dispatch surface, not an arbitrary
    device's advertised function set. A Reachy legitimately exposes ``nod``;
    rejecting it because it is not in the policy allowlist is the bug this
    function closes.

    What it DOES enforce (defence-in-depth, since the function name and params
    flow into the device runtime, RPC subjects, and audit logs):

    * ``function``: non-empty ``str`` matching :data:`_DC_RPC_FUNC_RE`
      (``[A-Za-z_][A-Za-z0-9_]*``), at most :data:`MAX_DC_RPC_FUNC_LEN` chars.
      No dots / slashes / whitespace / control bytes / shell metacharacters.
    * ``params``: ``None`` or a JSON object (``dict``) whose keys are
      identifier-safe strings and whose JSON-encoded size is bounded by
      :data:`MAX_DC_RPC_PARAMS_BYTES`. Values are left opaque (the device
      contract defines them) but the whole object must be JSON-serialisable.

    Returns ``(function, params_dict)`` -- ``params_dict`` is ``{}`` when no
    params were supplied. Raises :class:`ValidationError` on any violation.
    """
    if not isinstance(function, str) or not function:
        raise ValidationError("device_rpc requires a non-empty function name (string)")
    if len(function) > MAX_DC_RPC_FUNC_LEN:
        raise ValidationError(
            f"device_rpc function name length {len(function)} > MAX_DC_RPC_FUNC_LEN ({MAX_DC_RPC_FUNC_LEN})."
        )
    if not _DC_RPC_FUNC_RE.fullmatch(function):
        raise ValidationError(
            f"device_rpc function={function!r} must match [A-Za-z_][A-Za-z0-9_]* "
            "(no dots, slashes, whitespace, control chars, or shell metacharacters)."
        )

    if params is None:
        return function, {}
    if not isinstance(params, dict):
        raise ValidationError("device_rpc params must be a JSON object (dict) or null")

    # Keys must be identifier-safe; the device defines what values mean, so we
    # leave them opaque but require the whole object to be JSON-serialisable
    # and size-bounded.
    for key in params:
        if not isinstance(key, str) or not key:
            raise ValidationError("device_rpc params keys must be non-empty strings")
        if len(key) > MAX_DC_RPC_FUNC_LEN:
            raise ValidationError(
                f"device_rpc params key length {len(key)} > MAX_DC_RPC_FUNC_LEN ({MAX_DC_RPC_FUNC_LEN})."
            )
        if not _DC_RPC_FUNC_RE.fullmatch(key):
            raise ValidationError(
                f"device_rpc params key {key!r} must match [A-Za-z_][A-Za-z0-9_]* "
                "(no dots, slashes, whitespace, control chars, or shell metacharacters)."
            )
    try:
        encoded = json.dumps(params)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"device_rpc params is not JSON-serialisable: {exc}") from exc
    if len(encoded.encode("utf-8")) > MAX_DC_RPC_PARAMS_BYTES:
        raise ValidationError(f"device_rpc params encoded size > MAX_DC_RPC_PARAMS_BYTES ({MAX_DC_RPC_PARAMS_BYTES}).")

    return function, dict(params)


def validate_input_frame(action: Any) -> dict[str, float]:
    """Validate and sanitise a teleop input frame, returning a clean copy.

    A teleop input frame is the flat ``{motor_name: float}`` payload
    streamed by :class:`~strands_robots.mesh.input.InputPublisher` at up
    to 50 Hz and applied by
    :class:`~strands_robots.mesh.input.InputReceiver` via
    ``robot.send_action()``. Unlike :func:`validate_command`, this is not
    a dispatch envelope -- it is raw actuator data -- so it gets its own
    bounded validator.

    ``InputReceiver._on_input`` applied
    frames straight to ``send_action()`` with no validation, so any
    LAN-adjacent peer that discovered a source peer_id could drive the
    follower's joints directly, bypassing the action allowlist + rate
    limit that guard the command path.

    Performed checks:

    * Frame must be a ``dict``.
    * At most :data:`MAX_INPUT_FRAME_KEYS` keys (DoS bound).
    * Each key: ``str``, ``<= MAX_INPUT_KEY_LEN`` chars, matching
      :data:`_INPUT_KEY_RE` (no control bytes / path separators / shell
      metacharacters).
    * Each value: coercible to ``float``, **finite** (no ``nan`` /
      ``inf``), and within ``+/- MAX_INPUT_VALUE_ABS``.

    Returns a sanitised ``dict[str, float]`` containing only validated
    entries. Raises :class:`ValidationError` on any violation.
    """
    if not isinstance(action, dict):
        raise ValidationError(f"input frame must be a dict (got {type(action).__name__})")
    if len(action) > MAX_INPUT_FRAME_KEYS:
        raise ValidationError(f"input frame has too many keys: {len(action)} > {MAX_INPUT_FRAME_KEYS}")

    out: dict[str, float] = {}
    for key, value in action.items():
        if not isinstance(key, str):
            raise ValidationError(f"input frame key must be a string (got {type(key).__name__})")
        if not key or len(key) > MAX_INPUT_KEY_LEN:
            raise ValidationError(f"input frame key length out of range: {key!r}")
        if not _INPUT_KEY_RE.fullmatch(key):
            raise ValidationError(f"input frame key has illegal characters: {key!r}")

        # Coerce to float with a finite check. bool is an int subclass;
        # reject it explicitly so a stray True/False can't masquerade as
        # a 1.0/0.0 actuator command.
        if isinstance(value, bool):
            raise ValidationError(f"input frame value for {key!r} must be numeric, not bool")
        if hasattr(value, "item"):
            # numpy scalar / 0-d array -> python scalar
            try:
                value = value.item()
            except Exception as exc:  # pragma: no cover - defensive
                raise ValidationError(f"input frame value for {key!r} is not a scalar") from exc
        if not isinstance(value, (int, float)):
            raise ValidationError(f"input frame value for {key!r} must be numeric (got {type(value).__name__})")
        fval = float(value)
        if not math.isfinite(fval):
            raise ValidationError(f"input frame value for {key!r} must be finite, got {fval}")
        _value_abs = _input_value_abs()
        if abs(fval) > _value_abs:
            raise ValidationError(f"input frame value for {key!r} out of range: |{fval}| > {_value_abs}")
        out[key] = fval

    return out


__all__ = [
    "ALLOWED_ACTIONS",
    "MAX_DURATION_S",
    "MAX_INSTRUCTION_LEN",
    "MAX_MODEL_PATH_LEN",
    "MAX_INPUT_FRAME_KEYS",
    "MAX_INPUT_KEY_LEN",
    "MAX_INPUT_VALUE_ABS",
    "MAX_OVERRIDE_CODE_LEN",
    "MAX_PASSTHROUGH_LEN",
    "MAX_PEER_ID_LEN",
    "MAX_SERVER_ADDRESS_LEN",
    "MAX_TARGET_JOINTS",
    "MAX_WORLD_UPDATE_BYTES",
    "SecurityError",
    "ValidationError",
    "is_safe_model_path",
    "is_safe_policy_host",
    "is_safe_policy_provider",
    "is_safe_policy_type",
    "is_safe_server_address",
    "validate_command",
    "validate_device_rpc",
    "validate_input_frame",
    "MAX_DC_RPC_FUNC_LEN",
    "MAX_DC_RPC_PARAMS_BYTES",
    "LockoutError",
]
