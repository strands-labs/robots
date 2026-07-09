"""Zenoh config builders for the strands-robots mesh.

This module owns every ``insert_json5`` call that hardens the Zenoh
session: namespace isolation, scouting policy, transport DoS bounds,
per-key-expression rate / size caps, mTLS, and access control.

Public functions return ``(path, json5_value)`` pairs ready to feed to
``zenoh.Config.insert_json5``. They never touch a ``zenoh.Config``
object directly so the builders can be unit-tested without the wheel
installed.

The mesh enables the Zenoh built-in security primitives by default;
operators do not get to disable them. There is no permissive fallback,
no PSK, no application-layer envelope. Identity is bound at the TLS
handshake (``cert_common_names``); authorisation is bound at the ACL
(``key_exprs`` + ``messages`` + ``flows``); rate / size caps are
enforced at the transport before bytes hit the deserialiser.

Configuration env vars
----------------------
``STRANDS_MESH_NAMESPACE``
    Fleet prefix prepended to every key-expression. Default
    ``strands``. Two fleets with different namespaces cannot
    collide on the same network.

``STRANDS_MESH_MULTICAST``
    ``true`` to enable multicast scouting. Default ``false`` -- gossip
    is the only discovery channel and ``connect/endpoints`` must be set
    explicitly. This closes the LAN-attacker-enrollment surface.

``STRANDS_MESH_MAX_SESSIONS``
    Hard cap on simultaneous unicast sessions. Default ``256``.

``STRANDS_MESH_MAX_CMD_BYTES``
    Per-message byte cap on ``cmd`` / ``broadcast`` topics enforced via
    ``low_pass_filter``. Default ``16384`` (mesh commands are small
    JSON; anything larger is jumbo-frame DoS).

``STRANDS_MESH_MAX_CAMERA_BYTES``
    Per-message byte cap on camera topics. Default ``1048576`` (1 MiB).

``STRANDS_MESH_CMD_RATE_HZ``
    Per-key-expression frequency cap for ``cmd`` topics enforced via
    ``downsampling``. Default ``20.0`` Hz.

``STRANDS_MESH_SAFETY_RATE_HZ``
    Per-key-expression frequency cap on ``safety/**`` topics. Default
    ``2.0`` Hz. Caps novel-``t`` estop/resume floods that bypass the
    receiver-side replay cache. Operators with a legitimate need for
    higher safety throughput (sensor-driven safety event streams) can
    raise this; the floor is the receiver-side HMAC + freshness cost.

``STRANDS_MESH_MAX_SAFETY_BYTES``
    Per-message byte cap on ``safety/**`` topics. Default ``4096``.
    Safety envelopes are small JSON dicts; jumbo-frame envelopes on
    this topic are DoS targeting the receiver HMAC + freshness math.

``STRANDS_MESH_AUTH_MODE``
    ``mtls`` (default) or ``none``. ``none`` is a development-only mode
    that skips the TLS terminator and ACL -- never run it on a network
    you do not fully trust. The mesh still emits namespace, scouting,
    and DoS-cap config in ``none`` mode; only the auth + ACL blocks are
    omitted.

``STRANDS_MESH_TLS_CA``
    Filesystem path to the CA bundle used to validate peer certificates.
    Required when ``STRANDS_MESH_AUTH_MODE=mtls``.

``STRANDS_MESH_TLS_CERT``
    Filesystem path to this peer's certificate (PEM).

``STRANDS_MESH_TLS_KEY``
    Filesystem path to this peer's private key (PEM, mode 0o600 on POSIX).
    On non-POSIX hosts (Windows) ``_resolve_tls_paths`` does not enforce
    the file mode -- the loader skips the ``stat().st_mode`` check because
    POSIX modes do not map cleanly onto NTFS ACLs. Operators on Windows
    must rely on filesystem ACLs (e.g. restrict the key file to a single
    Windows account) rather than the loader's mode gate.

``STRANDS_MESH_ACL_FILE``
    Filesystem path to a JSON5 ACL file. When unset, the built-in
    permissive ACL from :func:`~strands_robots.mesh._acl_config.default_acl`
    is used: any CA-signed peer may publish/subscribe on any key. Operators
    who require role separation between robots and operators must supply
    a custom ACL file (template at ``examples/mesh_acl_example.json5``).
    See CHANGELOG.md Section 8 for the rationale (Zenoh 1.x ACL CN-glob
    quirks made a true default-deny silently total-deny on first run).

``STRANDS_MESH_ACCEPT_PERMISSIVE_ACL``
    Set to ``1`` / ``true`` / ``yes`` to acknowledge the permissive
    built-in default ACL under ``STRANDS_MESH_AUTH_MODE=mtls``. When
    unset, ``Mesh.start`` refuses to bring up the wire if the ACL shape
    is permissive (default), preventing the "fleet thinks mTLS protects
    them, but ACL is wide open" silent-misconfiguration footgun. The
    opt-in is intended for dev/lab postures where role separation is
    deliberately deferred; it does NOT silence the per-session
    permissive WARNING (still emitted at INFO instead of ERROR so the
    posture stays auditable). Production fleets must supply
    ``STRANDS_MESH_ACL_FILE`` instead.

``STRANDS_MESH_I_KNOW_THIS_IS_INSECURE``
    Set to ``1`` / ``true`` / ``yes`` to acknowledge running with
    ``STRANDS_MESH_AUTH_MODE=none`` (no TLS, no ACL). Required as a
    second factor on top of ``AUTH_MODE=none`` -- the env var alone
    is not enough to bring up an insecure session. The intent is
    documentation: an operator searching for "why won't none mode
    start" is forced to read the warning text and the variable name
    before the wire comes up. Never set this on a network you do not
    fully trust.

``STRANDS_MESH_LOCAL_DEV``
    Set to ``1`` / ``true`` / ``yes`` for a one-variable localhost
    developer preset. Defaults ``AUTH_MODE`` to ``none`` AND satisfies
    the ``_I_KNOW_THIS_IS_INSECURE`` second factor by itself, so a
    fresh ``Robot()`` joins the mesh with zero security setup -- the
    "no setup" promise from the README. An explicit
    ``STRANDS_MESH_AUTH_MODE`` still overrides (force ``mtls`` even in
    local dev). Intended for single-machine experiments only; never
    set on a shared or production network.

``STRANDS_MESH_FILTER_INTERFACES``
    Comma-separated allowlist of network interface names (e.g.
    ``eth0,wlan0``) used by ``low_pass_filter`` when enumerating NICs
    for per-message size caps. Empty / unset means apply the default
    wildcard (``*``) -- all interfaces are subject to the cap.
    Operators on hosts with many virtual interfaces (containers, VPN
    tunnels) can use this to scope the cap to the physical/wire
    interfaces. The value is honoured by
    :func:`_filter_interfaces` and surfaced in the
    ``low_pass_filter`` block.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections import OrderedDict
from pathlib import Path

# One-shot flag for the non-POSIX TLS-mode warning. ``_resolve_tls_paths``
# is called once per session open, so without this flag a long-running
# Windows process would emit the same WARNING per peer connection. The
# module-level dict (rather than a plain bool) keeps mutation explicit at
# the call site without needing a ``global`` declaration.
# Key the one-shot WARNING on
# the resolved key path (with mtime fingerprint for rotation detection)
# rather than a single boolean cell. A long-running process that
# rotates ``STRANDS_MESH_TLS_KEY`` to a different file used to see
# the WARNING for the first key only; subsequent rotations were
# silently muted even though the documented Windows-mode-skip
# guarantee applies per-file. Bound at 16 entries to cap memory in
# the rotation-loop attacker case.
# #307: use an insertion-ordered dict (acting as an ordered set) so eviction
# is deterministic FIFO -- matching the _ACL_CACHE eviction order in
# _acl_config.py (``pop(next(iter(...)))``). Both bounded mesh caches now
# share one eviction discipline; see AGENTS.md cache-eviction note.
_NON_POSIX_TLS_WARNED_KEYS: OrderedDict[tuple[str, int], None] = OrderedDict()
_NON_POSIX_TLS_WARNED_LOCK = threading.Lock()
_NON_POSIX_TLS_WARNED_MAX = 16


def _is_posix() -> bool:
    """Indirection over ``os.name == "posix"`` for testability.

    ``monkeypatch.setattr`` on this module-level function lets a test
    exercise the non-POSIX TLS-mode-warning branch without setting
    ``os.name`` itself, which would corrupt ``pathlib.Path``
    instantiation on POSIX hosts (``Path`` chooses ``PosixPath`` vs
    ``WindowsPath`` based on ``os.name`` at construction time).
    """
    return os.name == "posix"


logger = logging.getLogger(__name__)


#: Fleet namespace fallback when ``STRANDS_MESH_NAMESPACE`` is unset.
#:
#: This must match the literal topic prefix every mesh component emits
#: (`mesh/core.py`, `mesh/sensors.py`, `mesh/input.py`, the IoT path).
#: The `namespace` Zenoh config field provides routing isolation --
#: two fleets with different namespaces cannot exchange messages even
#: when their key-expressions collide. The default below tracks the
#: hardcoded `strands/...` topic prefix so the built-in ACL key_exprs
#: match the wire keys exactly.
DEFAULT_NAMESPACE: str = "strands"

#: Hard cap on simultaneous Zenoh unicast sessions.
DEFAULT_MAX_SESSIONS: int = 256

#: Per-message byte cap on cmd / broadcast topics.
DEFAULT_MAX_CMD_BYTES: int = 16 * 1024

#: Per-message byte cap on camera frames.
DEFAULT_MAX_CAMERA_BYTES: int = 1 * 1024 * 1024

#: Per-key-expression frequency cap on cmd topics (Hz).
DEFAULT_CMD_RATE_HZ: float = 20.0

#: Per-key-expression frequency cap on safety topics (Hz).
#:
#: legitimate operator estop / resume traffic is far below
#: 1 Hz steady-state. A peer publishing on ``safety/**`` faster
#: than this rate is throttled at the transport, capping the
#: novel-`t` flood surface that bypasses the receiver-side replay defence
#: replay cache.
DEFAULT_SAFETY_RATE_HZ: float = 2.0

#: Per-message byte cap on safety topics. Safety envelopes are
#: small JSON dicts; a 100 KiB envelope on this topic is jumbo-
#: frame DoS targeting the receiver-side HMAC + freshness math.
DEFAULT_MAX_SAFETY_BYTES: int = 4 * 1024


def resolve_namespace() -> str:
    """Return the configured fleet namespace.

    Reads ``STRANDS_MESH_NAMESPACE`` and falls back to
    :data:`DEFAULT_NAMESPACE`. Empty / whitespace values fall through
    to the default so an operator setting ``STRANDS_MESH_NAMESPACE=""``
    does not accidentally produce keys like ``"//presence"``.
    """
    raw = os.getenv("STRANDS_MESH_NAMESPACE", "").strip()
    return raw or DEFAULT_NAMESPACE


def _local_dev_enabled() -> bool:
    """True when ``STRANDS_MESH_LOCAL_DEV`` is set to a truthy value.

    The localhost-only developer preset (GH #373 friction #7). When on, the
    mesh runs without mTLS/ACL for frictionless single-machine experiments,
    and ``LOCAL_DEV`` itself acts as the explicit "I accept insecure" second
    factor -- so the operator does not ALSO need
    ``STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1``. Truthy: ``1``, ``true``,
    ``yes`` (case-insensitive). This is intentionally a *separate* knob from
    ``STRANDS_MESH_AUTH_MODE`` so production code paths that read auth mode
    directly never accidentally inherit a dev default.
    """
    return os.getenv("STRANDS_MESH_LOCAL_DEV", "").strip().lower() in ("1", "true", "yes")


def resolve_auth_mode() -> str:
    """Return the configured auth mode.

    One of ``"mtls"`` (default) or ``"none"``. Any other value is
    rejected with a ``ValueError`` so a typo does not silently disable
    auth.

    ``"none"`` disables both the mTLS terminator and the ACL block --
    a single env var that turns the entire wire-layer security model
    off. To prevent a typo / forgotten env-var / leaked CI fixture
    from silently disabling wire auth in production, ``"none"`` is
    additionally gated on ``STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1``
    (case-insensitive: ``1``, ``true``, ``yes``). Without that explicit
    second factor, ``"none"`` raises ``ValueError`` at config-build
    time -- the burden of proof lives with the operator who is turning
    auth off.

    **Local-dev shortcut** (GH #373 friction #7): setting
    ``STRANDS_MESH_LOCAL_DEV=1`` defaults the auth mode to ``"none"`` AND
    satisfies the insecure-acknowledgement second factor on its own -- one
    env var, not two, to run a frictionless localhost mesh. An explicit
    ``STRANDS_MESH_AUTH_MODE`` still wins (so you can force ``mtls`` even in
    local dev), and an explicit ``AUTH_MODE=none`` under ``LOCAL_DEV`` no
    longer needs the ``_I_KNOW_THIS_IS_INSECURE`` factor because ``LOCAL_DEV``
    is the acknowledgement.
    """
    local_dev = _local_dev_enabled()
    default_mode = "none" if local_dev else "mtls"
    raw = os.getenv("STRANDS_MESH_AUTH_MODE", default_mode).strip().lower()
    if raw not in ("mtls", "none"):
        raise ValueError(f"STRANDS_MESH_AUTH_MODE={raw!r} not supported (expected 'mtls' or 'none')")
    if raw == "none" and not local_dev:
        ack = os.getenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", "").strip().lower()
        if ack not in ("1", "true", "yes"):
            raise ValueError(
                "STRANDS_MESH_AUTH_MODE=none disables BOTH the mTLS "
                "terminator AND the ACL block -- the entire wire-layer "
                "security model. Refusing without an explicit second "
                "factor: set STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1 to "
                "confirm. This guard prevents a typo / forgotten env-var "
                "/ leaked CI fixture from silently disabling wire auth "
                "in production."
            )
    return raw


def _bool_env(name: str, default: bool) -> bool:
    """Parse a boolean env var with a strict truthy/falsy mapping."""
    raw = os.getenv(name, "").strip().lower()
    if raw == "":
        return default
    if raw in ("true", "1", "yes", "on"):
        return True
    if raw in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean (use true/false)")


def _int_env(name: str, default: int, *, lo: int = 1, hi: int = 1 << 30) -> int:
    """Parse an integer env var clamped to ``[lo, hi]``."""
    raw = os.getenv(name, "").strip()
    if raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc
    if value < lo or value > hi:
        raise ValueError(f"{name}={value} out of bounds [{lo}, {hi}]")
    return value


def _float_env(name: str, default: float, *, lo: float = 0.0, hi: float = 1e6) -> float:
    """Parse a float env var clamped to ``[lo, hi]``.

    Rejects NaN and +/-inf explicitly. IEEE-754 ``NaN`` compares False
    against any bound (both ``nan < lo`` and ``nan > hi`` evaluate to
    False), so a naive ``value < lo or value > hi`` clamp silently
    accepts ``STRANDS_MESH_CMD_RATE_HZ=nan``; the downstream Zenoh
    ``downsampling`` rule's ``freq`` field then becomes NaN and the
    rate cap is effectively disabled with no operator-visible signal.
    Raising on non-finite input keeps the misconfig surface clean: the
    ``ValueError`` fires at module-load time with the env-var name and
    the offending value, instead of bubbling out of ``zenoh.open()``
    several frames deeper as an opaque "config invalid".
    """
    raw = os.getenv(name, "").strip()
    if raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a float") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name}={raw!r} must be finite (got {value})")
    if value < lo or value > hi:
        raise ValueError(f"{name}={value} out of bounds [{lo}, {hi}]")
    return value


def namespace_block() -> tuple[str, str]:
    """Return ``("namespace", <json5>)`` for the configured fleet namespace."""
    return ("namespace", json.dumps(resolve_namespace()))


def scouting_block() -> list[tuple[str, str]]:
    """Return scouting config: multicast off (default) + gossip on.

    Multicast on a hostile LAN is a discovery attack surface -- any host
    that joins the multicast group (224.0.0.224:7446) sees every peer's
    presence broadcast. Gossip-only with explicit ``connect/endpoints``
    is the production posture.

    Operators on a controlled LAN can opt back into multicast with
    ``STRANDS_MESH_MULTICAST=true``. We do NOT recommend it.
    """
    multicast = _bool_env("STRANDS_MESH_MULTICAST", default=False)
    return [
        ("scouting/multicast/enabled", "true" if multicast else "false"),
        ("scouting/gossip/enabled", "true"),
    ]


def transport_caps_block() -> list[tuple[str, str]]:
    """Return transport-level DoS bounds.

    Currently emits ``transport/unicast/max_sessions``. Future caps
    (timeouts, queue sizes) land here.
    """
    max_sessions = _int_env(
        "STRANDS_MESH_MAX_SESSIONS",
        DEFAULT_MAX_SESSIONS,
        lo=1,
        hi=65535,
    )
    return [("transport/unicast/max_sessions", str(max_sessions))]


def downsampling_block() -> tuple[str, str]:
    """Return ``("downsampling", <json5>)`` capping the cmd-publish rate.

    A peer publishing to ``{namespace}/*/cmd`` faster than the
    configured frequency is throttled at the transport layer -- the
    extra messages are dropped before they reach the JSON parser, so
    flood attacks cost the receiver almost nothing.
    """
    freq = _float_env(
        "STRANDS_MESH_CMD_RATE_HZ",
        DEFAULT_CMD_RATE_HZ,
        lo=0.001,
        hi=10000.0,
    )
    # safety topics need their own (lower) rate cap. Without it
    # a peer with any CA-signed cert can flood ``safety/estop`` at
    # line rate with novel ``t`` on each envelope -- bypassing the
    # receiver-side replay cache (key=(issuer_id, t)) and consuming
    # CPU on freshness arithmetic + per-receiver replay-cache pressure.
    safety_freq = _float_env(
        "STRANDS_MESH_SAFETY_RATE_HZ",
        DEFAULT_SAFETY_RATE_HZ,
        lo=0.001,
        hi=1000.0,
    )
    # See ``low_pass_filter_block`` for the namespace-vs-key_expr note:
    # ``**/cmd`` matches any prefix including the namespace one;
    # ``f"{namespace}/*/cmd"`` would not.
    rules = [
        {"key_expr": "**/cmd", "freq": freq},
        {"key_expr": "**/broadcast", "freq": freq},
        {"key_expr": "**/safety/**", "freq": safety_freq},
    ]
    return (
        "downsampling",
        json.dumps(
            [
                {
                    "id": "strands_cmd_rate_cap",
                    "messages": ["put"],
                    "flows": ["ingress"],
                    "rules": rules,
                }
            ]
        ),
    )


def _filter_interfaces() -> list[str] | None:
    """Return the operator-supplied interface allowlist, or ``None``.

    Zenoh's ``low_pass_filter`` block treats an absent ``interfaces``
    field as ``SubjectProperty::Wildcard`` (matches every link). See
    ``zenoh/src/net/routing/interceptor/low_pass.rs`` (1.x):

        let interfaces = lpf_config.interfaces
            .map(...)
            .unwrap_or(vec![SubjectProperty::Wildcard]);

    A wildcard binding is the correct posture for a fleet-wide cap:
    the cap applies regardless of which NIC a peer's link rides on,
    and there is no NIC enumeration that needs to stay in sync with
    the actual deployment topology.

    Operators with a *specific* need to bind the cap to a subset of
    NICs (e.g. excluding a high-volume telemetry NIC from the cmd
    cap) can set ``STRANDS_MESH_FILTER_INTERFACES`` (comma-separated)
    and we honour it literally. Empty / unset returns ``None`` so the
    builder omits the field entirely -- not the empty list, which
    Zenoh's ``Option<NEVec<String>>`` parser rejects with
    ``Found empty interface value`` (deny_unknown_fields + non-empty
    vec).
    """
    raw = os.getenv("STRANDS_MESH_FILTER_INTERFACES", "").strip()
    if not raw:
        return None
    parts = [iface.strip() for iface in raw.split(",") if iface.strip()]
    return parts or None


def low_pass_filter_block() -> tuple[str, str]:
    """Return ``("low_pass_filter", <json5>)`` capping per-message bytes.

    Three filters:

    * cmd / broadcast topics: 16 KiB default cap, both flows.
    * camera topics: 1 MiB default cap, ingress-only.
    * safety topics: 4 KiB default cap, both flows.

    Anything over the cap is dropped at the transport before the
    JSON parser runs.

    Interface binding: ``interfaces`` is OMITTED so Zenoh applies the
    cap to every link (``SubjectProperty::Wildcard``). Operators with a
    specific need to scope the cap to a subset of NICs supply
    ``STRANDS_MESH_FILTER_INTERFACES`` (comma-separated); see
    :func:`_filter_interfaces`. Earlier revisions enumerated every
    local NIC via psutil with a hardcoded fallback; that pattern
    silently bypassed the cap on hosts with non-canonical interface
    names (``enp0s3``, ``wlp2s0``, ``cni0``, ``wg0``,...) when psutil
    was absent. Wildcard-by-default removes that footgun.
    """
    cmd_bytes = _int_env(
        "STRANDS_MESH_MAX_CMD_BYTES",
        DEFAULT_MAX_CMD_BYTES,
        lo=128,
        hi=16 * 1024 * 1024,
    )
    cam_bytes = _int_env(
        "STRANDS_MESH_MAX_CAMERA_BYTES",
        DEFAULT_MAX_CAMERA_BYTES,
        lo=1024,
        hi=128 * 1024 * 1024,
    )
    safety_bytes = _int_env(
        "STRANDS_MESH_MAX_SAFETY_BYTES",
        DEFAULT_MAX_SAFETY_BYTES,
        lo=128,
        hi=1 * 1024 * 1024,
    )
    interfaces = _filter_interfaces()

    def _rule(rule: dict) -> dict:
        # Only attach `interfaces` when the operator explicitly opted into a
        # subset; otherwise leave it unset so Zenoh treats the rule as
        # SubjectProperty::Wildcard (applies to every link).
        if interfaces is not None:
            rule["interfaces"] = interfaces
        return rule

    return (
        "low_pass_filter",
        json.dumps(
            [
                # NOTE on key_expr globs: the Zenoh ``namespace`` config
                # field prefixes keys on the wire (see
                # zenoh/src/net/routing/namespace.rs). The interceptor
                # matches against the wire key (post-prefix), but ``**``
                # matches any prefix including the namespace one, so
                # ``**/cmd`` is robust regardless of the configured
                # namespace.
                _rule(
                    {
                        "id": "strands_cmd_size_cap",
                        "messages": ["put"],
                        "flows": ["ingress", "egress"],
                        "key_exprs": ["**/cmd", "**/broadcast"],
                        "size_limit": cmd_bytes,
                    }
                ),
                _rule(
                    {
                        "id": "strands_camera_size_cap",
                        "messages": ["put"],
                        "flows": ["ingress"],  # ingress-only, publisher trusts own frames
                        "key_exprs": ["**/camera/**"],
                        "size_limit": cam_bytes,
                    }
                ),
                _rule(
                    {
                        # safety topics need their own (smaller)
                        # byte cap. A 100 KiB safety envelope is jumbo-
                        # frame DoS targeting the receiver-side HMAC and
                        # freshness math; legitimate safety envelopes are
                        # well under 1 KiB.
                        "id": "strands_safety_size_cap",
                        "messages": ["put"],
                        "flows": ["ingress", "egress"],
                        "key_exprs": ["**/safety/**"],
                        "size_limit": safety_bytes,
                    }
                ),
            ]
        ),
    )


# --- mTLS ---------------------------------------------------------------


def _resolve_tls_paths() -> tuple[Path, Path, Path]:
    """Return ``(ca, cert, key)`` paths from env vars.

    Raises :class:`FileNotFoundError` on a missing path so
    misconfiguration fails loud at session-open time rather than
    silently downgrading to plain TCP.
    """
    ca = os.getenv("STRANDS_MESH_TLS_CA", "").strip()
    cert = os.getenv("STRANDS_MESH_TLS_CERT", "").strip()
    key = os.getenv("STRANDS_MESH_TLS_KEY", "").strip()
    if not ca or not cert or not key:
        raise ValueError(
            "STRANDS_MESH_AUTH_MODE=mtls requires "
            "STRANDS_MESH_TLS_CA, STRANDS_MESH_TLS_CERT, "
            "and STRANDS_MESH_TLS_KEY to be set"
        )
    paths = (Path(ca), Path(cert), Path(key))
    # the existence + symlink check must come
    # before any other inspection. ``is_file`` follows symlinks; we
    # do an explicit ``is_symlink`` reject first so the path used for
    # mode + load is always the real file, never an attacker-redirected
    # link target.
    for label, p in zip(("CA", "cert", "key"), paths, strict=True):
        if p.is_symlink():
            raise ValueError(
                f"mTLS {label} file {p} is a SYMLINK "
                f"(target: {os.readlink(p)!r}). Refusing -- mTLS files "
                "must be real regular files at the operator-supplied path."
            )
        if not p.is_file():
            raise FileNotFoundError(f"mTLS {label} file does not exist: {p}")
    # enforce the mode 0o600 contract that the docstring (line 73)
    # and README env-var matrix promise for the private key. A 0o644 key
    # file on a shared host is a real exfiltration surface; the operator
    # who set STRANDS_MESH_TLS_KEY thinks they get the documented protection.
    # On non-POSIX (Windows) the mode 0o600 contract documented at line 73
    # and in the README env-var matrix cannot be enforced via ``stat()``.
    # Emit a one-shot WARNING so an operator running mTLS on Windows is
    # not silently led to believe the loader is verifying key-file mode
    # -- they must rely on filesystem ACLs (NTFS DACL) instead. This
    # matches the loud-on-misconfig posture of the rest of this module
    # (``_float_env`` raises on NaN, ``_int_env`` raises on out-of-bounds,
    # ``_load_acl_file`` raises on bad ``enabled``).
    #
    # On POSIX, the symlink-reject + ``lstat()`` + mode check below uses
    # so a symlink to an attacker-writable file does not silently pass
    # the mode check. Without this, ``STRANDS_MESH_TLS_KEY=/safe/key.pem``
    # pointing at a co-tenant-controlled ``/tmp/evil.pem`` (which the
    # attacker has chmod'd 0o600) would pass while the actual TLS load
    # later opens the symlink target. Symmetric with the
    # ``O_NOFOLLOW`` + lstat-reject discipline applied across
    # ``audit.py:_ensure_paths``, ``_load_seq_counters``, and
    # ``_acl_config.py:_load_acl_file``.
    #
    if not _is_posix():
        # Atomic check-and-set under lock so concurrent _build_config
        # calls (e.g. multi-threaded test harness) don't both fire the
        # WARNING. Key the
        # one-shot on (key_path, mtime_ns) so rotating
        # ``STRANDS_MESH_TLS_KEY`` to a different file (or replacing
        # the file in-place) re-arms the warning.
        try:
            _key_st = os.stat(str(paths[2]), follow_symlinks=False)
            _key_id: tuple[str, int] = (str(paths[2]), _key_st.st_mtime_ns)
        except OSError:
            # Cannot stat the key path -- fall back to path-only keying
            # so we still differentiate between rotations even when
            # filesystem ACLs hide the timestamp.
            _key_id = (str(paths[2]), 0)
        with _NON_POSIX_TLS_WARNED_LOCK:
            should_warn = _key_id not in _NON_POSIX_TLS_WARNED_KEYS
            if should_warn:
                # Bound the dict so a rogue caller looping over key files
                # cannot inflate memory. Eviction is deterministic FIFO via
                # ``popitem(last=False)`` -- the oldest-inserted key is evicted
                # first, matching _ACL_CACHE. Worst case on eviction is a
                # re-emitted WARNING for an evicted key, which is benign.
                if len(_NON_POSIX_TLS_WARNED_KEYS) >= _NON_POSIX_TLS_WARNED_MAX:
                    _NON_POSIX_TLS_WARNED_KEYS.popitem(last=False)
                _NON_POSIX_TLS_WARNED_KEYS[_key_id] = None
        if should_warn:
            logger.warning(
                "[mesh] mTLS key mode (0o600) check is SKIPPED on non-POSIX "
                "platform (%s). The README env-var matrix promises mode "
                "0o600 enforcement; on this platform that guarantee is "
                "not enforced -- rely on filesystem ACLs (e.g. NTFS DACL) "
                "to restrict %s.",
                os.name,
                paths[2],
            )
    else:
        key_path = paths[2]
        # Symlink reject already applied to all three TLS paths in the
        # CA/cert/key loop above; not re-checked here. ``lstat()``
        # returns the link's own metadata -- since the loop already
        # rejected symlinks, ``lstat`` is equivalent to ``stat`` for
        # the path we are looking at, but we keep ``lstat`` explicit to
        # match the discipline of audit.py:_ensure_paths,
        # _load_seq_counters, and _acl_config.py:_load_acl_file.
        #
        # Residual TOCTOU window: between this lstat() and Zenoh's
        # eventual open() of the same path, an attacker who controls
        # the parent directory can swap the file. Python cannot pass
        # ``O_NOFOLLOW`` into the C-level Zenoh wheel, so the residual
        # window is upstream-bound. Operators must keep the parent
        # directory non-attacker-writable (chmod 700). Tracked in
        # Zenoh upstream and the strands-robots threat model docs.
        key_mode = key_path.lstat().st_mode & 0o777
        if key_mode & 0o077:
            raise ValueError(
                f"mTLS private key {key_path} has mode 0o{key_mode:03o}; "
                "refusing world/group readable key. "
                "Run: chmod 600 " + str(key_path)
            )
    return paths


def tls_block() -> tuple[str, str]:
    """Return ``("transport/link/tls", <json5>)`` for mTLS terminator.

    Both listen-side and connect-side present the same cert (a peer can
    be either initiator or responder depending on who reaches whom
    first). Mutual TLS is mandatory; ``verify_name_on_connect`` is on so
    a peer that swaps its cert at the network layer cannot bypass CN
    matching on the ACL side.
    """
    ca, cert, key = _resolve_tls_paths()
    return (
        "transport/link/tls",
        json.dumps(
            {
                "root_ca_certificate": str(ca),
                "listen_private_key": str(key),
                "listen_certificate": str(cert),
                "connect_private_key": str(key),
                "connect_certificate": str(cert),
                "enable_mtls": True,
                "verify_name_on_connect": True,
                "close_link_on_expiration": True,
            }
        ),
    )


def link_protocols_block() -> tuple[str, str]:
    """Restrict the transport to TLS only when mTLS is on.

    Without this, an attacker could downgrade to plain TCP by being
    the first to bind a TCP listener -- Zenoh would happily accept the
    cleartext peer.
    """
    return ("transport/link/protocols", json.dumps(["tls"]))


# --- adminspace ---------------------------------------------------------


def adminspace_block() -> tuple[str, str]:
    """Lock down the admin space.

    Default in upstream Zenoh is already disabled but we set it
    explicitly so an operator who later toggles it on at the env-var
    layer can find the override centralised here.
    """
    return (
        "adminspace",
        json.dumps({"enabled": False, "permissions": {"read": False, "write": False}}),
    )
