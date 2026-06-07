"""ACL config builder for the strands-robots mesh.

Reads a JSON5 ACL file at ``STRANDS_MESH_ACL_FILE`` and returns the
serialised ``access_control`` block ready for
``zenoh.Config.insert_json5``. When the env var is unset, returns the
permissive :func:`default_acl` skeleton.

Zenoh 1.x quirks (each verified against a live session in
``tests/mesh/test_zenoh_transport_security.py``):

* ``enabled: true`` is required -- without it the entire block is a
  no-op even if rules and subjects are populated.
* ``cert_common_names`` matches LITERAL CNs only; globs and regexes
  match nothing. Operators tighten the default by enumerating each
  peer's exact cert CN in ``STRANDS_MESH_ACL_FILE``.
* Subject ``interfaces`` is OPTIONAL -- omitting it causes the subject
  to match on every link (wildcard). An empty list ``[]`` is rejected.
* ``key_exprs`` match the user-side key (the namespace prefix is
  stripped from the matcher's view), so ``**/cmd`` is the robust
  glob; ``"<namespace>/*/cmd"`` never matches.
* ``declare_subscriber`` rules live in the ``egress`` flow (the
  declare goes from subscriber to publisher); ``put`` rules live in
  ``ingress`` (the publisher's cert CN is known to the receiver).

JSON5 is the on-disk format (line + block comments, trailing commas,
unquoted keys). The loader delegates to the ``json5`` PyPI dependency
(declared in ``pyproject.toml``'s ``[mesh]`` extra and imported lazily
inside :func:`_parse_json5` so operators running without an ACL file
don't pay the import cost).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PermissiveACLError(RuntimeError):
    """Raised when an operator-supplied ACL uses the blacklist footgun
    (``default_permission='allow'`` with explicit rules) without opting
    in via ``STRANDS_MESH_ACCEPT_PERMISSIVE_ACL``. Pentest B-08 / F-14.

    The built-in permissive default (allow + EMPTY rules) is exempt --
    it is gated separately by the ``Mesh.start`` refuse-to-start path
    (:meth:`Mesh._refuse_under_permissive_default_acl`) which fires under
    mTLS. This error closes the *operator-file blacklist* footgun: a
    hand-written ``allow + rules`` ACL is an explicit, load-bearing
    anti-pattern that should fail loud rather than silently expose any
    rule gap.
    """


#: Maximum bytes of an ACL file we will load. Anything larger is almost
#: certainly an attacker probing for an OOM.
ACL_FILE_MAX_BYTES: int = 256 * 1024


# --- JSON5 parser (vendored via the ``json5`` PyPI dep) ----------------

# We delegate JSON5 parsing to the ``json5`` library (MIT, audited, ~3kLOC,
# pure Python, no native deps). Earlier revisions carried a four-pass hand-
# rolled preprocessor (``_strip_json5_comments`` -> ``_strip_trailing_commas``
# -> ``_quote_unquoted_keys`` -> ``_convert_single_quoted_strings``) that
# silently truncated on unterminated ``/*`` blocks, mis-quoted keys after
# ``[`` (object-in-array case), and produced ``json.JSONDecodeError`` column
# numbers pointing at the post-preprocessor string -- making operator
# debugging painful. The dep swap eliminates ~250 LOC of fragile state-
# machine code and gives operators precise diagnostics on malformed input.
#
# Why a third-party dep is acceptable here: the ACL file gates wire
# authorisation, so a parser that fails *closed* with a clear error is
# strictly safer than a hand-rolled approximation. ``json5`` is already
# transitively available in many Python deployments; we add it to the
# ``mesh`` extra so it ships with the rest of the wire-layer code.
# imported lazily inside ``_parse_json5`` -- only paid by
# operators who actually load an ACL file.

# json5 is imported lazily inside
# ``_parse_json5`` rather than at module top-level. Importing it eagerly every
# import of ``strands_robots.mesh`` (including ``session.py`` for
# ``auth_mode=none`` dev paths) triggered the json5 import even when
# no ACL file is loaded. Operators running with no ACL file (the
# permissive default) and no ``mesh`` extra installed got an
# ``ImportError`` at import time when they didn't need the dep.
# The loader is the only consumer; lazy-import there.


def _parse_json5(raw: str, path: Path) -> Any:
    """Parse *raw* JSON5 text into a Python object.

    Raises :class:`ValueError` with operator-friendly diagnostics on any
    malformed input. The ACL loader treats this as a fail-closed
    boundary: a malformed file does NOT silently degrade to the
    permissive default.
    """
    # Review thread _acl_config.py:85 -- use the project-standard
    # ``require_optional`` helper so the operator-facing import error
    # carries the canonical install-hint format used elsewhere in the
    # SDK (groot, libero, etc.). This still lazy-imports
    # (only operators with an ACL file pay the cost).
    from strands_robots.utils import require_optional

    try:
        _json5_mod: Any = require_optional(
            "json5",
            extra="mesh",
            purpose="parsing STRANDS_MESH_ACL_FILE (JSON5 format)",
        )
    except ImportError as exc:
        raise ImportError(
            "json5 is required to parse STRANDS_MESH_ACL_FILE -- install "
            "via ``pip install strands-robots[mesh]`` (which pulls in "
            "json5) or ``pip install json5``"
        ) from exc
    try:
        return _json5_mod.loads(raw)
    except ValueError as exc:
        # json5 raises ValueError (subclass) with a useful message that
        # includes line/column. Re-raise with the path attached so an
        # operator looking at the log sees exactly which file failed.
        raise ValueError(f"ACL file {path} is not valid JSON5: {exc}") from exc


# --- ACL file loader ---------------------------------------------------


def _load_acl_file(path: Path) -> dict[str, Any]:
    """Load and validate an ACL file.

    Refuses any file that omits ``enabled: true`` -- Zenoh silently
    no-ops the block in that case, and the loader fails closed rather
    than ship a quietly-disabled gate.
    """
    # Defence: refuse to follow symlinks AND bound the read at
    # ACL_FILE_MAX_BYTES + 1 so an attacker who races content between
    # stat() and read() cannot bypass the size cap. Mirrors the
    # O_NOFOLLOW + bounded-read discipline used for the audit log
    # (audit.py:_ensure_paths). The ACL file gates wire authorisation,
    # so the same TOCTOU + symlink-swap defences apply.
    if path.is_symlink():
        raise ValueError(
            f"refusing to load ACL file {path}: it is a SYMLINK "
            f"(target: {os.readlink(path)!r}). ACL files must be regular files."
        )
    if not path.is_file():
        raise FileNotFoundError(f"ACL file not found: {path}")
    flags = os.O_RDONLY
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags | nofollow)
    except OSError as exc:
        # ELOOP under O_NOFOLLOW = symlink raced ahead of the static check.
        raise ValueError(f"refusing to load ACL file {path}: {exc}") from exc
    try:
        # Read at most MAX+1 bytes so we can detect overflow without
        # an unbounded read.
        chunks = []
        remaining = ACL_FILE_MAX_BYTES + 1
        while remaining > 0:
            buf = os.read(fd, remaining)
            if not buf:
                break
            chunks.append(buf)
            remaining -= len(buf)
    finally:
        os.close(fd)
    raw_bytes = b"".join(chunks)
    if len(raw_bytes) > ACL_FILE_MAX_BYTES:
        raise ValueError(f"ACL file {path} is >{ACL_FILE_MAX_BYTES} bytes; refusing to load.")
    try:
        raw = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"ACL file {path} is not valid UTF-8: {exc}") from exc
    data = _parse_json5(raw, path)

    if not isinstance(data, dict):
        raise ValueError(f"ACL file {path} root must be an object")
    for required in ("default_permission", "rules", "subjects", "policies"):
        if required not in data:
            raise ValueError(f"ACL file {path} missing required field: {required!r}")
    # Require literal boolean ``True`` (identity check) rather than truthy
    # non-bool. ``enabled: 1`` (JSON5 int), ``enabled: "true"`` (string
    # typo), ``enabled: [false]`` (non-empty list) all pass ``not False``
    # but the downstream Zenoh deserializer expects a strict ``bool`` and
    # fails with an opaque "expected boolean" several frames deeper. The
    # whole point of this gate is to fail closed before Zenoh sees the
    # config, so accept only the literal type.
    if data.get("enabled") is not True:
        raise ValueError(
            f"ACL file {path} must set ``enabled: true`` (literal boolean) -- "
            f"got {data.get('enabled')!r}. Without it Zenoh silently "
            f"disables the access_control block."
        )
    if data["default_permission"] not in ("allow", "deny"):
        raise ValueError(f"ACL file {path} default_permission={data['default_permission']!r} must be 'allow' or 'deny'")
    if data["default_permission"] == "allow":
        # Reserve the blacklist warning for the actual anti-pattern --
        # ``allow`` + non-empty ``rules``. The built-in ``default_acl()``
        # (used when STRANDS_MESH_ACL_FILE is unset) ships ``allow +
        # empty rules/subjects/policies``; warning operators who copy
        # that shape into a file is asymmetric scolding. Per review
        # thread _acl_config.py:199, scope the trigger to ``rules``
        # specifically (the load-bearing part of the blacklist
        # anti-pattern) and phrase the warning to match: "allow +
        # rules" is the foot-gun, not "allow + anything".
        if data.get("rules"):
            logger.warning(
                "[acl] %s uses default_permission='allow' with %d rule(s) -- "
                "this is a blacklist policy and any rule gap exposes "
                "the mesh. Prefer 'deny' with explicit allow rules.",
                path,
                len(data["rules"]),
            )
            # B-08 / F-14: the warning above is the first-line signal; this
            # is the hard gate. An operator-supplied blacklist ACL
            # (allow + explicit rules) is a load-bearing anti-pattern --
            # any gap in the rule set exposes the mesh. Refuse to load it
            # unless the operator has explicitly acknowledged the posture
            # via STRANDS_MESH_ACCEPT_PERMISSIVE_ACL. The built-in default
            # (allow + EMPTY rules) does not reach this branch and stays
            # gated by Mesh.start's refuse-to-start path.
            accept = os.getenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            if not accept:
                raise PermissiveACLError(
                    f"ACL file {path} uses default_permission='allow' with "
                    f"{len(data['rules'])} rule(s) -- a blacklist policy where any "
                    "rule gap exposes the mesh. Refusing to load. Remediate one of:\n"
                    "  1. Rewrite the ACL with default_permission='deny' and "
                    "explicit allow rules (see examples/mesh_acl_example.json5).\n"
                    "  2. Set STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1 to acknowledge "
                    "the dev/lab posture."
                )
    _validate_acl_shape(data, path)
    return data


def _validate_acl_shape(data: dict[str, Any], path: Path) -> None:
    """Validate the shape of subjects/rules/policies after JSON parse.

    a typo like ``interface:``
    (singular) or a missing ``cert_common_names`` field silently
    degrades a role-separated ACL to "match nothing" at the Zenoh
    layer, which manifests as a silent total outage operators must
    debug from Zenoh logs. We refuse these shapes loudly at parse
    time -- the same posture the ``enabled: true`` check (added
    earlier in this function) is built around.

    Validates:

    1. ``subjects``, ``rules``, ``policies`` are lists.
    2. Every subject has ``id``. ``interfaces`` is OPTIONAL -- when
       omitted, Zenoh treats the interface dimension as
       ``SubjectProperty::Wildcard`` (matches every link); when present,
       it must be a non-empty list of non-empty strings (Zenoh rejects
       ``[]`` with ``Found empty interface value``). ``cert_common_names``
       is OPTIONAL -- when present must be a list. Subjects with
       neither ``interfaces`` nor ``cert_common_names`` match every
       peer (effectively wildcard) and operators should use them only
       when a permissive ``default_permission: "allow"`` is desired.
    3. Every rule has ``id``, ``key_exprs`` (non-empty list of
       strings), ``messages`` (non-empty list), ``flows`` (non-empty
       list), and ``permission`` (``allow`` or ``deny``).
    4. Every policy has ``rules`` and ``subjects`` referencing
       existing rule / subject ids.

    Raises ``ValueError`` with a path-prefixed message on the first
    failure. Callers should treat any failure here as a deployment
    blocker -- a malformed ACL is worse than no ACL because the
    operator believes role separation is enforced when it is not.
    """
    # 1. Top-level lists.
    for field in ("subjects", "rules", "policies"):
        if not isinstance(data[field], list):
            raise ValueError(f"ACL file {path}: {field!r} must be a list, got {type(data[field]).__name__}")

    # 2. Subjects.
    subject_ids: set[str] = set()
    for i, subj in enumerate(data["subjects"]):
        if not isinstance(subj, dict):
            raise ValueError(f"ACL file {path}: subjects[{i}] must be an object")
        sid = subj.get("id")
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"ACL file {path}: subjects[{i}].id must be a non-empty string")
        subject_ids.add(sid)
        # ``interfaces`` is OPTIONAL per Zenoh's AclConfigSubjects
        # schema (``Option<NEVec<...>>``). When omitted, Zenoh treats
        # the subject's interface dimension as ``SubjectProperty::Wildcard``
        # (matches every link); see authorization.rs:446-454. The
        # cleanest CN-only ACL pattern is therefore:
        #
        #  subjects: [{ id: "ops", cert_common_names: ["op-1", "op-2"] }]
        #
        # which is exactly what Zenoh's own ``tests/authentication.rs``
        # uses. We still REJECT an empty list outright -- Zenoh's parser
        # raises ``Found empty interface value`` and the silent total-
        # outage failure mode is real (prior footgun). And we still treat
        # an unknown-key like ``interface:`` (singular typo) as an error
        # because the rest of the validator catches it via
        # ``deny_unknown_fields`` semantics in the Rust deserializer.
        if "interfaces" in subj:
            ifaces = subj["interfaces"]
            if not isinstance(ifaces, list):
                raise ValueError(
                    f"ACL file {path}: subjects[{i}={sid!r}].interfaces must be a list "
                    f"(or omitted), got {type(ifaces).__name__}."
                )
            if not ifaces:
                raise ValueError(
                    f"ACL file {path}: subjects[{i}={sid!r}].interfaces is an empty list. "
                    f"Zenoh rejects ``[]`` with ``Found empty interface value``; either "
                    f"omit the field (for a wildcard binding) or enumerate the NICs."
                )
            if not all(isinstance(x, str) and x for x in ifaces):
                raise ValueError(
                    f"ACL file {path}: subjects[{i}={sid!r}].interfaces must contain only non-empty strings"
                )
        cns = subj.get("cert_common_names")
        if cns is not None and not isinstance(cns, list):
            raise ValueError(
                f"ACL file {path}: subjects[{i}={sid!r}].cert_common_names must be a list "
                f"(or omitted), got {type(cns).__name__}. Common typo: cert_common_name (singular)."
            )
        # Review thread _acl_config.py:279/293 -- HARD-REJECT subjects
        # that constrain neither ``interfaces`` nor
        # ``cert_common_names``. A subject with only an ``id`` (or
        # with both fields explicitly empty) maps to
        # ``SubjectProperty::Wildcard`` on every dimension -- "any
        # peer on any link" -- which combined with
        # ``default_permission: "deny"`` and an ``allow`` rule
        # produces a wire-effectively permissive ACL the operator did
        # not intend. Symmetric with the empty-list rejection above
        # (which also looks like "match nothing" but in inverted form
        # -- "match everything" is the dangerous footgun the parser
        # MUST refuse, not warn about).
        #
        # Operators who deliberately want a wildcard binding can
        # express it with ``interfaces: ["*"]`` (which Zenoh accepts
        # as the explicit any-link wildcard) plus an explicit CN
        # list, so this rejection does not block a legitimate
        # use case.
        ifaces_constrains = "interfaces" in subj and bool(subj.get("interfaces"))
        cns_constrains = isinstance(cns, list) and bool(cns)
        if not ifaces_constrains and not cns_constrains:
            raise ValueError(
                f"ACL file {path}: subjects[{i}={sid!r}] has neither "
                f"'interfaces' nor 'cert_common_names' (or both are empty) -- "
                f"it would match every peer on every link "
                f"(SubjectProperty::Wildcard on both dimensions). Add at least "
                f"one constraint to restrict scope; for an explicit "
                f'any-link wildcard use ``interfaces: ["*"]``.'
            )

    # 3. Rules.
    rule_ids: set[str] = set()
    for i, rule in enumerate(data["rules"]):
        if not isinstance(rule, dict):
            raise ValueError(f"ACL file {path}: rules[{i}] must be an object")
        rid = rule.get("id")
        if not isinstance(rid, str) or not rid:
            raise ValueError(f"ACL file {path}: rules[{i}].id must be a non-empty string")
        rule_ids.add(rid)
        for field in ("key_exprs", "messages", "flows"):
            val = rule.get(field)
            if not isinstance(val, list) or not val:
                raise ValueError(f"ACL file {path}: rules[{i}={rid!r}].{field} must be a non-empty list")
            if not all(isinstance(x, str) for x in val):
                raise ValueError(f"ACL file {path}: rules[{i}={rid!r}].{field} must contain only strings")
        perm = rule.get("permission")
        if perm not in ("allow", "deny"):
            raise ValueError(f"ACL file {path}: rules[{i}={rid!r}].permission must be 'allow' or 'deny', got {perm!r}")

    # 4. Policies.
    for i, pol in enumerate(data["policies"]):
        if not isinstance(pol, dict):
            raise ValueError(f"ACL file {path}: policies[{i}] must be an object")
        pol_rules = pol.get("rules")
        pol_subjects = pol.get("subjects")
        if not isinstance(pol_rules, list) or not pol_rules:
            raise ValueError(f"ACL file {path}: policies[{i}].rules must be a non-empty list of rule ids")
        if not isinstance(pol_subjects, list) or not pol_subjects:
            raise ValueError(f"ACL file {path}: policies[{i}].subjects must be a non-empty list of subject ids")
        for r in pol_rules:
            if r not in rule_ids:
                raise ValueError(
                    f"ACL file {path}: policies[{i}].rules references unknown rule id {r!r} (known: {sorted(rule_ids)})"
                )
        for sid_ref in pol_subjects:
            if sid_ref not in subject_ids:
                raise ValueError(
                    f"ACL file {path}: policies[{i}].subjects references unknown subject id "
                    f"{sid_ref!r} (known: {sorted(subject_ids)})"
                )


# --- Default ACL -------------------------------------------------------


def default_acl(namespace: str) -> dict[str, Any]:
    """Return a permissive default ACL skeleton.

    The default allows any peer with a valid CA-signed cert (verified
    at the mTLS handshake) to publish and subscribe on any key.
    Operators who want per-role enforcement supply their own ACL via
    ``STRANDS_MESH_ACL_FILE`` enumerating each peer's exact cert CN
    (Zenoh 1.x cert_common_names does not support globs -- see
    ``examples/mesh_acl_example.json5`` for the canonical template).

    Why permissive default rather than default-deny: a default-deny
    skeleton with no enumerated subjects rejects every legitimate
    message -- silent total outage on first run. The mTLS handshake at
    the link layer already gates fleet membership; the application-
    layer ``validate_command`` gates payload semantics. ACL is the
    third line of defence and operators opt in explicitly.
    """
    # ``namespace`` parameter is kept for API symmetry with the public
    # functions in this module (``acl_block``, ``resolve_acl``,
    # ``snapshot_acl``, ``is_default_acl_in_use``) -- they all take a
    # namespace string so callers can pass it positionally without
    # special-casing ``default_acl``. The built-in default ACL itself is
    # namespace-independent (Zenoh's namespace config does the routing
    # isolation; ACL key_exprs are RELATIVE to the active namespace and
    # do not need a namespace prefix). Review thread PR#224 _acl_config.py:343.
    _ = namespace  # noqa: F841 -- kept for API symmetry
    return {
        "enabled": True,
        # Permissive default: any peer that survived the mTLS handshake may
        # publish and subscribe on any key. This is the documented behaviour
        # (CHANGELOG section 8, README "Default ACL -- permissive by design").
        # Operators wanting per-role enforcement supply STRANDS_MESH_ACL_FILE
        # (see examples/mesh_acl_example.json5 for the canonical template).
        #
        # Earlier versions of this default mixed default_permission='deny'
        # with two key_exprs=['**'] allow-rules; the effective behaviour was
        # identical (allow-any) but the code-vs-doc surface was confusing
        # and review-flagged 5x. Code now matches docs.
        "default_permission": "allow",
        "rules": [],
        "subjects": [],
        "policies": [],
    }


# TOCTOU defence. In an earlier revision ``Mesh.start``
# called ``is_default_acl_in_use()`` (which now reads the file) and then
# ``resolve_acl()`` (which reads it again) -- a small TOCTOU window where
# an attacker who can rewrite the ACL file between the two reads sees
# the gate observe the SAFE shape and the wire load the UNSAFE shape.
# We close it with a single-load cache keyed on the file's identity
# tuple ``(path, dev, ino, size, mtime_ns)``. Both functions take the
# same snapshot; if the file changes mid-flight the next call refreshes,
# but a single ``Mesh.start`` call sees one snapshot.
_ACL_CACHE_LOCK = threading.Lock()
_ACL_CACHE: dict[tuple, dict[str, Any]] = {}


def _file_identity(path: Path) -> tuple | None:
    """Return ``(path_str, dev, ino, size, mtime_ns)`` or None on stat err."""
    try:
        st = os.stat(str(path), follow_symlinks=False)
    except OSError:
        return None
    return (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _load_acl_cached(path: Path) -> dict[str, Any]:
    """Load + cache an ACL file, keyed on its identity tuple.

    Two callers in the same ``Mesh.start`` flow (the gate check and the
    config builder) get the same dict object instead of two independent
    reads -- closing the prior TOCTOU surface. If the file changes a
    later call computes a fresh identity tuple and re-loads.
    """
    identity = _file_identity(path)
    if identity is None:
        # Stat failed -- fall through to the loader so it raises with
        # the canonical error path (FileNotFoundError, etc.).
        return _load_acl_file(path)
    with _ACL_CACHE_LOCK:
        cached = _ACL_CACHE.get(identity)
        if cached is not None:
            # Review thread _acl_config.py:429 -- return a deep copy so
            # caller mutation does not poison the cache for subsequent
            # callers. The cost is a small dict copy on every hit (ACL
            # files are tiny by ACL_FILE_MAX_BYTES = 256KiB).
            return copy.deepcopy(cached)
    loaded = _load_acl_file(path)
    with _ACL_CACHE_LOCK:
        # Cap the cache at 4 entries -- ACL files are tiny and the
        # operator usually has one. Bound prevents an attacker who can
        # touch the file repeatedly from inflating memory.
        if len(_ACL_CACHE) >= 4:
            _ACL_CACHE.pop(next(iter(_ACL_CACHE)))
        # Store a deep copy in the cache so caller mutation of the
        # returned dict does NOT poison the cached entry (review
        # _acl_config.py:429). Symmetric with the deep-copy on hit.
        _ACL_CACHE[identity] = copy.deepcopy(loaded)
    # Review thread _acl_config.py:458 -- return a deep copy on miss
    # too, so the FIRST caller for a given file identity sees the same
    # immutability contract subsequent callers get from the hit branch.
    # The previous code returned ``loaded`` directly, giving the first
    # caller a mutable handle on the parsed dict while later callers
    # got fresh deep copies -- an asymmetry that would silently bite
    # any future caller that mutates the result (e.g.
    # ``acl_block_from(resolved)`` is one accidental refactor away
    # from a ``json.dumps(resolved)`` that mutates).
    return copy.deepcopy(loaded)


def _clear_acl_cache_for_test() -> None:
    """Test-only escape hatch -- pytest fixtures that mutate ACL files
    in tmp_path between assertions need to invalidate the cache."""
    with _ACL_CACHE_LOCK:
        _ACL_CACHE.clear()


# Issue #218 / review session.py:296 -- thread-local single-flight ACL
# snapshot. ``Mesh.start`` calls ``snapshot_acl`` once at the gate, then
# ``session._build_config`` runs inside the same call stack and would
# call ``snapshot_acl`` AGAIN. The previous identity-tuple cache could
# miss if an attacker rewrote the file between the two calls (size /
# mtime_ns delta). The thread-local stashes the gate's resolved dict
# so ``_build_config`` reuses it verbatim, closing the TOCTOU window
# regardless of cache state.
_THREAD_SNAPSHOT: threading.local = threading.local()


def _set_thread_snapshot(
    resolved: dict[str, Any] | None,
    *,
    auth_mode: str | None = None,
) -> None:
    """Stash a resolved ACL dict (and optional ``auth_mode``) on the
    current thread for downstream reuse.

    Review thread core.py:139 -- the ``auth_mode`` env var
    (``STRANDS_MESH_AUTH_MODE``) is read independently by
    ``Mesh._refuse_under_permissive_default_acl`` and
    ``session._build_config``. A flip between the two reads (concurrent
    test fixture, plugin mutating ``os.environ``) yields inconsistent
    state. Stashing both signals together preserves the
    "one snapshot per ``Mesh.start``" invariant the docstring
    promises.
    """
    _THREAD_SNAPSHOT.value = (resolved, auth_mode) if auth_mode is not None else resolved


def _get_thread_snapshot() -> dict[str, Any] | None:
    """Return the current thread's stashed ACL snapshot, if any.

    Returns the resolved dict only -- callers that need ``auth_mode``
    use :func:`_get_thread_auth_mode`.
    """
    val = getattr(_THREAD_SNAPSHOT, "value", None)
    if isinstance(val, tuple):
        first = val[0]
        if first is None or isinstance(first, dict):
            return first
        return None
    if val is None or isinstance(val, dict):
        return val
    return None


def _get_thread_auth_mode() -> str | None:
    """Return the thread-local ``auth_mode`` stashed alongside the ACL,
    or ``None`` if the snapshot was set without one (legacy callers)."""
    val = getattr(_THREAD_SNAPSHOT, "value", None)
    if isinstance(val, tuple) and len(val) == 2:
        second = val[1]
        return second if (second is None or isinstance(second, str)) else None
    return None


def _clear_thread_snapshot() -> None:
    """Clear the current thread's snapshot (called after Mesh.start)."""
    if hasattr(_THREAD_SNAPSHOT, "value"):
        _THREAD_SNAPSHOT.value = None


def _is_permissive_acl_shape(data: dict[str, Any]) -> bool:
    """inspect the resolved ACL *shape* for
    the permissive pattern, regardless of where the dict came from
    (built-in default or operator-supplied file).

    Two patterns are flagged as permissive-by-shape:

    1. ``default_permission == "allow"`` AND every explicit
       rule/subject/policy collection empty. The original built-in
       ``default_acl()`` shape -- "any CA-signed peer can publish/
       subscribe everywhere".

    2. ``default_permission == "deny"`` BUT the operator opens
       everything back up via a wildcard rule + wildcard subject:
       a rule whose ``key_exprs`` contains ``"**"`` AND
       ``permission == "allow"``, AND there exists a subject lacking
       BOTH ``interfaces`` AND ``cert_common_names`` (i.e.
       ``SubjectProperty::Wildcard`` on every dimension), AND that
       subject is referenced by the wildcard rule's policy.

       This is the gap flagged in review at _acl_config.py:456 --
       ``default_permission: "deny"`` plus a single ``key_exprs:
       ["**"]/permission: "allow"`` rule plus a wildcard subject
       ("any CA-signed peer publishes/subscribes everywhere") was
       wire-effectively permissive but bypassed the previous narrow
       check.

    Returns True when EITHER pattern matches, so the
    ``Mesh.start`` refuse-to-start gate triggers on the
    wire-effective posture, not on the env-var presence or on the
    superficial ``default_permission`` literal.
    """
    if not isinstance(data, dict):
        return False
    default_perm = data.get("default_permission")
    rules = data.get("rules") or []
    subjects = data.get("subjects") or []
    policies = data.get("policies") or []

    # Pattern 1: built-in default shape (allow + empty everything else)
    if default_perm == "allow" and not rules and not subjects and not policies:
        return True

    # Pattern 2: deny + wildcard-rule + wildcard-subject + cross-policy
    if default_perm != "deny":
        return False
    if not isinstance(rules, list) or not isinstance(subjects, list):
        return False
    if not isinstance(policies, list):
        return False

    def _is_wildcard_rule(r: Any) -> bool:
        if not isinstance(r, dict):
            return False
        if r.get("permission") != "allow":
            return False
        kes = r.get("key_exprs") or []
        return isinstance(kes, list) and "**" in kes

    def _is_wildcard_subject(s: Any) -> bool:
        if not isinstance(s, dict):
            return False
        # Wildcard on both dimensions: no interfaces AND no
        # cert_common_names. Either field absent OR explicitly empty
        # (the validator already rejects empty interfaces lists, but
        # belt-and-braces here for hand-rolled dicts).
        ifaces = s.get("interfaces")
        cns = s.get("cert_common_names")
        return not ifaces and not cns

    wildcard_rule_ids = {r.get("id") for r in rules if _is_wildcard_rule(r)}
    if not wildcard_rule_ids:
        return False
    wildcard_subject_ids = {s.get("id") for s in subjects if _is_wildcard_subject(s)}
    if not wildcard_subject_ids:
        return False

    # Pattern 2 matches iff any policy ties a wildcard rule to a wildcard subject.
    for pol in policies:
        if not isinstance(pol, dict):
            continue
        pol_rules = set(pol.get("rules") or [])
        pol_subjects = set(pol.get("subjects") or [])
        if pol_rules & wildcard_rule_ids and pol_subjects & wildcard_subject_ids:
            return True
    return False


def is_default_acl_in_use(namespace: str = "strands") -> bool:
    """Return True when the wire-effective ACL is permissive-by-shape.

    A True return means *the resolved ACL* (whether built-in default
    or operator-supplied) grants any CA-signed peer publish/subscribe
    on any key. The check is shape-based (see
    :func:`_is_permissive_acl_shape`) so an operator file with the
    same permissive pattern as :func:`default_acl` triggers the same
    refuse-to-start gate at :class:`Mesh` start.

    Callers (Mesh.start) emit ERROR + refuse-to-start when this is
    combined with ``STRANDS_MESH_AUTH_MODE=mtls`` and the operator has
    not opted in via ``STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1``.

    Earlier revisions returned ``not env_var_set``
    -- an operator who supplied a permissive file silenced the gate
    while running with the same posture the gate was supposed to
    refuse. Now we resolve the file (or fall back to default) and
    inspect its shape.

    Failure mode: if the operator-supplied file fails to load
    (parse error, bad shape, IO error), the gate fails CLOSED -- we
    return ``True`` so :class:`Mesh.start` refuses to bring up the
    wire. A broken ACL file is a configuration emergency, not a
    "fall back to permissive" situation.
    """
    path_env = os.getenv("STRANDS_MESH_ACL_FILE", "").strip()
    if not path_env:
        # Built-in default: known to be permissive-by-shape.
        return True
    try:
        resolved = _load_acl_cached(Path(path_env))
    except (OSError, ValueError) as exc:
        # fail closed. A broken ACL file is treated as the
        # most-dangerous-known posture so the operator hears about it
        # at start-up rather than silently degrading to permissive.
        logger.warning(
            "[mesh] ACL file %s could not be loaded for shape check (%s); "
            "treating as permissive-by-default for the start-time gate",
            path_env,
            exc,
        )
        return True
    return _is_permissive_acl_shape(resolved)


def resolve_acl(namespace: str) -> dict[str, Any]:
    """Return the ACL dict for the current configuration.

    Resolution order: ``STRANDS_MESH_ACL_FILE`` -> :func:`default_acl`.
    """
    path_env = os.getenv("STRANDS_MESH_ACL_FILE", "").strip()
    if path_env:
        # shared cache so the prior shape gate and the wire
        # config builder see the SAME snapshot of the ACL file.
        return _load_acl_cached(Path(path_env))
    return default_acl(namespace)


def snapshot_acl(namespace: str = "strands") -> tuple[bool, dict[str, Any]]:
    """Atomically resolve the ACL and report its permissive-by-shape state.

    Issue #218 + review thread session.py:296: closes the TOCTOU window
    between the ``Mesh.start`` refuse-to-start gate and the
    ``session._build_config`` wire-config builder.

    Two-tier single-flight:

    1. **Thread-local hit**: when ``Mesh.start`` has already taken the
       snapshot and stashed it via :func:`_set_thread_snapshot`, we
       return THAT exact dict without touching the filesystem -- so an
       attacker rewriting the file between gate and build cannot create
       a cache miss. The thread-local is cleared at the end of
       ``Mesh.start`` so subsequent calls re-resolve.

    2. **Identity-tuple cache hit**: the legacy
       ``(path, dev, ino, size, mtime_ns)``-keyed cache. Same dict for
       same file identity, but a rewrite between two ``snapshot_acl``
       calls in *separate* threads (or after the thread-local clear)
       still yields a fresh load. That is the by-design refresh window
       -- the TOCTOU defence is local to a single ``Mesh.start`` flow.

    Returns:
        (is_permissive_by_shape, resolved_acl_dict)
    """
    # Tier 1: thread-local single-flight (closes TOCTOU within a single
    # Mesh.start flow even across separate snapshot_acl() callers).
    cached = _get_thread_snapshot()
    if cached is not None:
        return _is_permissive_acl_shape(cached), cached

    path_env = os.getenv("STRANDS_MESH_ACL_FILE", "").strip()
    if not path_env:
        # Built-in default: known permissive-by-shape.
        return True, default_acl(namespace)
    try:
        resolved = _load_acl_cached(Path(path_env))
    except (OSError, ValueError) as exc:
        # fail closed: unloadable file is treated as permissive so the
        # gate at Mesh.start refuses to bring up the wire.
        logger.warning(
            "[mesh] ACL file %s could not be loaded for snapshot (%s); "
            "treating as permissive-by-default for the start-time gate",
            path_env,
            exc,
        )
        return True, default_acl(namespace)
    return _is_permissive_acl_shape(resolved), resolved


def acl_block_from(resolved: dict[str, Any]) -> tuple[str, str]:
    """Return ``("access_control", <json5>)`` from a pre-resolved dict.

    Companion to :func:`snapshot_acl` -- pass the dict returned by the
    snapshot to bypass a second file read. Use this in Mesh.start so
    the refuse-to-start gate and the wire config builder share exactly
    one snapshot of the ACL file.
    """
    return ("access_control", json.dumps(resolved))


def acl_block(namespace: str) -> tuple[str, str]:
    """Return ``("access_control", <json5>)`` for the current config.

    .. note::
        Prefer :func:`snapshot_acl` + :func:`acl_block_from` in new code
        to avoid the TOCTOU window between gate-shape-check and
        wire-config-build (issue #218).
    """
    return ("access_control", json.dumps(resolve_acl(namespace)))
