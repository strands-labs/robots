"""Append-only audit log for safety-critical mesh events.

Safety actions on a multi-robot mesh (most importantly :func:`emergency_stop`)
need a tamper-evident trail that lives independently of stdout, structured
loggers, or any process that may crash mid-event.  This module owns that
trail.

Layout
------
By default the log lives at ``~/.strands_robots/mesh_audit.jsonl`` with
file mode ``0o600`` (owner read/write only) and the parent directory at
``0o700``.  The location can be overridden with the
``STRANDS_MESH_AUDIT_DIR`` environment variable; the JSONL file is always
named ``mesh_audit.jsonl``.

Format
------
Each line is one JSON object with these keys:

* ``ts`` -- UNIX timestamp (float seconds, UTC)
* ``event`` -- short event type, e.g. ``"emergency_stop"``
* ``peer_id`` -- the mesh peer that owned the event
* ``payload`` -- free-form dict with event-specific fields
* ``seq`` -- process-monotonic sequence number. Useful for detecting
  truncation: gaps within a single peer's stream indicate missing events.
* ``sig`` -- HMAC-SHA256 hex over the rest of the record. Present only when
  ``STRANDS_MESH_AUDIT_PSK`` is configured. Verifies that the record
  content has not been edited after write.

Integrity verification
----------------------
The audit log is the forensic trail for emergency stops, command
rejections, and resume attempts. To frustrate post-incident tampering by a
compromised process the writer attaches a per-record HMAC signature when
``STRANDS_MESH_AUDIT_PSK`` is set. :func:`verify_audit_integrity` walks the
log and reports:

* records with broken signatures (content was edited or partially
  truncated mid-line),
* sequence gaps (records were deleted),
* records lacking a signature (mixed-mode log -- only arises when
  separate processes write to the same audit directory at different
  times; e.g. a pre-PSK process and a post-PSK process across a rollout
  restart. ``_sign_record`` hard-rejects any signed<->unsigned
  transition WITHIN a single process by raising
  ``AuditPSKDegradedError`` which the writer converts into a
  ``sig="PSK_DEGRADED"`` poison record, so a single process cannot
  silently flip signing modes mid-run).

The PSK lives in env / Secrets Manager, never in the file. A reader that
does not have the PSK can still read events; it just cannot verify them.

The file is opened in append mode for every write so concurrent writers
from multiple threads or processes never overwrite each other; ordering
across processes is best-effort.

Reading
-------
:func:`read_audit_log` parses the file line by line and returns a list of
event dicts.  Lines that fail to parse are silently skipped (defensive: the
audit log is forward-compatible with future fields).
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# fcntl is POSIX-only. Windows deployments fall back to in-process
# locking and lose the cross-process safety guarantee documented in
# the module docstring.
try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

logger = logging.getLogger(__name__)

_LOG_FILE_NAME = "mesh_audit.jsonl"
_DEFAULT_DIR = Path.home() / ".strands_robots"

# Audit-log rotation (Phase-4 / Cycle 6 / E1).
#
# Without a size cap, an attacker who can publish to a peer's
# safety/event topic (or, in permissive mode, anyone on the LAN) can
# fill the robot's disk by spamming events at line-rate. We cap the
# active log at ``_DEFAULT_LOG_MAX_BYTES`` and rotate to a numbered
# suffix on overflow. ``_DEFAULT_LOG_MAX_FILES`` rotated copies are
# kept; the oldest is discarded. Operators tune via
# ``STRANDS_MESH_AUDIT_MAX_BYTES`` and ``STRANDS_MESH_AUDIT_MAX_FILES``.
#
# We deliberately don't use logging.handlers.RotatingFileHandler -- that
# class doesn't honour O_NOFOLLOW and would re-open the file via the
# default open() on rollover, defeating the prior symlink guard. The
# rotation here uses os.rename + os.open(O_NOFOLLOW) consistently.
_DEFAULT_LOG_MAX_BYTES: int = 100 * 1024 * 1024  # 100 MiB
_DEFAULT_LOG_MAX_FILES: int = 5
_LOG_MAX_BYTES_CAP: int = 10 * 1024 * 1024 * 1024  # 10 GiB hard upper bound
_LOG_MAX_FILES_CAP: int = 100

# Serialise writes inside a single process so two threads can't interleave
# bytes inside one append. Different processes still need filesystem-level
# atomicity (one open(..., "a") write per event).
_WRITE_LOCK = threading.Lock()

# Per-peer monotonic sequence counters. Each peer_id has its own counter so
# the (peer_id, seq) pair is unique within one process AND consecutive
# values within a single peer's stream are guaranteed to be adjacent. This
# makes :func:`verify_audit_integrity` gap detection meaningful even in
# processes that host multiple Mesh peers (test harnesses, ``Simulation``).
#
# Counters persist to a sidecar file (``mesh_audit.seq.json`` next to the
# audit log) so a process restart does NOT reset them. Without the
# sidecar, a compromised process could delete records, restart, and
# yield a clean ``verify_audit_integrity()`` because every peer's seq
# would start over at 1 -- defeating the gap-detection half of the
# threat model. The sidecar is reloaded inside the cross-process
# lockfile (``mesh_audit.seq.lock``) on every ``_next_seq`` call
# and rewritten before the lock is released, so two processes sharing
# the same ``STRANDS_MESH_AUDIT_DIR`` cannot roll the counter back.
# Writes are fail-soft: a write failure logs at WARNING but does not
# break the safety code path.
#
# Cross-process guarantee: POSIX (``fcntl.flock``) only. Windows
# deployments fall back to in-process locking; running multiple
# writer processes against the same audit dir on Windows is not
# safe and not supported.
#
# audit-write amplification (accepted limitation): every event
# triggers a synchronous double write (audit line + sidecar
# tmp+os.replace+chmod+fsync). The Zenoh transport-layer
# ``downsampling`` cap (``STRANDS_MESH_CMD_RATE_HZ``, default 20 Hz
# per-key-expression on ``**/cmd`` and ``**/broadcast``) drops flood
# traffic before it reaches the audit path, so the worst-case write
# rate is bounded by that cap times the number of distinct key
# expressions a peer can publish on. If your deployment runs
# pathologically high audit volumes, batch the sidecar persistence by
# subclassing ``_persist_seq_counters`` to write at most once per N
# events with an atexit flush -- the on-disk counter can then lose at
# most that many seconds of seq state on a hard kill, which
# ``verify_audit_integrity`` already detects. The default is per-event
# fsync because durability beats throughput for the safety log.
_SEQ_LOCK = threading.Lock()
_SEQ_COUNTERS: dict[str, int] = {}

#: Upper bound on a per-peer seq value when seeding ``_SEQ_COUNTERS`` from
#: an external source (sidecar OR audit-log walk). Real per-peer audit
#: volumes are tens of millions per year; a seq value above this cap is
#: almost certainly the result of a forged record / corrupted sidecar /
#: deliberate poisoning attempt. Capping the seed prevents a single bad
#: input from silently denying the legitimate writer the next ~billion
#: seq values.
_MAX_SEED_SEQ: int = 100_000_000

# the PSK fingerprint snapshot at
# ``_AUDIT_STATE.psk_fingerprint`` is read+modified+compared on every
# call to :func:`_sign_record`. Without a dedicated lock, two threads
# racing on the very first record could both observe ``snapshot is None``
# and both perform the assignment -- benign on the first record, but
# the same read-modify-compare path is exercised on every subsequent
# write where one thread can land between ``_audit_psk()`` and the
# ``snapshot != current_fp`` comparison and observe a stale view that
# defeats the PSK-degrade defence . Hold this lock around
# the entire fingerprint check so the compare-and-set is atomic.
_PSK_STATE_LOCK = threading.Lock()


class _ProcessAuditState:
    """Container for module-level mutable flags.

    Same rationale as ``mesh/security.py::_ProcessSecurityState``: we
    keep the one-shot ``loaded`` flag on an instance attribute so static
    analysers see a normal attribute read+write rather than a
    ``global`` declaration on a module-level scalar (which CodeQL's
    "unused global variable" rule mis-classifies -- alert #222).

    ``psk_fingerprint`` snapshots a fingerprint of the
    ``STRANDS_MESH_AUDIT_PSK`` value seen on the first record this
    process writes. Subsequent records compare to this snapshot --
    if the PSK gets unset, set, or rotated to a different value
    mid-run, ``_sign_record`` raises :class:`AuditPSKDegradedError`
    and the record is rejected. This closes:
    * (writer cleared PSK to forge unsigned records);
    * (writer rotated PSK value mid-run -- a verifier holding
      either key would fail signature on the other segment with
      no record-internal signal of which PSK was active when).

    The fingerprint is the first 16 bytes of ``sha256(psk)`` -- the
    same length used to attribute traces to runs in observability
    backends. Storing the fingerprint never leaks the PSK itself.
    """

    __slots__ = ("seq_loaded", "audit_log_seeded", "psk_fingerprint")

    def __init__(self) -> None:
        self.seq_loaded: bool = False
        # ``audit_log_seeded`` is the once-per-process flag for the
        # audit-log fallback path inside ``_load_seq_counters``. The
        # sidecar path is cheap (one fstat + one JSON parse, O(peers))
        # and runs on every ``_next_seq`` call so peer-process
        # increments are merged inside the flock; the audit-log walk
        # is O(records) and runs only when the sidecar is unusable.
        # Without this flag a degraded sidecar made every safety event
        # walk the entire audit log, turning a 100 MiB rotation set
        # into seconds-per-event latency on the safety code path. The
        # flag is set once the audit-log walk has run AND ``seq_loaded``
        # has been observed True at least once; resetting ``seq_loaded``
        # in ``_next_seq`` does not clear ``audit_log_seeded`` so the
        # walk does not repeat.
        self.audit_log_seeded: bool = False
        # ``None`` (unset sentinel)  = not yet observed.
        # ``b""`` (empty bytes sentinel)  = first call observed NO PSK.
        # any other ``bytes``  = first call observed a PSK,
        #  fingerprint = sha256(psk)[:16].
        self.psk_fingerprint: bytes | None = None


_AUDIT_STATE = _ProcessAuditState()

__all__ = [
    "AuditPSKDegradedError",
    "audit_log_path",
    "log_safety_event",
    "read_audit_log",
    "verify_audit_integrity",
]


def _audit_psk() -> bytes | None:
    """Return the audit-log PSK as bytes, or None when not configured."""
    psk = os.getenv("STRANDS_MESH_AUDIT_PSK")
    if not psk:
        return None
    return psk.encode("utf-8")


def _resolve_log_max_bytes() -> int:
    """Read STRANDS_MESH_AUDIT_MAX_BYTES with a hard upper cap.

    Reject obviously-broken values (negative, zero, larger than 10 GiB)
    and fall back to the default. The hard cap exists so a typo or
    misguided "disable rotation" attempt cannot turn the audit log
    back into an unbounded growth surface.
    """
    raw = os.getenv("STRANDS_MESH_AUDIT_MAX_BYTES")
    if not raw:
        return _DEFAULT_LOG_MAX_BYTES
    try:
        v = int(raw)
    except ValueError:
        logger.warning("[audit] STRANDS_MESH_AUDIT_MAX_BYTES=%r invalid -- using default", raw)
        return _DEFAULT_LOG_MAX_BYTES
    if v <= 0:
        return _DEFAULT_LOG_MAX_BYTES
    if v > _LOG_MAX_BYTES_CAP:
        logger.warning(
            "[audit] STRANDS_MESH_AUDIT_MAX_BYTES=%d exceeds hard cap %d -- clamping",
            v,
            _LOG_MAX_BYTES_CAP,
        )
        return _LOG_MAX_BYTES_CAP
    return v


def _resolve_log_max_files() -> int:
    """Read STRANDS_MESH_AUDIT_MAX_FILES with a hard upper cap."""
    raw = os.getenv("STRANDS_MESH_AUDIT_MAX_FILES")
    if not raw:
        return _DEFAULT_LOG_MAX_FILES
    try:
        v = int(raw)
    except ValueError:
        return _DEFAULT_LOG_MAX_FILES
    if v < 1:
        return 1
    if v > _LOG_MAX_FILES_CAP:
        return _LOG_MAX_FILES_CAP
    return v


def _rotate_log_if_needed(path: Path, current_size: int) -> None:
    """Rotate the audit log when it exceeds the configured size cap.

    Caller MUST hold :data:`_WRITE_LOCK` so two threads don't both
    rotate. We rename ``mesh_audit.jsonl`` -> ``mesh_audit.jsonl.1``,
    cascading older rotations up the chain, and discarding any
    rotation past ``max_files``.

    Rotation keeps the audit history within bounded disk usage
    (default: 100 MiB x 5 files = 500 MiB). Older records are
    discarded -- operators who need long-term retention should ship
    rotated files to durable storage out-of-band.

    Defence: also reject rotation if ``path`` is a symlink (paranoid
    repeat of the prior check; an attacker who races us between the
    write check and rotation could otherwise redirect the rotated
    name).
    """
    max_bytes = _resolve_log_max_bytes()
    if current_size < max_bytes:
        return
    if path.is_symlink():
        logger.warning("[audit] refusing to rotate symlinked audit log %s", path)
        return

    max_files = _resolve_log_max_files()
    # cascade.{n} ->.{n+1} for n in [max_files, max_files-1,..., 1].
    # The previous range(max_files - 1, 0, -1) walked [max_files-1.. 1] and
    # the predicate `n + 1 > max_files - 1` discarded n=max_files-1 instead
    # of rolling it to.{max_files}, so rotated suffixes only ever reached
    # .{max_files-1}. Walk from max_files now: file at.{max_files} (if
    # any leftover from a misconfig) is unlinked, then.{max_files-1}
    # through.1 cascade up by one.
    for n in range(max_files, 0, -1):
        src_p = path.with_suffix(path.suffix + f".{n}")
        dst_p = path.with_suffix(path.suffix + f".{n + 1}")
        if src_p.exists():
            try:
                if n + 1 > max_files:
                    # Discard files past the cap. Use os.unlink so a
                    # symlink at this position cannot redirect a delete.
                    if src_p.is_symlink():
                        logger.warning("[audit] discarding symlinked rotated log %s", src_p)
                        src_p.unlink(missing_ok=True)
                        continue
                    src_p.unlink(missing_ok=True)
                else:
                    os.replace(src_p, dst_p)
            except OSError as exc:
                logger.warning("[audit] rotation cascade failed at %s: %s", src_p, exc)
    # Finally, rename the active log to.1 and let the next write
    # create a fresh empty file via O_CREAT.
    try:
        os.replace(path, path.with_suffix(path.suffix + ".1"))
        logger.info("[audit] rotated %s (size=%d bytes)", path, current_size)
    except OSError as exc:
        logger.warning("[audit] could not rotate %s: %s", path, exc)


def _seq_sidecar_path() -> Path:
    """Return the location of the sequence-counter sidecar file."""
    return audit_log_path().parent / "mesh_audit.seq.json"


def _seq_lockfile_path() -> Path:
    """Path to the cross-process lockfile guarding the seq sidecar.

    two processes that host the same peer_id (multi-Mesh test
    harness, supervised restart racing the parent, fleet duplicate)
    could otherwise both load the sidecar at seq=N, increment in
    memory to N+1 and N+2 independently, persist whichever arrives
    last, and roll the counter back. We use a separate lockfile
    rather than ``flock``-ing the sidecar itself so the rename in
    ``_persist_seq_counters`` (which atomically replaces the inode)
    cannot strand the lock.
    """
    return audit_log_path().parent / "mesh_audit.seq.lock"


@contextlib.contextmanager
def _seq_flock() -> Iterator[None]:
    """Hold an exclusive flock on the seq lockfile for the block.

    Caller MUST already hold :data:`_SEQ_LOCK` (intra-process). Lock
    ordering: intra-process first, inter-process second. The lock is
    released on context exit even if the caller raises.

    On Windows ``fcntl`` is unavailable; we fall back to in-process
    locking only and document the cross-process limitation in the
    module docstring. POSIX deployments (the supported surface) get
    the full guarantee.
    """
    if not _HAS_FCNTL:
        yield
        return
    lockfile = _seq_lockfile_path()
    try:
        lockfile.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.debug("[audit] cannot create seq lockfile dir: %s", exc)
        yield
        return
    # seq lockfile open lacked ``O_NOFOLLOW`` while the audit log
    # itself (``_ensure_paths``), the sidecar (``_load_seq_counters`` /
    # ``_persist_seq_counters``), and the ACL loader all set it. An
    # attacker with write access to ``STRANDS_MESH_AUDIT_DIR`` who
    # pre-creates ``mesh_audit.seq.lock`` as a symlink to e.g.
    # ``/dev/null`` or a co-tenant file would otherwise have ``flock``
    # land on the link target rather than fail closed, breaking the
    # cross-process lock the docstring above promises. Restore the
    # symmetric defence here.
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(lockfile), os.O_RDWR | os.O_CREAT | nofollow, 0o600)
    except OSError as exc:
        # Issue #238: hard-fail on ELOOP rather than silently yielding
        # without a lock. The previous behaviour silently downgraded the
        # cross-process flock to no-lock when an attacker pre-created
        # ``mesh_audit.seq.lock`` as a symlink, defeating the
        # per-peer-monotonic-seq guarantee the docstring promises.
        # Symmetric with ``_ensure_paths`` which raises on the audit
        # log being a symlink.
        import errno

        if getattr(exc, "errno", None) == errno.ELOOP:
            # Hard-fail: raise into _next_seq's caller so the safety
            # event records a SEQ_LOCK_DEGRADED poison entry (mirrors
            # the PSK_DEGRADED discipline). The caller's try/except in
            # log_safety_event downgrades to a poison record on this
            # exception type, preserving forensic visibility.
            raise SeqLockSymlinkError(
                f"audit seq lockfile {lockfile} is a symlink (O_NOFOLLOW rejected); "
                "refusing to silently downgrade cross-process serialisation"
            ) from exc
        # Non-ELOOP errors (e.g. EACCES, ENOSPC) still degrade to
        # yield-without-lock with a DEBUG log -- those are operational
        # failures, not active attacker symlink swaps. Preserves the
        # existing best-effort posture for non-attack failure modes.
        logger.debug("[audit] cannot open seq lockfile: %s", exc)
        yield
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            # Best-effort unlock; on close the kernel releases anyway.
            pass
        try:
            os.close(fd)
        except OSError:
            # Already closed; no leak.
            pass


def _load_seq_counters() -> None:
    """Restore ``_SEQ_COUNTERS`` from the sidecar file. Idempotent.

    Caller MUST hold :data:`_SEQ_LOCK`. Stores the one-shot "loaded"
    flag on :data:`_AUDIT_STATE` so static analysers don't trip on a
    bare ``global`` for a module-level scalar.

    opens the sidecar with ``O_NOFOLLOW`` and refuses ``is_symlink``
    paths -- mirrors :func:`_persist_seq_counters` so an attacker who
    swaps the sidecar with a symlink between two process invocations
    cannot redirect the counter restore. Without this guard, the inter-
    process flock at :func:`_seq_flock` would still serialise writers
    but the reader would happily follow a symlink to attacker-chosen
    state (e.g. a sidecar from a different audit dir, or ``/dev/null``
    returning zero counters and rolling the cursor back). Asymmetric
    defence flagged earlier.

    when sidecar load fails OR is rejected as symlink, seed
    from the audit log by walking all records and taking max(seq) per
    peer_id. This prevents an attacker writing garbage to the sidecar
    from resetting all sequence counters to 0 on next boot.
    """
    if _AUDIT_STATE.seq_loaded:
        return
    sidecar = _seq_sidecar_path()
    sidecar_loaded = False
    try:
        if sidecar.is_symlink():
            logger.warning(
                "[audit] refusing to load seq sidecar at %s: it is a SYMLINK "
                "(target: %r). Counter restore will fail-soft.",
                sidecar,
                os.readlink(sidecar),
            )
        elif sidecar.exists():
            # O_NOFOLLOW on POSIX defeats a symlink-swap between
            # the is_symlink() check above and the open() below. On
            # Windows ``O_NOFOLLOW`` is 0 and the static check is the
            # only defence; this matches the audit-log open at L730+.
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(str(sidecar), os.O_RDONLY | nofollow)
            with os.fdopen(fd, encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                # Track whether any entry was actually merged; an empty
                # dict (or a dict with no valid entries) must NOT flip
                # sidecar_loaded=True, otherwise the audit-log fallback
                # is skipped and an attacker writing ``{}`` gets every
                # peer's seq reset.
                merged_any = False
                for key, value in payload.items():
                    if not (isinstance(key, str) and isinstance(value, int) and value >= 0):
                        continue
                    # Cap the seed even on the healthy-sidecar path. An
                    # attacker with audit-dir write access can drop a
                    # syntactically valid sidecar like
                    # ``{"victim_peer_id": 999999999}``; without the
                    # cap, the legitimate writer's next event would jump
                    # the seq counter by ~10^9 with no upper bound. The
                    # the prior HMAC-verify defence applies only to the
                    # audit-log fallback; the sidecar itself is not
                    # signed, so the cap is the only defence on this
                    # path.
                    if value > _MAX_SEED_SEQ:
                        logger.warning(
                            "[audit] refusing to seed seq counter for %r "
                            "from sidecar value %d (cap=%d, possibly "
                            "tampered sidecar)",
                            key,
                            value,
                            _MAX_SEED_SEQ,
                        )
                        continue
                    # Only restore if our in-memory value is lower --
                    # never roll a counter backwards even if the file
                    # somehow has a stale value.
                    if value > _SEQ_COUNTERS.get(key, 0):
                        _SEQ_COUNTERS[key] = value
                        merged_any = True
                # Only mark the sidecar as loaded when at least one
                # valid entry was merged. An empty dict (``{}``) parses
                # as JSON but seeds zero counters; treating it as a
                # successful load would skip the audit-log fallback
                # below and silently let the attacker reset every
                # peer's seq counter to 0 just by writing ``{}`` into
                # the sidecar. The sidecar guard is now all-or-nothing
                # AND fall-through-on-empty: either we merged real
                # entries or we fall through to the integrity-checked
                # audit-log seed.
                sidecar_loaded = merged_any
            else:
                logger.warning(
                    "[audit] sidecar %s parsed as non-dict (%s) -- falling through to audit-log seed",
                    sidecar,
                    type(payload).__name__,
                )
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[audit] could not load seq sidecar %s: %s", sidecar, exc)

    # If sidecar failed to load (corrupt/symlink/missing), seed
    # from the audit log to prevent fail-open sequence reset.
    #
    # when STRANDS_MESH_AUDIT_PSK is configured,
    # ONLY seed from records whose HMAC ``sig`` we can verify. Without
    # this, the previous version trusted every record in the log
    # (signed or not) -- an attacker who could write to the audit log
    # path (no PSK in dev posture, or PSK exfiltrated long enough to
    # forge one record then cleared) could append a forged record
    # with seq=10**9 and on the next process restart with a corrupt
    # sidecar that value would become the new floor for that peer,
    # silently denying the legitimate writer a working seq counter.
    # The fix: trust only HMAC-verified records when a PSK is present.
    # When no PSK is configured, everything is trusted (the dev
    # posture has no integrity gate and the threat model accepts
    # writers in the audit dir).
    #
    # ``audit_log_seeded`` gates the walk to once per
    # process. ``_next_seq`` resets ``seq_loaded`` on every call so the
    # cheap sidecar path runs inside the flock (peer-process increments
    # have to be merged), but the audit-log walk is O(records) and a
    # degraded sidecar would otherwise re-walk the entire rotation set
    # on every safety event -- seconds of CPU on the hot path. Once
    # the walk has run, the in-memory ``_SEQ_COUNTERS`` floor is
    # established; subsequent calls only need the cheap sidecar merge,
    # and the walk does not repeat unless a fresh process restart
    # clears the flag.
    if not sidecar_loaded and not _AUDIT_STATE.audit_log_seeded:
        try:
            records = read_audit_log()
            psk = _audit_psk()
            verified = 0
            unverified_skipped = 0
            for record in records:
                peer_id = record.get("peer_id")
                seq = record.get("seq")
                if not (isinstance(peer_id, str) and isinstance(seq, int) and seq > 0):
                    continue
                # when a PSK is configured, only trust signed records
                # whose HMAC we can verify. This breaks the circular trust
                # surface where an attacker who can write the audit log
                # could poison the seq counter restore.
                if psk is not None:
                    sig = record.get("sig")
                    if not isinstance(sig, str) or sig in ("PSK_DEGRADED", "SIGN_FAILED"):
                        unverified_skipped += 1
                        continue
                    expected = hmac.new(psk, _canonical_bytes(record), hashlib.sha256).hexdigest()
                    if not hmac.compare_digest(sig, expected):
                        unverified_skipped += 1
                        continue
                # Cap the seed even without a PSK so a single forged log
                # line cannot poison the counter to 10**9 and silently
                # deny the legitimate writer the next ~billion seq
                # values. The cap (:data:`_MAX_SEED_SEQ`) is shared
                # with the sidecar path so both seed sources have the
                # same fail-loud-on-tamper posture.
                if seq > _MAX_SEED_SEQ:
                    logger.warning(
                        "[audit] refusing to seed seq counter for %r from "
                        "audit-log seq=%d (cap=%d, possibly forged record)",
                        peer_id,
                        seq,
                        _MAX_SEED_SEQ,
                    )
                    unverified_skipped += 1
                    continue
                if seq > _SEQ_COUNTERS.get(peer_id, 0):
                    _SEQ_COUNTERS[peer_id] = seq
                    verified += 1
            if _SEQ_COUNTERS:
                logger.info(
                    "[audit] seeded %d peer counters from audit log after sidecar load failed (verified=%d, skipped_unverified=%d)",
                    len(_SEQ_COUNTERS),
                    verified,
                    unverified_skipped,
                )
            elif unverified_skipped:
                logger.warning(
                    "[audit] sidecar load failed and %d audit records were unverified -- counters NOT seeded "
                    "(this is the circular-trust defence; an attacker who wrote unsigned forgeries "
                    "to the audit log cannot poison the seq restore when STRANDS_MESH_AUDIT_PSK is set)",
                    unverified_skipped,
                )
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as log_exc:
            # Narrow per AGENTS.md > "Exception Clauses Must Be Narrow":
            # OSError covers disk failures, JSONDecodeError covers
            # malformed records, ValueError covers _validate_acl_shape-
            # style schema violations, TypeError covers record-shape
            # mismatches. An unexpected exception type is a programmer
            # bug we want to see in tests, not silently degrade the
            # the prior counter-reset defence.
            logger.warning(
                "[audit] could not seed from audit log after sidecar failure: %s",
                log_exc,
            )
        # Mark the audit-log walk as done regardless of whether it
        # found anything. A second call into ``_load_seq_counters``
        # while the sidecar is still degraded should not re-walk the
        # log -- the in-memory floor we built is the best we have, and
        # walking again only burns CPU on the safety code path.
        _AUDIT_STATE.audit_log_seeded = True

    _AUDIT_STATE.seq_loaded = True


def _persist_seq_counters() -> None:
    """Write ``_SEQ_COUNTERS`` to the sidecar file. Fail-soft.

    Caller MUST hold :data:`_SEQ_LOCK`.

    Defence (Phase-4 / Cycle 4): if the sidecar is a symlink,
    refuse to write. Same threat model as the audit log itself --
    attacker swaps the file with a symlink to redirect counter state
    or null-route it. The atomic ``tmp + os.replace`` already prevents
    half-written sidecars; this adds protection against tamper.
    """
    sidecar = _seq_sidecar_path()
    if sidecar.is_symlink():
        logger.warning(
            "[audit] refusing to persist seq sidecar at %s: it is a SYMLINK "
            "(target: %r). Counter persistence will fail-soft.",
            sidecar,
            os.readlink(sidecar),
        )
        return
    try:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file then rename so a crash mid-write cannot
        # leave a half-formed sidecar that fails to parse on next load.
        tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
        # Open the tmp file with O_NOFOLLOW too, so a TOCTOU between
        # the is_symlink check above and this open is foiled.
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(tmp, flags | nofollow, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(_SEQ_COUNTERS, fh, sort_keys=True, separators=(",", ":"))
                fh.flush()
                # fsync the temp fd before rename so a power-loss
                # cannot leave the audit log ahead of the sidecar. After
                # restart, ``_load_seq_counters`` would otherwise pick up
                # stale counters and the next event would write a duplicate
                # seq value, defeating per-peer adjacency in
                # ``verify_audit_integrity``.
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    # Best-effort on filesystems that reject fsync; the
                    # data is in the kernel page cache and durability is
                    # weaker than ideal but the safety code path stays
                    # alive (audit persistence is fail-soft by contract).
                    pass
        except OSError:
            # Defence-in-depth against the rare path where ``os.fdopen``
            # itself raises *before* adopting the fd (e.g. invalid mode
            # string, EMFILE while constructing the buffered wrapper).
            # On the common path this branch is unreachable: ``with
            # os.fdopen(fd, "w", ...) as fh:`` transfers fd ownership
            # to the file object, and the context manager's ``__exit__``
            # closes fd on any exception inside the with-block. Calling
            # ``os.close(fd)`` here would then hit EBADF; we suppress
            # that with the inner except so the original exception
            # propagates unchanged via the explicit ``raise`` below.
            #
            # narrowed from ``except Exception`` because the only thing
            # this path ever needs to handle is an OS-level failure
            # acquiring the fd; fh.flush / fsync / chmod / replace all
            # raise OSError too, and a programmer bug above (TypeError,
            # AttributeError) should propagate without being caught.
            try:
                os.close(fd)
            except OSError:
                # Already closed by fdopen's context manager exit (or
                # never opened). The raise below propagates the original
                # error; this cleanup branch only matters on the rare
                # crash path where fdopen itself raised.
                pass
            raise
        os.replace(tmp, sidecar)
        # fsync the parent directory so the rename is durable
        # too. POSIX-only -- Windows treats os.fsync on a directory fd
        # as undefined behaviour. Skip the dir fsync there; the rename
        # is atomic on NTFS so the visible-state ordering still holds.
        if os.name == "posix":
            try:
                dir_fd = os.open(str(sidecar.parent), os.O_RDONLY)
            except OSError:
                # Best-effort; if the parent is unreadable the rename
                # still happened and we just lose dir-level durability.
                dir_fd = None
            if dir_fd is not None:
                try:
                    os.fsync(dir_fd)
                except OSError:
                    # Some filesystems reject directory fsync; the
                    # rename is still on disk in the page cache.
                    pass
                finally:
                    try:
                        os.close(dir_fd)
                    except OSError:
                        # Best-effort close of a read-only dir fd; if
                        # the dir was unmounted between open and close
                        # this can race but no leak occurs because the
                        # process is exiting.
                        pass
        try:
            os.chmod(sidecar, 0o600)
        except OSError:
            # chmod is best-effort: filesystems that don't honour POSIX
            # permissions (FAT32, NFS without uid map, mounted volumes
            # under restricted mount options) silently fail this call,
            # but the sidecar itself is still written and readable. We
            # would rather have a working audit log without 0o600 than
            # crash safety persistence over a chmod failure.
            pass
    except OSError as exc:
        logger.warning("[audit] could not persist seq sidecar %s: %s", sidecar, exc)


def _next_seq(peer_id: str) -> int:
    """Return the next monotonic sequence number for *peer_id*.

    the load+increment+persist sequence runs under TWO locks:

    * :data:`_SEQ_LOCK` (intra-process) so multiple Mesh instances in
      one process don't interleave increments.
    * an ``fcntl.flock`` on the sidecar lockfile (inter-process) so
      two processes that share the same audit dir cannot both load
      seq=N and persist different increments -- which would roll the
      counter back. Inside the flock we **re-read** the sidecar so
      our in-memory ``_SEQ_COUNTERS`` cache is reconciled with whatever
      a peer process has written since our last increment.

    Lock ordering: intra-process first, inter-process second. Always.
    """
    with _SEQ_LOCK:
        with _seq_flock():
            # Re-read the sidecar inside the flock so a peer process's
            # increments are merged into our in-memory cache before
            # we decide our next value.
            _AUDIT_STATE.seq_loaded = False
            _load_seq_counters()
            next_value = _SEQ_COUNTERS.get(peer_id, 0) + 1
            _SEQ_COUNTERS[peer_id] = next_value
            _persist_seq_counters()
            return next_value


def _canonical_bytes(record: dict[str, Any]) -> bytes:
    """Stable byte encoding for HMAC. Excludes the ``sig`` field."""
    return json.dumps(
        {k: v for k, v in record.items() if k != "sig"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class SeqLockSymlinkError(RuntimeError):
    """Raised when the audit seq lockfile is a symlink (issue #238).

    An attacker with write access to ``STRANDS_MESH_AUDIT_DIR`` could
    pre-create ``mesh_audit.seq.lock`` as a symlink to e.g. ``/dev/null``
    so that ``fcntl.flock`` lands on the link target rather than fail
    closed -- silently downgrading the cross-process serialisation the
    audit log's per-peer monotonic seq guarantee depends on. Hard-fail
    posture (matching ``_ensure_paths`` for the audit log itself) is
    safer than the silent yield-without-lock that this defends against.
    """


class AuditPSKDegradedError(RuntimeError):
    """Raised when STRANDS_MESH_AUDIT_PSK was set at first write but is
    no longer set at a subsequent write.

    Round-4 / a process that briefly clears its env to write a run
    of unsigned forgeries -- then re-sets the PSK -- would otherwise yield
    records that ``verify_audit_integrity`` reports as ``missing_sig``
    while ``ok`` (the boolean reader-helpers check) stays True. We snap
    PSK presence on the first signed record and refuse to write further
    records under a downgraded configuration.
    """


def _psk_fingerprint(psk: bytes | None) -> bytes:
    """Return ``b""`` if PSK is unset, else the first 16 bytes of
    sha256(psk). Used by :data:`_AUDIT_STATE` to detect mid-run PSK
    transitions (set, unset, OR rotation to a different value).

    The fingerprint is one-way -- storing it never leaks the PSK
    itself. 16 bytes (128 bits) is enough to make accidental
    fingerprint collisions vanishingly unlikely while keeping the
    snapshot small.
    """
    if psk is None:
        return b""
    return hashlib.sha256(psk).digest()[:16]


def _sign_record(record: dict[str, Any]) -> str | None:
    """Compute the per-record HMAC signature, or ``None`` when no PSK
    is configured.

    Round-4 / snapshot a fingerprint of the PSK observed on
    the first call. If a subsequent call sees a different fingerprint
    (PSK unset, set, or rotated to a different value), raise
    ``AuditPSKDegradedError`` so the audit log cannot silently
    degrade to unsigned, silently start signing on top of an
    unverifiable unsigned prefix, OR silently switch keys mid-run
    (which would make every record post-rotation unverifiable
    against the pre-rotation key and vice versa).

    The caller is the safety code path; we let the error propagate
    to ``log_safety_event`` which writes a poison record
    (``sig="PSK_DEGRADED"``) and logs at ERROR. Audit failures must
    not crash the safety path, but we DO refuse the unsigned /
    rotated write.
    """
    psk = _audit_psk()
    current_fp = _psk_fingerprint(psk)
    # Atomic compare-and-set AND comparison
    # under one lock. Earlier the lock only held the snapshot fetch +
    # first-time set; the elif compare ran outside the lock. The
    # docstring at lines 150-158 promised "the entire fingerprint
    # check" was atomic; that claim now actually holds. Concretely:
    # Without the swap-after-compare, two threads racing on a PSK rotation could each
    # read the same pre-rotation snapshot, then both pass the
    # ``snapshot == current_fp`` check (both seeing post-rotation
    # current_fp), and both write under the new key without one
    # raising AuditPSKDegradedError.
    with _PSK_STATE_LOCK:
        snapshot = _AUDIT_STATE.psk_fingerprint
        if snapshot is None:
            # First record this process -- snap the observed state and
            # treat the current call as matching itself (no transition).
            _AUDIT_STATE.psk_fingerprint = current_fp
            snapshot = current_fp

        if snapshot != current_fp:
            transition_detected = True
        else:
            transition_detected = False
    if transition_detected:
        # PSK transition: set->unset, unset->set, OR rotated value.
        # All three break verifiability symmetrically; refuse.
        if snapshot != b"" and current_fp == b"":
            reason = (
                "STRANDS_MESH_AUDIT_PSK was set when the audit log first "
                "started signing this run, but is now unset. Refusing to "
                "write an unsigned record (would silently degrade audit "
                "integrity). Restore the PSK or restart the process to "
                "transition to unsigned mode deliberately."
            )
        elif snapshot == b"" and current_fp != b"":
            reason = (
                "STRANDS_MESH_AUDIT_PSK was unset when the audit log first "
                "started this run, but is now set. Refusing to start signing "
                "mid-run (would create an unverifiable unsigned prefix that "
                "a forensic walker cannot distinguish from an attacker-forged "
                "forgery window). Restart the process to transition to "
                "signed mode deliberately."
            )
        else:
            # Both non-empty but different: rotated value.
            # the post-rotation segment would be unverifiable
            # against the pre-rotation key and vice versa, with NO
            # record-internal signal of which key was active for
            # which records. Restart to rotate deliberately.
            reason = (
                "STRANDS_MESH_AUDIT_PSK changed value mid-run "
                "(rotation detected via fingerprint). Refusing to "
                "sign records under the new key: a verifier holding "
                "either key would fail signature on the other "
                "segment with no way to attribute records to keys. "
                "Restart the process to rotate the PSK deliberately."
            )
        raise AuditPSKDegradedError(reason)
    if psk is None:
        return None
    return hmac.new(psk, _canonical_bytes(record), hashlib.sha256).hexdigest()


def audit_log_path() -> Path:
    """Return the resolved path of the audit log file.

    Honours ``STRANDS_MESH_AUDIT_DIR`` (override) or falls back to
    ``~/.strands_robots``.  Does not create the directory.
    """
    override = os.getenv("STRANDS_MESH_AUDIT_DIR")
    base = Path(override).expanduser() if override else _DEFAULT_DIR
    return base / _LOG_FILE_NAME


def _ensure_paths(path: Path) -> None:
    """Make sure the parent directory exists (mode 0o700) and the file
    exists with mode 0o600.

    Re-applies permissions on every call so a fresh deploy or a manual
    ``touch`` cannot leave the file world-readable by accident.

    Defence: if the audit log path is a SYMLINK (potentially pointing
    to attacker-controlled territory like ``/dev/null`` or another
    process's file), refuse to operate. The audit log must always be
    a real regular file at the canonical location. See
    review feedback round 4 (symlink-swap defence).
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(parent, 0o700)
    except OSError as exc:  # pragma: no cover - best-effort on exotic FS
        logger.debug("[audit] could not chmod %s: %s", parent, exc)

    # Symlink check on the audit log itself. ``Path.is_symlink`` returns
    # False on a missing file and does NOT raise on permission errors,
    # so the previous try/except OSError wrapper was dead code that
    # could silently swallow a TOCTOU race. rely on the static
    # check for the eager-fail path AND on O_NOFOLLOW for the file
    # creation below so a symlink swap between the two cannot redirect
    # the create.
    if path.is_symlink():
        raise OSError(
            f"refusing to use audit log at {path}: it is a SYMLINK "
            f"(target: {os.readlink(path)!r}). This may indicate "
            f"tampering. Set STRANDS_MESH_AUDIT_DIR if you need to "
            f"relocate the log."
        )

    if not path.exists():
        # create with O_NOFOLLOW so an attacker who races a
        # symlink in between the is_symlink check above and this open
        # cannot redirect the create. ``Path.touch`` follows symlinks
        # (the symlink-refusal and fail-soft contracts are pinned in
        # tests/mesh/test_audit_log_symlink_refused.py). On Windows where
        # O_NOFOLLOW is 0 the static check above is the only line of
        # defence; this matches the residual-risk note in the module
        # docstring.
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags | nofollow, 0o600)
        except FileExistsError:
            # Another writer created it concurrently; that's fine --
            # they raced ahead of us and the next is_symlink check
            # would catch a swap if one happened.
            pass
        else:
            os.close(fd)

    try:
        os.chmod(path, 0o600)
    except OSError as exc:  # pragma: no cover
        logger.debug("[audit] could not chmod %s: %s", path, exc)


def log_safety_event(event_type: str, peer_id: str, payload: dict[str, Any]) -> None:
    """Append a single safety event to the audit log.

    Args:
        event_type: Short, lowercase event identifier
            (e.g. ``"emergency_stop"``).
        peer_id: The mesh peer that originated the event.
        payload: Event-specific fields.  Must be JSON-serialisable.

    Raises:
        Nothing - write errors are logged at WARNING and swallowed because
        an audit-log failure must never propagate up into the safety code
        path that called this function.
    """
    seq_lock_degraded_reason: str | None = None
    next_seq_degraded_reason: str | None = None
    try:
        seq = _next_seq(peer_id)
    except SeqLockSymlinkError as exc:
        # PR#221 R3 (issue #238): the seq lockfile is a symlink. Raise
        # the visible signal: write a poison record with
        # ``sig="SEQ_LOCK_DEGRADED"`` so verify_audit_integrity walkers
        # see a gap on this peer's stream. seq is unknown -- use 0 as
        # the placeholder so the record still serialises; the poison
        # ``sig`` is the discriminator a verifier keys on.
        logger.error(
            "[audit] SEQ_LOCK_DEGRADED for peer_id=%r: %s -- writing poison record",
            peer_id,
            exc,
        )
        seq = 0
        seq_lock_degraded_reason = str(exc)
    except Exception as exc:  # noqa: BLE001 -- audit log failures MUST be soft per contract
        # #324: a non-symlink _next_seq failure (e.g. OSError on the seq
        # sidecar, a corrupt counter file, a permissions flip) previously
        # dropped the record silently -- leaving a gap on this peer's stream
        # that no verifier could attribute. Mirror the SEQ_LOCK_DEGRADED /
        # PSK_DEGRADED / SIGN_FAILED poison-record discipline: emit a
        # NEXT_SEQ_DEGRADED poison record with seq=0 so verify_audit_integrity
        # walkers see (and can classify) the seq-counter integrity gap instead
        # of a silent hole.
        logger.error(
            "[audit] NEXT_SEQ_DEGRADED for peer_id=%r: %s -- writing poison record",
            peer_id,
            exc,
        )
        seq = 0
        next_seq_degraded_reason = str(exc)
    record: dict[str, Any] = {
        "ts": time.time(),
        "event": event_type,
        "peer_id": peer_id,
        "payload": payload,
        "seq": seq,
    }
    if next_seq_degraded_reason is not None:
        # #324: non-symlink _next_seq failure -- poison the record so the gap
        # is attributable on this peer's stream (symmetry with SEQ_LOCK_DEGRADED).
        record["sig"] = "NEXT_SEQ_DEGRADED"
    elif seq_lock_degraded_reason is not None:
        # Poison-record discipline: signal SEQ_LOCK_DEGRADED so a
        # verifier flags the seq-counter integrity gap. Mirrors the
        # PSK_DEGRADED / SIGN_FAILED poison patterns below.
        record["sig"] = "SEQ_LOCK_DEGRADED"
        record["seq_lock_degraded"] = seq_lock_degraded_reason
    sig: str | None = None
    if seq_lock_degraded_reason is None and next_seq_degraded_reason is None:
        try:
            sig = _sign_record(record)
        except AuditPSKDegradedError as exc:
            # STRANDS_MESH_AUDIT_PSK transitioned mid-run
            # (signed->unsigned or unsigned->signed). We refuse to forge a
            # signature, but instead of silently dropping the record we
            # write a "poison" record with sig="PSK_DEGRADED" and a
            # ``psk_degraded`` reason field. A signed-record verifier
            # (verify_audit_integrity with PSK present) reports it as
            # ``missing_sig`` (sig is not a valid HMAC), which forces
            # ``ok=False`` and surfaces the transition to forensics.
            #
            logger.error("[audit] %s -- writing poison record (sig=PSK_DEGRADED): %s", exc, record)
            record["sig"] = "PSK_DEGRADED"
            record["psk_degraded"] = str(exc)
        except Exception as sign_exc:  # noqa: BLE001 -- audit must be soft per contract (lines 768-771)
            # widen the fail-soft contract beyond
            # AuditPSKDegradedError so the safety code path never crashes
            # on a sign-time error.
            #
            # if a PSK was configured at this
            # moment, write a poison record (``sig="SIGN_FAILED"``) so a
            # forensic walker holding the same PSK sees the gap as a bad
            # signature and forces ``ok=False``. Without this, an
            # unsigned record would be invisible to a verifier running
            # without the PSK (psk_present=False -> missing_sig branch
            # not exercised) and silently weaken the documented
            # PSK-degrade contract.
            logger.error(
                "[audit] _sign_record raised %s: %s",
                type(sign_exc).__name__,
                sign_exc,
            )
            if _audit_psk() is not None:
                # PSK is configured, but signing failed transiently. Write
                # a poison record so verify_audit_integrity flags the gap.
                record["sig"] = "SIGN_FAILED"
                record["sign_error"] = f"{type(sign_exc).__name__}: {sign_exc}"
            # else: no PSK configured -- the unsigned write is the
            # documented dev-mode posture.
        else:
            if sig is not None:
                record["sig"] = sig

    try:
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    except (TypeError, ValueError) as exc:
        logger.warning(
            "[audit] could not serialise record for peer_id=%r: %s -- record dropped",
            peer_id,
            exc,
        )
        return
    path = audit_log_path()

    with _WRITE_LOCK:
        try:
            _ensure_paths(path)
            # Phase-4 / E1: rotate BEFORE writing if the active log
            # has grown past the size cap. Rotation is bounded so an
            # attacker flooding events cannot exhaust disk; only the
            # last (max_files * max_bytes) of audit history is kept.
            try:
                cur_size = path.stat().st_size if path.exists() else 0
            except OSError:
                cur_size = 0
            if cur_size > 0:
                _rotate_log_if_needed(path, cur_size)
            # Open with O_NOFOLLOW (POSIX) to defeat a symlink-swap
            # race between _ensure_paths and this open(). On a
            # symlink target the open() raises ELOOP and we reject the
            # write -- matching the static check in _ensure_paths.
            #
            # Fall back to plain open() if O_NOFOLLOW is unavailable
            # (Windows). On those platforms the static is_symlink
            # check in _ensure_paths is the only defence; we accept
            # that as residual risk because the supported deployment
            # surface is POSIX (Linux + macOS).
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            # NOTE: O_NOFOLLOW raises ELOOP on a symlink, which is the
            # intended behaviour -- a symlinked audit log is the threat
            # documented at the top of this module. Do not retry without
            # O_NOFOLLOW; let the OSError propagate so the audit-log path
            # is rejected loudly rather than redirected.
            fd = os.open(path, flags | nofollow, 0o600)
            try:
                with os.fdopen(fd, "a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())  # durable write before returning
                    except OSError:
                        # best-effort on filesystems that reject fsync;
                        # the data is in the kernel page cache and
                        # we'd rather lose durability than crash safety.
                        pass
            except Exception:
                # Make sure fd is closed if fdopen raises.
                try:
                    os.close(fd)
                except OSError:
                    # Already closed by fdopen context-manager exit.
                    # Nothing to do; the original error propagates via
                    # the raise below.
                    pass
                raise
        except OSError as exc:
            logger.warning("[audit] failed to write %s: %s", path, exc)


def _audit_log_files_in_order() -> list[Path]:
    """Return the audit log file set in chronological order.

    Rotated files are named ``mesh_audit.jsonl.N`` where ``.1`` is the
    most recently rotated and higher numbers are older. To iterate in
    chronological order we read the highest-numbered rotation first,
    then descend, then the active log last. When rotation has not
    happened the list is just ``[active]``.

    Returns an empty list when no audit file exists at all.
    """
    active = audit_log_path()
    out: list[Path] = []
    parent = active.parent
    if not parent.is_dir():
        return [active] if active.exists() else []

    # Find every rotated copy (mesh_audit.jsonl.<N>).
    rotations: list[tuple[int, Path]] = []
    for entry in parent.iterdir():
        name = entry.name
        if not name.startswith(active.name + "."):
            continue
        suffix = name[len(active.name) + 1 :]
        if not suffix.isdigit():
            continue
        rotations.append((int(suffix), entry))

    # Rotated suffixes: higher number = older, so sort DESC and prepend
    # the active log at the end.
    rotations.sort(reverse=True)
    out.extend(p for _, p in rotations)
    if active.exists():
        out.append(active)
    return out


def read_audit_log(since: float | None = None) -> list[dict[str, Any]]:
    """Read the audit log and return parsed event records.

    Reads rotated copies (``mesh_audit.jsonl.N``) in chronological
    order before the active log, so verification spans the entire
    persisted history rather than just whatever is in the current
    file. Files older than the rotation cap have already been
    discarded by :func:`_rotate_log_if_needed`; what remains is the
    full retained window.

    Args:
        since: Optional UNIX timestamp.  When provided, only records
            with ``ts >= since`` are returned.

    Returns:
        List of event dicts in chronological order. Returns an empty
        list if no audit file exists.
    """
    # every other open in this module
    # carefully refuses symlinks via static is_symlink() + O_NOFOLLOW
    # (see _ensure_paths, _persist_seq_counters, _load_seq_counters).
    # ``read_audit_log`` is the forensic walker AND the seed source
    # for ``_load_seq_counters`` on a corrupt sidecar; opening with
    # the bare stdlib ``open()`` would let an attacker who swapped a
    # rotated ``.1`` log file to a symlink redirect the read to
    # attacker-controlled bytes (``/dev/null`` for fail-open seq
    # reset, or attacker-forged content). Mirror the discipline
    # applied across the rest of the module.
    out: list[dict[str, Any]] = []
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    for path in _audit_log_files_in_order():
        try:
            if path.is_symlink():
                logger.warning(
                    "[audit] refusing to read %s: it is a SYMLINK (target: %r). Audit log files must be regular files.",
                    path,
                    os.readlink(path),
                )
                continue
            fd = os.open(str(path), os.O_RDONLY | nofollow)
            with os.fdopen(fd, encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError as parse_exc:
                        # forward-compatibility
                        # justifies skipping malformed lines, but
                        # silent skip interacts badly with the prior
                        # seq-seed walk in ``_load_seq_counters``: a
                        # malformed line for peer X with seq=N is
                        # invisible to the walk's max(seq), so on
                        # next process restart the seq starts below
                        # the highest seq actually written and the
                        # next legit write produces a duplicate.
                        # Emit a DEBUG breadcrumb so operators have
                        # a forensic signal that the seq seed may
                        # be incomplete; the line is still skipped
                        # for forward-compatibility, but the
                        # invisibility is no longer total.
                        logger.debug(
                            "[audit] skipping malformed line in %s: %s",
                            path,
                            parse_exc,
                        )
                        continue
                    if since is not None:
                        ts = record.get("ts")
                        if not isinstance(ts, (int, float)) or ts < since:
                            continue
                    out.append(record)
            # OSError / UnicodeDecodeError raised inside the with-block
            # propagate to the outer ``except OSError`` below. The
            # context manager already closed ``fd`` via ``fh.close``,
            # so no inner cleanup wrapper is needed here -- the
            # previous inner ``try / except (OSError, UnicodeDecodeError):
            # raise`` was a no-op (caught and re-raised with no extra
            # work) and obscured the real flow.
        except OSError as exc:  # pragma: no cover -- best-effort read
            # ELOOP under O_NOFOLLOW is the symlink-raced-after-static-
            # check path; treated as silent skip same as a missing file.
            logger.debug("[audit] failed to read %s: %s", path, exc)
    return out


def verify_audit_integrity(records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Walk the audit log and report tamper / truncation evidence.

    Args:
        records: Optional pre-loaded records list. When None, the current
            audit file is read fresh.

    Returns:
        Dict with keys::

            {
                "total":  <int>,  # records examined
                "signed":  <int>,  # records with a sig field
                "verified":  <int>,  # records whose sig validated
                "bad_signature":<int>,  # records whose sig failed
                "missing_sig":  <int>,  # signed log expected but missing
                "unverifiable_signed":<int>, # signed records, verifier lacks PSK
                "psk_present":  <bool>,  # whether STRANDS_MESH_AUDIT_PSK was set
                "sequence_gaps":[(prev_seq, this_seq),...],
                "ok":  <bool>,  # True iff bad_signature == 0 and
                                         # sequence_gaps == [].
            }
    """
    if records is None:
        records = read_audit_log()

    psk = _audit_psk()
    psk_present = psk is not None

    total = len(records)
    signed = 0
    verified = 0
    bad_signature = 0
    missing_sig = 0
    unverifiable_signed = 0
    gaps: list[tuple[int, int]] = []

    # Track sequence per peer_id -- each process has its own counter so the
    # only stream where consecutive seq values must be adjacent is a single
    # peer's contributions.
    last_seq_by_peer: dict[str, int] = {}

    for record in records:
        sig = record.get("sig")
        seq = record.get("seq")
        peer = record.get("peer_id", "")

        record_is_bad = False
        if sig is not None:
            signed += 1
            if psk is None:
                # verifier lacks PSK while the log carries signed
                # records. Count so ``ok`` fails closed -- a forensic
                # walker missing the PSK MUST NOT see a green light on
                # a signed log it cannot actually verify.
                unverifiable_signed += 1
                continue
            expected = hmac.new(psk, _canonical_bytes(record), hashlib.sha256).hexdigest()
            if hmac.compare_digest(sig, expected):
                verified += 1
            else:
                bad_signature += 1
                record_is_bad = True
        else:
            if psk_present:
                # When the verifier has a PSK, an unsigned record is forged by definition.
                # Mark it as bad so the per-peer cursor does not advance past it --
                # otherwise an attacker who simply omits the ``sig`` field
                # (the natural attack for someone who cannot compute the
                # HMAC) jumps the cursor to whatever ``seq`` they wrote
                # and hides arbitrary deletions from
                # :func:`verify_audit_integrity` 's gap detection.
                missing_sig += 1
                record_is_bad = True

        # Only advance the per-peer cursor on records we actually trust.
        # If we let a tampered record update last_seq_by_peer, an attacker
        # who edits a record's claimed seq value could hide a real gap
        # caused by deleting subsequent records -- the cursor would jump
        # to the forged value and the next legit record would look adjacent.
        if record_is_bad:
            continue

        if isinstance(seq, int) and isinstance(peer, str):
            prev = last_seq_by_peer.get(peer)
            if prev is not None and seq != prev + 1:
                gaps.append((prev, seq))
            # Refuse to roll the cursor backward. A forged record carrying
            # a seq <= prev would let a (forged-low-seq + delete-newer)
            # tamper sequence look adjacent on the next legit record.
            # Keep the highest seq seen for this peer.
            if prev is None or seq > prev:
                last_seq_by_peer[peer] = seq

    return {
        "total": total,
        "signed": signed,
        "verified": verified,
        "bad_signature": bad_signature,
        "missing_sig": missing_sig,
        "unverifiable_signed": unverifiable_signed,
        "psk_present": psk_present,
        "sequence_gaps": gaps,
        # when a PSK is configured at verification time, an
        # unsigned record (missing_sig > 0) is treated as a failure --
        # otherwise an attacker who briefly cleared the env mid-run
        # could write a stretch of unsigned forgeries and the
        # ``ok=True`` reader path would not flag them.
        # when the verifier lacks a PSK but the log carries
        # signed records, fail closed. A forensic walker missing
        # the PSK MUST NOT report ok=True on an unverifiable log.
        "ok": (bad_signature == 0 and not gaps and not (psk_present and missing_sig > 0) and unverifiable_signed == 0),
    }
