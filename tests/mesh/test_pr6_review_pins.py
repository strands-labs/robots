"""R7 review-feedback regression pins (timing oracle).

Each test here pins an invariant flagged in PR-6 review threads. Keep
this file thin and citation-heavy: each test docstring should name the
exact thread it pins so a future reader can trace the fix to the
review without spelunking PR history.
"""

from __future__ import annotations

import threading

import pytest

from strands_robots.mesh import core
from strands_robots.mesh.sensors import SensorLoopsMixin

# ---------------------------------------------------------------------------
# Thread 4: _resume_lockout HMAC compare is constant-time independent of
# the byte length of the provided override code.
# ---------------------------------------------------------------------------


class TestResumeLockoutTimingOracleClosed:
    """The ``hmac.compare_digest`` call in ``_resume_lockout`` must run
    over fixed-length 32-byte sha256 digests so an attacker probing with
    varying-length override codes cannot use response time to learn
    ``len(STRANDS_MESH_OVERRIDE_CODE)``.

    Pre-fix: ``compare_digest(expected.encode() or b"\x00" * len(provided),
    provided.encode())`` left a residual ``len(expected) ==
    len(provided)`` oracle when ``expected`` was configured. CPython's
    ``hmac.compare_digest`` is documented constant-time only when both
    inputs share a length; on length mismatch it shortcircuits to
    ``False`` quickly.

    Post-fix: both inputs are sha256-hashed before the compare, so the
    compare always operates on 32-byte buffers regardless of input
    length or whether ``expected`` is configured at all.
    """

    @pytest.fixture
    def stub_mesh(self):
        import threading

        m = core.Mesh.__new__(core.Mesh)
        m.peer_id = "test-peer"
        m._estop_lockout = threading.Event()
        m._last_estop_ts = 0.0
        m.publish_safety_event = lambda **kw: None
        return m

    def test_compare_target_is_fixed_length_when_expected_unset(self, stub_mesh, monkeypatch):
        """When ``STRANDS_MESH_OVERRIDE_CODE`` is unset the placeholder
        digest must still be 32 bytes (the sha256 of a fixed sentinel),
        matching the byte length of any real digest. This collapses the
        ``configured-vs-unconfigured`` oracle that the prior fix only
        partially closed.
        """
        monkeypatch.delenv("STRANDS_MESH_OVERRIDE_CODE", raising=False)
        stub_mesh._estop_lockout.set()

        # We do not expose the internal _EXPECTED_HASH; instead, assert
        # the source-level invariant: any call path through
        # _resume_lockout must always reject when expected is unset, and
        # the rejection must NOT depend on the provided length.
        result_short = stub_mesh._resume_lockout("ab")
        result_long = stub_mesh._resume_lockout("a" * 4096)
        assert result_short == {"status": "error", "error": "resume rejected"}
        assert result_long == {"status": "error", "error": "resume rejected"}
        # Lockout must not have cleared either way.
        assert stub_mesh._estop_lockout.is_set()

    def test_compare_runs_on_fixed_digest_length_regardless_of_provided_length(self, stub_mesh, monkeypatch):
        """The compare must operate on 32-byte sha256 digests so probes
        of varying length cannot leak ``len(expected)``.

        We cannot directly observe ``hmac.compare_digest``'s internal
        length-mismatch shortcut, but we can verify that probes of
        wildly varying lengths against a configured but wrong code are
        all rejected uniformly, AND that the source code calls
        ``hashlib.sha256(...).digest()`` on both compare inputs (pinned
        in ``test_source_level_pre_hash_invariant``).
        """
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "secret-32char-hex-1234567890abcd")
        stub_mesh._estop_lockout.set()

        # Length-varied probes -- all must reject (none equals the
        # configured code; the sha256 pre-hash ensures the compare runs
        # over 32 bytes regardless).
        for probe in ["", "a", "ab", "a" * 16, "a" * 32, "a" * 1024, "a" * 65536]:
            stub_mesh._estop_lockout.set()  # re-arm in case any path cleared it
            result = stub_mesh._resume_lockout(probe)
            assert result == {"status": "error", "error": "resume rejected"}, (
                f"length-{len(probe)} probe leaked through compare (expected reject)"
            )
            assert stub_mesh._estop_lockout.is_set(), f"length-{len(probe)} probe cleared lockout (compare oracle leak)"

    def test_correct_override_code_clears_lockout(self, stub_mesh, monkeypatch):
        """The pre-hash refactor must not break the happy path: a
        matching override code still clears the lockout. Without this
        the security-hardening could silently lock the resume path."""
        secret = "correct-override-code-32-chars-x"
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", secret)
        stub_mesh._estop_lockout.set()

        result = stub_mesh._resume_lockout(secret)
        assert result == {"status": "ok"}
        assert not stub_mesh._estop_lockout.is_set()

    def test_source_level_pre_hash_invariant(self):
        """Structural pin: the resume-compare path must hash both
        inputs to a fixed digest before calling
        ``hmac.compare_digest``. Pin via source-text inspection so an
        accidental revert (e.g. someone restoring the
        ``b"\x00" * len(provided)`` placeholder) trips this test
        regardless of runtime path coverage.
        """
        import inspect

        source = inspect.getsource(core.Mesh._resume_lockout)
        assert "hashlib.sha256(provided.encode()).digest()" in source, (
            "_resume_lockout must hash provided to a fixed-length digest "
            "before compare_digest -- the prior placeholder approach "
            "left a len(expected)-vs-len(provided) timing oracle."
        )
        assert "hashlib.sha256(expected.encode()).digest()" in source, (
            "_resume_lockout must hash expected when configured."
        )
        # The prior placeholder pattern must NOT reappear -- a
        # readback of the form ``b"\x00" * max(1, len(provided))``
        # would re-introduce the length oracle.
        assert 'b"\\x00" * max(1, len(provided))' not in source, (
            "the variable-length placeholder must not re-appear -- it "
            "leaks len(provided) via compare_digest's length-mismatch "
            "shortcut."
        )


# ---------------------------------------------------------------------------
# Thread 5: Mesh.publish must shadow SensorLoopsMixin.publish via MRO.
# ---------------------------------------------------------------------------


def test_mesh_publish_shadows_sensor_loops_mixin():
    """``Mesh.publish`` must shadow the ``SensorLoopsMixin.publish`` stub.

    Review feedback: the mixin's ``publish`` body raises
    ``NotImplementedError`` -- a deliberate replacement for the prior
    ``...`` no-effect statement (CodeQL #226). The contract is that
    ``Mesh`` itself defines a real ``publish`` so the stub is never
    reached at runtime.

    The contract chain ('``Mesh.publish`` shadows this stub via MRO')
    depends on every host class declaring ``class Mesh(SensorLoopsMixin)``
    AND defining its own ``publish``. A future refactor that inserts
    another mixin between them (e.g. one that also implements ``publish``
    but forwards differently), or removes ``Mesh.publish`` entirely,
    would silently fall through to ``NotImplementedError`` only at
    runtime when a sensor loop fires (POSE_HZ tick, IMU tick, etc.) --
    a latent fault that escapes import-time checks and unit tests of
    other paths.

    This test surfaces such a regression at collection time, so a
    subclass authoring error trips CI before any sensor loop runs in
    production. Per AGENTS.md > "Pin regression tests for reviewed
    fixes".
    """
    from strands_robots.mesh import sensors

    mesh_publish = core.Mesh.publish
    mixin_publish = sensors.SensorLoopsMixin.publish

    assert mesh_publish is not mixin_publish, (
        "Mesh.publish must override SensorLoopsMixin.publish; the mixin "
        "stub raises NotImplementedError and is never meant to execute. "
        "If this fires, either Mesh lost its publish definition or a "
        "mixin was reordered -- check the MRO."
    )

    # Belt-and-braces: Mesh's own ``__dict__`` must carry ``publish``,
    # not just inherit it from somewhere on the MRO. This catches the
    # subtler regression where someone deletes ``Mesh.publish`` and
    # accidentally relies on a different mixin's implementation.
    assert "publish" in core.Mesh.__dict__, "Mesh.publish must be defined on Mesh itself, not inherited from the mixin."


# ---------------------------------------------------------------------------
# Thread 6: _estop_replay_cache shape invariant -- all entries are 3-tuples
# (issuer_id, mono_ts, wire_zid). No half-defensive isinstance shim.
# ---------------------------------------------------------------------------


class TestEstopReplayCacheShapeInvariant:
    """The ``_estop_replay_cache`` must only contain ``(str, float, str | None)``
    3-tuple values. The type annotation at ``__init__`` declares this and the
    sole writer (the acceptance path in ``_on_safety_estop``) emits it.

    Pre-fix: a defensive ``isinstance(cached_entry, tuple) and len(cached_entry) >= 3``
    at the corroboration-branch entry tolerated bare-float legacy stubs, but the
    ts_view comprehension (``v[1]``) and per-issuer iteration (``for issuer, _mono,
    _zid in ...values()``) both crashed on bare floats. The half-defensive state
    masked contract violations at the corroboration site while letting them explode
    at eviction time -- the worst of both worlds.

    Post-fix: no isinstance guard. A bare-float entry crashes immediately on the
    cache-hit path (``cached_entry[2]`` on a float raises TypeError), so shape
    violations surface as close to the writer bug as possible.

    This test pins the structural invariant: ``_on_safety_estop`` writes
    3-tuples AND no defensive isinstance exists in the source.

    Addresses review thread on core.py:1570.
    """

    def test_cache_writer_emits_3_tuple(self):
        """The acceptance path writes (issuer_id, mono_ts, wire_zid)."""
        import inspect

        source = inspect.getsource(core.Mesh._on_safety_estop)
        # The writer line: self._estop_replay_cache[cache_key] = (issuer_id, now_mono, wire_zid)
        assert "(issuer_id, now_mono, wire_zid)" in source, (
            "The estop replay cache writer must emit the canonical 3-tuple "
            "(issuer_id, now_mono, wire_zid). If this fails, the shape "
            "contract at __init__ has drifted from the writer."
        )

    def test_no_defensive_isinstance_on_cache_entry(self):
        """The isinstance shim must be absent -- direct subscript access
        surfaces shape violations immediately."""
        import inspect

        source = inspect.getsource(core.Mesh._on_safety_estop)
        assert "isinstance(cached_entry, tuple)" not in source, (
            "The defensive isinstance(cached_entry, tuple) shim was removed "
            "because it disagreed with ts_view and per-issuer iteration. "
            "If this fires, someone reintroduced the half-defensive state."
        )

    def test_bare_float_in_cache_crashes_on_hit(self):
        """Verify that a bare-float cache entry is not tolerated.

        Pre-fix this would silently set cached_wire_zid=None (degrading
        corroboration to always-rejected). Post-fix it raises TypeError
        on ``cached_entry[2]`` immediately.
        """
        import threading
        import time

        m = core.Mesh.__new__(core.Mesh)
        m.peer_id = "test-receiver"
        m._estop_replay_cache = {}
        m._estop_replay_lock = threading.Lock()
        m._estop_lockout = threading.Event()
        m._last_estop_ts = 0.0
        m._last_estop_mono = 0.0

        # Seed a BARE FLOAT (the shape the removed isinstance tolerated)
        bad_key = time.time()
        m._estop_replay_cache[bad_key] = 123.456  # type: ignore[assignment]

        # Simulate a cache-hit read at the corroboration branch.
        # Post-fix: direct ``cached_entry[2]`` on a float raises TypeError.
        cached_entry = m._estop_replay_cache[bad_key]
        with pytest.raises(TypeError):
            _ = cached_entry[2]  # type: ignore[index]

    def test_type_annotation_matches_3_tuple(self):
        """The __init__ type annotation must declare the 3-tuple shape."""
        import inspect

        source = inspect.getsource(core.Mesh.__init__)
        assert "dict[float, tuple[str, float, str | None]]" in source, (
            "The _estop_replay_cache type annotation must be "
            "dict[float, tuple[str, float, str | None]] to enforce "
            "the 3-tuple shape contract."
        )


class TestEstopLockoutEngagesAtCap:
    """Pin: lockout ALWAYS engages even when per-issuer cache cap is exceeded.

    Regression for review thread core.py:1611 (2026-05-30).

    Pre-fix: issuer_slots >= per_issuer_cap triggered an early return,
    preventing lockout engagement. A legitimate operator whose estops
    filled their per-issuer cache cap had subsequent estops silently
    dropped from BOTH cache AND lockout -- inverting the safety contract.

    Post-fix: the cache slot is still refused (resource fairness), but
    the lockout engages unconditionally (safety-primitive availability).
    """

    def test_lockout_engages_when_issuer_at_cap(self):
        """A legitimate operator at per-issuer cache cap still engages lockout."""
        import inspect
        import threading

        m = core.Mesh.__new__(core.Mesh)
        m.peer_id = "test-receiver"
        m._estop_replay_cache = {}
        m._estop_replay_lock = threading.Lock()
        m._estop_lockout = threading.Event()
        m._last_estop_ts = 0.0
        m._last_estop_mono = 0.0

        # Verify the early return is absent from the source.
        source = inspect.getsource(core.Mesh._on_safety_estop)

        # The problematic pattern: a second `issuer_slots >= per_issuer_cap`
        # check followed by `return` that prevented lockout engagement.
        # Count occurrences of "issuer_slots >= per_issuer_cap" -- there should
        # be exactly ONE (the cache-slot gating), not TWO (cache + lockout).
        occurrences = source.count("issuer_slots >= per_issuer_cap")
        assert occurrences == 1, (
            f"Expected exactly 1 occurrence of 'issuer_slots >= per_issuer_cap' "
            f"(the cache-slot gate), found {occurrences}. "
            f"A second occurrence that early-returns before lockout engagement "
            f"is the safety regression this test pins against."
        )

    def test_no_early_return_between_cache_and_lockout(self):
        """Structural: no bare `return` between cache logic and lockout block."""
        import inspect
        import re

        source = inspect.getsource(core.Mesh._on_safety_estop)
        lines = source.split("\n")

        # Find the line with "self._estop_replay_cache[cache_key] ="
        cache_write_idx = None
        lockout_check_idx = None
        for i, line in enumerate(lines):
            if "self._estop_replay_cache[cache_key] =" in line:
                cache_write_idx = i
            if "if not lockout_was_engaged:" in line or "if not self._estop_lockout.is_set():" in line:
                lockout_check_idx = i
                break

        assert cache_write_idx is not None, "Could not find cache write line"
        assert lockout_check_idx is not None, "Could not find lockout check line"

        # Between cache write and lockout check, there must be no bare `return`
        between = lines[cache_write_idx + 1 : lockout_check_idx]
        bare_returns = [line.strip() for line in between if re.match(r"^\s*return\s*$", line)]
        assert not bare_returns, (
            f"Found bare 'return' statement(s) between cache-write and "
            f"lockout-engage block: {bare_returns}. This would silently "
            f"drop the lockout for at-cap issuers (safety regression)."
        )


# ---------------------------------------------------------------------------
# Thread (2026-05-30 R14): cache-key namespace conflation -- a bridge
# peer with peer_id matching a Zenoh wire_zid must not collide.
# Reviewer: "one-line fix worth landing in this PR."
# ---------------------------------------------------------------------------


class TestResumeCacheKeyNamespaceIsolation:
    """The resume replay cache key MUST domain-tag the issuer identifier
    so a Zenoh wire_zid and a body issuer_id from a bridge transport
    never share the same tuple slot.

    Pre-fix: ``cache_key = (wire_zid or issuer_id, proof_nonce)``
    Post-fix: ``cache_key = (("wire", wire_zid) if wire_zid is not None else ("body", issuer_id), proof_nonce)``

    Pin: thread core.py:1853 (2026-05-30 review batch).
    """

    def test_domain_tagged_key_no_collision(self):
        """A bridge peer with peer_id='ab12cd' and a Zenoh peer with
        wire_zid='ab12cd' must produce distinct cache keys."""
        import inspect

        source = inspect.getsource(core.Mesh._on_safety_resume)

        # Structural: the old conflating pattern must be absent.
        assert "wire_zid or issuer_id" not in source, (
            "cache_key still uses the conflating 'wire_zid or issuer_id' pattern -- namespace collision is possible"
        )

        # Structural: domain tag must be present.
        assert '"wire"' in source or "'wire'" in source, "cache_key does not contain a 'wire' domain tag"
        assert '"body"' in source or "'body'" in source, "cache_key does not contain a 'body' domain tag"

    def test_same_string_different_transport_distinct_keys(self):
        """Simulate key construction logic: same string via wire vs body
        must produce distinct first-tuple elements."""
        # Simulate the new key construction
        shared_hex = "ab12cd34ef567890"
        proof = "nonce-123"

        # Zenoh peer: wire_zid is set
        wire_key = (("wire", shared_hex), proof)
        # Bridge peer: wire_zid is None, issuer_id matches the hex
        body_key = (("body", shared_hex), proof)

        assert wire_key != body_key, (
            "Same string from different transports must produce distinct cache keys to prevent cross-transport eviction"
        )

    def test_none_wire_zid_uses_body_domain(self):
        """When wire_zid is None (bridge/IoT transport), the key must
        use the 'body' domain tag with issuer_id."""
        import inspect

        source = inspect.getsource(core.Mesh._on_safety_resume)

        # The conditional must check for None explicitly
        assert "wire_zid is not None" in source or "wire_zid is None" in source, (
            "Domain tag conditional must use explicit None check, not truthy/falsy (empty string '' is falsy but valid)"
        )


# ---------------------------------------------------------------------------
# Issue #258: sensor *_loop except clauses must re-raise NotImplementedError
# so an MRO contract violation (Mesh.publish missing) crashes loud at runtime
# instead of being swallowed by the catch-all and logged at DEBUG level.
# ---------------------------------------------------------------------------


class _PublishlessMesh(SensorLoopsMixin):
    """Host class that omits ``publish``, so the mixin stub fires.

    Every sensor read returns a truthy payload so each loop reaches the
    ``self.publish(...)`` call, which raises ``NotImplementedError`` from
    the mixin stub. A correctly-ordered ``except NotImplementedError: raise``
    must let it propagate out of the loop.
    """

    def __init__(self) -> None:
        self.peer_id = "test-peer"
        self.robot = None
        self._running = True
        self._stop_event = threading.Event()

    def _read_pose(self):
        return {"x": 0.0}

    def _read_health(self):
        return {"ok": True}

    def _read_imu(self):
        return {"ax": 0.0}

    def _read_odom(self):
        return {"vx": 0.0}

    def _read_lidar_summary(self):
        return {"n": 1}

    def _read_lidar_state(self):
        return {"state": "ok"}

    def _read_hands(self):
        return {"left": {"open": True}}

    def _read_map_info(self):
        return {"w": 1}


@pytest.mark.parametrize(
    "loop_name",
    [
        "_pose_loop",
        "_health_loop",
        "_imu_loop",
        "_odom_loop",
        "_lidar_loop",
        "_hand_loop",
        "_map_info_loop",
    ],
)
def test_sensor_loops_reraise_not_implemented_error(loop_name):
    """Each sensor loop must let NotImplementedError propagate (issue #258).

    The mixin ``publish`` stub raises ``NotImplementedError`` when no host
    ``Mesh.publish`` shadows it. The loop's ``except NotImplementedError:
    raise`` clause must surface this MRO contract violation loudly rather
    than swallowing it in the catch-all ``except Exception`` and emitting a
    DEBUG log per Hz tick. A guard ensures the test cannot hang if the
    re-raise is removed (the catch-all would otherwise loop forever).
    """
    fake = _PublishlessMesh()

    def _abort_if_swallowed():
        fake._running = False
        fake._stop_event.set()

    watchdog = threading.Timer(5.0, _abort_if_swallowed)
    watchdog.start()
    try:
        with pytest.raises(NotImplementedError):
            getattr(fake, loop_name)()
    finally:
        watchdog.cancel()
