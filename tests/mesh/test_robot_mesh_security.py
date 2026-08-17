"""Tests for the LLM-injection defences in ``robot_mesh``.

The ``robot_mesh`` tool exposes physical actuation primitives to a Strands
agent. Four layers of defence sit between the LLM call and the wire:

* A Strands SDK human-in-the-loop interrupt for ``emergency_stop`` and
  ``broadcast``. The operator response is delivered out-of-band of the
  tool's argument flow so a prompt cannot smuggle approval.
* A per-action sliding-window rate limit (e.g. ``emergency_stop`` capped
  at 3 calls/minute).
* Server-side :func:`strands_robots.mesh.security.validate_command` run
  client-side too, so attacker-controlled ``policy_host`` values, unknown
  actions, etc. are rejected before they ever leave the agent.
* Every safety-significant call is recorded through the audit log.

These tests pin each layer.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import strands_robots.tools.robot_mesh as rmt


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch, tmp_path):
    """Each test gets a clean rate-limit history and audit dir."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("STRANDS_MESH_AUDIT_PSK", raising=False)
    monkeypatch.delenv("STRANDS_MESH_PSK", raising=False)
    monkeypatch.delenv("STRANDS_MESH_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("STRANDS_MESH_POLICY_HOST_ALLOW", raising=False)
    # #10 gateway fallback: in a robot-less test process ``robot_mesh()``
    # would lazily bring up a REAL zenoh gateway Mesh (session + discovery
    # sleep + 30s RPC timeouts) before these mocks are consulted. That both
    # breaks hermeticity and stretches a rate-limit burst past its window.
    # These tests pin the defence layers, not transport - forbid the gateway.
    monkeypatch.setattr(rmt, "_gateway_mesh", lambda: None)
    rmt._reset_rate_limits()
    from strands_robots.mesh import audit

    audit._SEQ_COUNTER = 0
    yield
    rmt._reset_rate_limits()


def _make_ctx(response: str = "y", *, raises: bool = False) -> MagicMock:
    """Stand-in ToolContext whose interrupt() returns *response*."""
    ctx = MagicMock(name="ToolContext")
    if raises:
        ctx.interrupt.side_effect = RuntimeError("interrupts unavailable")
    else:
        ctx.interrupt.return_value = response
    return ctx


def _call(action, *, ctx: MagicMock | None = None, **kw):
    """Invoke the underlying function with a stub ToolContext."""
    fn = getattr(rmt.robot_mesh, "__wrapped__", None) or rmt.robot_mesh
    return fn(action=action, tool_context=ctx or _make_ctx(), **kw)


def _stub_mesh() -> MagicMock:
    """Mesh-shaped mock with the methods the tool calls."""
    m = MagicMock()
    m.tell.return_value = {"status": "ok"}
    m.send.return_value = {"status": "ok"}
    m.broadcast.return_value = [{"status": "ok"}]
    m.emergency_stop.return_value = [{"status": "ok"}]
    return m


# --- Interrupt gate (replaces former confirm=bool) ----------------------


class TestInterruptGate:
    def test_emergency_stop_raises_interrupt(self):
        """Without an approving response, emergency_stop must NOT call mesh."""
        ctx = _make_ctx(response="n")  # operator denies
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call("emergency_stop", ctx=ctx)
        assert r["status"] == "error"
        assert "declined" in r["content"][0]["text"].lower()
        m.emergency_stop.assert_not_called()
        ctx.interrupt.assert_called_once()
        # The interrupt name MUST be namespaced to the action.
        args, kwargs = ctx.interrupt.call_args
        assert args[0] == "robot_mesh-emergency_stop-approval"
        assert kwargs["reason"]["action"] == "emergency_stop"

    def test_emergency_stop_runs_when_operator_approves(self):
        ctx = _make_ctx(response="y")
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call("emergency_stop", ctx=ctx)
        assert r["status"] == "success"
        m.emergency_stop.assert_called_once()

    def test_emergency_stop_yes_full_word_approves(self):
        ctx = _make_ctx(response="YES")
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call("emergency_stop", ctx=ctx)
        assert r["status"] == "success"

    def test_broadcast_raises_interrupt(self):
        ctx = _make_ctx(response="n")
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call("broadcast", command='{"action": "status"}', ctx=ctx)
        assert r["status"] == "error"
        assert "declined" in r["content"][0]["text"].lower()
        m.broadcast.assert_not_called()

    def test_broadcast_runs_when_operator_approves(self):
        ctx = _make_ctx(response="y")
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call("broadcast", command='{"action": "status"}', ctx=ctx)
        assert r["status"] == "success"
        m.broadcast.assert_called_once()

    def test_tell_raises_interrupt_by_default(self):
        """tell is a single-peer physical-actuation action and is in the
        DEFAULT interrupt set -- a prompt-injected agent cannot drive a
        robot without an out-of-band operator approval."""
        ctx = _make_ctx(response="n")  # operator denies
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call("tell", target="peer-a", instruction="pick up cube", ctx=ctx)
        assert r["status"] == "error"
        assert "declined" in r["content"][0]["text"].lower()
        m.tell.assert_not_called()
        ctx.interrupt.assert_called_once()
        args, kwargs = ctx.interrupt.call_args
        assert args[0] == "robot_mesh-tell-approval"
        assert kwargs["reason"]["action"] == "tell"

    def test_tell_runs_when_operator_approves(self):
        ctx = _make_ctx(response="y")
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call("tell", target="peer-a", instruction="pick up cube", ctx=ctx)
        assert r["status"] == "success"
        m.tell.assert_called_once()

    def test_tell_not_gated_when_opted_out(self, monkeypatch):
        """Consumers can narrow the gate via STRANDS_MESH_HITL_ACTIONS.
        With 'none', tell dispatches without an interrupt (back-compat
        escape hatch for trusted single-tenant deployments)."""
        monkeypatch.setenv("STRANDS_MESH_HITL_ACTIONS", "none")
        rmt._reset_interrupt_actions_cache()
        try:
            ctx = _make_ctx()
            m = _stub_mesh()
            with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
                r = _call("tell", target="peer-a", instruction="pick up cube", ctx=ctx)
            assert r["status"] == "success"
            ctx.interrupt.assert_not_called()
            m.tell.assert_called_once()
        finally:
            rmt._reset_interrupt_actions_cache()

    def test_interrupt_unavailable_fails_closed(self):
        """When the runtime can't deliver interrupts (e.g. direct
        agent.tool.X path), the tool MUST refuse rather than execute."""
        ctx = _make_ctx(raises=True)
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call("emergency_stop", ctx=ctx)
        assert r["status"] == "error"
        assert "interrupt" in r["content"][0]["text"].lower()
        m.emergency_stop.assert_not_called()


# --- rate limiting ------------------------------------------------------


class TestRateLimit:
    def test_emergency_stop_capped_at_3_per_window(self):
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            for _ in range(3):
                r = _call("emergency_stop")
                assert r["status"] == "success"
            r = _call("emergency_stop")
        assert r["status"] == "error"
        assert "rate limit" in r["content"][0]["text"]
        assert m.emergency_stop.call_count == 3

    def test_tell_capped_at_30_per_window(self):
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            successes = 0
            for _ in range(35):
                r = _call("tell", target="peer-a", instruction="ping")
                if r["status"] == "success":
                    successes += 1
            assert successes == 30

    def test_distinct_actions_dont_share_buckets(self):
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            for _ in range(3):
                _call("emergency_stop")  # exhausts emergency_stop bucket
            r = _call("tell", target="peer-a", instruction="ping")
            assert r["status"] == "success"  # tell bucket unaffected


# --- command validation ------------------------------------------------


class TestCommandValidation:
    def test_send_rejects_attacker_policy_host(self):
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call(
                "send",
                target="peer-a",
                command='{"action": "execute", "instruction": "go", "policy_provider": "mock", "policy_host": "evil.example.com"}',
            )
        assert r["status"] == "error"
        assert "policy_host" in r["content"][0]["text"]
        m.send.assert_not_called()

    def test_send_rejects_unknown_action(self):
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call("send", target="peer-a", command='{"action": "rm_rf"}')
        assert r["status"] == "error"
        m.send.assert_not_called()

    def test_send_rejects_non_json(self):
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call("send", target="peer-a", command="not json")
        assert r["status"] == "error"

    def test_send_rejects_non_dict_json(self):
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call("send", target="peer-a", command='["not a dict"]')
        assert r["status"] == "error"

    def test_broadcast_validates_after_interrupt(self):
        """Broadcast goes through interrupt FIRST, then validation. An
        approved-but-malformed broadcast must still be rejected by
        validation, not silently shipped."""
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call(
                "broadcast",
                command='{"action": "execute", "instruction": "go", "policy_provider": "mock", "policy_host": "evil.com"}',
            )
        assert r["status"] == "error"
        assert "policy_host" in r["content"][0]["text"]
        m.broadcast.assert_not_called()

    def test_tell_too_long_instruction_rejected(self):
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call("tell", target="peer-a", instruction="x" * 3000)
        assert r["status"] == "error"
        assert "exceeds" in r["content"][0]["text"]

    def test_tell_passes_with_valid_args(self):
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call("tell", target="peer-a", instruction="pick up cube", duration=10)
        assert r["status"] == "success"


# --- audit logging ------------------------------------------------------


class TestAudit:
    def test_successful_emergency_stop_audited(self):
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            _call("emergency_stop")
        from strands_robots.mesh.audit import read_audit_log

        events = [r for r in read_audit_log() if r.get("event") == "llm_tool_action"]
        assert any(r["payload"]["action"] == "emergency_stop" and r["payload"]["success"] for r in events)

    def test_declined_call_audited(self):
        ctx = _make_ctx(response="no")
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=_stub_mesh()):
            _call("emergency_stop", ctx=ctx)
        from strands_robots.mesh.audit import read_audit_log

        events = [r for r in read_audit_log() if r.get("event") == "llm_tool_action"]
        assert any(
            r["payload"]["action"] == "emergency_stop"
            and r["payload"]["success"] is False
            and "declined" in r["payload"]["detail"]
            for r in events
        )


class TestRateLimitTOCTOU:
    """pre-fix, two concurrent emergency_stop calls could
    each pass _rate_limit_check, both get operator-approved, both record,
    briefly exceeding the configured 3/60s limit. The post-approval
    re-check under _RATE_LOCK closes the race.
    """

    def test_concurrent_approvals_capped_at_configured_limit(self, monkeypatch):
        # Lower the limit to make the race deterministic.
        rmt._reset_rate_limits()

        # Pretend the limit is 2 per 60s.
        monkeypatch.setitem(rmt._RATE_LIMITS, "emergency_stop", (2, 60.0))

        # Two pre-interrupt checks pass (no slots consumed).
        assert rmt._rate_limit_check("emergency_stop") is None
        assert rmt._rate_limit_check("emergency_stop") is None

        # First post-approval atomic check+record: succeeds (slot 1).
        assert rmt._rate_limit_check_and_record("emergency_stop") is None
        # Second: succeeds (slot 2).
        assert rmt._rate_limit_check_and_record("emergency_stop") is None
        # Third: must FAIL even though _rate_limit_check would have
        # passed pre-interrupt. This is the race the fix closes.
        race_err = rmt._rate_limit_check_and_record("emergency_stop")
        assert race_err is not None
        assert "rate limit exceeded" in race_err
        assert "raced past" in race_err

    def test_atomic_helper_does_not_consume_on_full_bucket(self, monkeypatch):
        rmt._reset_rate_limits()
        monkeypatch.setitem(rmt._RATE_LIMITS, "broadcast", (1, 60.0))
        assert rmt._rate_limit_check_and_record("broadcast") is None
        # Full -- second call rejects.
        assert rmt._rate_limit_check_and_record("broadcast") is not None
        # Verify no spurious extra slot was added.
        assert len(rmt._RATE_HISTORY["broadcast"]) == 1


class TestDeclineSideChannel322:
    """#322: a declined HITL response must not leak the operator's literal
    reply back into the LLM context (the human-as-content-side-channel)."""

    def test_declined_response_not_echoed_to_llm(self):
        """The operator's literal interrupt reply must NOT appear in the
        tool result returned to the model. A flat sentinel is returned; the
        full reply is kept only in the local audit row."""
        secret = "no-because-CEO-said-PROJECT-TITAN-ships-friday"
        ctx = _make_ctx(response=secret)
        m = _stub_mesh()
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call("emergency_stop", ctx=ctx)
        assert r["status"] == "error"
        text = r["content"][0]["text"]
        assert "declined" in text.lower()
        # The operator's literal reply MUST NOT be echoed to the LLM.
        assert secret not in text, "operator's literal decline reply leaked to the LLM result (#322 side-channel)"
        m.emergency_stop.assert_not_called()


# --- rate-limit sliding-window expiry -----------------------------------


class _FakeClock:
    """Deterministic stand-in for ``time.monotonic`` used to drive the
    rate-limit sliding window without sleeping. ``advance`` moves the
    virtual clock forward so stale bucket entries fall outside the
    window and get pruned.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class TestRateLimitWindowExpiry:
    """The sliding window must *forget* calls older than the configured
    window. Without expiry, three ``emergency_stop`` calls issued long
    ago would permanently lock the agent out of ever issuing another -
    the opposite of the safety property (bound nuisance, never inhibit a
    genuine emergency). These pin the stale-entry pruning in both
    rate-limit helpers: the read-only check and the atomic reservation
    that is now the only way a slot is consumed.
    """

    def test_check_clears_after_window_elapses(self, monkeypatch):
        clock = _FakeClock()
        monkeypatch.setattr(rmt.time, "monotonic", clock)
        rmt._reset_rate_limits()

        # Fill emergency_stop (cap 3 / 60s window) at t=0.
        for _ in range(3):
            assert rmt._rate_limit_check_and_record("emergency_stop") is None
        # Bucket full: a check (which does not consume) reports rejection.
        assert rmt._rate_limit_check("emergency_stop") is not None

        # Just before the window closes the slots are still held.
        clock.advance(59.0)
        assert rmt._rate_limit_check("emergency_stop") is not None

        # Past the window the stale entries are pruned and a slot frees up.
        clock.advance(2.0)
        assert rmt._rate_limit_check("emergency_stop") is None

    def test_check_and_record_frees_slot_after_window(self, monkeypatch):
        clock = _FakeClock()
        monkeypatch.setattr(rmt.time, "monotonic", clock)
        rmt._reset_rate_limits()

        # Atomically reserve all 3 slots at t=0; the 4th is rejected.
        for _ in range(3):
            assert rmt._rate_limit_check_and_record("emergency_stop") is None
        assert rmt._rate_limit_check_and_record("emergency_stop") is not None

        # Once the window elapses the stale entries are pruned and the
        # atomic check+record reserves a fresh slot.
        clock.advance(61.0)
        assert rmt._rate_limit_check_and_record("emergency_stop") is None
        assert len(rmt._RATE_HISTORY["emergency_stop"]) == 1
