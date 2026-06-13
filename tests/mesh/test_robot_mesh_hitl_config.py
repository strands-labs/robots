"""Pin: STRANDS_MESH_HITL_ACTIONS makes the robot_mesh interrupt set configurable.

The default interrupt set gates every physical-actuation action
(emergency_stop, broadcast, tell, send, stop). Consumers narrow or widen
the gate via the env var:

* unset      -> default set (the five actuation actions)
* "all"      -> every gateable action incl. subscribe / watch
* "none"     -> no gate (explicit opt-out; logs one warning)
* CSV subset -> exactly those actions
* bad token  -> structured error, nothing dispatched

These tests fail on pre-fix code (which hard-coded
``_INTERRUPT_REQUIRED = {emergency_stop, broadcast}`` with no override).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import strands_robots.tools.robot_mesh as rmt


@pytest.fixture(autouse=True)
def _reset_caches():
    rmt._reset_rate_limits()
    rmt._reset_interrupt_actions_cache()
    yield
    rmt._reset_rate_limits()
    rmt._reset_interrupt_actions_cache()


def _make_ctx(response: str = "y") -> MagicMock:
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = response
    return ctx


def _call(action, *, ctx=None, **kw):
    fn = getattr(rmt.robot_mesh, "__wrapped__", None) or rmt.robot_mesh
    return fn(action=action, tool_context=ctx or _make_ctx(), **kw)


def _stub_mesh() -> MagicMock:
    m = MagicMock()
    m.tell.return_value = {"status": "ok"}
    m.send.return_value = {"status": "ok"}
    m.broadcast.return_value = [{"status": "ok"}]
    m.emergency_stop.return_value = [{"status": "ok"}]
    m.subscribe.return_value = "sub-name"
    return m


# --- resolver unit tests ------------------------------------------------


def test_resolver_default_is_actuation_set(monkeypatch):
    monkeypatch.delenv("STRANDS_MESH_HITL_ACTIONS", raising=False)
    rmt._reset_interrupt_actions_cache()
    assert rmt._resolve_interrupt_actions() == rmt._DEFAULT_INTERRUPT_ACTIONS
    assert "tell" in rmt._resolve_interrupt_actions()
    assert "send" in rmt._resolve_interrupt_actions()
    assert "stop" in rmt._resolve_interrupt_actions()


def test_resolver_all_expands_to_every_gateable(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_HITL_ACTIONS", "all")
    rmt._reset_interrupt_actions_cache()
    assert rmt._resolve_interrupt_actions() == rmt._GATEABLE_ACTIONS
    assert "subscribe" in rmt._resolve_interrupt_actions()
    assert "watch" in rmt._resolve_interrupt_actions()


def test_resolver_none_is_empty(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_HITL_ACTIONS", "none")
    rmt._reset_interrupt_actions_cache()
    assert rmt._resolve_interrupt_actions() == frozenset()


def test_resolver_explicit_subset(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_HITL_ACTIONS", "emergency_stop,broadcast")
    rmt._reset_interrupt_actions_cache()
    assert rmt._resolve_interrupt_actions() == frozenset({"emergency_stop", "broadcast"})


def test_resolver_bad_token_raises(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_HITL_ACTIONS", "frobnicate")
    rmt._reset_interrupt_actions_cache()
    with pytest.raises(rmt._InterruptConfigError):
        rmt._resolve_interrupt_actions()


# --- dispatcher integration --------------------------------------------


def test_subscribe_gated_when_all(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_HITL_ACTIONS", "all")
    rmt._reset_interrupt_actions_cache()
    ctx = _make_ctx(response="n")  # deny
    m = _stub_mesh()
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("subscribe", target="**/presence", ctx=ctx)
    assert r["status"] == "error"
    assert "declined" in r["content"][0]["text"].lower()
    ctx.interrupt.assert_called_once()


def test_bad_env_returns_structured_error_no_dispatch(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_HITL_ACTIONS", "frobnicate")
    rmt._reset_interrupt_actions_cache()
    m = _stub_mesh()
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("tell", target="peer-a", instruction="go")
    assert r["status"] == "error"
    assert "unknown action" in r["content"][0]["text"].lower()
    m.tell.assert_not_called()


def test_declined_actuation_does_not_consume_rate_slot(monkeypatch):
    """A declined HITL approval must NOT burn a rate-limit slot -- otherwise
    repeated nuisance prompts an operator declines would lock out a genuine
    command. tell's bucket is 30/min; decline 5 then approve and confirm
    the approve still goes through."""
    monkeypatch.delenv("STRANDS_MESH_HITL_ACTIONS", raising=False)
    rmt._reset_interrupt_actions_cache()
    m = _stub_mesh()
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        for _ in range(5):
            r = _call("tell", target="peer-a", instruction="go", ctx=_make_ctx(response="n"))
            assert r["status"] == "error"
        # The 6th, approved, must succeed -- declines consumed no slots.
        r = _call("tell", target="peer-a", instruction="go", ctx=_make_ctx(response="y"))
    assert r["status"] == "success"
    assert m.tell.call_count == 1
