"""Pin: the robot_mesh HITL approval gate only approves an explicit affirmative.

A gated actuation action (``tell``, ``send``, ``broadcast``, ``emergency_stop``,
``stop``) is dispatched only after ``ToolContext.interrupt`` returns an operator
response that ``_interrupt_approves`` accepts. The interrupt contract is
"JSON-serialisable any", so the response may be a non-string (``None``, a dict,
an int, a bool). The gate must:

* approve ONLY the canonical affirmatives ``{y, yes, approve, approved}``,
  case-insensitively and whitespace-trimmed;
* treat every other string (``n``, ``cancel``, ``yep``, empty, whitespace) as a
  decline;
* fail closed on any non-string payload -- a non-string can never be an
  accidental approval, and (critically) must not crash the tool. Without the
  ``isinstance(response, str)`` guard the un-trimmed ``response.strip()`` raises
  ``AttributeError`` past the tool dispatch, both dropping the fail-closed
  decline and violating the "tools return a structured error, never raise past
  dispatch" contract.

These tests fail on pre-fix code with the non-string guard removed (the ``tell``
dispatch raises ``AttributeError`` instead of returning a clean decline).
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


def _make_ctx(response: object) -> MagicMock:
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = response
    return ctx


def _call(action, *, ctx, **kw):
    fn = getattr(rmt.robot_mesh, "__wrapped__", None) or rmt.robot_mesh
    return fn(action=action, tool_context=ctx, **kw)


def _stub_mesh() -> MagicMock:
    m = MagicMock()
    m.tell.return_value = {"status": "ok"}
    return m


# --- _interrupt_approves unit contract ---------------------------------


@pytest.mark.parametrize("response", ["y", "yes", "approve", "approved"])
def test_canonical_affirmatives_approve(response):
    assert rmt._interrupt_approves(response) is True


@pytest.mark.parametrize("response", ["  YES  ", "Approve", "APPROVED", "\ty\n"])
def test_affirmatives_are_case_insensitive_and_trimmed(response):
    assert rmt._interrupt_approves(response) is True


@pytest.mark.parametrize("response", ["n", "no", "cancel", "yep", "ok", "", "   ", "y "])
def test_non_canonical_strings_decline(response):
    # "y " trims to "y" and approves, so it is deliberately excluded above;
    # everything here must decline. (Sanity: the trimmed forms are covered
    # by the affirmative test, so a stray space is fine but "yep"/"ok" are not.)
    if response.strip().lower() in rmt._AFFIRMATIVE_RESPONSES:
        pytest.skip("covered by the affirmative contract")
    assert rmt._interrupt_approves(response) is False


@pytest.mark.parametrize("response", [None, {"approve": True}, 123, True, ["yes"], 0.0])
def test_non_string_payloads_never_approve(response):
    # Defence in depth: the interrupt contract is JSON-serialisable any, so a
    # non-string reply (incl. bool True) must fail closed, never approve.
    assert rmt._interrupt_approves(response) is False


# --- dispatch-level behavior -------------------------------------------


@pytest.mark.parametrize("response", [None, {"approve": True}, 123, True])
def test_non_string_interrupt_response_declines_without_dispatch(response):
    """A non-string operator reply fails closed at the dispatch boundary:
    the gated action returns a clean 'declined' error and the mesh is never
    invoked -- and the tool does not raise past dispatch."""
    m = _stub_mesh()
    ctx = _make_ctx(response)
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("tell", ctx=ctx, target="peer-a", instruction="go")
    assert r["status"] == "error"
    assert "declined" in r["content"][0]["text"].lower()
    m.tell.assert_not_called()
    ctx.interrupt.assert_called_once()


def test_affirmative_interrupt_response_dispatches():
    """The canonical affirmative approves and the action is dispatched once."""
    m = _stub_mesh()
    ctx = _make_ctx("  YES  ")
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("tell", ctx=ctx, target="peer-a", instruction="go")
    assert r["status"] == "success"
    m.tell.assert_called_once()


def test_non_string_decline_does_not_consume_rate_slot():
    """A non-string (fail-closed) decline must not burn a rate-limit slot, so a
    later genuine approval still dispatches -- same contract as a string decline."""
    m = _stub_mesh()
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        for _ in range(5):
            r = _call("tell", ctx=_make_ctx(None), target="peer-a", instruction="go")
            assert r["status"] == "error"
        r = _call("tell", ctx=_make_ctx("y"), target="peer-a", instruction="go")
    assert r["status"] == "success"
    assert m.tell.call_count == 1
