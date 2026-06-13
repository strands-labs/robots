"""Pin: robot_mesh subscribe is allowlist-scoped and inbox reads are audited.

Defence in depth for the cross-peer telemetry-leak surface. Even on a mesh
running the permissive default ACL, the tool layer must:

* allow subscribing only to low-impact shared topic classes by default,
* reject subscribing to another peer's cmd / state / camera streams,
* let operators extend the allowlist via STRANDS_MESH_SUBSCRIBE_ALLOW,
* audit every inbox read (which sub, how many frames).

These fail on pre-fix code (subscribe accepted any key expr; inbox reads
were never audited).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import strands_robots.tools.robot_mesh as rmt


@pytest.fixture(autouse=True)
def _reset():
    rmt._reset_rate_limits()
    rmt._reset_interrupt_actions_cache()
    rmt._reset_subscribe_allowlist_cache()
    yield
    rmt._reset_rate_limits()
    rmt._reset_interrupt_actions_cache()
    rmt._reset_subscribe_allowlist_cache()


def _make_ctx(response: str = "y") -> MagicMock:
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = response
    return ctx


def _call(action, *, ctx=None, **kw):
    fn = getattr(rmt.robot_mesh, "__wrapped__", None) or rmt.robot_mesh
    return fn(action=action, tool_context=ctx or _make_ctx(), **kw)


def _stub_mesh() -> MagicMock:
    m = MagicMock()
    m.subscribe.return_value = "sub-name"
    m.inbox = {}
    return m


# --- matcher unit tests -------------------------------------------------


def test_ke_matches_exact():
    assert rmt._ke_matches("**/presence", "**/presence") is True


def test_ke_matches_trailing_doublestar():
    assert rmt._ke_matches("**/safety/**", "**/safety/event") is True
    assert rmt._ke_matches("**/safety/**", "**/safety/estop") is True
    assert rmt._ke_matches("**/safety/**", "**/safety") is True


def test_ke_matches_rejects_unrelated():
    assert rmt._ke_matches("**/presence", "reachy/cmd") is False
    assert rmt._ke_matches("**/safety/**", "**/state/x") is False


def test_default_allowlist_blocks_cmd_and_state():
    assert rmt._is_allowed_subscribe_target("reachy/cmd") is False
    assert rmt._is_allowed_subscribe_target("peer-b/state/joints") is False
    assert rmt._is_allowed_subscribe_target("peer-b/camera/rgb") is False


def test_default_allowlist_permits_shared_classes():
    assert rmt._is_allowed_subscribe_target("**/presence") is True
    assert rmt._is_allowed_subscribe_target("**/health") is True
    assert rmt._is_allowed_subscribe_target("**/safety/event") is True


# --- dispatcher integration --------------------------------------------


def test_subscribe_allows_presence():
    m = _stub_mesh()
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("subscribe", target="**/presence", name="p")
    assert r["status"] == "success"
    m.subscribe.assert_called_once()


def test_subscribe_blocks_cmd_stream():
    m = _stub_mesh()
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("subscribe", target="victim/cmd", name="x")
    assert r["status"] == "error"
    assert "allowed topic set" in r["content"][0]["text"]
    m.subscribe.assert_not_called()


def test_subscribe_env_extends_allowlist(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH_SUBSCRIBE_ALLOW", "**/state/**")
    rmt._reset_subscribe_allowlist_cache()
    m = _stub_mesh()
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("subscribe", target="**/state/joints", name="s")
    assert r["status"] == "success"
    m.subscribe.assert_called_once()


def test_inbox_read_is_audited():
    m = _stub_mesh()
    m.inbox = {"sub-a": [("topic", {"x": 1}), ("topic", {"x": 2})]}
    with (
        patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m),
        patch("strands_robots.tools.robot_mesh._audit_tool_action") as audit,
    ):
        r = _call("inbox", name="sub-a")
    assert r["status"] == "success"
    # An inbox read must emit exactly one audit event recording the count.
    inbox_audits = [c for c in audit.call_args_list if c.args and c.args[0] == "inbox"]
    assert inbox_audits, "inbox read was not audited"
    # detail string carries the read count
    assert any("read=2" in (c.args[3] if len(c.args) > 3 else "") for c in inbox_audits)


# --- watch allowlist enforcement (parallel channel to subscribe) --------


def test_watch_rejects_off_allowlist_peer():
    """watch(target='peer-b') subscribes to strands/peer-b/stream which carries
    observations + policy actions. Without the allowlist check it provides a
    parallel telemetry-exfiltration channel bypassing the subscribe gate.
    This test fails on pre-fix code (watch accepted any target unconditionally).
    """
    m = _stub_mesh()
    m.on_stream.return_value = "stream-peer-b"
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("watch", target="peer-b")
    assert r["status"] == "error"
    assert "allowed topic set" in r["content"][0]["text"]
    assert "strands/peer-b/stream" in r["content"][0]["text"]
    m.on_stream.assert_not_called()


def test_watch_rejects_arbitrary_peer_ids():
    """Parametric: no peer ID passes by default (stream key never matches
    **/presence, **/health, or **/safety/**)."""
    m = _stub_mesh()
    m.on_stream.return_value = "stream-x"
    for peer in ("reachy", "arm-a", "robot_1", "attacker"):
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call("watch", target=peer)
        assert r["status"] == "error", f"watch should reject peer '{peer}'"
    m.on_stream.assert_not_called()


def test_watch_allowed_when_operator_extends_allowlist(monkeypatch):
    """Operators who want watch access can extend STRANDS_MESH_SUBSCRIBE_ALLOW
    to include the stream key pattern."""
    monkeypatch.setenv("STRANDS_MESH_SUBSCRIBE_ALLOW", "strands/*/stream")
    rmt._reset_subscribe_allowlist_cache()
    m = _stub_mesh()
    m.on_stream.return_value = "stream-peer-b"
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("watch", target="peer-b")
    assert r["status"] == "success"
    m.on_stream.assert_called_once_with("peer-b")


def test_watch_allowed_when_in_hitl_set_and_approved(monkeypatch):
    """If watch is in STRANDS_MESH_HITL_ACTIONS and the operator approved the
    interrupt, the allowlist is bypassed (same semantics as subscribe)."""
    monkeypatch.setenv("STRANDS_MESH_HITL_ACTIONS", "watch")
    rmt._reset_interrupt_actions_cache()
    m = _stub_mesh()
    m.on_stream.return_value = "stream-peer-b"
    ctx = _make_ctx("y")  # operator approves
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("watch", target="peer-b", ctx=ctx)
    assert r["status"] == "success"
    m.on_stream.assert_called_once_with("peer-b")


def test_watch_blocked_when_in_hitl_set_and_declined(monkeypatch):
    """If watch is in the HITL set but the operator declines, the call does
    not proceed (interrupt rejection takes precedence over allowlist)."""
    monkeypatch.setenv("STRANDS_MESH_HITL_ACTIONS", "watch")
    rmt._reset_interrupt_actions_cache()
    m = _stub_mesh()
    m.on_stream.return_value = "stream-peer-b"
    ctx = _make_ctx("n")  # operator declines
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("watch", target="peer-b", ctx=ctx)
    assert r["status"] == "error"
    m.on_stream.assert_not_called()


# --- single-star segment wildcard matcher tests ─────────────────────────


def test_ke_matches_single_star_segment():
    """Single * matches exactly one segment."""
    assert rmt._ke_matches("strands/*/stream", "strands/peer-b/stream") is True
    assert rmt._ke_matches("strands/*/stream", "strands/robot_1/stream") is True


def test_ke_matches_single_star_rejects_extra_segments():
    """Single * does not match multiple segments."""
    assert rmt._ke_matches("strands/*/stream", "strands/a/b/stream") is False


def test_ke_matches_single_star_rejects_empty_segment():
    """Single * does not match an empty segment (wrong segment count)."""
    assert rmt._ke_matches("strands/*/stream", "strands//stream") is False


# --- R7 must-fix: watch(target="*") wildcard-bypass regression ───────────


@pytest.mark.parametrize(
    "target",
    [
        "*",  # single Zenoh wildcard — matches strands/*/stream by equality
        "**",  # double-star wildcard — matches via trailing /** branch
        "*/state",  # embedded wildcard
        "peer-a/state",  # path separator (would broaden the keyexpr)
        "../etc",  # path traversal shape
        "",  # already covered by missing-target check, kept for completeness
    ],
)
def test_watch_rejects_wildcard_targets_even_with_permissive_allowlist(monkeypatch, target):
    """R7 must-fix regression: ``target`` is interpolated into
    ``strands/{target}/stream``, and a Zenoh wildcard segment in ``target``
    defeats per-peer scoping even when an operator extended the allowlist
    with ``strands/*/stream`` (the value the README example documents). The
    pre-fix code reached ``mesh.on_stream("*")`` and subscribed fleet-wide.

    Fails on pre-fix code: ``target="*"`` with ``STRANDS_MESH_SUBSCRIBE_ALLOW=
    strands/*/stream`` reached ``on_stream`` (cross-peer telemetry leak).
    """
    monkeypatch.setenv("STRANDS_MESH_SUBSCRIBE_ALLOW", "strands/*/stream")
    rmt._reset_subscribe_allowlist_cache()
    m = _stub_mesh()
    m.on_stream.return_value = "stream-x"
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("watch", target=target)
    # Empty target hits the missing-target branch first; non-empty wildcards
    # hit the new peer-id shape check. Either way, error + on_stream silent.
    assert r["status"] == "error", f"watch must reject target={target!r}"
    m.on_stream.assert_not_called()


def test_watch_rejects_wildcard_even_when_in_hitl_set_and_approved(monkeypatch):
    """Even an HITL approval cannot legitimise a wildcard target — the HITL
    bypass is for the allowlist gate, not for shape validation. Otherwise the
    operator would be tricked into approving a single-peer-shaped reason
    string while the agent slipped a wildcard past."""
    monkeypatch.setenv("STRANDS_MESH_HITL_ACTIONS", "watch")
    monkeypatch.setenv("STRANDS_MESH_SUBSCRIBE_ALLOW", "strands/*/stream")
    rmt._reset_interrupt_actions_cache()
    rmt._reset_subscribe_allowlist_cache()
    m = _stub_mesh()
    m.on_stream.return_value = "stream-x"
    ctx = _make_ctx("y")  # operator approves
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("watch", target="*", ctx=ctx)
    assert r["status"] == "error"
    m.on_stream.assert_not_called()


@pytest.mark.parametrize(
    "peer",
    [
        "peer-b",
        "arm-a",
        "robot_1",
        "robot.local",
        "reachy",
        "AB12",
        "z" * 64,  # max length boundary
    ],
)
def test_watch_accepts_literal_peer_ids_with_extended_allowlist(monkeypatch, peer):
    """Confirm the literal-peer-id check does not over-block legitimate ids
    (alphanumerics + ``._-``, max 64 chars)."""
    monkeypatch.setenv("STRANDS_MESH_SUBSCRIBE_ALLOW", "strands/*/stream")
    rmt._reset_subscribe_allowlist_cache()
    m = _stub_mesh()
    m.on_stream.return_value = f"stream-{peer}"
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("watch", target=peer)
    assert r["status"] == "success", f"watch should accept literal peer id {peer!r}"
    m.on_stream.assert_called_once_with(peer)


def test_watch_rejects_overlong_peer_id(monkeypatch):
    """Peer ids longer than 64 chars are rejected (Zenoh practical limit)."""
    monkeypatch.setenv("STRANDS_MESH_SUBSCRIBE_ALLOW", "strands/*/stream")
    rmt._reset_subscribe_allowlist_cache()
    m = _stub_mesh()
    m.on_stream.return_value = "stream-x"
    with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
        r = _call("watch", target="z" * 65)
    assert r["status"] == "error"
    m.on_stream.assert_not_called()


def test_watch_rejects_leading_special_char(monkeypatch):
    """First char must be alphanumeric (no leading ``-``, ``.``, ``_``).
    Prevents argument-injection-shaped ids reaching downstream Zenoh APIs."""
    monkeypatch.setenv("STRANDS_MESH_SUBSCRIBE_ALLOW", "strands/*/stream")
    rmt._reset_subscribe_allowlist_cache()
    m = _stub_mesh()
    m.on_stream.return_value = "stream-x"
    for bad in ("-rm", ".dot", "_under", "/abs"):
        with patch("strands_robots.tools.robot_mesh._resolve_mesh", return_value=m):
            r = _call("watch", target=bad)
        assert r["status"] == "error", f"watch must reject leading-special target {bad!r}"
    m.on_stream.assert_not_called()
