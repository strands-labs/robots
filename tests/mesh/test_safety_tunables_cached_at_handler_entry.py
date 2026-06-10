"""Pin: safety handlers read freshness/skew tunables once at handler entry.

Issue #265. ``_resume_forward_skew_s`` and ``_resume_freshness_window_s``
each parse ``os.getenv`` plus a regex validation on every call.
``_on_safety_estop`` referenced them 5-6 times per envelope, and two of
those reads happened inside the ``_estop_replay_lock`` critical section,
extending the lock and exposing a mid-handler env-mutation inconsistency
window against the timing-sensitive corroboration check.

The fix caches both values into locals immediately after the
``isinstance(data, dict)`` guard and uses those locals for the rest of
the method. The next envelope still re-reads the env, so the operator
tunable contract is preserved.
"""

from __future__ import annotations

import ast
import inspect
import json
import threading
import time
from types import SimpleNamespace

from strands_robots.mesh import core


def _stub_mesh() -> core.Mesh:
    m = core.Mesh.__new__(core.Mesh)
    m.peer_id = "test-peer"
    m._estop_replay_cache = {}
    m._resume_replay_cache = {}
    m._estop_replay_lock = threading.Lock()
    m._resume_replay_lock = threading.Lock()
    m._estop_lockout = threading.Event()
    m._last_estop_ts = 0.0
    m._last_estop_mono = 0.0
    m.publish_safety_event = lambda **kw: None  # type: ignore[method-assign]
    return m


def _envelope(t: float, peer_id: str = "issuer", **extra):
    body = {"peer_id": peer_id, "t": t, **extra}
    raw = json.dumps(body).encode()
    return SimpleNamespace(payload=SimpleNamespace(to_bytes=lambda r=raw: r))


def test_tunables_read_once_per_handler_invocation(monkeypatch):
    """A single estop dispatch reads each tunable at most once."""
    calls: dict[str, int] = {"skew": 0, "fresh": 0}
    real_skew = core._resume_forward_skew_s
    real_fresh = core._resume_freshness_window_s

    def counting_skew():
        calls["skew"] += 1
        return real_skew()

    def counting_fresh():
        calls["fresh"] += 1
        return real_fresh()

    monkeypatch.setattr(core, "_resume_forward_skew_s", counting_skew)
    monkeypatch.setattr(core, "_resume_freshness_window_s", counting_fresh)

    mesh = _stub_mesh()
    mesh._on_safety_estop(_envelope(time.time()))

    assert calls["skew"] <= 1, f"forward_skew tunable read {calls['skew']} times, expected <= 1"
    assert calls["fresh"] <= 1, f"freshness tunable read {calls['fresh']} times, expected <= 1"


def test_env_change_mid_handler_does_not_affect_current_envelope(monkeypatch):
    """An env mutation after handler entry must not change the value used by that envelope."""
    seen: list[float] = []
    call_index = {"n": 0}

    def first_read_then_mutate():
        # First (and only legitimate) read returns 60; if the handler
        # erroneously re-reads, the env will have been mutated to 1 and
        # the second read would return 1.
        call_index["n"] += 1
        if call_index["n"] == 1:
            monkeypatch.setenv("STRANDS_MESH_RESUME_FRESHNESS_S", "1")
            value = 60.0
        else:
            value = core._parse_positive_float_env("STRANDS_MESH_RESUME_FRESHNESS_S", "60")
        seen.append(value)
        return value

    monkeypatch.setattr(core, "_resume_freshness_window_s", first_read_then_mutate)

    mesh = _stub_mesh()
    # Envelope is 30s old: passes against the cached 60s window but would
    # FAIL against the mutated 1s window if the handler re-read mid-flight.
    mesh._on_safety_estop(_envelope(time.time() - 30.0))

    assert mesh._estop_lockout.is_set(), (
        "envelope must engage lockout using the freshness window cached at entry; "
        "a mid-handler env re-read to 1s would have rejected this 30s-old envelope"
    )
    assert seen == [60.0], f"freshness window must be read exactly once at entry; reads={seen}"


def _lock_acquired_without_env_reads(method) -> bool:
    """Static check: no _estop_replay_lock ``with`` body contains a call
    to a tunable reader (``_resume_forward_skew_s`` /
    ``_resume_freshness_window_s``) or to ``_parse_positive_float_env``.
    """
    source = inspect.getsource(method)
    # Dedent so ast can parse the standalone method.
    import textwrap

    tree = ast.parse(textwrap.dedent(source))
    banned = {
        "_resume_forward_skew_s",
        "_resume_freshness_window_s",
        "_parse_positive_float_env",
    }

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.violation = False

        def visit_With(self, node: ast.With):
            uses_estop_lock = any(
                isinstance(item.context_expr, ast.Attribute) and item.context_expr.attr == "_estop_replay_lock"
                for item in node.items
            )
            if uses_estop_lock:
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        fn = inner.func
                        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
                        if name in banned:
                            self.violation = True
            self.generic_visit(node)

    v = _Visitor()
    v.visit(tree)
    return not v.violation


def test_lock_held_duration_does_not_include_env_reads():
    """The _estop_replay_lock critical section must not parse env tunables."""
    assert _lock_acquired_without_env_reads(core.Mesh._on_safety_estop), (
        "_estop_replay_lock body must not call _resume_forward_skew_s / "
        "_resume_freshness_window_s / _parse_positive_float_env; cache them "
        "at handler entry (issue #265)"
    )
