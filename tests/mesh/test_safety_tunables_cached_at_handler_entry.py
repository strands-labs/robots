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

The static half of that pin originally graded one lock (``_estop_replay_lock``),
one method (``_on_safety_estop``) and the two float tunables. ``Mesh`` now keeps
three replay caches -- estop, resume and inbound-command dedup -- behind three
locks, and the eviction bound they all share is resolved by a third lazy
resolver, ``_resume_replay_cache_max``. The scan below therefore derives both the
lock set and the methods that take them from the class, so a fourth replay cache
is graded the hour it lands rather than inheriting the same gap.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
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


#: Every lazy env resolver that feeds a replay-cache critical section, plus the
#: two parsers behind them. Each is an ``os.getenv`` and a validating parse, and
#: on an unusable operator value it logs as well -- so the cost is not constant
#: and has no reason to be paid while holding a lock other peers wait on.
_BANNED_IN_LOCK = frozenset(
    {
        "_resume_forward_skew_s",
        "_resume_freshness_window_s",
        "_resume_replay_cache_max",
        "_parse_positive_float_env",
        "_parse_positive_int_env",
    }
)


def _class_tree(cls) -> ast.AST:
    """Parse *cls* from source, dedented so :mod:`ast` accepts it standalone."""
    return ast.parse(textwrap.dedent(inspect.getsource(cls)))


def _replay_lock_names(cls) -> set[str]:
    """Return the ``*_replay_lock`` attributes *cls* assigns to itself.

    Derived from the constructor rather than listed, so a replay cache added
    later is graded without editing this file.
    """
    names: set[str] = set()
    for node in ast.walk(_class_tree(cls)):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr.endswith("_replay_lock")
            ):
                names.add(target.attr)
    return names


def _env_reads_inside_replay_locks(cls) -> list[str]:
    """Report every banned resolver call made inside a replay-cache lock body.

    Args:
        cls: The class to scan, read from source.

    Returns:
        One human-readable entry per offending call, empty when every replay
        lock body is free of environment parsing.
    """
    locks = _replay_lock_names(cls)
    offenders: list[str] = []
    for method in ast.walk(_class_tree(cls)):
        if not isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for block in ast.walk(method):
            if not isinstance(block, ast.With):
                continue
            held = {
                item.context_expr.attr
                for item in block.items
                if isinstance(item.context_expr, ast.Attribute) and item.context_expr.attr in locks
            }
            if not held:
                continue
            for inner in ast.walk(block):
                if not isinstance(inner, ast.Call):
                    continue
                func = inner.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name in _BANNED_IN_LOCK:
                    offenders.append(f"{method.name} holds {sorted(held)} and calls {name}() (line {inner.lineno})")
    return offenders


def test_the_scan_reaches_every_replay_cache_lock() -> None:
    """Non-vacuity: a clean report must mean the scan found the locks.

    Without this, collapsing the lock discovery to an empty set would report a
    clean tree while grading nothing.
    """
    locks = _replay_lock_names(core.Mesh)
    assert len(locks) >= 3, f"expected at least three replay-cache locks, found {sorted(locks)}"
    tree = _class_tree(core.Mesh)
    takers = {
        method.name
        for method in ast.walk(tree)
        if isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef)
        for block in ast.walk(method)
        if isinstance(block, ast.With)
        for item in block.items
        if isinstance(item.context_expr, ast.Attribute) and item.context_expr.attr in locks
    }
    assert len(takers) >= 3, f"expected at least three methods taking a replay lock, found {sorted(takers)}"


def test_no_replay_cache_lock_body_parses_the_environment() -> None:
    """No replay-cache critical section resolves an env tunable (issue #265)."""
    offenders = _env_reads_inside_replay_locks(core.Mesh)
    assert not offenders, (
        "a replay-cache lock body must not parse the environment -- resolve the "
        "tunable into a local before taking the lock:\n  " + "\n  ".join(offenders)
    )


class _WatchingLock:
    """A real lock that records which resolvers are called while it is held."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.entered = 0
        self.held = False
        self.calls: list[str] = []

    def __enter__(self) -> _WatchingLock:
        self._lock.acquire()
        self.entered += 1
        self.held = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.held = False
        self._lock.release()


def _watch_resolvers(monkeypatch, watch: _WatchingLock) -> None:
    """Record any :data:`_BANNED_IN_LOCK` resolver call made while *watch* is held."""
    for name in ("_resume_forward_skew_s", "_resume_freshness_window_s", "_resume_replay_cache_max"):
        real = getattr(core, name)

        def spy(_real=real, _name=name):
            if watch.held:
                watch.calls.append(_name)
            return _real()

        monkeypatch.setattr(core, name, spy)


class _FakeRobot:
    """Minimal robot adapter: enough surface for an actuating command."""

    def status(self) -> dict:
        return {"status": "idle"}

    def stop_task(self) -> dict:
        return {"ok": True}


def test_command_dedup_lock_holds_no_env_parse(monkeypatch) -> None:
    """The inbound-command replay lock is held without parsing the environment.

    ``_exec_cmd`` dedups actuating commands through a third replay cache added
    after issue #265. It resolves the same tunables the safety handlers do, so it
    must resolve them the same way: into locals, before the lock.
    """
    mesh = core.Mesh(_FakeRobot(), peer_id="robot-a")
    monkeypatch.setattr(mesh, "publish", lambda key, payload, **kw: None)
    monkeypatch.setattr(core, "log_safety_event", lambda event_type, peer_id, detail: None)

    watch = _WatchingLock()
    monkeypatch.setattr(mesh, "_cmd_replay_lock", watch)
    _watch_resolvers(monkeypatch, watch)

    mesh._exec_cmd({"sender_id": "op1", "turn_id": "t1", "command": {"action": "stop"}})

    assert watch.entered >= 1, "premise: the command must reach the dedup critical section"
    assert not watch.calls, f"env tunables parsed while holding _cmd_replay_lock: {watch.calls}"


def test_estop_replay_lock_holds_no_env_parse(monkeypatch) -> None:
    """Nothing under the estop lock reads the environment, bound included.

    Issue #265 hoisted the two float tunables out of this critical section but
    left the eviction bound resolved inside it, so the handler paid an
    ``os.getenv`` and a parse under the lock it had just been cleared of. The
    bound is resolved beside ``per_issuer_cap`` before the lock now.
    """
    mesh = _stub_mesh()
    watch = _WatchingLock()
    monkeypatch.setattr(mesh, "_estop_replay_lock", watch)
    _watch_resolvers(monkeypatch, watch)

    mesh._on_safety_estop(_envelope(time.time()))

    assert watch.entered >= 1, "premise: the envelope must reach the estop critical section"
    assert not watch.calls, f"env tunables parsed while holding _estop_replay_lock: {watch.calls}"
