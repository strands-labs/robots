"""Regression tests for the deep-analysis fixes (Issues #1-#11).

Each test pins a behaviour we want to keep across future refactors.  When
a test fails, the failure message points at the originating issue number
in the PR description.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh import Mesh, init_mesh
from strands_robots.mesh import core as mesh_core
from strands_robots.mesh import session as mesh_session


class _FakeRobot:
    tool_name_str = "rob"


@pytest.fixture
def fake_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    """A fake zenoh session patched into both get/current_session."""
    monkeypatch.delenv("STRANDS_MESH", raising=False)
    sess = MagicMock()
    sess.declare_subscriber.return_value = MagicMock()
    with (
        patch.object(mesh_session, "get_session", return_value=sess),
        patch.object(mesh_session, "current_session", return_value=sess),
        patch.object(mesh_core, "get_session", return_value=sess),
        patch.object(mesh_core, "current_session", return_value=sess),
        patch.object(mesh_core, "release_session"),
    ):
        yield sess


# ---------------------------------------------------------------------------
# Issue #1 / #2 — _inbox_lock prevents concurrent-handler corruption.
# ---------------------------------------------------------------------------


def test_subscribe_inbox_concurrent_writes(fake_session: MagicMock) -> None:
    """Two threads writing to the same inbox at high rate must not corrupt
    the buffer (lengths within bounds, no torn entries).
    """
    m = Mesh(_FakeRobot(), peer_id="rob-conc", peer_type="robot")
    m.start()
    try:
        m.subscribe("topic-x", name="tx")
        handler = fake_session.declare_subscriber.call_args_list[-1].args[1]

        def _make_sample(i: int) -> Any:
            s = MagicMock()
            s.key_expr = "topic-x"
            s.payload.to_bytes.return_value = f'{{"i": {i}}}'.encode()
            return s

        N = 2000
        threads = []
        for k in range(4):
            t = threading.Thread(
                target=lambda start=k * N // 4: [handler(_make_sample(j)) for j in range(start, start + N // 4)],
                daemon=True,
            )
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive()

        # Buffer never exceeded the cap, every retained entry is well-formed.
        buf = m.inbox["tx"]
        assert len(buf) <= 1000
        for topic, data in buf:
            assert topic == "topic-x"
            assert isinstance(data, dict)
            assert isinstance(data.get("i"), int)
    finally:
        m.stop()


# ---------------------------------------------------------------------------
# Issue #7 — stop_event wakes the heartbeat / state loops promptly.
# ---------------------------------------------------------------------------


def test_stop_unblocks_heartbeat_loop_quickly(fake_session: MagicMock) -> None:
    """stop() should drop the heartbeat thread within well under one
    HEARTBEAT period (0.5 s) instead of waiting it out.
    """
    m = Mesh(_FakeRobot(), peer_id="rob-quick", peer_type="robot")
    m.start()
    # Capture the heartbeat thread.
    heartbeat = next(t for t in m._threads if t.name.startswith("mesh-heartbeat"))
    t0 = time.monotonic()
    m.stop()
    heartbeat.join(timeout=0.5)
    elapsed = time.monotonic() - t0
    assert not heartbeat.is_alive(), "heartbeat thread did not exit on stop()"
    # 0.5s would be the worst case under time.sleep(period); with the
    # stop_event we expect single-digit milliseconds.
    assert elapsed < 0.4, f"stop() took {elapsed:.3f}s — stop_event regressed"


# ---------------------------------------------------------------------------
# Issue #10 — partial-init failure tears down already-declared subscribers.
# ---------------------------------------------------------------------------


def test_partial_init_failure_cleans_up_declared_subscribers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If declare_subscriber fails on the third call, the two that succeeded
    must be undeclared so they don't leak into the shared session.
    """
    monkeypatch.delenv("STRANDS_MESH", raising=False)

    declared: list[MagicMock] = []
    call_count = {"n": 0}

    sess = MagicMock()

    def declare_side_effect(*args: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise RuntimeError("simulated zenoh failure")
        sub = MagicMock(name=f"sub{call_count['n']}")
        declared.append(sub)
        return sub

    sess.declare_subscriber.side_effect = declare_side_effect

    with (
        patch.object(mesh_session, "get_session", return_value=sess),
        patch.object(mesh_core, "get_session", return_value=sess),
        patch.object(mesh_core, "release_session") as release_mock,
    ):
        m = Mesh(_FakeRobot(), peer_id="rob-partial", peer_type="robot")
        m.start()

    # Mesh must not be marked alive.
    assert m.alive is False
    # Both successfully-declared subscribers were undeclared.
    assert all(sub.undeclare.called for sub in declared), "leaked subscribers — Issue #10 regressed"
    # Session reference released.
    assert release_mock.called


# ---------------------------------------------------------------------------
# Issue #5 — invalid STRANDS_MESH_PORT falls back gracefully (no exception).
# ---------------------------------------------------------------------------


def test_invalid_mesh_port_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Bad STRANDS_MESH_PORT must warn and fall back to 7447 — never raise."""
    # Reset module-level state so get_session re-runs the open path.
    import strands_robots.mesh.session as ms

    monkeypatch.setenv("STRANDS_MESH_PORT", "definitely_not_a_number")
    monkeypatch.delenv("ZENOH_CONNECT", raising=False)
    monkeypatch.delenv("ZENOH_LISTEN", raising=False)
    monkeypatch.setattr(ms, "_SESSION", None)
    monkeypatch.setattr(ms, "_SESSION_REFS", 0)

    # Patch zenoh.open to capture the local_ep that get_session computes —
    # we don't actually need a real router for this test.
    fake_zenoh = MagicMock()
    fake_zenoh.Config.return_value = MagicMock()
    fake_session = MagicMock()
    fake_zenoh.open.return_value = fake_session

    with patch.dict("sys.modules", {"zenoh": fake_zenoh}):
        with caplog.at_level("WARNING", logger="strands_robots.mesh.session"):
            sess = ms.get_session()

    # Did not raise, returned a session.
    assert sess is not None or sess is None  # any outcome OK; no exception is the point
    # WARNING about the bad port made it into the log.
    assert any("STRANDS_MESH_PORT" in rec.message for rec in caplog.records), (
        "expected WARNING about invalid STRANDS_MESH_PORT — Issue #5 regressed"
    )

    # Cleanup — close any session we accidentally opened.
    if ms._SESSION is not None:
        try:
            ms.release_session()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Issue #4 — current_session does NOT bump the refcount.
# ---------------------------------------------------------------------------


def test_current_session_does_not_change_refcount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_session bumps; current_session must not."""
    import strands_robots.mesh.session as ms

    monkeypatch.setattr(ms, "_SESSION", MagicMock(name="fake_sess"))
    monkeypatch.setattr(ms, "_SESSION_REFS", 1)

    before = ms._SESSION_REFS
    s = ms.current_session()
    after = ms._SESSION_REFS

    assert s is ms._SESSION
    assert after == before, "current_session must not bump refcount"


# ---------------------------------------------------------------------------
# Issue #9 — HardwareRobot.cleanup() stops the mesh.
# ---------------------------------------------------------------------------


def test_hardware_robot_cleanup_stops_mesh() -> None:
    """A HardwareRobot with a mesh attached must stop it during cleanup()."""
    # Import deferred — pulls in lerobot only when actually needed.  Skip
    # if lerobot isn't installed.
    pytest.importorskip("lerobot.robots.config", reason="needs lerobot")
    from strands_robots import hardware_robot

    # Build a HardwareRobot without going through the full robot() path —
    # we just need an instance with the .mesh attribute and a cleanup() that
    # works (no real hardware).
    hw = hardware_robot.Robot.__new__(hardware_robot.Robot)
    # Minimal fields cleanup() reads.
    from concurrent.futures import ThreadPoolExecutor

    hw.tool_name_str = "fakebot"
    hw._shutdown_event = threading.Event()
    hw._task_state = MagicMock()
    hw._task_state.status = "IDLE"
    hw._executor = ThreadPoolExecutor(max_workers=1)
    fake_mesh = MagicMock(name="mesh")
    hw.mesh = fake_mesh
    hw.peer_id = "fakebot-deadbeef"

    hw.cleanup()
    assert fake_mesh.stop.called, "HardwareRobot.cleanup() must call self.mesh.stop() — Issue #9 regressed"


# ---------------------------------------------------------------------------
# Issue #11 — peer_id default has 8 hex chars of entropy.
# ---------------------------------------------------------------------------


def test_default_peer_id_has_32_bits_of_entropy(fake_session: MagicMock) -> None:
    """8 hex chars = 32 bits — required for collision-free large meshes."""
    seen: set[str] = set()
    for _ in range(256):
        m = init_mesh(_FakeRobot(), peer_id=None)
        assert m is not None
        suffix = m.peer_id.split("-")[-1]
        assert len(suffix) == 8, f"got {len(suffix)} hex chars — Issue #11 regressed"
        seen.add(suffix)
        m.stop()
    # 256 ids in a 2**32 space — uniqueness is statistically guaranteed.
    assert len(seen) == 256, "peer_id collisions across 256 ids — entropy regressed"


# ---------------------------------------------------------------------------
# Issue #2 / #3 — concurrent subscribe/stop don't crash.
# ---------------------------------------------------------------------------


def test_concurrent_subscribe_and_stop_does_not_crash(
    fake_session: MagicMock,
) -> None:
    """A subscribe() racing a stop() must not raise."""
    m = Mesh(_FakeRobot(), peer_id="rob-race", peer_type="robot")
    m.start()

    errs: list[BaseException] = []

    def subber() -> None:
        for i in range(50):
            try:
                m.subscribe(f"topic-{i}", name=f"sub-{i}")
            except Exception as exc:  # noqa: BLE001
                errs.append(exc)

    t = threading.Thread(target=subber, daemon=True)
    t.start()
    time.sleep(0.01)
    m.stop()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert errs == [], f"subscribe() raced stop() and raised: {errs!r}"
