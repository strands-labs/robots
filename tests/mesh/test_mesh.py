"""Tests for strands_robots.mesh — Mesh component, presence + state loops.

All tests are 100% mocked.  No ``eclipse-zenoh`` install required: a
``MagicMock`` session is injected in place of the real Zenoh session by
patching :func:`strands_robots.mesh_session.get_session`.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from strands_robots import mesh as mesh_mod
from strands_robots.mesh import Mesh, get_local_robots, init_mesh
from strands_robots.mesh import session as mesh_session

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeRobot:
    """Minimal duck-typed robot used to drive presence/state introspection."""

    def __init__(
        self,
        tool_name: str = "fakebot",
        with_world: bool = False,
        with_task: bool = False,
        with_inner_robot: bool = False,
        with_action_features: bool = False,
    ) -> None:
        self.tool_name_str = tool_name
        if with_world:
            self._world = MagicMock()
            self._world._data.time = 12.5
            self._world.robots = {"arm0": object(), "arm1": object()}
        if with_task:
            ts = MagicMock()
            ts.status.value = "running"
            ts.instruction = "pick the cube"
            ts.step_count = 7
            ts.duration = 1.25
            self._task_state = ts
        if with_inner_robot:
            inner = MagicMock()
            inner.is_connected = True
            inner.name = "so100"
            inner.config.cameras = {"wrist_cam": object()}
            inner.get_observation.return_value = {
                "joint_0": MagicMock(tolist=lambda: [0.1, 0.2]),
                "joint_1": 0.5,
                "wrist_cam": MagicMock(shape=(3, 240, 320)),  # excluded by cam_keys
                "depth": MagicMock(shape=(240, 320), tolist=lambda: [[0]]),  # 2D - excluded by shape
            }
            self.robot = inner
        if with_action_features:
            self._action_features = {"action_0": float, "action_1": float}


@pytest.fixture(autouse=True)
def _clean_state() -> Iterator[None]:
    """Reset peer registry and local-robot registry between tests."""
    mesh_session.clear_peers()
    with mesh_mod._LOCAL_ROBOTS_LOCK:
        mesh_mod._LOCAL_ROBOTS.clear()
    yield
    mesh_session.clear_peers()
    with mesh_mod._LOCAL_ROBOTS_LOCK:
        mesh_mod._LOCAL_ROBOTS.clear()


@pytest.fixture
def fake_session() -> MagicMock:
    """A mock zenoh session that returns mock subscribers from declare_subscriber."""
    session = MagicMock()
    session.declare_subscriber.return_value = MagicMock()
    return session


@pytest.fixture
def patch_session(fake_session: MagicMock, monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    """Patch get_session in the mesh module so start() succeeds.

    Also clears the ``STRANDS_MESH`` env var (set by ``tests/conftest.py``
    for the rest of the suite) so :func:`init_mesh` is not short-circuited
    by the global kill switch.
    """
    monkeypatch.delenv("STRANDS_MESH", raising=False)
    with patch("strands_robots.mesh.core.get_session", return_value=fake_session):
        # Also stop start() from acquiring a real session ref count by
        # patching release_session to a no-op MagicMock so we can assert calls.
        with patch("strands_robots.mesh.core.release_session") as release_mock:
            yield release_mock


@pytest.fixture
def patch_no_session() -> Iterator[None]:
    """Patch get_session to return None (zenoh unavailable)."""
    with patch("strands_robots.mesh.core.get_session", return_value=None):
        yield


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    """start() / stop() / alive — basic lifecycle invariants."""

    def test_starts_and_alive(self, patch_session: MagicMock) -> None:
        m = Mesh(_FakeRobot(), peer_id="bot-1")
        assert m.alive is False
        m.start()
        assert m.alive is True
        m.stop()
        assert m.alive is False

    def test_start_idempotent(self, patch_session: MagicMock, fake_session: MagicMock) -> None:
        """Two start() calls should not double-subscribe or double-acquire."""
        m = Mesh(_FakeRobot(), peer_id="bot-2")
        m.start()
        first_calls = fake_session.declare_subscriber.call_count
        m.start()  # second call
        assert fake_session.declare_subscriber.call_count == first_calls
        m.stop()

    def test_stop_idempotent(self, patch_session: MagicMock) -> None:
        m = Mesh(_FakeRobot(), peer_id="bot-3")
        m.start()
        m.stop()
        m.stop()  # must not raise
        assert m.alive is False

    def test_stop_without_start_is_noop(self, patch_session: MagicMock) -> None:
        """stop() before start() must not call release_session()."""
        m = Mesh(_FakeRobot(), peer_id="bot-4")
        m.stop()
        # release_session is the patched MagicMock from patch_session fixture
        assert patch_session.call_count == 0

    def test_start_when_zenoh_unavailable(self, patch_no_session: None) -> None:
        """When get_session() returns None, start() is a clean no-op."""
        m = Mesh(_FakeRobot(), peer_id="bot-5")
        m.start()
        assert m.alive is False
        # No registration in _LOCAL_ROBOTS
        assert "bot-5" not in get_local_robots()

    def test_subscribers_undeclared_on_stop(self, patch_session: MagicMock, fake_session: MagicMock) -> None:
        m = Mesh(_FakeRobot(), peer_id="bot-6")
        m.start()
        # Capture the subscribers created during start().
        subs = fake_session.declare_subscriber.return_value
        m.stop()
        assert subs.undeclare.called

    def test_release_session_called_on_stop(self, patch_session: MagicMock) -> None:
        m = Mesh(_FakeRobot(), peer_id="bot-7")
        m.start()
        m.stop()
        # release_session was patched as a MagicMock and recorded the call.
        assert patch_session.call_count == 1

    def test_stop_after_start_failure_does_not_double_release(self, fake_session: MagicMock) -> None:
        """If declare_subscriber raises, start() must release_session()
        immediately and stop() must not release a second time."""
        fake_session.declare_subscriber.side_effect = RuntimeError("boom")
        with patch("strands_robots.mesh.core.get_session", return_value=fake_session):
            with patch("strands_robots.mesh.core.release_session") as release_mock:
                m = Mesh(_FakeRobot(), peer_id="bot-8")
                m.start()
                # start() failed cleanly
                assert m.alive is False
                # release_session called exactly once during start() rollback
                assert release_mock.call_count == 1
                # stop() now is a no-op — must NOT call release_session again
                m.stop()
                assert release_mock.call_count == 1


# ---------------------------------------------------------------------------
# _LOCAL_ROBOTS registry
# ---------------------------------------------------------------------------


class TestLocalRegistry:
    """Mesh start/stop registers and unregisters with the in-process registry."""

    def test_registered_on_start(self, patch_session: MagicMock) -> None:
        m = Mesh(_FakeRobot(), peer_id="reg-1")
        m.start()
        assert "reg-1" in get_local_robots()
        m.stop()
        assert "reg-1" not in get_local_robots()

    def test_get_local_robots_returns_copy(self, patch_session: MagicMock) -> None:
        """Mutating the result must not affect the underlying registry."""
        m = Mesh(_FakeRobot(), peer_id="reg-2")
        m.start()
        snap = get_local_robots()
        snap.clear()
        # original registry untouched
        assert "reg-2" in get_local_robots()
        m.stop()


# ---------------------------------------------------------------------------
# Presence
# ---------------------------------------------------------------------------


class TestBuildPresence:
    """_build_presence enriches the payload from whatever the robot exposes."""

    def test_minimal(self) -> None:
        m = Mesh(_FakeRobot(tool_name="x"), peer_id="x-1", peer_type="robot")
        p = m._build_presence()
        assert p["robot_id"] == "x-1"
        assert p["robot_type"] == "robot"
        assert p["tool_name"] == "x"
        assert "hostname" in p
        assert "timestamp" in p

    def test_with_task_state(self) -> None:
        m = Mesh(_FakeRobot(with_task=True), peer_id="t-1")
        p = m._build_presence()
        assert p["task_status"] == "running"
        assert p["instruction"] == "pick the cube"

    def test_with_inner_hardware(self) -> None:
        m = Mesh(_FakeRobot(with_inner_robot=True), peer_id="hw-1")
        p = m._build_presence()
        assert p["connected"] is True
        assert p["hw"] == "so100"

    def test_with_action_features(self) -> None:
        m = Mesh(_FakeRobot(with_action_features=True), peer_id="af-1")
        p = m._build_presence()
        assert sorted(p["action_keys"]) == ["action_0", "action_1"]

    def test_with_world(self) -> None:
        m = Mesh(_FakeRobot(with_world=True), peer_id="sim-1", peer_type="sim")
        p = m._build_presence()
        assert p["world"] is True
        assert sorted(p["sim_robots"]) == ["arm0", "arm1"]

    def test_robot_attribute_errors_swallowed(self) -> None:
        """A robot whose attributes raise on access must not break presence."""

        class BrokenRobot:
            tool_name_str = "broken"

            @property
            def _task_state(self) -> Any:
                raise RuntimeError("no task state for you")

            @property
            def _world(self) -> Any:
                raise ValueError("no world either")

        m = Mesh(BrokenRobot(), peer_id="br-1")
        p = m._build_presence()  # must not raise
        assert p["robot_id"] == "br-1"
        assert p["tool_name"] == "broken"


class TestOnPresence:
    """_on_presence updates the peer registry, ignoring self and bad payloads."""

    def _make_sample(self, payload: dict[str, Any] | bytes) -> MagicMock:
        sample = MagicMock()
        if isinstance(payload, bytes):
            sample.payload.to_bytes.return_value = payload
        else:
            sample.payload.to_bytes.return_value = json.dumps(payload).encode()
        return sample

    def test_updates_peer_registry(self) -> None:
        m = Mesh(_FakeRobot(), peer_id="self")
        m._on_presence(self._make_sample({"robot_id": "other", "robot_type": "sim", "hostname": "h"}))
        peers = mesh_session.get_peers()
        assert any(p["peer_id"] == "other" for p in peers)

    def test_ignores_self_report(self) -> None:
        m = Mesh(_FakeRobot(), peer_id="me")
        m._on_presence(self._make_sample({"robot_id": "me"}))
        peers = mesh_session.get_peers()
        assert all(p["peer_id"] != "me" for p in peers)

    def test_ignores_invalid_json(self) -> None:
        m = Mesh(_FakeRobot(), peer_id="me")
        m._on_presence(self._make_sample(b"not a json{"))  # must not raise
        assert mesh_session.peer_count() == 0

    def test_ignores_missing_robot_id(self) -> None:
        m = Mesh(_FakeRobot(), peer_id="me")
        m._on_presence(self._make_sample({"hostname": "no_id"}))
        assert mesh_session.peer_count() == 0

    def test_ignores_non_string_robot_id(self) -> None:
        m = Mesh(_FakeRobot(), peer_id="me")
        m._on_presence(self._make_sample({"robot_id": 12345}))
        assert mesh_session.peer_count() == 0


class TestPeersProperty:
    """Mesh.peers returns the session peer list filtered to exclude self."""

    def test_excludes_self(self) -> None:
        m = Mesh(_FakeRobot(), peer_id="self-id")
        # Force-inject self into the session registry.
        mesh_session.update_peer("self-id", "robot", "h", {"robot_id": "self-id"})
        mesh_session.update_peer("other-id", "sim", "h", {"robot_id": "other-id"})
        ids = [p["peer_id"] for p in m.peers]
        assert "self-id" not in ids
        assert "other-id" in ids


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class TestReadState:
    """_read_state filters out images, surfaces joint and sim data."""

    def test_returns_none_when_no_useful_state(self) -> None:
        m = Mesh(_FakeRobot(), peer_id="empty")
        assert m._read_state() is None

    def test_publishes_joints_filtered_against_cameras(self) -> None:
        m = Mesh(_FakeRobot(with_inner_robot=True), peer_id="hw")
        s = m._read_state()
        assert s is not None
        assert "joints" in s
        assert "joint_0" in s["joints"]
        assert "joint_1" in s["joints"]
        # camera key excluded
        assert "wrist_cam" not in s["joints"]
        # 2D depth array also excluded by shape
        assert "depth" not in s["joints"]
        # tolist() applied to numpy-like
        assert s["joints"]["joint_0"] == [0.1, 0.2]

    def test_publishes_task_progress(self) -> None:
        m = Mesh(_FakeRobot(with_task=True), peer_id="t")
        s = m._read_state()
        assert s is not None
        assert s["task"]["status"] == "running"
        assert s["task"]["steps"] == 7
        assert s["task"]["instruction"] == "pick the cube"

    def test_publishes_sim_clock(self) -> None:
        m = Mesh(_FakeRobot(with_world=True), peer_id="sim")
        s = m._read_state()
        assert s is not None
        assert s["sim_time"] == 12.5
        assert "robots" in s
        assert sorted(s["robots"].keys()) == ["arm0", "arm1"]


# ---------------------------------------------------------------------------
# Heartbeat / state loops
# ---------------------------------------------------------------------------


class TestLoops:
    """Heartbeat and state loops publish to the right keys and can be stopped."""

    def test_heartbeat_publishes_and_prunes(self, patch_session: MagicMock) -> None:
        with (
            patch("strands_robots.mesh.core.put") as put_mock,
            patch("strands_robots.mesh.core.prune_peers") as prune_mock,
        ):
            m = Mesh(_FakeRobot(), peer_id="hb-1")
            m.start()
            # Wait for at least one heartbeat tick (>500ms at 2Hz).
            time.sleep(0.7)
            m.stop()

            # At least one presence publish.
            keys = [c.args[0] for c in put_mock.call_args_list]
            assert any(k == "strands/hb-1/presence" for k in keys)
            # prune_peers called at least once.
            assert prune_mock.called

    def test_state_loop_publishes_when_state_present(self, patch_session: MagicMock) -> None:
        with patch("strands_robots.mesh.core.put") as put_mock:
            m = Mesh(_FakeRobot(with_world=True), peer_id="st-1", peer_type="sim")
            m.start()
            time.sleep(0.3)  # >100ms at 10Hz → at least one state tick
            m.stop()
            keys = [c.args[0] for c in put_mock.call_args_list]
            assert any(k == "strands/st-1/state" for k in keys)

    def test_state_loop_skips_publish_when_no_state(self, patch_session: MagicMock) -> None:
        with patch("strands_robots.mesh.core.put") as put_mock:
            # _FakeRobot() with nothing → _read_state returns None → no state publish
            m = Mesh(_FakeRobot(), peer_id="st-2")
            m.start()
            time.sleep(0.3)
            m.stop()
            keys = [c.args[0] for c in put_mock.call_args_list]
            # presence still published, state never
            assert "strands/st-2/state" not in keys

    def test_loop_survives_publish_error(self, patch_session: MagicMock) -> None:
        """If put() raises, the loop logs and keeps going."""
        with patch("strands_robots.mesh.put", side_effect=RuntimeError("net down")):
            m = Mesh(_FakeRobot(), peer_id="err-1")
            m.start()
            time.sleep(0.6)
            assert m.alive is True  # loop did not crash mesh
            m.stop()


# ---------------------------------------------------------------------------
# init_mesh — the public constructor
# ---------------------------------------------------------------------------


class TestInitMesh:
    """init_mesh is the only blessed way to build a Mesh."""

    def test_returns_none_when_mesh_arg_false(self, patch_session: MagicMock) -> None:
        assert init_mesh(_FakeRobot(), peer_id="x", mesh=False) is None

    def test_returns_none_when_env_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH", "false")
        with patch("strands_robots.mesh.core.get_session", return_value=MagicMock()):
            assert init_mesh(_FakeRobot(), peer_id="x") is None

    def test_env_kill_switch_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH", "  FALSE  ")
        with patch("strands_robots.mesh.core.get_session", return_value=MagicMock()):
            assert init_mesh(_FakeRobot(), peer_id="x") is None

    def test_returns_started_mesh(self, patch_session: MagicMock) -> None:
        m = init_mesh(_FakeRobot(), peer_id="im-1")
        assert m is not None
        assert isinstance(m, Mesh)
        assert m.alive is True
        m.stop()

    def test_default_peer_id_format(self, patch_session: MagicMock) -> None:
        """Default peer_id is f'{tool_name_str}-{8 hex chars}'."""
        m = init_mesh(_FakeRobot(tool_name="bot"), peer_id=None)
        assert m is not None
        assert m.peer_id.startswith("bot-")
        # 8 hex chars (32 bits) — sized to avoid collisions on a busy mesh.
        assert len(m.peer_id) == len("bot-") + 8
        # Suffix is hex.
        assert all(c in "0123456789abcdef" for c in m.peer_id.split("-")[-1])
        m.stop()

    def test_default_peer_id_when_no_tool_name(self, patch_session: MagicMock) -> None:
        class NoName:
            pass

        m = init_mesh(NoName(), peer_id=None)
        assert m is not None
        assert m.peer_id.startswith("robot-")
        m.stop()


# ---------------------------------------------------------------------------
# Two meshes share the session (refcount semantics)
# ---------------------------------------------------------------------------


class TestSessionSharing:
    """Two Mesh instances in the same process share one session reference."""

    def test_both_register_and_unregister_independently(self, patch_session: MagicMock) -> None:
        a = Mesh(_FakeRobot(tool_name="a"), peer_id="a-1")
        b = Mesh(_FakeRobot(tool_name="b"), peer_id="b-1")
        a.start()
        b.start()
        assert "a-1" in get_local_robots()
        assert "b-1" in get_local_robots()
        a.stop()
        assert "a-1" not in get_local_robots()
        assert "b-1" in get_local_robots()
        b.stop()
        assert "b-1" not in get_local_robots()

    def test_release_session_called_per_stop(self, patch_session: MagicMock) -> None:
        """Each successful stop() calls release_session() exactly once."""
        a = Mesh(_FakeRobot(), peer_id="a-2")
        b = Mesh(_FakeRobot(), peer_id="b-2")
        a.start()
        b.start()
        a.stop()
        b.stop()
        # release_session called twice total (once per Mesh.stop()).
        assert patch_session.call_count == 2


# ---------------------------------------------------------------------------
# Concurrent stress
# ---------------------------------------------------------------------------


class TestConcurrentLifecycle:
    """Stress test: many start/stop cycles from threads, no deadlock or leaks."""

    def test_repeated_start_stop_from_threads(self, patch_session: MagicMock) -> None:
        m = Mesh(_FakeRobot(), peer_id="stress-1")

        def worker() -> None:
            for _ in range(20):
                m.start()
                m.stop()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
        # No leaked registration.
        assert "stress-1" not in get_local_robots()
        assert m.alive is False
