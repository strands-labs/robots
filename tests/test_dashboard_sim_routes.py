"""Sim sessions, MJPEG, telemetry and the e-stop, on a fake engine so the routes are graded alone.

The fake answers the five engine calls the session uses (``robot_joint_names``,
``list_cameras``, ``mj_model.opt.timestep``, ``mj_data.time/qpos``, ``step``,
``get_frame``, ``reset``, ``set_joint_positions``, ``get_robot_state``) with
the shapes the MuJoCo engine returns. ``test_real_engine_*`` at the bottom
runs the same session on the real engine and is skipped without ``mujoco``.
"""

from __future__ import annotations

import importlib.util
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")

#: Only ``test_real_engine_*`` needs MuJoCo. Asking for it with
#: ``importorskip`` inside a decorator skips this whole module, because
#: that call raises at import time - so the flag is read instead.
_HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None

from fastapi.testclient import TestClient  # noqa: E402

from strands_robots.dashboard import routes_sim, settings, sim_session  # noqa: E402
from strands_robots.dashboard.server import create_app  # noqa: E402


class FakeEngine:
    def __init__(self, robot: str, joints: int = 3):
        self.robot = robot
        self.mj_model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        self.mj_data = SimpleNamespace(
            time=0.0, qpos=np.zeros(joints), geom_xpos=np.zeros((2, 3)), geom_xmat=np.tile(np.eye(3).ravel(), (2, 1))
        )
        self.steps = 0
        self.writes = 0
        self.holds: list = []
        self.resets = 0
        self.closed = False

    def robot_joint_names(self, robot):
        return [f"j{i}" for i in range(len(self.mj_data.qpos))]

    def list_cameras(self):
        return ["default"]

    def step(self, n=1):
        self.steps += n
        self.mj_data.time += n * 0.002
        self.mj_data.qpos = self.mj_data.qpos + 0.001 * n
        return {"status": "success", "content": [{"text": f"+{n}"}]}

    def get_frame(self, camera_name="default", width=None, height=None):
        return np.full((height or 8, width or 8, 3), 128, dtype=np.uint8), np.zeros((8, 8))

    def reset(self):
        self.resets += 1
        self.mj_data.time = 0.0
        self.mj_data.qpos = np.zeros_like(self.mj_data.qpos)
        return {"status": "success", "content": [{"text": "reset"}]}

    def set_joint_positions(self, positions, robot_name=None, hold=False):
        self.last_hold = hold
        if isinstance(positions, dict) and any(k not in self.robot_joint_names(robot_name) for k in positions):
            return {"status": "error", "content": [{"text": "unknown joint"}]}
        self.writes += 1
        self.holds.append(hold)
        return {"status": "success", "content": [{"text": "set"}]}

    def get_robot_state(self, robot_name=None):
        return {"status": "success", "content": [{"json": {"state": {}}}]}

    def close(self):
        self.closed = True


class ExplodingEngine:
    def __init__(self, robot):
        raise RuntimeError("no GL here")


@pytest.fixture()
def fake_factory(monkeypatch):
    made: list[FakeEngine] = []

    def factory(robot: str):
        e = FakeEngine(robot)
        made.append(e)
        return e

    monkeypatch.setattr(sim_session, "_default_factory", factory)  # looked up at call time
    return made


@pytest.fixture()
def client(tmp_path, monkeypatch, fake_factory):
    monkeypatch.setenv("STRANDS_DASH_AUTH_STORE", str(tmp_path / "auth.json"))
    monkeypatch.delenv("STRANDS_DASH_AUTH_ENABLED", raising=False)
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    settings.clear_overrides()
    settings.load(refresh=True)
    app = create_app()
    with TestClient(app) as c:
        yield c
    app.state.safety.store.shutdown()


def _until(predicate, timeout: float = 5.0) -> bool:
    """True as soon as *predicate* holds, so no cell sleeps a guess at a thread."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _create(client, robot="so101"):
    r = client.post("/api/sim", json={"robot": robot})
    assert r.status_code == 201, r.text
    return r.json()


class FakeModel:
    """Two geoms (a plane and a mesh), one 4-vertex / 2-face mesh - the fields scene.py reads."""

    ngeom, nmesh, ncam, nlight, nbody = 2, 1, 0, 1, 1
    geom_type = np.array([0, 7])
    geom_size = np.array([[5.0, 5.0, 0.01], [0.1, 0.1, 0.1]])
    geom_rgba = np.array([[0.5, 0.5, 0.5, 1.0], [1.0, 0.0, 0.0, 0.5]])
    geom_matid = np.array([-1, -1])
    geom_group = np.array([0, 3])
    geom_dataid = np.array([-1, 0])
    geom_bodyid = np.array([0, 0])
    mat_rgba = np.zeros((0, 4))
    mesh_vertadr = np.array([0])
    mesh_vertnum = np.array([4])
    mesh_faceadr = np.array([0])
    mesh_facenum = np.array([2])
    mesh_vert = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    mesh_face = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    cam_fovy = np.zeros(0)


# -- session -------------------------------------------------------------------


class TestSimSession:
    def test_starts_steps_and_renders(self, fake_factory):
        s = sim_session.SimSession("so101")
        assert s.wait_ready(5)
        time.sleep(0.25)
        snap = s.snapshot
        assert snap.state == "running"
        assert snap.joint_names == ("j0", "j1", "j2")
        assert snap.steps > 0 and snap.sim_time > 0
        assert s.latest_frame() is not None
        s.stop()
        assert s.snapshot.state == "stopped"
        assert fake_factory[0].closed

    def test_freeze_stops_stepping_but_keeps_answering(self, fake_factory):
        s = sim_session.SimSession("so101")
        s.wait_ready(5)
        time.sleep(0.1)
        s.freeze()
        time.sleep(0.1)
        t1 = s.snapshot.sim_time
        time.sleep(0.15)
        assert s.snapshot.sim_time == t1
        assert s.snapshot.state == "frozen"
        assert s.latest_frame() is not None, "a frozen robot is still drawn where it stopped"
        s.thaw()
        time.sleep(0.15)
        assert s.snapshot.sim_time > t1
        s.stop()

    def test_commands_run_on_the_worker(self, fake_factory):
        s = sim_session.SimSession("so101")
        s.wait_ready(5)
        assert s.command("reset")["status"] == "success"
        assert s.command("set_joints", positions={"j0": 0.1})["status"] == "success"
        assert fake_factory[0].last_hold is True, "servo setpoints must move with the pose"
        assert s.command("set_joints", positions={"zz": 0.1})["status"] == "error"
        assert s.command("bogus")["status"] == "error"
        s.stop()
        with pytest.raises(RuntimeError):
            s.command("reset")

    def test_a_motion_command_queued_while_frozen_is_refused_not_applied(self, fake_factory):
        """A write the worker had not reached yet does not run because the e-stop landed.

        The route gate refuses a command an operator sends after the stop; this is
        the other half - a command already on the queue when the freeze landed,
        which the worker would otherwise apply on its next pass, frozen or not.
        """
        s = sim_session.SimSession("so101")
        assert s.wait_ready(5)
        engine = fake_factory[0]
        s.freeze()

        refused = s.command("set_joints", positions={"j0": 0.4})
        assert refused["status"] == "error"
        assert "frozen by an e-stop" in refused["content"][0]["text"]
        assert engine.writes == 0, "the frozen session applied the write anyway"
        assert s.command("reset")["status"] == "error", "a reset moves the robot too"
        assert engine.resets == 0
        assert s.command("state")["status"] == "success", "a read moves nothing, so it is answered"

        s.thaw()
        assert s.command("set_joints", positions={"j0": 0.4})["status"] == "success"
        assert engine.writes == 1
        s.stop()

    def test_a_factory_failure_is_an_error_state_not_a_hang(self, monkeypatch):
        s = sim_session.SimSession("so101", engine_factory=ExplodingEngine)
        assert s.wait_ready(5)
        assert s.snapshot.state == "error"
        assert "no GL here" in (s.snapshot.error or "")

    def test_store_caps_live_sessions(self, fake_factory):
        store = sim_session.SessionStore(limit=2)
        a = store.create("so101")
        store.create("so101")
        with pytest.raises(RuntimeError, match="2 sessions"):
            store.create("so101")
        removed = store.remove(a.id)
        removed_again = store.remove(a.id)
        assert removed and not removed_again
        store.create("so101")  # room again
        store.shutdown()
        assert all(s.snapshot.state == "stopped" for s in store.all())


# -- routes --------------------------------------------------------------------


class TestSimRoutes:
    def test_create_get_delete(self, client):
        snap = _create(client)
        assert snap["state"] == "running" and snap["joint_names"] == ["j0", "j1", "j2"]
        assert client.get(f"/api/sim/{snap['id']}").json()["robot"] == "so101"
        assert client.get("/api/sim").json()["sessions"][0]["id"] == snap["id"]
        deleted = client.delete(f"/api/sim/{snap['id']}")
        gone = client.get(f"/api/sim/{snap['id']}")
        deleted_again = client.delete(f"/api/sim/{snap['id']}")
        assert (deleted.status_code, gone.status_code, deleted_again.status_code) == (200, 404, 404)

    def test_only_registry_robots_with_a_sim_asset(self, client):
        assert client.post("/api/sim", json={"robot": "not-a-robot"}).status_code == 400
        assert client.post("/api/sim", json={"robot": 3}).status_code == 400
        assert client.post("/api/sim", json=[]).status_code == 400

    @pytest.mark.parametrize(
        ("token", "reason"),
        [
            # The three bare tokens ``json.loads`` accepts and ``isinstance(v, float)`` admits.
            ("Infinity", "must be a finite number"),
            ("-Infinity", "must be a finite number"),
            ("NaN", "must be a finite number"),
            # A ``bool`` is an ``int``, so the type test admits it and
            # ``math.isfinite(True)`` is True: without the shared domain a
            # checkbox posted as a position was a 1 rad target the engine took.
            ("true", "must be a finite number"),
            # Finite, and past the float64 range: it needs a reason of its own,
            # because "must be a finite number" would be false of it - and
            # ``math.isfinite`` raises ``OverflowError`` on it rather than
            # answering, which is a 500 out of the guard that exists to answer.
            ("9" * 400, "must be within the range of a 64-bit float"),
            ('"0.2"', "must be a finite number"),
        ],
    )
    def test_a_value_that_is_not_a_joint_angle_is_refused_at_the_door(self, client, fake_factory, token, reason):
        """The route refuses what it cannot carry to a robot, in both body shapes.

        A position is a signed physical quantity passed verbatim to the engine,
        so the domain is ``utils.finite_number_error``'s - the one every surface
        that carries such a quantity shares. Each row states the reason it is
        answered with, and the reason names the joint that carried the value, so
        an operator is not left guessing which one of six was refused.
        """
        sid = _create(client)["id"]
        engine = fake_factory[0]
        for body, named in (
            (f'{{"positions": {{"j0": {token}}}}}', "positions['j0']"),
            # Not first in the list: the reason must name the index that
            # carried the value, which "positions[0]" would satisfy by accident.
            (f'{{"positions": [0.1, {token}, 0.2]}}', "positions[1]"),
        ):
            r = client.post(f"/api/sim/{sid}/joints", content=body, headers={"content-type": "application/json"})
            assert r.status_code == 400, body
            error = r.json()["error"]
            assert reason in error, error
            assert named in error, error
        assert engine.writes == 0, "a value that is not an angle reached the engine"

        # Both signs are angles: a joint turns either way, so the domain is the
        # signed one and a negative target is a pose, not a refusal.
        for good in (0.2, -0.2):
            ok = client.post(f"/api/sim/{sid}/joints", json={"positions": {"j0": good}})
            assert ok.status_code == 200, ok.text
        assert engine.writes == 2

    def test_an_alias_resolves_to_the_canonical_name(self, client):
        assert _create(client, "so-101")["robot"] == "so101"

    def test_engine_failure_is_500_and_the_session_is_not_kept(self, client, monkeypatch):
        monkeypatch.setattr(sim_session, "_default_factory", ExplodingEngine)
        r = client.post("/api/sim", json={"robot": "so101"})
        assert r.status_code == 500 and "no GL here" in r.json()["error"]
        assert client.get("/api/sim").json()["sessions"] == []

    def test_joints_validate_before_reaching_the_engine(self, client):
        sid = _create(client)["id"]
        assert client.post(f"/api/sim/{sid}/joints", json={"positions": {}}).status_code == 400
        assert client.post(f"/api/sim/{sid}/joints", json={"positions": {"j0": "x"}}).status_code == 400
        assert client.post(f"/api/sim/{sid}/joints", json={"positions": {"zz": 0.1}}).status_code == 400
        assert client.post(f"/api/sim/{sid}/joints", json={"positions": {"j0": 0.1}}).status_code == 200
        assert client.post(f"/api/sim/{sid}/reset").status_code == 200

    def test_joints_are_written_as_a_target_the_servos_hold(self, client, fake_factory):
        """The route says "set joint targets", so the write has to survive the next tick.

        ``set_joint_positions`` is a kinematic qpos write; on a robot held by
        position servos the servos are still commanded to their previous
        setpoint and the worker's next ``step`` pulls the pose back. Passing
        ``hold`` moves the setpoints with the pose. Without it the route answered
        200 for a pose the robot had left before the next telemetry frame.
        """
        sid = _create(client)["id"]
        assert client.post(f"/api/sim/{sid}/joints", json={"positions": {"j0": 0.5}}).status_code == 200
        (engine,) = fake_factory
        assert engine.holds == [True], "a joints request is a target, so the servo setpoints move with the pose"

    def test_stream_is_multipart_mjpeg_and_bounded_on_request(self, client):
        sid = _create(client)["id"]
        time.sleep(0.2)
        r = client.get(f"/api/sim/{sid}/stream.mjpg?frames=2")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
        assert r.content.count(b"Content-Type: image/jpeg") == 2
        assert client.get(f"/api/sim/{sid}/stream.mjpg?frames=0").status_code == 400

    def test_telemetry_websocket_carries_the_lockout(self, client):
        sid = _create(client)["id"]
        with client.websocket_connect(f"/ws/telemetry/{sid}") as ws:
            m = ws.receive_json()
        assert m["id"] == sid and m["lockout"]["state"] == "clear"
        assert len(m["qpos"]) == 3

    def test_telemetry_for_an_unknown_session_closes(self, client):
        """Accepted, then closed with 4404, which is what carries the code to the page."""
        from starlette.websockets import WebSocketDisconnect

        with client.websocket_connect("/ws/telemetry/nope") as ws, pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 4404

    def test_telemetry_from_another_origin_is_closed_before_it_is_accepted(self, client):
        """WebSockets have no CORS: a page anywhere can open ws://127.0.0.1:8090.
        Its Origin is the one thing it cannot hide, and admission reads it."""
        from starlette.websockets import WebSocketDisconnect

        sid = client.post("/api/sim", json={"robot": "so101"}).json()["id"]
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect(f"/ws/telemetry/{sid}", headers={"origin": "http://evil.example"}),
        ):
            pass
        assert exc.value.code == 4401


class TestTwinGeometry:
    def test_scene_describes_the_compiled_model(self, monkeypatch):
        from strands_robots.dashboard import scene

        monkeypatch.setattr(scene, "_name", lambda model, kind, i: f"{kind.lower()}{i}")
        d = scene.describe(FakeModel())
        assert d["ngeom"] == 2 and d["pose_row_floats"] == 12
        plane, mesh = d["geoms"]
        assert plane["type"] == "plane" and plane["mesh"] is None and plane["rgba"] == [0.5, 0.5, 0.5, 1.0]
        assert mesh["type"] == "mesh" and mesh["mesh"] == 0 and mesh["group"] == 3 and mesh["body"] == "body0"
        assert d["meshes"] == [{"id": 0, "name": "mesh0", "vertices": 4, "faces": 2, "url": "mesh/0"}]

    def test_mesh_bytes_round_trip(self):
        import struct

        from strands_robots.dashboard import scene

        b = scene.mesh_bytes(FakeModel(), 0)
        assert b[:4] == b"SRM1"
        nvert, nface = struct.unpack("<II", b[4:12])
        assert (nvert, nface) == (4, 2)
        verts = np.frombuffer(b[12 : 12 + nvert * 12], dtype="<f4").reshape(4, 3)
        faces = np.frombuffer(b[12 + nvert * 12 :], dtype="<u4").reshape(2, 3)
        assert verts[3].tolist() == [0.0, 0.0, 1.0] and faces[1].tolist() == [0, 2, 3]
        with pytest.raises(IndexError):
            scene.mesh_bytes(FakeModel(), 1)

    def test_poses_are_packed_as_12_float32_per_geom(self, fake_factory):
        s = sim_session.SimSession("so101")
        s.wait_ready(5)
        time.sleep(0.15)
        poses = np.frombuffer(s.snapshot.poses, dtype="<f4").reshape(-1, 12)
        assert poses.shape == (2, 12)
        assert poses[0, 3:].tolist() == [1, 0, 0, 0, 1, 0, 0, 0, 1]
        s.stop()

    def test_scene_and_mesh_routes(self, client, monkeypatch):
        sid = _create(client)["id"]
        monkeypatch.setattr(sim_session.SimSession, "model", property(lambda self: FakeModel()))
        from strands_robots.dashboard import scene

        monkeypatch.setattr(scene, "_name", lambda model, kind, i: None)
        assert client.get(f"/api/sim/{sid}/scene").json()["ngeom"] == 2
        r = client.get(f"/api/sim/{sid}/mesh/0")
        assert r.status_code == 200 and r.headers["content-type"] == "application/octet-stream"
        assert r.content[:4] == b"SRM1" and "max-age" in r.headers["cache-control"]
        assert client.get(f"/api/sim/{sid}/mesh/7").status_code == 404
        assert client.get("/api/sim/nope/scene").status_code == 404

    def test_scene_before_the_engine_exists_is_409(self, client, monkeypatch):
        sid = _create(client)["id"]
        monkeypatch.setattr(sim_session.SimSession, "model", property(lambda self: None))
        assert client.get(f"/api/sim/{sid}/scene").status_code == 409

    def test_telemetry_sends_binary_poses_only_when_asked(self, client):
        sid = _create(client)["id"]
        time.sleep(0.15)
        with client.websocket_connect(f"/ws/telemetry/{sid}?poses=1") as ws:
            snap = ws.receive_json()
            raw = ws.receive_bytes()
        assert snap["id"] == sid and len(raw) == 2 * 12 * 4
        with client.websocket_connect(f"/ws/telemetry/{sid}") as ws:
            ws.receive_json()
            ws.receive_json()  # two JSON frames in a row: no binary interleaved


class TestEstop:
    def test_estop_freezes_and_gates_then_resume_needs_proof(self, client):
        sid = _create(client)["id"]
        assert client.get("/api/safety").json()["lockout"]["state"] == "clear"
        e = client.post("/api/safety/estop").json()
        assert e["lockout"]["state"] == "locked" and e["frozen"] == [sid]
        time.sleep(0.1)
        t1 = client.get(f"/api/sim/{sid}").json()["sim_time"]
        time.sleep(0.1)
        assert client.get(f"/api/sim/{sid}").json()["sim_time"] == t1
        assert client.get(f"/api/sim/{sid}").json()["state"] == "frozen"
        for path, body in (
            (f"/api/sim/{sid}/joints", {"positions": {"j0": 0.1}}),
            (f"/api/sim/{sid}/reset", None),
            ("/api/sim", {"robot": "so101"}),
        ):
            r = client.post(path, json=body)
            assert r.status_code == 423, path
            assert "e-stop engaged" in r.json()["error"]
        stopped = client.delete(f"/api/sim/{sid}")
        assert stopped.status_code == 200, "stopping is never refused"
        r = client.post("/api/safety/resume").json()
        assert r["lockout"]["state"] == "unknown", "a resume is a request, not proof"
        snap = _create(client)
        assert client.get("/api/safety").json()["lockout"]["state"] == "clear", "an accepted command is the proof"
        assert snap["state"] == "running"

    def test_an_estop_reaches_a_session_whose_engine_is_still_building(self, fake_factory):
        """The window an e-stop exists for: Start pressed, engine not built yet, then E-STOP.

        Building the engine takes real time (a model compile plus a renderer),
        so a session sits in ``starting`` for that long. It has not stepped yet
        and starts the instant the build returns, so the e-stop must reach it.
        """
        from strands_robots.dashboard.routes_sim import Safety

        building, release = threading.Event(), threading.Event()
        rendering, hold = threading.Event(), threading.Event()

        class SlowToBuild(FakeEngine):
            def __init__(self, robot):
                building.set()
                release.wait(5)
                super().__init__(robot)

            def get_frame(self, *a, **kw):
                # Parked in the first render, which precedes ready: the worker
                # has published exactly once since the engine appeared, and
                # that publish is what is read.
                rendering.set()
                hold.wait(5)
                return super().get_frame(*a, **kw)

        store = sim_session.SessionStore()
        safety = Safety(store)
        session = store.create("so101", engine_factory=SlowToBuild)
        assert building.wait(5) and session.snapshot.state == "starting"

        assert safety.estop(by="operator")["frozen"] == [session.id]
        release.set()
        assert rendering.wait(5)
        snap = session.snapshot
        assert snap.state == "frozen", "the engine arrived into an e-stop, so it reports frozen"
        assert (snap.steps, snap.sim_time) == (0, 0.0), "no physics ran after the e-stop"

        hold.set()
        assert session.wait_ready(5), "ready follows the first frame"
        safety.resume(by="operator")
        assert _until(lambda: session.snapshot.steps > 0), "a resume thaws the session frozen while starting"
        store.shutdown()

    def test_an_estop_during_the_build_refuses_that_create_and_stays_latched(self, client, monkeypatch):
        """The create was admitted before the e-stop, so it is neither served nor taken as proof."""
        building, release = threading.Event(), threading.Event()

        def slow(robot):
            building.set()
            release.wait(5)
            return FakeEngine(robot)

        monkeypatch.setattr(sim_session, "_default_factory", slow)
        reply: dict = {}

        def create():
            r = client.post("/api/sim", json={"robot": "so101"})
            reply.update(status=r.status_code, body=r.json())

        worker = threading.Thread(target=create)
        worker.start()
        assert building.wait(5), "the create never reached the engine build"
        safety = client.app.state.safety
        session = safety.store.all()[0]
        safety.estop(by="operator")
        release.set()
        worker.join(10)

        assert reply["status"] == 423 and "e-stop engaged" in reply["body"]["error"]
        assert safety.lockout.state == "locked", "an in-flight create is not proof that the lockout lifted"
        assert safety.store.all() == [], "the refused session is not left running"
        assert session.snapshot.steps == 0

    def test_an_estop_in_the_instant_before_the_create_is_folded_drops_the_session(self, client, monkeypatch):
        """The build finished clear, and the red button lands as the create is about to be folded.

        Between the last read of the lockout in the route and the fold in
        ``accepted`` there is no work left for the request but the fold itself,
        so the e-stop is fired on the way into it - the narrowest window the
        latch can land in. The refusal is 423 as for the wider windows; what is
        pinned here is that the refused session leaves the store with it. Left
        behind it holds one of the slots, frozen, and thaws into a running robot
        on the next resume - one no operator was ever handed.
        """
        safety = client.app.state.safety
        fold = type(safety).accepted

        def an_estop_lands_first():
            safety.estop(by="operator")
            fold(safety)

        monkeypatch.setattr(safety, "accepted", an_estop_lands_first)
        r = client.post("/api/sim", json={"robot": "so101"})

        assert r.status_code == 423 and "e-stop engaged" in r.json()["error"]
        assert safety.lockout.state == "locked", "the refused create is not proof that the lockout lifted"
        assert safety.store.all() == [], "a create the caller was told was refused holds no slot"
        assert safety.resume(by="operator")["thawed"] == [], "and there is nothing for a resume to set running"

    def test_an_estop_during_a_write_refuses_that_request_and_stays_latched(self, client, monkeypatch):
        """The joints request was admitted while clear, and the red button was pressed mid-write.

        The write cannot be recalled - the worker is already inside the engine
        call - so what has to hold is the latch. Folding this command in as proof
        would report the lockout clear while every session sits frozen, and the
        next command would be admitted.
        """
        inside, release = threading.Event(), threading.Event()

        class HoldsTheWrite(FakeEngine):
            def set_joint_positions(self, positions, robot_name=None, hold=False):
                inside.set()
                release.wait(5)
                return super().set_joint_positions(positions, robot_name=robot_name, hold=hold)

        monkeypatch.setattr(sim_session, "_default_factory", HoldsTheWrite)
        sid = _create(client)["id"]
        reply: dict = {}

        def send_joints():
            r = client.post(f"/api/sim/{sid}/joints", json={"positions": {"j0": 0.5}})
            reply.update(status=r.status_code, body=r.json())

        worker = threading.Thread(target=send_joints)
        worker.start()
        assert inside.wait(5), "the command never reached the engine"
        safety = client.app.state.safety
        safety.estop(by="operator")
        release.set()
        worker.join(10)

        assert reply["status"] == 423 and "e-stop engaged" in reply["body"]["error"]
        assert safety.lockout.state == "locked", "an in-flight command is not proof the lockout lifted"
        blocked = client.post("/api/sim", json={"robot": "so101"})
        assert blocked.status_code == 423, "the latch still refuses the next command"

    def test_a_command_the_freeze_refused_answers_the_estop_not_a_bad_request(self, client, monkeypatch):
        """A queued command refused because its session froze is an e-stop answer, not a bad body.

        The worker is parked in a render, so the joints command sits on the queue
        when the e-stop lands; the drain then refuses it. That refusal is an
        error envelope, and reading it before the lockout would answer 400 for
        what is a 423.
        """
        rendering, hold = threading.Event(), threading.Event()

        class ParksInTheRender(FakeEngine):
            def get_frame(self, *a, **kw):
                rendering.set()
                hold.wait(5)
                return super().get_frame(*a, **kw)

        monkeypatch.setattr(sim_session, "_default_factory", ParksInTheRender)
        sid = _create(client)["id"]
        assert rendering.wait(5), "the worker never reached a render"
        reply: dict = {}

        def send_joints():
            r = client.post(f"/api/sim/{sid}/joints", json={"positions": {"j0": 0.5}})
            reply.update(status=r.status_code, body=r.json())

        worker = threading.Thread(target=send_joints)
        worker.start()
        safety = client.app.state.safety
        assert _until(lambda: not safety.store.get(sid)._commands.empty()), "the command never queued"
        safety.estop(by="operator")
        hold.set()
        worker.join(10)

        assert reply["status"] == 423, "a frozen session's refusal is the e-stop's answer"
        assert "e-stop engaged" in reply["body"]["error"]
        assert safety.lockout.state == "locked"

    def test_an_estop_does_not_relabel_a_session_that_failed_to_start(self):
        """``error`` is not a state an e-stop can freeze, and saying so would hide the failure."""
        from strands_robots.dashboard.routes_sim import Safety

        store = sim_session.SessionStore()
        session = store.create("so101", engine_factory=ExplodingEngine)
        assert session.wait_ready(5) and session.snapshot.state == "error"
        assert Safety(store).estop(by="operator")["frozen"] == []
        assert session.snapshot.state == "error"
        store.shutdown()

    def test_an_estop_that_lands_inside_a_resume_leaves_every_session_frozen(self, fake_factory):
        """resume() folds ``unknown``, then thaws; estop() freezes, then latches.

        If the fold and the thaw are not one step, an e-stop can land between
        them: it freezes and latches, and then the resume's thaw runs anyway.
        The lockout then reads ``locked`` while every session steps in realtime,
        the inversion of the invariant ``docs/dashboard.md`` promises. This cell
        parks the thaw, fires the e-stop, and checks the invariant afterwards
        for both interleavings: the e-stop must wait for the whole resume, or
        run whole before it.
        """
        from strands_robots.dashboard.routes_sim import Safety

        entered, go = threading.Event(), threading.Event()

        class ParkedThaw(sim_session.SessionStore):
            def thaw_all(self):
                entered.set()
                go.wait(5)
                return super().thaw_all()

        store = ParkedThaw()
        safety = Safety(store)
        session = store.create("so101", engine_factory=fake_factory)
        assert session.wait_ready(5)
        safety.estop(by="operator")
        assert session.frozen

        resume = threading.Thread(target=safety.resume, kwargs={"by": "operator"})
        resume.start()
        assert entered.wait(5), "the resume reached its thaw"
        estop = threading.Thread(target=safety.estop, kwargs={"by": "operator"})
        estop.start()
        time.sleep(0.2)  # long enough for an unguarded freeze_all to run inside the resume
        go.set()
        resume.join(5)
        estop.join(5)
        assert not resume.is_alive() and not estop.is_alive()

        assert safety.lockout.state == "locked", "the e-stop landed last, so the lockout is latched"
        assert session.frozen, "a latched lockout means every session is frozen, whatever the resume did"
        assert session.snapshot.state == "frozen"
        store.shutdown()

    def test_estop_is_never_refused(self, client):
        client.post("/api/safety/estop")
        assert client.post("/api/safety/estop").status_code == 200

    def test_everything_is_guarded(self, client, monkeypatch):
        sid = _create(client)["id"]
        monkeypatch.setenv("STRANDS_DASH_AUTH_ENABLED", "1")
        for method, path in (
            ("get", "/api/fleet"),
            ("get", "/api/robots/so101"),
            ("get", "/api/sim"),
            ("post", "/api/sim"),
            ("get", f"/api/sim/{sid}"),
            ("get", f"/api/sim/{sid}/stream.mjpg?frames=1"),
            ("get", f"/api/sim/{sid}/scene"),
            ("get", f"/api/sim/{sid}/mesh/0"),
            ("post", f"/api/sim/{sid}/joints"),
            ("delete", f"/api/sim/{sid}"),
            ("get", "/api/safety"),
            ("post", "/api/safety/estop"),
            ("post", "/api/safety/resume"),
        ):
            assert getattr(client, method)(path).status_code == 401, (method, path)
        from starlette.websockets import WebSocketDisconnect

        with client.websocket_connect(f"/ws/telemetry/{sid}") as ws, pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 4401


class TestFleet:
    def test_fleet_lists_the_registry_and_says_mesh_off(self, client):
        f = client.get("/api/fleet").json()
        names = {r["name"] for r in f["robots"]}
        assert {"so101", "unitree_g1", "panda"} <= names
        assert f["count"] == len(f["robots"])
        assert f["mesh"]["status"] == "off"
        assert all("model_local" in r for r in f["robots"] if r["has_sim"])

    def test_fleet_mode_filter(self, client):
        assert client.get("/api/fleet?mode=nope").status_code == 400
        both = client.get("/api/fleet?mode=both").json()["robots"]
        assert both and all(r["has_sim"] and r["has_real"] for r in both)

    def test_robot_detail_and_alias(self, client):
        r = client.get("/api/robots/so-101").json()
        assert r["name"] == "so101" and r["entry"]["category"]
        assert client.get("/api/robots/nope").status_code == 404


class TestReadyMeansItRenders:
    """The first frame is built before ready, so a session that cannot render is
    reported as ``error`` by the create route, not served as one that never streams."""

    def test_a_renderer_that_fails_is_an_error_before_ready_and_the_engine_is_closed(self):
        closed = threading.Event()

        class NoGL(FakeEngine):
            def get_frame(self, *a, **kw):
                raise RuntimeError("no OpenGL context")

            def close(self):
                closed.set()

        s = sim_session.SimSession("so101", engine_factory=NoGL)
        assert s.wait_ready(5)
        snap = s.snapshot
        assert snap.state == "error" and "no OpenGL context" in (snap.error or "")
        assert (snap.steps, snap.sim_time) == (0, 0.0)
        assert closed.wait(5), "the engine is released with the session"
        assert s.latest_frame() is None

    def test_ready_carries_a_frame(self):
        s = sim_session.SimSession("so101", engine_factory=FakeEngine)
        assert s.wait_ready(5)
        assert s.latest_frame() is not None, "no client sees a ready session without a frame"
        s.stop()

    def test_the_create_route_reports_a_missing_renderer_as_500_and_keeps_no_session(self, client, monkeypatch):
        class NoGL(FakeEngine):
            def get_frame(self, *a, **kw):
                raise RuntimeError("no OpenGL context")

        monkeypatch.setattr(sim_session, "_default_factory", NoGL)
        r = client.post("/api/sim", json={"robot": "so101"})
        assert r.status_code == 500 and "no OpenGL context" in r.json()["error"]
        assert client.get("/api/sim").json()["sessions"] == []

    def test_a_first_render_that_never_returns_is_dropped_not_served_as_running(self, client, monkeypatch):
        """The state published before the first frame is ``running``. A renderer that
        never comes back would be handed to the operator under that name, streaming
        nothing and holding a session slot, so the create route drops it instead."""
        hold = threading.Event()

        class ParksInTheFirstRender(FakeEngine):
            def get_frame(self, *a, **kw):
                hold.wait(30)
                return super().get_frame(*a, **kw)

        monkeypatch.setattr(routes_sim, "READY_TIMEOUT", 0.2)
        monkeypatch.setattr(sim_session, "_default_factory", ParksInTheFirstRender)
        try:
            r = client.post("/api/sim", json={"robot": "so101"})
            assert r.status_code == 504, r.text
            assert "did not render a first frame" in r.json()["error"]
            assert client.get("/api/sim").json()["sessions"] == [], "the session that cannot stream is kept"
        finally:
            hold.set()
        # The slot it held is free again: a working engine still starts afterwards.
        monkeypatch.setattr(sim_session, "_default_factory", FakeEngine)
        assert _create(client)["state"] == "running"


# -- the real engine -----------------------------------------------------------


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_real_engine_session_steps_and_renders(monkeypatch):
    s = sim_session.SimSession("so101")
    assert s.wait_ready(60), "engine did not start"
    if s.snapshot.state == "error":
        pytest.skip(f"no renderer here: {s.snapshot.error}")
    # Ready already carries the first frame; the physics is read when it has
    # visibly advanced, not after a sleep that guesses how slow this GL is.
    frame = s.latest_frame()
    assert frame is not None and frame.shape == (384, 512, 3)
    assert _until(lambda: s.snapshot.sim_time > 0.2, timeout=15.0), s.snapshot
    snap = s.snapshot
    assert snap.joint_names == ("1", "2", "3", "4", "5", "6")

    from strands_robots.dashboard import scene

    d = scene.describe(s.model)
    assert d["ngeom"] == 31 and len(d["meshes"]) == 13 and d["geoms"][0]["type"] == "plane"
    assert len(s.snapshot.poses) == 31 * 12 * 4
    assert scene.mesh_bytes(s.model, 0)[:4] == b"SRM1"
    assert s.command("set_joints", positions={"2": 0.3})["status"] == "success"
    s.stop()


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_real_engine_joints_target_survives_the_steps_that_follow():
    """On ``so101`` a joints write that is not held springs back toward home.

    Measured before the fix: ``{"2": 0.5}`` answered ``success`` and read 0.03 rad
    half a second of sim time later - the position servo was still commanded to
    its old setpoint. Held, it reads within a few hundredths of the target.
    """
    s = sim_session.SimSession("so101")
    assert s.wait_ready(60), "engine did not start"
    if s.snapshot.state == "error":
        pytest.skip(f"no renderer here: {s.snapshot.error}")
    try:
        assert _until(lambda: s.snapshot.sim_time > 0.2, timeout=15.0), s.snapshot
        joint = s.snapshot.joint_names.index("2")
        assert s.command("set_joints", positions={"2": 0.5})["status"] == "success"
        settled_at = s.snapshot.sim_time + 0.5
        assert _until(lambda: s.snapshot.sim_time > settled_at, timeout=30.0), s.snapshot
        held = s.snapshot.qpos[joint]
        assert abs(held - 0.5) < 0.15, f"joint 2 read {held:.3f} rad half a second after a 0.5 rad target"
    finally:
        s.stop()
