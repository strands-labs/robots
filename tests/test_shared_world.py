"""The thesis: two robots, one world. A control step is one control step, a teleport lands, one render shows both."""

import importlib.util
import json
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from strands_robots import Robot, Simulation

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


@pytest.mark.parametrize(
    "written, id_, name",
    [
        ("g1", "g1", "unitree_g1"),
        ("Unitree-G1", "unitree_g1", "unitree_g1"),
        ("so101_follower", "so101_follower", "so101"),
    ],
)
def test_a_robot_is_addressed_by_the_name_it_was_written_with(written, id_, name):
    # Robot("g1").id was "unitree_g1": the URL, the card, sim.run maps and the agent tool all wanted the other name
    sim = Simulation()
    r = Robot(written, sim=sim)
    assert (r.id, r.name, list(sim.robots)) == (id_, name, [id_])
    assert Robot(written, sim=sim, id="mine").id == "mine"  # a given id still wins
    sim.close()


@pytest.fixture
def pair():
    return Robot("so101"), Robot("g1", position=[1, 0, 0])


def test_two_robots_share_the_default_world(pair):
    arm, g1 = pair
    assert arm.sim is g1.sim is Simulation.default()
    assert list(arm.sim.robots) == ["so101", "g1"]
    assert Robot("so101", position=[0, -0.5, 0]).id == "so101_2"
    assert json.loads(json.dumps(arm.sim.describe()))["robots"]["g1"]["position"] == [1, 0, 0]


def test_each_robot_reads_its_own_joints_and_pose(pair):
    arm, g1 = pair
    a, g = arm.observe(), g1.observe()
    assert a.joints.names == ARM_JOINTS
    assert len(g.joints.names) == 29 and g.joints.names[0] == "left_hip_pitch_joint"
    assert np.allclose(g.pose.position, [1, 0, 0.79], atol=1e-6)  # the "stand" keyframe, moved to the spawn frame
    assert np.allclose(a.pose.position, [0, 0, 0], atol=1e-9)


def test_act_advances_one_control_step_and_leaves_the_other_robot(pair):
    arm, g1 = pair
    sim = arm.sim
    before_g1 = g1.observe().joints.positions.copy()
    t0 = sim.time
    arm.act({"shoulder_pan": 0.3})
    assert sim.physics.control_dt == pytest.approx(0.02)
    assert sim.time == pytest.approx(t0 + sim.physics.control_dt)
    for _ in range(49):
        arm.act({"shoulder_pan": 0.3})
    assert arm.observe()["shoulder_pan"] == pytest.approx(0.3, abs=0.02)
    assert np.abs(g1.observe().joints.positions - before_g1).max() < np.radians(1.0)  # g1 holds its keyframe


def test_manual_time_hands_stepping_to_the_caller(pair):
    arm, _ = pair
    sim = arm.sim
    t0 = sim.time
    with sim.manual_time():
        arm.act({"shoulder_pan": 0.2})
        assert sim.time == t0 and sim.time_is_manual
        sim.step(10)
    assert sim.time == pytest.approx(t0 + 10 * sim.physics.timestep)
    arm.act({"shoulder_pan": 0.2})
    assert sim.time == pytest.approx(t0 + 10 * sim.physics.timestep + sim.physics.control_dt)


def test_set_pose_lands_for_a_floating_and_a_fixed_base(pair):
    arm, g1 = pair
    sim = arm.sim
    g1.set_pose(position=[2, 0, 0.8])
    assert np.allclose(g1.pose.position, [2, 0, 0.8], atol=1e-9)
    assert np.allclose(g1.pose.linear_velocity, 0.0)
    for _ in range(5):
        arm.act({"shoulder_pan": 0.3})
    moved, t = arm.observe()["shoulder_pan"], sim.time
    arm.set_pose(position=[0, 0.5, 0])  # fixed base: spec edit + recompile, joint state and time survive
    assert np.allclose(arm.pose.position, [0, 0.5, 0], atol=1e-9)
    assert arm.observe()["shoulder_pan"] == pytest.approx(moved, abs=1e-9) and sim.time == t
    with pytest.raises(KeyError, match="ghost"):
        sim.set_pose("ghost", position=[0, 0, 0])


def test_a_fixed_teleport_keeps_the_randomized_physics():
    with Simulation() as sim:
        arm = Robot("so101", sim=sim)
        base_friction, base_mass = sim.backend.model.geom_friction[:, 0].copy(), sim.backend.model.body_mass.copy()
        sim.randomize(friction=1.5, mass=2.0)
        sim.set_pose(arm, position=[0, 0.5, 0])
        assert np.allclose(sim.backend.model.geom_friction[:, 0], 1.5 * base_friction)
        assert np.allclose(sim.backend.model.body_mass, 2.0 * base_mass)


@pytest.mark.parametrize(
    ("spawn", "target"),
    [
        ({"position": [1, 0, 0]}, [0, 0.5, 0]),  # an offset spawn frame
        ({"position": [0.2, 0, 0], "orientation": [0.7071068, 0, 0, 0.7071068]}, [0.5, 0, 0]),  # a yawed one
    ],
)
def test_a_fixed_base_teleport_lands_in_world_coordinates_and_survives_a_reload(spawn, target):
    with Simulation() as sim:
        arm = Robot("so101", sim=sim, **spawn)
        sim.set_pose(arm, position=target)
        assert np.allclose(arm.pose.position, target, atol=1e-6)
        sim.add_object("box", shape="box", position=[0.3, 0, 0.1])  # an unrelated rebuild does not move it
        assert np.allclose(arm.pose.position, target, atol=1e-6)


def test_a_locked_reader_never_sees_a_half_added_or_half_removed_robot():
    with Simulation() as sim:
        Robot("so101", sim=sim)
        stop, errors = threading.Event(), []

        def reader() -> None:
            try:
                while not stop.is_set():
                    with sim.lock:
                        sim.describe()
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
        for t in threads:
            t.start()
        for _ in range(15):
            sim.remove(Robot("so101", sim=sim))
        stop.set()
        for t in threads:
            t.join(2)
        assert not errors, errors[:3]


def test_one_render_shows_both_bodies(pair):
    arm, _ = pair
    img = arm.sim.render()
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    overview = arm.sim.scene.cameras[-1]
    with Simulation() as empty:  # the same camera over an empty floor: what differs is a robot
        empty.camera("overview", position=overview.position, target=overview.target)
        diff = np.abs(img.astype(int) - empty.render("overview").astype(int)).sum(axis=2) > 60
    h, w, _ = img.shape
    left, right = int(diff[:, : w // 2].sum()), int(diff[:, w // 2 :].sum())
    assert left > 150 and right > 1000, (left, right)  # the 30 cm arm left of centre, the G1 right of it
    assert diff.sum() < 0.25 * h * w


def _robot_pixels(sim):
    """Where the robots are in ``sim.render()``: the pixels that differ from the same camera over an empty world."""
    img, cam = sim.render(), sim.scene.cameras[-1]
    with Simulation(ground=False) as empty:
        empty.camera("overview", position=cam.position, target=cam.target)
        return np.abs(img.astype(int) - empty.render("overview").astype(int)).sum(axis=2) > 30


@pytest.mark.parametrize(("robots", "min_height"), [((("so101", None),), 0.6), ((("g1", None),), 0.8)])
def test_the_overview_frames_the_robots_not_the_floor(robots, min_height):
    """A lone 30 cm arm filled 68 of 480 rows (a 28 x 68 px box at 1.8 m); the overview now fits the bodies."""
    with Simulation(ground=False) as sim:
        for name, position in robots:
            Robot(name, sim=sim, position=position)
        ys, xs = np.nonzero(_robot_pixels(sim))
    h, w = 480, 640
    assert (ys.max() - ys.min()) / h >= min_height, (ys.min(), ys.max())
    assert ys.min() > 0 and ys.max() < h - 1 and xs.min() > 0 and xs.max() < w - 1  # in view, not cropped


@pytest.mark.parametrize(
    "outline",
    [np.array([[0, -0.4, 0], [0.1, 0.1, 0.3]]), np.array([[-1, 0, 0], [1.2, 0.2, 1.3]]), np.zeros((1, 3))],
    ids=["a reaching arm", "two bodies a metre apart", "a point"],
)
def test_overview_keeps_every_outline_point_in_view(outline):
    from strands_robots.simulation.scene import Scene

    cam = Scene().overview(outline, aspect=4 / 3)
    pos, fwd = np.array(cam["position"]), np.subtract(cam["target"], cam["position"])
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1]) / np.linalg.norm(np.cross(fwd, [0, 0, 1]))
    up, t = np.cross(right, fwd), np.tan(np.radians(22.5))
    q = outline - pos
    x, y = q @ right / (q @ fwd) / (t * 4 / 3), q @ up / (q @ fwd) / t
    assert np.abs(x).max() <= 1 and np.abs(y).max() <= 1
    assert max(np.abs(x).max(), np.abs(y).max()) > 0.85 or len(outline) == 1  # fills the view, not a speck


def test_render_without_a_camera_is_the_overview_even_after_user_cameras(pair):
    arm, _ = pair
    sim = arm.sim
    sim.camera("wrist", position=[0.15, -0.25, 0.25], target=[0.2, 0, 0.05], size=(64, 48), attach=arm)
    assert sim.render().shape == (480, 640, 3)
    assert [c.name for c in sim.scene.cameras] == ["wrist", "overview"]
    assert sim.render("wrist").shape == (48, 64, 3)
    with pytest.raises(KeyError, match="no camera 'nope'"):
        sim.render("nope")


def test_any_live_thread_can_render_after_another_did(pair):
    """A GL context is bound to the thread that made it current; renders from a script, a run loop and a server
    thread that all stay alive must share the world's cameras (EGL_BAD_ACCESS before, a hang on macOS)."""
    sim = pair[0].sim
    sim.camera("eye", position=[1.0, -1.0, 0.8], target=[0.3, 0, 0.3], size=(64, 48))
    first = sim.render("eye")  # this thread renders first and stays alive
    out: dict[str, object] = {}
    go, done = threading.Event(), threading.Event()

    def worker() -> None:
        try:
            out["img"] = sim.render("eye")
        except Exception as exc:  # noqa: BLE001 - the failure is the assertion
            out["img"] = exc
        done.set()
        go.wait(5)  # still alive while the first thread renders again

    t = threading.Thread(target=worker)
    t.start()
    assert done.wait(10)
    again = sim.render("eye")
    go.set()
    t.join(5)
    assert isinstance(out["img"], np.ndarray), out["img"]
    assert out["img"].shape == first.shape == again.shape == (48, 64, 3)


def test_a_script_that_rendered_exits_cleanly_without_closing_its_world(tmp_path):
    """The renderers live on the GL thread; a context the collector frees from the main thread at interpreter
    exit is a segfault (exit code -11) once torch is loaded, as it is by every LeRobot policy and by record=.
    They are freed on that thread before the interpreter goes."""
    if importlib.util.find_spec("torch") is None:  # only the script imports it; this process never does
        pytest.skip("needs torch")
    script = tmp_path / "render_and_leave.py"
    script.write_text(
        "import torch  # noqa: F401\n"
        "from strands_robots import Robot\n"
        "arm = Robot('so101')\n"
        "arm.sim.camera('eye', size=(64, 48), attach=arm)\n"
        "print(arm.sim.render('eye').shape)\n"  # the renderer is still alive when the script ends
    )
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-800:]
    assert "(48, 64, 3)" in out.stdout


def test_state_survives_add_object_and_remove(pair):
    arm, g1 = pair
    sim = arm.sim
    for _ in range(30):
        arm.act({"elbow_flex": 0.4})
    before = arm.observe()["elbow_flex"]
    box = sim.add_object("box", shape="box", position=[0.3, 0, 0.1], size=[0.03, 0.03, 0.03])
    assert arm.observe()["elbow_flex"] == pytest.approx(before, abs=1e-6)
    assert np.allclose(box.pose.position, [0.3, 0, 0.1])
    sim.remove(g1)
    assert list(sim.robots) == ["so101"] and g1.sim is None
    with pytest.raises(RuntimeError, match="removed"):
        g1.observe()
    assert arm.observe()["elbow_flex"] == pytest.approx(before, abs=1e-6)


def test_world_time_survives_every_scene_change(pair):
    arm, g1 = pair
    sim = arm.sim
    sim.step(50)
    t = sim.time
    sim.add_object("box", shape="box", position=[0.3, 0, 0.1])
    sim.camera("cam", position=[1, -1, 1], target=[0, 0, 0.2])
    sim.randomize(friction=1.2)
    sim.remove(g1)
    assert t > 0 and sim.time == t


def _gains(robot):
    """(kp, kd) per actuator, read from the compiled world (a position servo's biasprm is [0, -kp, -kd])."""
    m = robot.sim.backend.model
    ids = [m.actuator(f"{robot.id}/{a}").id for a in robot.description.actuators]
    return m.actuator_gainprm[ids, 0].copy(), -m.actuator_biasprm[ids, 2]


@pytest.mark.parametrize(
    "change",
    [
        lambda sim, arm: sim.render(),  # the first render adds the overview camera
        lambda sim, arm: sim.add_object("box", shape="box", position=[0.3, 0, 0.1]),
        lambda sim, arm: sim.camera("cam", position=[1, -1, 1], target=[0, 0, 0.2]),
        lambda sim, arm: Robot("so101", sim=sim, position=[0, 0.5, 0]),
        lambda sim, arm: sim.remove(arm),
        lambda sim, arm: sim.remove("crate"),  # by name
        lambda sim, arm: sim.randomize(friction=1.2),
        lambda sim, arm: sim.set_pose(arm, position=[0, -0.5, 0]),  # a fixed base is rebuilt where it lands
    ],
    ids=[
        "first_render",
        "add_object",
        "camera",
        "a_robot_joins",
        "a_robot_leaves",
        "an_object_leaves",
        "randomize",
        "fixed_base_teleport",
    ],
)
def test_the_gains_set_gains_wrote_survive_every_scene_change(pair, change):
    arm, g1 = pair
    arm.sim.add_object("crate", shape="box", position=[-0.6, 0.6, 0.1])  # there before the gains: one case removes it
    n = g1.description.dof
    g1.set_gains(np.full(n, 40.0), np.full(n, 1.0))  # what run() does with a policy's PD (#4749 decision 6)
    change(arm.sim, arm)
    kp, kd = _gains(g1)
    assert np.allclose(kp, 40.0) and np.allclose(kd, 1.0)  # dev: back to the MJCF's, kp 500


def test_a_removed_robots_gains_leave_with_it(pair):
    arm, _ = pair
    sim = arm.sim
    mjcf = _gains(arm)
    arm.set_gains(np.full(6, 5.0), np.full(6, 0.1))
    sim.remove(arm)
    again = Robot("so101", sim=sim)  # the same id, a new body: it starts from its MJCF
    assert again.id == "so101" and all(np.allclose(a, b) for a, b in zip(_gains(again), mjcf, strict=True))


def test_revision_bumps_on_every_scene_change_and_not_on_motion(pair):
    arm, g1 = pair
    sim = arm.sim
    r0 = sim.revision
    sim.add_object("box", shape="box", position=[0.3, 0, 0.1])
    sim.camera("cam", position=[1, -1, 1], target=[0, 0, 0.2])
    sim.set_pose(g1, position=[2, 0, 0.8])
    sim.randomize(friction=1.2)
    sim.remove(g1)
    assert sim.revision == r0 + 5
    arm.act({"shoulder_pan": 0.1})
    sim.step(5)
    assert sim.revision == r0 + 5


def test_the_world_lock_serialises_steps_and_reads_across_a_recompile(pair):
    arm, _ = pair
    sim = arm.sim
    errors: list[BaseException] = []
    stop = threading.Event()

    def reader() -> None:
        try:
            while not stop.is_set():
                with sim.lock:
                    arm.observe(images=False)
                    sim.describe()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    for _ in range(100):
        arm.act({"shoulder_pan": 0.3})
    Robot("so101", position=[0, 0.5, 0])  # a recompile under the reader
    stop.set()
    t.join(2)
    assert not errors


def test_an_explicit_world_is_separate_and_closes():
    with Simulation(timestep=0.001, substeps=5) as sim:
        arm = Robot("so101", sim=sim)
        assert arm.sim is sim and not sim.is_default and sim.physics.control_dt == pytest.approx(0.005)
    assert arm.sim is None and "so101" not in Simulation.default().robots


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"backend": "bullet"}, ValueError, "unknown simulation backend"),
        ({"physics": "warp"}, ValueError, "physics="),
        ({"substeps": 0}, ValueError, "substeps"),
        ({"num_envs": 4}, ValueError, "num_envs"),
        ({"bogus": 1}, TypeError, "unexpected keyword"),
    ],
)
def test_a_world_refuses_what_it_cannot_be(kwargs, error, match):
    with pytest.raises(error, match=match):
        Simulation(**kwargs)


def test_step_cost_is_measured(pair, capsys):
    sim = pair[0].sim
    n = 2000
    t = time.perf_counter()
    sim.step(n)
    dt = time.perf_counter() - t
    with capsys.disabled():
        print(f"\n[measured] {n} physics steps (so101 + g1) in {dt * 1e3:.1f} ms = {dt / n * 1e6:.1f} us/step")
    assert dt / n < 0.002  # slower than real time would be a regression


def test_contacts_name_both_bodies_and_reset_stands_a_teleported_g1_on_its_keyframe(pair):
    _, g1 = pair
    sim = g1.sim
    sim.step(50)
    feet = [c for c in sim.contacts() if "ground" in (c.body_a, c.body_b) or c.body_a == "world"]
    assert any(c.body_b.startswith("g1/") and c.force > 0 for c in feet), sim.contacts()[:3]
    sim.set_pose(g1, position=[2, 0, 0.8])
    sim.add_object("box", shape="box", position=[0.3, 0, 0.1])  # a reload keeps the teleported base where it is
    assert np.allclose(g1.pose.position, [2, 0, 0.8], atol=1e-9)
    sim.reset()
    assert sim.time == 0.0
    assert np.allclose(g1.pose.position, [1, 0, 0.79], atol=1e-6)


@pytest.mark.parametrize("name", ["g1", "panda"])  # a directional and a spot scene light
def test_the_world_owns_the_lighting(name):
    look = {"position": [1.9, -2.1, 1.3], "target": [0.45, 0, 0.45], "size": (160, 100)}
    with Simulation() as empty, Simulation() as sim:
        empty.camera("c", **look)
        Robot(name, sim=sim)
        sim.camera("c", **look)
        assert sim.backend.model.nlight == empty.backend.model.nlight
        assert sim.render("c")[-10:].mean() == pytest.approx(empty.render("c")[-10:].mean(), abs=5)  # the floor


def test_duplicate_names_get_unique_ids_and_their_own_joints(pair):
    arm, _ = pair
    a2 = Robot("so101", position=[0, -0.5, 0])  # a duplicate AT the origin would collide with the first arm
    assert a2.id == "so101_2" and Robot("so101", position=[0, 0.5, 0]).id == "so101_3"
    for _ in range(10):
        a2.act({"gripper": 0.5})
    assert a2.observe()["gripper"] > 0.05
    assert arm.observe()["gripper"] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize(
    ("action", "error", "match"),
    [({"elbow": 1.0}, KeyError, "unknown joint"), (np.zeros(5), ValueError, "6 action keys")],
)
def test_an_action_the_body_cannot_take_is_refused(pair, action, error, match):
    with pytest.raises(error, match=match):
        pair[0].act(action)


def test_lockstep_run_drives_both_robots_one_control_step_at_a_time(pair):
    arm, g1 = pair
    t0 = arm.sim.time
    res = arm.sim.run({arm: lambda o: {"shoulder_pan": 0.3}, g1: lambda o: o.joints.positions}, steps=20)
    assert set(res) == {"so101", "g1"} and res["so101"].steps == 20
    assert arm.sim.time == pytest.approx(t0 + 20 * arm.sim.physics.control_dt)
    assert res["so101"].final["shoulder_pan"] == pytest.approx(0.3, abs=0.03)
    with Simulation() as other, pytest.raises(ValueError, match="not in this world"):
        other.run({arm: lambda o: o.joints.positions}, steps=1)


def test_an_actuator_that_drives_a_tendon_reads_its_transmission():
    with Simulation() as sim:
        panda = Robot("panda", sim=sim)
        joints = panda.observe().joints
        assert joints.names[-1] == "actuator8" and len(joints.names) == 8  # the gripper drives a tendon, not a joint
        opened = joints["actuator8"]  # the home keyframe holds the fingers open: 255, 0.04 m of tendon
        for _ in range(40):
            panda.act({"actuator8": 0.0})  # close
        assert panda.observe()["actuator8"] < opened - 1.0


TENDON_KEYS = [("panda", "actuator8"), ("robotiq_2f85", "fingers_actuator"), ("stretch3", "arm")]


@pytest.mark.parametrize(("name", "key"), TENDON_KEYS)
def test_a_tendon_actuator_key_reads_the_control_that_holds_it(name, key):
    """A key naming a tendon actuator has no joint: it reads the control at which its servo pushes nothing."""
    sim = Simulation()
    try:
        body = Robot(name, sim=sim)
        body.act(body.observe(images=False).joints.as_dict())
        m, d = sim.backend.model, sim.backend.data
        a = m.actuator(f"{body.id}/{key}").id
        push = m.actuator_gainprm[a, 0] * body.observe(images=False)[key] + m.actuator_biasprm[a, 0]
        assert push + m.actuator_biasprm[a, 1] * d.actuator_length[a] == pytest.approx(0.0, abs=1e-9)
    finally:
        sim.close()


def test_holding_what_the_panda_reads_keeps_its_hand_open():
    sim = Simulation()
    try:
        panda = Robot("panda", sim=sim)
        assert panda.observe(images=False)["actuator8"] == pytest.approx(
            255.0, abs=0.01
        )  # home: 0.04 m of tendon, fully open
        for _ in range(50):
            panda.act(panda.observe(images=False).joints.as_dict())
        a = sim.backend.model.actuator(f"{panda.id}/actuator8").id
        assert sim.backend.data.actuator_length[a] == pytest.approx(0.04, abs=1e-3)
    finally:
        sim.close()


@pytest.mark.parametrize(
    "use",
    [
        lambda sim: sim.add(Robot("so101", sim=Simulation())),
        lambda sim: sim.add_object("cube", shape="box"),
        lambda sim: sim.step(1),
        lambda sim: sim.control_step(),
        lambda sim: sim.reset(),
        lambda sim: sim.run({}, steps=1),
    ],
)
def test_a_closed_simulation_refuses_to_be_used(use):
    sim = Simulation()
    sim.close()
    sim.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        use(sim)
    assert sim.robots == {} and sim._closed
