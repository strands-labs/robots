"""The scene a sim-to-real user builds: a wrist camera, per-body randomization, policy noise and latency,
batched edits that rebuild once, an honest determinism flag, and an Isaac EULA the user accepts themself."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import sys
import types

import mujoco
import numpy as np
import pytest

from strands_robots import Robot, Simulation
from strands_robots.policy import Perturb, SequencePolicy
from strands_robots.robot.device.build import manifest_for
from strands_robots.simulation.backends.mujoco.build import link_names
from strands_robots.types import JointState, Observation

has_warp = importlib.util.find_spec("mujoco_warp") is not None and importlib.util.find_spec("warp") is not None


@pytest.fixture
def world():
    sim = Simulation()
    arm = Robot("so101", sim=sim)
    sim.add_object("cube", shape="box", size=[0.015] * 3, position=[0.25, 0, 0.3], mass=0.05)
    yield sim, arm
    sim.close()


def _body_of_camera(sim, name):
    m = sim.backend.model
    return m.body(m.cam_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)]).name


def test_a_camera_rides_any_link_the_world_describes(world):
    sim, arm = world
    links = sim.describe()["robots"][arm.id]["links"]
    assert "gripper" in links and links[0] == "base"
    sim.camera("wrist", size=(32, 24), attach=f"{arm.id}/gripper")
    sim.camera("base_cam", size=(32, 24), attach=arm)
    assert _body_of_camera(sim, "wrist") == f"{arm.id}/gripper"
    assert _body_of_camera(sim, "base_cam") == f"{arm.id}/base"
    assert sim.render("wrist").shape == (24, 32, 3)
    assert set(arm.observe().images) == {"wrist", "base_cam"}  # a link camera reaches the policy and the recorder
    assert [s.name for s in manifest_for(arm).sensors if s.kind == "camera"] == ["wrist.image", "base_cam.image"]
    with pytest.raises(KeyError, match="no link 'so101/nope'.*links"):
        sim.camera("bad", attach=f"{arm.id}/nope")
    assert [c.name for c in sim.scene.cameras] == ["wrist", "base_cam"]  # a refused camera leaves no trace
    sim.remove(arm)
    assert sim.scene.cameras == []  # a robot takes its cameras, base or link, with it


def test_pose_answers_every_link_the_world_describes(world):
    sim, arm = world
    tip = sim.pose(f"{arm.id}/gripper").position
    arm.act({"shoulder_pan": 0.8})
    sim.step(200)
    moved = sim.pose(f"{arm.id}/gripper").position
    assert np.linalg.norm(moved - tip) > 0.05  # an end effector, live: it follows the joints
    assert np.allclose(sim.pose(f"{arm.id}/base").position, sim.pose(arm.id).position)
    for bad, hint in [("gripper", "so101/gripper"), ("so101/nope", None), ("ghost/gripper", "so101/gripper")]:
        with pytest.raises(KeyError, match=f"no body '{bad}'" + (f".*'<robot>/<link>'.*{hint}" if hint else "")):
            sim.pose(bad)


ARM = {"base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"}


@pytest.mark.parametrize(
    "kwargs, changed",
    [
        ({"target": "cube", "mass": 2.0}, {"cube"}),
        ({"target": "so101", "mass": 2.0}, ARM),
        ({"mass": 2.0}, ARM | {"cube"}),
    ],
)
def test_randomize_mass_scales_only_its_target(world, kwargs, changed):
    sim, _ = world
    before = sim.backend.model.body_mass.copy()
    drawn = sim.randomize(**kwargs)
    if "target" in kwargs:  # the return reports the draw where it landed, and the world factor stays 1.0
        assert drawn["scales"][kwargs["target"]]["mass_scale"] == 2.0 and drawn["mass_scale"] == 1.0
        assert drawn["scales"][kwargs["target"]] == dict(
            zip(("friction_scale", "mass_scale"), sim.scene.scales[kwargs["target"]], strict=True)
        )
    else:
        assert drawn["mass_scale"] == 2.0 and drawn["scales"] == {}
    m = sim.backend.model
    doubled = np.isclose(m.body_mass, 2.0 * before) & (before > 0)
    assert {m.body(b).name.split("/")[-1] for b in np.flatnonzero(doubled)} == changed
    assert np.array_equal(m.body_mass[~doubled], before[~doubled])


def test_randomize_friction_of_one_object_leaves_the_robot_alone(world):
    sim, _ = world
    m0 = sim.backend.model
    cube_geoms = m0.geom_bodyid == mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, "cube")
    before = m0.geom_friction[:, 0].copy()
    drawn = sim.randomize(target="cube", friction=(0.2, 0.4), seed=1)
    f = drawn["scales"]["cube"]["friction_scale"]
    assert 0.2 <= f <= 0.4 and (f, drawn["scales"]["cube"]["mass_scale"]) == sim.scene.scales["cube"]
    assert drawn["friction_scale"] == 1.0  # the world factor was not drawn
    after = sim.backend.model.geom_friction[:, 0]
    assert np.all((after[cube_geoms] / before[cube_geoms] >= 0.2) & (after[cube_geoms] / before[cube_geoms] <= 0.4))
    assert np.array_equal(after[~cube_geoms], before[~cube_geoms])


def test_randomize_jitters_cameras_within_the_bound_and_refuses_a_bad_target(world):
    sim, _ = world
    sim.camera("front", position=[0.5, -0.5, 0.4], size=(32, 24))
    sim.randomize(camera=0.02, seed=3)
    m = sim.backend.model
    off = m.cam_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front")] - np.array([0.5, -0.5, 0.4])
    assert 0 < np.abs(off).max() <= 0.02
    with pytest.raises(KeyError, match="not a robot or object"):
        sim.randomize(target="ghost", mass=2.0)
    with pytest.raises(ValueError, match="lighting is world-wide"):
        sim.randomize(target="cube", lighting=0.5)


def test_links_skip_an_unnamed_body_and_a_robot_not_built_yet(world):
    body = '<body name="r/base"><geom size=".1"/><body><geom size=".1"/></body></body>'
    xml = f"<mujoco><worldbody>{body}</worldbody></mujoco>"
    assert link_names(mujoco.MjModel.from_xml_string(xml), 1) == ["base"]  # reachy_mini has an unnamed body
    sim, _ = world
    with sim.editing():
        Robot("so101", sim=sim, id="late", position=[0.6, 0, 0])
        assert sim.describe()["robots"]["late"]["links"] == []
        assert sim.describe()["robots"]["late"]["pose"]["position"] == [0.6, 0.0, 0.0]  # the spawn frame, not built yet


def test_editing_rebuilds_the_world_once_and_a_failed_block_changes_nothing(world, monkeypatch):
    sim, arm = world
    loads = []
    real = sim.backend.load
    monkeypatch.setattr(sim.backend, "load", lambda scene: (loads.append(1), real(scene)))
    with sim.editing():
        for i in range(4):
            sim.add_object(f"o{i}", shape="sphere", size=[0.02], position=[0.3, 0.1 * i, 0.2])
        sim.camera("top", position=[0, 0, 1.0], target=[0, 0, 0], size=(32, 24))
    assert len(loads) == 1 and sim.render("top").shape == (24, 32, 3)
    names = [o.name for o in sim.scene.objects]
    with pytest.raises(KeyError, match="no link"), sim.editing():
        sim.add_object("late", shape="box", position=[0, 0.3, 0.2])
        sim.camera("bad", attach=f"{arm.id}/nope")
    assert [o.name for o in sim.scene.objects] == names and "bad" not in [c.name for c in sim.scene.cameras]
    sim.step(1)  # the world still steps: the old scene is still the built one
    assert sim.scene is sim.backend._scene  # restored in place: the backend rebuilds from the same scene
    sim.set_pose(arm, position=[0.5, 0, 0])  # a fixed base rebuilds; the undone camera is not in it
    with pytest.raises(KeyError, match="no link"), sim.editing():
        late = Robot("so101", sim=sim)
        sim.remove(arm)
        sim.camera("bad", attach=f"{late.id}/nope")
    assert list(sim.robots) == [arm.id] and arm.observe().joints.positions.shape == (6,)
    with pytest.raises(RuntimeError, match="removed"):
        late.observe()
    with sim.editing():
        sim.remove(arm)
        assert arm.sim is sim  # detached only once the block's rebuild holds
    assert arm.sim is None and list(sim.robots) == []


def test_a_block_that_raises_in_its_body_restores_the_world_and_set_pose_is_part_of_its_one_rebuild(world, monkeypatch):
    sim, arm = world
    loads = []
    real = sim.backend.load
    monkeypatch.setattr(sim.backend, "load", lambda scene: (loads.append(1), real(scene)))
    with sim.editing():  # a fixed base teleported inside the block: the block's rebuild places it, not a second one
        sim.add_object("o0", shape="sphere", size=[0.02], position=[0.3, 0.1, 0.2])
        sim.set_pose(arm, position=[0.4, 0, 0])
    assert len(loads) == 1 and np.allclose(sim.backend.body_pose(arm.id).position[:2], [0.4, 0])
    names = [o.name for o in sim.scene.objects]
    with pytest.raises(KeyError, match="no link"), sim.editing():  # the mistyped link + the advertised verb
        sim.camera("bad", attach=f"{arm.id}/nope")
        sim.set_pose(arm, position=[0.5, 0, 0])  # deferred: the error is the block's, at exit, and rolled back
    assert "bad" not in [c.name for c in sim.scene.cameras] and not sim._pending
    sim.step(1)  # not wedged
    with pytest.raises(RuntimeError, match="whatever"), sim.editing():  # the user's own code raising in the body
        sim.add_object("late", shape="box", size=[0.015] * 3, position=[0.25, 0, 0.3])
        raise RuntimeError("whatever the user's code did")
    assert [o.name for o in sim.scene.objects] == names and "late" not in str(sim.describe()["objects"])
    with pytest.raises(KeyError):
        sim.backend.body_pose("late")  # describe() and the physics agree: no such body anywhere
    assert not sim._pending
    sim.step(1)
    arm.act(dict.fromkeys(arm.action_keys, 0.0))  # the policy's world is the described one


def test_a_caught_inner_block_rolls_back_only_its_own_edits(world):
    sim, arm = world
    with sim.editing():
        sim.add_object("a", shape="box", size=[0.015] * 3, position=[0.25, 0, 0.3])
        sim.remove(arm)
        with pytest.raises(RuntimeError, match="helper"), sim.editing():
            sim.add_object("b", shape="box", size=[0.015] * 3, position=[0.3, 0, 0.3])
            raise RuntimeError("a helper's validation failed")
    assert sim.backend.body_pose("a") is not None and [o["name"] for o in sim.describe()["objects"]] == ["cube", "a"]
    assert arm.sim is None and list(sim.robots) == []  # the outer remove still detaches on the outer rebuild


def test_set_pose_on_a_body_added_in_the_same_editing_block_is_placed_by_the_blocks_one_rebuild(world, monkeypatch):
    """Spawn-then-place inside one block is the per-episode reset shape editing() exists for; it used to KeyError on
    the not-yet-built body and roll the whole block back."""
    sim, _ = world
    loads = []
    real = sim.backend.load
    monkeypatch.setattr(sim.backend, "load", lambda scene: (loads.append(1), real(scene))[1])
    with sim.editing():
        ball = sim.add_object("ball", shape="sphere", size=0.02, position=[0.4, 0, 0.3], free=False)
        got = ball.set_pose(position=[0.3, 0.1, 0.3])
        late = Robot("so101", sim=sim, id="late", position=[0.6, 0, 0])
        sim.set_pose(late, position=[0.7, 0, 0])
        assert loads == []  # nothing rebuilt yet
    assert loads == [1]  # the block's one rebuild
    assert np.allclose(got.position, [0.3, 0.1, 0.3])
    assert np.allclose(sim.pose("ball").position, [0.3, 0.1, 0.3])
    assert np.allclose(sim.pose("late").position, [0.7, 0, 0], atol=1e-6) and "late" in sim.robots
    assert np.allclose(sim.describe()["robots"]["late"]["position"], [0.7, 0, 0])


@pytest.mark.parametrize("spawn", [(0.0, 0.0, 0.0), (0.2, 0.3, 0.0)])
def test_a_second_set_pose_of_a_fixed_robot_in_one_editing_block_lands_where_asked(spawn):
    """Inside a block the backend is not rebuilt, so ``cur`` stays the pre-block pose while the scene entry moved on
    the first call; composing the second delta onto the moved entry added the first delta again ([1.5, 0, 0])."""
    sim = Simulation()
    arm = Robot("so101", sim=sim, position=spawn)
    try:
        with sim.editing():
            sim.set_pose(arm, position=(0.5, 0, 0))
            sim.set_pose(arm, position=(1.0, 0, 0))
        assert np.allclose(sim.pose(arm.id).position, [1.0, 0, 0], atol=1e-6), sim.pose(arm.id).position
        sim.set_pose(arm, position=(0.7, 0.1, 0))  # and outside a block, after the rebuild, still exact
        assert np.allclose(sim.pose(arm.id).position, [0.7, 0.1, 0], atol=1e-6)
        with sim.editing():  # a later block composes onto the frame the LAST rebuild compiled
            sim.set_pose(arm, position=(0.2, 0.2, 0))
        assert np.allclose(sim.pose(arm.id).position, [0.2, 0.2, 0], atol=1e-6)
    finally:
        sim.close()


def test_a_partial_set_pose_of_a_built_fixed_body_in_one_editing_block_keeps_the_earlier_move():
    """Yin on #4790: inside a block ``cur`` is the pre-block backend pose, so a position-only move followed by an
    orientation-only move of the same built fixed body filled the omitted position from ``cur`` and silently
    dropped the first move, for a fixed-base robot and a ``free=False`` object alike. The omitted component now
    comes from the pending pose (the moved scene entry), so both calls land."""
    sim = Simulation()
    arm = Robot("so101", sim=sim, position=(0.1, 0.2, 0))
    try:
        sim.add_object("post", shape="box", size=[0.02] * 3, position=[0.4, 0.4, 0.1], free=False)
        sim.set_pose(arm, position=(0.5, 0, 0), orientation=(0.7071, 0, 0, 0.7071))
        with sim.editing():
            sim.set_pose(arm, position=(0.9, 0.1, 0))
            second = sim.set_pose(arm, orientation=(1, 0, 0, 0))
            assert np.allclose(second.position, [0.9, 0.1, 0], atol=1e-6), second.position
            sim.set_pose("post", position=(0.6, 0.6, 0.1))
            sim.set_pose("post", orientation=(0.7071, 0, 0, 0.7071))
        assert np.allclose(sim.pose(arm.id).position, [0.9, 0.1, 0], atol=1e-6), sim.pose(arm.id).position
        assert np.allclose(np.abs(sim.pose(arm.id).orientation), [1, 0, 0, 0], atol=1e-4), sim.pose(arm.id).orientation
        assert np.allclose(sim.pose("post").position, [0.6, 0.6, 0.1], atol=1e-6), sim.pose("post").position
        assert np.allclose(np.abs(sim.pose("post").orientation), [0.7071, 0, 0, 0.7071], atol=1e-3)
        with sim.editing():  # the other order: turn, then move
            sim.set_pose(arm, orientation=(0.7071, 0, 0, 0.7071))
            sim.set_pose(arm, position=(0.3, 0.3, 0))
        assert np.allclose(sim.pose(arm.id).position, [0.3, 0.3, 0], atol=1e-6)
        assert np.allclose(np.abs(sim.pose(arm.id).orientation), [0.7071, 0, 0, 0.7071], atol=1e-3)
    finally:
        sim.close()


def test_remove_leaves_no_draw_behind_for_a_body_or_camera_re_added_under_the_same_name(world):
    sim, _ = world
    base = sim.backend.model.body_mass[sim.backend.model.body("cube").id]
    sim.randomize(target="cube", mass=3.0)
    sim.remove("cube")
    sim.add_object("cube", shape="box", size=[0.015] * 3, position=[0.25, 0, 0.3], mass=0.05)
    assert sim.scene.scales == {} and sim.backend.model.body_mass[sim.backend.model.body("cube").id] == pytest.approx(
        base
    )

    sim.camera("front", position=[0.5, -0.5, 0.4], size=(32, 24))
    sim.randomize(camera=0.02, seed=3)
    assert sim.scene.camera_jitter.get("front")
    sim.remove("front")
    sim.camera("front", position=[0.5, -0.5, 0.4], size=(32, 24))
    assert "front" not in sim.scene.camera_jitter
    cam = sim.backend.model.camera("front")
    assert np.allclose(sim.backend.model.cam_pos[cam.id], [0.5, -0.5, 0.4], atol=1e-9)

    sim.camera("wrist", attach="so101", size=(32, 24))  # a robot's camera leaves with the robot, draw included
    sim.randomize(camera=0.02, seed=4)
    assert "wrist" in sim.scene.camera_jitter
    sim.remove("so101")
    assert "wrist" not in sim.scene.camera_jitter and "so101" not in sim.scene.scales


def test_mujoco_replays_bit_exact_and_says_so():
    def episode():
        with Simulation() as sim:
            arm = Robot("so101", sim=sim)
            sim.add_object("cube", shape="box", size=[0.015] * 3, position=[0.25, 0, 0.3], mass=0.05)
            out = []
            with sim.manual_time():
                for i in range(50):
                    arm.act({k: float(0.3 * np.sin(i / 20 + j)) for j, k in enumerate(arm.action_keys)})
                    sim.step(10)
                    out.append(np.concatenate([arm.observe().joints.positions, sim.backend.body_pose("cube").position]))
            return sim.describe()["physics"]["deterministic"], np.array(out)

    (flag_a, a), (_, b) = episode(), episode()
    assert flag_a is True and np.array_equal(a, b)


@pytest.mark.skipif(not has_warp, reason="mujoco_warp/warp not installed (needs a CUDA box)")
@pytest.mark.parametrize("num_envs, deterministic", [(1, True), (8, False)])
def test_newton_is_deterministic_only_in_one_world(num_envs, deterministic):
    with Simulation(physics="newton", num_envs=num_envs) as sim:
        assert sim.describe()["physics"]["deterministic"] is deterministic


def test_newton_names_a_slow_kernel_compile_and_the_way_out(monkeypatch, caplog):
    from contextlib import nullcontext

    from strands_robots.simulation.backends.newton import stepper as st

    s = object.__new__(st.NewtonStepper)
    s.device, s.num_envs = "cpu", 1
    s.mjw = types.SimpleNamespace(put_model=lambda m: m, put_data=lambda m, d, nworld: d)
    s.wp = types.SimpleNamespace(
        ScopedDevice=lambda d: nullcontext(), config=types.SimpleNamespace(kernel_cache_dir="/k")
    )
    monkeypatch.setattr(st, "SLOW_COMPILE_S", -1.0)
    with caplog.at_level(logging.WARNING):
        s.load(object(), object())
    assert (
        "compile this scene shape" in caplog.text
        and "WARP_CACHE_PATH" in caplog.text
        and "sim.editing()" in caplog.text
    )


def _obs(q, img=None):
    names = ("a", "b")
    return Observation(0.0, JointState(names, np.asarray(q, float), np.zeros(2), np.zeros(2)), images=img or {})


def test_perturb_delays_actions_and_holds_the_measured_pose_until_the_first_lands():
    p = Perturb(SequencePolicy([np.array([1.0, 1.0]), np.array([2.0, 2.0]), np.array([3.0, 3.0])]), action_delay=2)
    out = [p.act(_obs([0.1, 0.2])) for _ in range(4)]
    assert np.allclose(out[0], [0.1, 0.2])
    assert [float(np.asarray(a)[0]) for a in out[2:]] == [1.0, 2.0]
    p.reset()
    assert np.allclose(p.act(_obs([0.0, 0.0])), [0.0, 0.0])  # reset empties the pipe


class _UnitsFor(SequencePolicy):  # declares its units the way a LeRobot checkpoint does
    def units_for(self, description):
        from strands_robots.units.scheme import Units

        return Units.for_robot(description, "lerobot")


@pytest.mark.parametrize("policy, units", [(_UnitsFor, "auto"), (SequencePolicy, "lerobot"), (SequencePolicy, "auto")])
def test_perturb_holds_in_whatever_units_the_loop_converts(world, policy, units):
    sim, arm = world
    q = np.zeros(len(arm.description.action_keys))
    arm.run(Perturb(policy([q] * 5), action_delay=2), steps=5, units=units)  # the hold is a vector the loop accepts


def test_perturb_noise_reaches_only_the_policy_and_is_seeded():
    seen = []
    p = Perturb(lambda o: seen.append(o) or o.joints.positions, joint_noise=0.01, image_noise=0.05, seed=7)
    img = np.full((4, 4, 3), 128, np.uint8)
    obs = _obs([0.5, -0.5], {"wrist": img})
    p.act(obs)
    assert not np.array_equal(seen[0].joints.positions, obs.joints.positions) and np.allclose(
        seen[0].joints.positions, obs.joints.positions, atol=0.1
    )
    assert not np.array_equal(seen[0].images["wrist"], img) and np.array_equal(obs.images["wrist"], img)
    p.reset()
    p.act(obs)
    assert np.array_equal(seen[0].joints.positions, seen[1].joints.positions)  # same seed, same noise
    with pytest.raises(ValueError, match=">= 0"):
        Perturb(lambda o: o, action_delay=-1)


def test_isaac_never_accepts_its_eula_for_the_user(monkeypatch):
    from strands_robots.simulation.backends.isaac import app

    started = []
    fake = types.ModuleType("isaacsim")
    fake.__spec__ = importlib.machinery.ModuleSpec("isaacsim", None)
    fake.SimulationApp = lambda cfg: started.append(cfg)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaacsim", fake)
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    monkeypatch.setattr(app, "_APP", None)
    with pytest.raises(RuntimeError, match="accept the NVIDIA Isaac Sim EULA yourself.*OMNI_KIT_ACCEPT_EULA=YES"):
        app.simulation_app(headless=True)
    assert started == [] and "OMNI_KIT_ACCEPT_EULA" not in __import__("os").environ
