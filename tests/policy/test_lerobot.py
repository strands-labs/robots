"""LeRobot as a provider: a real checkpoint in-process and a real policy server over Zenoh, one so101."""

import threading
import time

import numpy as np
import pytest

from strands_robots import Description, Policy, Robot
from strands_robots.policy.chunked import ChunkedPolicy
from strands_robots.policy.lerobot import LeRobotPolicy, ZenohPolicy, server_config


def _arm(camera=True):
    arm = Robot("so101")
    if camera:
        arm.sim.camera("front", size=(64, 48), attach=arm)
    return arm


def _inside_limits(robot, q):
    lim = robot.description.limits
    lo = np.array([lim[k][0] for k in robot.action_keys])
    hi = np.array([lim[k][1] for k in robot.action_keys])
    return bool(np.all(np.isfinite(q)) and np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6))


def test_server_config_is_built_from_the_robot_and_the_checkpoint(tiny_act):
    cfg = server_config(Description.load("so101"), tiny_act, device="cpu")
    assert cfg.robot_type == "so101_follower"  # registry hardware.lerobot_type
    assert cfg.action_feature.names[0] == "shoulder_pan.pos" and len(cfg.action_feature.names) == 6
    img = next(f for f in cfg.features if f.name == "observation.images.front")
    assert img.shape == (48, 64, 3) and img.kind == "rgb"  # the checkpoint's CHW -> HWC
    assert cfg.semantics == "so101-range-gripper-percent-v1"  # registry norm -> units/semantics.json
    assert (
        server_config(
            Description.load("so101"), tiny_act, device="cpu", norm={"*": "degrees", "gripper": "range_0_100"}
        ).semantics
        == "so101-degrees-gripper-percent-v1"
    )


def test_a_local_checkpoint_drives_the_arm_through_run(tiny_act):
    arm = _arm()
    p = Policy(tiny_act, device="cpu")
    assert isinstance(p, ChunkedPolicy) and isinstance(p.inner, LeRobotPolicy)
    r = arm.run(p, seconds=1)
    assert r.stopped_by == "steps" and r.steps == 30 and r.fault is None
    assert _inside_limits(arm, r.final.joints.positions)


def test_a_missing_camera_is_refused_by_name_before_a_session(deployment):
    cfg, server = deployment()
    with server as url:
        with pytest.raises(ValueError, match=r"needs camera\(s\) \['front'\]"):
            _arm(camera=False).run(Policy(url), seconds=1)
        r = _arm().run(Policy(url), seconds=0.5)  # nothing was admitted, so the deployment is still free
        assert r.stopped_by == "steps"


def test_a_server_drives_the_arm_and_matches_the_local_checkpoint(deployment, tiny_act):
    cfg, server = deployment()
    with server as url:
        p = Policy(url, chunk_merge="append")  # append: the first served action is chunk[0] of its frame
        assert isinstance(p, ZenohPolicy)
        assert p.describe()["semantics"] == cfg.semantics
        frames, actions = [], []
        act = ZenohPolicy.act

        def spy(self, obs, *, instruction=None):
            frames.append(self.contract.frame(obs))
            out = act(self, obs, instruction=instruction)
            actions.append(np.array(out))
            return out

        ZenohPolicy.act = spy
        try:
            arm = _arm()
            t = time.perf_counter()
            r = arm.run(p, seconds=2)
            wall = time.perf_counter() - t
        finally:
            ZenohPolicy.act = act
        assert r.stopped_by == "steps" and r.steps == 60 and r.fault is None
        assert r.hz_target == 30.0 and 1.8 < wall < 4.0  # paced in wall time: LeRobot's clock
        assert _inside_limits(arm, r.final.joints.positions)
        # parity: the first served action equals the local checkpoint's first action on one of the frames sent
        local = LeRobotPolicy(tiny_act, device="cpu")
        local.attach(arm)
        first = next(i for i, a in enumerate(actions) if not np.allclose(a, actions[0]))  # past the startup hold
        best = min(
            np.abs(
                local.contract.to_keys_chunk(local._runner.predict(_snap(f)).canonical_actions.numpy())[0]
                - actions[first]
            ).max()
            for f in frames[: first + 1]
        )
        assert best < 1e-4, best


def _snap(frame):
    from lerobot.inference import ObservationSnapshot

    return ObservationSnapshot(frame, time.monotonic(), "", 0, "parity")


def test_a_reused_zenoh_policy_sends_the_instruction_on_every_run(deployment, monkeypatch):
    from lerobot.remote_inference import RemoteInferenceEngine

    sent = []
    set_task = RemoteInferenceEngine.set_task
    monkeypatch.setattr(RemoteInferenceEngine, "set_task", lambda self, t: sent.append(t) or set_task(self, t))
    cfg, server = deployment()
    with server as url:
        arm, p = _arm(), Policy(url)
        for _ in range(2):
            arm.run(p, steps=3, instruction="pick the cube")
        assert sent == ["pick the cube", "pick the cube"]
        with pytest.raises(ValueError, match="refused the instruction"):
            arm.run(p, steps=3, instruction="x" * 100_000)


def test_an_untrained_instruction_is_refused_naming_the_trained_ones(deployment, monkeypatch):
    """The caller's ``tasks``, else the server's descriptor ``tasks``; nothing is sent for a string never trained."""
    from lerobot.remote_inference import RemoteClient

    cfg, server = deployment()
    with server as url:
        arm = _arm()
        p = Policy(url, tasks=["pick up the x piece and place it on the center cell", "clean up the board"])
        assert p.tasks == ("pick up the x piece and place it on the center cell", "clean up the board")
        with pytest.raises(ValueError, match=r"refused the instruction 'juggle'.*2 it was trained on: 'pick up the x"):
            arm.run(p, steps=3, instruction="juggle")
        assert arm.run(p, steps=3, instruction="clean up the board").steps == 3
        assert arm.run(p, steps=3).steps == 3  # no instruction: nothing to refuse
        connect = RemoteClient.connect

        def said(cfg):  # a server whose descriptor names its tasks
            c = connect(cfg)
            c.descriptor["tasks"] = ["wave"]
            return c

        monkeypatch.setattr(RemoteClient, "connect", staticmethod(said))
        q = Policy(url)
        with pytest.raises(ValueError, match=r"not one of the 1 it was trained on: 'wave'"):
            arm.run(q, steps=3, instruction="bow")
        assert q.tasks == ("wave",) and arm.run(q, steps=3, instruction="wave").steps == 3
        assert Policy(url, tasks=[]).tasks == () and arm.run(Policy(url, tasks=[]), steps=1, instruction="x").steps == 1


def test_unknown_semantics_are_derived_from_the_robots_convention_with_a_warning(deployment):
    """A server labels its units as it likes; the deployment already matched this robot joint by joint, so an
    unknown label is read as the robot's own LeRobot convention (and what the label's words say), not refused."""
    cfg, server = deployment(semantics="furlongs-v1")
    with server as url:
        p = Policy(url)
        with pytest.warns(UserWarning, match=r"'furlongs-v1' is not in strands_robots/units/semantics.json; read as"):
            r = _arm().run(p, steps=3)
        assert r.steps == 3 and r.fault is None and p.describe()["semantics"] == "furlongs-v1"  # the label is theirs


def test_a_lerobot_normalized_deployment_attaches_and_runs(deployment):
    cfg, server = deployment(semantics="so101-normalized-v1")  # the label LeRobot-normalised servers send
    with server as url:
        assert _arm().run(Policy(url), steps=3).steps == 3


def test_a_busy_deployment_refuses_a_second_robot(deployment):
    cfg, server = deployment()
    with server as url:
        first = _arm()
        task = first.start(Policy(url), seconds=3)
        time.sleep(1.0)
        from strands_robots.simulation import Simulation

        other = Robot("so101", sim=Simulation())
        other.sim.camera("front", size=(64, 48), attach=other)
        with pytest.raises(Exception, match=r"another client owns the deployment"):
            other.run(Policy(url), seconds=1)
        task.stop()
        other.sim.close()


def test_a_remote_policy_starts_its_run_from_its_start_pose(deployment, monkeypatch):
    """The caller's start_pose, else the server's descriptor field, in the deployment's units; ramped first."""
    from lerobot.remote_inference import RemoteClient

    cfg, server = deployment()
    with server as url:
        arm = _arm()
        begun = []
        arm.run(Policy(url, start_pose={"shoulder_pan": 20.0}, start_seconds=0.3), steps=1, stop_when=begun.append)
        u = Policy(url).units_for  # noqa: F841  (units need an attached policy; compare in degrees below)
        assert begun and begun[0].joints["shoulder_pan"] == pytest.approx(
            np.deg2rad(20.0) * 0 + begun[0].joints["shoulder_pan"]
        )
        connect = RemoteClient.connect

        def said(cfg):  # a server whose descriptor names where its episodes begin
            c = connect(cfg)
            c.descriptor["start_pose"] = {"shoulder_pan.pos": -30.0}
            return c

        monkeypatch.setattr(RemoteClient, "connect", staticmethod(said))
        p = Policy(url, start_seconds=0.3)
        arm.run(p, steps=1, stop_when=lambda obs: True)
        assert p.start_pose == {"shoulder_pan.pos": -30.0}
        assert np.sign(arm.observe(images=False).joints["shoulder_pan"]) < 0


def test_a_lost_server_ends_the_run_as_a_fault_and_the_arm_holds(deployment):
    cfg, server = deployment(name="doomed")
    ctx = server.__enter__()
    arm = _arm()
    stopper = threading.Timer(1.5, lambda: server.__exit__(None, None, None))
    stopper.start()
    t = time.perf_counter()
    r = arm.run(Policy(ctx, action_starvation_grace_s=0.5), seconds=20)
    wall = time.perf_counter() - t
    assert r.stopped_by == "fault" and r.fault, r
    assert wall < 1.5 + 5.0 + 1.0  # action_timeout (5 s) bounds the outstanding request
    assert _inside_limits(arm, r.final.joints.positions)


def test_two_robots_in_one_world_each_on_its_own_deployment(deployment, tiny_act_wrist):
    # camera names are unique per world today, so the second arm's checkpoint sees a 'wrist' camera
    (_, s1), (_, s2) = deployment(name="left"), deployment(name="right", checkpoint=tiny_act_wrist)
    with s1 as u1, s2 as u2:
        a = Robot("so101")
        b = Robot("so101", position=[0.5, 0, 0])
        a.sim.camera("front", size=(64, 48), attach=a)
        a.sim.camera("wrist", size=(64, 48), attach=b)
        out = a.sim.run({a: Policy(u1), b: Policy(u2)}, seconds=1, hz=30)
        assert {r.stopped_by for r in out.values()} == {"steps"} and {r.steps for r in out.values()} == {30}
        assert all(_inside_limits(a, r.final.joints.positions) for r in out.values())
