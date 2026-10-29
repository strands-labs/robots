"""invoke(): rpcs reach the device through its safety chain; refusals arrive verbatim; estop has its own path."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pytest

from strands_robots import Robot
from strands_robots.mesh import RequestRefused
from strands_robots.robot.safety import RequestRefused as DeviceRefused


@pytest.fixture
def served(mesh_pair):
    a, b = mesh_pair
    arm = Robot("so101")
    a.mount(arm, device_id="so101-a", labels={"role": "arm"})
    assert b.discover("so101-a", min_count=1, timeout=3.0)
    yield arm, a, b
    a.close()  # unmount before the robot goes
    arm.close()


def test_describe_and_observe_over_the_mesh(served):
    arm, _, b = served
    d = b.invoke("so101-a", "describe")
    assert d["device_id"] == arm.manifest.device_id and d["estop"] is False
    obs = b.invoke("so101-a", "observe")
    assert obs.joints.names == arm.observe().joints.names
    np.testing.assert_allclose(obs.joints.positions, arm.observe().joints.positions)


def test_act_moves_the_device_and_returns_its_observation(served):
    arm, _, b = served
    obs = b.invoke("so101-a", "act", action={"shoulder_pan": 0.3}, steps=50)
    assert obs.joints["shoulder_pan"] == pytest.approx(arm.observe().joints["shoulder_pan"])
    assert arm.observe().joints["shoulder_pan"] > 0.05


def test_out_of_range_act_is_refused_verbatim_at_the_caller(served):
    arm, _, b = served
    with pytest.raises(DeviceRefused) as local:
        arm.confirm_act({"shoulder_pan": 99.0})
    with pytest.raises(RequestRefused) as remote:
        b.invoke("so101-a", "act", action={"shoulder_pan": 99.0})
    assert str(remote.value) == str(local.value)
    assert RequestRefused is DeviceRefused


def test_unknown_rpc_and_unknown_args_are_refused(served):
    _, _, b = served
    with pytest.raises(LookupError, match="no rpc 'fly'"):
        b.invoke("so101-a", "fly")
    with pytest.raises(TypeError, match="speed"):
        b.invoke("so101-a", "act", action={"shoulder_pan": 0.1}, speed=3)


def test_an_absent_device_is_a_lookup_error(served):
    _, _, b = served
    with pytest.raises(LookupError, match="nobody"):
        b.invoke("nobody", "describe")


def test_estop_over_the_mesh_latches_and_does_not_wait_for_a_busy_device(served):
    arm, _, b = served
    started = threading.Event()
    real_act = arm.act

    def slow_act(*a, **k):
        started.set()
        time.sleep(1.0)
        return real_act(*a, **k)

    arm.act = slow_act
    outcome = []

    def busy():
        try:
            outcome.append(b.invoke("so101-a", "act", action={"shoulder_pan": 0.1}, timeout=5.0))
        except RequestRefused as exc:
            outcome.append(exc)

    t = threading.Thread(target=busy)
    t.start()
    assert started.wait(3.0)
    t0 = time.perf_counter()
    b.invoke("so101-a", "estop")
    assert time.perf_counter() - t0 < 0.5
    assert arm.estopped
    t.join(5.0)
    assert isinstance(outcome[0], RequestRefused), "the in-flight act must not land after the e-stop"
    with pytest.raises(RequestRefused):
        b.invoke("so101-a", "act", action={"shoulder_pan": 0.1})


def test_images_over_the_mesh_render_on_one_thread(mesh_pair):
    from strands_robots import Simulation

    a, b = mesh_pair
    sim = Simulation("mujoco", render_size=(64, 48))
    arm = Robot("so101", sim=sim, id="cam-arm")
    sim.camera("cam", position=(0.5, -0.4, 0.4), target=(0.0, 0.0, 0.15), size=(64, 48), attach=arm)
    a.mount(arm, watchdog_ms=None)
    assert b.discover("cam-arm", min_count=1, timeout=3.0)
    from concurrent.futures import ThreadPoolExecutor

    def grab(_):
        return b.invoke("cam-arm", "observe", images=True, timeout=10.0).images["cam"]

    with ThreadPoolExecutor(4) as pool:  # concurrent calls land on different rpc workers; one thread renders
        imgs = list(pool.map(grab, range(8)))
    assert all(img.shape == (48, 64, 3) and img.dtype == np.uint8 for img in imgs)
    a.close()
    sim.close()


class _LimpBus:
    """A hardware Body that counts holds: a hold on a limp arm is torque on plus a goal write."""

    supports_position_hold, calibration, resolution = True, None, None

    def __init__(self, names):
        self.names, self.holds = names, 0
        self.q = np.zeros(len(names))

    def read(self):
        from strands_robots import JointState

        return JointState(self.names, self.q, self.q * 0, self.q * 0)

    def write(self, targets):
        self.q = np.asarray(targets, dtype=float)

    def hold(self):
        self.holds += 1

    pose = lambda self: None  # noqa: E731
    images = extras = lambda self: {}  # noqa: E731
    estop = close = lambda self: None  # noqa: E731

    def calibrate(self, *, interactive=True):
        raise RuntimeError("no calibration")


@pytest.mark.parametrize("operator, holds", [(False, 0), (True, 1)])
def test_a_remote_hold_on_a_real_arm_asks_the_operator_first(mesh_pair, operator, holds):
    a, b = mesh_pair
    names = Robot("so101").description.action_keys
    bus = _LimpBus(names)
    real = Robot("so101", driver=bus, confirm=lambda targets: operator, twin=False)
    a.mount(real, device_id="real-arm")
    assert b.discover("real-arm", min_count=1, timeout=3.0)
    if operator:
        b.invoke("real-arm", "hold")
    else:
        with pytest.raises(RequestRefused, match="operator gate refused"):
            b.invoke("real-arm", "hold")
    assert bus.holds == holds  # refused: torque never went on
    a.close()
    real.close()


def test_a_device_slower_than_the_timeout_raises_timeout_error(served):
    arm, _, b = served
    real = arm.observe
    arm.observe = lambda **kw: (time.sleep(1.0), real(**kw))[1]
    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="no reply within 0.2"):
        b.invoke("so101-a", "observe", timeout=0.2)
    assert time.monotonic() - t0 < 0.8


def test_every_rpc_leaves_an_audit_record_naming_who_and_the_outcome(served, caplog):
    _, _, b = served
    caplog.set_level(logging.INFO, logger="strands_robots.audit")
    b.invoke("so101-a", "act", action={"shoulder_pan": 0.1})
    with pytest.raises(RequestRefused):
        b.invoke("so101-a", "act", action={"shoulder_pan": 99.0})
    acts = [r for r in caplog.records if r.name == "strands_robots.audit" and r.rpc == "act"]
    assert [(r.by, r.peer, r.outcome) for r in acts] == [
        (b.operator, b.peer_id, "ok"),
        (b.operator, b.peer_id, "RequestRefused"),
    ]


def test_one_act_cannot_hold_the_world_for_an_unbounded_number_of_steps(served):
    _, _, b = served
    with pytest.raises(ValueError, match="steps= must be an integer"):
        b.invoke("so101-a", "act", action={"shoulder_pan": 0.1}, steps=10**9)


def test_every_act_argument_the_manifest_advertises_is_served(served):
    arm, _, _ = served
    from strands_robots.mesh.serve import ARGS

    (act,) = [r for r in arm.manifest.rpcs if r.name == "act"]
    assert set(act.params["properties"]) <= set(ARGS["act"])


def test_act_seconds_reaches_the_pose_not_one_step_toward_it(served):
    arm, _, b = served
    obs = b.invoke("so101-a", "act", action={"shoulder_pan": 0.4}, seconds=1.0)
    assert obs.joints["shoulder_pan"] == pytest.approx(0.4, abs=0.02)
    assert arm.observe().joints["shoulder_pan"] == pytest.approx(0.4, abs=0.02)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"seconds": 0}, "seconds= must be"),
        ({"seconds": -1.0}, "seconds= must be"),
        ({"seconds": 11.0}, "seconds= must be"),
        ({"seconds": True}, "seconds= must be"),
        ({"seconds": "1"}, "seconds= must be"),
        ({"seconds": 1.0, "steps": 5}, "not both"),
    ],
)
def test_act_seconds_is_bounded(served, kw, match):
    _, _, b = served
    with pytest.raises(ValueError, match=match):
        b.invoke("so101-a", "act", action={"shoulder_pan": 0.1}, **kw)
