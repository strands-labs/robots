"""invoke(): rpcs reach the device through its safety chain; refusals arrive verbatim; estop has its own path."""

from __future__ import annotations

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
