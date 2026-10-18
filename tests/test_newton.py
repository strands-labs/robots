"""Simulation("mujoco", physics="newton"): the same MuJoCo world stepped by MuJoCo-Warp on the GPU, ``num_envs`` copies.

Newton is a solver choice inside a world, never a third engine. Runs where mujoco_warp + warp are installed (a CUDA
box); everywhere else the GPU tests skip with the reason and only the refusals run.
"""

from __future__ import annotations

import importlib.util
import threading
import time

import numpy as np
import pytest

from strands_robots import Robot, Simulation

has_warp = importlib.util.find_spec("mujoco_warp") is not None and importlib.util.find_spec("warp") is not None
gpu = pytest.mark.skipif(not has_warp, reason="mujoco_warp/warp not installed (needs a CUDA box)")


def test_many_envs_need_newton_physics():
    with pytest.raises(ValueError, match="num_envs=4 needs physics='newton'"):
        Simulation(num_envs=4)


@pytest.mark.skipif(has_warp, reason="checks the install hint, only where mujoco_warp is missing")
def test_newton_without_mujoco_warp_says_how_to_install():
    with pytest.raises(ImportError, match=r"physics='newton' needs mujoco_warp \+ warp"):
        Simulation(physics="newton")


@gpu
@pytest.mark.parametrize("num_envs", [1, 1024])
def test_newton_steps_two_robots_in_one_world(num_envs):
    sim = Simulation("mujoco", physics="newton", num_envs=num_envs)
    try:
        arm = Robot("so101", sim=sim)
        g1 = Robot("g1", sim=sim, position=[1, 0, 0])
        assert sim.physics.name == "newton" and sim.envs == num_envs and arm.sim is g1.sim is sim
        sim.step(10)  # kernel compilation + graph capture
        n = 200
        t = time.perf_counter()
        sim.step(n)
        dt = time.perf_counter() - t
        key = arm.description.action_keys[0]
        t0 = sim.time
        for _ in range(50):
            arm.act({key: 0.3})
        assert sim.time > t0
        assert arm.observe()[key] == pytest.approx(0.3, abs=0.05)
        g1.set_pose(position=[2, 0, 0.8])
        assert g1.pose.position[0] == pytest.approx(2.0, abs=1e-3)
        print(
            f"\n[measured] newton num_envs={num_envs}: {n} steps in {dt * 1e3:.0f} ms = "
            f"{n * num_envs / dt:,.0f} env-steps/s ({n / dt:,.0f} steps/s)",
            flush=True,
        )
    finally:
        sim.close()


@gpu
def test_newton_time_is_exact_across_load_reset_and_teleport():
    sim = Simulation("mujoco", physics="newton")
    try:
        g1 = Robot("g1", sim=sim)
        ts = sim.physics.timestep
        assert sim.time == 0.0  # loading compiles and captures, it never steps
        # GPU time is float32: 1e-6 is far below one step (2e-3), so one extra step would fail every assert below
        sim.step(10)
        assert sim.time == pytest.approx(10 * ts, abs=1e-6)
        g1.set_pose(position=[0.5, 0, 0.8])  # a teleport re-uploads the world, it does not step it
        assert sim.time == pytest.approx(10 * ts, abs=1e-6)
        sim.reset()
        sim.step(10)
        assert sim.time == pytest.approx(10 * ts, abs=1e-6)
    finally:
        sim.close()


@gpu
def test_unlocked_reads_survive_a_background_run():
    """``sim.time`` (the watchdog's clock), ``contacts()`` and ``pose`` read while ``Robot.start`` steps the world."""
    sim = Simulation("mujoco", physics="newton")
    try:
        arm = Robot("so101", sim=sim)
        cube = sim.add_object("cube", shape="box", size=[0.02] * 3, position=[0.25, 0, 0.02])
        key = arm.description.action_keys[0]
        task = arm.start(lambda o: {key: 0.3}, hz=200, seconds=2)
        reads, t = 0, time.perf_counter()
        while time.perf_counter() - t < 2.5:
            _ = sim.time, sim.contacts(), arm.pose, cube.pose
            reads += 1
        task.stop()
        assert reads > 100 and sim.time > 0.0
    finally:
        sim.close()


@gpu
def test_concurrent_estops_stop_a_background_run():
    """The e-stop takes no world lock; on newton its hold must not race the run thread on the mirrored ``MjData``."""
    sim = Simulation("mujoco", physics="newton")
    try:
        arm = Robot("so101", sim=sim)
        key = arm.description.action_keys[0]
        for _ in range(10):
            task = arm.start(lambda o: {key: 0.3}, hz=500, seconds=30)
            time.sleep(0.05)
            stops = [threading.Thread(target=arm.estop) for _ in range(8)]
            for s in stops:
                s.start()
            for s in stops:
                s.join()
            t = time.perf_counter()
            while task.running and time.perf_counter() - t < 2.0:
                time.sleep(0.001)
            assert not task.running and task.result.stopped_by == "estop"
            arm.resume(approved_by="test")
    finally:
        sim.close()


def test_a_scene_change_puts_the_kept_gains_on_the_device_in_one_upload(monkeypatch):
    """No GPU needed: a stand-in for the device stepper records the gains each upload carries."""
    from strands_robots.simulation.backends.newton import backend as newton

    uploads: list[np.ndarray] = []

    class Device:
        def __init__(self, device, num_envs):
            self.d = None

        def load(self, model, data):
            uploads.append(model.actuator_gainprm[:, 0].copy())

        def set_ctrl(self, ids, targets):
            pass

        def close(self):
            pass

    monkeypatch.setattr(newton, "NewtonStepper", Device)
    with Simulation(physics="newton") as sim:
        arms = [Robot("so101", sim=sim), Robot("so101", sim=sim, position=[0, 0.5, 0])]
        for arm in arms:
            arm.set_gains(np.full(6, 33.0), np.full(6, 1.0))
        n = len(uploads)
        sim.add_object("box", shape="box", position=[0.3, 0, 0.1])
        ids = [sim.backend.model.actuator(f"{a.id}/{k}").id for a in arms for k in a.description.actuators]
        assert len(uploads) == n + 1  # one per scene change, not one more per robot with gains
        assert np.allclose(uploads[-1][ids], 33.0)  # dev: the MJCF's kp 17.8 went to the device
