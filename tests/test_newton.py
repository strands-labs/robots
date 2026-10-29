"""Simulation("mujoco", physics="newton"): the same MuJoCo world stepped by MuJoCo-Warp on the GPU, ``num_envs`` copies.

Newton is a solver choice inside a world, never a third engine. Runs where mujoco_warp + warp are installed (a CUDA
box); everywhere else the GPU tests skip with the reason and only the refusals run.
"""

from __future__ import annotations

import importlib.util
import threading
import time

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


@pytest.mark.parametrize(("verb", "args"), [("reach", ("so101",)), ("move_camera", (None, None))])
def test_a_view_read_takes_the_mirror_lock_and_syncs_first(verb, args, monkeypatch):
    # reach() / move_camera() run mj_* on the shared MjData; the e-stop syncs it under _mirror only (no GPU needed)
    from strands_robots.simulation.backends.mujoco.view import View
    from strands_robots.simulation.backends.newton.backend import NewtonBackend

    b = object.__new__(NewtonBackend)
    b._mirror, seen = threading.RLock(), []
    monkeypatch.setattr(b, "_fresh", lambda: seen.append("fresh"), raising=False)
    monkeypatch.setattr(View, verb, lambda self, *a: seen.append(b._mirror._is_owned()))
    getattr(b, verb)(*args)
    assert seen == ["fresh", True]
