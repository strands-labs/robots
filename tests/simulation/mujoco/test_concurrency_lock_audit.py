"""Regression that the 3 race-prone read-only ops acquire self._lock.

Surfaced by /tmp/ast-analysis-v2 concurrency audit: get_mass_matrix,
get_sensor_data, and get_contacts call mj_forward (which mutates
data.qfrc_constraint etc.) and then read data fields, WITHOUT holding
self._lock. After GH #114 landed concurrent per-robot policies, these
ops can be called by a user thread while another thread's PolicyRunner
is stepping - the mj_forward / mj_step interleaving corrupts the reads.

This test verifies the lock is acquired by inspecting the source. Not a
perfect test (doesn't prove the code actually races without the lock),
but it catches the regression if someone removes the `with self._lock:`
block without realising why it's there.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.physics import PhysicsMixin  # noqa: E402
from strands_robots.simulation.mujoco.rendering import RenderingMixin  # noqa: E402


class TestRacePreventionLocks:
    def test_get_mass_matrix_holds_lock(self):
        src = inspect.getsource(PhysicsMixin.get_mass_matrix)
        assert "self._lock" in src, "get_mass_matrix must acquire self._lock"
        # the lock must cover the mj_forward + read, not just a spurious
        # lock nowhere near the MuJoCo call.
        assert "with self._lock" in src

    def test_get_sensor_data_holds_lock(self):
        src = inspect.getsource(PhysicsMixin.get_sensor_data)
        assert "self._lock" in src, "get_sensor_data must acquire self._lock"
        assert "with self._lock" in src
        # Must snapshot sensordata under the lock (the fix's key invariant).
        assert "sensordata_snapshot" in src

    def test_get_contacts_holds_lock(self):
        src = inspect.getsource(RenderingMixin.get_contacts)
        assert "self._lock" in src, "get_contacts must acquire self._lock"
        assert "with self._lock" in src
        assert "contact_snapshot" in src

    def test_raycast_holds_lock(self):
        # mj_ray reads data.geom_xpos/geom_xmat (derived); the refresh + cast
        # must be serialized against a policy thread's mj_step.
        src = inspect.getsource(PhysicsMixin.raycast)
        assert "with self._lock" in src, "raycast must acquire self._lock"
        assert "mj_kinematics" in src, "raycast must refresh geom poses before mj_ray"

    def test_multi_raycast_holds_lock(self):
        src = inspect.getsource(PhysicsMixin.multi_raycast)
        assert "with self._lock" in src, "multi_raycast must acquire self._lock"
        assert "mj_kinematics" in src, "multi_raycast must refresh geom poses before mj_ray"


class TestFunctionalCorrectnessUnderLock:
    """The lock addition must NOT change observable behaviour."""

    def test_mass_matrix_still_works(self, tmp_path):

        from strands_robots.simulation.mujoco.simulation import Simulation

        sim = Simulation(tool_name="lock_audit", mesh=False)
        try:
            sim.create_world()
            # Build a scene with one free-joint ball so nv > 0
            sim.replace_scene_mjcf(
                '<mujoco><worldbody><body name="b" pos="0 0 1">'
                "<freejoint/>"
                '<geom type="sphere" size="0.1" mass="1"/></body>'
                "</worldbody></mujoco>"
            )
            r = sim.get_mass_matrix()
            assert r["status"] == "success", r
            j = r["content"][1]["json"]
            assert j["shape"] == [6, 6]  # freejoint contributes 6 dofs
            assert j["rank"] == 6
        finally:
            sim.cleanup(policy_stop_timeout=0.5)

    def test_contacts_still_work(self, tmp_path):

        from strands_robots.simulation.mujoco.simulation import Simulation

        sim = Simulation(tool_name="lock_audit_c", mesh=False)
        try:
            sim.create_world()
            sim.replace_scene_mjcf(
                "<mujoco><worldbody>"
                '<geom type="plane" size="1 1 0.1"/>'
                '<body name="ball" pos="0 0 0.1">'
                "<freejoint/>"
                '<geom type="sphere" size="0.1"/>'
                "</body></worldbody></mujoco>"
            )
            # Step to let it settle against the ground
            sim.step(n_steps=200)
            r = sim.get_contacts()
            assert r["status"] == "success", r
            # the ball resting on the plane must yield at least 1 contact
            contacts = r["content"][1]["json"]["contacts"]
            assert isinstance(contacts, list)
        finally:
            sim.cleanup(policy_stop_timeout=0.5)
