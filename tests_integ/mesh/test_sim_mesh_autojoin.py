"""Integration test: simulated robots auto-join zenoh mesh with full topic schema.

Verifies that simulated robots spawned via Simulation.add_robot() auto-join
the Zenoh mesh and publish their state/presence topics.

Problem statement: when robots are spawned inside a MuJoCo simulation, they
sometimes don't appear in the mesh or appear without their state topics
published. The root cause is that the child Mesh created for a SimRobot
dataclass cannot read joint state (since SimRobot lacks the .robot.get_observation()
path that _read_state() expects from hardware robots).

This test verifies:
1. N=3 simulated robots each get their own mesh peer on add_robot()
2. All 3 appear in the peer registry within 2s
3. Each peer publishes state data (joints/sim_time) on the mesh
4. A 4th observer peer can discover all 3 sim-robot peers
5. remove_robot() stops the child mesh peer

Requires: eclipse-zenoh, mujoco, MUJOCO_GL=egl (headless)
"""

from __future__ import annotations

import json
import time

import pytest

zenoh = pytest.importorskip("zenoh", reason="mesh integ tests require eclipse-zenoh")
mujoco = pytest.importorskip("mujoco", reason="sim mesh tests require mujoco")


@pytest.fixture(autouse=True)
def _mesh_local_dev(monkeypatch):
    """Enable mesh without TLS for integration testing."""
    monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.delenv("STRANDS_MESH", raising=False)


@pytest.fixture
def sim_with_mesh():
    """Create a Simulation instance that is itself on a mesh."""
    from strands_robots.mesh import init_mesh
    from strands_robots.simulation.mujoco.simulation import Simulation

    sim = Simulation()

    # Give the sim a mesh identity (normally done by the Robot wrapper)
    sim_mesh = init_mesh(sim, peer_id="sim-integ-test", peer_type="sim", mesh=True)
    sim.mesh = sim_mesh
    sim.peer_id = sim_mesh.peer_id if sim_mesh else ""

    yield sim

    # Cleanup
    sim.cleanup()
    if sim_mesh:
        sim_mesh.stop()


@pytest.fixture
def observer_mesh():
    """Create an observer Mesh peer that watches but doesn't control."""
    from strands_robots.mesh import Mesh

    class _Observer:
        tool_name_str = "observer"

    obs = Mesh(_Observer(), peer_id="observer-integ", peer_type="dashboard")
    obs.start()
    yield obs
    obs.stop()


class TestSimMeshAutojoin:
    """Simulated robots auto-join mesh with full topic presence."""

    def test_spawn_3_sim_robots_appear_in_peer_registry(self, sim_with_mesh):
        """Spawn N=3 in MuJoCo; all 3 appear in zenoh peer registry within 3s."""
        sim = sim_with_mesh
        if not sim.mesh or not sim.mesh.alive:
            pytest.skip("Mesh not available (zenoh session failed to start)")

        result = sim.create_world()
        assert result["status"] == "success", f"create_world failed: {result}"

        robot_names = ["arm_a", "arm_b", "arm_c"]
        for name in robot_names:
            result = sim.add_robot(name, data_config="so100")
            assert result["status"] == "success", f"add_robot {name} failed: {result}"

        # Verify all 3 robots got mesh peers
        for name in robot_names:
            robot = sim._world.robots[name]
            assert robot.peer_id, f"Robot '{name}' has no peer_id after add_robot"
            assert robot.mesh is not None, f"Robot '{name}' has no mesh after add_robot"
            assert robot.mesh.alive, f"Robot '{name}' mesh is not alive"

        # Wait for peers to appear in registry (heartbeat at 2 Hz)
        deadline = time.time() + 3.0
        expected_peer_ids = {sim._world.robots[n].peer_id for n in robot_names}

        while time.time() < deadline:
            visible_ids = {p["peer_id"] for p in sim.mesh.peers}
            if expected_peer_ids.issubset(visible_ids):
                break
            time.sleep(0.2)

        visible_ids = {p["peer_id"] for p in sim.mesh.peers}
        missing = expected_peer_ids - visible_ids
        assert not missing, f"Robots not visible in mesh after 3s: {missing}. Visible peers: {visible_ids}"

    def test_observer_discovers_all_sim_robots(self, sim_with_mesh, observer_mesh):
        """A 4th observer peer sees all 3 sim-robot peers."""
        sim = sim_with_mesh
        obs = observer_mesh
        if not sim.mesh or not sim.mesh.alive:
            pytest.skip("Mesh not available")
        if not obs.alive:
            pytest.skip("Observer mesh not available")

        result = sim.create_world()
        assert result["status"] == "success"

        robot_names = ["obs_a", "obs_b", "obs_c"]
        for name in robot_names:
            result = sim.add_robot(name, data_config="so100")
            assert result["status"] == "success", f"add_robot {name}: {result}"

        expected_peer_ids = {sim._world.robots[n].peer_id for n in robot_names}

        deadline = time.time() + 3.0
        while time.time() < deadline:
            obs_visible = {p["peer_id"] for p in obs.peers}
            if expected_peer_ids.issubset(obs_visible):
                break
            time.sleep(0.2)

        obs_visible = {p["peer_id"] for p in obs.peers}
        missing = expected_peer_ids - obs_visible
        assert not missing, f"Observer cannot see robots: {missing}. Observer sees: {obs_visible}"

    def test_sim_robot_presence_includes_robot_type(self, sim_with_mesh):
        """Each sim robot's presence payload has peer_type='robot'."""
        sim = sim_with_mesh
        if not sim.mesh or not sim.mesh.alive:
            pytest.skip("Mesh not available")

        result = sim.create_world()
        assert result["status"] == "success"
        result = sim.add_robot("typed_bot", data_config="so100")
        assert result["status"] == "success"

        robot = sim._world.robots["typed_bot"]
        assert robot.mesh is not None
        assert robot.mesh.peer_type == "robot"

        time.sleep(2.0)
        peer_info = sim.mesh.get_peer(robot.peer_id)
        if peer_info is not None:
            assert peer_info.get("robot_type") == "robot"

    def test_sim_robot_state_published_on_mesh(self, sim_with_mesh):
        """Each sim robot publishes joint state on the mesh (the core bug).

        The bug: _read_state() in Mesh reads from self.robot.robot (expecting
        a lerobot Robot) but for sim robots self.robot is a SimRobot dataclass
        with no .robot attribute. Result: state is never published.

        After the fix, the child mesh should read joint positions from the
        parent Simulation's world data via a bridge on the SimRobot.
        """
        sim = sim_with_mesh
        if not sim.mesh or not sim.mesh.alive:
            pytest.skip("Mesh not available")

        result = sim.create_world()
        assert result["status"] == "success"
        result = sim.add_robot("state_bot", data_config="so100")
        assert result["status"] == "success"

        robot = sim._world.robots["state_bot"]
        if not robot.mesh or not robot.mesh.alive:
            pytest.skip("Robot mesh not alive")

        # Step physics so joints have non-trivial values
        sim.step(n_steps=10)

        # Subscribe to the state topic for this robot
        state_received: list[dict] = []
        from strands_robots.mesh.session import current_session

        session = current_session()
        if session is None:
            pytest.skip("No zenoh session")

        topic = f"strands/{robot.peer_id}/state"

        def on_state(sample):
            try:
                data = json.loads(sample.payload.to_bytes().decode())
                state_received.append(data)
            except Exception:
                pass

        sub = session.declare_subscriber(topic, on_state)
        try:
            # Wait for state publication (STATE_HZ=10, so ~100ms per tick)
            deadline = time.time() + 3.0
            while time.time() < deadline and not state_received:
                time.sleep(0.1)

            # THIS IS THE BUG ASSERTION: if no state is received, the
            # sim robot's mesh is not publishing joints.
            assert state_received, (
                f"No state published on '{topic}' within 3s. "
                "SimRobot mesh child is not bridging joint state from the "
                "parent Simulation world."
            )

            # Verify the state payload has joint data
            last_state = state_received[-1]
            assert "joints" in last_state or "sim_time" in last_state, (
                f"State payload missing joints/sim_time: {last_state}"
            )
        finally:
            sub.undeclare()

    def test_peer_id_encodes_parent_and_robot_name(self, sim_with_mesh):
        """Child peer_id follows the <parent>__<robot> convention."""
        sim = sim_with_mesh
        if not sim.mesh or not sim.mesh.alive:
            pytest.skip("Mesh not available")

        result = sim.create_world()
        assert result["status"] == "success"
        result = sim.add_robot("naming_test", data_config="so100")
        assert result["status"] == "success"

        robot = sim._world.robots["naming_test"]
        assert "__naming_test" in robot.peer_id, f"peer_id should contain '__naming_test', got: {robot.peer_id}"

    def test_remove_robot_detaches_from_mesh(self, sim_with_mesh):
        """remove_robot() stops the child mesh peer."""
        sim = sim_with_mesh
        if not sim.mesh or not sim.mesh.alive:
            pytest.skip("Mesh not available")

        result = sim.create_world()
        assert result["status"] == "success"
        result = sim.add_robot("removable", data_config="so100")
        assert result["status"] == "success"

        robot = sim._world.robots["removable"]
        child_mesh = robot.mesh
        assert child_mesh is not None and child_mesh.alive

        # Remove the robot
        result = sim.remove_robot("removable")
        assert result["status"] == "success"

        # Child mesh should be stopped (immediate, no need to wait for prune)
        assert not child_mesh.alive, "Child mesh still alive after remove_robot"
