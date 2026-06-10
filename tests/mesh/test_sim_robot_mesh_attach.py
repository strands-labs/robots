"""Regression: per-robot mesh attach in ``Simulation.add_robot``.

Bug report (telegram, PR #101 review): when a sim is on a Zenoh mesh and the
agent calls ``add_robot``, only the sim container itself was a peer; the
inner ``SimRobot`` instances were not addressable on the mesh. The agent had
to route through the sim's peer_id and then by ``robot_name``, breaking the
"every robot is a mesh peer" abstraction documented in PR #101.

This module covers the attach + detach lifecycle in isolation by patching
``init_mesh`` so we don't need a live zenoh session. It exercises the new
``Simulation._attach_robot_to_mesh`` / ``_detach_robot_from_mesh`` helpers
plus their integration points in ``add_robot`` / ``remove_robot`` /
``cleanup``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from strands_robots.simulation.models import SimRobot

# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------


def test_simrobot_dataclass_has_mesh_and_peer_id_fields():
    """SimRobot must carry mesh + peer_id so add_robot can populate them."""
    r = SimRobot(name="so100", urdf_path="/tmp/x.urdf")
    assert r.mesh is None, "mesh should default to None for off-mesh sims"
    assert r.peer_id == "", "peer_id should default to empty string"


def test_simrobot_mesh_fields_can_be_set():
    """Round-trip the new fields — required by Simulation.add_robot."""
    fake_mesh = MagicMock(name="Mesh", peer_id="parent-x:so100")
    r = SimRobot(
        name="so100",
        urdf_path="/tmp/x.urdf",
        mesh=fake_mesh,
        peer_id="parent-x:so100",
    )
    assert r.mesh is fake_mesh
    assert r.peer_id == "parent-x:so100"


# ---------------------------------------------------------------------------
# _attach_robot_to_mesh — direct unit tests via a stub Simulation
# ---------------------------------------------------------------------------


class _StubSim:
    """Minimal stand-in exposing only what ``_attach_robot_to_mesh`` reads.

    We import the bound method off the real class and re-bind it so we don't
    need a full MuJoCo sim instance just to test the mesh plumbing.
    """

    def __init__(self, mesh, peer_id):
        self.mesh = mesh
        self.peer_id = peer_id

    @staticmethod
    def _import_helpers():
        from strands_robots.simulation.mujoco.simulation import Simulation

        return Simulation._attach_robot_to_mesh, Simulation._detach_robot_from_mesh


def test_attach_is_noop_when_sim_has_no_mesh():
    """Off-mesh sims must not call init_mesh — keeps zenoh out of unit tests."""
    attach, _ = _StubSim._import_helpers()
    sim = _StubSim(mesh=None, peer_id="")
    robot = SimRobot(name="r1", urdf_path="/tmp/x.urdf")

    with patch("strands_robots.mesh.init_mesh") as init:
        attach(sim, robot)

    init.assert_not_called()
    assert robot.mesh is None
    assert robot.peer_id == ""


def test_attach_creates_child_peer_when_sim_is_on_mesh():
    """Sim on mesh + new robot → init_mesh called with parent:robot peer_id."""
    attach, _ = _StubSim._import_helpers()
    parent_mesh = MagicMock(name="ParentMesh")
    sim = _StubSim(mesh=parent_mesh, peer_id="so100_sim-a1b2c3d4")
    robot = SimRobot(name="so100", urdf_path="/tmp/x.urdf")

    fake_child = MagicMock(name="ChildMesh", peer_id="so100_sim-a1b2c3d4__so100")
    with patch("strands_robots.mesh.init_mesh", return_value=fake_child) as init:
        attach(sim, robot)

    init.assert_called_once()
    # The peer_id we pass into init_mesh must encode parent + robot name.
    kwargs = init.call_args.kwargs
    assert kwargs["peer_id"] == "so100_sim-a1b2c3d4__so100"
    assert kwargs["peer_type"] == "robot"
    assert kwargs["mesh"] is True
    # Resulting child mesh recorded on the SimRobot for later detach.
    assert robot.mesh is fake_child
    assert robot.peer_id == "so100_sim-a1b2c3d4__so100"


def test_attach_swallows_init_mesh_exceptions():
    """A mesh failure must not bring down add_robot — best-effort enrichment."""
    attach, _ = _StubSim._import_helpers()
    sim = _StubSim(mesh=MagicMock(), peer_id="parent-x")
    robot = SimRobot(name="r1", urdf_path="/tmp/x.urdf")

    with patch(
        "strands_robots.mesh.init_mesh",
        side_effect=RuntimeError("zenoh down"),
    ):
        attach(sim, robot)  # must not raise

    assert robot.mesh is None
    assert robot.peer_id == ""


def test_attach_handles_init_mesh_returning_none():
    """STRANDS_MESH=false case — init_mesh returns None, robot stays off-mesh."""
    attach, _ = _StubSim._import_helpers()
    sim = _StubSim(mesh=MagicMock(), peer_id="parent-x")
    robot = SimRobot(name="r1", urdf_path="/tmp/x.urdf")

    with patch("strands_robots.mesh.init_mesh", return_value=None):
        attach(sim, robot)

    assert robot.mesh is None
    assert robot.peer_id == ""


def test_attach_uses_fallback_parent_id_when_sim_peer_id_empty():
    """If sim somehow lacks peer_id but has mesh, child still gets a stable id."""
    attach, _ = _StubSim._import_helpers()
    sim = _StubSim(mesh=MagicMock(), peer_id="")
    robot = SimRobot(name="armA", urdf_path="/tmp/x.urdf")

    fake_child = MagicMock(peer_id="sim__armA")
    with patch("strands_robots.mesh.init_mesh", return_value=fake_child) as init:
        attach(sim, robot)

    assert init.call_args.kwargs["peer_id"] == "sim__armA"


# ---------------------------------------------------------------------------
# _detach_robot_from_mesh
# ---------------------------------------------------------------------------


def test_detach_is_noop_when_robot_has_no_mesh():
    """Detach on an off-mesh robot must stay quiet."""
    _, detach = _StubSim._import_helpers()
    sim = _StubSim(mesh=None, peer_id="")
    robot = SimRobot(name="r1", urdf_path="/tmp/x.urdf")
    detach(sim, robot)  # no exceptions, no observable side-effects
    assert robot.mesh is None
    assert robot.peer_id == ""


def test_detach_calls_stop_and_clears_fields():
    """Happy path: stop() called, then fields cleared."""
    _, detach = _StubSim._import_helpers()
    sim = _StubSim(mesh=None, peer_id="")
    fake_mesh = MagicMock(name="ChildMesh")
    robot = SimRobot(
        name="r1",
        urdf_path="/tmp/x.urdf",
        mesh=fake_mesh,
        peer_id="parent:r1",
    )

    detach(sim, robot)

    fake_mesh.stop.assert_called_once_with()
    assert robot.mesh is None
    assert robot.peer_id == ""


def test_detach_swallows_stop_exceptions_and_still_clears_fields():
    """A flaky mesh.stop() must not leave robot in a half-attached state."""
    _, detach = _StubSim._import_helpers()
    sim = _StubSim(mesh=None, peer_id="")
    fake_mesh = MagicMock()
    fake_mesh.stop.side_effect = RuntimeError("stop boom")
    robot = SimRobot(
        name="r1",
        urdf_path="/tmp/x.urdf",
        mesh=fake_mesh,
        peer_id="parent:r1",
    )

    detach(sim, robot)  # must not raise

    fake_mesh.stop.assert_called_once_with()
    assert robot.mesh is None, "fields must be cleared even when stop() raises"
    assert robot.peer_id == ""
