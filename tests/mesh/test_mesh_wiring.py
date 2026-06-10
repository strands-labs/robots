"""Tests for PR6 wiring: Robot() factory → init_mesh().

Verifies that the public ``Robot()`` factory constructs a Simulation (or
HardwareRobot in the future) with a working mesh component attached, and
that the ``mesh=False`` and ``STRANDS_MESH=false`` kill switches both
disable mesh creation cleanly.

Heavy dependencies (mujoco, lerobot) are mocked at import time via
``pytest.importorskip`` plus carefully scoped patches.  When mujoco is
not installed the sim tests skip gracefully.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# Skip the entire module if mujoco is not installed — the Robot()/Simulation
# wiring depends on a working sim backend.
pytest.importorskip("mujoco", reason="Robot()/Simulation wiring needs mujoco")


@pytest.fixture
def patched_init_mesh(monkeypatch):
    """Replace strands_robots.mesh.init_mesh with a controllable spy.

    The conftest sets ``STRANDS_MESH=false`` to keep the rest of the suite
    isolated from real zenoh; we delete that override here so the factory
    actually invokes our patched ``init_mesh``.

    Returns ``(mock, fake_mesh_instance)`` so each test can assert call
    arguments and inspect the resulting attached mesh.
    """
    from unittest.mock import MagicMock

    monkeypatch.delenv("STRANDS_MESH", raising=False)
    fake = MagicMock(name="MockMesh")
    fake.peer_id = "fakebot-test"
    fake.alive = True

    with patch("strands_robots.mesh.init_mesh", return_value=fake) as m:
        yield m, fake


def test_robot_factory_attaches_mesh_in_sim_mode(patched_init_mesh):
    """Robot() with default mesh=True attaches the mesh produced by init_mesh."""
    from strands_robots import Robot

    mock_init_mesh, fake_mesh = patched_init_mesh
    sim = Robot("so100", mode="sim")
    try:
        # init_mesh was called with the constructed sim and peer_type='sim'.
        assert mock_init_mesh.called
        kw = mock_init_mesh.call_args.kwargs
        assert kw["peer_type"] == "sim"
        assert kw["mesh"] is True
        assert sim.mesh is fake_mesh
        assert sim.peer_id == "fakebot-test"
    finally:
        # Best-effort cleanup; destroy may fail under heavily mocked envs.
        try:
            sim.destroy()
        except Exception:
            pass


def test_robot_factory_mesh_false_skips_init_mesh(monkeypatch):
    """Robot(..., mesh=False) keeps the sim alive but does not create a mesh."""
    from strands_robots import Robot

    # Make sure mesh would otherwise be enabled — tests run with
    # STRANDS_MESH=false from conftest by default.
    monkeypatch.delenv("STRANDS_MESH", raising=False)
    # But we must keep init_mesh from spinning up a real zenoh session.
    with patch("strands_robots.mesh.init_mesh", return_value=None) as m:
        sim = Robot("so100", mode="sim", mesh=False)
        # mesh=False short-circuits before init_mesh — but the factory may
        # still call init_mesh(... mesh=False) which returns None either way.
        if m.called:
            assert m.call_args.kwargs.get("mesh") is False
    try:
        # init_mesh returns None when mesh=False, so attribute stays None.
        assert sim.mesh is None
    finally:
        try:
            sim.destroy()
        except Exception:
            pass


def test_robot_factory_env_kill_switch_disables_mesh(monkeypatch):
    """STRANDS_MESH=false overrides mesh=True."""
    from strands_robots import Robot

    monkeypatch.setenv("STRANDS_MESH", "false")
    sim = Robot("so100", mode="sim")
    try:
        assert sim.mesh is None
    finally:
        try:
            sim.destroy()
        except Exception:
            pass


def test_robot_factory_passes_peer_id_through(patched_init_mesh):
    """Custom peer_id flows from Robot() into init_mesh()."""
    from strands_robots import Robot

    mock_init_mesh, _ = patched_init_mesh
    sim = Robot("so100", mode="sim", peer_id="my-custom-id")
    try:
        assert mock_init_mesh.call_args.kwargs["peer_id"] == "my-custom-id"
    finally:
        try:
            sim.destroy()
        except Exception:
            pass


def test_robot_factory_mesh_init_failure_does_not_break_sim(monkeypatch):
    """If init_mesh raises, the sim must still be returned."""
    from strands_robots import Robot

    monkeypatch.delenv("STRANDS_MESH", raising=False)

    def boom(*a, **kw):
        raise RuntimeError("zenoh blew up")

    monkeypatch.setattr("strands_robots.mesh.init_mesh", boom)

    # Should NOT raise — the user asked for a sim, mesh is best-effort.
    sim = Robot("so100", mode="sim")
    try:
        # mesh attribute remains the construction-time default (None / falsy).
        assert not sim.mesh
    finally:
        try:
            sim.destroy()
        except Exception:
            pass
