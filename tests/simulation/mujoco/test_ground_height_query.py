"""Regression tests for the public ``get_ground_height(x, y)`` terrain query.

``create_world(terrain=...)`` raises the local ground up to
``TERRAIN_ELEVATION * difficulty`` above ``z = 0``, and the terrain-relative
locomotion predicates + the spawn/reset base-seating already sample that local
surface internally (``_ground_height_at``). But there was no *public* way for a
caller to ask where the terrain surface is, so an object/camera/goal added at a
flat-ground ``z`` on a raised plateau spawns buried in the heightfield and sinks
through instead of resting on it. ``get_ground_height`` exposes the surface as a
facade query (and agent-dispatch action). These tests build real worlds and read
real geometry (GL-free -- no rendering): flat ground reports ``0.0``, a terrain
world reports the elevated local surface (matching the internal sampler), the
value is finite-validated, and it dispatches through the agent-tool router.
"""

from __future__ import annotations

import math

import numpy as np

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine


def test_get_ground_height_flat_ground_returns_zero() -> None:
    sim = MuJoCoSimEngine()
    try:
        sim.create_world(ground_plane=True)
        r = sim.get_ground_height(0.13, -0.21)
        assert r["status"] == "success"
        payload = r["content"][1]["json"]
        assert payload["x"] == 0.13 and payload["y"] == -0.21
        assert payload["height"] == 0.0
    finally:
        sim.destroy()


def test_get_ground_height_reports_elevated_terrain_surface() -> None:
    """On a raised terrain world the query returns the local surface z (not 0),
    matching the internal sampler that backs the locomotion predicates."""
    sim = MuJoCoSimEngine()
    try:
        sim.create_world(ground_plane=True, terrain="pyramid", difficulty=2.0)
        # Centre plateau of the pyramid is the highest ground.
        r = sim.get_ground_height(0.0, 0.0)
        assert r["status"] == "success"
        height = r["content"][1]["json"]["height"]
        # Elevated well above the flat-ground z=0 (the whole point of the query).
        assert height > 0.1, f"expected an elevated centre plateau, got {height}"
        # Public query must agree with the internal sampler the predicates use.
        assert math.isclose(height, sim._ground_height_at(0.0, 0.0), abs_tol=1e-9)
    finally:
        sim.destroy()


def test_get_ground_height_rejects_non_finite() -> None:
    sim = MuJoCoSimEngine()
    try:
        sim.create_world(ground_plane=True)
        for x, y in ((float("nan"), 0.0), (0.0, float("inf")), (0.0, float("-inf"))):
            r = sim.get_ground_height(x, y)
            assert r["status"] == "error"
        # non-numeric / bool are not valid coordinates
        assert sim.get_ground_height("a", 0.0)["status"] == "error"  # type: ignore[arg-type]
        assert sim.get_ground_height(True, 0.0)["status"] == "error"  # type: ignore[arg-type]
    finally:
        sim.destroy()


def test_get_ground_height_dispatches_via_tool_router() -> None:
    sim = MuJoCoSimEngine()
    try:
        sim.create_world(ground_plane=True, terrain="pyramid", difficulty=2.0)
        ok = sim(action="get_ground_height", x=0.0, y=0.0)
        assert ok["status"] == "success"
        assert ok["content"][1]["json"]["height"] > 0.1
        # both coordinates are required
        missing = sim(action="get_ground_height", x=0.0)
        assert missing["status"] == "error"
        assert "y" in missing["content"][0]["text"]
    finally:
        sim.destroy()


def test_get_ground_height_is_discoverable() -> None:
    """Advertised in the tool_spec enum and the describe() methods surface."""
    sim = MuJoCoSimEngine()
    try:
        enum = sim.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"]
        assert "get_ground_height" in enum
        assert "get_ground_height" in sim.describe()["methods"]
    finally:
        sim.destroy()


def test_get_ground_height_accepts_numpy_scalars() -> None:
    """Terrain coordinates come from ``mj_data`` / an observation (NumPy arrays),
    so a NumPy scalar must be accepted as a finite real number.

    The validation previously used ``isinstance(val, (int, float))``, which is
    False for ``np.float32`` / ``np.int64`` / ``np.int32`` (only ``np.float64``
    subclasses Python ``float``). That spuriously rejected the natural call
    ``get_ground_height(*obs["base_pos"][:2])`` on a float32 observation with a
    misleading "must be a finite number" error, even though the value IS finite.
    """
    sim = MuJoCoSimEngine()
    try:
        sim.create_world(ground_plane=True, terrain="pyramid", difficulty=2.0)
        expected = sim._ground_height_at(0.0, 0.0)
        # A scalar sliced from a float32 array is the realistic case (an
        # observation / mj_data coordinate), alongside the bare NumPy scalars.
        xy32 = np.array([0.0, 0.0], dtype=np.float32)
        for x, y in (
            (np.float32(0.0), np.float32(0.0)),
            (np.float64(0.0), np.float64(0.0)),
            (np.int64(0), np.int64(0)),
            (np.int32(0), np.int32(0)),
            (xy32[0], xy32[1]),
        ):
            r = sim.get_ground_height(x, y)
            assert r["status"] == "success", (type(x).__name__, r)
            assert math.isclose(r["content"][1]["json"]["height"], expected, abs_tol=1e-9)
        # NumPy booleans and non-finite NumPy scalars are still rejected.
        assert sim.get_ground_height(np.bool_(True), 0.0)["status"] == "error"
        assert sim.get_ground_height(np.float32("nan"), 0.0)["status"] == "error"
        assert sim.get_ground_height(np.float64("inf"), 0.0)["status"] == "error"
    finally:
        sim.destroy()


def test_get_ground_height_world_less_reports_flat_ground() -> None:
    """A world-less engine (before ``create_world``) reports flat ground.

    ``get_ground_height`` is deliberately *not* world-scoped like the sibling
    physics queries (``get_total_mass`` and friends return ``_NO_WORLD_MSG``
    before a world exists). A caller sizing a scene may ask where the ground is
    before building the world, and the documented contract is that any
    surface without a heightfield -- including a not-yet-built world -- reads as
    a flat ``0.0`` rather than raising. This pins that public behaviour so a
    later no-world guard cannot silently turn the query into an error.
    """
    sim = MuJoCoSimEngine()
    try:
        r = sim.get_ground_height(1.0, 2.0)
        assert r["status"] == "success"
        payload = r["content"][1]["json"]
        assert payload["x"] == 1.0 and payload["y"] == 2.0
        assert payload["height"] == 0.0
        # The internal sampler that backs the locomotion predicates agrees.
        assert sim._ground_height_at(1.0, 2.0) == 0.0
        # And dispatching through the agent-tool router is equally safe.
        ok = sim(action="get_ground_height", x=1.0, y=2.0)
        assert ok["status"] == "success"
        assert ok["content"][1]["json"]["height"] == 0.0
    finally:
        sim.destroy()


def test_seat_floating_bases_on_terrain_world_less_is_noop() -> None:
    """Terrain base-seating is a safe no-op before a world is built.

    ``_seat_floating_bases_on_terrain`` runs once per spawn / reset cycle. Its
    world-less guard must return without touching physics state (there is no
    model / data to seat onto), so an errant early call cannot raise.
    """
    sim = MuJoCoSimEngine()
    try:
        # Must not raise and must leave the engine world-less.
        sim._seat_floating_bases_on_terrain()
        assert sim._world is None
    finally:
        sim.destroy()
