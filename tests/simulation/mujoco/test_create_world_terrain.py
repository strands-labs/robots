"""Integration tests for ``create_world(terrain="rough")`` on the MuJoCo backend.

A flat ground plane measures a locomotion policy's command tracking but never
its robustness to terrain. ``create_world(terrain="rough")`` lays down a
deterministic rough-ground heightfield (:mod:`strands_robots.simulation.terrain`)
instead of the flat plane so a floating-base robot is spawned and evaluated on
non-flat ground - the ground-generation primitive a terrain curriculum builds
on. These tests build a real world and step real physics (GL-free - no
rendering) to verify the heightfield ground is laid down, actually collides
(an object rests higher on a bump than on flat ground), the flat default is
unchanged, an unknown kind is rejected, ``ground_plane=False`` stays the master
floor switch, an attached robot's own ground plane is stripped over terrain,
and the agent-dispatch router accepts the ``terrain`` kwarg.
"""

from __future__ import annotations

import tempfile

import mujoco

from strands_robots.simulation import terrain
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

_PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
_HFIELD = int(mujoco.mjtGeom.mjGEOM_HFIELD)


def _ground_geom_type(sim: MuJoCoSimEngine) -> int:
    assert sim._world is not None
    m = sim._world._model
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    return -1 if gid < 0 else int(m.geom_type[gid])


def _interior_peak_xy() -> tuple[float, float]:
    """World (x, y) of the highest heightfield cell well inside the field rim."""
    n = terrain.TERRAIN_RESOLUTION
    h = terrain.generate_heightfield("rough", resolution=n, seed=terrain.TERRAIN_SEED)
    best_v, best_i, best_j = -1.0, 0, 0
    for i in range(6, n - 6):
        for j in range(6, n - 6):
            v = h[i * n + j]
            if v > best_v:
                best_v, best_i, best_j = v, i, j
    # MuJoCo hfield: row-major, row 0 -> min y, col 0 -> min x, spanning +/-radius.
    y = -terrain.TERRAIN_RADIUS + (best_i / (n - 1)) * 2 * terrain.TERRAIN_RADIUS
    x = -terrain.TERRAIN_RADIUS + (best_j / (n - 1)) * 2 * terrain.TERRAIN_RADIUS
    return x, y


def _box_rest_z(terrain_kind: str | None, x: float, y: float) -> float:
    sim = MuJoCoSimEngine()
    try:
        r = sim.create_world(terrain=terrain_kind)
        assert r["status"] == "success", r
        sim.add_object("blk", shape="box", size=[0.1, 0.1, 0.04], position=[x, y, 0.5], color=[1, 0, 0, 1])
        assert sim._world is not None
        m, d = sim._world._model, sim._world._data
        for _ in range(2500):
            mujoco.mj_step(m, d)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "blk")
        return float(d.xpos[bid][2])
    finally:
        sim.destroy()


def _box_center_rest_z(terrain_kind: str, difficulty: float) -> float:
    """Rest z of a box dropped at the pyramid centre for a given difficulty."""
    sim = MuJoCoSimEngine()
    try:
        r = sim.create_world(terrain=terrain_kind, difficulty=difficulty)
        assert r["status"] == "success", r
        sim.add_object("blk", shape="box", size=[0.1, 0.1, 0.04], position=[0, 0, 0.6], color=[1, 0, 0, 1])
        assert sim._world is not None
        m, d = sim._world._model, sim._world._data
        for _ in range(2500):
            mujoco.mj_step(m, d)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "blk")
        return float(d.xpos[bid][2])
    finally:
        sim.destroy()


def _hfield_peak_elevation(**kw: object) -> float:
    sim = MuJoCoSimEngine()
    try:
        assert sim.create_world(terrain="pyramid", **kw)["status"] == "success"  # type: ignore[arg-type]
        assert sim._world is not None
        return float(sim._world._model.hfield_size[0][2])
    finally:
        sim.destroy()


def test_difficulty_default_matches_full_height_terrain() -> None:
    # difficulty=1.0 (default) is byte-identical to omitting it: same peak
    # elevation AND the same normalized heightfield (difficulty scales only the
    # metre-scale peak, never the [0, 1] field the generator returns).
    import numpy as np

    assert _hfield_peak_elevation() == _hfield_peak_elevation(difficulty=1.0) == terrain.TERRAIN_ELEVATION
    a = MuJoCoSimEngine()
    b = MuJoCoSimEngine()
    try:
        a.create_world(terrain="pyramid", difficulty=1.0)
        b.create_world(terrain="pyramid", difficulty=0.5)
        assert a._world is not None and b._world is not None
        assert np.array_equal(np.array(a._world._model.hfield_data), np.array(b._world._model.hfield_data))
    finally:
        a.destroy()
        b.destroy()


def test_difficulty_scales_peak_elevation_linearly() -> None:
    assert abs(_hfield_peak_elevation(difficulty=0.5) - terrain.TERRAIN_ELEVATION * 0.5) < 1e-9
    assert abs(_hfield_peak_elevation(difficulty=2.0) - terrain.TERRAIN_ELEVATION * 2.0) < 1e-9


def test_difficulty_scales_the_settled_terrain_height() -> None:
    # A box at the pyramid centre settles meaningfully LOWER on a lower-difficulty
    # (shorter) terrain and higher on a taller one - the curriculum physically
    # changes the ground the robot walks on, not just a stored number.
    z_full = _box_center_rest_z("pyramid", 1.0)
    z_half = _box_center_rest_z("pyramid", 0.5)
    assert z_full > z_half + 0.02, (z_full, z_half)


def test_bad_difficulty_is_rejected_without_half_building_a_world() -> None:
    for bad in (-1.0, 0.0, float("inf")):
        sim = MuJoCoSimEngine()
        try:
            r = sim.create_world(terrain="rough", difficulty=bad)
            assert r["status"] == "error"
            assert "difficulty" in r["content"][0]["text"]
            assert sim._world is None or sim._world._model is None
        finally:
            if sim._world is not None:
                sim.destroy()


def test_difficulty_without_terrain_is_rejected_as_a_no_op() -> None:
    # difficulty scales a heightfield; setting it != 1.0 with no terrain would
    # silently do nothing, so it is rejected actionably rather than ignored.
    sim = MuJoCoSimEngine()
    try:
        r = sim.create_world(difficulty=0.5)
        assert r["status"] == "error"
        assert "terrain" in r["content"][0]["text"]
        assert sim._world is None or sim._world._model is None
    finally:
        if sim._world is not None:
            sim.destroy()


def test_difficulty_without_terrain_error_names_every_supported_kind() -> None:
    # The actionable "difficulty needs a terrain" error steers the caller to a
    # real terrain kind; it must list EVERY supported kind (derived from
    # SUPPORTED_TERRAINS) so a kind added later (e.g. "slope") is never silently
    # dropped from the guidance and left undiscoverable in the message.
    sim = MuJoCoSimEngine()
    try:
        msg = sim.create_world(difficulty=0.5)["content"][0]["text"]
        for kind in terrain.SUPPORTED_TERRAINS:
            assert repr(kind) in msg, f"error omits supported terrain kind {kind!r}: {msg}"
    finally:
        if sim._world is not None:
            sim.destroy()


def test_flat_world_default_difficulty_still_succeeds() -> None:
    # The common flat-world path (no terrain, default difficulty) is unaffected.
    sim = MuJoCoSimEngine()
    try:
        assert sim.create_world()["status"] == "success"
        assert _ground_geom_type(sim) == _PLANE
    finally:
        sim.destroy()


def test_router_dispatch_accepts_difficulty_kwarg() -> None:
    sim = MuJoCoSimEngine()
    try:
        r = sim(action="create_world", terrain="pyramid", difficulty=0.5)
        assert r["status"] == "success", r
        assert sim._world is not None
        assert abs(float(sim._world._model.hfield_size[0][2]) - terrain.TERRAIN_ELEVATION * 0.5) < 1e-9
    finally:
        sim.destroy()


def test_terrain_builds_a_heightfield_ground() -> None:
    sim = MuJoCoSimEngine()
    try:
        assert sim.create_world(terrain="rough")["status"] == "success"
        assert sim._world is not None
        m = sim._world._model
        assert _ground_geom_type(sim) == _HFIELD
        assert int(m.nhfield) == 1
        # the compiled heightfield is genuinely non-flat
        assert float(m.hfield_data.max()) - float(m.hfield_data.min()) > 0.5
    finally:
        sim.destroy()


def test_flat_ground_default_is_a_plane() -> None:
    sim = MuJoCoSimEngine()
    try:
        assert sim.create_world()["status"] == "success"
        assert _ground_geom_type(sim) == _PLANE
        assert sim._world is not None
        assert int(sim._world._model.nhfield) == 0
    finally:
        sim.destroy()


def test_terrain_collides_and_lifts_object_above_flat() -> None:
    x, y = _interior_peak_xy()
    z_terrain = _box_rest_z("rough", x, y)
    z_flat = _box_rest_z(None, x, y)
    # The object rests on the heightfield bump, meaningfully higher than on the
    # flat plane at the identical (x, y) -> the terrain both renders AND collides.
    assert z_terrain > z_flat + 0.03, (z_terrain, z_flat)


def test_unknown_terrain_is_rejected() -> None:
    sim = MuJoCoSimEngine()
    try:
        r = sim.create_world(terrain="bogus")
        assert r["status"] == "error"
        assert "rough" in r["content"][0]["text"]
        # no world was left half-built
        assert sim._world is None or sim._world._model is None
    finally:
        if sim._world is not None:
            sim.destroy()


def test_ground_plane_false_is_the_master_switch() -> None:
    sim = MuJoCoSimEngine()
    try:
        assert sim.create_world(terrain="rough", ground_plane=False)["status"] == "success"
        assert _ground_geom_type(sim) == -1  # no ground geom at all
        assert sim._world is not None
        assert int(sim._world._model.nhfield) == 0
    finally:
        sim.destroy()


_PLANEBOT_MJCF = """<mujoco model="planebot">
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="link" pos="0 0 0.4">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="link_g" type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_attached_robot_ground_plane_is_stripped_over_terrain() -> None:
    sim = MuJoCoSimEngine()
    try:
        assert sim.create_world(terrain="rough")["status"] == "success"
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as fh:
            fh.write(_PLANEBOT_MJCF)
            path = fh.name
        assert sim.add_robot("planebot", urdf_path=path)["status"] == "success"
        assert sim._world is not None
        m = sim._world._model
        plane_geoms = [g for g in range(m.ngeom) if int(m.geom_type[g]) == _PLANE]
        hfield_geoms = [g for g in range(m.ngeom) if int(m.geom_type[g]) == _HFIELD]
        # The world owns the (heightfield) ground, so the robot's own z=0 plane
        # is stripped -> exactly one hfield ground, no leftover flat plane
        # coplanar with (and hiding) the terrain.
        assert len(hfield_geoms) == 1, hfield_geoms
        assert len(plane_geoms) == 0, plane_geoms
    finally:
        sim.destroy()


def test_router_dispatch_accepts_terrain_kwarg() -> None:
    sim = MuJoCoSimEngine()
    try:
        r = sim(action="create_world", terrain="rough")
        assert r["status"] == "success", r
        assert _ground_geom_type(sim) == _HFIELD
    finally:
        sim.destroy()


def test_router_dispatch_rejects_unknown_terrain() -> None:
    sim = MuJoCoSimEngine()
    try:
        r = sim(action="create_world", terrain="bogus")
        assert r["status"] == "error"
        assert "rough" in r["content"][0]["text"]
    finally:
        if sim._world is not None:
            sim.destroy()


def test_stairs_builds_a_heightfield_ground() -> None:
    sim = MuJoCoSimEngine()
    try:
        assert sim.create_world(terrain="stairs")["status"] == "success"
        assert sim._world is not None
        m = sim._world._model
        assert _ground_geom_type(sim) == _HFIELD
        assert int(m.nhfield) == 1
        # discrete step plateaus -> the compiled heightfield still spans the
        # full normalized range (flush at z=0, up to the top step).
        assert float(m.hfield_data.max()) - float(m.hfield_data.min()) > 0.5
    finally:
        sim.destroy()


def test_stairs_collides_and_climbs_along_x() -> None:
    # A box dropped on a +x plateau rests meaningfully higher than one on a -x
    # plateau: the staircase both renders AND collides, and rises along +x.
    z_top = _box_rest_z("stairs", 4.0, 0.0)  # near the top step
    z_bot = _box_rest_z("stairs", -4.0, 0.0)  # near the bottom step
    assert z_top > z_bot + 0.04, (z_top, z_bot)
    # ...and both settle on the terrain, not fallen through to a hole.
    assert z_bot > -0.01, z_bot


def test_pyramid_builds_a_heightfield_ground() -> None:
    sim = MuJoCoSimEngine()
    try:
        assert sim.create_world(terrain="pyramid")["status"] == "success"
        assert sim._world is not None
        m = sim._world._model
        assert _ground_geom_type(sim) == _HFIELD
        assert int(m.nhfield) == 1
        # concentric step plateaus -> the compiled heightfield spans the full
        # normalized range (flush at z=0 on the outer ring, up to the top plateau).
        assert float(m.hfield_data.max()) - float(m.hfield_data.min()) > 0.5
    finally:
        sim.destroy()


def test_slope_builds_a_heightfield_ground() -> None:
    sim = MuJoCoSimEngine()
    try:
        assert sim.create_world(terrain="slope")["status"] == "success"
        assert sim._world is not None
        m = sim._world._model
        assert _ground_geom_type(sim) == _HFIELD
        assert int(m.nhfield) == 1
        # a constant-grade ramp -> the compiled heightfield still spans the full
        # normalized range (flush at z=0 at the bottom, up to the top of the ramp).
        assert float(m.hfield_data.max()) - float(m.hfield_data.min()) > 0.5
    finally:
        sim.destroy()


def test_pyramid_climbs_toward_center_and_is_radially_isotropic() -> None:
    # A box near the centre rests meaningfully higher than one out at the ring:
    # the pyramid both renders AND collides, rising toward the centre.
    z_center = _box_rest_z("pyramid", 0.0, 0.0)
    z_px = _box_rest_z("pyramid", 4.0, 0.0)  # out along +x
    z_py = _box_rest_z("pyramid", 0.0, 4.0)  # out along +y (same ring distance)
    assert z_center > z_px + 0.03, (z_center, z_px)
    # The distinguishing property vs the +x-only staircase: the +x and +y rings
    # are at the SAME height (the climb is omnidirectional, depending only on the
    # distance from the centre) - a staircase would raise +x while +y stays flat.
    assert abs(z_px - z_py) < 0.02, (z_px, z_py)
    # ...and the outer ring settles on the flush base, not fallen into a hole.
    assert z_px > -0.01, z_px


def test_slope_collides_and_climbs_along_x() -> None:
    # A box dropped near the top of the ramp (+x) rests meaningfully higher than
    # one near the bottom (-x): the slope both renders AND collides, and rises
    # along +x.
    z_top = _box_rest_z("slope", 4.0, 0.0)  # near the top of the ramp
    z_bot = _box_rest_z("slope", -4.0, 0.0)  # near the bottom of the ramp
    assert z_top > z_bot + 0.04, (z_top, z_bot)
    # ...and both settle on the terrain, not fallen through to a hole.
    assert z_bot > -0.01, z_bot
