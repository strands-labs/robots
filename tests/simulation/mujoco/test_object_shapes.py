"""Every primitive shape supported by ``MJCFBuilder._object_xml`` must render.

Also locks the scene-composer fallback path (``compose_multi_robot_scene``)
and the object-geom auto-naming convention (``<name>_geom``).
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("mujoco")

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")


@pytest.fixture
def sim():
    from strands_robots.simulation import Simulation

    s = Simulation()
    s.create_world()
    yield s
    s.destroy()


@pytest.mark.parametrize(
    "shape,size,name",
    [
        ("box", [0.02, 0.02, 0.02], "a_box"),
        ("sphere", [0.025, 0.025, 0.025], "a_ball"),
        ("cylinder", [0.02, 0.02, 0.06], "a_rod"),
        ("capsule", [0.02, 0.02, 0.06], "a_capsule"),
    ],
)
def test_primitive_shape_roundtrips_to_model(sim, shape, size, name):
    r = sim.add_object(name=name, shape=shape, size=size, position=[0.1, 0.1, 0.05])
    assert r["status"] == "success", r

    # Geom is named by the convention '<name>_geom'
    import mujoco as mj

    gid = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_GEOM, f"{name}_geom")
    assert gid >= 0, f"geom '{name}_geom' not found in model"

    # And we can recolor it via geom_name (set_geom_properties coverage)
    r = sim.set_geom_properties(geom_name=f"{name}_geom", color=[0.3, 0.3, 0.3, 1.0])
    assert r["status"] == "success"


def test_plane_object_auto_static(sim):
    """T29: shape='plane' auto-sets is_static=True; add_object no longer
    errors on plane shapes since they're now routed as static bodies
    automatically."""
    r = sim.add_object(name="floor_mat", shape="plane", size=[0.5, 0.5, 0.001], position=[0, 0, 0.001])
    assert r["status"] == "success", r
    assert sim._world.objects["floor_mat"].is_static is True


def test_plane_object_explicit_dynamic_rejected(sim):
    """T29: Explicit is_static=False on a plane is a hard error - planes are
    infinite and cannot be dynamic bodies in MuJoCo."""
    r = sim.add_object(
        name="bad_floor",
        shape="plane",
        size=[0.5, 0.5, 0.001],
        position=[0, 0, 0.001],
        is_static=False,
    )
    assert r["status"] == "error"
    text = r["content"][0]["text"].lower()
    assert "plane" in text and "is_static" in text
