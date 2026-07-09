"""Integration tests for ``load_scene`` interacting with downstream mutations.

Regression suite for GH #115: ``load_scene`` previously did not populate
``_backend_state["xml"]`` / ``_backend_state["scene_loaded"]``, so subsequent
``add_object`` / ``add_camera`` / ``remove_object`` calls either:

* recompiled the world via ``MJCFBuilder.build_objects_only``, silently
  discarding every body/mesh from the loaded scene, or
* hit the XML round-trip path which fell through to ``mj_saveLastXML``
  global state and emitted the wrong (robot, not scene) XML.

Each test here loads a scene, performs a mutation, and asserts the original
scene content survives and the mutation is reflected in the compiled model.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from tests.simulation.mujoco._gl_probe import gl_available  # noqa: E402

# Minimal scene: a ground plane + a named block body. This is *not* a robot -
# there are no joints/actuators/sensors. The original bug triggered when
# ``self._world.robots`` was empty, which is the case here.
SCENE_XML = """
<mujoco model="test_scene">
  <option timestep="0.002"/>
  <worldbody>
    <light name="scene_light" pos="0 0 3" dir="0 0 -1"/>
    <geom name="scene_ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="scene_block" pos="1.0 0 0.1">
      <geom name="scene_block_geom" type="box" size="0.1 0.1 0.1" rgba="0.2 0.6 0.9 1"/>
    </body>
    <body name="scene_cylinder" pos="-1.0 0 0.1">
      <geom name="scene_cylinder_geom" type="cylinder" size="0.08 0.1" rgba="0.9 0.6 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def scene_path() -> Generator[str, None, None]:
    """Write the minimal scene XML to a temp file."""
    tmpdir = tempfile.mkdtemp(prefix="test_load_scene_")
    path = os.path.join(tmpdir, "test_scene.xml")
    with open(path, "w") as f:
        f.write(SCENE_XML)
    try:
        yield path
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sim() -> Generator[Simulation, None, None]:
    s = Simulation()
    try:
        yield s
    finally:
        s.cleanup()


def _world(sim: Simulation):
    """Narrow `sim._world` from `SimWorld | None` to `SimWorld` for mypy.

    All tests here construct the world via load_scene / create_world before
    inspecting state, so `sim._world` is definitely non-None at that point.
    Wrap in this helper to keep assertions tidy.
    """
    assert sim._world is not None
    return sim._world


# _backend_state population contract


def test_load_scene_populates_backend_xml(sim: Simulation, scene_path: str) -> None:
    """load_scene must cache the on-disk XML in _backend_state["xml"]."""
    result = sim.load_scene(scene_path)
    assert result["status"] == "success"

    stored = _world(sim)._backend_state.get("xml")
    assert stored is not None, "scene XML must be cached for injection round-trip"
    assert "<mujoco" in stored
    assert "scene_block" in stored


def test_load_scene_marks_scene_loaded(sim: Simulation, scene_path: str) -> None:
    """load_scene must set the scene_loaded flag for downstream mutation gating."""
    sim.load_scene(scene_path)
    assert _world(sim)._backend_state.get("scene_loaded") is True


def test_load_scene_records_scene_base_dir(sim: Simulation, scene_path: str) -> None:
    """load_scene must record the scene's base dir for mesh path resolution."""
    sim.load_scene(scene_path)
    base = _world(sim)._backend_state.get("scene_base_dir")
    assert base is not None
    assert os.path.isdir(base)
    assert os.path.abspath(base) == os.path.dirname(os.path.abspath(scene_path))


# Scene survives downstream add_* mutations


def test_add_object_after_load_scene_preserves_scene_bodies(sim: Simulation, scene_path: str) -> None:
    """add_object after load_scene must inject via XML round-trip, not rebuild.

    The original bug: with no robots registered, add_object fell through to
    _recompile_world() which called MJCFBuilder.build_objects_only - that
    builder only knows about ``world.objects`` and rebuilt from scratch,
    silently deleting every body from the loaded scene.
    """
    sim.load_scene(scene_path)
    mj = sim._mj

    # Establish baseline: the loaded scene has scene_block + scene_cylinder.
    block_id_before = mj.mj_name2id(_world(sim)._model, mj.mjtObj.mjOBJ_BODY, "scene_block")
    cyl_id_before = mj.mj_name2id(_world(sim)._model, mj.mjtObj.mjOBJ_BODY, "scene_cylinder")
    assert block_id_before >= 0, "baseline: scene_block should exist in loaded scene"
    assert cyl_id_before >= 0, "baseline: scene_cylinder should exist in loaded scene"

    # Now add an object. Bug: this used to wipe the scene.
    result = sim.add_object(name="my_new_cube", shape="box", position=[0.0, 1.0, 0.1])
    assert result["status"] == "success", result

    # Loaded scene bodies must still exist.
    block_id_after = mj.mj_name2id(_world(sim)._model, mj.mjtObj.mjOBJ_BODY, "scene_block")
    cyl_id_after = mj.mj_name2id(_world(sim)._model, mj.mjtObj.mjOBJ_BODY, "scene_cylinder")
    assert block_id_after >= 0, "scene_block was wiped by add_object (regression)"
    assert cyl_id_after >= 0, "scene_cylinder was wiped by add_object (regression)"

    # And the newly added object must be in the model too.
    # add_object injects a geom named '{name}_geom' under a body called '{name}'.
    new_body_id = mj.mj_name2id(_world(sim)._model, mj.mjtObj.mjOBJ_BODY, "my_new_cube")
    assert new_body_id >= 0, "newly added object not found in compiled model"


def test_add_camera_after_load_scene_preserves_scene_bodies(sim: Simulation, scene_path: str) -> None:
    """add_camera after load_scene must also use the XML round-trip path.

    Same failure mode as add_object: the ``else`` branch called
    ``_recompile_world()`` which wiped the loaded scene.
    """
    sim.load_scene(scene_path)
    mj = sim._mj

    result = sim.add_camera(name="top_cam", position=[0.0, 0.0, 5.0], target=[0.0, 0.0, 0.0])
    assert result["status"] == "success", result

    # Scene bodies survive
    assert mj.mj_name2id(_world(sim)._model, mj.mjtObj.mjOBJ_BODY, "scene_block") >= 0
    assert mj.mj_name2id(_world(sim)._model, mj.mjtObj.mjOBJ_BODY, "scene_cylinder") >= 0
    # Camera injected
    assert mj.mj_name2id(_world(sim)._model, mj.mjtObj.mjOBJ_CAMERA, "top_cam") >= 0


def test_remove_object_after_load_scene_preserves_other_bodies(sim: Simulation, scene_path: str) -> None:
    """remove_object on a loaded-scene world must use ejection round-trip.

    Previously it called _recompile_world() and wiped everything except
    ``world.objects`` (which is empty post-load_scene).
    """
    sim.load_scene(scene_path)
    # Add, then remove. Both mutations must preserve the loaded scene.
    add_res = sim.add_object(name="temp_obj", shape="box", position=[0.5, 0.5, 0.5])
    assert add_res["status"] == "success"

    rm_res = sim.remove_object(name="temp_obj")
    assert rm_res["status"] == "success", rm_res

    mj = sim._mj
    # Loaded scene bodies survived the round-trip add + remove
    assert mj.mj_name2id(_world(sim)._model, mj.mjtObj.mjOBJ_BODY, "scene_block") >= 0
    assert mj.mj_name2id(_world(sim)._model, mj.mjtObj.mjOBJ_BODY, "scene_cylinder") >= 0
    # temp_obj is gone
    assert mj.mj_name2id(_world(sim)._model, mj.mjtObj.mjOBJ_BODY, "temp_obj") < 0, (
        "remove_object did not actually eject the body from the scene"
    )


def test_create_world_does_not_set_scene_loaded(sim: Simulation) -> None:
    """create_world (the non-load_scene path) must leave scene_loaded unset.

    Regression guard: if create_world accidentally set the flag, add_object
    would mistakenly try to inject into a scene it can freely rebuild, which
    is slower and goes through more code paths.
    """
    result = sim.create_world()
    assert result["status"] == "success"
    assert not _world(sim)._backend_state.get("scene_loaded", False)


# load_scene + add_robot: the original scenario from the BRUTAL_REVIEW.md


ROBOT_XML_FOR_INJECTION = """
<mujoco model="inject_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="arm_base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="arm_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
    </body>
  </worldbody>
  <actuator>
    <position name="arm_pan_act" joint="arm_pan" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def robot_for_injection_path() -> Generator[str, None, None]:
    tmpdir = tempfile.mkdtemp(prefix="test_inject_robot_")
    path = os.path.join(tmpdir, "inject_arm.xml")
    with open(path, "w") as f:
        f.write(ROBOT_XML_FOR_INJECTION)
    try:
        yield path
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def test_add_robot_after_load_scene_preserves_scene_and_robot(
    sim: Simulation, scene_path: str, robot_for_injection_path: str
) -> None:
    """Load a scene, then inject a robot. Scene bodies + robot joints survive.

    This is the exact scenario flagged in the second-opinion review:

        sim.load_scene(...)
        sim.add_robot(...)
        # Expected: scene bodies still there, robot is present
        # Observed before fix: inject_robot_into_scene hits the
        # stored_xml-is-None branch, mj_saveLastXML emits the wrong XML,
        # and the merge breaks.
    """
    # Step 1: load the scene.
    res_scene = sim.load_scene(scene_path)
    assert res_scene["status"] == "success"

    # Step 2: inject the robot.
    res_robot = sim.add_robot(name="my_arm", urdf_path=robot_for_injection_path)
    assert res_robot["status"] == "success", res_robot

    mj = sim._mj
    model = _world(sim)._model

    # Scene bodies survive
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "scene_block") >= 0, (
        "scene_block was lost after add_robot (regression)"
    )
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "scene_cylinder") >= 0, (
        "scene_cylinder was lost after add_robot (regression)"
    )

    # Robot is namespaced under my_arm/
    # inject_robot_into_scene prefixes body/joint/actuator names with 'my_arm/'
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "my_arm/arm_base") >= 0
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "my_arm/arm_pan") >= 0


def test_add_robot_then_add_object_after_load_scene(
    sim: Simulation, scene_path: str, robot_for_injection_path: str
) -> None:
    """Full chain: load_scene → add_robot → add_object → all survive."""
    sim.load_scene(scene_path)
    assert sim.add_robot(name="my_arm", urdf_path=robot_for_injection_path)["status"] == "success"
    assert sim.add_object(name="box_a", shape="box", position=[0.3, 0.3, 0.3])["status"] == "success"
    assert sim.add_object(name="box_b", shape="box", position=[0.5, 0.5, 0.5])["status"] == "success"

    mj = sim._mj
    model = _world(sim)._model
    # All four things from all three sources coexist.
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "scene_block") >= 0
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "my_arm/arm_base") >= 0
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "box_a") >= 0
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "box_b") >= 0


# load_scene must leave MjData forward-evaluated (#168 round 15 bug D)


def test_load_scene_leaves_data_forward_evaluated(sim: Simulation, scene_path: str) -> None:
    """``load_scene`` must call ``mj_forward(model, data)`` before returning
    so ``data.xpos`` / ``data.xmat`` are populated.

    Pin for #168 round-15 bug-D fix: pre-fix, ``MjData(model)``
    zeros these arrays at construction. ``mj_forward`` is what computes
    them from ``qpos`` / kinematic tree. Until ``mj_forward`` runs,
    ``Renderer.update_scene`` finds the body transforms unset and
    returns a skybox-only gradient on every call (the bug-D pattern).

    Round 14 attempted to fix this by calling ``mj_forward`` in
    :meth:`LiberoAdapter.prewarm`, but ``LiberoAdapter.on_episode_start``
    immediately calls ``load_scene`` again, which resets the MjData
    via ``mj.MjData(model)``. The race window between the second
    ``load_scene`` and ``_apply_canonical_state`` (which forwards) is
    where the recorder thread captures gradient frames. The round-15
    fix moves ``mj_forward`` into ``load_scene`` itself - any caller
    of ``load_scene`` (LIBERO adapter or otherwise) gets a sim that's
    safe to render from immediately.
    """
    import numpy as np

    result = sim.load_scene(scene_path)
    assert result["status"] == "success"

    data = _world(sim)._data
    # Body 0 is the world body (always at origin); check a real scene
    # body. ``scene_block`` is at pos="1.0 0 0.1" in the scene XML.
    mj = sim._mj
    block_id = mj.mj_name2id(_world(sim)._model, mj.mjtObj.mjOBJ_BODY, "scene_block")
    assert block_id > 0, "scene_block should exist after load_scene"

    # data.xpos[block_id] should be approximately (1.0, 0, 0.1) AFTER
    # mj_forward has been called. Without the fix, it would be (0, 0, 0).
    expected_pos = np.array([1.0, 0.0, 0.1])
    actual_pos = np.asarray(data.xpos[block_id])
    np.testing.assert_allclose(actual_pos, expected_pos, atol=1e-6)


def test_load_scene_render_returns_real_geometry_immediately(sim: Simulation, scene_path: str) -> None:
    """End-to-end: a render call IMMEDIATELY after ``load_scene`` returns
    real geometry, not the skybox-only gradient.

    Pin for #168 round-15: this is the user-visible symptom of the
    mj_forward fix. Pre-fix, the first render after load_scene
    returned mean RGB (138, 150, 177) col-std 0.62 (skybox gradient).
    Post-fix, the first render returns real geometry (col-std > 5).

    Gated behind ``_requires_mujoco`` because it needs a real GL
    context.
    """
    pytest.importorskip("mujoco")
    if not gl_available():
        pytest.skip("no usable OpenGL context; force-skip with ROBOT_TEST_MUJOCO=0")

    os.environ.setdefault("MUJOCO_GL", "glfw")
    import io

    import numpy as np
    from PIL import Image

    sim.load_scene(scene_path)
    # Add a camera so render() has something to point at.
    sim.add_camera(name="cam", position=[2.0, 2.0, 1.0], target=[0.0, 0.0, 0.1])

    # Render immediately - no step, no further setup. Pre-fix, this
    # would return a gradient.
    r = sim.render(camera_name="cam", width=64, height=48)
    if r["status"] != "success":
        pytest.skip(f"render unavailable in this environment: {r}")

    # Decode the PNG and check column std.
    import base64 as _b64

    image_block = next(c for c in r["content"] if isinstance(c, dict) and "image" in c)
    png_raw = image_block["image"]["source"]["bytes"]
    if isinstance(png_raw, str):
        png_raw = _b64.b64decode(png_raw)
    img = np.asarray(Image.open(io.BytesIO(png_raw)).convert("RGB"))
    col_std = float(img.std(axis=0).mean())
    # Real geometry: col-std typically > 30 for our test scene's
    # mix of plane + colored bodies. Cold-start gradient: col-std ~0.6.
    # Use a conservative threshold of 5 to be robust across GL backends.
    assert col_std > 5.0, (
        f"first render after load_scene appears to be cold-start gradient "
        f"(col_std={col_std:.2f}); expected real geometry (col_std > 5). "
        f"This indicates load_scene didn't call mj_forward, regressing #168 round 15."
    )
