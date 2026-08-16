"""Regression tests: every image key in a MuJoCo observation carries that camera's view.

``_get_sim_observation`` publishes one image per camera key, and those keys come
from two places: the compiled model's own camera names, and the python-side
``world.cameras`` registry. The two do not use the same spelling. ``add_robot``
registers a robot's own MJCF cameras under their SHORT name - the stable,
config-level schema the method documents for joints as well, so a policy sees
``wrist`` whatever the robot instance is called - while the compiled model holds
them namespaced (``arm0/wrist``). The registered ``SimCamera`` carries that
namespaced name; the key does not.

The render loop resolved the model camera from the KEY alone, so every short key
missed, and the miss branch rendered the FREE camera and published it under that
key. The scene overview was therefore served as ``observation["wrist"]``: a
policy reading ``observation.images.wrist``, a recorded LeRobot dataset column,
and the agent-tool observation all received the wrong view under a success
result, with nothing to distinguish it from the camera they asked for. Two short
keys on one robot were byte-identical to each other for the same reason.

The same branch answered for a key that names no compiled camera at all.
``replace_scene_mjcf`` produces that state by documented contract: it swaps the
compiled scene while leaving the python-side registries untouched, so a robot's
camera keys outlive the compiled cameras they name. (These tests originally
manufactured the stranded key through ``remove_robot``, which back then left
behind a duplicate entry that ``add_robot`` had mis-registered against a later
robot. That route closed when ``add_robot`` started registering only the
robot's OWN cameras, each carrying ``origin_robot``, so removal now takes every
one of its keys with it - pinned in ``test_add_robot_camera_ownership.py`` -
and the registry-outlives-the-scene contract of ``replace_scene_mjcf`` is the
route that remains.)

So the contract is one sentence - an observation key that names a camera carries
that camera's view, or is absent - and it is pinned here from both directions:

==================================================  =========================
case                                                pre-fix
==================================================  =========================
short key carries its own robot's view               FAILS (shows overview)
two short keys carry two different views             FAILS (identical)
key with no compiled camera is absent                FAILS (shows overview)
model-named keys unaffected                          passes
joint state survives an unrenderable key             passes
==================================================  =========================

The last two rows are the control: the fix must not disturb the keys that
already resolved, and dropping an image must not cost a caller its
proprioception. They pass before and after, which is what shows the change is
scoped to the keys that were wrong.

The oracle is a renderer driven directly against the compiled model, so the
expectation comes from MuJoCo rather than from the code under test: each model
camera by name, plus the free camera the buggy branch substituted. Comparing
against BOTH is what makes the assertions non-vacuous - asserting only "not the
overview" would pass for any other wrong camera.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.mujoco.simulation import Simulation
from tests.simulation.mujoco._gl_probe import requires_gl

# A robot carrying two cameras with deliberately dissimilar poses: a close
# body-mounted view and a wide side view. They must differ from each other and
# from the free overview for the assertions below to distinguish them.
TWO_CAMERA_ROBOT_XML = """
<mujoco model="two_camera_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="key_light" pos="0 0 3" dir="0 0 -1"/>
    <body name="link" pos="0 0 0.3">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      <geom name="link_geom" type="box" size="0.05 0.05 0.25" rgba="0.9 0.1 0.1 1"/>
      <camera name="wrist" pos="0 -0.2 0.25" xyaxes="1 0 0 0 0 1" fovy="70"/>
    </body>
    <body name="post" pos="0.6 0 0.2">
      <geom name="post_geom" type="capsule" size="0.03" fromto="0 0 0 0 0 0.4" rgba="0.1 0.9 0.1 1"/>
    </body>
    <camera name="side" pos="0 -1.2 0.4" xyaxes="1 0 0 0 0.3 1" fovy="55"/>
  </worldbody>
  <actuator>
    <position name="shoulder_act" joint="shoulder" kp="20"/>
  </actuator>
</mujoco>
"""

# The scene ``replace_scene_mjcf`` swaps in for the stranded-key tests below.
# It keeps the robot's namespaced joint (so proprioception still has something
# to report) and a compiled camera named ``default`` (so the observation still
# carries an image, making "the stranded key is absent" non-vacuous), but none
# of the robot's cameras - while the registry entries for those cameras survive
# the swap by ``replace_scene_mjcf``'s documented contract.
REPLACED_SCENE_XML = """
<mujoco model="replaced_scene">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="key_light" pos="0 0 3" dir="0 0 -1"/>
    <camera name="default" pos="1.5 1.5 1.2" xyaxes="-0.707 0.707 0 -0.4 -0.4 0.82"/>
    <body name="arm0/link" pos="0 0 0.3">
      <joint name="arm0/shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      <geom name="arm0/link_geom" type="box" size="0.05 0.05 0.25" rgba="0.9 0.1 0.1 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="arm0/shoulder_act" joint="arm0/shoulder" kp="20"/>
  </actuator>
</mujoco>
"""


def _write(xml: str) -> str:
    path = os.path.join(tempfile.mkdtemp(), "model.xml")
    with open(path, "w") as handle:
        handle.write(xml)
    return path


@pytest.fixture
def sim() -> Any:
    engine = Simulation()
    assert engine.create_world()["status"] == "success"
    try:
        yield engine
    finally:
        engine.destroy()


def _image_keys(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """The image entries of an observation, keyed as the observation keys them."""
    return {
        key: value
        for key, value in observation.items()
        if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] == 3
    }


def _reference_views(engine: Simulation, height: int, width: int) -> dict[str, np.ndarray]:
    """Ground-truth renders straight from MuJoCo: each model camera, plus the free view.

    The expectation for every assertion below comes from here rather than from
    the observation under test. ``"<free>"`` is the view the miss branch used to
    publish, so naming it lets a failure say which wrong camera was served.
    """
    import mujoco as mj

    world = engine._world
    assert world is not None
    model, data = world._model, world._data
    views: dict[str, np.ndarray] = {}
    with mj.Renderer(model, height=height, width=width) as renderer:
        for cam_id in range(int(model.ncam)):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, cam_id)
            if not name:
                continue
            mj.mj_forward(model, data)
            renderer.update_scene(data, camera=name)
            views[name] = renderer.render().copy()
        mj.mj_forward(model, data)
        renderer.update_scene(data)
        views["<free>"] = renderer.render().copy()
    return views


def _view_shown(image: np.ndarray, views: dict[str, np.ndarray]) -> str:
    """Which reference view ``image`` is, by closest mean absolute pixel difference."""
    return min(views, key=lambda name: float(np.abs(views[name].astype(int) - image.astype(int)).mean()))


def _observe_with_views(engine: Simulation) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    observation = engine.get_observation()
    images = _image_keys(observation)
    assert images, "the observation carried no images at all, so nothing below is exercised"
    height, width = next(iter(images.values())).shape[:2]
    return images, _reference_views(engine, height, width)


@requires_gl
def test_short_key_carries_the_robots_own_camera_view(sim: Simulation) -> None:
    """``observation["wrist"]`` is the robot's wrist camera, not the scene overview.

    The key is the short name; the compiled camera is ``arm0/wrist``. Resolving
    only the key missed and served the free camera under it.
    """
    assert sim.add_robot("arm0", urdf_path=_write(TWO_CAMERA_ROBOT_XML))["status"] == "success"
    images, views = _observe_with_views(sim)
    assert "arm0/wrist" in views, "premise: the robot's camera compiled under a namespaced name"

    assert "wrist" in images, "the short camera key is part of the observation schema"
    assert _view_shown(images["wrist"], views) == "arm0/wrist"


@requires_gl
def test_two_short_keys_carry_two_different_views(sim: Simulation) -> None:
    """Two cameras on one robot contribute two distinct frames.

    Both short keys resolved to the same substituted free camera, so they were
    byte-identical - a two-camera rig recorded the same image twice.
    """
    assert sim.add_robot("arm0", urdf_path=_write(TWO_CAMERA_ROBOT_XML))["status"] == "success"
    images, views = _observe_with_views(sim)

    assert {"wrist", "side"} <= set(images)
    assert not np.array_equal(images["wrist"], images["side"]), (
        "each camera must contribute its own frame; identical frames mean both keys resolved to one substituted camera"
    )
    assert _view_shown(images["wrist"], views) == "arm0/wrist"
    assert _view_shown(images["side"], views) == "arm0/side"


def _stranded_camera_keys(engine: Simulation) -> list[str]:
    """The registry keys the compiled model cannot answer for.

    Mirrors the render loop's two-step resolution: a key is stranded only when
    neither the key itself nor the ``name`` its ``SimCamera`` carries resolves
    to a compiled camera.
    """
    import mujoco as mj

    world = engine._world
    assert world is not None
    model = world._model
    return [
        key
        for key, cam in world.cameras.items()
        if mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, key) < 0
        and mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, cam.name) < 0
    ]


@requires_gl
def test_key_naming_no_compiled_camera_is_absent(sim: Simulation) -> None:
    """A camera key the compiled model cannot answer for is omitted, not filled in.

    The stranded key comes from ``replace_scene_mjcf``, whose documented
    contract is to leave the python-side registries untouched: the robot's
    camera keys survive the swap while the compiled cameras they name do not.
    There is no view to report under such a key, and the overview is not a
    stand-in. The ``default`` key is the control: the replacement scene
    compiles a camera it CAN answer for, so its presence shows the loop
    rendered and only the stranded keys were dropped.
    """
    assert sim.add_robot("arm0", urdf_path=_write(TWO_CAMERA_ROBOT_XML))["status"] == "success"
    assert sim.replace_scene_mjcf(REPLACED_SCENE_XML)["status"] == "success"

    stranded = _stranded_camera_keys(sim)
    assert set(stranded) == {"wrist", "side"}, (
        "premise: the robot's camera keys outlive the compiled cameras they name across replace_scene_mjcf"
    )

    observation = sim.get_observation("arm0")
    images = _image_keys(observation)
    assert "default" in images, "control: a key the compiled model answers for still renders"
    for key in stranded:
        assert key not in images, (
            f"observation[{key!r}] names a camera that is no longer in the model, so it must be "
            "absent rather than carrying some other camera's view"
        )


@requires_gl
def test_model_named_keys_are_unaffected(sim: Simulation) -> None:
    """The keys that already resolved keep resolving to the same cameras.

    Control: the namespaced model names and the built-in ``default`` view were
    never mis-resolved, so the fix must leave them exactly as they were.
    """
    assert sim.add_robot("arm0", urdf_path=_write(TWO_CAMERA_ROBOT_XML))["status"] == "success"
    images, views = _observe_with_views(sim)

    for key in ("arm0/wrist", "arm0/side", "default"):
        assert key in images, f"observation[{key!r}] is part of the schema"
        assert _view_shown(images[key], views) == key


@requires_gl
def test_joint_state_survives_a_key_with_no_compiled_camera(sim: Simulation) -> None:
    """Dropping an unrenderable image does not cost the caller its joint state.

    Control: the loop's whole reason for tolerating a per-camera failure is that
    proprioception must still be reported, so omitting an image must be as
    survivable as the render failure the loop already handled. The replacement
    scene keeps the robot's namespaced joint, so the joint state has a real
    value to report while the robot's camera keys go unanswered.
    """
    assert sim.add_robot("arm0", urdf_path=_write(TWO_CAMERA_ROBOT_XML))["status"] == "success"
    assert sim.replace_scene_mjcf(REPLACED_SCENE_XML)["status"] == "success"
    assert _stranded_camera_keys(sim), "premise: at least one camera key goes unanswered by the compiled model"

    observation = sim.get_observation("arm0")
    assert observation["shoulder"] == pytest.approx(0.0, abs=1e-6)
    assert "shoulder.vel" in observation
