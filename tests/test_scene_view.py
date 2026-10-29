"""A world's view verbs: a named camera re-aims in place, ``light()`` adds one, describe() says how an arm reaches."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from strands_robots import Robot  # noqa: E402
from strands_robots.simulation import Simulation  # noqa: E402
from strands_robots.simulation.backends.isaac import build as isaac_build  # noqa: E402
from strands_robots.simulation.scene import Scene, SceneCamera, SceneLight  # noqa: E402

FRONT, BACK = ((1.0, -1.0, 1.0), (0.0, 0.0, 0.2)), ((-1.0, -1.0, 1.0), (0.0, 0.0, 0.2))


@pytest.fixture
def world():
    sim = Simulation()
    arm = Robot("so101", sim=sim)
    yield sim, arm
    sim.close()


def _cam(sim, pose, **kw):
    return sim.camera("cam", position=pose[0], target=pose[1], **kw)


def test_naming_a_camera_again_re_aims_it_in_place_and_draws_what_a_fresh_one_draws(world, monkeypatch):
    sim, _ = world
    _cam(sim, FRONT, size=(96, 72))
    loads, rev = [], sim.revision
    monkeypatch.setattr(sim.backend, "load", lambda scene: loads.append(scene))
    cam = _cam(sim, BACK)  # no size: the camera keeps 96x72
    moved = sim.render("cam")
    monkeypatch.undo()
    fresh = Simulation()
    Robot("so101", sim=fresh)
    _cam(fresh, BACK, size=(96, 72))
    try:
        assert loads == [] and sim.revision == rev + 1 and cam.size == (96, 72)
        assert [c.name for c in sim.scene.cameras].count("cam") == 1
        assert np.array_equal(moved, fresh.render("cam"))
    finally:
        fresh.close()


def test_a_camera_moved_inside_editing_waits_for_the_block_and_still_lands(world):
    sim, _ = world
    _cam(sim, FRONT, size=(96, 72))
    with sim.editing():
        _cam(sim, BACK)
    fresh = Simulation()
    Robot("so101", sim=fresh)
    _cam(fresh, BACK, size=(96, 72))
    try:
        assert np.array_equal(sim.render("cam"), fresh.render("cam"))
    finally:
        fresh.close()


def test_a_light_brightens_the_frame_recolors_by_name_and_remove_takes_it_out(world):
    sim, _ = world
    _cam(sim, FRONT, size=(96, 72))
    dark = sim.render("cam").astype(float)
    sim.light("lamp", position=(0.0, 0.0, 1.5), color=(0.2, 0.2, 1.0))
    blue = sim.render("cam").astype(float)
    sim.light("lamp", position=(0.0, 0.0, 1.5), color=(1.0, 0.2, 0.2))  # same name: recolored, not a second lamp
    red = sim.render("cam").astype(float)
    assert blue[..., 2].mean() - dark[..., 2].mean() > 5 and red[..., 0].mean() - dark[..., 0].mean() > 5
    assert [li["name"] for li in sim.describe()["lights"]] == ["lamp"]
    sim.remove("lamp")
    back = sim.render("cam").astype(float)
    assert sim.describe()["lights"] == [] and np.abs(back - dark).mean() < 0.5


@pytest.mark.parametrize(
    ("kw", "err"),
    [
        ({"color": (2.0, 0.0, 0.0)}, ValueError),
        ({"color": "blue"}, TypeError),
        ({"direction": (0.0, 0.0, 0.0)}, ValueError),
        ({"position": (0.0, float("nan"), 1.0)}, ValueError),
        ({"position": (0.0, 1.0)}, ValueError),
    ],
)
def test_a_light_refuses_what_it_cannot_draw(world, kw, err):
    sim, _ = world
    with pytest.raises(err):
        sim.light("lamp", **kw)
    assert sim.scene.lights == []


def test_a_light_cannot_take_a_name_the_world_already_uses(world):
    sim, arm = world
    sim.add_object("cube", shape="box")
    for name in (arm.id, "cube"):
        with pytest.raises(ValueError, match="already used"):
            sim.light(name)


def test_describe_says_where_the_arm_reaches_and_which_sign_raises_it(world):
    sim, arm = world
    reach = sim.describe()["robots"][arm.id]["reach"]
    x, y, z = reach["position"]
    assert y < -0.2 and abs(x) < 0.1 and z > 0.1  # an SO-101 at zero reaches along -y
    assert reach["per_rad"]["shoulder_lift"][2] < 0  # +shoulder_lift lowers the tip: negative raises it
    assert set(reach["per_rad"]) <= set(arm.description.action_keys)
    assert arm.describe()["reach"] == reach  # the robot tool's describe says the same


def test_isaac_refuses_a_light_by_name_before_it_touches_the_stage():
    with pytest.raises(NotImplementedError, match="lights"):
        isaac_build.refuse(Scene(lights=[SceneLight("lamp")]), {}, {})


def test_isaac_refuses_an_attached_camera_even_when_its_name_is_on_the_stage():
    # a re-aim onto a link must not pass as the world camera already built under that name
    with pytest.raises(NotImplementedError, match="attach"):
        isaac_build.refuse(Scene(cameras=[SceneCamera("front", attach="so101/gripper")]), {}, {"front": object()})
