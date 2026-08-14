"""Regression tests: a floating base survives a namespace-dropping scene replace.

``add_robot(name=...)`` records ``robot.namespace = "<name>/"`` and every joint
lookup in ``simulation/mujoco/rendering.py`` prefixes with it, because a
multi-robot scene injects each robot's joints namespaced (``arm0/shoulder_pan``)
so two same-config robots can coexist. ``replace_scene_mjcf`` then recompiles the
model from caller-supplied MJCF, which need not reproduce that prefix -- and the
registry is *not* rewritten, so ``robot.namespace`` stays ``"<name>/"`` while the
compiled model carries bare joint names. Three byte-identical retries are what
keep the robot observable across that gap::

    if jnt_id < 0 and pfx:
        jnt_id = mj_name_to_id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)

one in ``_robot_base_free_joint`` (the kinematic-tree walk), one in
``_robot_free_base_joint_id`` (the named-free-joint scan) and one in
``_get_sim_observation``'s per-joint read loop. All three bodies were unexecuted
by the suite, so the observable behaviour they protect -- a floating-base robot
still reports its base pose and its joint angles after such a replace -- was
pinned nowhere, and none of the three could be removed by a failing check.

Measured by removing one retry at a time from a `main` tree and re-running these
cases:

===============================  ===================================================
retry removed                    what fails here
===============================  ===================================================
``_robot_base_free_joint``       the mobile-base cases: all four ``base_*``
                                 observation keys, ``get_robot_state``'s ``base``
                                 block and ``_robot_free_base_joint_id`` all go
                                 empty. An unnamed ``<freejoint>`` is reachable
                                 only through the tree walk, so nothing masks it
``_get_sim_observation`` loop    both observation cases: the scalar joint keys
                                 (``hip`` / ``shoulder`` and their ``.vel``
                                 siblings) disappear while the base keys survive
``_robot_free_base_joint_id``    nothing. The named scan falls through to the tree
                                 walk, which resolves the same joint by its own
                                 retry, so this one is executed here but not
                                 separable from its own fallback
===============================  ===================================================

Two of the three are therefore individually decisive and the third is covered but
masked. Deduplicating the family into one shared helper is what would make the
third enforceable, and is deliberately not attempted here: the three call sites
do different things with the id (a free-joint type check, a tree walk, an
observation write), so that is a refactor with its own review. See #2262.

Which surface reaches which retry is not interchangeable, so the cases are split
along that line: ``get_observation`` runs its own inlined loop and delegates only
the UNNAMED base to the tree walk, and never calls
``_robot_free_base_joint_id`` at all -- that one is reached by
``start_recording``'s ``base_*`` schema columns and by terrain seating.

The negative case is what keeps the rest non-vacuous: with the joints under a
*third* namespace neither the prefixed nor the bare lookup resolves, and the base
must genuinely disappear from all three surfaces. Without it, a fallback that
resolved anything at all -- or an observation that emitted base keys
unconditionally -- would satisfy every positive assertion above.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Humanoid-style floating base: a NAMED free root joint, enumerated in
# ``robot.joint_names``, plus one actuated hinge. Exercises the named-free-joint
# scan in ``_robot_free_base_joint_id``.
NAMED_BASE_XML = """
<mujoco model="ns_named_base">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="torso" pos="0 0 0.5">
      <joint name="floating_base_joint" type="free"/>
      <geom type="box" size="0.1 0.1 0.2"/>
      <body name="thigh" pos="0 0 -0.2">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.2"/>
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_act" joint="hip"/>
  </actuator>
</mujoco>
"""

# Mobile base (LeKiwi-style): an UNNAMED ``<freejoint>``, absent from
# ``robot.joint_names``, so the base is reachable ONLY by walking up from an
# actuated joint -- which is what makes the tree walk's retry decisive here.
MOBILE_BASE_XML = """
<mujoco model="ns_mobile_base">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="base_plate" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.15 0.15 0.03"/>
      <body name="arm" pos="0 0 0.05">
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2"/>
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_act" joint="shoulder"/>
  </actuator>
</mujoco>
"""

# The two robot shapes, each paired with the joint names its replacement scene
# must declare. An empty ``base_joint`` spells the unnamed-``<freejoint/>`` shape.
SHAPES = [
    pytest.param(NAMED_BASE_XML, "floating_base_joint", "hip", id="named-floating-base"),
    pytest.param(MOBILE_BASE_XML, "", "shoulder", id="unnamed-mobile-base"),
]

_BASE_KEYS = ("base_pos", "base_quat", "base_lin_vel", "base_ang_vel")

# A known base pose + twist written straight to the free joint's qpos/qvel after
# the replace. Distinct in every component so a wrong joint id, or a base read
# off the wrong address, cannot coincide with it.
KNOWN_POS = [0.11, -0.22, 0.83]
KNOWN_QUAT = [0.7071068, 0.0, 0.7071068, 0.0]  # +90 deg about Y
KNOWN_LIN_VEL = [0.41, -0.52, 0.63]
KNOWN_ANG_VEL = [0.13, 0.24, 0.37]


def _replacement(model_name: str, base_joint: str, hinge: str) -> str:
    """MJCF for a ``replace_scene_mjcf`` that declares the joints UNPREFIXED.

    Agent-authored MJCF is written against the joint names the caller knows --
    the config-level ones -- and has no reason to reproduce the ``"<name>/"``
    prefix ``add_robot`` injected.
    """
    freejoint = f'<joint name="{base_joint}" type="free"/>' if base_joint else "<freejoint/>"
    return f"""
<mujoco model="{model_name}">
  <option timestep="0.002"/>
  <worldbody>
    <body name="torso" pos="0 0 0.5">
      {freejoint}
      <geom type="box" size="0.1 0.1 0.2"/>
      <body name="thigh" pos="0 0 -0.2">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.2"/>
        <joint name="{hinge}" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="act" joint="{hinge}"/>
  </actuator>
</mujoco>
"""


def _write(xml: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(xml)
    return path


@pytest.fixture
def sim():
    s = Simulation(tool_name="ns_fallback", mesh=False)
    s.create_world(ground_plane=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


def _world(sim: Simulation) -> Any:
    """The live world, asserted present -- the ``sim`` fixture created one.

    Narrows ``Simulation._world`` (``SimWorld | None``) for the reads below, so a
    fixture that stopped creating a world is reported here by name rather than as
    an attribute error raised from inside an assertion.
    """
    world = sim._world
    assert world is not None, "the sim fixture must have created a world"
    return world


def _spawn(sim: Simulation, name: str, xml: str) -> Any:
    res = sim.add_robot(name, urdf_path=_write(xml))
    assert res["status"] == "success", res
    robot = _world(sim).robots[name]
    assert robot.namespace == f"{name}/", f"the namespace this suite is about was not recorded: {robot.namespace!r}"
    return robot


def _replace_dropping_the_namespace(sim: Simulation, robot: Any, base_joint: str, hinge: str) -> None:
    """Recompile the scene with bare joint names, then assert that it really is bare.

    The precondition is asserted rather than assumed: if ``replace_scene_mjcf``
    ever re-namespaced the joints it compiles, the prefixed lookup would succeed,
    no retry would run, and every assertion in this module would still pass --
    pinning nothing. This is what fails loudly in that case instead.
    """
    import mujoco as mj

    res = sim.replace_scene_mjcf(_replacement("replaced", base_joint, hinge))
    assert res["status"] == "success", res
    model = _world(sim)._model
    prefixed = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, robot.namespace + hinge)
    bare = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, hinge)
    assert prefixed < 0, f"'{robot.namespace + hinge}' still resolves: the namespace was not dropped"
    assert bare >= 0, f"'{hinge}' does not resolve either: the replacement scene declares a different joint"


def _model_free_joint_id(sim: Simulation) -> int:
    """The scene's single free joint, located without asking the code under test.

    Scanning ``jnt_type`` keeps the expected id independent of the fallback whose
    result is being checked against it, so a case cannot agree with itself.
    """
    import mujoco as mj

    model = _world(sim)._model
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE]
    assert len(free) == 1, f"the replacement scene must carry exactly one free joint, found {len(free)}"
    return int(free[0])


def _plant_known_base_state(sim: Simulation) -> None:
    """Write the KNOWN_* pose + twist to that free joint and settle the model."""
    import mujoco as mj

    model, data = _world(sim)._model, _world(sim)._data
    jid = _model_free_joint_id(sim)
    qadr, vadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    data.qpos[qadr : qadr + 3] = KNOWN_POS
    data.qpos[qadr + 3 : qadr + 7] = KNOWN_QUAT
    data.qvel[vadr : vadr + 3] = KNOWN_LIN_VEL
    data.qvel[vadr + 3 : vadr + 6] = KNOWN_ANG_VEL
    mj.mj_forward(model, data)


@pytest.mark.parametrize(("spawn_xml", "base_joint", "hinge"), SHAPES)
def test_observation_survives_a_namespace_dropping_replace(
    sim: Simulation, spawn_xml: str, base_joint: str, hinge: str
) -> None:
    """The full observation -- base pose, base twist and joint scalars -- survives.

    A locomotion or whole-body-control policy consumes ``base_pos`` /
    ``base_quat`` / ``base_lin_vel`` / ``base_ang_vel`` and the per-joint scalars
    from the same dict. Losing either half after an agent replaced the scene is
    silent: ``get_observation`` reports ``success`` with keys simply absent, and a
    policy reading them through ``.get`` sees zeros rather than an error. Both
    halves are pinned in one case because they fail independently -- the base keys
    come from the free-joint retries, the scalars from the read loop's own.
    """
    robot = _spawn(sim, "hum", spawn_xml)
    _replace_dropping_the_namespace(sim, robot, base_joint, hinge)
    _plant_known_base_state(sim)

    obs = sim.get_observation("hum", skip_images=True)

    for key in _BASE_KEYS:
        assert key in obs, f"{key} lost after a namespace-dropping replace; got {sorted(obs)}"
    assert obs["base_pos"] == pytest.approx(KNOWN_POS, abs=1e-6)
    assert obs["base_quat"] == pytest.approx(KNOWN_QUAT, abs=1e-6)
    assert obs["base_lin_vel"] == pytest.approx(KNOWN_LIN_VEL, abs=1e-6)
    assert obs["base_ang_vel"] == pytest.approx(KNOWN_ANG_VEL, abs=1e-6)
    assert hinge in obs, f"the scalar joint '{hinge}' is not observed; got {sorted(obs)}"
    assert f"{hinge}.vel" in obs, f"the joint velocity '{hinge}.vel' is not observed; got {sorted(obs)}"
    # A free joint has no scalar angle (its qpos is a pose), so the retry must
    # not have smuggled one in alongside the structured base keys.
    assert base_joint not in obs


@pytest.mark.parametrize(("spawn_xml", "base_joint", "hinge"), SHAPES)
def test_free_base_joint_id_resolves_after_a_namespace_dropping_replace(
    sim: Simulation, spawn_xml: str, base_joint: str, hinge: str
) -> None:
    """``_robot_free_base_joint_id`` still names the base joint, for both shapes.

    This is the finder ``start_recording`` asks whether to give the dataset its
    ``base_pos`` / ``base_quat`` / ``base_lin_vel`` / ``base_ang_vel`` columns,
    and that terrain seating asks where the base is. It is not reached through
    ``get_observation``, which inlines its own loop -- so a regression here would
    be invisible to the cases above while quietly producing a base-blind dataset
    from a floating-base robot.

    Driven directly rather than through either consumer: the recording path needs
    lerobot installed, and the seating path needs a height field the replacement
    scene does not carry, so neither can observe this without a dependency that
    would turn the pin into a skip.
    """
    robot = _spawn(sim, "hum", spawn_xml)
    _replace_dropping_the_namespace(sim, robot, base_joint, hinge)

    jid = sim._robot_free_base_joint_id(_world(sim)._model, robot)

    assert jid == _model_free_joint_id(sim), (
        f"the base joint resolved to {jid}, not the free joint the compiled model carries"
    )


def test_robot_state_base_block_survives_for_an_unnamed_mobile_base(sim: Simulation) -> None:
    """``get_robot_state`` reports its ``base`` block through the same tree walk.

    A second consumer of the tree walk, and the reason its retry is worth pinning
    twice: ``get_robot_state`` runs its own joint loop but delegates an unnamed
    free base to ``_robot_base_free_joint``, exactly as ``_get_sim_observation``
    does. Pinning both means a regression is reported at whichever surface a
    caller happens to use.
    """
    robot = _spawn(sim, "mob", MOBILE_BASE_XML)
    _replace_dropping_the_namespace(sim, robot, "", "shoulder")
    _plant_known_base_state(sim)

    res = sim.get_robot_state("mob")
    assert res["status"] == "success", res
    payload = next(block["json"] for block in res["content"] if "json" in block)

    assert "base" in payload, f"the mobile base's 'base' block is missing: {payload}"
    assert payload["base"]["position"] == pytest.approx(KNOWN_POS, abs=1e-6)
    assert payload["base"]["quaternion"] == pytest.approx(KNOWN_QUAT, abs=1e-6)


def test_base_state_disappears_when_neither_name_resolves(sim: Simulation) -> None:
    """The pin is not vacuous: an unresolvable robot reports no base on any surface.

    The joints land under a *third* namespace, so neither ``"hum/hip"`` nor
    ``"hip"`` resolves and no retry can help. This is the case that gives the
    positive ones their meaning -- it fails if a fallback ever resolves a joint
    that is not the robot's (a sibling free-jointed task object, say), or if the
    base keys are emitted unconditionally.
    """
    import mujoco as mj

    robot = _spawn(sim, "hum", NAMED_BASE_XML)
    foreign = _replacement("foreign", "other/floating_base_joint", "other/hip")
    assert sim.replace_scene_mjcf(foreign)["status"] == "success"
    model = _world(sim)._model
    for lookup in (robot.namespace + "hip", "hip"):
        assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, lookup) < 0, f"'{lookup}' must not resolve in this scene"

    obs = sim.get_observation("hum", skip_images=True)

    assert obs == {}, f"an unresolvable robot must observe nothing, got {sorted(obs)}"
    assert sim._robot_free_base_joint_id(model, robot) == -1, "a base joint was resolved for an unresolvable robot"
    payload = next(block["json"] for block in sim.get_robot_state("hum")["content"] if "json" in block)
    assert "base" not in payload, f"a base block was reported for an unresolvable robot: {payload}"
