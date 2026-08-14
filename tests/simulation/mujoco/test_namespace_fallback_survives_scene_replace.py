"""Regression tests: the namespace-fallback retry keeps a floating base observable
after a scene replace whose MJCF drops the namespace.

``add_robot`` records ``robot.namespace = "<name>/"`` and every joint lookup in
:mod:`strands_robots.simulation.mujoco.rendering` prefixes with it.
``replace_scene_mjcf`` recompiles from caller-supplied MJCF, which need not
reproduce that prefix, and the registry is *not* rewritten - ``robot.namespace``
stays ``"<name>/"`` across the replace. So after such a replace the prefixed
lookup misses and only a retry on the **bare** joint name resolves the joint.

That retry is spelled three times, once per joint-resolving helper:

=================================  =========================================
helper                             reached by
=================================  =========================================
``_get_sim_observation`` loop      ``get_observation``
``_robot_base_free_joint``         ``get_observation`` / ``get_robot_state``
                                   for an UNNAMED ``<freejoint>``
``_robot_free_base_joint_id``      terrain seating + recording schema
=================================  =========================================

The three mask one another through the fall-through chain, which is why none of
them was driven before this module: a helper whose retry is gone still reaches
the base through the next helper's. Deleting one retry at a time and running
this file measures how far that masking survives the pin:

======================================  =============================
retry deleted                           this module
======================================  =============================
``_robot_base_free_joint``              3 of 8 tests fail
``_get_sim_observation`` loop           2 of 8 tests fail
``_robot_free_base_joint_id``           8 pass - still masked
all three                               5 of 8 tests fail
======================================  =============================

So two of the three are now individually observable. The third is masked
structurally rather than incidentally: ``_robot_free_base_joint_id`` ends by
delegating to ``_robot_base_free_joint``, which reaches the same free joint
through its own retry, so no assertion on the finder's return value can separate
them. Its test below therefore pins the finder's contract and covers the line;
the all-three row is what shows the line is load-bearing (without the family the
finder returns ``-1`` and ``get_observation`` surfaces no ``base_*`` key at all).

The last test is what keeps the rest non-vacuous: the fallback resolves by bare
NAME, so when the replacement scene's joints are renamed there is nothing to find
and the base state must be absent rather than borrowed from an unrelated free
joint.

Deleting a retry is not the only way this family breaks. The retry is a
*fallback*, so the other direction is a lookup that stops preferring the
namespaced name - a bare-name lookup, or a retry hoisted to run unconditionally
over a successful prefixed hit. Both spellings can be compiled at once, and then
the two answers differ, so the last two tests pin the precedence against a decoy
body owning the bare names:

==========================================  =============================
lookup rewritten                            this module
==========================================  =============================
bare name used unconditionally              2 of 8 tests fail
retry runs unconditionally                  2 of 8 tests fail
==========================================  =============================

Neither mutation is observable to any of the six tests above, because a scene
that carries only one spelling cannot tell the two readings apart. This matters
most for the refactor #2262 defers: collapsing the three copies into one shared
helper is precisely the change that could reorder the lookup, and the pin is what
would catch it.

The base signals read back are PLANTED (:func:`_plant_known_state`), not the
compiled rest state. That rest state - identity quaternion, zero twist, zero
hinge angle - is also what a read from the wrong joint, the wrong ``qpos`` offset
or the wrong ``qvel`` offset returns, so asserting it cannot separate a correct
address from several incorrect ones. ``base_lin_vel`` / ``base_ang_vel`` are the
sharpest case: they are addressed through ``jnt_dofadr`` into ``qvel``, computed
independently of ``qpos``'s ``jnt_qposadr``, so swapping the two blocks is
invisible at rest and a locomotion policy reading ``base_lin_vel`` is exactly the
consumer that would notice.

Planting makes the address itself observable, which deleting a retry never was:

==========================================  =============================
base read mis-addressed                     this module
==========================================  =============================
``base_lin_vel``/``base_ang_vel`` swapped   3 of 8 tests fail
``base_quat`` off the ``qpos`` offset       2 of 8 tests fail
==========================================  =============================

The swap is the one to note: both ``qvel`` blocks are zero at rest, so against
the rest-state assertions it was invisible in all 8. Reproducing that row means
swapping the pair on BOTH surfaces these cases read - ``base_lin_vel`` /
``base_ang_vel`` in :meth:`_get_sim_observation`, and ``linear_velocity`` /
``angular_velocity`` in ``get_robot_state``, which addresses ``qvel`` itself
rather than delegating. Swapping only the first fails 2 of 8, because the
``get_robot_state`` case reads the unmutated copy - a partial reproduction that
looks like the table has drifted.

Which spellings a replacement carries is the premise every case rests on, so it
is asserted on the recompiled model itself (:func:`_assert_replacement_premise`)
rather than on ``robot.namespace`` alone. Checking the registry establishes only
what the registry still believes: were ``replace_scene_mjcf`` ever to namespace
what it compiles, the prefixed lookup would resolve, no retry body would execute,
and the six namespace-dropping cases would all still pass - going green while
pinning nothing, and quietly ceasing to reproduce the tables above. The two
precedence cases opt into the opposite premise (``prefixed_resolves=True``),
since a decoy scene must carry both spellings for the answers to differ.

``_robot_free_base_joint_id`` is asserted through a direct call because neither
of its callers is reachable here. Terrain seating needs a heightfield, and the
replacement MJCF cannot carry one (a scene exported from a seated world fails to
recompile: ``XML Error: repeated default class name``); the recording schema
needs the ``lerobot`` extra, so a test routed through it would skip rather than
run wherever that extra is absent.
"""

import os
import tempfile
from typing import Any

import pytest

from strands_robots.simulation.mujoco.simulation import Simulation

# Humanoid-style floating base: the free joint is NAMED and therefore enumerated
# in ``robot.joint_names``, so the observation read loop resolves it directly.
NAMED_BASE_XML = """
<mujoco model="test_named_base">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="torso" pos="0 0 0.6">
      <freejoint name="floating_base_joint"/>
      <geom type="box" size="0.1 0.05 0.2" rgba="0.3 0.3 0.8 1"/>
      <body name="thigh" pos="0 0 -0.2">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.3" rgba="0.8 0.3 0.3 1"/>
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_act" joint="hip"/>
  </actuator>
</mujoco>
"""

# Mobile base (LeKiwi-style): the free joint is UNNAMED and absent from
# ``robot.joint_names``, so the base is recovered by walking up the kinematic
# tree from the actuated joint - which must itself be resolved by bare name.
UNNAMED_BASE_XML = """
<mujoco model="test_unnamed_base">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="base_plate" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.15 0.15 0.03" rgba="0.3 0.3 0.8 1"/>
      <body name="arm" pos="0 0 0.05">
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_act" joint="shoulder"/>
  </actuator>
</mujoco>
"""

# The namespace-dropping replacement scenes: the same joint NAMES as the robot
# the registry recorded, with no ``<name>/`` prefix anywhere.
NAMED_BASE_UNPREFIXED = NAMED_BASE_XML.replace('model="test_named_base"', 'model="replacement_named"')
UNNAMED_BASE_UNPREFIXED = UNNAMED_BASE_XML.replace('model="test_unnamed_base"', 'model="replacement_unnamed"')

# A replacement whose joints share no name with the registered robot, prefixed
# or bare. There IS a free joint in the scene, so a helper that picked any free
# joint rather than this robot's own would still answer - the base state must
# disappear instead.
RENAMED_JOINTS_XML = """
<mujoco model="replacement_renamed">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="other_plate" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.15 0.15 0.03" rgba="0.3 0.3 0.8 1"/>
      <body name="other_arm" pos="0 0 0.05">
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="elbow" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="elbow_act" joint="elbow"/>
  </actuator>
</mujoco>
"""

BASE_KEYS = {"base_pos", "base_quat", "base_lin_vel", "base_ang_vel"}

# Distinct in every component and different from the compiled rest state, so a
# wrong joint, a wrong ``qpos`` offset and a wrong ``qvel`` offset each fail with
# a value that names which. The quaternion is unit-norm (+90 deg about Y) so
# ``mj_forward`` cannot alter it by renormalising.
KNOWN_BASE_POS = [0.11, -0.22, 0.83]
KNOWN_BASE_QUAT = [0.7071068, 0.0, 0.7071068, 0.0]
KNOWN_BASE_LIN_VEL = [0.41, -0.52, 0.63]
KNOWN_BASE_ANG_VEL = [0.13, 0.24, 0.37]
KNOWN_HINGE_POS = 0.37
KNOWN_HINGE_VEL = -0.29

# A replacement carrying BOTH spellings of the robot's hinge: its own
# ``hum/hip`` and an unrelated ``hip`` on a decoy body. The retry is a FALLBACK,
# so the prefixed name must still win - a lookup rewritten to use the bare name,
# or a retry hoisted to run unconditionally, would report the decoy's angle as
# this robot's and pass every test above. That is the multi-robot failure the
# namespace exists to prevent, and the refactor #2262 defers (collapsing the
# three copies into one helper) is exactly the change that could reorder it.
BOTH_SPELLINGS_XML = """
<mujoco model="replacement_both_spellings">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="torso" pos="0 0 0.6">
      <freejoint name="hum/floating_base_joint"/>
      <geom type="box" size="0.1 0.05 0.2" rgba="0.3 0.3 0.8 1"/>
      <body name="thigh" pos="0 0 -0.2">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.3" rgba="0.8 0.3 0.3 1"/>
        <joint name="hum/hip" type="hinge" axis="0 1 0" range="-3 3"/>
      </body>
    </body>
    <body name="decoy" pos="1 0 0.3">
      <freejoint name="floating_base_joint"/>
      <geom type="box" size="0.05 0.05 0.05" rgba="0.5 0.5 0.5 1"/>
      <body name="decoy_link" pos="0 0 0.05">
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.1" rgba="0.5 0.5 0.5 1"/>
        <joint name="hip" type="hinge" axis="0 1 0" range="-3 3"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_namespace_fallback", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    s.cleanup()


def _write(xml: str) -> str:
    path = os.path.join(tempfile.mkdtemp(), "model.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


def _single_joint_of_type(sim: Simulation, model: Any, jnt_type: Any, label: str) -> int:
    """Locate the scene's only joint of ``jnt_type`` by scanning ``jnt_type``.

    Deliberately not obtained from the helper under test: deriving the address
    from the code whose result is compared against it would make the comparison
    circular.
    """
    ids = [i for i in range(model.njnt) if model.jnt_type[i] == jnt_type]
    assert len(ids) == 1, f"expected exactly one {label} joint in the replacement scene, found ids {ids}"
    return int(ids[0])


def _plant_known_state(sim: Simulation) -> None:
    """Write the known base pose/twist and hinge state into the replaced scene."""
    mj = sim._mj
    world = sim._world
    assert world is not None
    model, data = world._model, world._data

    free_id = _single_joint_of_type(sim, model, mj.mjtJoint.mjJNT_FREE, "free")
    qadr = int(model.jnt_qposadr[free_id])
    vadr = int(model.jnt_dofadr[free_id])
    data.qpos[qadr : qadr + 3] = KNOWN_BASE_POS
    data.qpos[qadr + 3 : qadr + 7] = KNOWN_BASE_QUAT
    data.qvel[vadr : vadr + 3] = KNOWN_BASE_LIN_VEL
    data.qvel[vadr + 3 : vadr + 6] = KNOWN_BASE_ANG_VEL

    hinge_id = _single_joint_of_type(sim, model, mj.mjtJoint.mjJNT_HINGE, "hinge")
    data.qpos[int(model.jnt_qposadr[hinge_id])] = KNOWN_HINGE_POS
    data.qvel[int(model.jnt_dofadr[hinge_id])] = KNOWN_HINGE_VEL

    # Recompute derived quantities without integrating, so the reads below see
    # exactly what was planted.
    mj.mj_forward(model, data)


def _assert_planted_base(
    pos: list[float], quat: list[float], lin_vel: list[float], ang_vel: list[float], *, where: str
) -> None:
    """Assert the four base signals are the planted ones, component by component."""
    assert pos == pytest.approx(KNOWN_BASE_POS, abs=1e-6), f"{where}: base position is not the planted pose"
    assert quat == pytest.approx(KNOWN_BASE_QUAT, abs=1e-6), f"{where}: base orientation is not the planted quat"
    assert lin_vel == pytest.approx(KNOWN_BASE_LIN_VEL, abs=1e-6), (
        f"{where}: base linear velocity is not the planted twist - a wrong qvel dof address reads zeros here"
    )
    assert ang_vel == pytest.approx(KNOWN_BASE_ANG_VEL, abs=1e-6), (
        f"{where}: base angular velocity is not the planted twist - a wrong qvel dof address reads zeros here"
    )


def _joint_id(sim: Simulation, name: str) -> int:
    world = sim._world
    assert world is not None
    return sim._mj.mj_name2id(world._model, sim._mj.mjtObj.mjOBJ_JOINT, name)


def _assert_replacement_premise(sim: Simulation, name: str, *, prefixed_resolves: bool, bare_resolves: bool) -> None:
    """Assert on the COMPILED MODEL which spellings of ``name``'s joints exist.

    ``robot.namespace`` only records what the registry believes. Without this the
    namespace-dropping cases would keep passing against a ``replace_scene_mjcf``
    that started namespacing its output, with no retry body executed at all. See
    the module docstring.
    """
    world = sim._world
    assert world is not None
    joint_names = world.robots[name].joint_names
    assert joint_names, f"{name!r} registered no joints, so no lookup is exercised"

    prefixed = [jnt for jnt in joint_names if _joint_id(sim, f"{name}/{jnt}") >= 0]
    bare = [jnt for jnt in joint_names if _joint_id(sim, jnt) >= 0]

    if prefixed_resolves:
        assert prefixed, (
            f"the replacement carries no '{name}/' joint, so the prefixed lookup cannot win and "
            "this case is not exercising precedence"
        )
    else:
        assert not prefixed, (
            f"the replacement still carries {[f'{name}/{jnt}' for jnt in prefixed]}: the prefixed "
            "lookup resolves, so no namespace-fallback retry is driven and this case would pass "
            "while pinning nothing"
        )

    if bare_resolves:
        assert bare, (
            f"no registered joint of {name!r} resolves by bare name either ({joint_names}): the "
            "retry has nothing to find, so the assertions below would be vacuous"
        )
    else:
        assert not bare, f"the replacement still resolves {bare} by bare name, so this case is not exercising a miss"


def _add_and_replace(
    sim: Simulation,
    name: str,
    robot_xml: str,
    replacement: str,
    *,
    prefixed_resolves: bool = False,
    bare_resolves: bool = True,
) -> None:
    """Register ``name`` (which namespaces its joints), then replace the scene."""
    sim.add_robot(name, urdf_path=_write(robot_xml))
    world = sim._world
    assert world is not None
    assert world.robots[name].namespace == f"{name}/", "add_robot must namespace the robot's joints"
    result = sim.replace_scene_mjcf(replacement)
    assert result["status"] == "success", result
    # The premise of the whole family: the registry is not rewritten, so every
    # lookup still prefixes with a namespace the compiled model no longer has.
    assert world.robots[name].namespace == f"{name}/", "replace_scene_mjcf must not rewrite the namespace"
    # ... and the other half of it, on the model that was actually compiled.
    _assert_replacement_premise(sim, name, prefixed_resolves=prefixed_resolves, bare_resolves=bare_resolves)


def test_named_floating_base_observation_survives_namespace_dropping_replace(sim):
    """A humanoid's NAMED free joint and its actuated hinge both resolve by bare
    name after the replace, so the observation keeps its base state AND its
    per-joint scalars."""
    _add_and_replace(sim, "hum", NAMED_BASE_XML, NAMED_BASE_UNPREFIXED)
    _plant_known_state(sim)

    obs = sim.get_observation(robot_name="hum", skip_images=True)

    assert BASE_KEYS <= set(obs), f"floating-base state lost after replace: {sorted(obs)}"
    _assert_planted_base(
        obs["base_pos"], obs["base_quat"], obs["base_lin_vel"], obs["base_ang_vel"], where="get_observation"
    )
    # The scalar joints go through the same retry, so they are part of the pin:
    # without it the observation would be base state and nothing else.
    assert obs["hip"] == pytest.approx(KNOWN_HINGE_POS, abs=1e-6)
    assert obs["hip.vel"] == pytest.approx(KNOWN_HINGE_VEL, abs=1e-6)
    # A free joint is never reported as a scalar (its qpos is [xyz + quat]).
    assert "floating_base_joint" not in obs


def test_unnamed_mobile_base_observation_survives_namespace_dropping_replace(sim):
    """A mobile base's UNNAMED free joint is reached by walking up from the
    actuated joint, and that joint is only findable by bare name after the
    replace - so the tree-walk fallback depends on the retry too."""
    _add_and_replace(sim, "mob", UNNAMED_BASE_XML, UNNAMED_BASE_UNPREFIXED)

    assert sim._world.robots["mob"].joint_names == ["shoulder"], "the base freejoint is unnamed, so not enumerated"

    _plant_known_state(sim)

    obs = sim.get_observation(robot_name="mob", skip_images=True)

    assert BASE_KEYS <= set(obs), f"mobile base observed as a fixed-base arm after replace: {sorted(obs)}"
    _assert_planted_base(
        obs["base_pos"], obs["base_quat"], obs["base_lin_vel"], obs["base_ang_vel"], where="get_observation"
    )
    assert obs["shoulder"] == pytest.approx(KNOWN_HINGE_POS, abs=1e-6)


def test_get_robot_state_base_entry_survives_namespace_dropping_replace(sim):
    """``get_robot_state`` reaches the same tree-walk fallback and must keep
    reporting the structured ``base`` entry, not just the scalar joints."""
    _add_and_replace(sim, "mob", UNNAMED_BASE_XML, UNNAMED_BASE_UNPREFIXED)

    _plant_known_state(sim)

    result = sim.get_robot_state(robot_name="mob")

    assert result["status"] == "success", result
    text = result["content"][0]["text"]
    assert "base: pos=" in text, f"structured base entry lost after replace: {text}"
    assert "quat=" in text
    assert "shoulder: pos=" in text
    # The rendered text is rounded to 4dp, so assert the planted values on the
    # structured payload that recording and policies actually consume.
    payload = result["content"][1]["json"]
    base = payload.get("base")
    assert base is not None, f"structured base entry lost after replace: {payload}"
    _assert_planted_base(
        base["position"],
        base["quaternion"],
        base["linear_velocity"],
        base["angular_velocity"],
        where="get_robot_state",
    )
    assert payload["state"]["shoulder"]["position"] == pytest.approx(KNOWN_HINGE_POS, abs=1e-6)


def test_free_base_joint_id_resolves_the_base_after_namespace_dropping_replace(sim):
    """The finder shared by terrain seating and the recording schema resolves the
    base for both floating-base shapes after the replace."""
    _add_and_replace(sim, "hum", NAMED_BASE_XML, NAMED_BASE_UNPREFIXED)
    world = sim._world

    named = sim._robot_free_base_joint_id(world._model, world.robots["hum"])
    assert named >= 0, "named floating base unresolvable after replace"
    assert world._model.jnt_type[named] == sim._mj.mjtJoint.mjJNT_FREE


def test_free_base_joint_id_resolves_an_unnamed_base_after_namespace_dropping_replace(sim):
    """Same finder, the unnamed-``<freejoint>`` shape."""
    _add_and_replace(sim, "mob", UNNAMED_BASE_XML, UNNAMED_BASE_UNPREFIXED)
    world = sim._world

    unnamed = sim._robot_free_base_joint_id(world._model, world.robots["mob"])
    assert unnamed >= 0, "unnamed floating base unresolvable after replace"
    assert world._model.jnt_type[unnamed] == sim._mj.mjtJoint.mjJNT_FREE


def test_base_state_disappears_when_the_bare_name_misses_too(sim):
    """The pin above is not vacuous: the fallback resolves by bare NAME, so a
    replacement scene whose joints are renamed leaves nothing to find - the base
    state must be absent rather than borrowed from an unrelated free joint."""
    _add_and_replace(sim, "mob", UNNAMED_BASE_XML, RENAMED_JOINTS_XML, bare_resolves=False)
    world = sim._world

    obs = sim.get_observation(robot_name="mob", skip_images=True)
    assert not (BASE_KEYS & set(obs)), f"base state reported for a robot with no resolvable joint: {sorted(obs)}"
    assert obs == {}, f"no registered joint resolves, so the observation is empty: {sorted(obs)}"

    assert sim._robot_free_base_joint_id(world._model, world.robots["mob"]) == -1


def test_the_namespaced_joint_is_preferred_over_a_bare_homonym(sim):
    """With both ``hum/hip`` and a decoy ``hip`` compiled, the observation reads
    ours. The retry is a fallback, so it must not fire when the prefixed lookup
    succeeds - otherwise an unrelated body's angle is reported as this robot's."""
    _add_and_replace(sim, "hum", NAMED_BASE_XML, BOTH_SPELLINGS_XML, prefixed_resolves=True)
    world = sim._world
    model, data = world._model, world._data

    ours, decoy = _joint_id(sim, "hum/hip"), _joint_id(sim, "hip")
    assert ours >= 0 and decoy >= 0 and ours != decoy, "the scene must compile both spellings"
    data.qpos[model.jnt_qposadr[ours]] = 0.25
    data.qpos[model.jnt_qposadr[decoy]] = -0.75
    sim._mj.mj_forward(model, data)

    obs = sim.get_observation(robot_name="hum", skip_images=True)
    assert obs["hip"] == pytest.approx(0.25), "the bare homonym was read instead of the robot's own joint"


def test_the_namespaced_free_base_is_preferred_over_a_bare_homonym(sim):
    """Same precedence on the free-base finder: the decoy body carries a free
    joint under the bare name, and the finder must return the robot's own."""
    _add_and_replace(sim, "hum", NAMED_BASE_XML, BOTH_SPELLINGS_XML, prefixed_resolves=True)
    world = sim._world

    resolved = sim._robot_free_base_joint_id(world._model, world.robots["hum"])
    assert resolved == _joint_id(sim, "hum/floating_base_joint"), "the finder resolved the decoy's free joint"
