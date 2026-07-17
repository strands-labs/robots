"""Multi-robot ejection: surviving-robot state + fail-soft rebuild contract.

``Simulation.remove_robot`` rebuilds the whole MJCF from the remaining
``world.robots`` and re-attaches every survivor (see
:func:`strands_robots.simulation.mujoco.scene_ops.eject_robot_from_scene`).
The guardrail suite pins the "no compiled world" and unknown-body early
returns; the snapshot/restore round-trip is pinned per joint-type width.

What was NOT pinned - and is pinned here - is the behaviour a scene with more
than one robot depends on:

* a surviving robot keeps its joint state across the rebuild AND has its
  actuator/joint ids re-resolved against the freshly compiled model, so it can
  still be driven after a sibling is removed; and
* the documented fail-soft contract: if the rebuild cannot compile, the eject
  reports failure rather than leaving a half-mutated world, and
  ``remove_robot`` surfaces that as a structured error.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco import scene_ops  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Two single-joint arms; attaching both namespaces them as ``armN/...`` so the
# rebuild has to re-resolve ids by fully-qualified name.
_ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link0" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def two_arm_sim(tmp_path):
    """A compiled world holding two attached single-joint arms."""
    arm_path = tmp_path / "arm.xml"
    arm_path.write_text(_ARM_XML)
    sim = Simulation(tool_name="devx_multi_eject", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm1", urdf_path=str(arm_path))
    sim.add_robot(name="arm2", urdf_path=str(arm_path))
    try:
        yield sim
    finally:
        sim.cleanup(policy_stop_timeout=0.5)


def _joint_qpos(sim: Simulation, joint_name: str) -> float:
    world = sim._world
    assert world is not None and world._model is not None
    mj = sim._mj
    jid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    assert jid >= 0, f"joint {joint_name!r} not in compiled model"
    adr = int(world._model.jnt_qposadr[jid])
    return float(world._data.qpos[adr])


def test_remove_robot_preserves_survivor_state_and_reresolves_actuators(two_arm_sim):
    """Removing one arm keeps the survivor's pose and leaves it drivable.

    The XML round-trip reallocates ``model``/``data`` and shifts every body/
    joint index, so the survivor's cached actuator ids must be rebuilt by name.
    A survivor that lost its actuator ids would silently stop responding to
    ``send_action`` - the regression this pins.
    """
    sim = two_arm_sim
    world = sim._world
    assert world is not None
    mj = sim._mj

    # Put the survivor at a distinct, non-zero pose before the rebuild.
    surv_jid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_JOINT, "arm2/pan")
    surv_adr = int(world._model.jnt_qposadr[surv_jid])
    world._data.qpos[surv_adr] = 0.5
    mj.mj_forward(world._model, world._data)

    result = sim.remove_robot("arm1")
    assert result["status"] == "success", result

    # Registry reflects the removal.
    remaining = sim.list_robots()
    assert "arm1" not in remaining
    assert "arm2" in remaining

    # Survivor kept its pose across the fresh compile.
    assert _joint_qpos(sim, "arm2/pan") == pytest.approx(0.5, abs=1e-6)

    # Actuator ids were re-resolved against the new model, so the survivor is
    # still drivable: a position command moves the joint toward its target.
    survivor = sim._world.robots["arm2"]
    assert survivor.actuator_ids, "survivor lost its actuator ids after rebuild"

    send = sim.send_action({"arm2/pan": 1.2}, robot_name="arm2")
    assert not (isinstance(send, dict) and send.get("status") == "error"), send
    for _ in range(200):
        sim.step()
    assert _joint_qpos(sim, "arm2/pan") > 0.5, "survivor did not track the new command"


def test_remove_robot_reports_error_when_eject_fails(two_arm_sim, monkeypatch):
    """A rebuild that cannot eject surfaces a structured error, not a crash.

    ``remove_robot`` pops the target from the registry before delegating to
    ``eject_robot_from_scene``; if the eject returns ``False`` the caller must
    report failure rather than pretend success on a half-mutated world.
    """
    sim = two_arm_sim
    monkeypatch.setattr(
        "strands_robots.simulation.mujoco.simulation.eject_robot_from_scene",
        lambda *a, **k: False,
    )
    result = sim.remove_robot("arm1")
    assert result["status"] == "error"
    assert "arm1" in result["content"][0]["text"]


def test_eject_from_scene_returns_false_when_recompile_fails(two_arm_sim, monkeypatch):
    """``eject_robot_from_scene`` fails soft (returns ``False``) on a bad compile.

    A rebuilt spec that will not compile must not install a broken
    ``model``/``data`` pair; the helper returns ``False`` and leaves the prior
    world intact so the caller can report the failure.
    """
    sim = two_arm_sim
    world = sim._world
    assert world is not None
    prior_model = world._model

    # Drop every robot so the survivor re-attach loop is a no-op, then force the
    # fresh compile of the rebuilt spec to raise the way MuJoCo would on an
    # invalid model.
    world.robots.clear()
    bad_spec = MagicMock()
    bad_spec.compile.side_effect = ValueError("compile boom")
    monkeypatch.setattr(scene_ops.SpecBuilder, "build", staticmethod(lambda _world: bad_spec))

    assert scene_ops.eject_robot_from_scene(world, "arm1") is False
    # The failed rebuild did not swap in the broken model.
    assert world._model is prior_model
