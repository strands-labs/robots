"""A model whose actuators outnumber their control slots is refused, not half-driven.

Every actuator surface in the MuJoCo backend addresses an actuator BY ITS
CONTROL INDEX: ``SimRobot.actuator_ids`` holds indices into ``data.ctrl``, and
the ``actuator_*`` model arrays are read at that same index. Through mujoco 3.11
that identity was free -- every actuator owned exactly one control slot, so
``model.nu`` was both the control width and the actuator count.

mujoco 3.12 separated them. ``<pid>`` takes two controls by default
(``input="pos vel"``) and ``<orientation>`` up to four, so a caller's MJCF can
compile with ``nu > nactuator``. Measured on 3.12.0 for a two-joint arm:

===================  ==  =========  =======  =======
actuators             nu  nactuator  ctrladr  ctrlnum
===================  ==  =========  =======  =======
position, position    2          2   [0, 1]   [1, 1]
position, pid         3          2   [0, 1]   [1, 2]
pid, position         3          2   [0, 2]   [2, 1]
===================  ==  =========  =======  =======

Both directions of the old identity now break. Iterating ``range(nu)`` indexes
the actuator-indexed arrays past their end -- which is what ``add_robot`` did,
reporting ``Failed to load: index 2 is out of bounds for axis 0 with size 2``, a
numpy message naming neither the actuator, the element, nor mujoco. Worse, that
raise escaped both the recompile refusal path and the caller's spec restore, so
the robot's bodies stayed compiled into a scene whose registry had already
dropped it: ``nbody`` went 1 -> 3 with ``arm/link`` and ``arm/link2`` present and
``list_robots() == []``.

Iterating ``range(nactuator)`` instead is not the fix. On the ``pid, position``
row the position servo's setpoint is ``ctrl[2]`` while such a loop writes
``ctrl[1]`` -- the pid's *velocity* setpoint -- so a pose would be applied as a
rate and the servo never commanded. Choosing which of an actuator's controls a
pose is written to, and what ``robot_action_keys`` reports for the rest, is a
control layout this surface does not have, so the model is declined with the
actuator named instead of driven by guesswork.

The refusal sits where the compiler's own refusal sits -- before the model is
installed -- so a refused add leaves the scene exactly as it was found, and
``inject_robot_into_scene`` now carries the reason out the way
``inject_object_into_scene`` already does rather than folding it into a bare
``False``.

These tests are GL-free (``mesh=False``, no render) so they run in CI without a
display. The reason itself is graded against stub models, so the rule is pinned
on every supported build; only the end-to-end half needs a build that can
compile a multi-control actuator.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import mujoco
import pytest

from strands_robots.simulation.mujoco import scene_ops
from strands_robots.simulation.mujoco.simulation import Simulation

# A two-joint arm. ``{acts}`` is the actuator block under test.
_ARM = """
<mujoco model="ctrlslots">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="link1" pos="0 0 0.3">
      <joint name="j1" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
      <geom name="g1" type="capsule" size="0.02 0.1" mass="0.5"/>
      <body name="link2" pos="0 0 -0.2">
        <joint name="j2" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom name="g2" type="capsule" size="0.02 0.08" mass="0.3"/>
      </body>
    </body>
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>
"""

_SINGLE_CONTROL = '<position name="a1" joint="j1" kp="20"/><position name="a2" joint="j2" kp="20"/>'
_PID_SECOND = '<position name="a1" joint="j1" kp="20"/><pid name="a2" joint="j2" kp="20"/>'
_PID_FIRST = '<pid name="a1" joint="j1" kp="20"/><position name="a2" joint="j2" kp="20"/>'


def _write(tmp_path: Any, acts: str, name: str) -> str:
    p = tmp_path / f"{name}.xml"
    p.write_text(_ARM.format(acts=acts))
    return str(p)


def _compiles_multi_control() -> bool:
    """Whether the installed build compiles an actuator wider than one control."""
    try:
        model = mujoco.MjModel.from_xml_string(_ARM.format(acts=_PID_SECOND))
    except (ValueError, RuntimeError):
        return False
    return int(getattr(model, "nactuator", model.nu)) != int(model.nu)


_needs_multi_control = pytest.mark.skipif(
    not _compiles_multi_control(),
    reason="this mujoco build has no actuator that spans more than one control slot",
)


def _stub_model(*, nu: int, nactuator: int | None, ctrladr: list[int] | None, ctrlnum: list[int] | None) -> Any:
    """A model stand-in carrying only the fields the reason reads."""
    fields: dict[str, Any] = {"nu": nu}
    if nactuator is not None:
        fields["nactuator"] = nactuator
    if ctrladr is not None:
        fields["actuator_ctrladr"] = ctrladr
    if ctrlnum is not None:
        fields["actuator_ctrlnum"] = ctrlnum
    return SimpleNamespace(**fields)


def _stub_mj(names: dict[int, str | None]) -> Any:
    """A ``mujoco`` stand-in whose ``mj_id2name`` answers from ``names``."""
    return SimpleNamespace(
        mj_id2name=lambda _model, _objtype, act_id: names.get(act_id),
        mjtObj=SimpleNamespace(mjOBJ_ACTUATOR=object()),
    )


class TestTheReasonIsDerivedFromTheModel:
    """Graded against stub models, so the rule is pinned on every build."""

    def test_one_control_per_actuator_is_addressable(self):
        # The identity every build through 3.11 gave for free.
        model = _stub_model(nu=2, nactuator=2, ctrladr=[0, 1], ctrlnum=[1, 1])
        assert scene_ops._unaddressable_actuator_reason(model, _stub_mj({0: "a1", 1: "a2"})) is None

    def test_a_build_that_does_not_report_an_actuator_count_is_addressable(self):
        # On a build with no ``nactuator``, ``nu`` IS the actuator count, so
        # there is no multi-control actuator for the reason to declare.
        model = _stub_model(nu=2, nactuator=None, ctrladr=None, ctrlnum=None)
        assert scene_ops._unaddressable_actuator_reason(model, _stub_mj({})) is None

    def test_a_trailing_wide_actuator_is_named_with_its_slots(self):
        model = _stub_model(nu=3, nactuator=2, ctrladr=[0, 1], ctrlnum=[1, 2])
        reason = scene_ops._unaddressable_actuator_reason(model, _stub_mj({0: "a1", 1: "a2"}))
        assert reason is not None
        assert "'a2' owns control slots [1, 2]" in reason
        # The actuator that is addressable is not blamed.
        assert "'a1'" not in reason

    def test_a_leading_wide_actuator_is_named_with_its_slots(self):
        # The row a count-only fix reads wrong: writing ``ctrl[1]`` for actuator
        # 1 here hits the pid's velocity setpoint, and ``a2`` at ``ctrl[2]``
        # never gets its pose.
        model = _stub_model(nu=3, nactuator=2, ctrladr=[0, 2], ctrlnum=[2, 1])
        reason = scene_ops._unaddressable_actuator_reason(model, _stub_mj({0: "a1", 1: "a2"}))
        assert reason is not None
        assert "'a1' owns control slots [0, 1]" in reason

    def test_an_unnamed_wide_actuator_is_still_reported(self):
        model = _stub_model(nu=3, nactuator=2, ctrladr=[0, 1], ctrlnum=[1, 2])
        reason = scene_ops._unaddressable_actuator_reason(model, _stub_mj({0: "a1", 1: None}))
        assert reason is not None
        assert "unnamed actuator 1" in reason

    def test_a_build_reporting_no_slot_layout_is_still_refused(self):
        # The counts alone establish that some actuator is wider than one slot.
        # Without the per-actuator layout the reason cannot name which, and
        # reporting the counts beats addressing the model anyway.
        model = _stub_model(nu=3, nactuator=2, ctrladr=None, ctrlnum=None)
        reason = scene_ops._unaddressable_actuator_reason(model, _stub_mj({}))
        assert reason is not None
        assert "2 actuator(s) spanning 3 control slots" in reason

    def test_the_reason_names_the_elements_and_a_remedy(self):
        model = _stub_model(nu=3, nactuator=2, ctrladr=[0, 1], ctrlnum=[1, 2])
        reason = scene_ops._unaddressable_actuator_reason(model, _stub_mj({0: "a1", 1: "a2"}))
        assert reason is not None
        # The element that produced it, so the MJCF line is findable.
        assert "<pid>" in reason
        # A remedy, so the refusal is not a dead end.
        assert "<position>" in reason


class TestAMultiControlModelIsRefusedNotHalfDriven:
    """End-to-end through ``add_robot`` on a build that compiles one."""

    @pytest.fixture()
    def sim(self):
        s = Simulation(tool_name="devx_multi_control", mesh=False)
        s.create_world()
        try:
            yield s
        finally:
            s.cleanup(policy_stop_timeout=0.5)

    def test_a_single_control_robot_still_adds(self, sim, tmp_path):
        # The control: the shape every supported build compiles is untouched.
        added = sim.add_robot(name="arm", urdf_path=_write(tmp_path, _SINGLE_CONTROL, "ok"))
        assert added["status"] == "success", added["content"][0]["text"]
        assert sim.robot_action_keys("arm") == ["a1", "a2"]

    @_needs_multi_control
    @pytest.mark.parametrize("acts", [_PID_SECOND, _PID_FIRST], ids=["pid_second", "pid_first"])
    def test_the_refusal_names_the_actuator_rather_than_an_index(self, sim, tmp_path, acts):
        refused = sim.add_robot(name="arm", urdf_path=_write(tmp_path, acts, "wide"))
        assert refused["status"] == "error"
        text = refused["content"][0]["text"]
        # Pre-fix: "Failed to load: index 2 is out of bounds for axis 0 with
        # size 2" -- a numpy message naming nothing the caller wrote.
        assert "out of bounds" not in text, text
        assert "control slot" in text, text
        assert "<pid>" in text, text

    @_needs_multi_control
    @pytest.mark.parametrize("acts", [_PID_SECOND, _PID_FIRST], ids=["pid_second", "pid_first"])
    def test_a_refused_robot_leaves_no_subtree_in_the_scene(self, sim, tmp_path, acts):
        before = sim.mj_model
        sizes = (int(before.nbody), int(before.nu), int(before.njnt))

        refused = sim.add_robot(name="arm", urdf_path=_write(tmp_path, acts, "wide"))
        assert refused["status"] == "error"
        assert sim.list_robots() == []

        after = sim.mj_model
        assert (int(after.nbody), int(after.nu), int(after.njnt)) == sizes
        names = [mujoco.mj_id2name(after, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(after.nbody)]
        orphans = [n for n in names if n and n.startswith("arm/")]
        assert orphans == [], f"the refused robot left {orphans} compiled into the scene"

    @_needs_multi_control
    def test_the_scene_still_takes_later_mutations(self, sim, tmp_path):
        refused = sim.add_robot(name="arm", urdf_path=_write(tmp_path, _PID_SECOND, "wide"))
        assert refused["status"] == "error"
        # A refused robot costs exactly one refused add: the spec it was
        # attached into is back, so every later mutation still compiles.
        marker = sim.add_object(name="marker", shape="box", position=[0.3, 0.0, 0.1])
        assert marker["status"] == "success", marker["content"][0]["text"]
        retry = sim.add_robot(name="arm", urdf_path=_write(tmp_path, _SINGLE_CONTROL, "ok"))
        assert retry["status"] == "success", retry["content"][0]["text"]

    @_needs_multi_control
    def test_the_remedy_the_refusal_offers_is_accepted(self, sim, tmp_path):
        refused = sim.add_robot(name="arm", urdf_path=_write(tmp_path, _PID_SECOND, "wide"))
        text = refused["content"][0]["text"]
        offer = re.search(r"Replace it with a single-control actuator \(([^)]*)\)", text)
        assert offer is not None, text

        # Apply the first element the refusal offers, exactly as written.
        element = offer.group(1).split(",")[0].strip().strip("<>")
        acts = _PID_SECOND.replace('<pid name="a2"', f'<{element} name="a2"')
        retry = sim.add_robot(name="arm", urdf_path=_write(tmp_path, acts, "remedied"))
        assert retry["status"] == "success", retry["content"][0]["text"]
        assert sim.robot_action_keys("arm") == ["a1", "a2"]
