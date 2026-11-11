"""A real robot drives only joints with a range: the safety chain reads a joint without one as unconstrained."""

from __future__ import annotations

import dataclasses
import sys

import pytest

from strands_robots import Description, Robot
from strands_robots.robot.safety import RequestRefused

ARM = """<mujoco model="arm">
  <default><default class="arm"><joint range="-100 100"/></default></default>
  <worldbody><body name="b"><joint name="shoulder" range="-100 100"/><geom size="0.1" mass="1"/>
    <body name="c" childclass="arm" pos="0 0 0.3"><joint name="elbow"/><geom size="0.1" mass="1"/>
      <body name="d" pos="0 0 0.3"><joint name="wrist" {wrist}/><geom size="0.1" mass="1"/></body></body></body>
  </worldbody>
  <actuator><position name="shoulder" joint="shoulder"/><position name="elbow" joint="elbow"/>
    <position name="wrist" joint="wrist"/></actuator></mujoco>"""


def _model(tmp_path, wrist: str) -> str:
    path = tmp_path / "arm.xml"
    path.write_text(ARM.format(wrist=wrist))
    return str(path)


@pytest.fixture(params=["mujoco", "xml-walk"])
def reader(request, monkeypatch):
    """Both ways a model is read: MuJoCo, and the XML walk used when the mujoco extra is not installed."""
    if request.param == "xml-walk":
        monkeypatch.setitem(sys.modules, "mujoco", None)
    else:
        pytest.importorskip("mujoco")
    return request.param


@pytest.mark.parametrize(
    "wrist", ['limited="false" range="-1 1"', 'class="main"'], ids=["limited-false", "no-range-in-its-class"]
)
def test_a_real_robot_refuses_a_model_with_a_driven_joint_without_a_range(tmp_path, reader, wrist):
    with pytest.raises(ValueError, match=r"joint\(s\) \['wrist'\] have no range"):
        Robot(_model(tmp_path, wrist), driver="memory")


def test_a_model_whose_joints_all_have_ranges_drives_a_real_robot(tmp_path, reader):
    real = Robot(_model(tmp_path, 'range="-45 45"'), driver="memory")
    assert set(real.description.limits) == {"shoulder", "elbow", "wrist"}  # degrees, the class's range, its own
    with pytest.raises(RequestRefused, match="outside its range"):
        real.confirm_act({"wrist": 1e6}, confirmed=True)
    real.confirm_act({"wrist": 0.5}, confirmed=True)
    real.close()


def test_a_simulated_robot_still_takes_a_joint_without_a_range(tmp_path):
    pytest.importorskip("mujoco")
    from strands_robots import Simulation

    sim = Simulation()
    arm = Robot(_model(tmp_path, 'limited="false"'), sim=sim)
    assert "wrist" not in arm.description.limits
    arm.act({"wrist": 0.5})
    sim.close()


def test_the_joint_limit_check_refuses_a_real_joint_without_a_range():
    real = Robot("so101", driver="memory")
    del real.description.limits["wrist_flex"]  # a description changed after the robot was built
    with pytest.raises(RequestRefused, match="wrist_flex has no range in the model"):
        real.confirm_act({"wrist_flex": 0.1}, confirmed=True)
    real.close()


def test_a_joint_that_turns_without_end_drives_a_real_robot_inside_its_declared_safe_bound(tmp_path, monkeypatch):
    model, load = _model(tmp_path, 'limited="false"'), Description.load

    def declared(name):  # the registry entry's hardware.safe bound for the continuous wrist
        d = load(name)
        return dataclasses.replace(d, hardware=dataclasses.replace(d.hardware, safe={"wrist": (-3.0, 3.0)}))

    monkeypatch.setattr(Description, "load", staticmethod(declared))
    real = Robot(model, driver="memory")
    real.confirm_act({"wrist": 2.0}, confirmed=True)  # the joint moves, its declared bound is the one checked
    with pytest.raises(RequestRefused, match="wrist=10.0000 rad outside safe"):
        real.gate({"wrist": 10.0}, approved_run=True)  # an approved run's step outside it is refused
    real.close()
