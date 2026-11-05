"""The XML walk (used when MuJoCo is not installed) reads joint limits the way MuJoCo's compiler does."""

from __future__ import annotations

import math

import pytest

from strands_robots import Description
from strands_robots.description.introspect.xml import introspect_xml


def _limits(tmp_path, body: str, *, compiler: str = "", default: str = "", extra: str = "") -> dict:
    xml = f"<mujoco>{compiler}{default}<worldbody>{body}</worldbody>{extra}</mujoco>"
    path = tmp_path / "m.xml"
    path.write_text(xml)
    return introspect_xml(path, "mjcf")["limits"]


def _close(a, b):
    return all(math.isclose(x, y, abs_tol=1e-9) for x, y in zip(a, b, strict=True))


def test_hinge_ranges_are_degrees_unless_the_compiler_says_radian(tmp_path):
    body = '<body><joint name="j" range="-90 90"/><geom size="0.1"/></body>'
    assert _close(_limits(tmp_path, body)["j"], (-math.pi / 2, math.pi / 2))
    assert _close(_limits(tmp_path, body, compiler='<compiler angle="degree"/>')["j"], (-math.pi / 2, math.pi / 2))
    assert _limits(tmp_path, body, compiler='<compiler angle="radian"/>')["j"] == (-90.0, 90.0)


def test_slide_ranges_are_never_converted(tmp_path):
    body = '<body><joint name="s" type="slide" range="0 0.5"/><geom size="0.1"/></body>'
    assert _limits(tmp_path, body)["s"] == (0.0, 0.5)


def test_ranges_come_from_the_default_class_tree(tmp_path):
    default = (
        '<default><joint range="-1 1"/><default class="arm"><joint range="-2 2"/>'
        '<default class="wrist"><joint damping="1"/></default></default></default>'
    )
    body = (
        '<body childclass="arm"><joint name="shoulder"/><geom size="0.1"/>'
        '<body><joint name="wrist" class="wrist"/><geom size="0.1"/></body></body>'
        '<body><joint name="base"/><geom size="0.1"/></body>'
    )
    lim = _limits(tmp_path, body, compiler='<compiler angle="radian"/>', default=default)
    assert lim == {"shoulder": (-2.0, 2.0), "wrist": (-2.0, 2.0), "base": (-1.0, 1.0)}


def test_limited_false_and_autolimits_off_leave_a_joint_unlimited(tmp_path):
    body = '<body><joint name="a" range="-1 1" limited="false"/><joint name="b" range="-1 1"/><geom size="0.1"/></body>'
    radian = '<compiler angle="radian"/>'
    assert _limits(tmp_path, body, compiler=radian) == {"b": (-1.0, 1.0)}
    off = '<compiler angle="radian" autolimits="false"/>'
    assert _limits(tmp_path, body, compiler=off) == {}


def test_default_joints_are_not_counted_as_joints(tmp_path):
    path = tmp_path / "m.xml"
    path.write_text(
        '<mujoco><default><joint range="-1 1"/></default><worldbody><body><joint name="j"/><geom size="0.1"/>'
        "</body></worldbody></mujoco>"
    )
    assert introspect_xml(path, "mjcf")["joints"] == ["j"]


def test_includes_are_read_where_they_appear(tmp_path):
    (tmp_path / "arm.xml").write_text(
        '<mujocoinclude><body><joint name="j" range="-45 45"/><geom size="0.1"/></body></mujocoinclude>'
    )
    (tmp_path / "m.xml").write_text('<mujoco><worldbody><include file="arm.xml"/></worldbody></mujoco>')
    assert _close(introspect_xml(tmp_path / "m.xml", "mjcf")["limits"]["j"], (-math.pi / 4, math.pi / 4))


@pytest.mark.parametrize("name", ["so100", "so101", "unitree_g1", "panda", "shadow_hand", "unitree_go2"])
def test_the_xml_walk_matches_mujoco_on_registry_models(name):
    pytest.importorskip("mujoco")
    from strands_robots.description.introspect.mujoco import introspect_mujoco

    path = Description.load(name).model_path
    a, b = introspect_mujoco(path), introspect_xml(path, "mjcf")
    assert a["joints"] == b["joints"] and a["action_joints"] == b["action_joints"]
    assert a["limits"].keys() == b["limits"].keys()
    for key, rng in a["limits"].items():
        assert _close(rng, b["limits"][key]), key
