"""End-effector frame auto-discovery regression for VERA eef-delta IK.

Exercises :func:`strands_robots.policies.vera.ee_frame.discover_ee_frame`, the
namespace-aware heuristic that resolves an IK target frame from a compiled
``mujoco.MjModel`` so eef-delta / cartesian-delta embodiments stay zero-config.

The heuristic is a first-match-wins ladder:
  1. a TCP-like *site* (``attachment_site`` / ``grasp`` / ``tcp`` / ...),
  2. otherwise a hand/tool *body* (``gripper`` / ``hand`` / ``wrist`` / ...),
  3. otherwise the *leaf body* of the robot's kinematic chain.

Discovery is scoped to a body/site namespace prefix so multi-robot worlds
resolve the right arm, and hint matching runs on the namespace-stripped
basename so the namespace text cannot itself trigger a false hint match.

Tests build genuine ``MjModel`` instances from inline MJCF (no mocks) so the
real ``mj_id2name`` / ``body_parentid`` traversal is exercised. Skips cleanly
when the ``sim-mujoco`` extra is absent.
"""

import sys

import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from strands_robots.policies.vera.ee_frame import discover_ee_frame  # noqa: E402


def _model(xml: str) -> "mujoco.MjModel":
    return mujoco.MjModel.from_xml_string(xml)


# A namespaced 2-link arm carrying a conventional ``attachment_site``.
_SITE_ARM = """
<mujoco><worldbody>
  <body name="panda/link0">
    <joint name="panda/j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>
    <body name="panda/link1">
      <joint name="panda/j1" type="hinge"/><geom type="box" size=".1 .1 .1"/>
      <site name="panda/attachment_site" pos="0 0 0.2"/>
    </body>
  </body>
</worldbody></mujoco>
"""


def test_site_hint_wins_over_bodies() -> None:
    """A TCP-like site is the preferred IK frame (rung 1 of the ladder)."""
    frame = discover_ee_frame(_model(_SITE_ARM), "panda/")
    assert frame == ("panda/attachment_site", "site")


def test_namespace_none_still_matches_site() -> None:
    """With no namespace, discovery searches every name in the world."""
    frame = discover_ee_frame(_model(_SITE_ARM), None)
    assert frame == ("panda/attachment_site", "site")


def test_body_hint_when_no_site() -> None:
    """Falls to a hand/tool body (rung 2) when no site hint is present."""
    xml = """
    <mujoco><worldbody>
      <body name="arm/base">
        <joint name="arm/j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        <body name="arm/gripper">
          <joint name="arm/j1" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        </body>
      </body>
    </worldbody></mujoco>
    """
    assert discover_ee_frame(_model(xml), "arm/") == ("arm/gripper", "body")


def test_site_preferred_over_body_hint() -> None:
    """When both a hint site and a hint body exist, the site wins."""
    xml = """
    <mujoco><worldbody>
      <body name="a/base">
        <joint name="a/j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        <body name="a/hand">
          <joint name="a/j1" type="hinge"/><geom type="box" size=".1 .1 .1"/>
          <site name="a/tcp" pos="0 0 0.1"/>
        </body>
      </body>
    </worldbody></mujoco>
    """
    assert discover_ee_frame(_model(xml), "a/") == ("a/tcp", "site")


def test_leaf_body_fallback_when_no_hints() -> None:
    """With only generic body names, the deepest chain leaf is returned (rung 3)."""
    xml = """
    <mujoco><worldbody>
      <body name="r/link0">
        <joint name="r/j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        <body name="r/link1">
          <joint name="r/j1" type="hinge"/><geom type="box" size=".1 .1 .1"/>
          <body name="r/link2">
            <joint name="r/j2" type="hinge"/><geom type="box" size=".1 .1 .1"/>
          </body>
        </body>
      </body>
    </worldbody></mujoco>
    """
    assert discover_ee_frame(_model(xml), "r/") == ("r/link2", "body")


def test_namespace_scoping_selects_the_right_arm() -> None:
    """Two arms in one world resolve to their own namespace's frame."""
    xml = """
    <mujoco><worldbody>
      <body name="panda/link0">
        <joint name="panda/j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        <site name="panda/grasp" pos="0 0 0.1"/>
      </body>
      <body name="ur/link0">
        <joint name="ur/j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        <site name="ur/tcp" pos="0 0 0.1"/>
      </body>
    </worldbody></mujoco>
    """
    model = _model(xml)
    assert discover_ee_frame(model, "panda/") == ("panda/grasp", "site")
    assert discover_ee_frame(model, "ur/") == ("ur/tcp", "site")


def test_hint_priority_ordering() -> None:
    """Earlier hints outrank later ones: ``grasp`` beats ``tcp`` even when the
    tcp site is declared first in the model."""
    xml = """
    <mujoco><worldbody>
      <body name="a/l0">
        <joint name="a/j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        <site name="a/tcp_site" pos="0 0 0.1"/>
        <site name="a/grasp_point" pos="0 0 0.2"/>
      </body>
    </worldbody></mujoco>
    """
    assert discover_ee_frame(_model(xml), "a/") == ("a/grasp_point", "site")


def test_basename_strip_prevents_false_namespace_match() -> None:
    """Hint matching runs on the namespace-stripped basename, so a namespace
    that itself contains a hint substring (``eebot/``) does not falsely match
    the ``ee`` body hint; discovery falls through to the leaf body."""
    xml = """
    <mujoco><worldbody>
      <body name="eebot/base">
        <joint name="eebot/j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        <body name="eebot/link1">
          <joint name="eebot/j1" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        </body>
      </body>
    </worldbody></mujoco>
    """
    assert discover_ee_frame(_model(xml), "eebot/") == ("eebot/link1", "body")


def test_returns_none_when_mujoco_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discovery degrades to None (caller then asks for an explicit frame)
    when mujoco cannot be imported, rather than raising."""
    monkeypatch.setitem(sys.modules, "mujoco", None)
    assert discover_ee_frame(object(), "x/") is None


def test_returns_none_when_namespace_matches_nothing() -> None:
    """A namespace that scopes out every body and site resolves to None so the
    caller can warn and request an explicit frame, rather than mis-picking
    another robot's link."""
    xml = """
    <mujoco><worldbody>
      <body name="panda/link0">
        <joint name="panda/j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        <site name="panda/attachment_site" pos="0 0 0.1"/>
      </body>
    </worldbody></mujoco>
    """
    assert discover_ee_frame(_model(xml), "ghost/") is None
