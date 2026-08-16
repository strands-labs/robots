"""End-effector hint matching is component-aware, not bare-substring.

:func:`strands_robots.simulation.ik.discover_ee_frame` resolves an IK target
frame by looking for hint words in body/site names. The hints name *components*
of a name, so they must match on word boundaries: the short hints (``"ee"``,
``"eef"``, ``"tcp"``) otherwise fire inside unrelated words - ``"ee"`` occurs in
``"knee"``, ``"wheel"`` and ``"unitree"`` - and resolve a leg link or a drive
wheel as a robot's end-effector, which then silently becomes the frame every
Cartesian target and every eef-delta policy chunk is applied to.

This is the same guarantee as the namespace-stripping one (a namespace that
contains a hint substring must not trigger a hint match), applied within the
name itself.

Models are built from inline MJCF so the real ``mj_id2name`` traversal runs;
skips cleanly when the ``sim-mujoco`` extra is absent.
"""

import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from strands_robots.simulation.ik import discover_ee_frame  # noqa: E402


def _model(xml: str) -> "mujoco.MjModel":
    return mujoco.MjModel.from_xml_string(xml)


def _chain(namespace: str, links: str) -> str:
    """A single serial chain of hinge-jointed bodies named by ``links``."""
    body_names = links.split()
    xml = f'<mujoco><worldbody><body name="{namespace}base">'
    xml += f'<joint name="{namespace}j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>'
    for i, name in enumerate(body_names):
        xml += f'<body name="{namespace}{name}">'
        xml += f'<joint name="{namespace}j{i + 1}" type="hinge"/><geom type="box" size=".05 .05 .05"/>'
    xml += "</body>" * len(body_names)
    xml += "</body></worldbody></mujoco>"
    return xml


# --------------------------------------------------------------------------
# A hint must not match inside an unrelated word.
# --------------------------------------------------------------------------


def test_knee_is_not_an_end_effector_when_a_wrist_exists() -> None:
    """A humanoid resolves its wrist, not the ``kn-ee`` the ``ee`` hint used to
    match: ``ee`` precedes ``wrist`` in the body-hint order, so a substring
    match on a leg link outranked the arm entirely."""
    xml = _chain("g1/", "left_knee_link left_elbow_link left_wrist_roll_link")
    assert discover_ee_frame(_model(xml), "g1/") == ("g1/left_wrist_roll_link", "body")


def test_drive_wheel_is_not_an_end_effector_on_a_mobile_manipulator() -> None:
    """A wheeled base carrying an arm resolves the arm's wrist, not a
    ``wh-ee-l`` link."""
    xml = _chain("lekiwi/", "wheel_hub_back_link wheel_back_link Wrist_Pitch_Roll")
    assert discover_ee_frame(_model(xml), "lekiwi/") == ("lekiwi/Wrist_Pitch_Roll", "body")


def test_hyphen_separated_knee_is_not_an_end_effector() -> None:
    """Hyphens delimit name components too, so ``left-knee`` does not match
    ``ee`` and discovery falls through to the leaf body."""
    xml = _chain("cassie/", "left-knee left-shin left-foot")
    assert discover_ee_frame(_model(xml), "cassie/") == ("cassie/left-foot", "body")


def test_site_hint_does_not_match_inside_a_word() -> None:
    """The site rung is subject to the same rule: a ``knee``-named site does not
    win rung 1, so a genuine hand body on rung 2 is resolved instead."""
    xml = """
    <mujoco><worldbody>
      <body name="r/upper">
        <joint name="r/j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        <site name="r/knee_marker" pos="0 0 .1"/>
        <body name="r/hand">
          <joint name="r/j1" type="hinge"/><geom type="box" size=".05 .05 .05"/>
        </body>
      </body>
    </worldbody></mujoco>
    """
    assert discover_ee_frame(_model(xml), "r/") == ("r/hand", "body")


def test_robot_name_in_an_unnamespaced_world_does_not_match() -> None:
    """With no namespace the full name is searched, so the ``unitr-ee`` in a
    robot's own name must not resolve that robot's pelvis site as its TCP."""
    xml = """
    <mujoco><worldbody>
      <body name="unitree_g1/pelvis">
        <joint name="unitree_g1/j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        <site name="unitree_g1/imu_in_pelvis" pos="0 0 .1"/>
        <body name="unitree_g1/left_wrist_roll_link">
          <joint name="unitree_g1/j1" type="hinge"/><geom type="box" size=".05 .05 .05"/>
        </body>
      </body>
    </worldbody></mujoco>
    """
    assert discover_ee_frame(_model(xml), None) == ("unitree_g1/left_wrist_roll_link", "body")


# --------------------------------------------------------------------------
# Every way a hint is legitimately spelled still matches.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("site_name", "hint_described"),
    [
        ("attachment_site", "the menagerie multi-token convention"),
        ("attachment", "a single-token prefix hint"),
        ("grasp_point", "a hint as the leading token"),
        ("left_pinch", "a hint as the trailing token"),
        ("tcp", "a whole-name hint"),
        ("ee_site", "a multi-token short hint"),
        ("ee", "a bare short hint as the whole name"),
        ("gripper_ee", "a short hint as the trailing token"),
        ("tool_flange", "a hint as the trailing token"),
    ],
)
def test_a_hinted_site_is_still_resolved(site_name: str, hint_described: str) -> None:
    """Component matching keeps every legitimate hint spelling resolving,
    including the short hints when they really are a name component."""
    xml = f"""
    <mujoco><worldbody>
      <body name="a/link0">
        <joint name="a/j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        <site name="a/{site_name}" pos="0 0 .2"/>
      </body>
    </worldbody></mujoco>
    """
    assert discover_ee_frame(_model(xml), "a/") == (f"a/{site_name}", "site"), hint_described


@pytest.mark.parametrize(
    "body_name",
    ["hand", "gripper", "tool0", "wristYawLeft", "eef_link", "end_effector_frame", "robotiq_2f85_flange"],
)
def test_a_hinted_body_is_still_resolved(body_name: str) -> None:
    """Body hints match across the separator conventions MuJoCo names use:
    snake_case, camelCase, and a trailing digit (``tool0``)."""
    xml = _chain("b/", f"link1 {body_name}")
    assert discover_ee_frame(_model(xml), "b/") == (f"b/{body_name}", "body")


def test_hint_priority_is_unchanged() -> None:
    """Earlier hints still outrank later ones within a rung."""
    xml = """
    <mujoco><worldbody>
      <body name="a/l0">
        <joint name="a/j0" type="hinge"/><geom type="box" size=".1 .1 .1"/>
        <site name="a/tcp_site" pos="0 0 .1"/>
        <site name="a/grasp_point" pos="0 0 .2"/>
      </body>
    </worldbody></mujoco>
    """
    assert discover_ee_frame(_model(xml), "a/") == ("a/grasp_point", "site")


def test_leaf_fallback_when_no_component_matches() -> None:
    """A chain whose names carry no hint component still resolves its leaf
    body, so a robot with no tool-like name keeps a usable frame."""
    xml = _chain("plain/", "link1 link2 link3")
    assert discover_ee_frame(_model(xml), "plain/") == ("plain/link3", "body")


# --------------------------------------------------------------------------
# ``list_bodies`` advertises the same mount, under the same rule.
#
# ``list_bodies(robot_name=...)`` answers "which body is this robot's gripper /
# end-effector mount" in its ``gripper_body`` field - the discovery surface a
# caller uses to resolve ``add_camera(parent_body=...)`` for a wrist view. It
# asked that question with its own bare-substring scan, so the same short hint
# fired inside the same unrelated words, and the mount it advertised for a
# humanoid or a mobile manipulator was a knee or a drive wheel. Both backends
# now match through the matcher above, so one rule answers the question
# wherever it is asked.
# --------------------------------------------------------------------------

import types  # noqa: E402

from strands_robots.simulation.models import SimRobot, SimWorld  # noqa: E402
from strands_robots.simulation.newton.simulation import NewtonSimEngine  # noqa: E402


def _mujoco_gripper_body(tmp_path, robot: str, links: str) -> str | None:
    """``list_bodies(robot_name=robot)["gripper_body"]`` for an inline chain.

    Drives the real backend end to end - ``create_world``, ``add_robot`` from a
    written MJCF, then ``list_bodies`` - so the namespacing and the model
    traversal are the production ones. Renders nothing, downloads nothing.
    """
    # `list_bodies` is a concrete-backend method, not a member of the `SimEngine`
    # ABC that `create_simulation` is annotated to return, so the backend is
    # constructed directly - the same class the factory resolves for
    # `backend="mujoco"`. Mirrors the sibling entity-name-domain tests.
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    path = tmp_path / f"{robot}.xml"
    path.write_text(_chain("", links))
    sim = MuJoCoSimEngine()
    try:
        sim.create_world(ground_plane=False)
        added = sim.add_robot(name=robot, urdf_path=str(path))
        assert added["status"] == "success", added["content"][0]["text"]
        payload = sim.list_bodies(robot_name=robot)["content"][1]["json"]
        assert payload["bodies"], "the chain must contribute bodies to search"
        return payload["gripper_body"]
    finally:
        sim.destroy()


def test_list_bodies_does_not_advertise_a_knee_as_the_end_effector_mount(tmp_path) -> None:
    """A legged robot with no gripper-like body advertises no mount at all.

    Pre-fix ``ee`` matched inside ``left_knee_link`` and the surface named a
    leg link as the robot's gripper/EEF mount - so a caller resolving a wrist
    camera mount from discovery aimed it at the knee.
    """
    assert _mujoco_gripper_body(tmp_path, "g1", "left_knee_link left_ankle_link torso_link") is None


def test_list_bodies_advertises_the_gripper_over_a_drive_wheel(tmp_path) -> None:
    """A wheeled base carrying an arm advertises the arm's gripper.

    The wheel comes first in body order, so pre-fix the ``ee`` in ``wh-ee-l``
    won and the gripper further down the chain was never reached.
    """
    got = _mujoco_gripper_body(tmp_path, "stretch", "link_right_wheel link_arm link_gripper_slider")
    assert got == "stretch/link_gripper_slider"


@pytest.mark.parametrize(
    "leaf",
    ["gripper", "hand", "ee_link", "gripper_ee", "tool0", "toolFlange", "left_hand_link"],
)
def test_list_bodies_still_advertises_a_hinted_body(tmp_path, leaf: str) -> None:
    """Every spelling of a hint this surface accepts still resolves, including a
    short hint that really is a whole name component - the fix narrows where a
    hint may fire, not which names carry one."""
    assert _mujoco_gripper_body(tmp_path, "arm", f"link1 {leaf}") == f"arm/{leaf}"


def _newton_gripper_body(labels: list[str]) -> str | None:
    """``gripper_body`` from Newton's ``list_bodies`` for ``labels``.

    Newton's ``list_bodies`` reads only the robot registry and the per-robot
    body map, so a stand-in ``self`` carrying those exercises the real method
    without a Newton runtime - the pattern its sibling backend-parity tests use.
    """
    world = SimWorld()
    world.robots["bot"] = SimRobot(name="bot", urdf_path="bot.xml", data_config="bot", joint_names=[])
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = world
    engine._model = types.SimpleNamespace(body_label=list(labels))
    engine._robot_body_map = {"bot": list(labels)}
    result = NewtonSimEngine.list_bodies(engine, "bot")
    assert result["status"] == "success"
    return result["content"][1]["json"]["gripper_body"]


def test_newton_list_bodies_does_not_advertise_a_knee_or_a_wheel() -> None:
    """The Newton backend answers the mount question under the same rule.

    It carried its own copy of the bare-substring scan, so the parity that
    matters to a caller - the same robot reports the same mount on either
    backend - held only for robots whose names happened not to contain ``ee``.
    """
    assert _newton_gripper_body(["bot/base/left_knee_link", "bot/base/wheel_hub_back_link"]) is None


def test_newton_list_bodies_still_advertises_a_jaw() -> None:
    """Newton's own extra hint (``jaw``) keeps matching a real jaw body."""
    labels = ["bot/base/left_knee_link", "bot/base/Moving_Jaw"]
    assert _newton_gripper_body(labels) == "bot/base/Moving_Jaw"


@pytest.mark.parametrize(
    "links,expected",
    [
        ("shoulder_link gripper_body", "gripper_body"),
        ("shoulder_link left_hand", "left_hand"),
        ("shoulder_link tool_flange", "tool_flange"),
        ("shoulder_link ee_link", "ee_link"),
        ("shoulder_link EE_BODY_R", "EE_BODY_R"),
        ("shoulder_link wristYawLeft toolTip", "toolTip"),
        ("shoulder_link tool0", "tool0"),
    ],
)
def test_list_bodies_still_advertises_every_mount_spelling(tmp_path, links: str, expected: str) -> None:
    """Every legitimate gripper-mount spelling still resolves at this surface.

    snake_case, SCREAMING_CASE, camelCase, a trailing digit, and a short hint
    that is one whole token of a longer name. The rule narrows what matches
    *inside* a word, not which names are gripper mounts, so narrowing it must
    not cost a single real mount.
    """
    assert _mujoco_gripper_body(tmp_path, "arm", links) == f"arm/{expected}"


def test_both_surfaces_agree_on_the_same_robot(tmp_path) -> None:
    """The IK target frame and the advertised mount resolve the same body.

    The two surfaces keep separate hint vocabularies but answer one question,
    so a robot whose only end-effector-like body is its gripper must resolve to
    that body at both. While each carried its own bare-substring scan a drive
    wheel could be the end-effector to one surface and not to the other - this
    measures the agreement through the two public surfaces rather than through
    the shared helper, so it holds regardless of how the rule is factored.
    """
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    path = tmp_path / "mobile.xml"
    path.write_text(_chain("", "link_right_wheel lift_link link_gripper_slider"))
    sim = MuJoCoSimEngine()
    try:
        sim.create_world(ground_plane=False)
        assert sim.add_robot(name="mobile", urdf_path=str(path))["status"] == "success"
        advertised = sim.list_bodies(robot_name="mobile")["content"][1]["json"]["gripper_body"]
        discovered = discover_ee_frame(sim.mj_model, "mobile/")
    finally:
        sim.destroy()

    assert discovered is not None
    frame, kind = discovered
    assert (frame, kind) == ("mobile/link_gripper_slider", "body")
    assert advertised == frame, (advertised, frame)
