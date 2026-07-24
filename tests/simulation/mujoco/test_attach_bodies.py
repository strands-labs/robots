"""Regression tests for ``attach_bodies`` / ``detach_bodies`` (GH #1533, PR 1).

The runtime grasp-assist primitives the ``so101_curobo`` example used to
hand-roll (per-waypoint ``move_object`` teleports + raw ``qvel`` zeroing).
Contracts pinned here:

* **weld** adds an equality constraint holding the CURRENT runtime relative
  pose (not the compile-time ``qpos0`` pose - MuJoCo's all-zero ``relpose``
  shortcut would silently weld at the spawn offset).
* **kinematic** teleport-follows the parent every physics step through both
  the ``step()`` batch loop and the policy-driven ``send_action`` substep
  loop, with the child's freejoint velocity zeroed (no teleport fling).
* ``detach_bodies`` releases: a welded child no longer follows; a kinematic
  child falls physically from the release point.
* Error contract: structured error dicts (never raises) for unknown bodies,
  duplicate attachments, bad mode, bad torquescale, kinematic children
  without a freejoint, and mismatched detach parents.
* ``remove_object`` / ``remove_robot`` refuse to remove a body an active
  attachment references (a dangling weld would fail the next recompile).
"""

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_attach_bodies_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    yield s
    s.cleanup(policy_stop_timeout=0.5)


def _body_xpos(sim, name):
    m, d = sim._world._model, sim._world._data
    bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"body {name!r} not found"
    return [float(v) for v in d.xpos[bid]]


def _freejoint_qvel(sim, name):
    m, d = sim._world._model, sim._world._data
    jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"{name}_joint")
    assert jid >= 0
    adr = m.jnt_dofadr[jid]
    return [float(v) for v in d.qvel[adr : adr + 6]]


def _offset(sim, parent, child):
    p = _body_xpos(sim, parent)
    c = _body_xpos(sim, child)
    return [c[i] - p[i] for i in range(3)]


@pytest.fixture
def two_boxes(sim):
    """A big dynamic 'carrier' box and a small 'cube', both airborne."""
    assert sim.add_object("carrier", shape="box", size=[0.1, 0.1, 0.1], position=[0, 0, 0.6])["status"] == "success"
    assert sim.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], position=[0.2, 0, 0.6])["status"] == "success"
    return sim


class TestWeldAttach:
    def test_weld_holds_current_runtime_pose_not_spawn_pose(self, two_boxes):
        """The weld must capture the pose AT ATTACH TIME - the core honesty fix.

        Move the cube away from its spawn offset first; after attaching, the
        held offset must be the runtime one (0.3 m), not the spawn one (0.2 m).
        """
        sim = two_boxes
        assert sim.move_object("cube", position=[0.3, 0.0, 0.6])["status"] == "success"
        result = sim.attach_bodies("carrier", "cube", mode="weld")
        assert result["status"] == "success", result

        sim.step(150)  # both fall + settle under gravity
        off = _offset(sim, "carrier", "cube")
        assert off[0] == pytest.approx(0.3, abs=0.02), f"weld must hold the RUNTIME offset, got {off}"

    def test_weld_survives_later_scene_recompile(self, two_boxes):
        """The equality lives on the spec, so add_object recompiles keep it."""
        sim = two_boxes
        assert sim.attach_bodies("carrier", "cube", mode="weld")["status"] == "success"
        assert sim.add_object("bystander", shape="box", size=[0.05, 0.05, 0.05], position=[1, 1, 0.5])["status"] == (
            "success"
        )
        sim.step(150)
        off = _offset(sim, "carrier", "cube")
        assert off[0] == pytest.approx(0.2, abs=0.02), f"weld lost across recompile: {off}"

    def test_detach_releases_weld(self, two_boxes):
        sim = two_boxes
        assert sim.attach_bodies("carrier", "cube", mode="weld")["status"] == "success"
        result = sim.detach_bodies("carrier", "cube")
        assert result["status"] == "success", result
        # Model must hold no equality constraints anymore.
        assert sim._world._model.neq == 0
        # And the pair can re-attach cleanly.
        assert sim.attach_bodies("carrier", "cube", mode="weld")["status"] == "success"


class TestKinematicAttach:
    def test_child_follows_parent_through_step(self, two_boxes):
        """The carrier free-falls; the kinematically attached cube rides along."""
        sim = two_boxes
        start_off = _offset(sim, "carrier", "cube")
        result = sim.attach_bodies("carrier", "cube", mode="kinematic")
        assert result["status"] == "success", result

        sim.step(200)  # carrier falls ~ to the ground
        assert _body_xpos(sim, "carrier")[2] < 0.3, "carrier should have fallen"
        off = _offset(sim, "carrier", "cube")
        for i in range(3):
            assert off[i] == pytest.approx(start_off[i], abs=0.01), f"cube did not follow: {off} vs {start_off}"

    def test_child_velocity_zeroed_while_carried(self, two_boxes):
        """No teleport fling: the carried child's freejoint velocity is zero."""
        sim = two_boxes
        assert sim.attach_bodies("carrier", "cube", mode="kinematic")["status"] == "success"
        sim.step(50)
        assert _freejoint_qvel(sim, "cube") == pytest.approx([0.0] * 6, abs=1e-12)

    def test_child_follows_through_send_action_path(self, sim):
        """The policy-driven substep loop (_apply_sim_action) also re-pins.

        A torque-actuated pusher scene drives via send_action; the attached
        cube must follow the pusher body, not stay behind.
        """
        r = sim.replace_scene_mjcf(
            "<mujoco><worldbody>"
            '<geom type="plane" size="2 2 0.1"/>'
            '<body name="pusher" pos="0 0 0.1">'
            '<joint name="slide_x" type="slide" axis="1 0 0"/>'
            '<geom type="box" size="0.05 0.05 0.05" mass="1"/>'
            "</body>"
            '<body name="cube" pos="0.2 0 0.05">'
            '<joint name="cube_joint" type="free"/>'
            '<geom type="box" size="0.02 0.02 0.02" mass="0.05"/>'
            "</body>"
            "</worldbody>"
            "<actuator>"
            '<velocity name="drive_x" joint="slide_x" kv="10"/>'
            "</actuator></mujoco>"
        )
        assert r["status"] == "success", r
        # Register a robot handle so send_action can route to the actuator.
        from strands_robots.simulation.models import SimRobot

        robot = SimRobot(name="pusher_bot", urdf_path="")
        robot.joint_names = ["slide_x"]
        robot.joint_ids = [mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_JOINT, "slide_x")]
        robot.actuator_ids = [0]
        sim._world.robots["pusher_bot"] = robot

        assert sim.attach_bodies("pusher", "cube", mode="kinematic")["status"] == "success"
        start_off = _offset(sim, "pusher", "cube")
        sim.send_action({"drive_x": 0.5}, robot_name="pusher_bot", n_substeps=400)
        assert _body_xpos(sim, "pusher")[0] > 0.05, "pusher should have moved"
        off = _offset(sim, "pusher", "cube")
        assert off[0] == pytest.approx(start_off[0], abs=0.01), f"cube did not follow the driven pusher: {off}"

    def test_detach_drops_child_physically(self, two_boxes):
        """After detach the cube is free again: it falls from the release point."""
        sim = two_boxes
        assert sim.attach_bodies("carrier", "cube", mode="kinematic")["status"] == "success"
        sim.step(20)
        assert sim.detach_bodies("carrier", "cube")["status"] == "success"
        z_release = _body_xpos(sim, "cube")[2]
        sim.step(100)
        assert _body_xpos(sim, "cube")[2] < z_release - 0.01, "detached cube should fall freely"

    def test_kinematic_requires_child_freejoint(self, sim):
        sim.add_object("carrier", shape="box", size=[0.1, 0.1, 0.1], position=[0, 0, 0.5])
        sim.add_object("fixture", shape="box", size=[0.05, 0.05, 0.05], position=[0.3, 0, 0.025], is_static=True)
        result = sim.attach_bodies("carrier", "fixture", mode="kinematic")
        assert result["status"] == "error"
        assert "freejoint" in result["content"][0]["text"]

    def test_stale_attachment_dropped_when_parent_body_vanishes(self, two_boxes, caplog):
        """A kinematic follow whose parent body disappears is dropped, not fatal.

        ``delete_body`` via ``patch_scene_mjcf`` can remove the parent out from
        under a live kinematic attachment (it does not go through the
        ``remove_object`` guard). The docstring contract: the next step drops
        the now-unresolvable entry with a warning so a stale name cannot
        silently teleport an unrelated joint - stepping must stay finite and
        both the kinematic-follow and the attachment registry must be cleared.
        """
        sim = two_boxes
        assert sim.attach_bodies("carrier", "cube", mode="kinematic")["status"] == "success"

        # Remove the parent body; the child's freejoint still exists, so the
        # entry becomes unresolvable on the parent side only.
        assert sim.patch_scene_mjcf([{"op": "delete_body", "name": "carrier"}])["status"] == "success"

        with caplog.at_level("WARNING", logger="strands_robots.simulation.mujoco.manipulation"):
            sim.step(3)  # exercises _apply_kinematic_attachments with a dead parent

        assert any("cube" in rec.message and "dropped" in rec.message for rec in caplog.records), (
            "the stale attachment must be dropped with a warning"
        )
        backend = sim._world._backend_state
        assert "cube" not in backend.get("kinematic_attachments", {}), "the kinematic-follow entry must be cleared"
        assert "cube" not in backend.get("attachments", {}), "the attachment registry entry must be cleared"
        assert bool(np.all(np.isfinite(sim._world._data.qacc))), "dropping the stale entry must leave state finite"

        # A second step is a clean no-op now that the registry is empty.
        sim.step(3)
        assert bool(np.all(np.isfinite(sim._world._data.qacc)))


class TestAttachErrorContract:
    def test_unknown_bodies_error(self, two_boxes):
        sim = two_boxes
        for parent, child, which in (("nope", "cube", "parent"), ("carrier", "nope", "child")):
            result = sim.attach_bodies(parent, child)
            assert result["status"] == "error"
            assert f"{which} body 'nope' not found" in result["content"][0]["text"]

    def test_unknown_mode_error(self, two_boxes):
        result = two_boxes.attach_bodies("carrier", "cube", mode="magnetic")
        assert result["status"] == "error"
        assert "unknown mode" in result["content"][0]["text"]

    def test_self_attach_error(self, two_boxes):
        result = two_boxes.attach_bodies("cube", "cube")
        assert result["status"] == "error"
        assert "same body" in result["content"][0]["text"]

    def test_bad_torquescale_error(self, two_boxes):
        for bad in (0.0, -1.0, float("nan"), "high"):
            result = two_boxes.attach_bodies("carrier", "cube", torquescale=bad)
            assert result["status"] == "error", f"torquescale={bad!r} must be rejected"
            assert "torquescale" in result["content"][0]["text"]

    def test_double_attach_error(self, two_boxes):
        sim = two_boxes
        assert sim.attach_bodies("carrier", "cube", mode="weld")["status"] == "success"
        result = sim.attach_bodies("carrier", "cube", mode="kinematic")
        assert result["status"] == "error"
        assert "already attached" in result["content"][0]["text"]

    def test_detach_unknown_attachment_error(self, two_boxes):
        result = two_boxes.detach_bodies("carrier", "cube")
        assert result["status"] == "error"
        assert "no attachment" in result["content"][0]["text"]

    def test_detach_wrong_parent_error(self, two_boxes):
        sim = two_boxes
        sim.add_object("other", shape="box", size=[0.05, 0.05, 0.05], position=[1, 1, 0.5])
        assert sim.attach_bodies("carrier", "cube", mode="kinematic")["status"] == "success"
        result = sim.detach_bodies("other", "cube")
        assert result["status"] == "error"
        assert "attached to 'carrier'" in result["content"][0]["text"]


class TestRemovalGuards:
    def test_remove_object_refuses_attached_child(self, two_boxes):
        sim = two_boxes
        assert sim.attach_bodies("carrier", "cube", mode="weld")["status"] == "success"
        for name in ("cube", "carrier"):
            result = sim.remove_object(name)
            assert result["status"] == "error", f"remove_object({name!r}) must refuse while attached"
            assert "detach_bodies" in result["content"][0]["text"]
        # After detaching, removal succeeds.
        assert sim.detach_bodies("carrier", "cube")["status"] == "success"
        assert sim.remove_object("cube")["status"] == "success"

    def test_remove_robot_refuses_attached_parent(self, sim, tmp_path):
        urdf = tmp_path / "mini.urdf"
        urdf.write_text(
            '<?xml version="1.0"?><robot name="mini">'
            '<link name="base_link"><inertial><mass value="1.0"/>'
            '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>'
            '<visual><geometry><box size="0.05 0.05 0.05"/></geometry></visual></link>'
            "</robot>"
        )
        assert sim.add_robot(name="arm", urdf_path=str(urdf))["status"] == "success"
        sim.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], position=[0.2, 0, 0.5])
        assert sim.attach_bodies("arm/base_link", "cube", mode="kinematic")["status"] == "success"
        result = sim.remove_robot("arm")
        assert result["status"] == "error"
        assert "detach_bodies" in result["content"][0]["text"]
        assert sim.detach_bodies("arm/base_link", "cube")["status"] == "success"
        assert sim.remove_robot("arm")["status"] == "success"


class TestDispatch:
    def test_attach_detach_dispatch_via_action_router(self, two_boxes):
        """The agent-facing dispatcher routes the new actions end to end."""
        sim = two_boxes
        result = sim._dispatch_action("attach_bodies", {"parent": "carrier", "child": "cube", "mode": "kinematic"})
        assert result["status"] == "success", result
        result = sim._dispatch_action("detach_bodies", {"parent": "carrier", "child": "cube"})
        assert result["status"] == "success", result
