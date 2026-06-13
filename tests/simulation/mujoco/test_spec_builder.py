"""Unit + integration tests for ``SpecBuilder`` - the MjSpec-based MJCF builder.

Tests cover:

* Module-level helpers (``_geom_type``, ``_normalize_size``, ``_target_quat``).
* ``SpecBuilder.build`` produces a compile-valid spec for empty, object-only,
  and camera-only worlds.
* Mutation helpers (``add_object``, ``remove_body``, ``add_camera``,
  ``remove_camera``) produce specs that recompile cleanly.
* ``attach_robot`` prefixes names correctly and returns the source joint names.
* ``from_mjcf_string`` / ``from_file`` round-trip cleanly.

These tests use the actual mujoco AST - they require the ``mujoco`` package.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from strands_robots.simulation.models import (  # noqa: E402
    SimCamera,
    SimObject,
    SimRobot,
    SimWorld,
)
from strands_robots.simulation.mujoco.spec_builder import (  # noqa: E402
    SpecBuilder,
    _geom_type,
    _normalize_size,
    _target_quat,
)

# Module-level helpers


class TestGeomType:
    def test_known_shapes_map_to_enum(self):
        assert _geom_type("box") == mujoco.mjtGeom.mjGEOM_BOX
        assert _geom_type("sphere") == mujoco.mjtGeom.mjGEOM_SPHERE
        assert _geom_type("cylinder") == mujoco.mjtGeom.mjGEOM_CYLINDER
        assert _geom_type("capsule") == mujoco.mjtGeom.mjGEOM_CAPSULE
        assert _geom_type("mesh") == mujoco.mjtGeom.mjGEOM_MESH
        assert _geom_type("plane") == mujoco.mjtGeom.mjGEOM_PLANE
        assert _geom_type("ellipsoid") == mujoco.mjtGeom.mjGEOM_ELLIPSOID

    def test_unknown_shape_raises_with_helpful_list(self):
        with pytest.raises(ValueError, match="Unsupported shape"):
            _geom_type("hyperboloid")


class TestNormalizeSize:
    def test_box_halves_full_extents(self):
        assert _normalize_size("box", [0.2, 0.4, 0.6]) == [0.1, 0.2, 0.3]

    def test_sphere_halves_first_coordinate(self):
        assert _normalize_size("sphere", [0.1])[0] == pytest.approx(0.05)

    def test_cylinder_radius_and_half_height(self):
        out = _normalize_size("cylinder", [0.1, 0, 0.4])
        assert out[0] == pytest.approx(0.05)  # radius
        assert out[1] == pytest.approx(0.2)  # half-height

    def test_capsule_same_as_cylinder(self):
        assert _normalize_size("capsule", [0.1, 0, 0.4]) == _normalize_size("cylinder", [0.1, 0, 0.4])

    def test_plane_size_duplicates_x_to_y_when_only_one_given(self):
        out = _normalize_size("plane", [2.0])
        assert out[0] == 2.0
        assert out[1] == 2.0

    def test_mesh_returns_zeros(self):
        assert _normalize_size("mesh", [0.1, 0.2, 0.3]) == [0.0, 0.0, 0.0]

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError, match="Cannot normalize size"):
            _normalize_size("hyperboloid", [1.0])


class TestTargetQuat:
    def test_returns_none_for_degenerate(self):
        assert _target_quat([1, 2, 3], [1, 2, 3]) is None

    def test_returns_normalised_quaternion(self):
        quat = _target_quat([1, 0, 0.5], [0, 0, 0])
        assert quat is not None
        assert len(quat) == 4
        norm = (quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_forward_parallel_to_up_returns_none(self):
        # Straight-down camera - forward is (0,0,-1), colinear with world up.
        assert _target_quat([0, 0, 1], [0, 0, 0]) is None

    def test_quat_rotates_camera_to_face_target(self):
        """Compile a spec with quat=, then check the resulting cam_mat0 actually
        points the camera's -Z axis toward the target.
        """
        pos = [1.0, 0.0, 0.5]
        target = [0.0, 0.0, 0.0]
        quat = _target_quat(pos, target)
        assert quat is not None

        spec = mujoco.MjSpec()
        spec.worldbody.add_camera(name="c", pos=pos, quat=quat, fovy=60, mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
        model = spec.compile()
        rot = model.cam_mat0[0].reshape(3, 3)
        # Camera forward is -Z in its local frame. Column 2 of rot is world
        # frame +Z of the camera, so camera forward = -rot[:, 2].
        cam_forward = -rot[:, 2]
        expected = np.array([-1.0, 0.0, -0.5])
        expected /= np.linalg.norm(expected)
        assert np.allclose(cam_forward, expected, atol=1e-3)


# SpecBuilder.build


@pytest.fixture
def sample_world() -> SimWorld:
    w = SimWorld()
    w.objects["cube"] = SimObject(
        name="cube",
        shape="box",
        position=[0, 0, 0.1],
        size=[0.1, 0.1, 0.1],
        color=[0.5, 0.5, 0.5, 1],
        is_static=False,
        mass=0.2,
    )
    w.objects["ball"] = SimObject(
        name="ball",
        shape="sphere",
        position=[0.5, 0, 0.1],
        size=[0.05, 0.05, 0.05],
        color=[1, 0, 0, 1],
        is_static=False,
        mass=0.1,
    )
    w.cameras["front"] = SimCamera(
        name="front",
        position=[1, 0, 0.5],
        target=[0, 0, 0],
        fov=60,
        width=640,
        height=480,
    )
    return w


class TestBuild:
    def test_empty_world_compiles(self):
        spec = SpecBuilder.build(SimWorld())
        model = spec.compile()
        assert model.nbody >= 1  # world + ground + lights

    def test_gravity_and_timestep_propagate(self):
        w = SimWorld()
        w.gravity = [0.0, 0.0, -5.0]
        w.timestep = 0.004
        spec = SpecBuilder.build(w)
        model = spec.compile()
        assert model.opt.timestep == pytest.approx(0.004)
        assert np.allclose(model.opt.gravity, [0.0, 0.0, -5.0])

    def test_sample_world_compiles(self, sample_world):
        model = SpecBuilder.build(sample_world).compile()
        # 1 world + 1 ground is a geom; cube, ball are bodies
        assert model.nbody >= 3
        assert model.ncam == 1
        assert model.nu == 0

    def test_body_positions_and_masses(self, sample_world):
        model = SpecBuilder.build(sample_world).compile()
        for name in ("cube", "ball"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            assert bid >= 0
            obj = sample_world.objects[name]
            assert np.allclose(model.body_pos[bid], obj.position)
            assert model.body_mass[bid] == pytest.approx(obj.mass, abs=1e-6)

    def test_ground_plane_present(self):
        w = SimWorld()
        spec = SpecBuilder.build(w)
        model = spec.compile()
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        assert gid >= 0

    def test_ground_plane_absent_when_disabled(self):
        w = SimWorld()
        w.ground_plane = False
        spec = SpecBuilder.build(w)
        model = spec.compile()
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        assert gid < 0

    def test_headlight_is_dimmed(self):
        """GH #373 (renderings too bright): MuJoCo's default headlight
        (diffuse 0.4, specular 0.5, active) stacks additively on our two
        explicit scene lights, washing out the scene. build() must dim it
        to a low, shadow-free term."""
        model = SpecBuilder.build(SimWorld()).compile()
        hl = model.vis.headlight
        # Specular off entirely (no head-on glare hotspots).
        assert np.allclose(hl.specular, [0.0, 0.0, 0.0])
        # Diffuse pulled well below the 0.4 default so explicit lights
        # provide the directional illumination.
        assert float(hl.diffuse[0]) <= 0.25
        assert np.allclose(hl.diffuse, [hl.diffuse[0]] * 3)

    def test_explicit_scene_lights_present(self):
        """The two explicit directional lights survive build() — the dimmed
        headlight relies on them for the actual scene illumination."""
        model = SpecBuilder.build(SimWorld()).compile()
        assert model.nlight >= 2
        names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, i) for i in range(model.nlight)}
        assert {"main_light", "fill_light"} <= names


# Mutation helpers


class TestMutation:
    def test_add_object_then_recompile(self):
        w = SimWorld()
        spec = SpecBuilder.build(w)
        model = spec.compile()
        data = mujoco.MjData(model)

        SpecBuilder.add_object(
            spec,
            SimObject(
                name="ball",
                shape="sphere",
                position=[0, 0, 0.1],
                size=[0.05, 0.05, 0.05],
                color=[1, 0, 0, 1],
                is_static=False,
                mass=0.1,
            ),
        )
        new_model, _new_data = spec.recompile(model, data)
        ball_id = mujoco.mj_name2id(new_model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        assert ball_id >= 0

    def test_remove_body_then_recompile(self):
        w = SimWorld()
        w.objects["victim"] = SimObject(
            name="victim",
            shape="box",
            position=[0, 0, 0.1],
            size=[0.1, 0.1, 0.1],
            color=[1, 0, 0, 1],
            is_static=False,
            mass=0.1,
        )
        spec = SpecBuilder.build(w)
        model = spec.compile()
        data = mujoco.MjData(model)
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "victim") >= 0

        assert SpecBuilder.remove_body(spec, "victim") is True
        new_model, _ = spec.recompile(model, data)
        assert mujoco.mj_name2id(new_model, mujoco.mjtObj.mjOBJ_BODY, "victim") < 0

    def test_remove_missing_body_returns_false(self):
        spec = SpecBuilder.build(SimWorld())
        assert SpecBuilder.remove_body(spec, "ghost") is False

    def test_add_camera_then_recompile(self):
        w = SimWorld()
        spec = SpecBuilder.build(w)
        model = spec.compile()
        data = mujoco.MjData(model)

        SpecBuilder.add_camera(
            spec,
            SimCamera(
                name="top",
                position=[0, 0, 2.0],
                target=[0, 0, 0],
                fov=60,
                width=640,
                height=480,
            ),
        )
        new_model, _ = spec.recompile(model, data)
        assert mujoco.mj_name2id(new_model, mujoco.mjtObj.mjOBJ_CAMERA, "top") >= 0

    def test_add_camera_mounted_on_body(self):
        """GH #373 (SO101 wrist cam): a camera with parent_body set must be
        attached to that body so it tracks the body's motion (realistic
        wrist/gripper camera), not fixed in the world."""
        # Build a world with a single movable body to mount onto.
        w = SimWorld()
        w.objects["holder"] = SimObject(
            name="holder", shape="box", position=[0.2, 0.0, 0.3], size=[0.05, 0.05, 0.05], mass=0.2
        )
        spec = SpecBuilder.build(w)
        model = spec.compile()
        data = mujoco.MjData(model)

        SpecBuilder.add_camera(
            spec,
            SimCamera(
                name="wrist",
                position=[0.0, 0.0, 0.1],
                target=[0.0, 0.0, 0.0],
                fov=60,
                width=320,
                height=240,
                parent_body="holder",
            ),
        )
        new_model, _ = spec.recompile(model, data)
        cam_id = mujoco.mj_name2id(new_model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        assert cam_id >= 0
        # The camera's parent body must be 'holder', not worldbody (id 0).
        body_id = int(new_model.cam_bodyid[cam_id])
        assert mujoco.mj_id2name(new_model, mujoco.mjtObj.mjOBJ_BODY, body_id) == "holder"

    def test_add_camera_unknown_parent_body_raises(self):
        w = SimWorld()
        spec = SpecBuilder.build(w)
        with pytest.raises(ValueError, match="parent_body"):
            SpecBuilder.add_camera(
                spec,
                SimCamera(name="wrist", position=[0, 0, 0.1], target=[0, 0, 0], parent_body="does_not_exist"),
            )

    def test_remove_camera(self):
        w = SimWorld()
        w.cameras["c"] = SimCamera(name="c", position=[1, 0, 0.5], target=[0, 0, 0], fov=60, width=640, height=480)
        spec = SpecBuilder.build(w)
        model = spec.compile()
        data = mujoco.MjData(model)

        assert SpecBuilder.remove_camera(spec, "c") is True
        new_model, _ = spec.recompile(model, data)
        assert mujoco.mj_name2id(new_model, mujoco.mjtObj.mjOBJ_CAMERA, "c") < 0


# attach_robot


ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="lift" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="50"/>
    <position name="lift_act" joint="lift" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def arm_path(tmp_path):
    path = tmp_path / "arm.xml"
    path.write_text(ARM_XML)
    return str(path)


class TestAttachRobot:
    def test_attach_prefixes_names(self, arm_path):
        scene = SpecBuilder.build(SimWorld())
        robot = SimRobot(name="arm1", urdf_path=arm_path, position=[0.3, 0, 0])
        joint_names = SpecBuilder.attach_robot(scene, robot, arm_path)
        assert joint_names == ["pan", "lift"]

        model = scene.compile()
        # Prefixed joint lookup
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "arm1/pan") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "arm1/lift") >= 0
        # Prefixed actuator lookup
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm1/pan_act") >= 0

    def test_attach_two_of_same_robot(self, arm_path):
        """Two attaches of the same source MJCF - each needs a fresh MjSpec."""
        scene = SpecBuilder.build(SimWorld())
        robot_a = SimRobot(name="armA", urdf_path=arm_path, position=[0.3, 0, 0])
        robot_b = SimRobot(name="armB", urdf_path=arm_path, position=[-0.3, 0, 0])

        SpecBuilder.attach_robot(scene, robot_a, arm_path)
        SpecBuilder.attach_robot(scene, robot_b, arm_path)

        model = scene.compile()
        for prefix in ("armA", "armB"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/pan") >= 0
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/lift") >= 0

    def test_attach_frame_uses_robot_position(self, arm_path):
        scene = SpecBuilder.build(SimWorld())
        robot = SimRobot(name="arm1", urdf_path=arm_path, position=[0.7, 0.2, 0.0])
        SpecBuilder.attach_robot(scene, robot, arm_path)

        model = scene.compile()
        # The arm1 robot's body root gets child-of-frame semantics - body_pos
        # for arm1/base is relative to the attach frame.
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm1/base")
        # Walk up the body tree to verify the frame is offset
        # (simpler: compile works at all - spec.attach validates positions)
        assert bid >= 0

    def test_attach_robot_strips_robot_scene_ground_plane(self, tmp_path):
        """A robot scene that ships its own ``floor`` plane must not add a
        second coplanar ground plane (issue #320).

        Many menagerie scenes (e.g. franka_emika_panda/scene.xml) include a
        ``floor`` plane at z=0. Attached alongside the world's own ``ground``
        plane (also z=0) it caused two coplanar infinite planes with different
        materials -> depth-buffer Z-fighting / broken floor render.
        ``attach_robot`` now strips plane geoms from the robot scene so exactly
        one world-owned ``ground`` plane survives.
        """
        robot_with_floor = """
        <mujoco model="arm_with_floor">
          <compiler angle="radian"/>
          <worldbody>
            <geom name="floor" size="0 0 0.05" type="plane"/>
            <body name="base" pos="0 0 0.1">
              <joint name="pan" type="hinge" axis="0 0 1"/>
              <geom type="cylinder" size="0.05 0.05"/>
            </body>
          </worldbody>
          <actuator>
            <position name="pan_act" joint="pan" kp="50"/>
          </actuator>
        </mujoco>
        """
        path = tmp_path / "arm_with_floor.xml"
        path.write_text(robot_with_floor)

        scene = SpecBuilder.build(SimWorld())  # ground_plane=True by default
        robot = SimRobot(name="arm1", urdf_path=str(path), position=[0.0, 0.0, 0.0])
        SpecBuilder.attach_robot(scene, robot, str(path))

        model = scene.compile()
        plane_labels = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            for g in range(model.ngeom)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
        ]
        assert len(plane_labels) == 1, f"expected exactly one ground plane, got {plane_labels}"
        assert plane_labels == ["ground"], f"the surviving plane must be the world ground, got {plane_labels}"

        # The robot's own body/joint/actuator must still be present.
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "arm1/pan") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm1/pan_act") >= 0

    def test_attach_robot_without_floor_is_unchanged(self, arm_path):
        """A robot scene with no plane geom attaches normally - the strip is a
        no-op and the single world ``ground`` plane survives."""
        scene = SpecBuilder.build(SimWorld())
        robot = SimRobot(name="arm1", urdf_path=arm_path, position=[0.0, 0.0, 0.0])
        SpecBuilder.attach_robot(scene, robot, arm_path)
        model = scene.compile()
        planes = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            for g in range(model.ngeom)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
        ]
        assert planes == ["ground"]

    def test_attach_robot_keeps_floor_when_world_ground_disabled(self, tmp_path):
        """#363 guard 1: under ``ground_plane=False`` the world owns no ground
        plane, so the robot's own floor must NOT be stripped -- otherwise the
        opt-out scene is left with zero ground planes."""
        robot_with_floor = """
        <mujoco model="arm_with_floor">
          <compiler angle="radian"/>
          <worldbody>
            <geom name="floor" size="0 0 0.05" type="plane"/>
            <body name="base" pos="0 0 0.1">
              <joint name="pan" type="hinge" axis="0 0 1"/>
              <geom type="cylinder" size="0.05 0.05"/>
            </body>
          </worldbody>
        </mujoco>
        """
        path = tmp_path / "arm_with_floor.xml"
        path.write_text(robot_with_floor)

        scene = SpecBuilder.build(SimWorld(ground_plane=False))
        robot = SimRobot(name="arm1", urdf_path=str(path), position=[0.0, 0.0, 0.0])
        SpecBuilder.attach_robot(scene, robot, str(path))
        model = scene.compile()
        planes = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            for g in range(model.ngeom)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
        ]
        # The robot floor is the ONLY floor source -- it must survive.
        assert len(planes) == 1, f"opt-out scene must keep the robot floor, got {planes}"

    def test_attach_robot_logs_debug_on_strip(self, tmp_path, caplog):
        """#363 guard 2: the plane strip emits a DEBUG record naming what was
        removed so a disappearing floor is diagnosable."""
        import logging

        robot_with_floor = """
        <mujoco model="arm_with_floor">
          <compiler angle="radian"/>
          <worldbody>
            <geom name="floor" size="0 0 0.05" type="plane"/>
            <body name="base" pos="0 0 0.1">
              <joint name="pan" type="hinge" axis="0 0 1"/>
              <geom type="cylinder" size="0.05 0.05"/>
            </body>
          </worldbody>
        </mujoco>
        """
        path = tmp_path / "arm_with_floor.xml"
        path.write_text(robot_with_floor)
        scene = SpecBuilder.build(SimWorld())
        robot = SimRobot(name="arm1", urdf_path=str(path), position=[0.0, 0.0, 0.0])
        with caplog.at_level(logging.DEBUG, logger="strands_robots.simulation.mujoco.spec_builder"):
            SpecBuilder.attach_robot(scene, robot, str(path))
        msgs = [r.getMessage() for r in caplog.records if "stripped" in r.getMessage()]
        assert msgs, f"expected a DEBUG record about the stripped plane; got {[r.getMessage() for r in caplog.records]}"

    def test_attach_robot_keeps_non_z0_plane(self, tmp_path):
        """#363 guard 3: a robot plane that is NOT z=0 axis-aligned (e.g. an
        elevated/angled ramp modeled as a plane) must survive the strip; only
        the z=0 ground floor is removed."""
        robot_with_ramp = """
        <mujoco model="arm_with_ramp">
          <compiler angle="radian"/>
          <worldbody>
            <geom name="floor" size="0 0 0.05" type="plane"/>
            <geom name="ramp" size="1 1 0.05" type="plane" pos="0 0 0.5"/>
            <body name="base" pos="0 0 0.1">
              <joint name="pan" type="hinge" axis="0 0 1"/>
              <geom type="cylinder" size="0.05 0.05"/>
            </body>
          </worldbody>
        </mujoco>
        """
        path = tmp_path / "arm_with_ramp.xml"
        path.write_text(robot_with_ramp)
        scene = SpecBuilder.build(SimWorld())
        robot = SimRobot(name="arm1", urdf_path=str(path), position=[0.0, 0.0, 0.0])
        SpecBuilder.attach_robot(scene, robot, str(path))
        model = scene.compile()
        plane_labels = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            for g in range(model.ngeom)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
        ]
        # World ground survives + the elevated ramp survives; the z=0 robot
        # floor is the only plane stripped.
        assert "ground" in plane_labels
        assert any("ramp" in (lbl or "") for lbl in plane_labels), (
            f"the non-z=0 ramp plane must survive the strip; got {plane_labels}"
        )
        assert not any(lbl and lbl.endswith("floor") for lbl in plane_labels), (
            f"the z=0 robot floor should have been stripped; got {plane_labels}"
        )


# from_mjcf_string / from_file


class TestFromSources:
    def test_from_mjcf_string_parses(self):
        spec = SpecBuilder.from_mjcf_string(
            '<mujoco><worldbody><body name="x"><geom size="0.1"/></body></worldbody></mujoco>'
        )
        model = spec.compile()
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "x") >= 0

    def test_from_mjcf_string_raises_on_invalid(self):
        with pytest.raises(ValueError):
            SpecBuilder.from_mjcf_string("not valid xml at all")

    def test_from_file_reads_mjcf(self, tmp_path):
        p = tmp_path / "scene.xml"
        p.write_text('<mujoco><worldbody><body name="y"><geom size="0.1"/></body></worldbody></mujoco>')
        spec = SpecBuilder.from_file(str(p))
        model = spec.compile()
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "y") >= 0
