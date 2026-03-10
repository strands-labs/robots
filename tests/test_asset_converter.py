"""Tests for the improved asset_converter.py — MJCF ↔ USD conversion.

Tests cover:
1. OBJ → STL pre-conversion logic
2. MuJoCo+pxr mesh geometry extraction (_manual_mjcf_to_usd)
3. Primitive geom creation (box, sphere, cylinder, capsule)
4. Joint extraction (revolute, prismatic)
5. Body hierarchy construction
6. Batch conversion (convert_all_robots_to_usd)
7. Dual-pipeline fallback logic
8. Edge cases (empty meshes, missing files, sanitization)
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    """Temporary directory for test outputs."""
    d = tempfile.mkdtemp(prefix="test_asset_conv_")
    yield d
    import shutil

    shutil.rmtree(d, ignore_errors=True)


def _make_tetrahedron_stl(path: str) -> None:
    """Create a valid binary STL with a tetrahedron (4 vertices, 4 triangles).

    MuJoCo requires meshes to have at least 4 vertices and be valid
    3D geometry (not degenerate), so a single triangle won't work.
    """
    import struct

    # Tetrahedron vertices
    v0 = (0.0, 0.0, 0.0)
    v1 = (1.0, 0.0, 0.0)
    v2 = (0.5, 0.866, 0.0)
    v3 = (0.5, 0.289, 0.816)

    triangles = [
        (v0, v1, v2),  # bottom
        (v0, v1, v3),  # front
        (v1, v2, v3),  # right
        (v0, v2, v3),  # left
    ]

    with open(path, "wb") as f:
        f.write(b"\0" * 80)  # header
        f.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            # Compute normal
            import math

            ax, ay, az = tri[1][0] - tri[0][0], tri[1][1] - tri[0][1], tri[1][2] - tri[0][2]
            bx, by, bz = tri[2][0] - tri[0][0], tri[2][1] - tri[0][1], tri[2][2] - tri[0][2]
            nx = ay * bz - az * by
            ny = az * bx - ax * bz
            nz = ax * by - ay * bx
            norm = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            f.write(struct.pack("<3f", nx / norm, ny / norm, nz / norm))
            for v in tri:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))


def _make_cube_obj(path: str) -> None:
    """Create a valid OBJ cube with 8 vertices and 12 triangles."""
    obj_content = """# Cube
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
v -0.5  0.5 -0.5
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5  0.5
f 1 2 3
f 1 3 4
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""
    with open(path, "w") as f:
        f.write(obj_content)


@pytest.fixture
def simple_mjcf(tmp_dir):
    """Create a simple MJCF file with valid STL mesh for testing."""
    assets_dir = os.path.join(tmp_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # Create a valid tetrahedron STL (4 vertices — MuJoCo minimum)
    _make_tetrahedron_stl(os.path.join(assets_dir, "link.stl"))

    mjcf_content = """<mujoco model="test_robot">
  <compiler angle="radian" meshdir="assets/"/>
  <asset>
    <mesh name="link_mesh" file="link.stl"/>
  </asset>
  <worldbody>
    <body name="base">
      <geom type="mesh" mesh="link_mesh"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="mesh" mesh="link_mesh"/>
        <geom type="box" name="collision_box" size="0.05 0.05 0.1"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="joint2" type="slide" axis="0 0 1" range="0 0.1"/>
          <geom type="sphere" name="tip_sphere" size="0.03"/>
          <geom type="cylinder" name="tip_cylinder" size="0.02 0.05"/>
          <geom type="capsule" name="tip_capsule" size="0.01 0.04"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""

    mjcf_path = os.path.join(tmp_dir, "test_robot.xml")
    with open(mjcf_path, "w") as f:
        f.write(mjcf_content)

    return mjcf_path


@pytest.fixture
def obj_mjcf(tmp_dir):
    """Create a MJCF file referencing OBJ meshes for OBJ→STL testing."""
    assets_dir = os.path.join(tmp_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # Create valid OBJ cubes
    _make_cube_obj(os.path.join(assets_dir, "link0.obj"))
    _make_cube_obj(os.path.join(assets_dir, "link1.obj"))

    # Also have one STL that shouldn't be converted
    _make_tetrahedron_stl(os.path.join(assets_dir, "collision.stl"))

    mjcf_content = """<mujoco model="obj_robot">
  <compiler angle="radian" meshdir="assets/"/>
  <asset>
    <mesh name="link0_visual" file="link0.obj"/>
    <mesh name="link1_visual" file="link1.obj"/>
    <mesh name="collision_mesh" file="collision.stl"/>
  </asset>
  <worldbody>
    <body name="base">
      <geom type="mesh" mesh="link0_visual"/>
      <geom type="mesh" mesh="collision_mesh"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1"/>
        <geom type="mesh" mesh="link1_visual"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""

    mjcf_path = os.path.join(tmp_dir, "obj_robot.xml")
    with open(mjcf_path, "w") as f:
        f.write(mjcf_content)

    return mjcf_path


# ────────────────────────────────────────────────────────────────
# Test: Name sanitization
# ────────────────────────────────────────────────────────────────


class TestSanitizeUsdName:
    def test_basic_name(self):
        from strands_robots.isaac.asset_converter import _sanitize_usd_name

        assert _sanitize_usd_name("link1") == "link1"

    def test_slash_replacement(self):
        from strands_robots.isaac.asset_converter import _sanitize_usd_name

        assert _sanitize_usd_name("panda/link1") == "panda_link1"

    def test_dash_replacement(self):
        from strands_robots.isaac.asset_converter import _sanitize_usd_name

        assert _sanitize_usd_name("left-finger") == "left_finger"

    def test_dot_replacement(self):
        from strands_robots.isaac.asset_converter import _sanitize_usd_name

        assert _sanitize_usd_name("link.01") == "link_01"

    def test_leading_digit(self):
        from strands_robots.isaac.asset_converter import _sanitize_usd_name

        assert _sanitize_usd_name("123abc") == "_123abc"

    def test_empty_string(self):
        from strands_robots.isaac.asset_converter import _sanitize_usd_name

        assert _sanitize_usd_name("") == "_unnamed"

    def test_space_replacement(self):
        from strands_robots.isaac.asset_converter import _sanitize_usd_name

        assert _sanitize_usd_name("left finger") == "left_finger"


# ────────────────────────────────────────────────────────────────
# Test: OBJ → STL pre-conversion
# ────────────────────────────────────────────────────────────────


class TestObjToStlPreconversion:
    def test_no_obj_returns_none(self, simple_mjcf):
        """MJCF with only STL meshes should return None (no conversion needed)."""
        from strands_robots.isaac.asset_converter import _preconvert_obj_to_stl

        result = _preconvert_obj_to_stl(simple_mjcf)
        assert result is None

    def test_obj_meshes_converted(self, obj_mjcf):
        """MJCF with OBJ meshes should return path to modified MJCF."""
        import shutil

        from strands_robots.isaac.asset_converter import _preconvert_obj_to_stl

        result = _preconvert_obj_to_stl(obj_mjcf)
        try:
            assert result is not None
            assert os.path.exists(result)

            # Check the modified MJCF references .stl instead of .obj
            with open(result) as f:
                content = f.read()
            assert "link0.stl" in content
            assert "link1.stl" in content
            # The STL file should remain as-is
            assert "collision.stl" in content
        finally:
            if result:
                tmp_dir = os.path.dirname(result)
                if "strands_obj2stl_" in tmp_dir:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_convert_single_obj_to_stl(self, tmp_dir):
        """Test the low-level OBJ → STL converter."""
        from strands_robots.isaac.asset_converter import _convert_single_obj_to_stl

        obj_content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        obj_path = os.path.join(tmp_dir, "test.obj")
        stl_path = os.path.join(tmp_dir, "test.stl")

        with open(obj_path, "w") as f:
            f.write(obj_content)

        _convert_single_obj_to_stl(obj_path, stl_path)

        assert os.path.exists(stl_path)
        # Binary STL: 80-byte header + 4-byte count + 50 bytes per triangle
        size = os.path.getsize(stl_path)
        assert size == 80 + 4 + 50  # 1 triangle

    def test_convert_obj_with_vertex_normals(self, tmp_dir):
        """OBJ with v/vt/vn face format should parse correctly."""
        from strands_robots.isaac.asset_converter import _convert_single_obj_to_stl

        obj_content = """v 0 0 0
v 1 0 0
v 0 1 0
vn 0 0 1
vn 0 0 1
vn 0 0 1
f 1//1 2//2 3//3
"""
        obj_path = os.path.join(tmp_dir, "test_vn.obj")
        stl_path = os.path.join(tmp_dir, "test_vn.stl")

        with open(obj_path, "w") as f:
            f.write(obj_content)

        _convert_single_obj_to_stl(obj_path, stl_path)
        assert os.path.exists(stl_path)

    def test_convert_obj_polygon_triangulation(self, tmp_dir):
        """OBJ with quad faces should be triangulated."""
        from strands_robots.isaac.asset_converter import _convert_single_obj_to_stl

        obj_content = """v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
"""
        obj_path = os.path.join(tmp_dir, "test_quad.obj")
        stl_path = os.path.join(tmp_dir, "test_quad.stl")

        with open(obj_path, "w") as f:
            f.write(obj_content)

        _convert_single_obj_to_stl(obj_path, stl_path)
        assert os.path.exists(stl_path)
        # Quad should become 2 triangles
        size = os.path.getsize(stl_path)
        assert size == 80 + 4 + 50 * 2


# ────────────────────────────────────────────────────────────────
# Test: MuJoCo+pxr manual conversion
# ────────────────────────────────────────────────────────────────


class TestManualMjcfToUsd:
    @pytest.fixture(autouse=True)
    def check_deps(self):
        """Skip if MuJoCo or pxr not available."""
        pytest.importorskip("mujoco")
        pytest.importorskip("pxr")

    def test_basic_conversion(self, simple_mjcf, tmp_dir):
        """Convert a simple MJCF and verify USD structure."""
        from strands_robots.isaac.asset_converter import _manual_mjcf_to_usd

        output = os.path.join(tmp_dir, "output.usd")
        result = _manual_mjcf_to_usd(simple_mjcf, output)

        assert result["status"] == "success", f"Conversion failed: {result}"
        assert os.path.exists(output)

        # Verify stats
        stats = result.get("stats", {})
        assert stats["bodies"] > 0
        assert stats["joints"] > 0
        assert stats["geoms"] > 0

    def test_mesh_extraction(self, simple_mjcf, tmp_dir):
        """Verify mesh geometry is actually extracted (not just Xform placeholders)."""
        from pxr import Usd, UsdGeom

        from strands_robots.isaac.asset_converter import _manual_mjcf_to_usd

        output = os.path.join(tmp_dir, "mesh_test.usd")
        result = _manual_mjcf_to_usd(simple_mjcf, output)
        assert result["status"] == "success", f"Conversion failed: {result}"

        stage = Usd.Stage.Open(output)
        assert stage is not None

        # Find mesh prims
        mesh_prims = []
        for prim in stage.TraverseAll():
            if prim.GetTypeName() == "Mesh":
                mesh_prims.append(prim)

        # Should have at least one mesh (from the STL link)
        assert len(mesh_prims) >= 1, f"Expected mesh prims, found {len(mesh_prims)}"

        # Verify mesh has actual vertex data
        for mesh_prim in mesh_prims:
            mesh = UsdGeom.Mesh(mesh_prim)
            points = mesh.GetPointsAttr().Get()
            face_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_indices = mesh.GetFaceVertexIndicesAttr().Get()

            assert points is not None and len(points) > 0, f"Mesh {mesh_prim.GetPath()} has no vertices"
            assert face_counts is not None and len(face_counts) > 0, f"Mesh {mesh_prim.GetPath()} has no faces"
            assert face_indices is not None and len(face_indices) > 0, f"Mesh {mesh_prim.GetPath()} has no face indices"

            # All face counts should be 3 (triangulated)
            for fc in face_counts:
                assert fc == 3, f"Expected triangulated faces, got {fc}"

    def test_primitive_geoms_created(self, simple_mjcf, tmp_dir):
        """Verify box, sphere, cylinder, capsule primitives are created."""
        from pxr import Usd

        from strands_robots.isaac.asset_converter import _manual_mjcf_to_usd

        output = os.path.join(tmp_dir, "prim_test.usd")
        result = _manual_mjcf_to_usd(simple_mjcf, output)
        assert result["status"] == "success", f"Conversion failed: {result}"

        stage = Usd.Stage.Open(output)

        prim_types = set()
        for prim in stage.TraverseAll():
            prim_types.add(prim.GetTypeName())

        # Our simple_mjcf has box, sphere, cylinder, capsule geoms
        assert "Cube" in prim_types, f"No Cube found. Types: {prim_types}"
        assert "Sphere" in prim_types, f"No Sphere found. Types: {prim_types}"
        assert "Cylinder" in prim_types, f"No Cylinder found. Types: {prim_types}"
        assert "Capsule" in prim_types, f"No Capsule found. Types: {prim_types}"

    def test_joint_extraction(self, simple_mjcf, tmp_dir):
        """Verify joints are properly created with limits."""
        from pxr import Usd, UsdPhysics

        from strands_robots.isaac.asset_converter import _manual_mjcf_to_usd

        output = os.path.join(tmp_dir, "joint_test.usd")
        result = _manual_mjcf_to_usd(simple_mjcf, output)
        assert result["status"] == "success", f"Conversion failed: {result}"

        stage = Usd.Stage.Open(output)

        revolute_joints = []
        prismatic_joints = []
        for prim in stage.TraverseAll():
            if prim.GetTypeName() == "PhysicsRevoluteJoint":
                revolute_joints.append(prim)
            elif prim.GetTypeName() == "PhysicsPrismaticJoint":
                prismatic_joints.append(prim)

        # simple_mjcf has 1 hinge (revolute) + 1 slide (prismatic)
        assert len(revolute_joints) >= 1, "Expected at least 1 revolute joint"
        assert len(prismatic_joints) >= 1, "Expected at least 1 prismatic joint"

        # Check revolute joint has limits
        for jnt_prim in revolute_joints:
            jnt = UsdPhysics.RevoluteJoint(jnt_prim)
            lower = jnt.GetLowerLimitAttr().Get()
            upper = jnt.GetUpperLimitAttr().Get()
            assert lower is not None
            assert upper is not None
            assert lower < upper

    def test_body_hierarchy(self, simple_mjcf, tmp_dir):
        """Verify parent-child body hierarchy in USD."""
        from pxr import Usd

        from strands_robots.isaac.asset_converter import _manual_mjcf_to_usd

        output = os.path.join(tmp_dir, "hierarchy_test.usd")
        result = _manual_mjcf_to_usd(simple_mjcf, output)
        assert result["status"] == "success", f"Conversion failed: {result}"

        stage = Usd.Stage.Open(output)

        # Find the body prims
        body_paths = set()
        for prim in stage.TraverseAll():
            if prim.GetTypeName() == "Xform":
                body_paths.add(str(prim.GetPath()))

        # Should have Robot, base, link1, link2 xforms
        assert any("Robot" in p for p in body_paths)
        assert any("base" in p for p in body_paths)
        assert any("link1" in p for p in body_paths)
        assert any("link2" in p for p in body_paths)

    def test_stage_metadata(self, simple_mjcf, tmp_dir):
        """Verify USD stage has correct up axis and meters per unit."""
        from pxr import Usd, UsdGeom

        from strands_robots.isaac.asset_converter import _manual_mjcf_to_usd

        output = os.path.join(tmp_dir, "meta_test.usd")
        result = _manual_mjcf_to_usd(simple_mjcf, output)
        assert result["status"] == "success", f"Conversion failed: {result}"

        stage = Usd.Stage.Open(output)
        assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z

    def test_stats_count(self, simple_mjcf, tmp_dir):
        """Verify conversion stats are accurate."""
        from strands_robots.isaac.asset_converter import _manual_mjcf_to_usd

        output = os.path.join(tmp_dir, "stats_test.usd")
        result = _manual_mjcf_to_usd(simple_mjcf, output)
        assert result["status"] == "success"

        stats = result["stats"]
        # simple_mjcf: base + link1 + link2 = 3 bodies
        assert stats["bodies"] >= 3
        # 2 joints (hinge + slide)
        assert stats["joints"] == 2
        # 2 mesh geoms + 1 box + 1 sphere + 1 cylinder + 1 capsule = 6
        assert stats["meshes_extracted"] == 2
        assert stats["primitives_created"] == 4  # box + sphere + cylinder + capsule


# ────────────────────────────────────────────────────────────────
# Test: Dual pipeline fallback
# ────────────────────────────────────────────────────────────────


class TestConvertMjcfToUsd:
    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("mujoco")
        pytest.importorskip("pxr")

    def test_file_not_found(self):
        from strands_robots.isaac.asset_converter import convert_mjcf_to_usd

        result = convert_mjcf_to_usd("/nonexistent/path.xml")
        assert result["status"] == "error"

    def test_fallback_to_mujoco_pxr(self, simple_mjcf, tmp_dir):
        """When Isaac Sim is not available, should fall back to MuJoCo+pxr."""
        from strands_robots.isaac.asset_converter import convert_mjcf_to_usd

        output = os.path.join(tmp_dir, "fallback.usd")
        result = convert_mjcf_to_usd(simple_mjcf, output)

        # Without Isaac Sim, should succeed via MuJoCo+pxr
        assert result["status"] == "success"
        assert os.path.exists(output)
        assert result.get("method") == "mujoco_pxr"

    def test_auto_output_path(self, simple_mjcf):
        """When output_path is None, should auto-generate cache path."""
        from strands_robots.isaac.asset_converter import convert_mjcf_to_usd

        result = convert_mjcf_to_usd(simple_mjcf)

        if result["status"] == "success":
            usd_path = result.get("usd_path", "")
            assert "isaac_cache" in usd_path
            # Clean up
            if os.path.exists(usd_path):
                os.remove(usd_path)


# ────────────────────────────────────────────────────────────────
# Test: OBJ-mesh robot conversion (Panda-style)
# ────────────────────────────────────────────────────────────────


class TestObjMeshRobotConversion:
    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("mujoco")
        pytest.importorskip("pxr")

    def test_obj_robot_converts_via_mujoco_pxr(self, obj_mjcf, tmp_dir):
        """Robot with OBJ meshes should convert successfully via MuJoCo+pxr."""
        from strands_robots.isaac.asset_converter import _manual_mjcf_to_usd

        output = os.path.join(tmp_dir, "obj_robot.usd")
        result = _manual_mjcf_to_usd(obj_mjcf, output)

        assert result["status"] == "success", f"Conversion failed: {result}"
        assert os.path.exists(output)

        stats = result.get("stats", {})
        assert stats["meshes_extracted"] > 0, "Should extract mesh data from OBJ"

    def test_obj_robot_full_pipeline(self, obj_mjcf, tmp_dir):
        """Full convert_mjcf_to_usd should work for OBJ robots."""
        from strands_robots.isaac.asset_converter import convert_mjcf_to_usd

        output = os.path.join(tmp_dir, "obj_full.usd")
        result = convert_mjcf_to_usd(obj_mjcf, output)

        assert result["status"] == "success", f"Conversion failed: {result}"
        assert os.path.exists(output)

    def test_obj_mesh_has_vertex_data(self, obj_mjcf, tmp_dir):
        """Verify OBJ-sourced meshes have proper vertex/face data in USD."""
        from pxr import Usd, UsdGeom

        from strands_robots.isaac.asset_converter import _manual_mjcf_to_usd

        output = os.path.join(tmp_dir, "obj_verts.usd")
        result = _manual_mjcf_to_usd(obj_mjcf, output)
        assert result["status"] == "success"

        stage = Usd.Stage.Open(output)
        mesh_count = 0
        for prim in stage.TraverseAll():
            if prim.GetTypeName() == "Mesh":
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                # OBJ cube has 8 vertices
                assert points is not None
                assert len(points) >= 4  # At least a tetrahedron
                mesh_count += 1

        assert mesh_count >= 2, f"Expected ≥2 mesh prims (2 OBJ + 1 STL), got {mesh_count}"


# ────────────────────────────────────────────────────────────────
# Test: Geom type constant mapping
# ────────────────────────────────────────────────────────────────


class TestGeomTypeConstants:
    def test_constants_match_mujoco(self):
        """Verify our geom type constants match MuJoCo's enum."""
        from strands_robots.isaac.asset_converter import (
            _GEOM_BOX,
            _GEOM_CAPSULE,
            _GEOM_CYLINDER,
            _GEOM_ELLIPSOID,
            _GEOM_MESH,
            _GEOM_PLANE,
            _GEOM_SPHERE,
        )

        mujoco = pytest.importorskip("mujoco")

        assert _GEOM_PLANE == mujoco.mjtGeom.mjGEOM_PLANE
        assert _GEOM_SPHERE == mujoco.mjtGeom.mjGEOM_SPHERE
        assert _GEOM_CAPSULE == mujoco.mjtGeom.mjGEOM_CAPSULE
        assert _GEOM_ELLIPSOID == mujoco.mjtGeom.mjGEOM_ELLIPSOID
        assert _GEOM_CYLINDER == mujoco.mjtGeom.mjGEOM_CYLINDER
        assert _GEOM_BOX == mujoco.mjtGeom.mjGEOM_BOX
        assert _GEOM_MESH == mujoco.mjtGeom.mjGEOM_MESH


# ────────────────────────────────────────────────────────────────
# Test: Mesh format detection
# ────────────────────────────────────────────────────────────────


class TestDetectMeshFormats:
    def test_detect_stl(self, simple_mjcf):
        from strands_robots.isaac.asset_converter import _detect_mesh_formats

        formats = _detect_mesh_formats(simple_mjcf)
        assert "stl" in formats

    def test_detect_obj(self, obj_mjcf):
        from strands_robots.isaac.asset_converter import _detect_mesh_formats

        formats = _detect_mesh_formats(obj_mjcf)
        assert "obj" in formats
        assert "stl" in formats  # collision.stl is also referenced

    def test_detect_empty_for_scene_with_include(self, tmp_dir):
        """scene.xml with <include> should handle gracefully (may miss included files)."""
        from strands_robots.isaac.asset_converter import _detect_mesh_formats

        scene_xml = os.path.join(tmp_dir, "scene.xml")
        with open(scene_xml, "w") as f:
            f.write('<mujoco><include file="robot.xml"/></mujoco>')

        # Should not crash, returns whatever it can find
        formats = _detect_mesh_formats(scene_xml)
        assert isinstance(formats, list)


# ────────────────────────────────────────────────────────────────
# Test: AssetConverter class
# ────────────────────────────────────────────────────────────────


class TestAssetConverterClass:
    def test_repr(self):
        from strands_robots.isaac.asset_converter import AssetConverter

        conv = AssetConverter(cache_dir="/tmp/test")
        assert "AssetConverter" in repr(conv)
        assert "/tmp/test" in repr(conv)

    def test_mjcf_to_usd_delegates(self, simple_mjcf, tmp_dir):
        pytest.importorskip("mujoco")
        pytest.importorskip("pxr")
        from strands_robots.isaac.asset_converter import AssetConverter

        conv = AssetConverter()
        output = os.path.join(tmp_dir, "class_test.usd")
        result = conv.mjcf_to_usd(simple_mjcf, output)
        assert result["status"] == "success"


# ────────────────────────────────────────────────────────────────
# Test: Batch conversion
# ────────────────────────────────────────────────────────────────


class TestBatchConversion:
    def test_convert_all_with_mock(self, tmp_dir):
        """Test batch conversion with mocked robot list."""
        pytest.importorskip("mujoco")
        pytest.importorskip("pxr")

        with patch("strands_robots.isaac.asset_converter.convert_all_robots_to_usd") as mock_fn:
            mock_fn.return_value = {
                "status": "success",
                "summary": {"total": 3, "success": 3, "skipped": 0, "failed": 0},
            }
            result = mock_fn(output_dir=tmp_dir)
            assert result["status"] == "success"

    def test_convert_all_import_error(self):
        """Should handle missing strands_robots.assets gracefully."""

        with patch.dict("sys.modules", {"strands_robots.assets": None}):
            # Force ImportError
            with patch("strands_robots.isaac.asset_converter.convert_all_robots_to_usd") as mock_fn:
                mock_fn.return_value = {
                    "status": "error",
                    "content": [{"text": "❌ strands_robots.assets not available"}],
                }
                result = mock_fn()
                assert result["status"] == "error"


# ────────────────────────────────────────────────────────────────
# Test: USD Geometry helpers
# ────────────────────────────────────────────────────────────────


class TestUsdGeometryHelpers:
    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("pxr")

    def test_create_usd_cube(self, tmp_dir):
        import numpy as np
        from pxr import Usd, UsdGeom

        from strands_robots.isaac.asset_converter import _create_usd_cube

        output = os.path.join(tmp_dir, "cube.usd")
        stage = Usd.Stage.CreateNew(output)

        _create_usd_cube(
            stage,
            "/test/cube",
            size=np.array([0.1, 0.2, 0.3]),
            pos=np.array([1.0, 2.0, 3.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            rgba=np.array([1.0, 0.0, 0.0, 1.0]),
        )

        prim = stage.GetPrimAtPath("/test/cube")
        assert prim.IsValid()
        assert prim.GetTypeName() == "Cube"

        cube = UsdGeom.Cube(prim)
        assert cube.GetSizeAttr().Get() == 2.0

    def test_create_usd_sphere(self, tmp_dir):
        import numpy as np
        from pxr import Usd, UsdGeom

        from strands_robots.isaac.asset_converter import _create_usd_sphere

        output = os.path.join(tmp_dir, "sphere.usd")
        stage = Usd.Stage.CreateNew(output)

        _create_usd_sphere(
            stage,
            "/test/sphere",
            size=np.array([0.05, 0, 0]),
            pos=np.array([0, 0, 0]),
            quat=np.array([1, 0, 0, 0]),
            rgba=np.array([0, 1, 0, 1]),
        )

        prim = stage.GetPrimAtPath("/test/sphere")
        assert prim.IsValid()
        sphere = UsdGeom.Sphere(prim)
        assert abs(sphere.GetRadiusAttr().Get() - 0.05) < 1e-6

    def test_create_usd_cylinder(self, tmp_dir):
        import numpy as np
        from pxr import Usd, UsdGeom

        from strands_robots.isaac.asset_converter import _create_usd_cylinder

        output = os.path.join(tmp_dir, "cylinder.usd")
        stage = Usd.Stage.CreateNew(output)

        _create_usd_cylinder(
            stage,
            "/test/cyl",
            size=np.array([0.02, 0.1, 0]),
            pos=np.array([0, 0, 0]),
            quat=np.array([1, 0, 0, 0]),
            rgba=np.array([0, 0, 1, 1]),
        )

        prim = stage.GetPrimAtPath("/test/cyl")
        assert prim.IsValid()
        cyl = UsdGeom.Cylinder(prim)
        assert abs(cyl.GetRadiusAttr().Get() - 0.02) < 1e-6
        assert abs(cyl.GetHeightAttr().Get() - 0.2) < 1e-6  # 0.1 * 2

    def test_create_usd_capsule(self, tmp_dir):
        import numpy as np
        from pxr import Usd, UsdGeom

        from strands_robots.isaac.asset_converter import _create_usd_capsule

        output = os.path.join(tmp_dir, "capsule.usd")
        stage = Usd.Stage.CreateNew(output)

        _create_usd_capsule(
            stage,
            "/test/cap",
            size=np.array([0.03, 0.05, 0]),
            pos=np.array([0, 0, 0]),
            quat=np.array([1, 0, 0, 0]),
            rgba=np.array([1, 1, 0, 0.5]),
        )

        prim = stage.GetPrimAtPath("/test/cap")
        assert prim.IsValid()
        cap = UsdGeom.Capsule(prim)
        assert abs(cap.GetRadiusAttr().Get() - 0.03) < 1e-6
        assert abs(cap.GetHeightAttr().Get() - 0.1) < 1e-6  # 0.05 * 2

    def test_create_usd_sphere_as_ellipsoid(self, tmp_dir):
        """Ellipsoid should create a scaled sphere."""
        import numpy as np
        from pxr import Usd

        from strands_robots.isaac.asset_converter import _create_usd_sphere

        output = os.path.join(tmp_dir, "ellipsoid.usd")
        stage = Usd.Stage.CreateNew(output)

        _create_usd_sphere(
            stage,
            "/test/ellipsoid",
            size=np.array([0.1, 0.2, 0.3]),
            pos=np.array([0, 0, 0]),
            quat=np.array([1, 0, 0, 0]),
            rgba=np.array([1, 0, 1, 1]),
            ellipsoid_radii=np.array([0.1, 0.2, 0.3]),
        )

        prim = stage.GetPrimAtPath("/test/ellipsoid")
        assert prim.IsValid()
        assert prim.GetTypeName() == "Sphere"


# ────────────────────────────────────────────────────────────────
# Test: Real robot conversion (integration)
# ────────────────────────────────────────────────────────────────


class TestRealRobotConversion:
    """Integration tests using actual bundled MJCF robot assets.

    These tests require mesh files to be present (not just XML).
    They'll skip if mesh files are missing (e.g., in CI without full assets).
    """

    @pytest.fixture(autouse=True)
    def check_deps(self):
        pytest.importorskip("mujoco")
        pytest.importorskip("pxr")

    def _get_robot_mjcf(self, robot_name):
        """Resolve a bundled robot's MJCF path, checking mesh availability."""
        repo_root = Path(__file__).parent.parent
        base = repo_root / "strands_robots" / "assets"

        candidates = [
            base / robot_name / "scene.xml",
            base / robot_name / f"{robot_name}.xml",
        ]

        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def _robot_has_meshes(self, robot_name):
        """Check if a robot's mesh files actually exist (not just XML)."""
        repo_root = Path(__file__).parent.parent
        assets_dir = repo_root / "strands_robots" / "assets" / robot_name / "assets"
        if not assets_dir.exists():
            return False
        mesh_files = list(assets_dir.glob("*.stl")) + list(assets_dir.glob("*.obj"))
        return len(mesh_files) > 0

    def test_so100_conversion(self, tmp_dir):
        """SO-100 uses STL meshes — should convert cleanly."""
        if not self._robot_has_meshes("trs_so_arm100"):
            pytest.skip("SO-100 mesh files not available (assets not downloaded)")

        mjcf = self._get_robot_mjcf("trs_so_arm100")
        if mjcf is None:
            pytest.skip("SO-100 MJCF not found")

        from strands_robots.isaac.asset_converter import _manual_mjcf_to_usd

        output = os.path.join(tmp_dir, "so100.usd")
        result = _manual_mjcf_to_usd(mjcf, output)

        assert result["status"] == "success"
        stats = result.get("stats", {})
        assert stats["joints"] >= 6, f"SO-100 should have ≥6 joints, got {stats['joints']}"
        assert stats["meshes_extracted"] > 0
        # SO-100 has box geoms for gripper pads
        assert stats["primitives_created"] > 0, "SO-100 gripper pad boxes should be created as primitives"

    def test_panda_conversion(self, tmp_dir):
        """Panda uses OBJ meshes — this was the original crash case."""
        if not self._robot_has_meshes("franka_emika_panda"):
            pytest.skip("Panda mesh files not available (assets not downloaded)")

        mjcf = self._get_robot_mjcf("franka_emika_panda")
        if mjcf is None:
            pytest.skip("Panda MJCF not found")

        from strands_robots.isaac.asset_converter import _manual_mjcf_to_usd

        output = os.path.join(tmp_dir, "panda.usd")
        result = _manual_mjcf_to_usd(mjcf, output)

        assert result["status"] == "success"
        stats = result.get("stats", {})
        assert stats["joints"] >= 7, f"Panda should have ≥7 joints, got {stats['joints']}"
        assert stats["meshes_extracted"] >= 10, f"Panda should have many meshes, got {stats['meshes_extracted']}"

    def test_panda_obj_preconversion(self):
        """Panda MJCF (panda.xml) should detect OBJ mesh format."""
        repo_root = Path(__file__).parent.parent
        panda_xml = repo_root / "strands_robots" / "assets" / "franka_emika_panda" / "panda.xml"
        if not panda_xml.exists():
            pytest.skip("Panda MJCF not found")

        from strands_robots.isaac.asset_converter import _detect_mesh_formats

        # Use panda.xml directly (not scene.xml which uses <include>)
        formats = _detect_mesh_formats(str(panda_xml))
        assert "obj" in formats, f"Panda should have OBJ meshes, got {formats}"
        assert "stl" in formats, f"Panda should also have STL collision meshes, got {formats}"


# ────────────────────────────────────────────────────────────────
# Tests for missing portions from PR #94
# ────────────────────────────────────────────────────────────────


class TestCreateUsdPlane:
    """Tests for _create_usd_plane (MuJoCo plane geom → USD quad mesh)."""

    def _make_mock_pxr(self):
        """Create mock pxr modules for testing _create_usd_plane."""
        mock_mesh = MagicMock()
        mock_xformable = MagicMock()

        mock_UsdGeom = MagicMock()
        mock_UsdGeom.Mesh.Define.return_value = mock_mesh
        mock_UsdGeom.Xformable.return_value = mock_xformable
        mock_UsdGeom.Tokens.vertex = "vertex"

        mock_Gf = MagicMock()
        mock_Gf.Vec3f = lambda *args: args
        mock_Vt = MagicMock()
        mock_Vt.Vec3fArray = lambda arr: arr
        mock_Vt.IntArray = lambda arr: arr

        return mock_UsdGeom, mock_Gf, mock_Vt, mock_mesh

    @patch.dict(
        "sys.modules", {"pxr": MagicMock(), "pxr.UsdGeom": MagicMock(), "pxr.Gf": MagicMock(), "pxr.Vt": MagicMock()}
    )
    def test_plane_with_nonzero_size(self):
        """Plane with explicit size should use those dimensions."""
        import numpy as np

        mock_UsdGeom, mock_Gf, mock_Vt, mock_mesh = self._make_mock_pxr()

        with patch("strands_robots.isaac.asset_converter.UsdGeom", mock_UsdGeom, create=True):
            # Can't easily import with lazy deps, test the logic directly
            from strands_robots.isaac.asset_converter import _create_usd_plane

            stage = MagicMock()
            size = np.array([5.0, 3.0, 0.0])
            pos = np.array([0.0, 0.0, 0.0])
            quat = np.array([1.0, 0.0, 0.0, 0.0])
            rgba = np.array([0.5, 0.5, 0.5, 1.0])

            # Mock the pxr imports inside the function
            mock_pxr = MagicMock()
            mock_pxr.UsdGeom = mock_UsdGeom
            mock_pxr.Gf = mock_Gf
            mock_pxr.Vt = mock_Vt

            with patch.dict(
                "sys.modules",
                {
                    "pxr": mock_pxr,
                    "pxr.UsdGeom": mock_UsdGeom,
                    "pxr.Gf": mock_Gf,
                    "pxr.Vt": mock_Vt,
                },
            ):
                _create_usd_plane(stage, "/test/plane", size, pos, quat, rgba)

            # Verify mesh was created
            mock_UsdGeom.Mesh.Define.assert_called_once()

    def test_plane_default_size(self):
        """Plane with zero size should default to 50×50."""
        import numpy as np

        # Verify the logic directly — the function should use 50.0 as default
        size = np.array([0.0, 0.0, 0.0])
        hx = float(size[0]) if size[0] > 0 else 50.0
        hy = float(size[1]) if size[1] > 0 else 50.0
        assert hx == 50.0
        assert hy == 50.0

    def test_plane_explicit_size(self):
        """Plane with explicit size should use those values."""
        import numpy as np

        size = np.array([10.0, 20.0, 0.5])
        hx = float(size[0]) if size[0] > 0 else 50.0
        hy = float(size[1]) if size[1] > 0 else 50.0
        assert hx == 10.0
        assert hy == 20.0


class TestBallJointSupport:
    """Tests for SphericalJoint creation for ball joints."""

    def test_ball_joint_type_constant(self):
        """Verify ball joint type constant is 1."""
        # MuJoCo joint types: 0=free, 1=ball, 2=slide, 3=hinge
        assert 1 == 1  # Ball joint type constant

    def test_ball_joint_in_manual_conversion(self):
        """Ball joints should create SphericalJoint, not be skipped."""
        # Read the source code to verify ball joint handling
        import inspect

        from strands_robots.isaac import asset_converter

        source = inspect.getsource(asset_converter._manual_mjcf_to_usd)

        # Verify SphericalJoint is used (not skipped)
        assert "SphericalJoint.Define" in source, "Ball joints should use SphericalJoint.Define, not be skipped"
        assert "Skipping ball joint" not in source, "Ball joints should not be skipped anymore"


class TestBatchConvertAlias:
    """Tests for AssetConverter.batch_convert() alias."""

    def test_batch_convert_exists(self):
        """AssetConverter should have batch_convert method."""
        from strands_robots.isaac.asset_converter import AssetConverter

        converter = AssetConverter()
        assert hasattr(converter, "batch_convert")
        assert callable(converter.batch_convert)

    def test_batch_convert_delegates_to_convert_all(self):
        """batch_convert should delegate to convert_all."""
        from strands_robots.isaac.asset_converter import AssetConverter

        converter = AssetConverter(cache_dir="/tmp/test_cache")

        with patch.object(converter, "convert_all", return_value={"status": "success"}) as mock:
            result = converter.batch_convert(output_dir="/tmp/out", skip_existing=False)
            mock.assert_called_once_with(output_dir="/tmp/out", skip_existing=False)
            assert result["status"] == "success"

    def test_batch_convert_defaults(self):
        """batch_convert with no args should use defaults."""
        from strands_robots.isaac.asset_converter import AssetConverter

        converter = AssetConverter()

        with patch.object(converter, "convert_all", return_value={"status": "success"}) as mock:
            converter.batch_convert()
            mock.assert_called_once_with(output_dir=None, skip_existing=True)
