"""Asset Conversion: MJCF ↔ USD for cross-backend compatibility.

strands-robots bundles 32 robots (28 MuJoCo Menagerie + 3 community) as MJCF.
Isaac Sim uses Universal Scene Description (USD) format.
This module converts between them.

Conversion paths:
    MJCF → USD: For using strands-robots' 32 robots in Isaac Sim
    USD → MJCF: For using Isaac Lab's USD assets in our MuJoCo sim

Dual pipeline:
    Method 1 (Isaac Sim): Full-fidelity conversion via Isaac Lab's MjcfConverter.
        Includes OBJ→STL pre-conversion to work around Isaac Sim's MJCF importer
        crash when encountering OBJ mesh files (omni.kit.asset_converter fails
        to convert OBJ → USD, causing null pointer dereference).
    Method 2 (MuJoCo+pxr): Extract mesh geometry from MuJoCo's compiled model
        and write proper UsdGeom.Mesh / UsdGeom primitive prims via OpenUSD.
        Works for ALL robots regardless of mesh format.

Uses:
    - Isaac Lab's MjcfConverter (MJCF → USD via Isaac Sim)
    - MuJoCo's built-in URDF→MJCF (for URDF intermediary)
    - MuJoCo mesh data extraction (vertices, faces, normals)
    - OpenUSD (pxr) for direct USD stage authoring
    - trimesh/numpy for OBJ→STL pre-conversion

Requirements:
    - Isaac Sim for Method 1 (MJCF → USD full-fidelity conversion)
    - MuJoCo + usd-core for Method 2 (fallback, always works)
"""

import logging
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# MuJoCo geom type constants (from mjtGeom enum)
_GEOM_PLANE = 0
_GEOM_HFIELD = 1
_GEOM_SPHERE = 2
_GEOM_CAPSULE = 3
_GEOM_ELLIPSOID = 4
_GEOM_CYLINDER = 5
_GEOM_BOX = 6
_GEOM_MESH = 7

_GEOM_TYPE_NAMES = {
    _GEOM_PLANE: "plane",
    _GEOM_HFIELD: "hfield",
    _GEOM_SPHERE: "sphere",
    _GEOM_CAPSULE: "capsule",
    _GEOM_ELLIPSOID: "ellipsoid",
    _GEOM_CYLINDER: "cylinder",
    _GEOM_BOX: "box",
    _GEOM_MESH: "mesh",
}


def _preconvert_obj_to_stl(mjcf_path: str) -> Optional[str]:
    """Pre-convert OBJ meshes to STL in MJCF assets to work around Isaac Sim crash.

    Isaac Sim's MJCF importer (omni.kit.asset_converter) crashes when it encounters
    OBJ mesh files — it tries to convert OBJ→USD internally, creates a temp directory
    (e.g. link1_tmp/), but fails to write the .tmp.usd file. This causes a null pointer
    crash: ``attempted member lookup on NULL TfRefPtr<UsdStage>``.

    This function:
    1. Parses the MJCF XML to find all <mesh> assets
    2. Identifies which reference .obj files
    3. Converts each OBJ → STL using trimesh (or numpy fallback)
    4. Creates a modified MJCF copy with .obj → .stl references
    5. Returns path to the modified MJCF (or None if no OBJ files found)

    Args:
        mjcf_path: Path to the original MJCF XML file

    Returns:
        Path to modified MJCF with STL meshes, or None if no conversion needed.
        The caller is responsible for cleaning up the temp directory.
    """
    import xml.etree.ElementTree as ET

    mjcf_path = os.path.abspath(mjcf_path)
    mjcf_dir = os.path.dirname(mjcf_path)

    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # Find meshdir from compiler
    meshdir = ""
    compiler = root.find(".//compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir", "")

    mesh_dir_abs = (
        os.path.normpath(os.path.join(mjcf_dir, meshdir)) if meshdir else mjcf_dir
    )

    # Collect all mesh elements referencing OBJ files
    obj_meshes = []
    for mesh_elem in root.iter("mesh"):
        mesh_file = mesh_elem.get("file", "")
        if mesh_file.lower().endswith(".obj"):
            obj_meshes.append((mesh_elem, mesh_file))

    if not obj_meshes:
        return None  # No OBJ files, no conversion needed

    logger.info("Found %s OBJ meshes to pre-convert to STL", len(obj_meshes))

    # Create a temp directory for converted files
    tmp_dir = tempfile.mkdtemp(prefix="strands_obj2stl_")

    # Copy the entire meshdir contents first
    tmp_mesh_dir = os.path.join(tmp_dir, "assets") if meshdir else tmp_dir
    if os.path.isdir(mesh_dir_abs):
        shutil.copytree(mesh_dir_abs, tmp_mesh_dir, dirs_exist_ok=True)

    # Convert each OBJ → STL
    converted = 0
    for mesh_elem, mesh_file in obj_meshes:
        obj_path = os.path.join(mesh_dir_abs, mesh_file)
        stl_file = mesh_file.rsplit(".", 1)[0] + ".stl"
        stl_path = os.path.join(tmp_mesh_dir, stl_file)

        if not os.path.exists(obj_path):
            logger.warning("OBJ file not found: %s", obj_path)
            continue

        try:
            _convert_single_obj_to_stl(obj_path, stl_path)
            mesh_elem.set("file", stl_file)
            converted += 1
            logger.debug("Converted: %s → %s", mesh_file, stl_file)
        except Exception as e:
            logger.warning("Failed to convert %s: %s", mesh_file, e)

    if converted == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    # Update compiler meshdir to point to tmp assets
    if compiler is not None:
        compiler.set("meshdir", tmp_mesh_dir)
    else:
        # Add compiler element
        comp = ET.SubElement(root, "compiler")
        comp.set("meshdir", tmp_mesh_dir)

    # Write modified MJCF
    tmp_mjcf = os.path.join(tmp_dir, os.path.basename(mjcf_path))
    tree.write(tmp_mjcf, xml_declaration=True, encoding="unicode")

    logger.info(
        f"Pre-converted {converted}/{len(obj_meshes)} OBJ→STL, modified MJCF: {tmp_mjcf}"
    )
    return tmp_mjcf


def _convert_single_obj_to_stl(obj_path: str, stl_path: str) -> None:
    """Convert a single OBJ file to binary STL.

    Tries trimesh first (robust, handles complex OBJ), falls back to
    a minimal numpy-based parser for simple triangle meshes.
    """
    try:
        import trimesh

        mesh = trimesh.load(obj_path, force="mesh")
        mesh.export(stl_path, file_type="stl")
        return
    except ImportError:
        pass

    # Fallback: minimal OBJ→STL via numpy
    import struct

    import numpy as np

    vertices = []
    faces = []

    with open(obj_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                # Parse face indices (OBJ is 1-indexed, may have v/vt/vn format)
                face_verts = []
                for p in parts[1:]:
                    idx = int(p.split("/")[0]) - 1
                    face_verts.append(idx)
                # Triangulate polygons
                for i in range(1, len(face_verts) - 1):
                    faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])

    if not vertices or not faces:
        raise ValueError(f"Empty mesh in {obj_path}")

    verts = np.array(vertices, dtype=np.float32)
    tris = np.array(faces, dtype=np.int32)

    # Write binary STL
    with open(stl_path, "wb") as f:
        f.write(b"\0" * 80)  # header
        f.write(struct.pack("<I", len(tris)))
        for tri in tris:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            # Compute normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal = normal / norm_len
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))  # attribute byte count


def _strip_meshes_from_mjcf(mjcf_path: str) -> Optional[str]:
    """Create a temporary MJCF copy with all mesh references removed.

    When bundled robots ship without their mesh asset files (e.g. MuJoCo
    Menagerie XMLs that reference ``meshdir=assets/`` but the STL/OBJ files
    are not included in the package), MuJoCo's ``MjModel.from_xml_path()``
    raises ``ValueError``.

    This function creates a modified copy that:
    - Removes all ``<mesh>`` elements from ``<asset>``
    - Removes all ``<geom>`` elements with ``type="mesh"`` or ``mesh="..."``
    - Preserves all primitive geoms (box, sphere, capsule, cylinder)
    - Preserves all joints, bodies, actuators, and kinematic structure

    The resulting USD will contain only the kinematic chain and primitive
    collision geometries — sufficient for physics simulation and RL training,
    just without visual mesh detail.

    Args:
        mjcf_path: Path to the original MJCF XML file.

    Returns:
        Path to a temporary MJCF file with meshes stripped, or ``None``
        if the MJCF has no mesh references.  Caller should clean up the
        temp file/directory when done.
    """
    import xml.etree.ElementTree as ET

    mjcf_path = os.path.abspath(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    removed_meshes = 0
    removed_geoms = 0

    # Remove <mesh> and <texture> elements from <asset> that reference
    # files not present on disk.  Textures (.png, .jpg) are also stripped
    # because MuJoCo fails to load MJCF when referenced files are missing.
    mjcf_dir_abs = os.path.dirname(mjcf_path)

    # Resolve meshdir from compiler
    compiler_elem = root.find(".//compiler")
    meshdir = ""
    if compiler_elem is not None:
        meshdir = compiler_elem.get("meshdir", "")
    mesh_dir_abs = (
        os.path.normpath(os.path.join(mjcf_dir_abs, meshdir))
        if meshdir
        else mjcf_dir_abs
    )
    texturedir = ""
    if compiler_elem is not None:
        texturedir = compiler_elem.get("texturedir", "")
    texture_dir_abs = (
        os.path.normpath(os.path.join(mjcf_dir_abs, texturedir))
        if texturedir
        else mjcf_dir_abs
    )

    removed_textures = 0
    removed_materials = 0

    for asset_elem in root.iter("asset"):
        # Remove mesh elements
        mesh_elems = list(asset_elem.findall("mesh"))
        for mesh_elem in mesh_elems:
            mesh_file = mesh_elem.get("file", "")
            if mesh_file:
                asset_elem.remove(mesh_elem)
                removed_meshes += 1

        # Remove texture elements referencing missing files
        tex_elems = list(asset_elem.findall("texture"))
        for tex_elem in tex_elems:
            tex_file = tex_elem.get("file", "")
            if tex_file:
                # Check if file exists relative to texturedir
                full_path = os.path.join(texture_dir_abs, tex_file)
                if not os.path.exists(full_path):
                    # Also check relative to meshdir and mjcf dir
                    alt_path = os.path.join(mesh_dir_abs, tex_file)
                    alt_path2 = os.path.join(mjcf_dir_abs, tex_file)
                    if not os.path.exists(alt_path) and not os.path.exists(alt_path2):
                        asset_elem.remove(tex_elem)
                        removed_textures += 1

        # Remove material elements that reference removed textures
        # (MuJoCo will error on dangling texture references)
        mat_elems = list(asset_elem.findall("material"))
        for mat_elem in mat_elems:
            mat_tex = mat_elem.get("texture", "")
            if mat_tex:
                # Check if the referenced texture still exists in the asset
                remaining_textures = {
                    t.get("name", "") for t in asset_elem.findall("texture")
                }
                if mat_tex not in remaining_textures:
                    # Remove the texture attribute rather than the whole material
                    del mat_elem.attrib["texture"]
                    removed_materials += 1

    # ── Build mesh_classes set BEFORE removing geoms ──
    # Must happen first because _remove_mesh_geoms also strips <geom>
    # children inside <default> elements, which would prevent us from
    # detecting which classes define type="mesh".
    mesh_classes = set()
    for default_elem in root.iter("default"):
        cls_name = default_elem.get("class", "")
        for geom_def in default_elem.findall("geom"):
            if geom_def.get("type") == "mesh":
                if cls_name:
                    mesh_classes.add(cls_name)

    # Remove <geom> elements that reference meshes
    # We need to walk the entire tree and remove mesh-typed geoms
    def _remove_mesh_geoms(elem):
        nonlocal removed_geoms
        to_remove = []
        for child in list(elem):
            if child.tag == "geom":
                geom_type = child.get("type", "")
                geom_mesh = child.get("mesh", "")
                if geom_type == "mesh" or geom_mesh:
                    to_remove.append(child)
                    removed_geoms += 1
            else:
                _remove_mesh_geoms(child)
        for child in to_remove:
            elem.remove(child)

    _remove_mesh_geoms(root)

    # Remove geoms that inherit mesh type from class defaults.
    # After mesh stripping, such geoms would cause errors in MuJoCo.
    removed_classonly_geoms = 0

    def _remove_mesh_class_geoms(elem):
        nonlocal removed_classonly_geoms
        to_remove = []
        for child in list(elem):
            if child.tag == "geom":
                geom_class = child.get("class", "")
                has_type = child.get("type")
                has_size = child.get("size")
                has_mesh = child.get("mesh")
                has_fromto = child.get("fromto")
                # Remove geoms that inherit mesh type from their class
                # and have no explicit type/size override
                if geom_class in mesh_classes and not has_type and not has_mesh:
                    to_remove.append(child)
                    removed_classonly_geoms += 1
                # Also remove geoms with no class, type, size, mesh, or fromto
                # (these cannot define valid geometry)
                elif (
                    not geom_class
                    and not has_type
                    and not has_size
                    and not has_mesh
                    and not has_fromto
                ):
                    to_remove.append(child)
                    removed_classonly_geoms += 1
            else:
                _remove_mesh_class_geoms(child)
        for child in to_remove:
            elem.remove(child)

    _remove_mesh_class_geoms(root)

    # ── Add default inertial to bodies that lost all geoms ──
    # When all geoms on a body are mesh-type and get stripped, the body
    # has zero mass.  MuJoCo rejects this with:
    #   "mass and inertia of moving bodies must be larger than mjMINVAL"
    # Fix: add a minimal <inertial> element to bodies that have no
    # remaining child geoms AND no existing <inertial> element.
    added_inertials = 0
    for body_elem in root.iter("body"):
        has_geom = any(child.tag == "geom" for child in body_elem)
        has_inertial = any(child.tag == "inertial" for child in body_elem)
        if not has_geom and not has_inertial:
            inertial = ET.SubElement(body_elem, "inertial")
            inertial.set("pos", "0 0 0")
            inertial.set("mass", "0.001")
            inertial.set("diaginertia", "1e-8 1e-8 1e-8")
            added_inertials += 1

    if removed_meshes == 0 and removed_geoms == 0 and removed_textures == 0:
        return None  # No mesh/texture references found

    logger.info(
        f"Stripped {removed_meshes} mesh assets, {removed_geoms} mesh geoms, "
        f"{removed_classonly_geoms} class-only geoms, "
        f"{removed_textures} textures, {removed_materials} material refs "
        f"from {os.path.basename(mjcf_path)}; added {added_inertials} default "
        f"inertials (asset files not available)"
    )

    # Write to a temp file
    tmp_dir = tempfile.mkdtemp(prefix="strands_meshstrip_")
    tmp_mjcf = os.path.join(tmp_dir, os.path.basename(mjcf_path))

    # Copy any included XML files to the temp directory
    mjcf_dir = os.path.dirname(mjcf_path)
    for inc_elem in root.iter("include"):
        inc_file = inc_elem.get("file", "")
        if inc_file:
            src = os.path.join(mjcf_dir, inc_file)
            dst = os.path.join(tmp_dir, inc_file)
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

    tree.write(tmp_mjcf, xml_declaration=True, encoding="unicode")
    return tmp_mjcf


def convert_mjcf_to_usd(
    mjcf_path: str,
    output_path: Optional[str] = None,
    fix_base: bool = True,
    make_instanceable: bool = False,
    *,
    usd_path: Optional[str] = None,
    fix_collision_meshes: bool = False,
    generate_physics: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Convert a MuJoCo MJCF file to USD format for Isaac Sim.

    Uses a dual-pipeline approach:
    1. Try Isaac Lab's MjcfConverter (full fidelity, physics schemas, materials)
       - Pre-converts OBJ meshes → STL to avoid Isaac Sim crash
    2. Fall back to MuJoCo + OpenUSD (pxr) extraction (always works)
       - Extracts actual mesh vertices/faces from MuJoCo's compiled model
       - Creates proper UsdGeom.Mesh and primitive geoms

    Args:
        mjcf_path: Path to MJCF XML file
        output_path: Output USD path (auto-generated if None)
        fix_base: Fix the base link (no floating base)
        make_instanceable: Create instanceable USD for parallel envs

    Returns:
        Dict with status, output path, and conversion metadata

    Example:
        result = convert_mjcf_to_usd(
            "/path/to/unitree_go2/scene.xml",
            "/path/to/unitree_go2.usd",
        )
    """
    # Accept usd_path as alias for output_path (used by tool APIs)
    if usd_path is not None and output_path is None:
        output_path = usd_path

    mjcf_path = os.path.abspath(mjcf_path)
    if not os.path.exists(mjcf_path):
        return {
            "status": "error",
            "content": [{"text": f"❌ File not found: {mjcf_path}"}],
        }

    if output_path is None:
        stem = Path(mjcf_path).stem
        cache_dir = Path.home() / ".strands_robots" / "isaac_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(cache_dir / f"{stem}.usd")

    tmp_mjcf = None
    try:
        # ── Method 1: Isaac Lab's MjcfConverter (preferred, full fidelity) ──
        try:
            from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

            # Pre-convert OBJ → STL to avoid Isaac Sim's MJCF importer crash
            effective_mjcf = mjcf_path
            tmp_mjcf = _preconvert_obj_to_stl(mjcf_path)
            if tmp_mjcf:
                logger.info("Using OBJ→STL pre-converted MJCF: %s", tmp_mjcf)
                effective_mjcf = tmp_mjcf

            cfg = MjcfConverterCfg(
                asset_path=effective_mjcf,
                usd_dir=os.path.dirname(output_path),
                usd_file_name=os.path.basename(output_path),
                fix_base=fix_base,
                make_instanceable=make_instanceable,
            )

            converter = MjcfConverter(cfg)
            usd_path = converter.usd_path

            method = "Isaac Lab MjcfConverter"
            if tmp_mjcf:
                method += " (with OBJ→STL pre-conversion)"

            logger.info("✅ Converted MJCF → USD via %s: %s", method, usd_path)
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"✅ MJCF → USD conversion complete\n"
                            f"📥 Input: {os.path.basename(mjcf_path)}\n"
                            f"📤 Output: {usd_path}\n"
                            f"🔧 Method: {method}\n"
                            f"⚙️ fix_base={fix_base}, instanceable={make_instanceable}"
                        )
                    }
                ],
                "usd_path": usd_path,
                "method": "isaac_sim",
            }

        except ImportError:
            logger.info(
                "Isaac Lab MjcfConverter not available, trying alternative method"
            )
        except Exception as e:
            logger.warning(
                f"Isaac Lab conversion failed: {e}, falling back to MuJoCo+pxr"
            )

        # ── Method 2: Isaac Sim's built-in converter (if available) ──
        try:
            import isaacsim.utils.converter as converter_utils

            effective_mjcf = mjcf_path
            tmp_mjcf_2 = _preconvert_obj_to_stl(mjcf_path)
            if tmp_mjcf_2:
                effective_mjcf = tmp_mjcf_2
                if tmp_mjcf is None:
                    tmp_mjcf = tmp_mjcf_2

            converter_utils.convert_asset_to_usd(
                input_asset_path=effective_mjcf,
                output_usd_path=output_path,
            )

            logger.info("✅ Converted via Isaac Sim converter: %s", output_path)
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"✅ Converted: {os.path.basename(mjcf_path)} → {output_path}"
                    }
                ],
                "usd_path": output_path,
                "method": "isaac_sim_converter",
            }
        except ImportError:
            pass
        except Exception as e:
            logger.warning(
                f"Isaac Sim converter failed: {e}, falling back to MuJoCo+pxr"
            )

        # ── Method 3: MuJoCo + OpenUSD (pxr) — full mesh extraction ──
        return _manual_mjcf_to_usd(mjcf_path, output_path)

    except Exception as e:
        logger.error("MJCF → USD conversion failed: %s", e)
        return {"status": "error", "content": [{"text": f"❌ Conversion failed: {e}"}]}

    finally:
        # Clean up temp files from OBJ→STL pre-conversion
        if tmp_mjcf:
            tmp_dir = os.path.dirname(tmp_mjcf)
            if tmp_dir and "strands_obj2stl_" in tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)


def convert_usd_to_mjcf(
    usd_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a USD file to MJCF format for MuJoCo.

    This is less common but useful for:
    - Using Isaac Lab's USD assets in our MuJoCo simulation
    - Cross-validating physics between backends

    The conversion goes: USD → URDF (via Isaac Sim) → MJCF (via MuJoCo)

    Args:
        usd_path: Path to USD file
        output_path: Output MJCF path (auto-generated if None)

    Returns:
        Dict with status and output path
    """
    usd_path = os.path.abspath(usd_path)
    if not os.path.exists(usd_path):
        return {
            "status": "error",
            "content": [{"text": f"❌ File not found: {usd_path}"}],
        }

    if output_path is None:
        stem = Path(usd_path).stem
        cache_dir = Path.home() / ".strands_robots" / "mjcf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(cache_dir / f"{stem}.xml")

    try:
        # Step 1: USD → URDF (via Isaac Sim's exporter)
        urdf_tmp = tempfile.mktemp(suffix=".urdf", prefix="strands_")

        try:
            # Isaac Lab can import URDF but not export to it directly
            # We need to use Omniverse's USD → URDF exporter
            import omni.isaac.core.utils.extensions as ext_utils
            from isaaclab.sim.converters import UrdfConverterCfg  # noqa: F401

            ext_utils.enable_extension("omni.importer.urdf")

            from omni.importer.urdf import _urdf  # noqa: F401

            # This is highly dependent on Isaac Sim version
            logger.warning("USD → URDF export requires Isaac Sim runtime")

        except ImportError:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "❌ USD → MJCF requires Isaac Sim runtime.\n"
                            "Isaac Sim is needed for the USD → URDF step.\n"
                            "💡 Alternative: Use the MJCF version of the robot directly"
                        )
                    }
                ],
            }

        # Step 2: URDF → MJCF (via MuJoCo)
        if os.path.exists(urdf_tmp):
            import mujoco

            model = mujoco.MjModel.from_xml_path(urdf_tmp)
            mujoco.mj_saveLastXML(output_path, model)

            os.remove(urdf_tmp)

            return {
                "status": "success",
                "content": [
                    {
                        "text": f"✅ Converted: {os.path.basename(usd_path)} → {output_path}"
                    }
                ],
                "mjcf_path": output_path,
            }

    except Exception as e:
        logger.error("USD → MJCF conversion failed: %s", e)
        return {"status": "error", "content": [{"text": f"❌ Conversion failed: {e}"}]}


def _manual_mjcf_to_usd(mjcf_path: str, output_path: str) -> Dict[str, Any]:
    """Generate a full USD file from MJCF using MuJoCo + OpenUSD (pxr).

    This method extracts actual mesh geometry (vertices, faces, normals) from
    MuJoCo's compiled model data and writes proper USD prims:
    - UsdGeom.Mesh for mesh-type geoms (with full vertex/face data)
    - UsdGeom.Cube for box geoms
    - UsdGeom.Sphere for sphere geoms
    - UsdGeom.Cylinder for cylinder geoms
    - UsdGeom.Capsule for capsule geoms
    - UsdPhysics.RevoluteJoint / PrismaticJoint for joints

    Works for ALL robots regardless of source mesh format (STL, OBJ, etc.)
    because MuJoCo loads and triangulates all meshes internally.

    MuJoCo data model used:
        model.mesh_vertadr[i], mesh_vertnum[i] → vertex slice for mesh i
        model.mesh_faceadr[i], mesh_facenum[i] → face slice for mesh i
        model.mesh_vert → all vertices (N×3 float)
        model.mesh_face → all face indices (M×3 int, triangulated)
        model.mesh_normal → all vertex normals (N×3 float)
        model.geom_type[i] → geom type enum
        model.geom_size[i] → geom size parameters (semantics vary by type)
        model.geom_dataid[i] → mesh ID for mesh-type geoms
        model.geom_bodyid[i] → owning body ID
        model.geom_pos[i] → geom position relative to body
        model.geom_quat[i] → geom orientation relative to body
        model.geom_rgba[i] → geom color
    """
    try:
        import mujoco
        import numpy as np  # noqa: F401
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt  # noqa: F401

        # MuJoCo resolves relative mesh paths (from <compiler meshdir>)
        # relative to the CWD, not relative to the MJCF file.  We must
        # chdir to the MJCF directory so mesh files like "assets/foo.stl"
        # are found correctly.
        #
        # NOTE: os.chdir() is process-global and NOT thread-safe.  If this
        # function is ever called from multiple threads concurrently, the CWD
        # changes will corrupt each other.  For the current sequential asset
        # conversion use case this is acceptable.  For parallel conversion,
        # use multiprocessing (separate processes) instead of threading.
        mjcf_dir = os.path.dirname(os.path.abspath(mjcf_path))
        meshes_stripped = False
        tmp_meshstrip_dir = None
        saved_cwd = os.getcwd()
        try:
            os.chdir(mjcf_dir)
            try:
                model = mujoco.MjModel.from_xml_path(mjcf_path)
            except ValueError as load_err:
                # MuJoCo raises ValueError when mesh files are missing
                # (e.g. "Error opening file 'assets/link1.stl'").
                # Fall back to a mesh-stripped copy so we can still
                # produce a kinematic-only USD with primitive geoms.
                err_msg = str(load_err)
                if "Error opening file" in err_msg or "file" in err_msg.lower():
                    logger.warning(
                        f"Mesh files missing for {os.path.basename(mjcf_path)}: "
                        f"{err_msg}. Stripping mesh references for kinematic-only USD."
                    )
                    stripped_mjcf = _strip_meshes_from_mjcf(mjcf_path)
                    if stripped_mjcf is not None:
                        tmp_meshstrip_dir = os.path.dirname(stripped_mjcf)
                        os.chdir(os.path.dirname(stripped_mjcf))
                        model = mujoco.MjModel.from_xml_path(stripped_mjcf)
                        meshes_stripped = True
                    else:
                        raise  # No mesh references found, original error stands
                else:
                    raise
        finally:
            os.chdir(saved_cwd)

        # Create USD stage
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        stage = Usd.Stage.CreateNew(output_path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        # Root hierarchy
        UsdGeom.Xform.Define(stage, "/World")
        robot_path = "/World/Robot"
        robot_xform = UsdGeom.Xform.Define(stage, robot_path)

        # Add ArticulationRoot schema to the robot
        UsdPhysics.ArticulationRootAPI.Apply(robot_xform.GetPrim())

        # ── Build body hierarchy ──
        # MuJoCo body tree: body_parentid[i] gives the parent of body i
        # body 0 = "world" (root)
        body_usd_paths: Dict[int, str] = {}
        body_usd_paths[0] = robot_path  # world body maps to robot root

        stats = {
            "bodies": 0,
            "joints": 0,
            "geoms": 0,
            "meshes_extracted": 0,
            "primitives_created": 0,
        }

        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not body_name or body_name == "world":
                body_usd_paths[i] = robot_path
                continue

            safe_name = _sanitize_usd_name(body_name)
            parent_id = model.body_parentid[i]
            parent_path = body_usd_paths.get(parent_id, robot_path)
            body_path = f"{parent_path}/{safe_name}"

            body_xform = UsdGeom.Xform.Define(stage, body_path)
            body_usd_paths[i] = body_path

            # Set body transform relative to parent
            pos = model.body_pos[i]
            quat = model.body_quat[i]  # MuJoCo: (w, x, y, z)
            _set_xform_transform(body_xform, pos, quat)

            # Apply RigidBody physics
            UsdPhysics.RigidBodyAPI.Apply(body_xform.GetPrim())

            # Set mass properties if available
            mass = float(model.body_mass[i])
            if mass > 0:
                mass_api = UsdPhysics.MassAPI.Apply(body_xform.GetPrim())
                mass_api.CreateMassAttr(mass)

            stats["bodies"] += 1

        # ── Add geoms (mesh geometry + primitives) ──
        for i in range(model.ngeom):
            geom_type = int(model.geom_type[i])
            body_id = int(model.geom_bodyid[i])
            body_path = body_usd_paths.get(body_id, robot_path)

            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name:
                safe_geom = _sanitize_usd_name(geom_name)
            else:
                type_str = _GEOM_TYPE_NAMES.get(geom_type, "geom")
                safe_geom = f"{type_str}_{i}"

            geom_path = f"{body_path}/{safe_geom}"

            # Get geom transform relative to body
            geom_pos = model.geom_pos[i]
            geom_quat = model.geom_quat[i]
            geom_size = model.geom_size[i]
            geom_rgba = model.geom_rgba[i]

            if geom_type == _GEOM_MESH:
                mesh_id = int(model.geom_dataid[i])
                if mesh_id >= 0:
                    _create_usd_mesh_from_mujoco(
                        stage,
                        geom_path,
                        model,
                        mesh_id,
                        geom_pos,
                        geom_quat,
                        geom_rgba,
                    )
                    stats["meshes_extracted"] += 1
                    stats["geoms"] += 1

            elif geom_type == _GEOM_BOX:
                _create_usd_cube(
                    stage,
                    geom_path,
                    geom_size,
                    geom_pos,
                    geom_quat,
                    geom_rgba,
                )
                stats["primitives_created"] += 1
                stats["geoms"] += 1

            elif geom_type == _GEOM_SPHERE:
                _create_usd_sphere(
                    stage,
                    geom_path,
                    geom_size,
                    geom_pos,
                    geom_quat,
                    geom_rgba,
                )
                stats["primitives_created"] += 1
                stats["geoms"] += 1

            elif geom_type == _GEOM_CYLINDER:
                _create_usd_cylinder(
                    stage,
                    geom_path,
                    geom_size,
                    geom_pos,
                    geom_quat,
                    geom_rgba,
                )
                stats["primitives_created"] += 1
                stats["geoms"] += 1

            elif geom_type == _GEOM_CAPSULE:
                _create_usd_capsule(
                    stage,
                    geom_path,
                    geom_size,
                    geom_pos,
                    geom_quat,
                    geom_rgba,
                )
                stats["primitives_created"] += 1
                stats["geoms"] += 1

            elif geom_type == _GEOM_ELLIPSOID:
                # USD doesn't have ellipsoid — approximate as scaled sphere
                _create_usd_sphere(
                    stage,
                    geom_path,
                    geom_size,
                    geom_pos,
                    geom_quat,
                    geom_rgba,
                    ellipsoid_radii=geom_size[:3],
                )
                stats["primitives_created"] += 1
                stats["geoms"] += 1

            elif geom_type == _GEOM_PLANE:
                _create_usd_plane(
                    stage,
                    geom_path,
                    geom_size,
                    geom_pos,
                    geom_quat,
                    geom_rgba,
                )
                stats["primitives_created"] += 1
                stats["geoms"] += 1

            else:
                logger.debug("Skipping geom type %s (%s)", geom_type, safe_geom)
                continue

        # ── Add joints ──
        for i in range(model.njnt):
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if not jnt_name:
                continue

            body_id = int(model.jnt_bodyid[i])
            body_path = body_usd_paths.get(body_id, robot_path)

            safe_jnt = _sanitize_usd_name(jnt_name)
            jnt_path = f"{body_path}/{safe_jnt}"

            jnt_type = int(model.jnt_type[i])
            # MuJoCo joint types: 0=free, 1=ball, 2=slide, 3=hinge

            if jnt_type == 3:  # hinge (revolute)
                jnt_prim = UsdPhysics.RevoluteJoint.Define(stage, jnt_path)

                # Joint axis from MuJoCo
                axis = model.jnt_axis[i]
                _set_joint_axis(jnt_prim, axis)

                # Joint limits
                if model.jnt_limited[i]:
                    lower = float(model.jnt_range[i, 0])
                    upper = float(model.jnt_range[i, 1])
                    jnt_prim.CreateLowerLimitAttr(math.degrees(lower))
                    jnt_prim.CreateUpperLimitAttr(math.degrees(upper))

                # Connect bodies (child body ← joint → parent body)
                parent_body_id = model.body_parentid[body_id]
                parent_path = body_usd_paths.get(parent_body_id, robot_path)
                _set_joint_bodies(jnt_prim, parent_path, body_path)

                stats["joints"] += 1

            elif jnt_type == 2:  # slide (prismatic)
                jnt_prim = UsdPhysics.PrismaticJoint.Define(stage, jnt_path)

                axis = model.jnt_axis[i]
                _set_joint_axis(jnt_prim, axis)

                if model.jnt_limited[i]:
                    # Prismatic limits are in meters (no conversion needed)
                    lower = float(model.jnt_range[i, 0])
                    upper = float(model.jnt_range[i, 1])
                    jnt_prim.CreateLowerLimitAttr(lower)
                    jnt_prim.CreateUpperLimitAttr(upper)

                parent_body_id = model.body_parentid[body_id]
                parent_path = body_usd_paths.get(parent_body_id, robot_path)
                _set_joint_bodies(jnt_prim, parent_path, body_path)

                stats["joints"] += 1

            elif jnt_type == 0:  # free joint
                # Free joints don't need USD representation (floating base)
                logger.debug("Skipping free joint %s", safe_jnt)

            elif jnt_type == 1:  # ball joint (spherical)
                jnt_prim = UsdPhysics.SphericalJoint.Define(stage, jnt_path)

                # Connect bodies
                parent_body_id = model.body_parentid[body_id]
                parent_path = body_usd_paths.get(parent_body_id, robot_path)
                _set_joint_bodies(jnt_prim, parent_path, body_path)

                stats["joints"] += 1

        # ── Add collision filtering ──
        # Apply collision API to all body prims
        for body_id, body_path in body_usd_paths.items():
            prim = stage.GetPrimAtPath(body_path)
            if prim and prim.IsValid():
                UsdPhysics.CollisionAPI.Apply(prim)

        stage.GetRootLayer().Save()

        # Clean up temporary mesh-stripped MJCF
        if tmp_meshstrip_dir and os.path.isdir(tmp_meshstrip_dir):
            shutil.rmtree(tmp_meshstrip_dir, ignore_errors=True)

        file_size_kb = os.path.getsize(output_path) / 1024

        method_desc = (
            "MuJoCo+pxr (kinematic-only, mesh files unavailable)"
            if meshes_stripped
            else "MuJoCo+pxr (full mesh geometry)"
        )

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"✅ MJCF → USD via {method_desc}\n"
                        f"📥 Input: {os.path.basename(mjcf_path)}\n"
                        f"📤 Output: {output_path} ({file_size_kb:.0f} KB)\n"
                        f"🦴 Bodies: {stats['bodies']} | 🔩 Joints: {stats['joints']} | "
                        f"📐 Geoms: {stats['geoms']}\n"
                        f"🔷 Meshes extracted: {stats['meshes_extracted']} | "
                        f"⬡ Primitives: {stats['primitives_created']}"
                    )
                }
            ],
            "usd_path": output_path,
            "method": "mujoco_pxr" + ("_kinematic" if meshes_stripped else ""),
            "stats": stats,
            "meshes_stripped": meshes_stripped,
        }

    except ImportError as e:
        missing = str(e)
        packages = []
        if "pxr" in missing or "Usd" in missing:
            packages.append("pip install usd-core")
        if "mujoco" in missing:
            packages.append("pip install mujoco")

        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"❌ Missing dependency: {missing}\n"
                        f"Install with:\n  " + "\n  ".join(packages)
                        if packages
                        else f"❌ Import error: {missing}"
                    )
                }
            ],
        }
    except Exception as e:
        logger.error("MuJoCo+pxr conversion failed: %s", e, exc_info=True)
        return {
            "status": "error",
            "content": [{"text": f"❌ Manual conversion failed: {e}"}],
        }


# ────────────────────────────────────────────────────────────────
# USD Geometry Helpers
# ────────────────────────────────────────────────────────────────


def _sanitize_usd_name(name: str) -> str:
    """Sanitize a name for use as a USD prim path component."""
    # USD prim names can only contain [a-zA-Z0-9_]
    safe = name.replace("/", "_").replace("-", "_").replace(".", "_").replace(" ", "_")
    # Must start with letter or underscore
    if safe and safe[0].isdigit():
        safe = "_" + safe
    if not safe:
        safe = "_unnamed"
    return safe


def _set_xform_transform(xform, pos, quat) -> None:
    """Set translation and orientation on a UsdGeom.Xformable.

    Uses float precision (Quatf) for the orient op to ensure compatibility
    with all prim types (Mesh, Cube, Sphere, etc.), not just Xform.
    The default AddOrientOp() precision varies by prim type and can cause
    GfQuatf/GfQuatd mismatch errors.

    Args:
        xform: UsdGeom.Xformable (Xform, Mesh, Cube, Sphere, etc.)
        pos: [x, y, z] position
        quat: [w, x, y, z] quaternion (MuJoCo convention)
    """
    from pxr import Gf, UsdGeom

    xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    # MuJoCo quat: (w, x, y, z) → USD Gf.Quatf: (w, x, y, z) — same convention
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    # Only add orient if non-identity
    if not (abs(w - 1.0) < 1e-7 and abs(x) < 1e-7 and abs(y) < 1e-7 and abs(z) < 1e-7):
        # Explicitly use float precision — works on ALL prim types (Mesh, Cube, etc.)
        orient_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
        orient_op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _set_joint_axis(jnt_prim, axis) -> None:
    """Set the joint axis based on MuJoCo's axis vector."""

    ax = [float(axis[0]), float(axis[1]), float(axis[2])]
    # USD joints default to specific axes. We set the axis attribute.
    # For RevoluteJoint/PrismaticJoint, the axis is "X", "Y", or "Z"
    max_idx = max(range(3), key=lambda k: abs(ax[k]))
    axis_map = {0: "X", 1: "Y", 2: "Z"}
    jnt_prim.CreateAxisAttr(axis_map[max_idx])


def _set_joint_bodies(jnt_prim, parent_path: str, child_path: str) -> None:
    """Connect joint to parent and child bodies."""
    from pxr import Sdf

    jnt_prim.CreateBody0Rel().AddTarget(Sdf.Path(parent_path))
    jnt_prim.CreateBody1Rel().AddTarget(Sdf.Path(child_path))


def _create_usd_mesh_from_mujoco(
    stage,
    prim_path: str,
    model,
    mesh_id: int,
    geom_pos,
    geom_quat,
    geom_rgba,
) -> None:
    """Extract mesh geometry from MuJoCo's internal data and create UsdGeom.Mesh.

    MuJoCo stores all mesh data in flat arrays indexed by mesh ID:
        mesh_vertadr[mesh_id] → starting index into mesh_vert
        mesh_vertnum[mesh_id] → number of vertices
        mesh_faceadr[mesh_id] → starting index into mesh_face
        mesh_facenum[mesh_id] → number of triangular faces
        mesh_vert → (total_verts, 3) float array of all vertices
        mesh_face → (total_faces, 3) int array of all face vertex indices
        mesh_normal → (total_verts, 3) float array of all vertex normals
    """
    from pxr import Gf, UsdGeom, Vt

    # Extract vertex and face data for this mesh
    vert_adr = int(model.mesh_vertadr[mesh_id])
    vert_num = int(model.mesh_vertnum[mesh_id])
    face_adr = int(model.mesh_faceadr[mesh_id])
    face_num = int(model.mesh_facenum[mesh_id])

    if vert_num == 0 or face_num == 0:
        logger.warning("Empty mesh data for mesh_id=%s", mesh_id)
        return

    # Slice out this mesh's vertices and faces
    vertices = model.mesh_vert[vert_adr : vert_adr + vert_num].copy()
    faces = model.mesh_face[face_adr : face_adr + face_num].copy()

    # MuJoCo face indices are relative to the mesh's vertex start (already 0-based)
    # Each face is a triangle (3 indices)

    # Extract normals if available
    normals = None
    if hasattr(model, "mesh_normal") and model.mesh_normal.shape[0] > 0:
        normals = model.mesh_normal[vert_adr : vert_adr + vert_num].copy()

    # Create UsdGeom.Mesh
    mesh_prim = UsdGeom.Mesh.Define(stage, prim_path)

    # Set mesh points (vertices)
    points = Vt.Vec3fArray(
        [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
    )
    mesh_prim.CreatePointsAttr(points)

    # Set face vertex counts (all triangles = 3)
    face_counts = Vt.IntArray([3] * face_num)
    mesh_prim.CreateFaceVertexCountsAttr(face_counts)

    # Set face vertex indices (flatten the Nx3 face array)
    face_indices = Vt.IntArray(faces.flatten().tolist())
    mesh_prim.CreateFaceVertexIndicesAttr(face_indices)

    # Set normals if available
    if normals is not None and len(normals) == vert_num:
        normal_array = Vt.Vec3fArray(
            [Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in normals]
        )
        mesh_prim.CreateNormalsAttr(normal_array)
        mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    # Set subdivision scheme to none (we have exact triangulated geometry)
    mesh_prim.CreateSubdivisionSchemeAttr("none")

    # Set geom transform (position relative to body)
    xform = UsdGeom.Xformable(mesh_prim)
    _set_xform_transform(xform, geom_pos, geom_quat)

    # Set display color from MuJoCo rgba
    if geom_rgba is not None:
        r, g, b, a = (
            float(geom_rgba[0]),
            float(geom_rgba[1]),
            float(geom_rgba[2]),
            float(geom_rgba[3]),
        )
        if a > 0:  # Only set color if not fully transparent
            mesh_prim.CreateDisplayColorAttr([Gf.Vec3f(r, g, b)])
            if a < 1.0:
                mesh_prim.CreateDisplayOpacityAttr([a])


def _create_usd_cube(
    stage,
    prim_path: str,
    size,
    pos,
    quat,
    rgba,
) -> None:
    """Create a UsdGeom.Cube from MuJoCo box geom.

    MuJoCo box size = half-extents [hx, hy, hz].
    USD Cube has size=2.0 centered at origin, scaled by the half-extents.
    """
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, prim_path)
    # USD Cube has default size 2 (from -1 to 1), so we set size to 2*half_extent
    cube.CreateSizeAttr(2.0)

    xform = UsdGeom.Xformable(cube)
    _set_xform_transform(xform, pos, quat)
    # Scale by half-extents (MuJoCo size = half-size)
    xform.AddScaleOp().Set(Gf.Vec3f(float(size[0]), float(size[1]), float(size[2])))

    if rgba is not None:
        r, g, b, a = float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])
        if a > 0:
            cube.CreateDisplayColorAttr([Gf.Vec3f(r, g, b)])


def _create_usd_sphere(
    stage,
    prim_path: str,
    size,
    pos,
    quat,
    rgba,
    ellipsoid_radii=None,
) -> None:
    """Create a UsdGeom.Sphere from MuJoCo sphere/ellipsoid geom.

    MuJoCo sphere size[0] = radius.
    MuJoCo ellipsoid size[0:3] = semi-axes (x, y, z radii).
    """
    from pxr import Gf, UsdGeom

    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    radius = float(size[0])
    sphere.CreateRadiusAttr(radius)

    xform = UsdGeom.Xformable(sphere)
    _set_xform_transform(xform, pos, quat)

    if ellipsoid_radii is not None:
        # Approximate ellipsoid as non-uniformly scaled sphere
        # Normalize by the largest radius so the sphere radius is correct
        max_r = max(
            float(ellipsoid_radii[0]),
            float(ellipsoid_radii[1]),
            float(ellipsoid_radii[2]),
        )
        if max_r > 0:
            sphere.CreateRadiusAttr(max_r)
            sx = float(ellipsoid_radii[0]) / max_r
            sy = float(ellipsoid_radii[1]) / max_r
            sz = float(ellipsoid_radii[2]) / max_r
            xform.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))

    if rgba is not None:
        r, g, b, a = float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])
        if a > 0:
            sphere.CreateDisplayColorAttr([Gf.Vec3f(r, g, b)])


def _create_usd_cylinder(
    stage,
    prim_path: str,
    size,
    pos,
    quat,
    rgba,
) -> None:
    """Create a UsdGeom.Cylinder from MuJoCo cylinder geom.

    MuJoCo cylinder: size[0] = radius, size[1] = half-length.
    USD Cylinder: radius + height, centered at origin, along Y axis.
    """
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, prim_path)
    cyl.CreateRadiusAttr(float(size[0]))
    cyl.CreateHeightAttr(float(size[1]) * 2.0)  # MuJoCo half-length → full height
    cyl.CreateAxisAttr("Z")  # MuJoCo cylinders are along Z by default

    xform = UsdGeom.Xformable(cyl)
    _set_xform_transform(xform, pos, quat)

    if rgba is not None:
        r, g, b, a = float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])
        if a > 0:
            cyl.CreateDisplayColorAttr([Gf.Vec3f(r, g, b)])


def _create_usd_capsule(
    stage,
    prim_path: str,
    size,
    pos,
    quat,
    rgba,
) -> None:
    """Create a UsdGeom.Capsule from MuJoCo capsule geom.

    MuJoCo capsule: size[0] = radius, size[1] = half-length (cylinder part).
    USD Capsule: radius + height (full cylinder part), along configurable axis.
    """
    from pxr import Gf, UsdGeom

    cap = UsdGeom.Capsule.Define(stage, prim_path)
    cap.CreateRadiusAttr(float(size[0]))
    cap.CreateHeightAttr(float(size[1]) * 2.0)  # half-length → full height
    cap.CreateAxisAttr("Z")

    xform = UsdGeom.Xformable(cap)
    _set_xform_transform(xform, pos, quat)

    if rgba is not None:
        r, g, b, a = float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])
        if a > 0:
            cap.CreateDisplayColorAttr([Gf.Vec3f(r, g, b)])


def _create_usd_plane(
    stage,
    prim_path: str,
    size,
    pos,
    quat,
    rgba,
) -> None:
    """Create a large quad mesh for MuJoCo plane geom.

    MuJoCo plane: size[0] = half_x, size[1] = half_y, size[2] = spacing (unused).
    If size is zero, uses a large default plane (50×50 meters).

    USD has no native plane primitive, so we create a single-quad UsdGeom.Mesh.
    """
    from pxr import Gf, UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, prim_path)

    hx = float(size[0]) if size[0] > 0 else 50.0
    hy = float(size[1]) if size[1] > 0 else 50.0

    points = Vt.Vec3fArray(
        [
            Gf.Vec3f(-hx, -hy, 0),
            Gf.Vec3f(hx, -hy, 0),
            Gf.Vec3f(hx, hy, 0),
            Gf.Vec3f(-hx, hy, 0),
        ]
    )
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))

    # Set normals (single upward-facing normal for the quad)
    normals = Vt.Vec3fArray(
        [
            Gf.Vec3f(0, 0, 1),
            Gf.Vec3f(0, 0, 1),
            Gf.Vec3f(0, 0, 1),
            Gf.Vec3f(0, 0, 1),
        ]
    )
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    # Set transform
    xform = UsdGeom.Xformable(mesh)
    _set_xform_transform(xform, pos, quat)

    if rgba is not None:
        r, g, b, a = float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])
        if a > 0:
            mesh.CreateDisplayColorAttr([Gf.Vec3f(r, g, b)])


# ────────────────────────────────────────────────────────────────
# Batch Conversion
# ────────────────────────────────────────────────────────────────


def convert_all_robots_to_usd(
    output_dir: Optional[str] = None,
    method: str = "auto",
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """Batch convert all strands-robots MJCF models to USD format.

    Iterates over all 32 bundled robots and converts each MJCF → USD.
    Uses the dual-pipeline approach: Isaac Sim if available, MuJoCo+pxr fallback.

    Args:
        output_dir: Base directory for USD outputs (default: ~/.strands_robots/isaac_cache)
        method: "auto" (try Isaac Sim first), "isaac_sim", or "mujoco_pxr"
        skip_existing: Skip robots that already have a cached USD

    Returns:
        Dict with per-robot results and summary statistics

    Example:
        results = convert_all_robots_to_usd("/data/usd_robots")
        print(f"Converted {results['summary']['success']}/{results['summary']['total']} robots")
    """
    try:
        from strands_robots.assets import list_available_robots, resolve_model_path
    except ImportError:
        return {
            "status": "error",
            "content": [{"text": "❌ strands_robots.assets not available"}],
        }

    if output_dir is None:
        output_dir = str(Path.home() / ".strands_robots" / "isaac_cache")

    os.makedirs(output_dir, exist_ok=True)

    robots = list_available_robots()
    results = {}
    summary = {"total": 0, "success": 0, "skipped": 0, "failed": 0}

    for robot in robots:
        name = robot["name"] if isinstance(robot, dict) else robot
        summary["total"] += 1

        usd_output = os.path.join(output_dir, name, f"{name}.usd")

        if skip_existing and os.path.exists(usd_output):
            results[name] = {"status": "skipped", "usd_path": usd_output}
            summary["skipped"] += 1
            logger.info("⏭️ Skipping %s (already exists)", name)
            continue

        mjcf_path = resolve_model_path(name)
        if not mjcf_path or not mjcf_path.exists():
            results[name] = {"status": "error", "reason": "MJCF not found"}
            summary["failed"] += 1
            continue

        os.makedirs(os.path.dirname(usd_output), exist_ok=True)

        try:
            if method == "mujoco_pxr":
                result = _manual_mjcf_to_usd(str(mjcf_path), usd_output)
            else:
                result = convert_mjcf_to_usd(str(mjcf_path), usd_output)

            results[name] = result

            if result.get("status") == "success":
                summary["success"] += 1
                logger.info("✅ %s: %s", name, result.get('method', 'unknown'))
            else:
                summary["failed"] += 1
                logger.warning("❌ %s: %s", name, result)

        except Exception as e:
            results[name] = {"status": "error", "reason": str(e)}
            summary["failed"] += 1
            logger.error("❌ %s: %s", name, e)

    return {
        "status": "success",
        "content": [
            {
                "text": (
                    f"🔄 Batch MJCF → USD conversion complete\n"
                    f"✅ Success: {summary['success']}/{summary['total']}\n"
                    f"⏭️ Skipped: {summary['skipped']}\n"
                    f"❌ Failed: {summary['failed']}\n"
                    f"📂 Output: {output_dir}"
                )
            }
        ],
        "results": results,
        "summary": summary,
        "output_dir": output_dir,
    }


# ────────────────────────────────────────────────────────────────
# AssetConverter Class (OOP Wrapper)
# ────────────────────────────────────────────────────────────────


class AssetConverter:
    """Convenience wrapper around MJCF ↔ USD conversion functions.

    Provides an object-oriented interface for converting robot assets between
    MuJoCo MJCF and Universal Scene Description (USD) formats.

    Examples:
        converter = AssetConverter()
        result = converter.mjcf_to_usd("/path/to/robot.xml")
        result = converter.usd_to_mjcf("/path/to/robot.usd")
        assets = converter.list_convertible()
        batch = converter.convert_all("/data/usd_output")
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir

    def mjcf_to_usd(
        self,
        mjcf_path: str,
        output_path: Optional[str] = None,
        fix_base: bool = True,
        make_instanceable: bool = False,
    ) -> Dict[str, Any]:
        """Convert MJCF to USD. See :func:`convert_mjcf_to_usd`."""
        return convert_mjcf_to_usd(
            mjcf_path,
            output_path=output_path,
            fix_base=fix_base,
            make_instanceable=make_instanceable,
        )

    def usd_to_mjcf(
        self,
        usd_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert USD to MJCF. See :func:`convert_usd_to_mjcf`."""
        return convert_usd_to_mjcf(usd_path, output_path=output_path)

    def convert_all(
        self,
        output_dir: Optional[str] = None,
        method: str = "auto",
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """Batch convert all robots. See :func:`convert_all_robots_to_usd`."""
        return convert_all_robots_to_usd(
            output_dir=output_dir or self.cache_dir,
            method=method,
            skip_existing=skip_existing,
        )

    def batch_convert(
        self,
        output_dir: Optional[str] = None,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """Batch-convert all robots to USD. Alias for :meth:`convert_all`.

        See :func:`convert_all_robots_to_usd`.
        """
        return self.convert_all(
            output_dir=output_dir,
            skip_existing=skip_existing,
        )

    def list_convertible(self) -> Dict[str, Any]:
        """List assets that can be converted. See :func:`list_convertible_assets`."""
        return list_convertible_assets()

    def __repr__(self) -> str:
        return f"AssetConverter(cache_dir={self.cache_dir!r})"


def list_convertible_assets() -> Dict[str, Any]:
    """List strands-robots assets that can be converted to USD.

    Shows all MJCF robots from our asset manager that could be used
    in Isaac Sim after conversion.
    """
    try:
        from strands_robots.assets import list_available_robots, resolve_model_path

        robots = list_available_robots()
        convertible = []

        for robot in robots:
            name = robot["name"] if isinstance(robot, dict) else robot
            path = resolve_model_path(name)

            if path and path.exists():
                # Check if already converted
                cache_dir = Path.home() / ".strands_robots" / "isaac_cache"
                cached = cache_dir / f"{name}.usd"

                # Detect mesh format(s)
                mesh_formats = _detect_mesh_formats(str(path))

                convertible.append(
                    {
                        "name": name,
                        "mjcf_path": str(path),
                        "has_usd": cached.exists(),
                        "usd_path": str(cached) if cached.exists() else None,
                        "mesh_formats": mesh_formats,
                    }
                )

        return {
            "status": "success",
            "content": [
                {"text": f"🔄 {len(convertible)} robots convertible MJCF → USD"}
            ],
            "robots": convertible,
        }

    except ImportError:
        return {
            "status": "error",
            "content": [{"text": "❌ strands_robots.assets not available"}],
        }


def _detect_mesh_formats(mjcf_path: str) -> List[str]:
    """Detect which mesh file formats a MJCF file references."""
    import xml.etree.ElementTree as ET

    formats = set()
    try:
        # Parse the MJCF and any included files
        tree = ET.parse(mjcf_path)
        for mesh_elem in tree.iter("mesh"):
            mesh_file = mesh_elem.get("file", "")
            if "." in mesh_file:
                ext = mesh_file.rsplit(".", 1)[1].lower()
                formats.add(ext)
    except Exception:
        pass

    return sorted(formats)
