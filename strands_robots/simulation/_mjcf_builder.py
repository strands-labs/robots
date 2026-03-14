"""MJCF XML builder — programmatic scene construction."""

import logging
import os
import tempfile
from typing import Dict

from strands_robots.simulation._backend import _ensure_mujoco
from strands_robots.simulation._models import SimCamera, SimObject, SimRobot, SimWorld

logger = logging.getLogger(__name__)


class MJCFBuilder:
    """Builds MuJoCo MJCF XML from SimWorld state."""

    @staticmethod
    def build_objects_only(world: SimWorld) -> str:
        """Build MJCF XML for a world with only objects (robots loaded separately)."""
        _ensure_mujoco()

        parts = []
        parts.append('<mujoco model="strands_sim">')
        parts.append('  <compiler angle="radian" autolimits="true"/>')

        gx, gy, gz = world.gravity
        parts.append(f'  <option timestep="{world.timestep}" gravity="{gx} {gy} {gz}"/>')

        parts.append("  <visual>")
        parts.append('    <global offwidth="1280" offheight="960"/>')
        parts.append('    <quality shadowsize="4096"/>')
        parts.append("  </visual>")

        parts.append("  <asset>")
        parts.append(
            '    <texture type="2d" name="grid_tex" builtin="checker" '
            'width="512" height="512" rgb1=".9 .9 .9" rgb2=".7 .7 .7"/>'
        )
        parts.append('    <material name="grid_mat" texture="grid_tex" texrepeat="8 8" reflectance="0.1"/>')
        for obj in world.objects.values():
            if obj.shape == "mesh" and obj.mesh_path:
                parts.append(f'    <mesh name="mesh_{obj.name}" file="{obj.mesh_path}"/>')
        parts.append("  </asset>")

        parts.append("  <worldbody>")
        parts.append('    <light name="main_light" pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" specular="0.3 0.3 0.3"/>')
        parts.append('    <light name="fill_light" pos="1 1 2" dir="-0.5 -0.5 -1" diffuse="0.5 0.5 0.5"/>')

        if world.ground_plane:
            parts.append(
                '    <geom name="ground" type="plane" size="5 5 0.01" material="grid_mat" conaffinity="1" condim="3"/>'
            )

        for cam in world.cameras.values():
            px, py, pz = cam.position
            parts.append(f'    <camera name="{cam.name}" pos="{px} {py} {pz}" fovy="{cam.fov}" mode="fixed"/>')

        for obj in world.objects.values():
            parts.append(MJCFBuilder._object_xml(obj, indent=4))

        parts.append("  </worldbody>")
        parts.append("</mujoco>")

        return "\n".join(parts)

    @staticmethod
    def _object_xml(obj: SimObject, indent: int = 4) -> str:
        """Generate MJCF XML for a single object."""
        pad = " " * indent
        px, py, pz = obj.position
        qw, qx, qy, qz = obj.orientation
        r, g, b, a = obj.color
        lines = []

        lines.append(f'{pad}<body name="{obj.name}" pos="{px} {py} {pz}" quat="{qw} {qx} {qy} {qz}">')

        if not obj.is_static:
            lines.append(f'{pad}  <freejoint name="{obj.name}_joint"/>')
            lines.append(f'{pad}  <inertial pos="0 0 0" mass="{obj.mass}" diaginertia="0.001 0.001 0.001"/>')

        if obj.shape == "box":
            sx, sy, sz = [s / 2 for s in obj.size]
            lines.append(
                f'{pad}  <geom name="{obj.name}_geom" type="box" size="{sx} {sy} {sz}" '
                f'rgba="{r} {g} {b} {a}" condim="3" friction="1 0.5 0.001"/>'
            )
        elif obj.shape == "sphere":
            radius = obj.size[0] / 2 if obj.size else 0.025
            lines.append(
                f'{pad}  <geom name="{obj.name}_geom" type="sphere" size="{radius}" rgba="{r} {g} {b} {a}" condim="3"/>'
            )
        elif obj.shape == "cylinder":
            radius = obj.size[0] / 2 if obj.size else 0.025
            half_h = obj.size[2] / 2 if len(obj.size) > 2 else 0.05
            lines.append(
                f'{pad}  <geom name="{obj.name}_geom" type="cylinder" size="{radius} {half_h}" '
                f'rgba="{r} {g} {b} {a}" condim="3"/>'
            )
        elif obj.shape == "capsule":
            radius = obj.size[0] / 2 if obj.size else 0.025
            half_h = obj.size[2] / 2 if len(obj.size) > 2 else 0.05
            lines.append(
                f'{pad}  <geom name="{obj.name}_geom" type="capsule" size="{radius} {half_h}" '
                f'rgba="{r} {g} {b} {a}" condim="3"/>'
            )
        elif obj.shape == "mesh" and obj.mesh_path:
            lines.append(
                f'{pad}  <geom name="{obj.name}_geom" type="mesh" mesh="mesh_{obj.name}" '
                f'rgba="{r} {g} {b} {a}" condim="3"/>'
            )
        elif obj.shape == "plane":
            sx = obj.size[0] if obj.size else 1.0
            sy = obj.size[1] if len(obj.size) > 1 else sx
            lines.append(
                f'{pad}  <geom name="{obj.name}_geom" type="plane" size="{sx} {sy} 0.01" rgba="{r} {g} {b} {a}"/>'
            )

        lines.append(f"{pad}</body>")
        return "\n".join(lines)

    @staticmethod
    def compose_multi_robot_scene(
        robots: Dict[str, SimRobot],
        objects: Dict[str, SimObject],
        cameras: Dict[str, SimCamera],
        world: SimWorld,
    ) -> str:
        """Compose a multi-robot scene by merging URDF-derived MJCF fragments."""
        mj = _ensure_mujoco()
        world._tmpdir = tempfile.TemporaryDirectory(prefix="strands_sim_")
        tmpdir = world._tmpdir.name

        robot_xmls = {}
        for robot_name, robot in robots.items():
            try:
                model = mj.MjModel.from_xml_path(str(robot.urdf_path))
                robot_xml_path = os.path.join(tmpdir, f"{robot_name}.xml")
                mj.mj_saveLastXML(robot_xml_path, model)
                robot_xmls[robot_name] = robot_xml_path
                logger.debug("Converted %s → %s", robot.urdf_path, robot_xml_path)
            except Exception as e:
                logger.error("Failed to convert URDF for '%s': %s", robot_name, e)
                raise

        parts = []
        parts.append('<mujoco model="strands_sim_multi">')
        parts.append('  <compiler angle="radian" autolimits="true" meshdir="."/>')

        gx, gy, gz = world.gravity
        parts.append(f'  <option timestep="{world.timestep}" gravity="{gx} {gy} {gz}"/>')

        parts.append("  <visual>")
        parts.append('    <global offwidth="1280" offheight="960"/>')
        parts.append('    <quality shadowsize="4096"/>')
        parts.append("  </visual>")

        parts.append("  <asset>")
        parts.append(
            '    <texture type="2d" name="grid_tex" builtin="checker" '
            'width="512" height="512" rgb1=".9 .9 .9" rgb2=".7 .7 .7"/>'
        )
        parts.append('    <material name="grid_mat" texture="grid_tex" texrepeat="8 8" reflectance="0.1"/>')
        for obj in objects.values():
            if obj.shape == "mesh" and obj.mesh_path:
                parts.append(f'    <mesh name="mesh_{obj.name}" file="{obj.mesh_path}"/>')
        parts.append("  </asset>")

        parts.append("  <worldbody>")
        parts.append('    <light name="main_light" pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" specular="0.3 0.3 0.3"/>')
        parts.append('    <light name="fill_light" pos="1 1 2" dir="-0.5 -0.5 -1" diffuse="0.5 0.5 0.5"/>')

        if world.ground_plane:
            parts.append(
                '    <geom name="ground" type="plane" size="5 5 0.01" material="grid_mat" conaffinity="1" condim="3"/>'
            )

        for cam in cameras.values():
            px, py, pz = cam.position
            parts.append(f'    <camera name="{cam.name}" pos="{px} {py} {pz}" fovy="{cam.fov}" mode="fixed"/>')

        for robot_name, robot in robots.items():
            xml_path = robot_xmls[robot_name]
            parts.append(f"    <!-- Robot: {robot_name} -->")
            parts.append(f'    <include file="{xml_path}"/>')

        for obj in objects.values():
            parts.append(MJCFBuilder._object_xml(obj, indent=4))

        parts.append("  </worldbody>")
        parts.append("</mujoco>")

        master_xml = "\n".join(parts)
        master_path = os.path.join(tmpdir, "master_scene.xml")
        with open(master_path, "w") as f:
            f.write(master_xml)

        return master_path
