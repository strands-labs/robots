#!/usr/bin/env python3
"""
Robot Asset Manager for Strands Robots Simulation.

Provides auto-resolution of robot model files (MJCF XML) from:
1. Bundled assets (strands_robots/assets/ — from MuJoCo Menagerie)
2. Custom paths (STRANDS_URDF_DIR env var)
3. User home (~/.strands_robots/assets/)

32 robots are bundled (28 from MuJoCo Menagerie + 3 community):
- Arms: SO-100, SO-101, Koch v1.1, Franka Panda, FR3, UR5e, KUKA iiwa,
         Kinova Gen3, xArm7, Trossen VX300s, ARX L5, AgileX Piper, Unitree Z1
- Bimanual: ALOHA, Trossen WidowX AI
- Hands: Shadow Hand, LEAP Hand, Robotiq 2F-85
- Humanoids: Fourier N1, Unitree G1, H1, Apptronik Apollo, Agility Cassie,
         Open Duck Mini V2, Asimov V0
- Expressive: Pollen Reachy Mini
- Mobile: Unitree Go2, A1, Boston Dynamics Spot, Hello Robot Stretch 3
- Mobile Manip: Google Robot

"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Asset directory resolution
# ─────────────────────────────────────────────────────────────────────

_ASSETS_DIR = Path(__file__).parent


def get_assets_dir() -> Path:
    """Get the path to the bundled assets directory."""
    return _ASSETS_DIR


def get_search_paths() -> List[Path]:
    """Get ordered list of asset search paths."""
    paths = [_ASSETS_DIR]

    # Custom paths from env
    custom = os.getenv("STRANDS_URDF_DIR") or os.getenv("STRANDS_ASSETS_DIR")
    if custom:
        for p in custom.split(":"):
            paths.append(Path(p))

    # User home
    paths.append(Path.home() / ".strands_robots" / "assets")

    # CWD
    paths.append(Path.cwd() / "assets")

    return paths


# ─────────────────────────────────────────────────────────────────────
# Robot Model Registry
# ─────────────────────────────────────────────────────────────────────
# Maps friendly names → (menagerie_dir, model_xml, scene_xml, description)

_ROBOT_MODELS: Dict[str, Dict] = {
    # ── Arms ──
    "so100": {
        "dir": "trs_so_arm100",
        "model_xml": "so_arm100.xml",
        "scene_xml": "scene.xml",
        "description": "TrossenRobotics SO-ARM100 (6-DOF, Feetech servos)",
        "joints": 13,
        "category": "arm",
    },
    "so101": {
        "dir": "robotstudio_so101",
        "model_xml": "so101.xml",
        "scene_xml": "scene_box.xml",
        "description": "RobotStudio SO-101 (6-DOF, upgraded SO-100)",
        "joints": 9,
        "category": "arm",
    },
    "koch": {
        "dir": "low_cost_robot_arm",
        "model_xml": "low_cost_robot_arm.xml",
        "scene_xml": "scene.xml",
        "description": "Koch v1.1 Low Cost Robot Arm (6-DOF, Dynamixel)",
        "joints": 7,
        "category": "arm",
    },
    "panda": {
        "dir": "franka_emika_panda",
        "model_xml": "panda.xml",
        "scene_xml": "scene.xml",
        "description": "Franka Emika Panda (7-DOF + gripper)",
        "joints": 7,
        "category": "arm",
    },
    "fr3": {
        "dir": "franka_fr3",
        "model_xml": "fr3.xml",
        "scene_xml": "scene.xml",
        "description": "Franka Research 3 (7-DOF + gripper)",
        "joints": 8,
        "category": "arm",
    },
    "ur5e": {
        "dir": "universal_robots_ur5e",
        "model_xml": "ur5e.xml",
        "scene_xml": "scene.xml",
        "description": "Universal Robots UR5e (6-DOF industrial)",
        "joints": 8,
        "category": "arm",
    },
    "kuka_iiwa": {
        "dir": "kuka_iiwa_14",
        "model_xml": "iiwa14.xml",
        "scene_xml": "scene.xml",
        "description": "KUKA LBR iiwa 14 (7-DOF collaborative)",
        "joints": 11,
        "category": "arm",
    },
    "kinova_gen3": {
        "dir": "kinova_gen3",
        "model_xml": "gen3.xml",
        "scene_xml": "scene.xml",
        "description": "Kinova Gen3 (7-DOF lightweight)",
        "joints": 7,
        "category": "arm",
    },
    "xarm7": {
        "dir": "ufactory_xarm7",
        "model_xml": "xarm7.xml",
        "scene_xml": "scene.xml",
        "description": "UFactory xArm 7 (7-DOF + gripper)",
        "joints": 13,
        "category": "arm",
    },
    "vx300s": {
        "dir": "trossen_vx300s",
        "model_xml": "vx300s.xml",
        "scene_xml": "scene.xml",
        "description": "Trossen ViperX 300s (6-DOF + gripper)",
        "joints": 19,
        "category": "arm",
    },
    "arx_l5": {
        "dir": "arx_l5",
        "model_xml": "arx_l5.xml",
        "scene_xml": "scene.xml",
        "description": "ARX L5 (6-DOF lightweight arm)",
        "joints": 11,
        "category": "arm",
    },
    "piper": {
        "dir": "agilex_piper",
        "model_xml": "piper.xml",
        "scene_xml": "scene.xml",
        "description": "AgileX Piper (6-DOF + gripper)",
        "joints": 11,
        "category": "arm",
    },
    "z1": {
        "dir": "unitree_z1",
        "model_xml": "z1_gripper.xml",
        "scene_xml": "scene.xml",
        "description": "Unitree Z1 (6-DOF + gripper)",
        "joints": 8,
        "category": "arm",
    },
    "openarm": {
        "dir": "enactic_openarm",
        "model_xml": "openarm.xml",
        "scene_xml": "scene.xml",
        "description": "Enactic OpenArm (7-DOF, DAMIAO motors, CAN bus)",
        "joints": 9,
        "category": "arm",
    },
    # ── Bimanual ──
    "aloha": {
        "dir": "aloha",
        "model_xml": "aloha.xml",
        "scene_xml": "scene.xml",
        "description": "ALOHA Bimanual (2× ViperX 300s, 14-DOF + 2 grippers)",
        "joints": 28,
        "category": "bimanual",
    },
    "trossen_wxai": {
        "dir": "trossen_wxai",
        "model_xml": "trossen_ai_bimanual.xml",
        "scene_xml": "scene.xml",
        "description": "Trossen WidowX AI Bimanual",
        "joints": 17,
        "category": "bimanual",
    },
    # ── Hands ──
    "shadow_hand": {
        "dir": "shadow_hand",
        "model_xml": "left_hand.xml",
        "scene_xml": "scene_left.xml",
        "description": "Shadow Dexterous Hand (24-DOF)",
        "joints": 45,
        "category": "hand",
    },
    "leap_hand": {
        "dir": "leap_hand",
        "model_xml": "left_hand.xml",
        "scene_xml": "scene_left.xml",
        "description": "LEAP Hand (16-DOF dexterous)",
        "joints": 41,
        "category": "hand",
    },
    "robotiq_2f85": {
        "dir": "robotiq_2f85",
        "model_xml": "2f85.xml",
        "scene_xml": "scene.xml",
        "description": "Robotiq 2F-85 Gripper (2-finger adaptive)",
        "joints": 16,
        "category": "hand",
    },
    # ── Humanoids ──
    "fourier_n1": {
        "dir": "fourier_n1",
        "model_xml": "n1.xml",
        "scene_xml": "scene.xml",
        "description": "Fourier N1 / GR-1 Humanoid (26-DOF)",
        "joints": 26,
        "category": "humanoid",
    },
    "unitree_g1": {
        "dir": "unitree_g1",
        "model_xml": "g1.xml",
        "scene_xml": "scene.xml",
        "description": "Unitree G1 Humanoid (29-DOF + dexterous hands)",
        "joints": 46,
        "category": "humanoid",
    },
    "unitree_h1": {
        "dir": "unitree_h1",
        "model_xml": "h1.xml",
        "scene_xml": "scene.xml",
        "description": "Unitree H1 Humanoid (19-DOF)",
        "joints": 20,
        "category": "humanoid",
    },
    "apollo": {
        "dir": "apptronik_apollo",
        "model_xml": "apptronik_apollo.xml",
        "scene_xml": "scene.xml",
        "description": "Apptronik Apollo Humanoid (34-DOF)",
        "joints": 34,
        "category": "humanoid",
    },
    "cassie": {
        "dir": "agility_cassie",
        "model_xml": "cassie.xml",
        "scene_xml": "scene.xml",
        "description": "Agility Cassie Bipedal Robot",
        "joints": 28,
        "category": "humanoid",
    },
    "open_duck_mini": {
        "dir": "open_duck_mini_v2",
        "model_xml": "open_duck_mini_v2.xml",
        "scene_xml": "scene.xml",
        "description": "Open Duck Mini V2 (16-DOF expressive biped, Feetech servos)",
        "joints": 16,
        "category": "humanoid",
    },
    "asimov_v0": {
        "dir": "asimov_v0",
        "model_xml": "asimov_v0.xml",
        "scene_xml": "scene.xml",
        "description": "Asimov V0 Bipedal Legs (12-DOF + 2 passive toes)",
        "joints": 15,
        "category": "humanoid",
    },
    # ── Expressive ──
    "reachy_mini": {
        "dir": "reachy_mini",
        "model_xml": "mjcf/reachy_mini.xml",
        "scene_xml": "mjcf/scene.xml",
        "description": "Pollen Reachy Mini (6-DOF Stewart head + antennas, 9 actuators)",
        "joints": 21,
        "category": "expressive",
    },
    # ── Mobile ──
    "unitree_go2": {
        "dir": "unitree_go2",
        "model_xml": "go2.xml",
        "scene_xml": "scene.xml",
        "description": "Unitree Go2 Quadruped",
        "joints": 40,
        "category": "mobile",
    },
    "unitree_a1": {
        "dir": "unitree_a1",
        "model_xml": "a1.xml",
        "scene_xml": "scene.xml",
        "description": "Unitree A1 Quadruped",
        "joints": 16,
        "category": "mobile",
    },
    "spot": {
        "dir": "boston_dynamics_spot",
        "model_xml": "spot_arm.xml",
        "scene_xml": "scene_arm.xml",
        "description": "Boston Dynamics Spot (with arm)",
        "joints": 20,
        "category": "mobile",
    },
    "stretch3": {
        "dir": "hello_robot_stretch_3",
        "model_xml": "stretch.xml",
        "scene_xml": "scene.xml",
        "description": "Hello Robot Stretch 3 (mobile manipulator)",
        "joints": 41,
        "category": "mobile",
    },
    # ── Mobile Manipulation ──
    "google_robot": {
        "dir": "google_robot",
        "model_xml": "robot.xml",
        "scene_xml": "scene.xml",
        "description": "Google Robot (mobile base + arm, RT-X)",
        "joints": 10,
        "category": "mobile_manip",
    },
}

# ── Aliases (multiple names resolve to the same robot) ──
_ALIASES: Dict[str, str] = {
    # SO-100 variants (all use same model, different data configs)
    "so100_dualcam": "so100",
    "so100_4cam": "so100",
    "so_arm100": "so100",
    "trs_so_arm100": "so100",
    # SO-101
    "so101_follower": "so101",
    "so101_leader": "so101",
    "robotstudio_so101": "so101",
    # Panda variants
    "franka": "panda",
    "franka_panda": "panda",
    "franka_emika_panda": "panda",
    "bimanual_panda_gripper": "panda",  # data config → uses same panda model
    "libero_panda": "panda",
    # Franka FR3
    "franka_fr3": "fr3",
    "franka_fr3_v2": "fr3",
    # Fourier
    "fourier_gr1_arms_only": "fourier_n1",
    "fourier_gr1": "fourier_n1",
    "gr1": "fourier_n1",
    # Unitree
    "unitree_g1_locomanip": "unitree_g1",
    "g1": "unitree_g1",
    "h1": "unitree_h1",
    "go2": "unitree_go2",
    "a1": "unitree_a1",
    "unitree_z1": "z1",
    # Others
    "oxe_droid": "google_robot",  # OXE DROID → closest is Google Robot
    # "galaxea_r1_pro": "panda",  # Removed: silent fallback to Panda is misleading (see PR #9 review)
    "koch_v1.1": "koch",
    "low_cost_robot_arm": "koch",
    "viper_x300s": "vx300s",
    "trossen_vx300s": "vx300s",
    "trossen_ai_bimanual": "trossen_wxai",
    "ufactory_xarm7": "xarm7",
    "kuka_iiwa_14": "kuka_iiwa",
    "boston_dynamics_spot": "spot",
    "hello_robot_stretch": "stretch3",
    "hello_robot_stretch_3": "stretch3",
    "apptronik_apollo": "apollo",
    "agility_cassie": "cassie",
    "shadow_dexee": "shadow_hand",
    "robotiq": "robotiq_2f85",
    "robotiq_2f85_v4": "robotiq_2f85",
    "agilex_piper": "piper",
    # Open Duck Mini
    "open_duck": "open_duck_mini",
    "open_duck_mini_v2": "open_duck_mini",
    "open_duck_v2": "open_duck_mini",
    "mini_bdx": "open_duck_mini",
    "bdx": "open_duck_mini",
    # Asimov
    "asimov": "asimov_v0",
    # Reachy Mini
    "reachy": "reachy_mini",
    "pollen_reachy_mini": "reachy_mini",
    "reachy-mini": "reachy_mini",
    "reachymini": "reachy_mini",
    # OpenArm
    "enactic_openarm": "openarm",
    "open_arm": "openarm",
    "openarm_v10": "openarm",
}


def resolve_robot_name(name: str) -> str:
    """Resolve a robot name through aliases to canonical name."""
    name = name.lower().strip()
    return _ALIASES.get(name, name)


def resolve_model_path(
    name: str,
    prefer_scene: bool = False,
) -> Optional[Path]:
    """Resolve a robot name to its MJCF model XML path.

    Args:
        name: Robot name (canonical or alias, e.g. "so100", "panda", "unitree_g1")
        prefer_scene: If True, return scene XML instead of model XML

    Returns:
        Path to the MJCF XML file, or None if not found
    """
    canonical = resolve_robot_name(name)

    if canonical not in _ROBOT_MODELS:
        logger.warning(f"Unknown robot: {name} (resolved to: {canonical})")
        return None

    info = _ROBOT_MODELS[canonical]
    xml_file = info["scene_xml"] if prefer_scene else info["model_xml"]

    # Search all paths
    for search_dir in get_search_paths():
        model_path = search_dir / info["dir"] / xml_file
        if model_path.exists():
            logger.debug(f"Resolved {name} → {model_path}")
            return model_path

    logger.warning(f"Robot model not found: {name} → {info['dir']}/{xml_file}")
    logger.info(f"Expected at: {_ASSETS_DIR / info['dir'] / xml_file}")
    return None


def resolve_model_dir(name: str) -> Optional[Path]:
    """Resolve a robot name to its asset directory (containing XML + meshes).

    Args:
        name: Robot name (canonical or alias)

    Returns:
        Path to the robot's asset directory, or None if not found
    """
    canonical = resolve_robot_name(name)

    if canonical not in _ROBOT_MODELS:
        return None

    info = _ROBOT_MODELS[canonical]

    for search_dir in get_search_paths():
        dir_path = search_dir / info["dir"]
        if dir_path.exists():
            return dir_path

    return None


def get_robot_info(name: str) -> Optional[Dict]:
    """Get information about a robot model.

    Args:
        name: Robot name (canonical or alias)

    Returns:
        Dict with dir, model_xml, scene_xml, description, joints, category
    """
    canonical = resolve_robot_name(name)
    if canonical in _ROBOT_MODELS:
        info = dict(_ROBOT_MODELS[canonical])
        info["canonical_name"] = canonical
        info["resolved_path"] = str(resolve_model_path(name))
        info["available"] = resolve_model_path(name) is not None
        return info
    return None


def list_available_robots() -> List[Dict]:
    """List all available robot models with their info.

    Returns:
        List of dicts with name, description, joints, category, available
    """
    robots = []
    for name, info in sorted(_ROBOT_MODELS.items()):
        path = resolve_model_path(name)
        robots.append(
            {
                "name": name,
                "description": info["description"],
                "joints": info["joints"],
                "category": info["category"],
                "dir": info["dir"],
                "available": path is not None,
                "path": str(path) if path else None,
            }
        )
    return robots


def list_robots_by_category() -> Dict[str, List[Dict]]:
    """List robots grouped by category."""
    robots = list_available_robots()
    categories = {}
    for r in robots:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    return categories


def list_aliases() -> Dict[str, str]:
    """List all name aliases."""
    return dict(_ALIASES)


def format_robot_table() -> str:
    """Format a human-readable table of all robots."""
    lines = []
    lines.append(f"{'Name':<20} {'Category':<15} {'Joints':<8} {'Available':<10} Description")
    lines.append("─" * 100)

    by_cat = list_robots_by_category()
    for category in ["arm", "bimanual", "hand", "humanoid", "expressive", "mobile", "mobile_manip"]:
        if category not in by_cat:
            continue
        for r in by_cat[category]:
            avail = "✅" if r["available"] else "❌"
            lines.append(f"{r['name']:<20} {r['category']:<15} {r['joints']:<8} {avail:<10} {r['description']}")

    lines.append("")
    lines.append(
        f"Total: {len(_ROBOT_MODELS)} robots ({sum(1 for r in list_available_robots() if r['available'])} available)"
    )

    # Count aliases
    lines.append(f"Aliases: {len(_ALIASES)} (e.g. 'so100_dualcam' → 'so100', 'libero_panda' → 'panda')")

    return "\n".join(lines)


__all__ = [
    "resolve_model_path",
    "resolve_model_dir",
    "resolve_robot_name",
    "get_robot_info",
    "list_available_robots",
    "list_robots_by_category",
    "list_aliases",
    "format_robot_table",
    "get_assets_dir",
    "get_search_paths",
]
