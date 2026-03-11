"""Isaac Sim GPU-accelerated simulation backend for strands-robots.

Alternative to the MuJoCo Simulation class. Uses NVIDIA Isaac Sim for:
- GPU-parallel physics (thousands of envs simultaneously)
- RTX rendering (photorealistic cameras, ray-traced lighting)
- Deformable body simulation
- Advanced sensor simulation (LIDAR, IMU, contact)

Implements the same high-level API as simulation.py so it can be used
as a drop-in replacement via Robot("so100", backend="isaac").

Key difference from MuJoCo backend:
    MuJoCo: Single env, CPU physics, offscreen rendering
    Isaac:  N parallel envs, GPU physics, RTX rendering, USD scenes

Requires:
    - NVIDIA GPU (RTX 2070+ recommended)
    - Isaac Sim 5.1+ (pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics)
    - Isaac Lab (pip install isaaclab)
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Lazy imports — Isaac Sim/Lab are massive; only import when needed
# ─────────────────────────────────────────────────────────────────────

_sim_utils = None
_isaac_env = None


def _ensure_isaac():
    """Lazy-import Isaac Sim/Lab modules.

    Ensures ``_ensure_isaacsim()`` has been called first (which injects
    Isaac Sim + Isaac Lab onto ``sys.path`` when needed), then imports
    the Isaac Lab modules used by the backend.
    """
    global _sim_utils, _isaac_env
    if _sim_utils is not None:
        return True

    # Make sure Isaac Sim paths are on sys.path first
    _ensure_isaacsim()

    try:
        import isaaclab.sim as sim_utils_mod

        _sim_utils = sim_utils_mod

        from isaaclab.envs import DirectRLEnv

        _isaac_env = DirectRLEnv

        logger.info("✅ Isaac Lab loaded successfully")
        return True
    except ImportError as e:
        logger.warning("Isaac Lab not available: %s", e)
        return False


def _ensure_isaacsim():
    """Check if Isaac Sim runtime is available.

    Attempts a standard ``import isaacsim`` first. If that fails, looks
    for Isaac Sim at well-known filesystem locations (see
    :func:`get_isaac_sim_path`) and adds the Python packages directory
    to ``sys.path`` so subsequent imports succeed.

    Returns ``True`` if ``isaacsim`` can be imported, ``False`` otherwise.
    """
    try:
        import isaacsim  # noqa: F401

        return True
    except ImportError:
        pass

    # Attempt to discover Isaac Sim on the filesystem and
    # inject its Python packages into sys.path.
    from . import get_isaac_sim_path

    isaac_path = get_isaac_sim_path()
    if isaac_path is None:
        return False

    import sys

    python_pkgs = os.path.join(isaac_path, "python_packages")
    if os.path.isdir(python_pkgs) and python_pkgs not in sys.path:
        sys.path.insert(0, python_pkgs)
        logger.info("Added Isaac Sim python_packages to sys.path: %s", python_pkgs)

    # Also add Isaac Lab source if available alongside Isaac Sim
    for lab_candidate in [
        os.path.expanduser("~/IsaacLab/source/isaaclab"),
        os.path.join(os.path.dirname(isaac_path), "IsaacLab", "source", "isaaclab"),
        "/home/ubuntu/IsaacLab/source/isaaclab",
    ]:
        if os.path.isdir(lab_candidate) and lab_candidate not in sys.path:
            sys.path.insert(0, lab_candidate)
            logger.info("Added Isaac Lab to sys.path: %s", lab_candidate)
            break

    try:
        import isaacsim  # noqa: F401

        return True
    except ImportError:
        return False


def _setup_nucleus():
    """Configure Nucleus asset paths for Isaac Lab.

    Must be called AFTER SimulationApp is created but BEFORE importing isaaclab_tasks.
    Sets the carb settings that Isaac Lab reads for asset resolution.
    """
    try:
        import carb

        settings = carb.settings.get_settings()
        root = settings.get("/persistent/isaac/asset_root/cloud")
        if root is None or root == "":
            default_root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
            settings.set("/persistent/isaac/asset_root/cloud", default_root)
            root = default_root

        # Patch isaaclab assets module with correct root
        try:
            import isaaclab.utils.assets as am

            am.NUCLEUS_ASSET_ROOT_DIR = root
            am.NVIDIA_NUCLEUS_DIR = f"{root}/NVIDIA"
            am.ISAAC_NUCLEUS_DIR = f"{root}/Isaac"
            am.ISAACLAB_NUCLEUS_DIR = f"{root}/Isaac/IsaacLab"
            logger.info("Nucleus configured: %s", root)
        except ImportError:
            pass

        return root
    except ImportError:
        logger.debug("carb not available — Nucleus setup skipped")
        return None


# ─────────────────────────────────────────────────────────────────────
# Isaac Sim Backend — GPU-accelerated simulation
# ─────────────────────────────────────────────────────────────────────


@dataclass
class IsaacSimConfig:
    """Configuration for Isaac Sim backend."""

    num_envs: int = 1
    device: str = "cuda:0"
    physics_dt: float = 1.0 / 200.0  # 200Hz physics
    rendering_dt: float = 1.0 / 60.0  # 60Hz rendering
    headless: bool = True
    enable_cameras: bool = True
    camera_width: int = 640
    camera_height: int = 480
    render_mode: str = "rgb_array"  # "rgb_array" or "human"
    usd_scene_path: Optional[str] = None


class IsaacSimBackend:
    """GPU-accelerated simulation backend using NVIDIA Isaac Sim.

    This is the Isaac Sim equivalent of strands_robots.simulation.Simulation.
    It provides the same conceptual interface (create_world, add_robot, step, render)
    but runs on GPU with parallel environments.

    Differences from MuJoCo Simulation:
        - Parallel envs: num_envs parameter creates N identical envs on GPU
        - USD assets: Loads Universal Scene Description files instead of MJCF/URDF
        - RTX rendering: Photorealistic camera images via ray tracing
        - Domain randomization: Built-in Isaac Replicator integration
        - Deformable bodies: Soft objects, cloth, fluids
        - GPU tensor API: Observations/actions are torch.Tensor on GPU

    Usage:
        backend = IsaacSimBackend(config=IsaacSimConfig(num_envs=4096))
        backend.create_world()
        backend.add_robot("go2", usd_path="unitree_go2.usd")
        obs = backend.step(actions)  # GPU tensor, all 4096 envs
    """

    def __init__(self, config: Optional[IsaacSimConfig] = None):
        self.config = config or IsaacSimConfig()
        self._sim = None
        self._scene = None
        self._robot = None
        self._envs_created = False

        # Verify Isaac Sim availability
        if not _ensure_isaacsim():
            # Check if Isaac Sim is installed on the filesystem but not
            # importable in the current Python (common with Omniverse installs
            # which ship their own Python environment).
            from . import get_isaac_sim_path

            isaac_path = get_isaac_sim_path()
            if isaac_path:
                raise ImportError(
                    f"Isaac Sim found at {isaac_path} but cannot be imported "
                    f"in the current Python ({os.path.abspath(os.sys.executable)}).\n"
                    f"Isaac Sim requires its own Python runtime.\n"
                    f"Use: {isaac_path}/python.sh <your_script.py>\n"
                    f"Or install Isaac Sim Python packages: "
                    f"pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics"
                )
            raise ImportError(
                "Isaac Sim is required for the Isaac backend.\n"
                "Install with:\n"
                "  pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics\n"
                "Or use Omniverse Launcher: https://developer.nvidia.com/omniverse"
            )

        # Check if SimulationApp is actually callable (it may be None if
        # Isaac Sim extensions are not fully loaded in this Python env).
        try:
            from isaacsim import SimulationApp

            if SimulationApp is None:
                from . import get_isaac_sim_path

                isaac_path = get_isaac_sim_path()
                raise ImportError(
                    f"Isaac Sim imported but SimulationApp is None — "
                    f"the Omniverse Kit runtime is not available in this Python.\n"
                    f"Use Isaac Sim's own Python: {isaac_path}/python.sh <script.py>\n"
                    f"Or the CLI: {isaac_path}/isaac-sim.sh --headless"
                )
        except ImportError:
            pass  # Will be handled when create_world() is called

        logger.info(
            f"🎮 Isaac Sim backend initialized "
            f"(num_envs={self.config.num_envs}, device={self.config.device})"
        )

    def create_world(
        self,
        gravity: Optional[List[float]] = None,
        ground_plane: bool = True,
    ) -> Dict[str, Any]:
        """Create a new simulation world in Isaac Sim.

        Sets up the SimulationContext, physics scene, and ground plane.
        """
        if not _ensure_isaac():
            return {
                "status": "error",
                "content": [{"text": "❌ Isaac Lab not available"}],
            }

        try:
            from isaaclab.sim import SimulationCfg, SimulationContext

            sim_cfg = SimulationCfg(
                dt=self.config.physics_dt,
                render_interval=max(
                    1, int(self.config.rendering_dt / self.config.physics_dt)
                ),
                gravity=gravity or (0.0, 0.0, -9.81),
                device=self.config.device,
            )

            _setup_nucleus()
            self._sim = SimulationContext(sim_cfg)
            self._sim.set_camera_view(eye=[3.0, 3.0, 3.0], target=[0.0, 0.0, 0.0])

            # Add ground plane
            if ground_plane:
                ground_cfg = _sim_utils.GroundPlaneCfg()
                ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

            # Add dome light
            light_cfg = _sim_utils.DomeLightCfg(
                intensity=2000.0, color=(0.75, 0.75, 0.75)
            )
            light_cfg.func("/World/Light", light_cfg)

            self._envs_created = True

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"🌍 Isaac Sim world created (GPU: {self.config.device})\n"
                            f"⚙️ Physics: {1/self.config.physics_dt:.0f}Hz, "
                            f"Render: {1/self.config.rendering_dt:.0f}Hz\n"
                            f"🔢 Parallel envs: {self.config.num_envs}\n"
                            f"💡 Use add_robot() to spawn robots"
                        )
                    }
                ],
            }

        except Exception as e:
            logger.error("Failed to create Isaac world: %s", e)
            return {"status": "error", "content": [{"text": f"❌ Failed: {e}"}]}

    def add_robot(
        self,
        name: str,
        usd_path: Optional[str] = None,
        data_config: Optional[str] = None,
        position: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Add a robot to the Isaac Sim world.

        Args:
            name: Robot instance name
            usd_path: Direct path to USD file
            data_config: strands-robots data config name (auto-resolves to USD)
            position: [x, y, z] spawn position
        """
        if not self._envs_created:
            return {
                "status": "error",
                "content": [{"text": "❌ No world. Call create_world() first."}],
            }

        try:
            from isaaclab.assets import Articulation, ArticulationCfg

            # Resolve USD path
            resolved_usd = usd_path
            if not resolved_usd and data_config:
                resolved_usd = self._resolve_usd(data_config)

            if not resolved_usd:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"❌ No USD file found for '{data_config or name}'.\n"
                                "💡 Convert MJCF: from strands_robots.isaac import convert_mjcf_to_usd"
                            )
                        }
                    ],
                }

            pos = position or [0.0, 0.0, 0.0]

            # Create articulation config
            robot_cfg = ArticulationCfg(
                prim_path=f"/World/envs/env_.*/Robot_{name}",
                spawn=_sim_utils.UsdFileCfg(usd_path=resolved_usd),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=tuple(pos),
                ),
            )

            self._robot = Articulation(robot_cfg)

            n_joints = (
                self._robot.num_joints if hasattr(self._robot, "num_joints") else 0
            )

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"🤖 Robot '{name}' added to Isaac Sim\n"
                            f"📁 USD: {os.path.basename(resolved_usd or 'unknown')}\n"
                            f"📍 Position: {pos}\n"
                            f"🔩 Joints: {n_joints}\n"
                            f"🔢 Instances: {self.config.num_envs} parallel"
                        )
                    }
                ],
            }

        except Exception as e:
            logger.error("Failed to add robot: %s", e)
            return {"status": "error", "content": [{"text": f"❌ Failed: {e}"}]}

    def step(self, actions=None) -> Dict[str, Any]:
        """Step the simulation forward.

        Args:
            actions: torch.Tensor of shape (num_envs, action_dim)

        Returns:
            Dict with observations from all parallel envs
        """
        if self._sim is None:
            return {
                "status": "error",
                "content": [{"text": "❌ No simulation running"}],
            }

        try:

            if actions is not None and self._robot is not None:
                self._robot.set_joint_position_target(actions)

            self._sim.step()

            # Gather observations
            obs = {}
            if self._robot is not None:
                obs["joint_pos"] = self._robot.data.joint_pos.cpu().numpy()
                obs["joint_vel"] = self._robot.data.joint_vel.cpu().numpy()
                obs["root_pos"] = self._robot.data.root_pos_w.cpu().numpy()
                obs["root_quat"] = self._robot.data.root_quat_w.cpu().numpy()

            return {
                "status": "success",
                "content": [{"text": f"⏩ Stepped {self.config.num_envs} envs"}],
                "observations": obs,
            }

        except Exception as e:
            return {"status": "error", "content": [{"text": f"❌ Step failed: {e}"}]}

    def render(
        self,
        camera_name: str = "default",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Render a camera view using RTX ray tracing.

        Returns GPU-rendered image as PNG bytes.
        """
        if self._sim is None:
            return {"status": "error", "content": [{"text": "❌ No simulation"}]}

        w = width or self.config.camera_width
        h = height or self.config.camera_height

        try:
            # Use Isaac Sim's tiled camera for efficient GPU rendering
            from isaaclab.sensors import TiledCamera, TiledCameraCfg

            cam_cfg = TiledCameraCfg(
                prim_path="/World/Camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(3.0, 3.0, 3.0)),
                data_types=["rgb"],
                spawn=_sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    clipping_range=(0.01, 100.0),
                ),
                width=w,
                height=h,
            )

            camera = TiledCamera(cam_cfg)
            camera.update(dt=self.config.rendering_dt)

            rgb_data = camera.data.output["rgb"]
            if hasattr(rgb_data, "cpu"):
                rgb_np = rgb_data[0].cpu().numpy()
            else:
                rgb_np = np.array(rgb_data[0])

            import io

            from PIL import Image

            pil_img = Image.fromarray(rgb_np[:, :, :3].astype(np.uint8))
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            png_bytes = buffer.getvalue()

            return {
                "status": "success",
                "content": [
                    {"text": f"📸 RTX rendered {w}×{h} from '{camera_name}'"},
                    {"image": {"format": "png", "source": {"bytes": png_bytes}}},
                ],
            }

        except Exception as e:
            return {"status": "error", "content": [{"text": f"❌ Render failed: {e}"}]}

    def get_observation(self, robot_name: str = None) -> Dict[str, Any]:
        """Get observations from the simulation (GPU tensors).

        Returns observations in the same format as MuJoCo simulation.py
        for Policy ABC compatibility.
        """
        if self._robot is None:
            return {}

        obs = {}

        # Joint states
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        if hasattr(joint_pos, "cpu"):
            joint_pos = joint_pos.cpu().numpy()
            joint_vel = joint_vel.cpu().numpy()

        # Map to named joints (if single env, squeeze batch dim)
        if self.config.num_envs == 1:
            joint_pos = joint_pos.squeeze(0)
            joint_vel = joint_vel.squeeze(0)

        joint_names = (
            self._robot.joint_names if hasattr(self._robot, "joint_names") else []
        )
        for i, name in enumerate(joint_names):
            if i < len(joint_pos):
                obs[name] = (
                    float(joint_pos[i]) if joint_pos.ndim == 1 else joint_pos[:, i]
                )

        return obs

    def run_policy(
        self,
        robot_name: str,
        policy_provider: str = "mock",
        instruction: str = "",
        duration: float = 10.0,
        **policy_kwargs,
    ) -> Dict[str, Any]:
        """Run a strands-robots Policy on the Isaac Sim backend.

        Uses the same Policy ABC as MuJoCo simulation.py — zero code changes.
        """
        import asyncio
        import time

        from strands_robots.policies import create_policy

        policy = create_policy(policy_provider, **policy_kwargs)

        joint_names = (
            self._robot.joint_names
            if self._robot and hasattr(self._robot, "joint_names")
            else []
        )
        policy.set_robot_state_keys(joint_names)

        start_time = time.time()
        steps = 0

        while time.time() - start_time < duration:
            obs = self.get_observation(robot_name)

            try:
                actions = asyncio.run(policy.get_actions(obs, instruction))
            except RuntimeError:
                loop = asyncio.get_event_loop()
                actions = loop.run_until_complete(policy.get_actions(obs, instruction))

            if actions:
                import torch

                action_vals = []
                for jname in joint_names:
                    action_vals.append(actions[0].get(jname, 0.0))

                action_tensor = torch.tensor(
                    [action_vals],
                    device=self.config.device,
                    dtype=torch.float32,
                )
                if self.config.num_envs > 1:
                    action_tensor = action_tensor.expand(self.config.num_envs, -1)

                self._robot.set_joint_position_target(action_tensor)

            self._sim.step()
            steps += 1

        elapsed = time.time() - start_time
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"✅ Policy complete on Isaac Sim '{robot_name}'\n"
                        f"🧠 {policy_provider} | 🎯 {instruction}\n"
                        f"⏱️ {elapsed:.1f}s | 📊 {steps} steps | "
                        f"🔢 {self.config.num_envs} parallel envs"
                    )
                }
            ],
        }

    def destroy(self) -> Dict[str, Any]:
        """Destroy the simulation and release GPU resources."""
        if self._sim is not None:
            try:
                self._sim.stop()
            except Exception:
                pass
            self._sim = None
        self._robot = None
        self._envs_created = False
        return {
            "status": "success",
            "content": [{"text": "🗑️ Isaac Sim world destroyed"}],
        }

    # ── Helpers ──────────────────────────────────────────────────────

    def _resolve_usd(self, data_config: str) -> Optional[str]:
        """Resolve a strands-robots data_config name to a USD file path.

        Checks:
        1. Isaac Lab's bundled USD assets (Nucleus server)
        2. Local converted files from MJCF
        3. strands-robots asset manager → convert on the fly
        """
        # Isaac Lab built-in assets (via Nucleus)
        _ISAAC_ASSETS = {
            "unitree_go2": "ISAACLAB_NUCLEUS_DIR/Robots/Unitree/Go2/go2.usd",
            "unitree_g1": "ISAACLAB_NUCLEUS_DIR/Robots/Unitree/G1/g1.usd",
            "unitree_h1": "ISAACLAB_NUCLEUS_DIR/Robots/Unitree/H1/h1.usd",
            "unitree_a1": "ISAACLAB_NUCLEUS_DIR/Robots/Unitree/A1/a1.usd",
            "panda": "ISAACLAB_NUCLEUS_DIR/Robots/FrankaEmika/panda_instanceable.usd",
            "ur5e": "ISAACLAB_NUCLEUS_DIR/Robots/UniversalRobots/UR5e/ur5e.usd",
            "spot": "ISAACLAB_NUCLEUS_DIR/Robots/BostonDynamics/Spot/spot.usd",
            "shadow_hand": "ISAACLAB_NUCLEUS_DIR/Robots/ShadowHand/shadow_hand.usd",
            "kinova_gen3": "ISAACLAB_NUCLEUS_DIR/Robots/Kinova/Gen3/gen3.usd",
        }

        if data_config in _ISAAC_ASSETS:
            # Replace nucleus dir placeholder
            try:
                nucleus_dir = os.environ.get(
                    "ISAACLAB_NUCLEUS_DIR",
                    "omniverse://localhost/NVIDIA/Assets/Isaac/4.2",
                )
                return _ISAAC_ASSETS[data_config].replace(
                    "ISAACLAB_NUCLEUS_DIR", nucleus_dir
                )
            except Exception:
                pass

        # Check local converted cache
        cache_dir = Path.home() / ".strands_robots" / "isaac_cache"
        cached = cache_dir / f"{data_config}.usd"
        if cached.exists():
            return str(cached)

        # Try converting from MJCF
        try:
            from strands_robots.assets import resolve_model_path

            mjcf_path = resolve_model_path(data_config)
            if mjcf_path and mjcf_path.exists():
                from .asset_converter import convert_mjcf_to_usd

                cache_dir.mkdir(parents=True, exist_ok=True)
                output = str(cached)
                result = convert_mjcf_to_usd(str(mjcf_path), output)
                if result.get("status") == "success":
                    return output
        except ImportError:
            pass

        return None

    @property
    def is_available(self) -> bool:
        """Check if Isaac Sim runtime is available."""
        return _ensure_isaacsim()

    # ── Extended API (Phase 1.1 completion) ─────────────────────

    def add_object(
        self,
        name: str = "object",
        object_type: str = "box",
        position: Optional[List[float]] = None,
        size: Optional[List[float]] = None,
        color: Optional[List[float]] = None,
        usd_path: Optional[str] = None,
        mass: float = 1.0,
    ) -> Dict[str, Any]:
        """Spawn an object into the Isaac Sim world.

        Args:
            name: Instance name for the object
            object_type: Primitive type — "box", "sphere", "cylinder", "cone", "mesh"
            position: [x, y, z] spawn position
            size: Size parameters (type-dependent)
            color: [r, g, b, a] color values (0-1)
            usd_path: Path to USD file for mesh objects
            mass: Mass in kg

        Returns:
            Dict with status and object info
        """
        if not self._envs_created:
            return {
                "status": "error",
                "content": [{"text": "❌ No world. Call create_world() first."}],
            }

        pos = position or [0.0, 0.0, 0.5]
        sz = size or [0.1, 0.1, 0.1]
        clr = color or [0.8, 0.2, 0.2, 1.0]

        try:
            prim_path = f"/World/Objects/{name}"

            if object_type == "box":
                cfg = _sim_utils.CuboidCfg(
                    size=tuple(sz[:3]) if len(sz) >= 3 else (sz[0], sz[0], sz[0]),
                    rigid_props=_sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=_sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=_sim_utils.CollisionPropertiesCfg(),
                    visual_material=_sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(clr[:3]),
                    ),
                )
            elif object_type == "sphere":
                radius = sz[0] if sz else 0.1
                cfg = _sim_utils.SphereCfg(
                    radius=radius,
                    rigid_props=_sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=_sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=_sim_utils.CollisionPropertiesCfg(),
                    visual_material=_sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(clr[:3]),
                    ),
                )
            elif object_type == "cylinder":
                radius = sz[0] if len(sz) >= 1 else 0.05
                height = sz[1] if len(sz) >= 2 else 0.2
                cfg = _sim_utils.CylinderCfg(
                    radius=radius,
                    height=height,
                    rigid_props=_sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=_sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=_sim_utils.CollisionPropertiesCfg(),
                    visual_material=_sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(clr[:3]),
                    ),
                )
            elif object_type == "mesh" and usd_path:
                cfg = _sim_utils.UsdFileCfg(usd_path=usd_path)
            else:
                return {
                    "status": "error",
                    "content": [{"text": f"❌ Unknown object type: {object_type}"}],
                }

            cfg.func(prim_path, cfg, translation=tuple(pos))

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"📦 Object '{name}' added\n"
                            f"  Type: {object_type} | Mass: {mass}kg\n"
                            f"  Position: {pos} | Size: {sz}"
                        )
                    }
                ],
            }

        except Exception as e:
            logger.error("Failed to add object: %s", e)
            return {"status": "error", "content": [{"text": f"❌ Failed: {e}"}]}

    def add_terrain(
        self,
        terrain_type: str = "flat",
        size: Optional[List[float]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add terrain to the Isaac Sim world.

        Args:
            terrain_type: "flat", "rough", "stairs", "pyramid_stairs",
                         "discrete_obstacles", "wave", "heightfield"
            size: [width, length] in meters
            params: Type-specific parameters (roughness, step_height, etc.)

        Returns:
            Dict with status
        """
        if not self._envs_created:
            return {
                "status": "error",
                "content": [{"text": "❌ No world. Call create_world() first."}],
            }

        sz = size or [20.0, 20.0]
        # params available for future terrain customization

        try:
            from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
            from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

            if terrain_type == "flat":
                # Already have ground plane from create_world
                return {
                    "status": "success",
                    "content": [
                        {"text": "🏔️ Flat terrain (ground plane already present)"}
                    ],
                }

            elif terrain_type in ("rough", "stairs", "pyramid_stairs", "wave"):
                terrain_cfg = TerrainImporterCfg(
                    prim_path="/World/Terrain",
                    terrain_type="generator",
                    terrain_generator=ROUGH_TERRAINS_CFG,
                    max_init_terrain_level=5,
                    debug_vis=False,
                )
                TerrainImporter(terrain_cfg)

                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"🏔️ Terrain '{terrain_type}' added\n"
                                f"  Size: {sz[0]}×{sz[1]}m\n"
                                f"  Using Isaac Lab ROUGH_TERRAINS_CFG"
                            )
                        }
                    ],
                }

            else:
                return {
                    "status": "error",
                    "content": [{"text": f"❌ Unknown terrain: {terrain_type}"}],
                }

        except ImportError as e:
            return {
                "status": "error",
                "content": [{"text": f"❌ Terrain requires Isaac Lab: {e}"}],
            }
        except Exception as e:
            return {"status": "error", "content": [{"text": f"❌ Terrain failed: {e}"}]}

    def set_joint_positions(
        self,
        positions: List[float],
        env_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Set joint positions directly (teleport).

        Args:
            positions: Joint position values
            env_ids: Specific environment indices (None = all)

        Returns:
            Dict with status
        """
        if self._robot is None:
            return {"status": "error", "content": [{"text": "❌ No robot loaded"}]}

        try:
            import torch

            pos_tensor = torch.tensor(
                [positions], device=self.config.device, dtype=torch.float32
            )

            if env_ids is not None:
                idx = torch.tensor(env_ids, device=self.config.device)
                # Only set for specific envs
                current = self._robot.data.joint_pos.clone()
                current[idx] = pos_tensor
                self._robot.write_joint_state_to_sim(
                    current, self._robot.data.joint_vel
                )
            else:
                if self.config.num_envs > 1:
                    pos_tensor = pos_tensor.expand(self.config.num_envs, -1)
                self._robot.write_joint_state_to_sim(
                    pos_tensor,
                    torch.zeros_like(pos_tensor),  # zero velocity
                )

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"🎯 Joint positions set ({len(positions)} joints)\n"
                            f"  Envs: {env_ids if env_ids else 'all'}"
                        )
                    }
                ],
            }

        except Exception as e:
            return {"status": "error", "content": [{"text": f"❌ Failed: {e}"}]}

    def get_contact_forces(
        self,
        body_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get contact forces from the simulation.

        Args:
            body_name: Specific body to query (None = all)

        Returns:
            Dict with contact force data
        """
        if self._robot is None:
            return {"status": "error", "content": [{"text": "❌ No robot loaded"}]}

        try:
            # Isaac Lab provides net contact forces via the robot data
            net_forces = self._robot.data.net_contact_forces
            if hasattr(net_forces, "cpu"):
                forces_np = net_forces.cpu().numpy()
            else:
                forces_np = np.array(net_forces)

            total_force = float(np.linalg.norm(forces_np))
            max_force = float(np.max(np.abs(forces_np)))

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"📍 Contact Forces:\n"
                            f"  Shape: {forces_np.shape}\n"
                            f"  Total force norm: {total_force:.3f}N\n"
                            f"  Max component: {max_force:.3f}N"
                        )
                    }
                ],
                "forces": forces_np,
            }

        except Exception as e:
            return {
                "status": "error",
                "content": [{"text": f"❌ Contact query failed: {e}"}],
            }

    def set_camera_pose(
        self,
        eye: List[float],
        target: List[float],
    ) -> Dict[str, Any]:
        """Set the camera position and look-at target.

        Args:
            eye: Camera position [x, y, z]
            target: Look-at target [x, y, z]
        """
        if self._sim is None:
            return {"status": "error", "content": [{"text": "❌ No simulation"}]}

        try:
            self._sim.set_camera_view(eye=eye, target=target)
            return {
                "status": "success",
                "content": [{"text": f"📸 Camera set: eye={eye}, target={target}"}],
            }
        except Exception as e:
            return {"status": "error", "content": [{"text": f"❌ Camera failed: {e}"}]}

    def reset(
        self,
        env_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Reset simulation environments.

        Args:
            env_ids: Specific environments to reset (None = all)
        """
        if self._sim is None:
            return {"status": "error", "content": [{"text": "❌ No simulation"}]}

        try:
            import torch

            if self._robot is not None:
                if env_ids is not None:
                    idx = torch.tensor(env_ids, device=self.config.device)
                    # Reset specific envs to default joint positions
                    default_pos = self._robot.data.default_joint_pos[idx]
                    default_vel = self._robot.data.default_joint_vel[idx]
                    self._robot.write_joint_state_to_sim(
                        default_pos, default_vel, env_ids=idx
                    )
                    return {
                        "status": "success",
                        "content": [{"text": f"🔄 Reset {len(env_ids)} environments"}],
                    }
                else:
                    # Reset all
                    self._robot.write_joint_state_to_sim(
                        self._robot.data.default_joint_pos,
                        self._robot.data.default_joint_vel,
                    )

            self._sim.reset()
            return {
                "status": "success",
                "content": [
                    {"text": f"🔄 All {self.config.num_envs} environments reset"}
                ],
            }

        except Exception as e:
            return {"status": "error", "content": [{"text": f"❌ Reset failed: {e}"}]}

    def record_video(
        self,
        output_path: str = "/tmp/isaac_sim_recording.mp4",
        duration: float = 5.0,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
    ) -> Dict[str, Any]:
        """Record a video of the simulation.

        Args:
            output_path: Output MP4 file path
            duration: Recording duration in seconds
            fps: Frames per second
            width: Video width
            height: Video height

        Returns:
            Dict with status and output path
        """
        if self._sim is None:
            return {"status": "error", "content": [{"text": "❌ No simulation"}]}

        try:
            from strands_robots.video import VideoEncoder

            encoder = VideoEncoder(output_path, fps=fps)
            n_frames = int(duration * fps)
            physics_steps_per_frame = max(1, int(1.0 / (self.config.physics_dt * fps)))

            for frame_idx in range(n_frames):
                # Step physics
                for _ in range(physics_steps_per_frame):
                    self._sim.step()

                # Render frame
                render_result = self.render(width=width, height=height)
                if render_result.get("status") == "success":
                    for content in render_result.get("content", []):
                        if "image" in content:
                            img_bytes = content["image"]["source"]["bytes"]
                            import io

                            from PIL import Image

                            img = Image.open(io.BytesIO(img_bytes))
                            frame = np.array(img)
                            encoder.add_frame(frame)

            encoder.close()

            import os

            size_kb = os.path.getsize(output_path) / 1024

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"🎬 Video recorded!\n"
                            f"  Output: {output_path}\n"
                            f"  Frames: {n_frames} @ {fps}fps\n"
                            f"  Duration: {duration:.1f}s\n"
                            f"  Resolution: {width}×{height}\n"
                            f"  Size: {size_kb:.1f} KB"
                        )
                    }
                ],
                "video_path": output_path,
            }

        except ImportError as e:
            return {
                "status": "error",
                "content": [{"text": f"❌ Video recording requires: {e}"}],
            }
        except Exception as e:
            return {
                "status": "error",
                "content": [{"text": f"❌ Recording failed: {e}"}],
            }
