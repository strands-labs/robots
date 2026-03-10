"""Marble 3D World Generation for Robot Training Environments.

Integrates World Labs' Marble world model with strands-robots to generate
diverse 3D training environments for robot learning. Marble creates full
3D worlds from text, images, video, or coarse 3D layouts and exports as
Gaussian splats (.ply), meshes (.glb), or videos.

The pipeline converts Marble outputs into Isaac Sim-compatible USD scenes
and composes them with robots and task objects for use with LeIsaac
environments, Cosmos Transfer sim-to-real, and DreamGen neural trajectories.

Typical usage:

    from strands_robots.marble import MarblePipeline, MarbleConfig

    config = MarbleConfig(output_format="usdz", robot="so101")
    pipeline = MarblePipeline(config)

    # Generate a 3D world from text
    scene = pipeline.generate_world(
        prompt="A modern kitchen with wooden countertops and a sink",
        output_dir="./scenes/kitchen",
    )

    # Compose with robot + task objects for training
    composed = pipeline.compose_scene(
        scene_path=scene["usdz_path"],
        robot="so101",
        task_objects=["orange", "plate"],
        table_replacement=True,
    )

    # Use directly with LeIsaac
    from strands_robots.leisaac import LeIsaacEnv
    env = LeIsaacEnv.from_marble_scene(composed["scene_usd"])

Or use the convenience function:

    from strands_robots.marble import generate_world

    result = generate_world(
        prompt="A cluttered office desk with papers and mugs",
        robot="so101",
        compose=True,
    )

Pipeline:
    Text/Image → Marble API → .ply + .glb
                                    ↓
                   3DGrut: PLY → USDZ
                                    ↓
                   marble_compose: Robot + Table + Objects → scene.usd
                                    ↓
             ┌─────────────────────────────────────────┐
             │  LeIsaac env OR Isaac Sim direct load    │
             │  → Policy rollout (any of 18 providers)  │
             │  → Dataset recording (LeRobot format)    │
             │  → Training (GR00T/LeRobot/Cosmos)       │
             │  → Cosmos Transfer (sim→real video)       │
             └─────────────────────────────────────────┘

References:
    - Marble by World Labs: https://marble.worldlabs.ai
    - Marble API docs: https://docs.worldlabs.ai/api/reference/openapi
    - LeIsaac × Marble: https://lightwheelai.github.io/marble/
    - 3DGrut (PLY→USDZ): https://github.com/nv-tlabs/3dgrut
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile  # noqa: F401
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = [
    "MarblePipeline",
    "MarbleConfig",
    "MarbleScene",
    "generate_world",
    "compose_scene",
    "list_presets",
    "MARBLE_PRESETS",
    "MARBLE_MODELS",
    "SUPPORTED_ROBOTS",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARBLE_API_URL = "https://api.worldlabs.ai"

VALID_OUTPUT_FORMATS: Tuple[str, ...] = ("ply", "glb", "usdz", "video")

VALID_INPUT_MODES: Tuple[str, ...] = ("text", "image", "video", "multi-image")

# Models available in the Marble API
MARBLE_MODELS: Tuple[str, ...] = ("Marble 0.1-mini", "Marble 0.1-plus")

# Default polling configuration for async generation
DEFAULT_POLL_INTERVAL: float = 5.0  # seconds between polls
DEFAULT_POLL_TIMEOUT: float = 600.0  # max seconds to wait for generation

# Robots supported for scene composition (from LeIsaac + strands-robots)
SUPPORTED_ROBOTS: Dict[str, Dict[str, Any]] = {
    "so101": {
        "usd_path": "robots/so101/so101.usd",
        "type": "single_arm",
        "mount_height": 0.0,
        "description": "SO-101 Follower single arm",
    },
    "bi_so101": {
        "usd_path": "robots/bi_so101/bi_so101.usd",
        "type": "dual_arm",
        "mount_height": 0.0,
        "description": "SO-101 Follower bimanual (dual-arm)",
    },
    "panda": {
        "usd_path": "robots/panda/panda.usd",
        "type": "single_arm",
        "mount_height": 0.0,
        "description": "Franka Emika Panda 7-DOF",
    },
    "ur5e": {
        "usd_path": "robots/ur5e/ur5e.usd",
        "type": "single_arm",
        "mount_height": 0.0,
        "description": "Universal Robots UR5e 6-DOF",
    },
    "xarm7": {
        "usd_path": "robots/xarm7/xarm7.usd",
        "type": "single_arm",
        "mount_height": 0.0,
        "description": "UFactory xArm 7-DOF",
    },
    "lekiwi": {
        "usd_path": "robots/lekiwi/lekiwi.usd",
        "type": "mobile_manipulation",
        "mount_height": 0.3,
        "description": "LeKiwi mobile manipulation platform",
    },
}

# Predefined scene generation presets
MARBLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "kitchen": {
        "prompt": "A modern kitchen with wooden countertops, stainless steel sink, "
        "and a cutting board with fruits on the counter",
        "description": "Kitchen manipulation — pick-place, pouring, cutting prep",
        "task_objects": ["orange", "apple", "mug", "plate", "knife"],
        "category": "indoor",
    },
    "office_desk": {
        "prompt": "A cluttered office desk with papers, a coffee mug, pens, " "and a laptop next to a wooden bookshelf",
        "description": "Office desk manipulation — sorting, stacking, tidying",
        "task_objects": ["mug", "pen", "paper", "book"],
        "category": "indoor",
    },
    "workshop": {
        "prompt": "A workshop table with tools, screws in a box, a circuit board, "
        "and a soldering iron in a well-lit garage",
        "description": "Workshop assembly — precision pick-place, tool use",
        "task_objects": ["screw", "screwdriver", "circuit_board", "box"],
        "category": "indoor",
    },
    "living_room": {
        "prompt": "A cozy living room with a coffee table, scattered toys, "
        "folded towels, and a houseplant near a window",
        "description": "Home cleanup — tidying, folding, organizing",
        "task_objects": ["toy", "towel", "remote", "cup"],
        "category": "indoor",
    },
    "warehouse": {
        "prompt": "A warehouse aisle with metal shelving, cardboard boxes of " "varying sizes, and a packing station",
        "description": "Warehouse logistics — box manipulation, stacking",
        "task_objects": ["box_small", "box_medium", "box_large", "pallet"],
        "category": "industrial",
    },
    "lab_bench": {
        "prompt": "A clean laboratory bench with beakers, test tubes in a rack, " "a microscope, and a precision scale",
        "description": "Lab manipulation — precise handling, liquid transfer",
        "task_objects": ["beaker", "test_tube", "petri_dish"],
        "category": "scientific",
    },
    "outdoor_garden": {
        "prompt": "A garden table on a patio with potted plants, a watering can, "
        "garden tools, and a wooden fence background",
        "description": "Outdoor manipulation — robust to varied lighting",
        "task_objects": ["pot", "watering_can", "trowel", "seeds"],
        "category": "outdoor",
    },
    "restaurant": {
        "prompt": "A restaurant table with plates, napkins, wine glasses, "
        "cutlery, and a bread basket, warm ambient lighting",
        "description": "Table setting and service tasks",
        "task_objects": ["plate", "wine_glass", "fork", "napkin"],
        "category": "indoor",
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MarbleScene:
    """Represents a generated Marble 3D scene with all output artifacts."""

    scene_id: str
    prompt: str
    input_mode: str = "text"

    # Raw Marble outputs
    ply_path: Optional[str] = None  # Gaussian splat (.ply format)
    spz_path: Optional[str] = None  # Gaussian splat (.spz compressed format)
    glb_path: Optional[str] = None  # Mesh
    video_path: Optional[str] = None  # Rendered video
    pano_path: Optional[str] = None  # Downloaded panorama image

    # Marble API outputs
    world_id: Optional[str] = None  # Marble world identifier
    world_marble_url: Optional[str] = None  # World Marble viewer URL
    pano_url: Optional[str] = None  # Panorama image URL
    thumbnail_url: Optional[str] = None  # Thumbnail URL
    collider_mesh_url: Optional[str] = None  # Collider mesh URL
    spz_urls: Optional[Dict[str, str]] = None  # SPZ Gaussian splat URLs
    caption: Optional[str] = None  # AI-generated caption
    operation_id: Optional[str] = None  # Async operation ID
    model: Optional[str] = None  # Model used (Marble 0.1-mini/plus)

    # Converted outputs
    usdz_path: Optional[str] = None  # USDZ via 3DGrut conversion
    scene_usd: Optional[str] = None  # Final composed scene with robot

    # Metadata
    output_dir: str = ""
    robot: Optional[str] = None
    task_objects: List[str] = field(default_factory=list)
    composed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def splat_path(self) -> Optional[str]:
        """Best available Gaussian splat file (SPZ preferred, then PLY)."""
        return self.spz_path or self.ply_path

    @property
    def best_background(self) -> Optional[str]:
        """Best available background asset for scene composition."""
        return self.usdz_path or self.glb_path or self.spz_path or self.ply_path or self.pano_path

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Include computed properties
        d["splat_path"] = self.splat_path
        d["best_background"] = self.best_background
        return d


@dataclass
class MarbleConfig:
    """Configuration for the Marble 3D World Generation pipeline.

    Attributes:
        api_url: Marble API base URL or ``"local"`` for offline generation.
        api_key: Marble API key (``WLT-Api-Key`` header). Falls back to
            ``MARBLE_API_KEY`` or ``WLT_API_KEY`` env var.
        model: Marble model to use for generation. One of
            ``"Marble 0.1-mini"`` or ``"Marble 0.1-plus"``.
        output_format: Primary output format(s) requested from Marble.
            One of ``"ply"``, ``"glb"``, ``"usdz"``, ``"video"``.
        input_mode: Input modality. One of ``"text"``, ``"image"``,
            ``"video"``, ``"multi-image"``.
        robot: Default robot to compose into scenes. Key from
            ``SUPPORTED_ROBOTS``.
        table_replacement: Whether to replace detected table geometry with
            a physics-enabled table USD prim.
        auto_compose: Automatically compose scenes with robot after
            generation.
        convert_to_usdz: Convert PLY Gaussian splats to USDZ via 3DGrut.
        threedgrut_path: Path to 3DGrut installation. Falls back to
            ``THREEDGRUT_PATH`` env var.
        chisel_enabled: Enable Marble's Chisel 3D sculpting tool for
            interactive scene editing.
        num_variations: Number of scene variations to generate per prompt
            (for domain randomization).
        seed: Random seed for reproducibility.
        poll_interval: Seconds between polling the operation status.
        poll_timeout: Max seconds to wait for generation to complete.
        is_public: Whether generated worlds should be publicly visible.
    """

    api_url: str = MARBLE_API_URL
    api_key: Optional[str] = None
    model: str = "Marble 0.1-plus"
    output_format: str = "ply"
    input_mode: str = "text"
    robot: Optional[str] = None
    table_replacement: bool = True
    auto_compose: bool = False
    convert_to_usdz: bool = True
    threedgrut_path: Optional[str] = None
    chisel_enabled: bool = False
    num_variations: int = 1
    seed: int = 42
    poll_interval: float = DEFAULT_POLL_INTERVAL
    poll_timeout: float = DEFAULT_POLL_TIMEOUT
    is_public: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialisation."""
        if self.output_format not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output_format '{self.output_format}'. " f"Must be one of: {VALID_OUTPUT_FORMATS}"
            )

        if self.input_mode not in VALID_INPUT_MODES:
            raise ValueError(f"Invalid input_mode '{self.input_mode}'. " f"Must be one of: {VALID_INPUT_MODES}")

        if self.model not in MARBLE_MODELS:
            raise ValueError(f"Invalid model '{self.model}'. " f"Must be one of: {MARBLE_MODELS}")

        if self.robot and self.robot not in SUPPORTED_ROBOTS:
            raise ValueError(f"Unknown robot '{self.robot}'. " f"Supported: {sorted(SUPPORTED_ROBOTS.keys())}")

        if self.num_variations < 1:
            raise ValueError(f"num_variations must be >= 1, got {self.num_variations}")

        # Resolve API key (WLT-Api-Key header per OpenAPI spec)
        if self.api_key is None:
            self.api_key = os.environ.get("WLT_API_KEY") or os.environ.get("MARBLE_API_KEY")

        # Resolve 3DGrut path
        if self.threedgrut_path is None:
            self.threedgrut_path = os.environ.get("THREEDGRUT_PATH")

    def resolve_threedgrut_path(self) -> Optional[str]:
        """Resolve path to 3DGrut installation.

        Returns:
            Path to 3DGrut directory, or None if not found.
        """
        if self.threedgrut_path and os.path.isdir(self.threedgrut_path):
            return self.threedgrut_path

        env_path = os.environ.get("THREEDGRUT_PATH")
        if env_path and os.path.isdir(env_path):
            return env_path

        # Common install locations
        candidates = [
            os.path.expanduser("~/3dgrut"),
            os.path.expanduser("~/repos/3dgrut"),
            "/opt/3dgrut",
        ]
        for candidate in candidates:
            if os.path.isdir(candidate):
                logger.info("Found 3DGrut at %s", candidate)
                return candidate

        return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class MarblePipeline:
    """End-to-end pipeline for Marble 3D world generation and scene composition.

    The pipeline orchestrates:

    1. **Generate**: Create 3D worlds via Marble API or local generation
       from text, images, video, or 3D layouts.
    2. **Convert**: Transform Gaussian splat PLY files to USDZ via 3DGrut
       for Isaac Sim compatibility.
    3. **Compose**: Combine generated backgrounds with robots and task
       objects using ``marble_compose.py``-style scene assembly.
    4. **Export**: Produce Isaac Sim / LeIsaac-ready USD scenes for
       training, evaluation, or Cosmos Transfer augmentation.

    Args:
        config: A :class:`MarbleConfig` instance. When *None* a default
            configuration is used.
        **kwargs: Additional keyword arguments forwarded to
            :class:`MarbleConfig` when ``config`` is *None*.

    Example::

        pipeline = MarblePipeline(MarbleConfig(
            robot="so101",
            num_variations=10,
            convert_to_usdz=True,
        ))
        scenes = pipeline.generate_world(
            prompt="A kitchen countertop with fruits",
            output_dir="./scenes/kitchen",
        )
    """

    def __init__(
        self,
        config: Optional[MarbleConfig] = None,
        **kwargs: Any,
    ) -> None:
        if config is not None and kwargs:
            raise ValueError(
                "Cannot specify both 'config' and keyword arguments. "
                "Pass either a MarbleConfig instance OR keyword arguments."
            )

        if config is None:
            config = MarbleConfig(**kwargs)

        self.config = config
        self._tmp_dirs: List[str] = []
        self._generated_scenes: List[MarbleScene] = []

        logger.info(
            "Initialised MarblePipeline (format=%s, robot=%s, variations=%d)",
            self.config.output_format,
            self.config.robot or "none",
            self.config.num_variations,
        )

    # ------------------------------------------------------------------
    # Stage 1: Generate 3D World
    # ------------------------------------------------------------------

    def generate_world(
        self,
        prompt: str,
        output_dir: str = "./marble_scenes",
        input_image: Optional[str] = None,
        input_video: Optional[str] = None,
        input_layout: Optional[str] = None,
        num_variations: Optional[int] = None,
        **kwargs: Any,
    ) -> List[MarbleScene]:
        """Stage 1: Generate 3D world(s) from Marble.

        Creates one or more 3D worlds using the Marble world model. The
        input modality is determined by which arguments are provided:

        - **Text only** → text-to-3D generation
        - **Image provided** → image-to-3D generation
        - **Video provided** → video-to-3D generation
        - **Layout provided** → layout-guided 3D generation

        Args:
            prompt: Text prompt describing the desired 3D world.
            output_dir: Directory for generated outputs.
            input_image: Path to input image (for image-to-3D).
            input_video: Path to input video (for video-to-3D).
            input_layout: Path to coarse 3D layout (for layout-guided).
            num_variations: Override config.num_variations for this call.
            **kwargs: Additional generation parameters.

        Returns:
            List of :class:`MarbleScene` objects, one per variation.

        Raises:
            ValueError: If invalid inputs are provided.
            RuntimeError: If generation fails.
        """
        num_vars = num_variations or self.config.num_variations
        os.makedirs(output_dir, exist_ok=True)

        # Determine input mode
        if input_image and os.path.isfile(input_image):
            input_mode = "image"
        elif input_video and os.path.isfile(input_video):
            input_mode = "video"
        elif input_layout and os.path.isfile(input_layout):
            input_mode = "layout_3d"
        else:
            input_mode = "text"

        logger.info(
            "🌍 Stage 1: Generating %d world(s) from %s prompt",
            num_vars,
            input_mode,
        )
        logger.info("   Prompt: %s", prompt[:100])

        scenes: List[MarbleScene] = []

        for var_idx in range(num_vars):
            scene_id = f"marble_{uuid.uuid4().hex[:8]}"
            scene_dir = os.path.join(output_dir, scene_id)
            os.makedirs(scene_dir, exist_ok=True)

            scene = MarbleScene(
                scene_id=scene_id,
                prompt=prompt,
                input_mode=input_mode,
                output_dir=scene_dir,
                metadata={
                    "variation_index": var_idx,
                    "seed": self.config.seed + var_idx,
                    **kwargs,
                },
            )

            # Call Marble API or local generation
            try:
                result = self._call_marble_api(
                    prompt=prompt,
                    input_mode=input_mode,
                    input_image=input_image,
                    input_video=input_video,
                    input_layout=input_layout,
                    output_dir=scene_dir,
                    seed=self.config.seed + var_idx,
                    **kwargs,
                )

                scene.ply_path = result.get("ply_path")
                scene.spz_path = result.get("spz_path")
                scene.glb_path = result.get("glb_path")
                scene.video_path = result.get("video_path")
                scene.pano_path = result.get("pano_path")
                scene.metadata.update(result.get("metadata", {}))

                # Populate Marble API-specific fields
                meta = result.get("metadata", {})
                scene.world_id = meta.get("world_id")
                scene.world_marble_url = meta.get("world_marble_url")
                scene.pano_url = meta.get("pano_url")
                scene.thumbnail_url = meta.get("thumbnail_url")
                scene.collider_mesh_url = meta.get("collider_mesh_url")
                scene.spz_urls = meta.get("spz_urls")
                scene.caption = meta.get("caption")
                scene.operation_id = meta.get("operation_id")
                scene.model = meta.get("model") or self.config.model

            except Exception as exc:
                logger.error("Failed to generate world variation %d: %s", var_idx, exc)
                scene.metadata["error"] = str(exc)

            # Stage 2: Convert PLY → USDZ if requested
            # NOTE: Only convert actual .ply files. SPZ files are a different
            # compressed Gaussian splat format and need their own converter.
            if self.config.convert_to_usdz and scene.ply_path:
                ply_ext = Path(scene.ply_path).suffix.lower()
                if ply_ext == ".ply":
                    try:
                        usdz_path = self.convert_ply_to_usdz(
                            scene.ply_path,
                            output_path=os.path.join(scene_dir, f"{scene_id}.usdz"),
                        )
                        scene.usdz_path = usdz_path
                    except Exception as exc:
                        logger.warning("PLY→USDZ conversion failed: %s", exc)
                else:
                    logger.info(
                        "Skipping PLY→USDZ conversion for %s file: %s "
                        "(SPZ splats from the Marble API don't need conversion — "
                        "use the world_marble_url for viewing or download the "
                        "collider mesh GLB for Isaac Sim)",
                        ply_ext,
                        scene.ply_path,
                    )

            # Stage 3: Auto-compose if configured
            if self.config.auto_compose and self.config.robot:
                try:
                    bg_path = scene.best_background
                    if bg_path:
                        composed = self.compose_scene(
                            scene_path=bg_path,
                            robot=self.config.robot,
                            output_dir=scene_dir,
                        )
                        scene.scene_usd = composed.get("scene_usd")
                        scene.robot = self.config.robot
                        scene.composed = True
                except Exception as exc:
                    logger.warning("Auto-compose failed: %s", exc)

            scenes.append(scene)
            self._generated_scenes.append(scene)

        logger.info(
            "✅ Generated %d scene(s) → %s",
            len(scenes),
            output_dir,
        )
        return scenes

    # ------------------------------------------------------------------
    # Stage 2: Convert PLY → USDZ
    # ------------------------------------------------------------------

    def convert_ply_to_usdz(
        self,
        ply_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Stage 2: Convert Gaussian splat PLY to USDZ via 3DGrut.

        Uses NVIDIA's 3DGrut (3D Gaussian Ray-tracing Utility Toolkit) to
        convert Marble-generated Gaussian splat .ply files into .usdz
        format compatible with Isaac Sim and other USD-based tools.

        Args:
            ply_path: Path to the input .ply Gaussian splat file.
            output_path: Destination for the .usdz file. When *None*,
                generates alongside the input with ``.usdz`` extension.

        Returns:
            Path to the generated USDZ file.

        Raises:
            FileNotFoundError: If ``ply_path`` does not exist or 3DGrut
                is not installed.
            RuntimeError: If conversion fails.
        """
        ply_path = str(ply_path)
        if not os.path.isfile(ply_path):
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        if output_path is None:
            output_path = str(Path(ply_path).with_suffix(".usdz"))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        logger.info("🔄 Converting PLY → USDZ: %s", ply_path)

        # Strategy 1: 3DGrut CLI
        grut_path = self.config.resolve_threedgrut_path()
        if grut_path:
            return self._convert_via_3dgrut(ply_path, output_path, grut_path)

        # Strategy 2: pxr (OpenUSD) direct conversion
        if self._try_pxr_conversion(ply_path, output_path):
            return output_path

        # Strategy 3: trimesh fallback
        if self._try_trimesh_conversion(ply_path, output_path):
            return output_path

        raise RuntimeError(
            "No PLY→USDZ converter available. Install one of:\n"
            "  1. 3DGrut: git clone https://github.com/nv-tlabs/3dgrut\n"
            "  2. OpenUSD: pip install usd-core\n"
            "  3. trimesh: pip install trimesh\n"
            "Set THREEDGRUT_PATH env var for option 1."
        )

    def _convert_via_3dgrut(self, ply_path: str, output_path: str, grut_path: str) -> str:
        """Convert PLY to USDZ using 3DGrut CLI."""
        export_script = os.path.join(grut_path, "scripts", "export_usdz.py")
        if not os.path.isfile(export_script):
            # Try alternative script paths
            for alt in ["export.py", "convert.py", "ply2usdz.py"]:
                alt_path = os.path.join(grut_path, "scripts", alt)
                if os.path.isfile(alt_path):
                    export_script = alt_path
                    break
            else:
                # Use module-based invocation
                python = shutil.which("python3") or shutil.which("python")
                cmd = [
                    python,
                    "-m",
                    "threedgrut.export",
                    "--input",
                    ply_path,
                    "--output",
                    output_path,
                    "--format",
                    "usdz",
                ]
                return self._run_subprocess(cmd, "3DGrut export")

        python = shutil.which("python3") or shutil.which("python")
        cmd = [python, export_script, "--input", ply_path, "--output", output_path]

        return self._run_subprocess(cmd, "3DGrut PLY→USDZ")

    def _try_pxr_conversion(self, ply_path: str, output_path: str) -> bool:
        """Attempt PLY→USD conversion via OpenUSD (pxr)."""
        try:
            import numpy as np  # noqa: F401
            from pxr import Gf, Sdf, Usd, UsdGeom  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            logger.debug("OpenUSD (pxr) not available for PLY conversion.")
            return False

        try:
            # Read PLY vertices
            vertices, colors = self._read_ply_vertices(ply_path)
            if vertices is None or len(vertices) == 0:
                return False

            # Create USD stage
            stage = Usd.Stage.CreateNew(output_path)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

            UsdGeom.Xform.Define(stage, "/World")
            points_prim = UsdGeom.Points.Define(stage, "/World/MarbleScene")

            # Set points
            points_prim.GetPointsAttr().Set([Gf.Vec3f(*v) for v in vertices])

            # Set widths (point sizes)
            points_prim.GetWidthsAttr().Set([0.005] * len(vertices))

            # Set colors if available
            if colors is not None and len(colors) > 0:
                color_attr = points_prim.GetDisplayColorAttr()
                color_attr.Set([Gf.Vec3f(*c) for c in colors])

            stage.GetRootLayer().Save()
            logger.info("PLY→USD converted via OpenUSD: %s", output_path)
            return True

        except Exception:
            logger.debug("OpenUSD PLY conversion failed.", exc_info=True)
            return False

    def _try_trimesh_conversion(self, ply_path: str, output_path: str) -> bool:
        """Attempt PLY→GLB conversion via trimesh (as USDZ fallback)."""
        try:
            import trimesh  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("trimesh not available for PLY conversion.")
            return False

        try:
            mesh = trimesh.load(ply_path)
            # trimesh can export to GLB; we save alongside as intermediate
            glb_path = str(Path(output_path).with_suffix(".glb"))
            mesh.export(glb_path)
            logger.info(
                "PLY→GLB converted via trimesh: %s (USDZ requires 3DGrut or pxr)",
                glb_path,
            )
            # Copy to output path as a usable intermediate
            shutil.copy2(glb_path, output_path)
            return True
        except Exception:
            logger.debug("trimesh PLY conversion failed.", exc_info=True)
            return False

    @staticmethod
    def _read_ply_vertices(
        ply_path: str,
    ) -> Tuple[Optional[list], Optional[list]]:
        """Read vertices and optional colours from a PLY file.

        Supports both ASCII and binary PLY formats via a lightweight parser.
        For full Gaussian splat data (SH coefficients, opacity, scale,
        rotation) use 3DGrut's native reader.

        Args:
            ply_path: Path to the PLY file.

        Returns:
            Tuple of (vertices, colors) where each is a list of 3-tuples
            or None if parsing fails.
        """
        try:
            # Try plyfile first (requires numpy)
            try:
                import numpy as np  # noqa: F401
                from plyfile import PlyData  # type: ignore[import-untyped]

                plydata = PlyData.read(ply_path)
                vertex = plydata["vertex"]
                vertices = list(zip(vertex["x"], vertex["y"], vertex["z"]))

                colors = None
                if "red" in vertex.data.dtype.names:
                    colors = [
                        (
                            float(r) / 255.0,
                            float(g) / 255.0,
                            float(b) / 255.0,
                        )
                        for r, g, b in zip(vertex["red"], vertex["green"], vertex["blue"])
                    ]

                return vertices, colors

            except ImportError:
                pass

            # Fallback: manual ASCII PLY reader (pure Python, no deps)
            vertices = []
            colors = []
            header_ended = False
            vertex_count = 0
            has_color = False

            with open(ply_path, "r", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not header_ended:
                        if line.startswith("element vertex"):
                            vertex_count = int(line.split()[-1])
                        if "red" in line:
                            has_color = True
                        if line == "end_header":
                            header_ended = True
                        continue

                    if len(vertices) >= vertex_count:
                        break

                    parts = line.split()
                    if len(parts) >= 3:
                        vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
                        if has_color and len(parts) >= 6:
                            colors.append(
                                (
                                    float(parts[3]) / 255.0,
                                    float(parts[4]) / 255.0,
                                    float(parts[5]) / 255.0,
                                )
                            )

            return vertices if vertices else None, colors if colors else None

        except Exception:
            logger.debug("Failed to read PLY file: %s", ply_path, exc_info=True)
            return None, None

    # ------------------------------------------------------------------
    # Stage 3: Compose Scene
    # ------------------------------------------------------------------

    def compose_scene(
        self,
        scene_path: str,
        robot: Optional[str] = None,
        task_objects: Optional[List[str]] = None,
        table_replacement: Optional[bool] = None,
        robot_position: Optional[List[float]] = None,
        output_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Stage 3: Compose a Marble scene with robot and task objects.

        Takes a generated/converted Marble background and overlays:
        - Robot USD model at specified position
        - Task objects (cubes, cylinders, or custom USD prims)
        - Optional table replacement with physics-enabled mesh
        - Lighting and physics settings for Isaac Sim

        This mirrors the ``marble_compose.py`` script from the
        LeIsaac × Marble integration.

        Args:
            scene_path: Path to the background scene (.usdz, .glb, .ply).
            robot: Robot key from ``SUPPORTED_ROBOTS``. Falls back to
                ``config.robot``.
            task_objects: List of task object names to place in the scene.
            table_replacement: Override config table_replacement.
            robot_position: [x, y, z] position for the robot base.
            output_dir: Directory for the composed scene output.
            **kwargs: Additional composition parameters.

        Returns:
            Dictionary with:
                - ``scene_usd``: Path to the final composed USD scene.
                - ``robot``: Robot key used.
                - ``task_objects``: Objects placed in the scene.
                - ``background``: Path to the background asset.

        Raises:
            FileNotFoundError: If ``scene_path`` doesn't exist.
            ValueError: If ``robot`` is not a supported key.
        """
        scene_path = str(scene_path)
        if not os.path.isfile(scene_path):
            raise FileNotFoundError(f"Scene file not found: {scene_path}")

        robot = robot or self.config.robot
        if robot and robot not in SUPPORTED_ROBOTS:
            raise ValueError(f"Unknown robot '{robot}'. " f"Supported: {sorted(SUPPORTED_ROBOTS.keys())}")

        if table_replacement is None:
            table_replacement = self.config.table_replacement

        if task_objects is None:
            task_objects = []

        if output_dir is None:
            output_dir = str(Path(scene_path).parent)

        os.makedirs(output_dir, exist_ok=True)

        scene_stem = Path(scene_path).stem
        composed_usd = os.path.join(output_dir, f"{scene_stem}_composed.usda")

        logger.info(
            "🔧 Stage 3: Composing scene\n"
            "   Background: %s\n"
            "   Robot: %s\n"
            "   Objects: %s\n"
            "   Table replacement: %s",
            scene_path,
            robot or "none",
            task_objects or "none",
            table_replacement,
        )

        # Try USD composition via pxr
        if self._try_usd_composition(
            scene_path=scene_path,
            composed_usd=composed_usd,
            robot=robot,
            task_objects=task_objects,
            table_replacement=table_replacement,
            robot_position=robot_position,
            **kwargs,
        ):
            logger.info("✅ Composed scene: %s", composed_usd)
        else:
            # Fallback: create a composition manifest JSON
            manifest = {
                "background": scene_path,
                "robot": robot,
                "robot_info": SUPPORTED_ROBOTS.get(robot, {}) if robot else {},
                "robot_position": robot_position or [0.0, 0.0, 0.0],
                "task_objects": task_objects,
                "table_replacement": table_replacement,
                "output": composed_usd,
            }
            manifest_path = os.path.join(output_dir, f"{scene_stem}_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2)

            logger.info(
                "📋 Composition manifest saved (USD composition requires " "Isaac Sim runtime): %s",
                manifest_path,
            )
            composed_usd = manifest_path

        result = {
            "scene_usd": composed_usd,
            "background": scene_path,
            "robot": robot,
            "task_objects": task_objects,
            "table_replacement": table_replacement,
            "output_dir": output_dir,
        }

        return result

    def _try_usd_composition(
        self,
        scene_path: str,
        composed_usd: str,
        robot: Optional[str],
        task_objects: List[str],
        table_replacement: bool,
        robot_position: Optional[List[float]],
        **kwargs: Any,
    ) -> bool:
        """Attempt USD scene composition via OpenUSD."""
        try:
            from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # type: ignore  # noqa: F401
        except ImportError:
            logger.debug("OpenUSD not available for scene composition.")
            return False

        try:
            stage = Usd.Stage.CreateNew(composed_usd)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)

            # Create root
            UsdGeom.Xform.Define(stage, "/World")

            # Add physics scene
            physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
            physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
            physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

            # Reference the background scene
            bg_prim = UsdGeom.Xform.Define(stage, "/World/Background")
            bg_prim.GetPrim().GetReferences().AddReference(scene_path)

            # Add robot
            if robot and robot in SUPPORTED_ROBOTS:
                robot_info = SUPPORTED_ROBOTS[robot]
                robot_xform = UsdGeom.Xform.Define(stage, "/World/Robot")
                pos = robot_position or [0.0, 0.0, robot_info["mount_height"]]
                robot_xform.AddTranslateOp().Set(Gf.Vec3d(*pos))

                # Reference robot USD (path relative to Isaac Sim assets)
                robot_xform.GetPrim().GetReferences().AddReference(robot_info["usd_path"])

            # Add table if replacement requested
            if table_replacement:
                table = UsdGeom.Cube.Define(stage, "/World/Table")
                table.GetSizeAttr().Set(1.0)
                table_xform = UsdGeom.Xformable(table)
                table_xform.AddTranslateOp().Set(Gf.Vec3d(0.4, 0.0, 0.35))
                table_xform.AddScaleOp().Set(Gf.Vec3f(0.8, 0.6, 0.7))

                # Add collision
                UsdPhysics.CollisionAPI.Apply(table.GetPrim())

            # Add task objects
            for i, obj_name in enumerate(task_objects):
                obj_path = f"/World/Objects/{obj_name}_{i}"
                obj_prim = UsdGeom.Sphere.Define(stage, obj_path)
                obj_prim.GetRadiusAttr().Set(0.03)
                obj_xform = UsdGeom.Xformable(obj_prim)
                # Distribute objects on the table
                x = 0.3 + (i % 3) * 0.1
                y = -0.1 + (i // 3) * 0.1
                z = 0.75
                obj_xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))

                UsdPhysics.CollisionAPI.Apply(obj_prim.GetPrim())
                UsdPhysics.RigidBodyAPI.Apply(obj_prim.GetPrim())

            # Add default lighting
            UsdGeom.Xform.Define(stage, "/World/Light")
            # Dome light for ambient
            from pxr import UsdLux  # type: ignore

            dome = UsdLux.DomeLight.Define(stage, "/World/Light/Dome")
            dome.CreateIntensityAttr().Set(1000.0)

            stage.GetRootLayer().Save()
            return True

        except Exception:
            logger.debug("USD scene composition failed.", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Stage 4: Batch generation for domain randomisation
    # ------------------------------------------------------------------

    def generate_training_scenes(
        self,
        prompts: Optional[List[str]] = None,
        preset: Optional[str] = None,
        num_per_prompt: int = 10,
        robot: Optional[str] = None,
        output_dir: str = "./marble_training_scenes",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a diverse set of training scenes for domain randomisation.

        Creates multiple scene variations from one or more prompts, useful
        for training robust policies that generalise across environments.

        Args:
            prompts: List of text prompts for scene generation. Mutually
                exclusive with ``preset``.
            preset: Name of a preset from ``MARBLE_PRESETS``. Uses the
                preset's prompt and task objects.
            num_per_prompt: Number of variations per prompt.
            robot: Robot to compose into scenes.
            output_dir: Base output directory.
            **kwargs: Additional generation parameters.

        Returns:
            Dictionary with generation summary including paths to all scenes.

        Example::

            result = pipeline.generate_training_scenes(
                preset="kitchen",
                num_per_prompt=50,
                robot="so101",
            )
            # result["scenes"] contains 50 MarbleScene objects
        """
        if preset and preset in MARBLE_PRESETS:
            preset_info = MARBLE_PRESETS[preset]
            if prompts is None:
                prompts = [preset_info["prompt"]]
            task_objects = preset_info.get("task_objects", [])
            logger.info("🎯 Using preset '%s': %s", preset, preset_info["description"])
        else:
            task_objects = kwargs.pop("task_objects", [])

        if not prompts:
            raise ValueError(
                "Either 'prompts' or a valid 'preset' must be provided. "
                f"Available presets: {sorted(MARBLE_PRESETS.keys())}"
            )

        robot = robot or self.config.robot
        all_scenes: List[MarbleScene] = []

        for prompt_idx, prompt in enumerate(prompts):
            prompt_dir = os.path.join(output_dir, f"prompt_{prompt_idx:03d}")
            scenes = self.generate_world(
                prompt=prompt,
                output_dir=prompt_dir,
                num_variations=num_per_prompt,
                **kwargs,
            )

            # Compose each scene with robot and objects
            if robot:
                for scene in scenes:
                    bg_path = scene.best_background
                    if bg_path:
                        try:
                            composed = self.compose_scene(
                                scene_path=bg_path,
                                robot=robot,
                                task_objects=task_objects,
                            )
                            scene.scene_usd = composed.get("scene_usd")
                            scene.robot = robot
                            scene.task_objects = task_objects
                            scene.composed = True
                        except Exception as exc:
                            logger.warning(
                                "Composition failed for %s: %s",
                                scene.scene_id,
                                exc,
                            )

            all_scenes.extend(scenes)

        result = {
            "total_scenes": len(all_scenes),
            "composed_scenes": sum(1 for s in all_scenes if s.composed),
            "prompts": prompts,
            "preset": preset,
            "robot": robot,
            "task_objects": task_objects,
            "output_dir": output_dir,
            "scenes": all_scenes,
            "scene_paths": [
                s.scene_usd or s.usdz_path or s.ply_path for s in all_scenes if s.scene_usd or s.usdz_path or s.ply_path
            ],
        }

        logger.info(
            "✅ Training scene generation complete: %d total, %d composed",
            result["total_scenes"],
            result["composed_scenes"],
        )
        return result

    # ------------------------------------------------------------------
    # Marble API interaction (OpenAPI v1 — api.worldlabs.ai)
    # ------------------------------------------------------------------

    def _get_api_headers(self) -> Dict[str, str]:
        """Build API request headers with WLT-Api-Key auth."""
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["WLT-Api-Key"] = self.config.api_key
        return headers

    def _call_marble_api(
        self,
        prompt: str,
        input_mode: str,
        output_dir: str,
        seed: int,
        input_image: Optional[str] = None,
        input_video: Optional[str] = None,
        input_layout: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Call the Marble world model API.

        Dispatches to the real API at ``api.worldlabs.ai`` when an API key
        is configured, otherwise falls back to placeholder generation.

        The real API flow is:
        1. (Optional) Upload media via ``/marble/v1/media-assets:prepare_upload``
        2. Submit generation via ``POST /marble/v1/worlds:generate``
        3. Poll ``GET /marble/v1/operations/{operation_id}`` until ``done=true``
        4. Download assets from the returned ``World`` object

        Args:
            prompt: Generation prompt.
            input_mode: Input modality type (text, image, video, multi-image).
            output_dir: Output directory for downloaded results.
            seed: Random seed.
            input_image: Path to input image (if applicable).
            input_video: Path to input video (if applicable).
            input_layout: Path to 3D layout (if applicable).

        Returns:
            Dictionary with paths to generated outputs.
        """
        result: Dict[str, Any] = {"metadata": {"prompt": prompt, "seed": seed}}

        # Check if API key is available
        if self.config.api_key and self.config.api_url != "local":
            try:
                return self._call_marble_api_remote(
                    prompt=prompt,
                    input_mode=input_mode,
                    output_dir=output_dir,
                    seed=seed,
                    input_image=input_image,
                    input_video=input_video,
                    input_layout=input_layout,
                    **kwargs,
                )
            except Exception as exc:
                logger.warning("Marble API call failed, creating placeholder: %s", exc)

        # Create placeholder outputs for offline/local mode
        logger.info(
            "📝 Creating placeholder scene (Marble API not configured).\n"
            "   Set WLT_API_KEY or MARBLE_API_KEY env var for real generation.\n"
            "   API docs: https://docs.worldlabs.ai/api/reference/openapi\n"
            "   Placeholders can be replaced with actual Marble exports."
        )

        # Create a minimal placeholder PLY
        ply_path = os.path.join(output_dir, "scene.ply")
        self._create_placeholder_ply(ply_path, prompt)
        result["ply_path"] = ply_path

        # Create placeholder GLB metadata
        glb_path = os.path.join(output_dir, "scene.glb")
        result["glb_path"] = glb_path

        # Save generation metadata
        meta_path = os.path.join(output_dir, "generation_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "prompt": prompt,
                    "input_mode": input_mode,
                    "seed": seed,
                    "model": self.config.model,
                    "api_url": self.config.api_url,
                    "placeholder": True,
                    "instructions": (
                        "Replace scene.ply and scene.glb with actual Marble exports. "
                        "Set WLT_API_KEY env var and re-run, or upload manually at "
                        "https://marble.worldlabs.ai"
                    ),
                },
                fh,
                indent=2,
            )

        result["metadata"]["placeholder"] = True
        return result

    def _build_world_prompt(
        self,
        prompt: str,
        input_mode: str,
        input_image: Optional[str] = None,
        input_video: Optional[str] = None,
        media_asset_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build a ``world_prompt`` object matching the OpenAPI WorldsGenerateRequest schema.

        Supports the discriminated union types:
        - ``text`` → WorldTextPrompt
        - ``image`` → ImagePrompt
        - ``video`` → VideoPrompt
        - ``multi-image`` → MultiImagePrompt
        """
        if input_mode == "text":
            return {
                "type": "text",
                "text_prompt": prompt,
            }

        elif input_mode == "image":
            image_source: Dict[str, Any]
            if media_asset_id:
                image_source = {
                    "source": "media_asset",
                    "media_asset_id": media_asset_id,
                }
            elif input_image and os.path.isfile(input_image):
                with open(input_image, "rb") as f:
                    data_b64 = base64.b64encode(f.read()).decode()
                ext = Path(input_image).suffix.lstrip(".")
                image_source = {
                    "source": "data_base64",
                    "data_base64": data_b64,
                    "extension": ext or "jpg",
                }
            elif input_image and input_image.startswith(("http://", "https://")):
                image_source = {
                    "source": "uri",
                    "uri": input_image,
                }
            else:
                raise ValueError("image mode requires input_image (file path or URL) " "or media_asset_id")

            return {
                "type": "image",
                "image_prompt": image_source,
                "text_prompt": prompt or None,
                "is_pano": kwargs.get("is_pano", False),
            }

        elif input_mode == "video":
            video_source: Dict[str, Any]
            if media_asset_id:
                video_source = {
                    "source": "media_asset",
                    "media_asset_id": media_asset_id,
                }
            elif input_video and os.path.isfile(input_video):
                with open(input_video, "rb") as f:
                    data_b64 = base64.b64encode(f.read()).decode()
                ext = Path(input_video).suffix.lstrip(".")
                video_source = {
                    "source": "data_base64",
                    "data_base64": data_b64,
                    "extension": ext or "mp4",
                }
            elif input_video and input_video.startswith(("http://", "https://")):
                video_source = {
                    "source": "uri",
                    "uri": input_video,
                }
            else:
                raise ValueError("video mode requires input_video (file path or URL) " "or media_asset_id")

            return {
                "type": "video",
                "video_prompt": video_source,
                "text_prompt": prompt or None,
            }

        elif input_mode == "multi-image":
            # multi_image_prompt is a list of SphericallyLocatedContent
            multi_images = kwargs.get("multi_image_prompt", [])
            if not multi_images:
                raise ValueError(
                    "multi-image mode requires 'multi_image_prompt' kwarg — "
                    "a list of dicts with 'content' and optional 'azimuth'"
                )
            return {
                "type": "multi-image",
                "multi_image_prompt": multi_images,
                "text_prompt": prompt or None,
                "reconstruct_images": kwargs.get("reconstruct_images", False),
            }

        else:
            raise ValueError(f"Unsupported input_mode: {input_mode}")

    def _call_marble_api_remote(
        self,
        prompt: str,
        input_mode: str,
        output_dir: str,
        seed: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Call the Marble API at api.worldlabs.ai per OpenAPI v1 spec.

        Flow:
        1. (Optional) Upload media assets
        2. POST /marble/v1/worlds:generate → operation_id
        3. Poll GET /marble/v1/operations/{operation_id} until done
        4. Extract World assets and download them
        """
        try:
            import requests  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("requests is required for Marble API calls. " "Install with: pip install requests")

        headers = self._get_api_headers()

        # Step 1: Upload media asset if needed
        media_asset_id = kwargs.pop("media_asset_id", None)
        input_image = kwargs.pop("input_image", None)
        input_video = kwargs.pop("input_video", None)
        # Also remove input_layout if present (not used by API)
        kwargs.pop("input_layout", None)

        if not media_asset_id and input_mode == "image" and input_image and os.path.isfile(input_image):
            try:
                media_asset_id = self._upload_media_asset(file_path=input_image, kind="image", headers=headers)
                logger.info("Uploaded image as media asset: %s", media_asset_id)
            except Exception as exc:
                logger.debug("Media asset upload failed, using data_base64 fallback: %s", exc)

        if not media_asset_id and input_mode == "video" and input_video and os.path.isfile(input_video):
            try:
                media_asset_id = self._upload_media_asset(file_path=input_video, kind="video", headers=headers)
                logger.info("Uploaded video as media asset: %s", media_asset_id)
            except Exception as exc:
                logger.debug("Media asset upload failed, using data_base64 fallback: %s", exc)

        # Step 2: Build WorldsGenerateRequest
        world_prompt = self._build_world_prompt(
            prompt=prompt,
            input_mode=input_mode,
            input_image=input_image,
            input_video=input_video,
            media_asset_id=media_asset_id,
            **kwargs,
        )

        display_name = kwargs.get("display_name") or prompt[:80]
        tags = kwargs.get("tags")

        payload: Dict[str, Any] = {
            "world_prompt": world_prompt,
            "model": self.config.model,
            "permission": {
                "public": self.config.is_public,
            },
        }
        if display_name:
            payload["display_name"] = display_name
        if seed is not None:
            payload["seed"] = seed
        if tags:
            payload["tags"] = tags

        logger.info(
            "🌍 POST %s/marble/v1/worlds:generate (model=%s, type=%s)",
            self.config.api_url,
            self.config.model,
            input_mode,
        )

        # Step 3: Submit generation
        response = requests.post(
            f"{self.config.api_url}/marble/v1/worlds:generate",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        gen_data = response.json()
        operation_id = gen_data["operation_id"]
        logger.info("Generation submitted: operation_id=%s", operation_id)

        # Step 4: Poll for completion
        world = self._poll_operation(operation_id, headers)

        # Step 5: Extract assets and download
        result: Dict[str, Any] = {
            "metadata": {
                "prompt": prompt,
                "seed": seed,
                "operation_id": operation_id,
                "world_id": world.get("world_id"),
                "model": world.get("model"),
                "world_marble_url": world.get("world_marble_url"),
                "display_name": world.get("display_name"),
                "caption": None,
            },
        }

        assets = world.get("assets") or {}

        # Caption
        if assets.get("caption"):
            result["metadata"]["caption"] = assets["caption"]

        # Thumbnail
        thumbnail_url = assets.get("thumbnail_url")
        if thumbnail_url:
            result["metadata"]["thumbnail_url"] = thumbnail_url
            self._download_file(
                thumbnail_url,
                os.path.join(output_dir, "thumbnail.jpg"),
                headers,
            )

        # Panorama image
        imagery = assets.get("imagery") or {}
        pano_url = imagery.get("pano_url")
        if pano_url:
            result["metadata"]["pano_url"] = pano_url
            local_pano = os.path.join(output_dir, "pano.jpg")
            self._download_file(pano_url, local_pano, headers)
            result["pano_path"] = local_pano

        # Mesh (collider)
        mesh_assets = assets.get("mesh") or {}
        collider_url = mesh_assets.get("collider_mesh_url")
        if collider_url:
            result["metadata"]["collider_mesh_url"] = collider_url
            glb_path = os.path.join(output_dir, "scene.glb")
            self._download_file(collider_url, glb_path, headers)
            result["glb_path"] = glb_path

        # Gaussian splats (SPZ format — compressed splat, NOT PLY)
        splat_assets = assets.get("splats") or {}
        spz_urls = splat_assets.get("spz_urls") or {}
        if spz_urls:
            result["metadata"]["spz_urls"] = spz_urls
            # Pick the best SPZ variant: prefer 500k (good balance),
            # then full_res, then 100k, then whatever is first
            preferred_order = ["500k", "full_res", "100k"]
            chosen_key = None
            for pref in preferred_order:
                if pref in spz_urls:
                    chosen_key = pref
                    break
            if chosen_key is None:
                chosen_key = next(iter(spz_urls))

            spz_url = spz_urls[chosen_key]
            spz_local = os.path.join(output_dir, f"scene_{chosen_key}.spz")
            self._download_file(spz_url, spz_local, headers)
            result["spz_path"] = spz_local
            logger.info("Downloaded SPZ Gaussian splat (%s): %s", chosen_key, spz_local)

        # Save full world metadata
        meta_path = os.path.join(output_dir, "world_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(world, fh, indent=2, default=str)

        return result

    def _poll_operation(
        self,
        operation_id: str,
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        """Poll ``GET /marble/v1/operations/{operation_id}`` until done.

        Returns the ``response`` field (a ``World`` object) when complete.

        Raises:
            RuntimeError: If the operation fails or times out.
        """
        import requests  # type: ignore[import-untyped]

        url = f"{self.config.api_url}/marble/v1/operations/{operation_id}"
        deadline = time.time() + self.config.poll_timeout

        while time.time() < deadline:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get("done"):
                # Check for error
                error = data.get("error")
                if error and (error.get("message") or error.get("code")):
                    raise RuntimeError(
                        f"Marble generation failed: " f"code={error.get('code')}, message={error.get('message')}"
                    )
                # Return the World object
                world = data.get("response")
                if world is None:
                    raise RuntimeError("Operation completed but no world in response")
                return world

            # Log progress
            metadata = data.get("metadata") or {}
            progress = metadata.get("progress", "unknown")
            logger.info(
                "⏳ Operation %s in progress (%s)...",
                operation_id,
                progress,
            )

            time.sleep(self.config.poll_interval)

        raise RuntimeError(
            f"Marble generation timed out after {self.config.poll_timeout}s " f"(operation_id={operation_id})"
        )

    def _upload_media_asset(
        self,
        file_path: str,
        kind: str,
        headers: Dict[str, str],
    ) -> str:
        """Upload a media asset via the two-step prepare + PUT flow.

        1. POST /marble/v1/media-assets:prepare_upload
        2. PUT the file to the signed upload_url

        Args:
            file_path: Local path to the file.
            kind: ``"image"`` or ``"video"``.
            headers: API request headers.

        Returns:
            The ``media_asset_id`` for use in generation requests.
        """
        import requests  # type: ignore[import-untyped]

        file_name = Path(file_path).name
        extension = Path(file_path).suffix.lstrip(".")

        # Step 1: Prepare upload
        prepare_payload = {
            "file_name": file_name,
            "kind": kind,
            "extension": extension or None,
        }

        resp = requests.post(
            f"{self.config.api_url}/marble/v1/media-assets:prepare_upload",
            headers=headers,
            json=prepare_payload,
            timeout=30,
        )
        resp.raise_for_status()
        prepare_data = resp.json()

        media_asset = prepare_data["media_asset"]
        upload_info = prepare_data["upload_info"]
        media_asset_id = media_asset["media_asset_id"]
        upload_url = upload_info["upload_url"]
        required_headers = upload_info.get("required_headers") or {}

        # Step 2: Upload the file via PUT
        with open(file_path, "rb") as f:
            file_data = f.read()

        upload_headers = dict(required_headers)
        resp = requests.request(
            upload_info.get("upload_method", "PUT"),
            upload_url,
            headers=upload_headers,
            data=file_data,
            timeout=120,
        )
        resp.raise_for_status()

        logger.info("📤 Uploaded media asset: %s (%s)", media_asset_id, file_name)
        return media_asset_id

    def get_world(self, world_id: str) -> Dict[str, Any]:
        """Get a world by ID via ``GET /marble/v1/worlds/{world_id}``.

        Args:
            world_id: The world identifier.

        Returns:
            World object dictionary.
        """
        import requests  # type: ignore[import-untyped]

        headers = self._get_api_headers()
        resp = requests.get(
            f"{self.config.api_url}/marble/v1/worlds/{world_id}",
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def list_worlds(
        self,
        page_size: int = 20,
        status: Optional[str] = None,
        model: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_public: Optional[bool] = None,
        sort_by: str = "created_at",
        page_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List worlds via ``POST /marble/v1/worlds:list``.

        Args:
            page_size: Number of results per page (1-100).
            status: Filter by status (SUCCEEDED, PENDING, FAILED, RUNNING).
            model: Filter by model (``"Marble 0.1-mini"`` or ``"Marble 0.1-plus"``).
            tags: Filter by tags (any match).
            is_public: Filter by visibility.
            sort_by: Sort by ``"created_at"`` or ``"updated_at"``.
            page_token: Pagination cursor from a previous response.

        Returns:
            ``ListWorldsResponse`` with ``worlds`` list and optional
            ``next_page_token``.
        """
        import requests  # type: ignore[import-untyped]

        headers = self._get_api_headers()
        payload: Dict[str, Any] = {
            "page_size": page_size,
            "sort_by": sort_by,
        }
        if status:
            payload["status"] = status
        if model:
            payload["model"] = model
        if tags:
            payload["tags"] = tags
        if is_public is not None:
            payload["is_public"] = is_public
        if page_token:
            payload["page_token"] = page_token

        resp = requests.post(
            f"{self.config.api_url}/marble/v1/worlds:list",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _download_file(
        url: str,
        local_path: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """Download a file from URL to local path."""
        import requests  # type: ignore[import-untyped]

        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        resp = requests.get(url, headers=headers, timeout=120, stream=True)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.debug("Downloaded: %s → %s", url[:80], local_path)
        return local_path

    @staticmethod
    def _create_placeholder_ply(ply_path: str, prompt: str) -> None:
        """Create a minimal placeholder PLY file with a simple point cloud."""
        import random

        random.seed(hash(prompt) % (2**32))

        num_points = 1000
        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {num_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )

        with open(ply_path, "w") as f:
            f.write(header)
            for _ in range(num_points):
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                z = random.uniform(0, 1)
                r = random.randint(100, 200)
                g = random.randint(100, 200)
                b = random.randint(100, 200)
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _run_subprocess(self, cmd: List[str], label: str) -> str:
        """Run a subprocess with logging and error handling."""
        logger.info("Running %s: %s", label, " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{label} timed out after 300s")
        except OSError as exc:
            raise RuntimeError(f"Failed to launch {label}: {exc}") from exc

        if result.returncode != 0:
            logger.error("%s STDOUT:\n%s", label, result.stdout)
            logger.error("%s STDERR:\n%s", label, result.stderr)
            raise RuntimeError(f"{label} failed (rc={result.returncode}): " f"{result.stderr[-500:]}")

        logger.debug("%s output:\n%s", label, result.stdout)

        # Try to find the output file from stdout
        for line in result.stdout.strip().split("\n"):
            if line.strip().endswith((".usdz", ".usda", ".usd")):
                return line.strip()

        return cmd[-1]  # Return the last argument (typically output path)

    def get_generated_scenes(self) -> List[MarbleScene]:
        """Return all scenes generated in this pipeline session."""
        return list(self._generated_scenes)

    def cleanup(self) -> None:
        """Remove temporary directories created during pipeline execution."""
        for tmp_dir in self._tmp_dirs:
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
                logger.debug("Cleaned up: %s", tmp_dir)
        self._tmp_dirs.clear()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"MarblePipeline("
            f"format={self.config.output_format!r}, "
            f"robot={self.config.robot!r}, "
            f"variations={self.config.num_variations}, "
            f"scenes={len(self._generated_scenes)})"
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def generate_world(
    prompt: str,
    output_dir: str = "./marble_scenes",
    robot: Optional[str] = None,
    compose: bool = False,
    num_variations: int = 1,
    config: Optional[MarbleConfig] = None,
    **kwargs: Any,
) -> Union[MarbleScene, List[MarbleScene]]:
    """One-shot convenience function for Marble world generation.

    Creates a :class:`MarblePipeline`, generates world(s), optionally
    composes with a robot, and returns the result.

    Args:
        prompt: Text prompt describing the desired 3D world.
        output_dir: Output directory.
        robot: Robot to compose into the scene.
        compose: Whether to auto-compose with robot.
        num_variations: Number of scene variations.
        config: Optional :class:`MarbleConfig`.
        **kwargs: Additional parameters forwarded to the pipeline.

    Returns:
        A single :class:`MarbleScene` if ``num_variations == 1``,
        otherwise a list of scenes.

    Example::

        scene = generate_world(
            prompt="A kitchen table with fruits and plates",
            robot="so101",
            compose=True,
        )
        print(scene.scene_usd)
    """
    config_fields = {f.name for f in MarbleConfig.__dataclass_fields__.values()}
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
    gen_kwargs = {k: v for k, v in kwargs.items() if k not in config_fields}

    if config is None:
        config = MarbleConfig(
            robot=robot,
            auto_compose=compose,
            num_variations=num_variations,
            **config_kwargs,
        )

    pipeline = MarblePipeline(config)
    try:
        scenes = pipeline.generate_world(
            prompt=prompt,
            output_dir=output_dir,
            num_variations=num_variations,
            **gen_kwargs,
        )
        if len(scenes) == 1:
            return scenes[0]
        return scenes
    finally:
        pipeline.cleanup()


def compose_scene(
    scene_path: str,
    robot: str,
    task_objects: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """One-shot convenience function for scene composition.

    Args:
        scene_path: Path to background scene.
        robot: Robot key.
        task_objects: Task objects to place.
        output_dir: Output directory.

    Returns:
        Composition result dictionary.
    """
    pipeline = MarblePipeline(MarbleConfig(robot=robot))
    return pipeline.compose_scene(
        scene_path=scene_path,
        robot=robot,
        task_objects=task_objects,
        output_dir=output_dir,
        **kwargs,
    )


def list_presets() -> List[Dict[str, str]]:
    """List available Marble scene presets.

    Returns:
        List of preset info dictionaries.
    """
    presets = []
    for name, info in MARBLE_PRESETS.items():
        presets.append(
            {
                "name": name,
                "description": info["description"],
                "category": info["category"],
                "prompt": info["prompt"][:80] + "...",
                "objects": ", ".join(info.get("task_objects", [])),
            }
        )
    return presets
