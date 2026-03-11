"""Cosmos Transfer 2.5 integration module for strands-robots.

Provides sim-to-real visual augmentation by leveraging NVIDIA's Cosmos Transfer
model to transform simulated robot videos into photorealistic outputs using
controllable generation (depth, edge, segmentation, etc.).

Typical usage:

    from strands_robots.cosmos_transfer import CosmosTransferPipeline, CosmosTransferConfig

    config = CosmosTransferConfig(model_variant="depth", num_gpus=2)
    pipeline = CosmosTransferPipeline(config)

    result = pipeline.transfer_video(
        sim_video_path="sim_output.mp4",
        prompt="A robot arm picking up a red cube on a wooden table",
        output_path="real_output.mp4",
        control_types=["depth", "edge"],
    )

Or use the convenience function:

    from strands_robots.cosmos_transfer import transfer_video

    result = transfer_video(
        sim_video_path="sim_output.mp4",
        prompt="A robot arm picking up a red cube on a wooden table",
        output_path="real_output.mp4",
    )

References:
    - Cosmos Transfer 2.5: https://github.com/NVIDIA/Cosmos
    - Video Depth Anything: https://github.com/DepthAnything/Video-Depth-Anything
    - SAM2: https://github.com/facebookresearch/sam2
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["CosmosTransferPipeline", "CosmosTransferConfig", "transfer_video"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_MODEL_VARIANTS: Tuple[str, ...] = (
    "depth",
    "edge",
    "seg",
    "vis",
    "robot/multiview-agibot-depth",
    "robot/multiview-agibot-edge",
    "robot/multiview-agibot-seg",
    "robot/multiview-gr1-depth",
    "robot/multiview-gr1-edge",
    "robot/multiview-gr1-seg",
)

VALID_OUTPUT_RESOLUTIONS: Tuple[str, ...] = ("480", "720", "1080")

VALID_EDGE_THRESHOLDS: Dict[str, Tuple[int, int]] = {
    "low": (50, 100),
    "medium": (100, 200),
    "high": (200, 400),
}

DEFAULT_NEGATIVE_PROMPT: str = (
    "The video captures a scene with low visual quality, "
    "appearing blurry, oversaturated, or poorly lit. "
    "The content is generic and unoriginal, lacking any creative flair. "
    "The composition is uninspired, with awkward framing and an "
    "unbalanced layout. The lighting is flat and uninteresting, "
    "casting harsh, unflattering shadows. The colors are dull "
    "and washed out, failing to create any visual impact. Overall, "
    "the video feels amateurish and hastily produced, with no "
    "attention to detail or artistic expression. "
    "The video has bad CG quality, is poorly rendered, and looks "
    "like a low-budget simulation or video game."
)

COSMOS_INFERENCE_SCRIPT: str = "cosmos_transfer2.inference"
COSMOS_EXAMPLES_INFERENCE: str = "examples/inference.py"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CosmosTransferConfig:
    """Configuration for the Cosmos Transfer 2.5 pipeline.

    Attributes:
        model_variant: The model variant to use. One of ``"depth"``,
            ``"edge"``, ``"seg"``, ``"vis"``,
            ``"robot/multiview-agibot-depth"``, etc.
        checkpoint_path: Explicit path to a Cosmos checkpoint directory.
            When *None* the pipeline will attempt to resolve the checkpoint
            from the ``COSMOS_CHECKPOINT_DIR`` environment variable or a
            default HuggingFace cache location.
        num_gpus: Number of GPUs to use for inference.  When > 1 the
            pipeline launches inference via ``torchrun``.
        guidance: Classifier-free guidance scale.
        num_steps: Number of denoising steps.
        control_weight: Weight applied to the control signal(s).
        seed: Random seed for reproducibility.
        negative_prompt: Negative prompt used for classifier-free guidance.
        output_resolution: Output video resolution height as a string.
            One of ``"480"``, ``"720"``, or ``"1080"``.
        enable_autoregressive: If *True*, enable autoregressive generation
            for long videos.  The video is split into overlapping chunks
            and generated sequentially.
        num_chunks: Number of autoregressive chunks (only used when
            ``enable_autoregressive`` is *True*).
        chunk_overlap: Number of overlapping frames between consecutive
            chunks (only used when ``enable_autoregressive`` is *True*).
    """

    model_variant: str = "depth"
    checkpoint_path: Optional[str] = None
    cosmos_transfer_path: Optional[str] = None
    num_gpus: int = 1
    guidance: float = 3.0
    num_steps: int = 35
    control_weight: float = 1.0
    seed: int = 2025
    disable_guardrails: bool = True
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    output_resolution: str = "720"
    enable_autoregressive: bool = False
    num_chunks: int = 2
    chunk_overlap: int = 1

    def __post_init__(self) -> None:
        """Validate configuration values after initialisation."""
        if self.model_variant not in VALID_MODEL_VARIANTS:
            raise ValueError(
                f"Invalid model_variant '{self.model_variant}'. "
                f"Must be one of: {VALID_MODEL_VARIANTS}"
            )

        if self.output_resolution not in VALID_OUTPUT_RESOLUTIONS:
            raise ValueError(
                f"Invalid output_resolution '{self.output_resolution}'. "
                f"Must be one of: {VALID_OUTPUT_RESOLUTIONS}"
            )

        if self.num_gpus < 1:
            raise ValueError(f"num_gpus must be >= 1, got {self.num_gpus}")

        if self.guidance < 0.0:
            raise ValueError(f"guidance must be >= 0.0, got {self.guidance}")

        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")

        if not 0.0 <= self.control_weight <= 2.0:
            raise ValueError(
                f"control_weight should be in [0.0, 2.0], got {self.control_weight}"
            )

        if self.num_chunks < 1:
            raise ValueError(f"num_chunks must be >= 1, got {self.num_chunks}")

        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")

    def resolve_checkpoint_path(self) -> str:
        """Resolve the checkpoint path from config, env var, or default.

        Returns:
            The resolved checkpoint directory path.

        Raises:
            FileNotFoundError: If no valid checkpoint path can be resolved.
        """
        if self.checkpoint_path and os.path.isdir(self.checkpoint_path):
            return self.checkpoint_path

        env_path = os.environ.get("COSMOS_CHECKPOINT_DIR")
        if env_path and os.path.isdir(env_path):
            logger.info(
                "Using checkpoint path from COSMOS_CHECKPOINT_DIR: %s", env_path
            )
            return env_path

        # Fallback: HuggingFace hub cache default location
        default_hf = os.path.expanduser(
            "~/.cache/huggingface/hub/models--nvidia--Cosmos-Transfer2-7B"
        )
        if os.path.isdir(default_hf):
            logger.info("Using default HuggingFace cache checkpoint: %s", default_hf)
            return default_hf

        raise FileNotFoundError(
            "Could not resolve a Cosmos checkpoint path. "
            "Set CosmosTransferConfig.checkpoint_path, the "
            "COSMOS_CHECKPOINT_DIR environment variable, or download the "
            "model to the default HuggingFace cache."
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class CosmosTransferPipeline:
    """End-to-end pipeline for Cosmos Transfer 2.5 sim-to-real augmentation.

    The pipeline orchestrates:
    1. Control signal generation (depth, edge, segmentation) from simulation
       video.
    2. Building a Cosmos-compatible inference specification.
    3. Launching Cosmos inference via ``torchrun`` (multi-GPU) or ``python``
       (single GPU).

    Args:
        config: A :class:`CosmosTransferConfig` instance.  When *None* a
            default configuration is used.
        **kwargs: Additional keyword arguments forwarded to
            :class:`CosmosTransferConfig` when ``config`` is *None*.

    Example::

        pipeline = CosmosTransferPipeline(
            CosmosTransferConfig(num_gpus=2, guidance=4.0)
        )
        result = pipeline.transfer_video(
            sim_video_path="sim.mp4",
            prompt="Robot arm on a desk",
            output_path="output.mp4",
        )
    """

    def __init__(
        self,
        config: Optional[CosmosTransferConfig] = None,
        **kwargs: Any,
    ) -> None:
        if config is not None and kwargs:
            raise ValueError(
                "Cannot specify both 'config' and keyword arguments. "
                "Pass either a CosmosTransferConfig instance OR keyword "
                "arguments, not both."
            )

        if config is None:
            config = CosmosTransferConfig(**kwargs)

        self.config = config
        self._tmp_dirs: List[str] = []

        logger.info(
            "Initialised CosmosTransferPipeline (variant=%s, gpus=%d)",
            self.config.model_variant,
            self.config.num_gpus,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transfer_video(
        self,
        sim_video_path: str,
        prompt: str,
        output_path: str,
        control_types: Optional[List[str]] = None,
        control_weights: Optional[List[float]] = None,
        depth_video_path: Optional[str] = None,
        edge_video_path: Optional[str] = None,
        seg_video_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run Cosmos Transfer on a simulation video.

        This is the primary entry-point.  It will:

        1. Auto-generate any missing control videos (depth from the
           simulator or a monocular depth model, edges via Canny, and
           segmentation masks via SAM2).
        2. Build a Cosmos inference spec JSON file.
        3. Shell out to the Cosmos inference script.
        4. Return a result dictionary with metadata.

        Args:
            sim_video_path: Path to the simulation video (MP4).
            prompt: Text prompt describing the desired real-world scene.
            output_path: Destination path for the generated video.
            control_types: List of control signal types to use.  Defaults
                to ``["depth"]``.  Valid values: ``"depth"``, ``"edge"``,
                ``"seg"``.
            control_weights: Per-control-type weights.  When *None* all
                controls use ``config.control_weight``.
            depth_video_path: Pre-computed depth control video.  When
                *None* depth is generated automatically.
            edge_video_path: Pre-computed edge control video.  When
                *None* edges are generated automatically.
            seg_video_path: Pre-computed segmentation control video.  When
                *None* segmentation is generated automatically (SAM2).
            **kwargs: Override any :class:`CosmosTransferConfig` field for
                this invocation only.

        Returns:
            A dictionary containing:
                - ``output_path`` (str): Path to the generated video.
                - ``frame_count`` (int): Number of frames in the output.
                - ``control_types`` (list[str]): Controls that were used.
                - ``control_video_paths`` (dict[str, str]): Mapping from
                  control type to the control video that was used.
                - ``spec_path`` (str): Path to the inference spec JSON.
                - ``seed`` (int): The random seed that was used.

        Raises:
            FileNotFoundError: If ``sim_video_path`` does not exist.
            ValueError: If invalid control types are specified.
            RuntimeError: If Cosmos inference fails.
        """
        # -- Validate inputs ------------------------------------------------
        sim_video_path = str(sim_video_path)
        output_path = str(output_path)

        if not os.path.isfile(sim_video_path):
            raise FileNotFoundError(f"Simulation video not found: {sim_video_path}")

        if control_types is None:
            control_types = ["depth"]

        valid_control_types = {"depth", "edge", "seg"}
        for ct in control_types:
            if ct not in valid_control_types:
                raise ValueError(
                    f"Invalid control type '{ct}'. "
                    f"Must be one of: {sorted(valid_control_types)}"
                )

        if control_weights is not None and len(control_weights) != len(control_types):
            raise ValueError(
                f"control_weights length ({len(control_weights)}) must match "
                f"control_types length ({len(control_types)})"
            )

        # -- Merge per-call overrides with config --------------------------
        effective_config = self._merge_config_overrides(**kwargs)

        # -- Ensure output directory exists ---------------------------------
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        work_dir = tempfile.mkdtemp(prefix="cosmos_transfer_")
        self._tmp_dirs.append(work_dir)
        logger.debug("Working directory: %s", work_dir)

        # -- Generate / resolve control videos -----------------------------
        control_video_paths: Dict[str, str] = {}
        precomputed_map = {
            "depth": depth_video_path,
            "edge": edge_video_path,
            "seg": seg_video_path,
        }

        for ct in control_types:
            precomputed = precomputed_map.get(ct)
            if precomputed and os.path.isfile(precomputed):
                logger.info("Using pre-computed %s control: %s", ct, precomputed)
                control_video_paths[ct] = precomputed
            else:
                if precomputed:
                    logger.warning(
                        "Pre-computed %s control not found at '%s'; "
                        "generating automatically.",
                        ct,
                        precomputed,
                    )
                logger.info("Auto-generating %s control video…", ct)
                generated = self._generate_control(ct, sim_video_path, work_dir)
                control_video_paths[ct] = generated

        # -- Build weights list --------------------------------------------
        if control_weights is None:
            control_weights = [effective_config.control_weight] * len(control_types)

        # -- Build inference spec ------------------------------------------
        spec = self._build_inference_spec(
            sim_video_path=sim_video_path,
            prompt=prompt,
            output_path=output_path,
            control_types=control_types,
            control_video_paths=control_video_paths,
            control_weights=control_weights,
            config=effective_config,
        )

        spec_path = os.path.join(work_dir, "inference_spec.json")
        with open(spec_path, "w", encoding="utf-8") as fh:
            json.dump(spec, fh, indent=2)
        logger.info("Inference spec written to %s", spec_path)

        # -- Run inference -------------------------------------------------
        generated_video = self._run_inference(
            spec_path=spec_path,
            output_dir=os.path.dirname(output_path) or ".",
            config=effective_config,
        )

        # -- If cosmos writes to a different location, copy to output_path -
        if generated_video != output_path and os.path.isfile(generated_video):
            shutil.copy2(generated_video, output_path)
            logger.info("Copied result to %s", output_path)

        # -- Count frames in the output ------------------------------------
        frame_count = self._count_frames(output_path)

        result: Dict[str, Any] = {
            "output_path": output_path,
            "frame_count": frame_count,
            "control_types": control_types,
            "control_video_paths": control_video_paths,
            "spec_path": spec_path,
            "seed": effective_config.seed,
        }

        logger.info("Transfer complete → %s (%d frames)", output_path, frame_count)
        return result

    def set_sim_context(self, model, data):
        """Set MuJoCo model/data for direct depth buffer access."""
        self._sim_model = model
        self._sim_data = data
        logger.info("MuJoCo sim context set for depth generation")

    def generate_depth_control(
        self,
        sim_video_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a depth control video from simulation footage.

        Strategy:
        1. **MuJoCo depth buffer** – If ``mujoco`` is importable and the
           video was produced by a MuJoCo-based simulation, attempt to read
           the depth buffer directly from the renderer.  The raw depth is
           normalised to 0–255 and encoded as a greyscale video.
        2. **Video Depth Anything fallback** – If MuJoCo is unavailable or
           the depth buffer cannot be obtained, run the Video Depth Anything
           monocular depth estimator on the RGB frames.

        Args:
            sim_video_path: Path to the simulation video.
            output_path: Destination path for the depth video.  When *None*
                a temporary file is created alongside the input.

        Returns:
            Path to the generated depth control video.

        Raises:
            FileNotFoundError: If ``sim_video_path`` does not exist.
            RuntimeError: If depth generation fails with all strategies.
        """
        sim_video_path = str(sim_video_path)
        if not os.path.isfile(sim_video_path):
            raise FileNotFoundError(f"Video not found: {sim_video_path}")

        if output_path is None:
            stem = Path(sim_video_path).stem
            output_path = str(Path(sim_video_path).parent / f"{stem}_depth_control.mp4")

        # Strategy 1: MuJoCo depth buffer
        if self._try_mujoco_depth(sim_video_path, output_path):
            logger.info("Depth control generated via MuJoCo: %s", output_path)
            return output_path

        # Strategy 2: Video Depth Anything
        if self._try_video_depth_anything(sim_video_path, output_path):
            logger.info(
                "Depth control generated via Video Depth Anything: %s",
                output_path,
            )
            return output_path

        # Strategy 3: OpenCV-based simple disparity (last resort)
        logger.warning(
            "MuJoCo and Video Depth Anything unavailable. "
            "Falling back to greyscale luminance as a pseudo-depth proxy."
        )
        self._fallback_greyscale_depth(sim_video_path, output_path)
        return output_path

    def generate_edge_control(
        self,
        sim_video_path: str,
        output_path: Optional[str] = None,
        threshold: str = "medium",
    ) -> str:
        """Generate an edge control video using Canny edge detection.

        Each frame of the input video is converted to greyscale, blurred
        slightly to reduce noise, and passed through ``cv2.Canny`` with
        configurable thresholds.

        Args:
            sim_video_path: Path to the simulation video.
            output_path: Destination path for the edge video.  When *None*
                a temporary file is created alongside the input.
            threshold: Edge detection sensitivity.  One of ``"low"``,
                ``"medium"``, or ``"high"``.

        Returns:
            Path to the generated edge control video.

        Raises:
            FileNotFoundError: If ``sim_video_path`` does not exist.
            ValueError: If ``threshold`` is not a recognised preset.
            ImportError: If OpenCV is not installed.
            RuntimeError: If edge generation fails.
        """
        try:
            import cv2  # type: ignore[import-untyped]
            import numpy as np  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "OpenCV (cv2) and NumPy are required for edge control "
                "generation. Install them with: pip install opencv-python numpy"
            ) from exc

        sim_video_path = str(sim_video_path)
        if not os.path.isfile(sim_video_path):
            raise FileNotFoundError(f"Video not found: {sim_video_path}")

        if threshold not in VALID_EDGE_THRESHOLDS:
            raise ValueError(
                f"Invalid threshold '{threshold}'. "
                f"Must be one of: {sorted(VALID_EDGE_THRESHOLDS.keys())}"
            )

        low_thresh, high_thresh = VALID_EDGE_THRESHOLDS[threshold]

        if output_path is None:
            stem = Path(sim_video_path).stem
            output_path = str(Path(sim_video_path).parent / f"{stem}_edge_control.mp4")

        cap = cv2.VideoCapture(sim_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {sim_video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not writer.isOpened():
                raise RuntimeError(f"Cannot create video writer for: {output_path}")

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(grey, (5, 5), 1.4)
                edges = cv2.Canny(blurred, low_thresh, high_thresh)
                # Convert back to 3-channel for video encoding
                edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                writer.write(edges_bgr)
                frame_idx += 1

            writer.release()
        finally:
            cap.release()

        logger.info(
            "Edge control generated (%d frames, threshold=%s): %s",
            frame_idx,
            threshold,
            output_path,
        )
        return output_path

    def generate_seg_control(
        self,
        sim_video_path: str,
        output_path: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """Generate a segmentation control video.

        This method attempts to use **SAM2** (Segment Anything Model 2) to
        produce per-frame segmentation masks.  If SAM2 is not available it
        falls back to a simple colour-based segmentation placeholder.

        Args:
            sim_video_path: Path to the simulation video.
            output_path: Destination path for the segmentation video.
                When *None* a temporary file is created alongside the input.
            prompt: Optional text prompt for prompted segmentation (SAM2).
                When *None* automatic mask generation is used.

        Returns:
            Path to the generated segmentation control video.

        Raises:
            FileNotFoundError: If ``sim_video_path`` does not exist.
            RuntimeError: If segmentation generation fails.
        """
        try:
            import cv2  # type: ignore[import-untyped]
            import numpy as np  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "OpenCV (cv2) and NumPy are required for segmentation control "
                "generation. Install them with: pip install opencv-python numpy"
            ) from exc

        sim_video_path = str(sim_video_path)
        if not os.path.isfile(sim_video_path):
            raise FileNotFoundError(f"Video not found: {sim_video_path}")

        if output_path is None:
            stem = Path(sim_video_path).stem
            output_path = str(Path(sim_video_path).parent / f"{stem}_seg_control.mp4")

        # Try SAM2 first
        if self._try_sam2_segmentation(sim_video_path, output_path, prompt):
            logger.info("Segmentation control generated via SAM2: %s", output_path)
            return output_path

        # Fallback: simple colour-quantised segmentation
        logger.warning(
            "SAM2 not available. Falling back to colour-quantised "
            "segmentation placeholder."
        )
        self._fallback_colour_segmentation(sim_video_path, output_path)
        return output_path

    # ------------------------------------------------------------------
    # Inference spec & execution
    # ------------------------------------------------------------------

    def _build_inference_spec(
        self,
        sim_video_path: str,
        prompt: str,
        output_path: str,
        control_types: List[str],
        control_video_paths: Dict[str, str],
        control_weights: List[float],
        config: CosmosTransferConfig,
    ) -> Dict[str, Any]:
        """Build the JSON inference specification for Cosmos.

        The spec follows the schema expected by the Cosmos Transfer 2.5
        inference script, including control inputs, generation parameters,
        and autoregressive settings.

        Args:
            sim_video_path: Path to the original simulation video.
            prompt: Text prompt for generation.
            output_path: Desired output video path.
            control_types: List of control signal types.
            control_video_paths: Mapping of control type → video path.
            control_weights: Per-control weights.
            config: Effective configuration for this run.

        Returns:
            A dictionary representing the inference specification.
        """
        # Build spec matching Cosmos Transfer 2.5 real inference format
        # See: cosmos-transfer2.5/assets/robot_example/depth/robot_depth_spec.json
        spec: Dict[str, Any] = {
            "name": os.path.splitext(os.path.basename(output_path))[0],
            "prompt": prompt,
            "video_path": os.path.abspath(sim_video_path),
            "guidance": config.guidance,
            "num_steps": config.num_steps,
        }

        # Add negative prompt if non-default
        if config.negative_prompt:
            spec["negative_prompt"] = config.negative_prompt

        # Add control configs (depth, edge, seg, vis) as top-level keys
        for ct, weight in zip(control_types, control_weights):
            spec[ct] = {
                "control_path": os.path.abspath(control_video_paths[ct]),
                "control_weight": weight,
            }

        return spec

    def _run_inference(
        self,
        spec_path: str,
        output_dir: str,
        config: Optional[CosmosTransferConfig] = None,
    ) -> str:
        """Execute Cosmos inference as a subprocess.

        For multi-GPU (``num_gpus > 1``) the inference is launched via
        ``torchrun --nproc_per_node=<num_gpus>``.  For single-GPU it
        falls back to a plain ``python -m`` invocation.

        Args:
            spec_path: Path to the inference spec JSON.
            output_dir: Directory where Cosmos will write its output.
            config: Effective config (uses ``self.config`` when *None*).

        Returns:
            Path to the generated video (as reported by Cosmos or inferred
            from ``output_dir``).

        Raises:
            RuntimeError: If the subprocess exits with a non-zero code.
            FileNotFoundError: If ``torchrun`` or ``python`` cannot be found.
        """
        config = config or self.config

        # Read spec to find the expected output path
        with open(spec_path, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
        expected_output = spec.get("output", {}).get("video_path", "")

        # Resolve Cosmos Transfer installation path
        cosmos_root = (
            config.cosmos_transfer_path
            or os.environ.get("COSMOS_TRANSFER_PATH")
            or os.environ.get("COSMOS_TRANSFER2_PATH")
        )
        # Auto-detect from installed package location
        if not cosmos_root:
            try:
                import cosmos_transfer2

                cosmos_root = str(Path(cosmos_transfer2.__file__).parent.parent)
            except (ImportError, AttributeError):
                pass

        inference_script = None
        if cosmos_root:
            candidate = os.path.join(cosmos_root, COSMOS_EXAMPLES_INFERENCE)
            if os.path.isfile(candidate):
                inference_script = candidate

        python_bin = shutil.which("python3") or shutil.which("python")
        if python_bin is None:
            raise FileNotFoundError("python/python3 not found on PATH.")

        # Build command — use examples/inference.py with real CLI format
        if inference_script:
            # Real CLI: python examples/inference.py -i spec.json -o outdir [--disable-guardrails] depth
            # Determine control subcommand from spec
            control_subcommand = "depth"  # default
            with open(spec_path, "r") as _f:
                _spec = json.load(_f)
                for ct in ("edge", "seg", "vis", "depth"):
                    if ct in _spec:
                        control_subcommand = ct
                        break

            if config.num_gpus > 1:
                torchrun = shutil.which("torchrun")
                if torchrun is None:
                    raise FileNotFoundError("torchrun not found.")
                cmd = [
                    torchrun,
                    f"--nproc_per_node={config.num_gpus}",
                    inference_script,
                    "-i",
                    spec_path,
                    "-o",
                    os.path.dirname(spec_path) or output_dir,
                ]
            else:
                cmd = [
                    python_bin,
                    inference_script,
                    "-i",
                    spec_path,
                    "-o",
                    os.path.dirname(spec_path) or output_dir,
                ]

            if getattr(config, "disable_guardrails", True):
                cmd.append("--disable-guardrails")
            cmd.append(control_subcommand)
        else:
            # Fallback: try module invocation (may not work with all versions)
            logger.warning(
                "Cosmos examples/inference.py not found. Trying module invocation."
            )
            cmd = [python_bin, "-m", COSMOS_INFERENCE_SCRIPT, "--spec", spec_path]

        logger.info("Running inference: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                cwd=output_dir,
            )
        except OSError as exc:
            raise RuntimeError(f"Failed to launch inference subprocess: {exc}") from exc

        if result.returncode != 0:
            logger.error("Cosmos inference STDOUT:\n%s", result.stdout)
            logger.error("Cosmos inference STDERR:\n%s", result.stderr)
            raise RuntimeError(
                f"Cosmos inference failed with return code "
                f"{result.returncode}. See logs for details.\n"
                f"STDERR (last 500 chars): {result.stderr[-500:]}"
            )

        logger.debug("Cosmos inference STDOUT:\n%s", result.stdout)

        if expected_output and os.path.isfile(expected_output):
            return expected_output

        # Try to find the output in the output directory
        mp4_files = sorted(
            Path(output_dir).glob("*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if mp4_files:
            return str(mp4_files[0])

        logger.warning(
            "Could not locate generated video in %s; returning expected path.",
            output_dir,
        )
        return expected_output

    # ------------------------------------------------------------------
    # Control generation helpers (private)
    # ------------------------------------------------------------------

    def _generate_control(
        self, control_type: str, sim_video_path: str, work_dir: str
    ) -> str:
        """Dispatch control generation by type.

        Args:
            control_type: One of ``"depth"``, ``"edge"``, ``"seg"``.
            sim_video_path: Input simulation video.
            work_dir: Temporary working directory for outputs.

        Returns:
            Path to the generated control video.
        """
        stem = Path(sim_video_path).stem
        output_path = os.path.join(work_dir, f"{stem}_{control_type}_control.mp4")

        if control_type == "depth":
            return self.generate_depth_control(sim_video_path, output_path)
        elif control_type == "edge":
            return self.generate_edge_control(sim_video_path, output_path)
        elif control_type == "seg":
            return self.generate_seg_control(sim_video_path, output_path)
        else:
            raise ValueError(f"Unknown control type: {control_type}")

    def _try_mujoco_depth(self, sim_video_path: str, output_path: str) -> bool:
        """Attempt depth generation via MuJoCo depth buffer.

        Tries to import ``mujoco`` and ``strands_robots.video.VideoEncoder``
        to read the depth buffer directly from the MuJoCo renderer and
        encode it as a normalised greyscale video.

        Args:
            sim_video_path: Path to the simulation video (used to read
                metadata such as frame count / fps).
            output_path: Destination for the depth video.

        Returns:
            *True* if depth was generated successfully, *False* otherwise.
        """
        try:
            import cv2  # type: ignore[import-untyped]
            import mujoco  # type: ignore[import-untyped]
            import numpy as np  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("MuJoCo or OpenCV/NumPy not available for depth generation.")
            return False

        try:
            from strands_robots.video import VideoEncoder  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("strands_robots.video.VideoEncoder not available.")
            return False

        try:
            # Read the video to get frame dimensions and FPS
            cap = cv2.VideoCapture(sim_video_path)
            if not cap.isOpened():
                return False

            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Attempt to access the MuJoCo model/data from strands_robots
            # global simulation context (if one is active).
            # Try to get model/data from the pipeline's sim_context (if set)
            model = getattr(self, "_sim_model", None)
            data = getattr(self, "_sim_data", None)
            if model is None or data is None:
                logger.debug(
                    "No MuJoCo sim context set. Use pipeline.set_sim_context(model, data)."
                )
                return False

            renderer = mujoco.Renderer(model, height=height, width=width)
            renderer.enable_depth_rendering()

            encoder = VideoEncoder(output_path, fps=fps, width=width, height=height)

            frame_count = int(
                cv2.VideoCapture(sim_video_path).get(cv2.CAP_PROP_FRAME_COUNT)
            )
            for _ in range(frame_count):
                renderer.update_scene(data)
                depth = renderer.render()

                # Normalise depth to 0-255
                depth_min = depth.min()
                depth_max = depth.max()
                if depth_max - depth_min > 1e-6:
                    depth_norm = (depth - depth_min) / (depth_max - depth_min) * 255.0
                else:
                    depth_norm = np.zeros_like(depth)

                depth_u8 = depth_norm.astype(np.uint8)
                depth_bgr = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)
                encoder.write_frame(depth_bgr)

                mujoco.mj_step(model, data)

            encoder.close()
            renderer.close()

            logger.info("MuJoCo depth buffer rendered %d frames.", frame_count)
            return True

        except Exception:
            logger.debug("MuJoCo depth generation failed.", exc_info=True)
            return False

    def _try_video_depth_anything(self, sim_video_path: str, output_path: str) -> bool:
        """Attempt depth generation via Video Depth Anything.

        Args:
            sim_video_path: Path to the input video.
            output_path: Destination for the depth video.

        Returns:
            *True* if depth was generated successfully, *False* otherwise.
        """
        try:
            import cv2  # type: ignore[import-untyped]
            import numpy as np  # type: ignore[import-untyped]
            from video_depth_anything.infer import (  # type: ignore[import-untyped]
                VideoDepthAnythingPredictor,
            )
        except ImportError:
            logger.debug("Video Depth Anything not available.")
            return False

        try:
            predictor = VideoDepthAnythingPredictor()

            cap = cv2.VideoCapture(sim_video_path)
            if not cap.isOpened():
                return False

            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            if not frames:
                return False

            depth_maps = predictor.predict(frames)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for depth in depth_maps:
                depth_arr = np.array(depth, dtype=np.float32)
                d_min, d_max = depth_arr.min(), depth_arr.max()
                if d_max - d_min > 1e-6:
                    depth_norm = (depth_arr - d_min) / (d_max - d_min) * 255.0
                else:
                    depth_norm = np.zeros_like(depth_arr)
                depth_u8 = depth_norm.astype(np.uint8)
                depth_bgr = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)
                writer.write(depth_bgr)

            writer.release()
            logger.info("Video Depth Anything generated %d depth frames.", len(frames))
            return True

        except Exception:
            logger.debug("Video Depth Anything inference failed.", exc_info=True)
            return False

    def _fallback_greyscale_depth(self, sim_video_path: str, output_path: str) -> None:
        """Last-resort depth proxy using luminance-based greyscale conversion.

        This is **not** a true depth map but provides a usable control
        signal when no depth estimator is available.

        Args:
            sim_video_path: Path to the input video.
            output_path: Destination for the pseudo-depth video.

        Raises:
            ImportError: If OpenCV is not available.
            RuntimeError: If the video cannot be processed.
        """
        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required for fallback depth generation."
            ) from exc

        cap = cv2.VideoCapture(sim_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {sim_video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not writer.isOpened():
                raise RuntimeError(f"Cannot create video writer: {output_path}")

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                grey_bgr = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
                writer.write(grey_bgr)
                frame_count += 1

            writer.release()
        finally:
            cap.release()

        logger.info(
            "Fallback greyscale depth generated (%d frames): %s",
            frame_count,
            output_path,
        )

    def _try_sam2_segmentation(
        self,
        sim_video_path: str,
        output_path: str,
        prompt: Optional[str] = None,
    ) -> bool:
        """Attempt segmentation via SAM2 (Segment Anything Model 2).

        Args:
            sim_video_path: Path to the input video.
            output_path: Destination for the segmentation video.
            prompt: Optional text prompt for prompted segmentation.

        Returns:
            *True* if segmentation was generated successfully, *False*
            otherwise.
        """
        try:
            import cv2  # type: ignore[import-untyped]
            import numpy as np  # type: ignore[import-untyped]
            import torch  # type: ignore[import-untyped]
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("SAM2 not available for segmentation.")
            return False

        try:
            # Build the SAM2 video predictor with default checkpoint
            sam2_checkpoint = os.environ.get("SAM2_CHECKPOINT", "sam2_hiera_large.pt")
            model_cfg = os.environ.get("SAM2_MODEL_CFG", "sam2_hiera_l.yaml")

            if not os.path.isfile(sam2_checkpoint):
                logger.debug("SAM2 checkpoint not found at %s", sam2_checkpoint)
                return False

            predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

            cap = cv2.VideoCapture(sim_video_path)
            if not cap.isOpened():
                return False

            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            if not frames:
                return False

            # Use automatic mask generation (no prompt)
            with torch.inference_mode():
                state = predictor.init_state(video_path=sim_video_path)

                # Add initial points at centre of frame for automatic
                # segmentation
                _, _, masks = predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    points=np.array([[width // 2, height // 2]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),
                )

                # Propagate through the video
                video_segments = {}
                for (
                    frame_idx,
                    obj_ids,
                    mask_logits,
                ) in predictor.propagate_in_video(state):
                    mask = (mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)
                    video_segments[frame_idx] = mask * 255

            # Encode segmentation masks as video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for idx in range(len(frames)):
                if idx in video_segments:
                    seg_mask = video_segments[idx]
                    if seg_mask.ndim == 2:
                        seg_bgr = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)
                    else:
                        seg_bgr = seg_mask
                else:
                    seg_bgr = np.zeros((height, width, 3), dtype=np.uint8)
                writer.write(seg_bgr)

            writer.release()
            logger.info("SAM2 segmentation generated %d frames.", len(frames))
            return True

        except Exception:
            logger.debug("SAM2 segmentation failed.", exc_info=True)
            return False

    def _fallback_colour_segmentation(
        self, sim_video_path: str, output_path: str
    ) -> None:
        """Fallback colour-quantised segmentation.

        Uses k-means colour clustering to produce a rough semantic-style
        segmentation map.

        Args:
            sim_video_path: Path to the input video.
            output_path: Destination for the segmentation video.

        Raises:
            ImportError: If OpenCV/NumPy are not available.
            RuntimeError: If the video cannot be processed.
        """
        try:
            import cv2  # type: ignore[import-untyped]
            import numpy as np  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "OpenCV and NumPy are required for fallback segmentation."
            ) from exc

        cap = cv2.VideoCapture(sim_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {sim_video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n_clusters = 8

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not writer.isOpened():
                raise RuntimeError(f"Cannot create video writer: {output_path}")

            # Predefined palette for segmentation visualisation
            palette = np.array(
                [
                    [0, 0, 0],  # background – black
                    [128, 0, 0],  # maroon
                    [0, 128, 0],  # green
                    [128, 128, 0],  # olive
                    [0, 0, 128],  # navy
                    [128, 0, 128],  # purple
                    [0, 128, 128],  # teal
                    [192, 192, 192],  # silver
                ],
                dtype=np.uint8,
            )

            frame_count = 0
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                10,
                1.0,
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                pixels = frame.reshape((-1, 3)).astype(np.float32)
                _, labels, _ = cv2.kmeans(
                    pixels,
                    n_clusters,
                    None,
                    criteria,
                    3,
                    cv2.KMEANS_PP_CENTERS,
                )
                labels = labels.flatten() % len(palette)
                seg_frame = palette[labels].reshape((height, width, 3))
                writer.write(seg_frame)
                frame_count += 1

            writer.release()
        finally:
            cap.release()

        logger.info(
            "Fallback colour segmentation generated (%d frames): %s",
            frame_count,
            output_path,
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _merge_config_overrides(self, **kwargs: Any) -> CosmosTransferConfig:
        """Create a new config with per-call overrides merged in.

        Args:
            **kwargs: Fields of :class:`CosmosTransferConfig` to override.

        Returns:
            A new (validated) :class:`CosmosTransferConfig` instance.
        """
        if not kwargs:
            return self.config

        base = asdict(self.config)
        base.update(kwargs)
        return CosmosTransferConfig(**base)

    @staticmethod
    def _count_frames(video_path: str) -> int:
        """Count the number of frames in a video file.

        Args:
            video_path: Path to the video.

        Returns:
            Number of frames, or ``0`` if the file cannot be read.
        """
        try:
            import cv2  # type: ignore[import-untyped]

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return max(count, 0)
        except Exception:
            logger.debug("Could not count frames in %s", video_path, exc_info=True)
            return 0

    def cleanup(self) -> None:
        """Remove temporary working directories created during transfers.

        This is safe to call multiple times.  Directories that have already
        been removed are silently skipped.
        """
        for tmp_dir in self._tmp_dirs:
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
                logger.debug("Cleaned up temporary directory: %s", tmp_dir)
        self._tmp_dirs.clear()

    def __del__(self) -> None:
        """Destructor – attempt to clean up temp directories."""
        try:
            self.cleanup()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"CosmosTransferPipeline("
            f"variant={self.config.model_variant!r}, "
            f"gpus={self.config.num_gpus}, "
            f"guidance={self.config.guidance}, "
            f"steps={self.config.num_steps})"
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def transfer_video(
    sim_video_path: str,
    prompt: str,
    output_path: str,
    control_types: Optional[List[str]] = None,
    control_weights: Optional[List[float]] = None,
    depth_video_path: Optional[str] = None,
    edge_video_path: Optional[str] = None,
    seg_video_path: Optional[str] = None,
    config: Optional[CosmosTransferConfig] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """One-shot convenience function for Cosmos Transfer sim-to-real.

    Creates a :class:`CosmosTransferPipeline`, runs
    :meth:`~CosmosTransferPipeline.transfer_video`, cleans up, and
    returns the result.

    Args:
        sim_video_path: Path to the simulation video (MP4).
        prompt: Text prompt describing the desired real-world scene.
        output_path: Destination path for the generated video.
        control_types: List of control signal types (default: ``["depth"]``).
        control_weights: Per-control-type weights.
        depth_video_path: Pre-computed depth control video.
        edge_video_path: Pre-computed edge control video.
        seg_video_path: Pre-computed segmentation control video.
        config: Optional :class:`CosmosTransferConfig`.  When *None* a
            default config is used, potentially augmented by ``**kwargs``.
        **kwargs: Additional keyword arguments forwarded to
            :class:`CosmosTransferConfig` if ``config`` is *None*, or to
            :meth:`CosmosTransferPipeline.transfer_video` otherwise.

    Returns:
        A dictionary with the same schema as
        :meth:`CosmosTransferPipeline.transfer_video`.

    Example::

        result = transfer_video(
            sim_video_path="sim.mp4",
            prompt="A Franka robot picking up a mug in a kitchen",
            output_path="real.mp4",
            control_types=["depth", "edge"],
            num_gpus=2,
            guidance=4.0,
        )
    """
    # Separate config kwargs from transfer_video kwargs
    config_fields = {f.name for f in CosmosTransferConfig.__dataclass_fields__.values()}

    config_kwargs = {}
    transfer_kwargs = {}
    for key, value in kwargs.items():
        if key in config_fields:
            config_kwargs[key] = value
        else:
            transfer_kwargs[key] = value

    if config is None:
        config = CosmosTransferConfig(**config_kwargs)
    elif config_kwargs:
        logger.warning(
            "Config keyword arguments (%s) are ignored when an explicit "
            "'config' is provided.",
            list(config_kwargs.keys()),
        )

    pipeline = CosmosTransferPipeline(config)
    try:
        return pipeline.transfer_video(
            sim_video_path=sim_video_path,
            prompt=prompt,
            output_path=output_path,
            control_types=control_types,
            control_weights=control_weights,
            depth_video_path=depth_video_path,
            edge_video_path=edge_video_path,
            seg_video_path=seg_video_path,
            **transfer_kwargs,
        )
    finally:
        pipeline.cleanup()
