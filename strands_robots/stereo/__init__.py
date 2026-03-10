"""Fast-FoundationStereo integration for strands-robots.

Provides real-time zero-shot stereo depth estimation using NVIDIA's
Fast-FoundationStereo (CVPR 2026) — a knowledge-distilled model that runs
10× faster than FoundationStereo while closely matching its zero-shot accuracy.

The pipeline converts rectified stereo image pairs into dense disparity maps,
metric depth maps, and optionally 3D point clouds — critical capabilities for
robot manipulation, navigation, and obstacle avoidance.

Typical usage:

    from strands_robots.stereo import StereoDepthPipeline, StereoConfig

    config = StereoConfig(model_variant="23-36-37", valid_iters=8)
    pipeline = StereoDepthPipeline(config)

    result = pipeline.estimate_depth(
        left_image="left.png",
        right_image="right.png",
        intrinsic_matrix=K,  # 3×3 numpy array
        baseline=0.12,       # metres between cameras
    )
    # result.disparity  — (H, W) numpy array
    # result.depth      — (H, W) numpy array in metres
    # result.point_cloud — (H, W, 3) numpy array (x, y, z)

Or use the convenience function:

    from strands_robots.stereo import estimate_depth

    result = estimate_depth(
        left_image="left.png",
        right_image="right.png",
        intrinsic_matrix=K,
        baseline=0.12,
    )

Pipeline overview:

    Stereo Pair (left, right)
            ↓
    [Pad to divisible-by-32]
            ↓
    [FastFoundationStereo forward — GRU iterative refinement]
            ↓
    [Unpad + upscale disparity]
            ↓
    disparity → depth = f·b / d
            ↓
    depth → XYZ point cloud (with intrinsics)

Supported cameras:
    - Intel RealSense D4XX (stereo IR or RGB)
    - Stereolabs ZED / ZED 2 / ZED Mini
    - Any rectified stereo pair

References:
    - Fast-FoundationStereo: https://github.com/NVlabs/Fast-FoundationStereo
    - Paper: https://arxiv.org/abs/2512.11130 (CVPR 2026)
    - FoundationStereo: https://github.com/NVlabs/FoundationStereo
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

__all__ = [
    "StereoDepthPipeline",
    "StereoConfig",
    "StereoResult",
    "estimate_depth",
    "VALID_MODEL_VARIANTS",
    "SUPPORTED_CAMERAS",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_MODEL_VARIANTS: Tuple[str, ...] = (
    "23-36-37",  # Highest accuracy (49.4ms PyTorch, 23.4ms TRT on RTX 3090)
    "20-26-39",  # Balanced (43.6ms PyTorch, 19.4ms TRT)
    "20-30-48",  # Fastest (38.4ms PyTorch, 16.6ms TRT)
)

VALID_ITER_OPTIONS: Tuple[int, ...] = (4, 8, 12)

SUPPORTED_CAMERAS: Dict[str, Dict[str, Any]] = {
    "realsense_d435": {
        "type": "stereo_ir",
        "default_baseline": 0.050,
        "default_resolution": (640, 480),
        "description": "Intel RealSense D435 (stereo IR, 50mm baseline)",
    },
    "realsense_d455": {
        "type": "stereo_ir",
        "default_baseline": 0.095,
        "default_resolution": (640, 480),
        "description": "Intel RealSense D455 (stereo IR, 95mm baseline)",
    },
    "zed2": {
        "type": "stereo_rgb",
        "default_baseline": 0.120,
        "default_resolution": (1280, 720),
        "description": "Stereolabs ZED 2 (stereo RGB, 120mm baseline)",
    },
    "zed_mini": {
        "type": "stereo_rgb",
        "default_baseline": 0.063,
        "default_resolution": (1280, 720),
        "description": "Stereolabs ZED Mini (stereo RGB, 63mm baseline)",
    },
    "custom": {
        "type": "any",
        "default_baseline": None,
        "default_resolution": None,
        "description": "Custom rectified stereo pair",
    },
}

DEFAULT_DIVIS_BY: int = 32
DEFAULT_MAX_DISP: int = 192
DEFAULT_VALID_ITERS: int = 8
DEFAULT_SCALE: float = 1.0
DEFAULT_ZFAR: float = 100.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class StereoResult:
    """Result of stereo depth estimation.

    Attributes:
        disparity: Dense disparity map of shape ``(H, W)`` in pixels.
            Higher values indicate closer objects.
        depth: Metric depth map of shape ``(H, W)`` in metres.  Computed
            as ``focal_length * baseline / disparity``.  Infinite/invalid
            pixels are set to ``np.inf``.
        point_cloud: 3D point cloud of shape ``(H, W, 3)`` in camera
            coordinates (x-right, y-down, z-forward).  Only present when
            ``intrinsic_matrix`` and ``baseline`` are provided.
        disparity_vis: Colourised disparity visualisation ``(H, W, 3)`` uint8.
            Generated when ``visualize=True``.
        metadata: Dictionary with timing, configuration, and diagnostics.
    """

    disparity: np.ndarray
    depth: Optional[np.ndarray] = None
    point_cloud: Optional[np.ndarray] = None
    disparity_vis: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        """Image height."""
        return self.disparity.shape[0]

    @property
    def width(self) -> int:
        """Image width."""
        return self.disparity.shape[1]

    @property
    def median_depth(self) -> Optional[float]:
        """Median valid depth in metres."""
        if self.depth is None:
            return None
        valid = self.depth[np.isfinite(self.depth) & (self.depth > 0)]
        return float(np.median(valid)) if valid.size > 0 else None

    @property
    def valid_ratio(self) -> float:
        """Fraction of pixels with valid (finite, positive) disparity."""
        valid = (self.disparity > 0) & np.isfinite(self.disparity)
        return float(valid.sum()) / max(self.disparity.size, 1)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dictionary (without large arrays)."""
        return {
            "height": self.height,
            "width": self.width,
            "median_depth": self.median_depth,
            "valid_ratio": self.valid_ratio,
            "has_depth": self.depth is not None,
            "has_point_cloud": self.point_cloud is not None,
            "has_visualisation": self.disparity_vis is not None,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class StereoConfig:
    """Configuration for the stereo depth estimation pipeline.

    Attributes:
        model_variant: Checkpoint variant to use.  One of the keys from
            :data:`VALID_MODEL_VARIANTS`.  Determines the speed/accuracy
            trade-off.
        model_path: Explicit path to the model checkpoint (``.pth`` file).
            When *None*, the pipeline resolves from ``STEREO_MODEL_DIR``
            env var or a default weights directory.
        valid_iters: Number of GRU refinement iterations during inference.
            More iterations = better accuracy but slower.  Recommended:
            4 (fastest) or 8 (best accuracy).
        max_disp: Maximum disparity for the cost volume.  192 is sufficient
            for most scenes.  Increase for very close objects (< 0.1m).
        scale: Input image scaling factor.  Use < 1.0 to reduce resolution
            for faster inference.  Depth is rescaled to original resolution.
        hierarchical: If *True*, use hierarchical (coarse-to-fine) inference
            for better results on high-resolution images.
        remove_invisible: Remove depth for non-overlapping (occluded) regions
            at the left boundary.
        mixed_precision: Use FP16 mixed precision for faster inference.
        low_memory: Reduce peak memory at a small speed cost.
        device: CUDA device string (e.g. ``"cuda:0"``).
        zfar: Maximum depth (metres) to include in point cloud output.
        camera: Optional camera preset from :data:`SUPPORTED_CAMERAS`.
        denoise_cloud: Apply statistical outlier removal to point cloud.
        denoise_nb_points: Number of neighbours for outlier removal.
        denoise_radius: Radius for outlier removal (metres).
    """

    model_variant: str = "23-36-37"
    model_path: Optional[str] = None
    valid_iters: int = DEFAULT_VALID_ITERS
    max_disp: int = DEFAULT_MAX_DISP
    scale: float = DEFAULT_SCALE
    hierarchical: bool = False
    remove_invisible: bool = True
    mixed_precision: bool = True
    low_memory: bool = False
    device: str = "cuda:0"
    zfar: float = DEFAULT_ZFAR
    camera: Optional[str] = None
    denoise_cloud: bool = False
    denoise_nb_points: int = 30
    denoise_radius: float = 0.03

    def __post_init__(self) -> None:
        """Validate configuration after initialisation."""
        if self.model_variant not in VALID_MODEL_VARIANTS:
            raise ValueError(
                f"Invalid model_variant '{self.model_variant}'. " f"Must be one of: {VALID_MODEL_VARIANTS}"
            )

        if self.valid_iters < 1:
            raise ValueError(f"valid_iters must be >= 1, got {self.valid_iters}")

        if self.max_disp < 1:
            raise ValueError(f"max_disp must be >= 1, got {self.max_disp}")

        if not 0.0 < self.scale <= 2.0:
            raise ValueError(f"scale must be in (0.0, 2.0], got {self.scale}")

        if self.zfar <= 0:
            raise ValueError(f"zfar must be > 0, got {self.zfar}")

        if self.camera and self.camera not in SUPPORTED_CAMERAS:
            raise ValueError(f"Unknown camera '{self.camera}'. " f"Supported: {sorted(SUPPORTED_CAMERAS.keys())}")

    def resolve_model_path(self) -> str:
        """Resolve the model checkpoint path.

        Resolution order:
        1. Explicit ``model_path``
        2. ``STEREO_MODEL_DIR`` / ``FAST_FOUNDATION_STEREO_DIR`` env var
        3. Default ``./weights/<variant>/model_best_bp2_serialize.pth``

        Returns:
            Path to the model checkpoint file.

        Raises:
            FileNotFoundError: If no valid checkpoint can be found.
        """
        # 1. Explicit path
        if self.model_path and os.path.isfile(self.model_path):
            return self.model_path

        # 2. Environment variable
        for env_var in ("STEREO_MODEL_DIR", "FAST_FOUNDATION_STEREO_DIR"):
            env_dir = os.environ.get(env_var)
            if env_dir:
                candidate = os.path.join(env_dir, self.model_variant, "model_best_bp2_serialize.pth")
                if os.path.isfile(candidate):
                    logger.info("Using stereo model from %s: %s", env_var, candidate)
                    return candidate

        # 3. Default weights directory
        default_path = os.path.join("weights", self.model_variant, "model_best_bp2_serialize.pth")
        if os.path.isfile(default_path):
            return default_path

        raise FileNotFoundError(
            f"Could not resolve stereo model checkpoint for variant "
            f"'{self.model_variant}'. Set StereoConfig.model_path, the "
            f"STEREO_MODEL_DIR environment variable, or download weights "
            f"from the Fast-FoundationStereo repository."
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class StereoDepthPipeline:
    """End-to-end pipeline for stereo depth estimation.

    Wraps NVIDIA's Fast-FoundationStereo model to provide:

    1. **Disparity estimation** from rectified stereo image pairs.
    2. **Metric depth** computation using camera intrinsics and baseline.
    3. **3D point cloud** generation in camera coordinates.
    4. **Disparity visualisation** with colourmap rendering.

    The model is loaded lazily on first use to avoid GPU memory allocation
    until needed.

    Args:
        config: A :class:`StereoConfig` instance.  When *None* a default
            configuration is used.
        **kwargs: Additional keyword arguments forwarded to
            :class:`StereoConfig` when ``config`` is *None*.

    Example::

        pipeline = StereoDepthPipeline(StereoConfig(
            model_variant="23-36-37",
            valid_iters=8,
            scale=0.5,
        ))
        result = pipeline.estimate_depth(
            left_image="left.png",
            right_image="right.png",
            intrinsic_matrix=K,
            baseline=0.12,
        )
        print(f"Median depth: {result.median_depth:.2f}m")
    """

    def __init__(
        self,
        config: Optional[StereoConfig] = None,
        **kwargs: Any,
    ) -> None:
        if config is not None and kwargs:
            raise ValueError(
                "Cannot specify both 'config' and keyword arguments. "
                "Pass either a StereoConfig instance OR keyword arguments."
            )

        if config is None:
            config = StereoConfig(**kwargs)

        self.config = config
        self._model = None
        self._model_loaded = False

        logger.info(
            "Initialised StereoDepthPipeline (variant=%s, iters=%d, " "scale=%.2f, max_disp=%d)",
            self.config.model_variant,
            self.config.valid_iters,
            self.config.scale,
            self.config.max_disp,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the Fast-FoundationStereo model onto GPU.

        The model is loaded lazily on first call to :meth:`estimate_depth`.

        Raises:
            ImportError: If ``torch`` is not available.
            FileNotFoundError: If the model checkpoint cannot be found.
            RuntimeError: If CUDA is not available.
        """
        try:
            import torch  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for stereo depth estimation. " "Install with: pip install torch"
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Fast-FoundationStereo inference. " "No CUDA device found.")

        model_path = self.config.resolve_model_path()
        logger.info("Loading stereo model from %s", model_path)

        t0 = time.monotonic()
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.args.valid_iters = self.config.valid_iters
        model.args.max_disp = self.config.max_disp

        model = model.to(self.config.device).eval()
        torch.set_grad_enabled(False)

        self._model = model
        self._model_loaded = True

        dt = time.monotonic() - t0
        logger.info("Stereo model loaded in %.2fs (device=%s)", dt, self.config.device)

    @property
    def model(self):
        """Lazily-loaded model instance."""
        if not self._model_loaded:
            self._load_model()
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_depth(
        self,
        left_image: Union[str, np.ndarray],
        right_image: Union[str, np.ndarray],
        intrinsic_matrix: Optional[np.ndarray] = None,
        baseline: Optional[float] = None,
        visualize: bool = False,
        **kwargs: Any,
    ) -> StereoResult:
        """Estimate depth from a rectified stereo image pair.

        Args:
            left_image: Path to left image or ``(H, W, 3)`` uint8 array (RGB).
            right_image: Path to right image or ``(H, W, 3)`` uint8 array (RGB).
            intrinsic_matrix: 3×3 camera intrinsic matrix.  Required for
                metric depth and point cloud output.  When using
                ``config.camera``, defaults from the camera preset are used.
            baseline: Distance between stereo cameras in metres.  Required
                for metric depth.  When using ``config.camera``, the default
                baseline for the camera is used.
            visualize: If *True*, generate colourised disparity visualisation.
            **kwargs: Override any :class:`StereoConfig` field for this call.

        Returns:
            A :class:`StereoResult` containing disparity, depth, point cloud,
            and metadata.

        Raises:
            FileNotFoundError: If image paths do not exist.
            ValueError: If images have different shapes.
            RuntimeError: If inference fails.
        """
        import torch  # type: ignore[import-untyped]

        effective_config = self._merge_config_overrides(**kwargs)
        metadata: Dict[str, Any] = {
            "config": asdict(effective_config),
        }

        # Load images
        t0 = time.monotonic()
        img_left = self._load_image(left_image)
        img_right = self._load_image(right_image)
        metadata["image_load_time"] = time.monotonic() - t0

        if img_left.shape != img_right.shape:
            raise ValueError(f"Image shapes must match. Left: {img_left.shape}, " f"Right: {img_right.shape}")

        H_orig, W_orig = img_left.shape[:2]
        metadata["original_size"] = (H_orig, W_orig)

        # Apply scaling
        scale = effective_config.scale
        if scale != 1.0:
            try:
                import cv2  # type: ignore[import-untyped]

                img_l_scaled = cv2.resize(
                    img_left,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_LINEAR,
                )
                # Verify resize returned a real numpy array
                if not isinstance(img_l_scaled, np.ndarray):
                    raise TypeError("cv2.resize returned non-array")
                img_right = cv2.resize(
                    img_right,
                    (img_l_scaled.shape[1], img_l_scaled.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                img_left = img_l_scaled
            except Exception:
                logger.warning("OpenCV not available for image scaling; using original size.")
                scale = 1.0

        H, W = img_left.shape[:2]
        metadata["inference_size"] = (H, W)
        metadata["scale"] = scale

        # Convert to tensors — auto-detect device (fall back to CPU if CUDA unavailable)
        device = effective_config.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
            logger.debug("CUDA not available; falling back to CPU for tensors.")
        img0 = torch.as_tensor(img_left, dtype=torch.float32).to(device)[None].permute(0, 3, 1, 2)  # (1, 3, H, W)
        img1 = torch.as_tensor(img_right, dtype=torch.float32).to(device)[None].permute(0, 3, 1, 2)

        # Pad to divisible-by-32
        padder = _InputPadder(img0.shape, divis_by=DEFAULT_DIVIS_BY)
        img0_padded, img1_padded = padder.pad(img0, img1)

        # Inference
        t_infer = time.monotonic()
        use_amp = effective_config.mixed_precision and device.startswith("cuda")
        with torch.amp.autocast(
            "cuda" if device.startswith("cuda") else "cpu",
            enabled=use_amp,
            dtype=torch.float16 if use_amp else torch.bfloat16,
        ):
            if effective_config.hierarchical:
                disp = self.model.run_hierachical(
                    img0_padded,
                    img1_padded,
                    iters=effective_config.valid_iters,
                    test_mode=True,
                    low_memory=effective_config.low_memory,
                    small_ratio=0.5,
                )
            else:
                disp = self.model.forward(
                    img0_padded,
                    img1_padded,
                    iters=effective_config.valid_iters,
                    test_mode=True,
                    low_memory=effective_config.low_memory,
                )

        metadata["inference_time"] = time.monotonic() - t_infer

        # Unpad and convert to numpy
        disp = padder.unpad(disp.float())
        disp_np = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

        # Remove invisible regions (non-overlapping at left boundary)
        if effective_config.remove_invisible:
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            invalid = (xx - disp_np) < 0
            disp_np[invalid] = np.inf

        # Compute metric depth and point cloud
        depth_np = None
        point_cloud = None

        # Resolve camera defaults
        if baseline is None and effective_config.camera:
            cam_info = SUPPORTED_CAMERAS.get(effective_config.camera, {})
            baseline = cam_info.get("default_baseline")

        if intrinsic_matrix is not None and baseline is not None:
            K = intrinsic_matrix.copy().astype(np.float64)
            K[:2] *= scale
            focal = K[0, 0]

            with np.errstate(divide="ignore", invalid="ignore"):
                depth_np = (focal * baseline) / disp_np
            depth_np = np.where(np.isfinite(depth_np) & (depth_np > 0), depth_np, np.inf)

            # Point cloud
            point_cloud = _depth_to_xyz(depth_np, K)

            metadata["focal_length"] = float(focal)
            metadata["baseline"] = float(baseline)

        # Visualisation
        disparity_vis = None
        if visualize:
            disparity_vis = _visualize_disparity(disp_np)

        metadata["total_time"] = time.monotonic() - t0

        return StereoResult(
            disparity=disp_np,
            depth=depth_np,
            point_cloud=point_cloud,
            disparity_vis=disparity_vis,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image: Union[str, np.ndarray]) -> np.ndarray:
        """Load an image from path or validate a numpy array.

        Args:
            image: File path or ``(H, W, 3)`` uint8 RGB array.

        Returns:
            ``(H, W, 3)`` uint8 RGB numpy array.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the array has an unexpected shape.
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
                raise ValueError(f"Expected (H, W, 3) array, got shape {image.shape}")
            return image[..., :3].copy()

        path = str(image)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")

        try:
            import imageio.v2 as imageio  # type: ignore[import-untyped]

            img = imageio.imread(path)
        except ImportError:
            try:
                import cv2  # type: ignore[import-untyped]

                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"Failed to read image: {path}")
                img = img[..., ::-1]  # BGR → RGB
            except ImportError:
                raise ImportError(
                    "Either imageio or cv2 is required for image loading. "
                    "Install with: pip install imageio  OR  pip install opencv-python"
                )

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img[..., :3].copy()

    def _merge_config_overrides(self, **kwargs: Any) -> StereoConfig:
        """Merge per-call overrides with the pipeline config."""
        if not kwargs:
            return self.config
        base = asdict(self.config)
        base.update(kwargs)
        return StereoConfig(**base)

    def __repr__(self) -> str:
        return (
            f"StereoDepthPipeline("
            f"variant={self.config.model_variant!r}, "
            f"iters={self.config.valid_iters}, "
            f"scale={self.config.scale}, "
            f"loaded={self._model_loaded})"
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _depth_to_xyz(
    depth: np.ndarray,
    K: np.ndarray,
    zmin: float = 0.1,
) -> np.ndarray:
    """Convert depth map to XYZ point cloud using camera intrinsics.

    Args:
        depth: ``(H, W)`` depth map in metres.
        K: 3×3 camera intrinsic matrix.
        zmin: Minimum valid depth.

    Returns:
        ``(H, W, 3)`` point cloud in camera coordinates.
    """
    H, W = depth.shape[:2]
    invalid_mask = (depth < zmin) | ~np.isfinite(depth)

    vs, us = np.meshgrid(np.arange(H), np.arange(W), sparse=False, indexing="ij")

    zs = depth.copy()
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]

    xyz = np.stack([xs, ys, zs], axis=-1).astype(np.float32)
    xyz[invalid_mask] = 0.0
    return xyz


def _visualize_disparity(
    disp: np.ndarray,
    invalid_thres: float = np.inf,
) -> np.ndarray:
    """Create a colourised disparity visualisation.

    Args:
        disp: ``(H, W)`` disparity map.
        invalid_thres: Disparity values ≥ this are treated as invalid.

    Returns:
        ``(H, W, 3)`` uint8 RGB visualisation.
    """
    # Try OpenCV colormap, with full fallback to greyscale
    disp = disp.copy()
    valid = np.isfinite(disp) & (disp < invalid_thres) & (disp >= 0)
    if not valid.any():
        return np.zeros((*disp.shape, 3), dtype=np.uint8)

    d_min, d_max = float(disp[valid].min()), float(disp[valid].max())
    if d_max - d_min < 1e-6:
        norm = np.zeros_like(disp)
    else:
        norm = ((disp - d_min) / (d_max - d_min)).clip(0, 1) * 255

    try:
        import cv2 as _cv2  # type: ignore[import-untyped]

        vis = _cv2.applyColorMap(norm.astype(np.uint8), _cv2.COLORMAP_TURBO)
        # Ensure we have a real numpy array (not a mock)
        if not isinstance(vis, np.ndarray):
            raise TypeError("cv2 returned non-array")
        vis = vis[..., ::-1].copy()  # BGR → RGB
    except Exception:
        # Greyscale fallback (no cv2 or cv2 is mocked)
        grey = norm.astype(np.uint8)
        vis = np.stack([grey, grey, grey], axis=-1)

    vis[~valid] = 0
    return vis.astype(np.uint8)


class _InputPadder:
    """Pad images so dimensions are divisible by a given factor.

    This is a pure-Python/PyTorch re-implementation of the ``InputPadder``
    from Fast-FoundationStereo, avoiding the need to import the upstream
    codebase at runtime.
    """

    def __init__(
        self,
        dims: Tuple[int, ...],
        divis_by: int = 32,
        mode: str = "sintel",
    ) -> None:
        ht, wd = dims[-2:]
        pad_ht = (((ht // divis_by) + 1) * divis_by - ht) % divis_by
        pad_wd = (((wd // divis_by) + 1) * divis_by - wd) % divis_by
        if mode == "sintel":
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        import torch.nn.functional as F  # type: ignore[import-untyped]

        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [
            self._pad[2],
            ht - self._pad[3],
            self._pad[0],
            wd - self._pad[1],
        ]
        return x[..., c[0] : c[1], c[2] : c[3]]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def estimate_depth(
    left_image: Union[str, np.ndarray],
    right_image: Union[str, np.ndarray],
    intrinsic_matrix: Optional[np.ndarray] = None,
    baseline: Optional[float] = None,
    visualize: bool = False,
    config: Optional[StereoConfig] = None,
    **kwargs: Any,
) -> StereoResult:
    """One-shot convenience function for stereo depth estimation.

    Creates a :class:`StereoDepthPipeline`, runs estimation, and returns
    the result.

    Args:
        left_image: Path or array for the left image.
        right_image: Path or array for the right image.
        intrinsic_matrix: 3×3 camera intrinsic matrix (optional).
        baseline: Stereo baseline in metres (optional).
        visualize: Generate colourised disparity visualisation.
        config: Optional :class:`StereoConfig`.
        **kwargs: Additional parameters forwarded to :class:`StereoConfig`.

    Returns:
        A :class:`StereoResult` with disparity, depth, and point cloud.

    Example::

        import numpy as np
        K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=float)
        result = estimate_depth(
            "left.png", "right.png",
            intrinsic_matrix=K, baseline=0.12,
            model_variant="20-30-48", valid_iters=4,
        )
    """
    config_fields = {f.name for f in StereoConfig.__dataclass_fields__.values()}
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
    estimate_kwargs = {k: v for k, v in kwargs.items() if k not in config_fields}

    if config is None:
        config = StereoConfig(**config_kwargs)

    pipeline = StereoDepthPipeline(config)
    return pipeline.estimate_depth(
        left_image=left_image,
        right_image=right_image,
        intrinsic_matrix=intrinsic_matrix,
        baseline=baseline,
        visualize=visualize,
        **estimate_kwargs,
    )
