"""Stereo depth estimation tool for Strands Agents.

Provides a ``@tool``-decorated function that allows agents to estimate depth
from stereo image pairs using Fast-FoundationStereo.  The tool wraps
:class:`~strands_robots.stereo.StereoDepthPipeline` with a simple,
agent-friendly interface.

Typical usage with Strands Agents:

    from strands import Agent
    from strands_robots.tools.stereo_depth import stereo_depth

    agent = Agent(tools=[stereo_depth])
    agent("Estimate depth from the stereo pair left.png and right.png "
          "with 120mm baseline and focal length 600px")

The tool supports:
    - File path inputs (PNG, JPG, etc.)
    - Automatic intrinsic matrix construction from focal length + principal point
    - Disparity, metric depth, and point cloud output
    - Optional disparity visualisation saved to disk
    - Camera presets for common stereo cameras
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from strands import tool

logger = logging.getLogger(__name__)


@tool
def stereo_depth(
    left_image: str,
    right_image: str,
    baseline: Optional[float] = None,
    focal_length: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    camera: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_variant: str = "23-36-37",
    valid_iters: int = 8,
    scale: float = 1.0,
    visualize: bool = True,
) -> Dict[str, Any]:
    """Estimate depth from a rectified stereo image pair.

    Uses NVIDIA's Fast-FoundationStereo to compute dense disparity and
    metric depth from left/right stereo images.  Returns depth statistics
    and optionally saves disparity visualisation and depth map to disk.

    Args:
        left_image: Path to the left (reference) stereo image.
        right_image: Path to the right stereo image.
        baseline: Distance between stereo cameras in metres.
        focal_length: Camera focal length in pixels.  Used with ``cx``/``cy``
            to construct the intrinsic matrix.
        cx: Principal point X coordinate (pixels).  Defaults to image width / 2.
        cy: Principal point Y coordinate (pixels).  Defaults to image height / 2.
        camera: Camera preset name (e.g. ``"realsense_d435"``, ``"zed2"``).
            Overrides ``baseline`` with the camera's default baseline.
        output_dir: Directory to save output files (disparity vis, depth map).
            When *None*, uses the directory of ``left_image``.
        model_variant: Model checkpoint variant.  One of ``"23-36-37"``
            (best accuracy), ``"20-26-39"`` (balanced), ``"20-30-48"``
            (fastest).
        valid_iters: Number of GRU refinement iterations (4=fast, 8=accurate).
        scale: Image scale factor for inference.  Use 0.5 for faster processing.
        visualize: If True, save colourised disparity image.

    Returns:
        Dictionary with depth estimation results including:
            - ``median_depth``: Median valid depth in metres.
            - ``valid_ratio``: Fraction of pixels with valid depth.
            - ``inference_time``: Model inference time in seconds.
            - ``disparity_vis_path``: Path to saved visualisation (if any).
            - ``depth_npy_path``: Path to saved depth array (if any).
    """
    import numpy as np

    from strands_robots.stereo import StereoConfig, StereoDepthPipeline

    # Validate inputs
    if not os.path.isfile(left_image):
        return {"error": f"Left image not found: {left_image}"}
    if not os.path.isfile(right_image):
        return {"error": f"Right image not found: {right_image}"}

    # Build configuration
    config = StereoConfig(
        model_variant=model_variant,
        valid_iters=valid_iters,
        scale=scale,
        camera=camera,
    )

    # Build intrinsic matrix if focal length provided
    intrinsic_matrix = None
    if focal_length is not None:
        # Load image to get dimensions for default principal point
        img = StereoDepthPipeline._load_image(left_image)
        h, w = img.shape[:2]
        px = cx if cx is not None else w / 2.0
        py = cy if cy is not None else h / 2.0
        intrinsic_matrix = np.array(
            [[focal_length, 0, px], [0, focal_length, py], [0, 0, 1]],
            dtype=np.float64,
        )

    # Run pipeline
    pipeline = StereoDepthPipeline(config)
    result = pipeline.estimate_depth(
        left_image=left_image,
        right_image=right_image,
        intrinsic_matrix=intrinsic_matrix,
        baseline=baseline,
        visualize=visualize,
    )

    # Prepare output
    out_dir = output_dir or os.path.dirname(left_image) or "."
    os.makedirs(out_dir, exist_ok=True)

    output: Dict[str, Any] = {
        "status": "success",
        "height": result.height,
        "width": result.width,
        "median_depth": result.median_depth,
        "valid_ratio": round(result.valid_ratio, 4),
        "inference_time": round(result.metadata.get("inference_time", 0), 4),
        "total_time": round(result.metadata.get("total_time", 0), 4),
        "model_variant": model_variant,
        "valid_iters": valid_iters,
        "scale": scale,
    }

    # Save outputs
    if result.depth is not None:
        depth_path = os.path.join(out_dir, "depth_meter.npy")
        np.save(depth_path, result.depth)
        output["depth_npy_path"] = depth_path

    if result.disparity_vis is not None and visualize:
        try:
            import imageio.v2 as imageio  # type: ignore[import-untyped]

            vis_path = os.path.join(out_dir, "disparity_vis.png")
            imageio.imwrite(vis_path, result.disparity_vis)
            output["disparity_vis_path"] = vis_path
        except ImportError:
            try:
                import cv2  # type: ignore[import-untyped]

                vis_path = os.path.join(out_dir, "disparity_vis.png")
                cv2.imwrite(vis_path, result.disparity_vis[..., ::-1])
                output["disparity_vis_path"] = vis_path
            except ImportError:
                output["disparity_vis_path"] = None
                output["warning"] = "Could not save visualisation (no imageio/cv2)"

    if result.point_cloud is not None:
        pc_path = os.path.join(out_dir, "point_cloud.npy")
        np.save(pc_path, result.point_cloud)
        output["point_cloud_npy_path"] = pc_path

    return output
