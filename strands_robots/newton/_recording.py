"""Video recording — run policy and capture frames."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ._core import NewtonBackend


def record_video(
    backend: NewtonBackend,
    robot_name: str = "",
    policy_provider: str = "mock",
    instruction: str = "",
    duration: float = 1.0,
    fps: int = 30,
    width: int = 1024,
    height: int = 768,
    output_path: Optional[str] = None,
    cosmos_transfer: bool = False,
    cosmos_prompt: Optional[str] = None,
    cosmos_control: Optional[str] = None,
) -> Dict[str, Any]:
    """Run policy and record video."""
    if not backend._world_created:
        return {"status": "error", "content": [{"text": "World not created"}]}

    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), "newton_video.mp4")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total_frames = int(duration * fps)
    dt = backend._config.physics_dt
    frames = []

    t0 = time.time()
    for _ in range(total_frames):
        substeps = max(1, int(1.0 / (fps * dt)))
        for _ in range(substeps):
            backend.step()

        r = backend.render(width=width, height=height)
        img = r.get("image")
        if img is not None:
            frames.append(img)

    elapsed = time.time() - t0

    # Save video
    try:
        import imageio

        writer = imageio.get_writer(output_path, fps=fps, quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()
    except ImportError:
        # Save as npz fallback
        output_path = output_path.replace(".mp4", ".npz")
        np.savez_compressed(output_path, frames=np.array(frames))

    # Get file size
    size_kb = 0
    if os.path.exists(output_path):
        size_kb = os.path.getsize(output_path) // 1024

    # Optional Cosmos transfer
    cosmos_out = None
    if cosmos_transfer and output_path.endswith(".mp4"):
        try:
            from strands_robots.cosmos_transfer import (
                CosmosTransferConfig,
                CosmosTransferPipeline,
            )

            cfg = CosmosTransferConfig()
            pipe = CosmosTransferPipeline(cfg)
            cosmos_out = pipe.transfer_video(
                input_path=output_path,
                prompt=cosmos_prompt or instruction,
                control=cosmos_control,
            )
        except Exception as exc:
            logger.warning("Cosmos transfer failed: %s", exc)

    text = (
        f"Recorded {len(frames)} frames in {elapsed:.1f}s → {output_path} ({size_kb}KB)\n"
        f"Solver: {backend._config.solver}, Envs: {backend._config.num_envs}"
    )
    if cosmos_out:
        text += f"\nCosmos transfer: {cosmos_out}"

    return {"status": "success", "content": [{"text": text}]}
