"""``num_envs`` copies of one compiled MuJoCo model on the GPU, stepped by MuJoCo-Warp (one CUDA graph per step)."""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any

import numpy as np

from ...._extras import hint

HINT = f"physics='newton' needs mujoco_warp + warp: {hint('newton')}"
SLOW_COMPILE_S = 5.0  # a warm kernel cache loads a scene in about a second; above this the kernels were compiled
_log = logging.getLogger(__name__)


class NewtonStepper:
    def __init__(self, device: str | None, num_envs: int) -> None:
        try:
            import mujoco_warp as mjw
            import warp as wp
        except ImportError as exc:
            raise ImportError(f"{HINT} ({exc})") from exc
        self.mjw, self.wp = mjw, wp
        wp.init()
        self.device = device or ("cuda:0" if wp.get_cuda_device_count() else "cpu")
        self.num_envs = num_envs
        self.m: Any = None
        self.d: Any = None
        self._model: Any = None
        self._graph: Any = None

    def load(self, model: Any, data: Any) -> None:
        """Put the model on the device and broadcast ``data`` to every env."""
        t = time.perf_counter()
        with self.wp.ScopedDevice(self.device):
            self._model = model
            self.m = self.mjw.put_model(model)
            self.d = self.mjw.put_data(model, data, nworld=self.num_envs)
            self._capture(data)
        if (took := time.perf_counter() - t) > SLOW_COMPILE_S:
            _log.warning(
                "newton: %.1f s to compile this scene shape's kernels; warp caches them in %s (WARP_CACHE_PATH), so "
                "the same shape loads in about a second next time. Batch scene edits with `with sim.editing():`.",
                took,
                self.wp.config.kernel_cache_dir,
            )

    def _capture(self, data: Any) -> None:
        """Record one step into a CUDA graph (kernel-by-kernel launches are ~10x slower); CPU devices run eagerly.

        The warm-up step that compiles the kernels runs on a scratch copy, and a capture records without executing,
        so loading, resetting or teleporting never advances the world.
        """
        self._graph = None
        if not str(self.device).startswith("cuda"):
            return
        try:
            self.mjw.step(self.m, self.mjw.put_data(self._model, data, nworld=self.num_envs))
            with self.wp.ScopedCapture() as cap:
                self.mjw.step(self.m, self.d)
            self._graph = cap.graph
        except Exception as exc:  # noqa: BLE001 — the graph is an optimisation; eager stepping is the contract
            msg = f"newton: CUDA graph capture failed ({exc}); stepping kernel by kernel, ~10x slower"
            warnings.warn(msg, stacklevel=2)
            self._graph = None

    def step(self, n: int) -> None:
        with self.wp.ScopedDevice(self.device):
            for _ in range(n):
                if self._graph is not None:
                    self.wp.capture_launch(self._graph)
                else:
                    self.mjw.step(self.m, self.d)

    def set_ctrl(self, ids: list[int], targets: np.ndarray) -> None:
        """The same targets in every env, written in place (the captured graph keeps the buffer)."""
        ctrl = self.d.ctrl.numpy()
        ctrl[:, ids] = targets
        self.d.ctrl.assign(ctrl)

    def sync_to(self, data: Any) -> None:
        """Env 0 into the CPU ``MjData`` (positions, velocities, time), so reads and renders see the GPU state."""
        import mujoco

        data.qpos[:] = self.d.qpos.numpy()[0]
        data.qvel[:] = self.d.qvel.numpy()[0]
        data.time = float(self.d.time.numpy()[0])
        mujoco.mj_forward(self._model, data)

    def sync_from(self, data: Any) -> None:
        """The CPU ``MjData`` into every env (after a reset, a teleport or a scene change)."""
        self.load(self._model, data)

    def close(self) -> None:
        self.m = self.d = self._graph = None
