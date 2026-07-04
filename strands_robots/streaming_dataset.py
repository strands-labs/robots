#!/usr/bin/env python3
"""Streaming read-back for LeRobotDataset - read frames directly from the Hub.

Primary use: in-process eval / replay / notebooks / agent loops (NOT a
precondition for streamed *training* - ``python -m lerobot.scripts.train
dataset.streaming=true`` already uses StreamingLeRobotDataset via
``lerobot.datasets.factory.make_dataset``).

Design mirrors :mod:`strands_robots.dataset_recorder`:
  * lerobot is NEVER imported at module top-level (numpy/pandas ABI safety on
    Jetson; see the :mod:`strands_robots.dataset_recorder` header).
  * Constructor kwargs are forwarded via ``inspect.signature`` introspection so
    a lerobot version bump can't break us (lerobot's dataset API drifted across
    0.5.0->0.5.2; streaming is newer and still changing - upstream has a
    multi-thread prefetch TODO).
"""

from __future__ import annotations

import inspect
import logging
import sys
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


# Only the POSITIVE result is cached. lerobot availability is a process
# capability that can transiently fail to resolve (a slow/locked import, or - in
# tests - a temporarily monkeypatched ``sys.modules``); caching a ``False``
# would permanently disable streaming for the rest of the process even after the
# condition clears. A failed probe is therefore re-attempted on the next call.
#
# The cache is a single-cell list rather than a module-level bool: a non-empty
# cell means "probed True". Mutating the cell (append/clear) records the result
# without rebinding a module global, so the memoization needs no ``global``
# statement.
_HAS_STREAMING_DATASET: list[bool] = []


def has_streaming_dataset() -> bool:
    """Return True if lerobot's ``StreamingLeRobotDataset`` is importable.

    A successful probe is cached; a failed probe is re-attempted on the next
    call so a transient import failure does not permanently disable streaming.
    """
    if _HAS_STREAMING_DATASET:
        return True
    try:
        from lerobot.datasets import StreamingLeRobotDataset  # noqa: F401
    except (ImportError, ValueError, RuntimeError) as exc:
        logger.debug("StreamingLeRobotDataset unavailable: %s", exc)
        return False
    _HAS_STREAMING_DATASET.append(True)
    return True


def _get_streaming_cls() -> Any:
    """Return StreamingLeRobotDataset, honoring a test-injected module override."""
    this_module = sys.modules[__name__]
    mock_cls = getattr(this_module, "StreamingLeRobotDataset", None)
    if mock_cls is not None:
        return mock_cls
    try:
        from lerobot.datasets import StreamingLeRobotDataset

        return StreamingLeRobotDataset
    except (ImportError, ValueError, RuntimeError) as exc:
        raise ImportError(
            f"StreamingLeRobotDataset unavailable ({exc}). "
            "Install with: pip install 'strands-robots[lerobot]' "
            "(needs torchcodec for video keys; on aarch64/Jetson that means "
            "torch>=2.11 + torchcodec>=0.11). "
            "For proprio-only streaming without torchcodec, use drop_videos=True."
        ) from exc


class StreamingDatasetReader:
    """Version-tolerant wrapper over lerobot's StreamingLeRobotDataset.

    Example (in-process eval / replay):
        reader = StreamingDatasetReader.open(
            "strands-robots/pick-place",
            delta_timestamps={"observation.images.front": [-0.2, -0.1, 0.0],
                              "action": [0.0, 0.1, 0.2]},
            shuffle=False,            # chronological for replay/eval
        )
        for frame in reader:
            ...  # raw tensors; normalize via reader.meta.stats if needed
    """

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    @classmethod
    def open(
        cls,
        repo_id: str,
        *,
        root: str | None = None,
        episodes: list[int] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        image_transforms: Callable | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        streaming: bool = True,
        buffer_size: int = 1000,
        max_num_shards: int = 16,
        seed: int = 42,
        shuffle: bool = True,
        return_uint8: bool = True,  # halves frame bandwidth; policies normalize
        validate_deltas: bool = True,  # parity with the materialized dataset path
        drop_videos: bool = False,  # proprio-only streaming (no torchcodec)
    ) -> StreamingDatasetReader:
        StreamingCls = _get_streaming_cls()
        init_sig = inspect.signature(StreamingCls).parameters
        # If the constructor accepts **kwargs, every candidate is forwardable.
        accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in init_sig.values())

        # Proprio-only: strip video keys from delta_timestamps so video decode
        # (torchcodec) is never invoked - lets constrained edge devices stream
        # state/action without a torchcodec wheel.
        if drop_videos and delta_timestamps:
            delta_timestamps = {
                k: v for k, v in delta_timestamps.items() if not k.startswith("observation.images.")
            } or None

        kwargs: dict[str, Any] = {"repo_id": repo_id}
        candidate = dict(
            root=root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            tolerance_s=tolerance_s,
            revision=revision,
            streaming=streaming,
            buffer_size=buffer_size,
            max_num_shards=max_num_shards,
            seed=seed,
            shuffle=shuffle,
            return_uint8=return_uint8,
        )
        for k, v in candidate.items():
            if not (accepts_var_kw or k in init_sig):
                continue
            if k in ("streaming", "shuffle", "return_uint8") or v is not None:
                kwargs[k] = v

        logger.info(
            "Opening StreamingLeRobotDataset: %s (streaming=%s, buffer=%d, shards=%d)",
            repo_id,
            streaming,
            buffer_size,
            max_num_shards,
        )
        ds = StreamingCls(**kwargs)

        # Tolerance grid-check - the streaming path skips check_delta_timestamps;
        # replicate it for parity with the materialized dataset.
        if delta_timestamps and validate_deltas:
            try:
                from lerobot.datasets.feature_utils import check_delta_timestamps

                check_delta_timestamps(delta_timestamps, ds.fps, tolerance_s, raise_value_error=True)
            except ImportError:
                logger.debug("check_delta_timestamps unavailable; skipping grid check")

        return cls(ds)

    def dataloader(self, batch_size: int = 64, num_workers: int = 0, **kw: Any) -> Any:
        """torch DataLoader over the streamed (Iterable) dataset.

        WARNING (verified upstream): StreamingLeRobotDataset is an
        IterableDataset that shuffles INTERNALLY (reservoir buffer). Do NOT pass
        shuffle=True here. With num_workers>0, video decode parallelizes across
        worker processes - the documented mitigation for the single-thread
        decode bottleneck. Never decode video in the main process while
        num_workers>0 (a known lerobot segfault).

        Note: lerobot's ``make_dataset`` (``lerobot.datasets.factory``) couples
        max_num_shards = num_workers. If you need that coupling, pass the same N
        to open(max_num_shards=N) and here num_workers=N.
        """
        import torch

        if kw.pop("shuffle", None):
            logger.warning("Ignoring shuffle=True: streaming shuffles internally.")
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers, **kw)

    @property
    def num_frames(self) -> Any:
        return self.dataset.num_frames

    @property
    def num_episodes(self) -> Any:
        return self.dataset.num_episodes

    @property
    def fps(self) -> Any:
        return self.dataset.fps

    @property
    def meta(self) -> Any:
        """Dataset metadata (incl. .stats for normalization). Always local."""
        return self.dataset.meta

    def __iter__(self) -> Any:
        return iter(self.dataset)
