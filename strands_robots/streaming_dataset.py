#!/usr/bin/env python3
"""Streaming read-back for LeRobotDataset - read frames directly from the Hub.

Primary use: in-process eval / replay / notebooks / agent loops (NOT a
precondition for streamed *training* - ``python -m lerobot.scripts.lerobot_train
--dataset.streaming=true`` already uses StreamingLeRobotDataset via
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

from strands_robots.utils import lerobot_version

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
            "For proprio-only streaming without torchcodec, use drop_videos=True "
            "with a delta_timestamps covering the non-video keys you need."
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
        repo_type: str = "dataset",  # "dataset" or "bucket"; non-default REQUIRES a lerobot that accepts it
    ) -> StreamingDatasetReader:
        """Open a version-tolerant streaming reader.

        Raises:
            RuntimeError: ``repo_type`` is not ``"dataset"`` but the installed
                ``StreamingLeRobotDataset`` does not accept a ``repo_type``
                parameter. No released lerobot does (the versioned-dataset vs
                bucket storage split is not upstream); the parameter is retained
                only for a lerobot build that adds it. Silently dropping the
                kwarg would stream from the versioned *dataset* namespace instead
                of the requested bucket - a different storage system, not a
                cosmetic difference - so this is never tolerant-dropped.
            ValueError: ``drop_videos=True`` but no non-video keys remain in
                ``delta_timestamps`` (or none were passed). Without a proprio
                ``delta_timestamps``, StreamingLeRobotDataset streams every
                feature - including camera keys via torchcodec decode - so the
                call would silently do the opposite of what was asked.
            ImportError: ``StreamingLeRobotDataset`` is not importable.
        """
        StreamingCls = _get_streaming_cls()
        init_sig = inspect.signature(StreamingCls).parameters
        # If the constructor accepts **kwargs, every candidate is forwardable.
        accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in init_sig.values())

        # repo_type selects WHICH storage system is read (versioned dataset
        # namespace vs bucket). No released lerobot accepts it; unlike the
        # cosmetic kwargs below, silently dropping a non-default value would open
        # the wrong storage system without error - fail fast instead.
        if repo_type != "dataset" and not (accepts_var_kw or "repo_type" in init_sig):
            raise RuntimeError(
                f"repo_type={repo_type!r} is not supported by any released lerobot "
                f"(installed: {lerobot_version()}): StreamingLeRobotDataset does not "
                "accept a repo_type parameter (the versioned-dataset vs bucket storage "
                "split is not upstream), and silently falling back to the 'dataset' "
                "namespace would stream from a different storage system. Pass "
                "repo_type='dataset' (the default) or use a lerobot build whose "
                "StreamingLeRobotDataset accepts repo_type."
            )

        # Proprio-only: strip video keys from delta_timestamps so video decode
        # (torchcodec) is never invoked - lets constrained edge devices stream
        # state/action without a torchcodec wheel.
        if drop_videos:
            if delta_timestamps:
                delta_timestamps = {
                    k: v for k, v in delta_timestamps.items() if not k.startswith("observation.images.")
                } or None
            if delta_timestamps is None:
                # Without delta_timestamps, StreamingLeRobotDataset streams
                # EVERY feature - camera keys included, invoking torchcodec -
                # so drop_videos=True would be a silent no-op doing the
                # opposite of what was asked. Refuse rather than no-op.
                raise ValueError(
                    "drop_videos=True requires delta_timestamps with at least "
                    "one non-video key: without it, every feature (including "
                    "camera keys, via torchcodec decode) is streamed and "
                    "drop_videos has no effect. Pass e.g. "
                    "delta_timestamps={'observation.state': [0.0], 'action': [0.0]}."
                )

        # return_uint8 absence does not change semantics (policies normalize
        # either way) but a lerobot that lacks it streams frames as float32 -
        # ~4x the bandwidth of uint8. That is a real cost, not a no-op, so warn
        # rather than drop it silently (unlike the truly cosmetic kwargs below).
        if return_uint8 and not (accepts_var_kw or "return_uint8" in init_sig):
            logger.warning(
                "return_uint8=True dropped: installed StreamingLeRobotDataset "
                "(lerobot %s) does not accept return_uint8, so frames stream as "
                "float32 (~4x the bandwidth of uint8). Policies still normalize "
                "correctly; upgrade lerobot for uint8 streaming.",
                lerobot_version(),
            )

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
            repo_type=repo_type,
        )
        for k, v in candidate.items():
            # Tolerant forwarding covers only kwargs whose absence does not
            # change semantics (a lerobot without the parameter behaves the
            # same for its default). repo_type is guarded above: a non-default
            # value on a lerobot that lacks the parameter raises instead of
            # being dropped here; the "dataset" default is safe to skip.
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
        """Total number of frames across all episodes in the streamed dataset."""
        return self.dataset.num_frames

    @property
    def num_episodes(self) -> Any:
        """Total number of episodes in the streamed dataset."""
        return self.dataset.num_episodes

    @property
    def fps(self) -> Any:
        """Capture frame rate (frames per second) of the streamed dataset."""
        return self.dataset.fps

    @property
    def meta(self) -> Any:
        """Dataset metadata (incl. .stats for normalization). Always local."""
        return self.dataset.meta

    def __iter__(self) -> Any:
        return iter(self.dataset)


def stream_dataset(repo_id: str, **kwargs: Any) -> StreamingDatasetReader:
    """Open a streaming reader for a LeRobotDataset without a simulator.

    Thin module-level alias for :meth:`StreamingDatasetReader.open`, exported
    at the package root (``strands_robots.stream_dataset``). Use this from
    training/eval scripts that only need to READ a dataset - it never touches
    MuJoCo, a GL context, or a thread pool, unlike constructing a ``Robot()``
    simulation. ``Simulation.stream_dataset`` is sugar delegating here so the
    "the same Robot() that records reads it back" flow still works.

    Args:
        repo_id: HF dataset id (e.g. ``"lerobot/svla_so100_pickplace"``) or a
            local repo_id paired with ``root=``.
        **kwargs: Forwarded to :meth:`StreamingDatasetReader.open` - e.g.
            ``root``, ``delta_timestamps``, ``episodes``, ``shuffle``,
            ``buffer_size``, ``max_num_shards``, ``drop_videos``
            (proprio-only, torchcodec-free), ``repo_type``.

    Returns:
        A :class:`StreamingDatasetReader`.

    Example:
        import strands_robots

        reader = strands_robots.stream_dataset(
            "lerobot/svla_so100_pickplace", shuffle=False
        )
        for frame in reader:
            ...
    """
    return StreamingDatasetReader.open(repo_id, **kwargs)
