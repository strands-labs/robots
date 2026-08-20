#!/usr/bin/env python3
"""Judge-agent tools for labeling recorded LeRobotDataset episodes.

The four tools the episode-judge agent drives (see issue-level design in
:mod:`strands_robots.episode_labels`, which owns the sidecar schema and the
verdict-precedence contract):

- :func:`load_episode` - episode/dataset metadata (frame count, features,
  whether a deterministic verdict and a judge label exist yet).
- :func:`sample_frames` - evenly spaced frames from one episode: state
  vectors always, decoded camera images optionally (for a multimodal judge).
- :func:`read_predicate_verdict` - the authoritative deterministic verdict.
- :func:`write_label` - the judge's annotation. Structurally unable to
  overturn the deterministic verdict: it writes only the ``judge`` block,
  and a disagreeing ``success_opinion`` is recorded as ``disputes_verdict``.

:func:`create_judge_agent` assembles them into a strands ``Agent`` whose
system prompt carries the two-stage doctrine. It is model-provider agnostic:
pass any strands model object (a Bedrock VLM, an OpenAI-compatible local
endpoint, ...) or none for the strands default.

Every tool returns the structured ``{"status", "content"}`` envelope and
never raises - a judge run over a hundred episodes must report the one
episode it could not read, not die on it.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from pathlib import Path
from typing import Any

from strands import Agent, tool

from strands_robots.tools._path_validation import validate_save_path
from strands_robots.utils import boolean_flag_error, non_negative_whole_number_error, positive_count_error

logger = logging.getLogger(__name__)

# The doctrine the agent operates under, stated where the model reads it.
# The quality-orthogonality sentence is load-bearing, not stylistic: measured
# on a graded five-recording ladder with exact ground truth, an unsteered
# VLM's grade tracks the OUTCOME - low for every failure, high
# only for the success - which re-derives the verdict the judge can never
# overturn and carries no information where the grade is consulted
# (filter_episodes already gates on the deterministic verdict).
JUDGE_SYSTEM_PROMPT = (
    "You are an episode-labeling judge for recorded robot datasets. The "
    "deterministic benchmark predicates have already scored each episode and "
    "their verdict is authoritative - you can never overturn it. Your job is "
    "to add what the predicates cannot measure: a quality grade (low / medium "
    "/ high), a failure-mode tag from the fixed taxonomy, and a short "
    "free-text note. The quality grade is about the EXECUTION visible in the "
    "recording - smoothness, directness, control - not about the outcome. The "
    "verdict already carries success and failure, so do not re-derive it in "
    "the grade: a failed episode executed cleanly up to the point of failure "
    "can be medium or high, and a successful episode with jerky or lucky "
    "motion can be low. Workflow per episode: call read_predicate_verdict, "
    "then load_episode and sample_frames to inspect the recording, then "
    "write_label exactly once. If your own read of success differs from the "
    "verdict, say so via success_opinion - it is recorded as a dispute "
    "annotation for human review, never applied to the verdict."
)


def _error(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": text}]}


def _resolve_root(root: str) -> Path:
    """Validate and resolve a caller-supplied dataset root path.

    LLM-provided strings are untrusted (AGENTS.md LLM-input-safety baseline):
    reject null bytes, ``..`` traversal and sensitive system directories
    before any filesystem read or write, then require the resolved directory
    to exist.
    """
    resolved = Path(validate_save_path(root, label="root"))
    if not resolved.is_dir():
        raise ValueError(f"Dataset root {resolved} is not an existing directory.")
    return resolved


def _read_info(root: Path) -> dict[str, Any]:
    """Read ``meta/info.json`` (empty dict when absent/unreadable)."""
    info_path = root / "meta" / "info.json"
    try:
        loaded = json.loads(info_path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError):
        return {}


def _episode_frame_rows(root: Path, episode: int) -> list[dict[str, Any]]:
    """Read one episode's frame rows from ``data/**/*.parquet`` (pure pyarrow).

    Returns rows sorted by ``frame_index``, each with ``frame_index``,
    ``timestamp`` and ``state`` (the flattened ``observation.state`` vector).
    Raises ValueError when the dataset has no data parquet or the episode has
    no frames.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - pyarrow ships with lerobot
        raise ValueError("sample_frames requires pyarrow (installed with the lerobot extra).") from e

    parquet_files = sorted((root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise ValueError(
            f"No data parquet under {root}. The dataset is empty or unfinalized "
            "(frames are flushed at save_episode / stop_recording)."
        )

    rows: list[dict[str, Any]] = []
    for pf in parquet_files:
        try:
            table = pq.read_table(pf)
        except (ValueError, OSError):
            # A corrupt shard loses its own frames only; the readable rest of
            # the dataset still answers for this episode (mirrors
            # read_dataset_episode_indices).
            continue
        columns = table.column_names
        if "episode_index" not in columns:
            continue
        data = table.to_pydict()
        for i, ep in enumerate(data["episode_index"]):
            if int(ep) != episode:
                continue
            state = data.get("observation.state", [None] * len(data["episode_index"]))[i]
            rows.append(
                {
                    "frame_index": int(data.get("frame_index", [i])[i]),
                    "timestamp": float(data["timestamp"][i]) if "timestamp" in data else None,
                    "state": [float(v) for v in state] if state is not None else None,
                }
            )
    if not rows:
        raise ValueError(f"Episode {episode} has no frames in {root}.")
    rows.sort(key=lambda r: r["frame_index"])
    return rows


def _decoded_image_blocks(root: Path, episode: int, positions: list[int]) -> list[dict[str, Any]]:
    """Decode camera frames at per-episode ``positions`` into image blocks.

    Uses ``LeRobotDataset`` for the decode (videos need lerobot's reader
    stack), so this path requires the ``lerobot`` extra; the state-only path
    above does not.
    """
    import io

    from strands_robots.utils import require_optionals

    require_optionals(
        ("lerobot", "PIL"),
        extra="lerobot",
        purpose="decoding recorded camera frames for a multimodal judge",
    )
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    # The dataset is addressed by its on-disk root; the repo_id is only a
    # display name once a root is given.
    ds = LeRobotDataset(repo_id=f"local/{root.name}", root=str(root))
    # Episode -> global frame range. LeRobot 0.6 records it per episode in
    # the episodes metadata (dataset_from_index); older builds expose the
    # same fact as episode_data_index tensors.
    if hasattr(ds, "episode_data_index"):
        start = int(ds.episode_data_index["from"][episode].item())
    else:
        start = int(ds.meta.episodes[episode]["dataset_from_index"])
    blocks: list[dict[str, Any]] = []
    for position in positions:
        frame = ds[start + position]
        for key in sorted(k for k in frame if k.startswith("observation.images.")):
            array = frame[key]
            if hasattr(array, "cpu"):
                array = array.cpu().numpy()
            array = np.asarray(array)
            if array.ndim == 3 and array.shape[0] in (1, 3):  # CHW float -> HWC
                array = np.moveaxis(array, 0, -1)
            if array.dtype != np.uint8:
                array = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(array.squeeze()).save(buffer, format="PNG")
            blocks.append({"image": {"format": "png", "source": {"bytes": buffer.getvalue()}}})
    return blocks


def _rms_state_jerk(rows: list[dict[str, Any]]) -> float | None:
    """RMS third difference of the state series - the statistic that carries
    smoothness.

    Jerk is the third derivative of position, so the max first difference
    (``max_state_delta``) cannot express it: a maximum over the episode is
    pinned by the gross traverse's own peak step, and a superimposed jitter
    never exceeds it - measured constant to 0.013% across a 4x range of true
    jerk on a real SO-101 recording. The third difference over the same rows
    tracks true rms jerk with correlation +1.0 on that same ladder, and
    recorded datasets carry positions only (no velocity channel), so a third
    difference of position is the available route.

    When the rows carry timestamps with a positive median spacing the value
    is scaled to state-units per second cubed; otherwise it is the raw
    per-step third difference. Returns ``None`` when fewer than four
    consecutive frames carry a state vector (jerk needs four points).
    """
    total = 0.0
    count = 0
    for s0, s1, s2, s3 in zip(
        (r["state"] for r in rows),
        (r["state"] for r in rows[1:]),
        (r["state"] for r in rows[2:]),
        (r["state"] for r in rows[3:]),
    ):
        if s0 is None or s1 is None or s2 is None or s3 is None:
            continue
        for a, b, c, d in zip(s0, s1, s2, s3):
            third = d - 3.0 * c + 3.0 * b - a
            total += third * third
            count += 1
    if count == 0:
        return None
    rms = math.sqrt(total / count)
    spacings = [
        current["timestamp"] - previous["timestamp"]
        for previous, current in zip(rows, rows[1:])
        if previous["timestamp"] is not None and current["timestamp"] is not None
    ]
    positive = [s for s in spacings if s > 0.0]
    if positive:
        rms /= statistics.median(positive) ** 3
    return rms


@tool
def load_episode(root: str, episode: int) -> dict[str, Any]:
    """Describe one recorded episode: length, features, and label state.

    The judge's first call on an episode - it reports how many frames there
    are (bounds for sample_frames), which cameras and state columns the
    dataset carries, and whether a deterministic verdict / judge label
    already exists in the sidecar.

    Args:
        root: Dataset root directory (the directory containing ``meta/``).
        episode: Episode index to describe, a non-negative whole number.

    Returns:
        Structured result whose JSON payload carries ``episode``, ``length``
        (frame count), ``total_episodes``, ``fps``, ``camera_keys``,
        ``state_names``, ``has_deterministic_verdict`` and ``has_judge_label``.
    """
    try:
        root_path = _resolve_root(root)
        if msg := non_negative_whole_number_error(episode, "episode", "load_episode"):
            return _error(msg)
        episode = int(episode)

        from strands_robots.dataset_recorder import read_dataset_episode_indices
        from strands_robots.episode_labels import labels_path, read_labels

        indices = read_dataset_episode_indices(root_path)
        if episode not in indices["episode_indices"]:
            return _error(f"Episode {episode} not in dataset {root_path} (episodes: {indices['episode_indices']}).")
        position = indices["episode_indices"].index(episode)
        length = indices["frames_per_episode"][position] if indices["frames_per_episode"] else 0

        info = _read_info(root_path)
        features = info.get("features", {})
        camera_keys = sorted(
            k.removeprefix("observation.images.") for k in features if k.startswith("observation.images.")
        )
        state_names = features.get("observation.state", {}).get("names") or []

        has_verdict = False
        has_label = False
        if labels_path(root_path).is_file():
            record = read_labels(root_path)["episodes"].get(str(episode), {})
            has_verdict = "deterministic" in record
            has_label = "judge" in record

        payload = {
            "episode": episode,
            "length": length,
            "total_episodes": indices["total_episodes"],
            "fps": info.get("fps"),
            "camera_keys": camera_keys,
            "state_names": state_names,
            "has_deterministic_verdict": has_verdict,
            "has_judge_label": has_label,
        }
        return {
            "status": "success",
            "content": [
                {"text": f"Episode {episode}: {length} frames, cameras={camera_keys or 'none'}."},
                {"json": payload},
            ],
        }
    except (ValueError, FileNotFoundError, OSError) as e:
        return _error(f"load_episode: {e}")


@tool
def sample_frames(root: str, episode: int, n_frames: int = 4, include_images: bool = False) -> dict[str, Any]:
    """Sample evenly spaced frames from one episode for the judge to inspect.

    Always returns the flattened ``observation.state`` vector and timestamp
    per sampled frame (pure pyarrow read, no lerobot import), plus a motion
    summary over the whole episode: ``rms_state_jerk`` (rms third difference
    of the state series) so a text-only judge can ground ``jerky_motion``
    from state alone, and ``max_state_delta`` (largest per-step state delta)
    for spotting discontinuities and teleports. Smoothness lives in the jerk
    field - a maximum first difference is a peak-velocity statistic pinned by
    the gross traverse, so it cannot see a superimposed jitter. With
    ``include_images`` a multimodal judge additionally receives the decoded
    camera frames as image content blocks.

    Args:
        root: Dataset root directory (the directory containing ``meta/``).
        episode: Episode index to sample, a non-negative whole number.
        n_frames: How many evenly spaced frames to sample, a positive count.
            Clamped to the episode length.
        include_images: When ``True``, decode the camera frames at the sampled
            positions into PNG image blocks (requires the ``lerobot`` extra;
            a dataset recorded without cameras reports an error). Must be a
            boolean - a posture flag is checked, never read by truthiness.

    Returns:
        Structured result whose JSON payload carries ``episode``, ``length``,
        ``samples`` (``frame_index`` / ``timestamp`` / ``state`` per sample),
        ``max_state_delta`` and ``rms_state_jerk`` (state units per second
        cubed when timestamps are present, per step cubed otherwise; ``null``
        when the episode is shorter than four frames). When images are
        requested, one image block follows per camera per sampled position -
        position-major, cameras in sorted key order within each position (the
        same order ``load_episode`` reports ``camera_keys``) - and the leading
        text block states the block count and that grouping, so a judge handed
        ``n_frames x n_cameras`` unlabelled images knows which are the same
        timestep from different viewpoints. Every camera is deliberately
        included rather than one canonical view: the same world motion can be
        legible in one view and below a judge's threshold in another (measured
        on a real two-camera recording, where a 185 mm slide read as 84 px of
        travel in one view and 22 px in the other), so sampling a single
        camera would drop verdicts.
    """
    try:
        root_path = _resolve_root(root)
        if msg := non_negative_whole_number_error(episode, "episode", "sample_frames"):
            return _error(msg)
        if msg := positive_count_error(n_frames, "n_frames", "sample_frames"):
            return _error(msg)
        if msg := boolean_flag_error(include_images, "include_images", "sample_frames"):
            return _error(msg)
        episode = int(episode)
        n_frames = int(n_frames)

        rows = _episode_frame_rows(root_path, episode)
        length = len(rows)
        count = min(n_frames, length)
        positions = [round(i * (length - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]

        max_delta = 0.0
        for previous, current in zip(rows, rows[1:]):
            if previous["state"] is None or current["state"] is None:
                continue
            for a, b in zip(previous["state"], current["state"]):
                max_delta = max(max_delta, abs(b - a))

        samples = [rows[p] for p in positions]
        content: list[dict[str, Any]] = [
            {"text": f"Episode {episode}: sampled {count} of {length} frames."},
            {
                "json": {
                    "episode": episode,
                    "length": length,
                    "samples": samples,
                    "max_state_delta": max_delta,
                    "rms_state_jerk": _rms_state_jerk(rows),
                }
            },
        ]
        if include_images:
            features = _read_info(root_path).get("features", {})
            camera_keys = sorted(
                k.removeprefix("observation.images.") for k in features if k.startswith("observation.images.")
            )
            if not camera_keys:
                return _error(
                    f"sample_frames: dataset {root_path} carries no observation.images.* features "
                    "to decode; record with cameras for a multimodal judge."
                )
            image_blocks = _decoded_image_blocks(root_path, episode, positions)
            # State the block count and grouping where the judge reads it: a
            # judge asked for n_frames and handed n_frames x n_cameras
            # unlabelled images has no other way to know that adjacent blocks
            # are the same timestep from different viewpoints.
            content[0]["text"] = (
                f"Episode {episode}: sampled {count} of {length} frames; "
                f"{len(image_blocks)} image blocks, position-major, "
                f"cameras sorted ({', '.join(camera_keys)})."
            )
            content.extend(image_blocks)
        return {"status": "success", "content": content}
    except (
        ValueError,
        FileNotFoundError,
        ImportError,
        OSError,
        KeyError,
        IndexError,
        AttributeError,
        RuntimeError,
    ) as e:
        # AttributeError / RuntimeError cover the optional decode stack
        # (lerobot dataset API drift, torch/torchcodec decode failures) - the
        # judge must report the episode it could not read, not die on it.
        return _error(f"sample_frames: {e}")


@tool
def read_predicate_verdict(root: str, episode: int) -> dict[str, Any]:
    """Read the authoritative deterministic predicate verdict for an episode.

    The benchmark predicates scored the episode from simulator state at
    rollout time; that verdict is stage one of the two-stage labeling and the
    judge's annotation layers on top of it. If none is recorded the judge
    must stop: there is nothing to annotate.

    Args:
        root: Dataset root directory (the directory containing ``meta/``).
        episode: Episode index, a non-negative whole number.

    Returns:
        Structured result whose JSON payload is the episode's
        ``deterministic`` block (``success`` / ``failure`` and, when present,
        ``steps`` / ``cumulative_reward`` / ``seed``).
    """
    try:
        root_path = _resolve_root(root)
        if msg := non_negative_whole_number_error(episode, "episode", "read_predicate_verdict"):
            return _error(msg)
        episode = int(episode)
        from strands_robots.episode_labels import deterministic_verdict

        verdict = deterministic_verdict(root_path, episode)
        return {
            "status": "success",
            "content": [
                {"text": f"Episode {episode} deterministic verdict: success={verdict['success']}."},
                {"json": verdict},
            ],
        }
    except (ValueError, FileNotFoundError, OSError) as e:
        return _error(f"read_predicate_verdict: {e}")


@tool
def write_label(
    root: str,
    episode: int,
    quality: str,
    failure_mode: str | None = None,
    note: str = "",
    success_opinion: bool | None = None,
    judge_model: str = "",
) -> dict[str, Any]:
    """Write the judge's annotation for an episode into the label sidecar.

    Annotation only, by construction: this writes the ``judge`` block and
    cannot reach the ``deterministic`` one, so no value passed here changes
    the benchmark verdict. A ``success_opinion`` that contradicts the verdict
    is recorded as ``disputes_verdict: true`` for human review - the verdict
    stands. An episode with no recorded deterministic verdict is refused.

    Args:
        root: Dataset root directory (the directory containing ``meta/``).
        episode: Episode index to label, a non-negative whole number.
        quality: Quality grade, one of ``low`` / ``medium`` / ``high``. Grades
            the execution visible in the recording (smoothness, directness,
            control), not the outcome - the deterministic verdict already
            carries success/failure, so a clean failure can be medium or high
            and a jerky or lucky success can be low.
        failure_mode: Optional tag from the fixed taxonomy (``jerky_motion``,
            ``near_miss``, ``camera_occlusion``, ``wrong_but_lucky``,
            ``drift``, ``collision``, ``incomplete``, ``other``). Legal on a
            successful episode too - ``near_miss`` and ``wrong_but_lucky``
            are exactly the annotations that make a success worth excluding
            from training data.
        note: Short free-text observation backing the grade and tag.
        success_opinion: The judge's own success read, or omit to offer none.
            Disagreement with the deterministic verdict is recorded as a
            dispute annotation, never applied.
        judge_model: Identifier of the labeling model (or ``"human"``),
            stored for provenance and calibration.

    Returns:
        Structured result whose JSON payload is the updated episode record
        (``episode_index`` / ``deterministic`` / ``judge``).
    """
    try:
        root_path = _resolve_root(root)
        if msg := non_negative_whole_number_error(episode, "episode", "write_label"):
            return _error(msg)
        episode = int(episode)
        from strands_robots.episode_labels import annotate_episode

        record = annotate_episode(
            root_path,
            episode,
            quality=quality,
            failure_mode=failure_mode,
            note=note,
            success_opinion=success_opinion,
            model=judge_model,
        )
        disputed = record["judge"]["disputes_verdict"]
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Episode {episode} labeled quality={quality}"
                        + (f", failure_mode={failure_mode}" if failure_mode else "")
                        + (
                            "; success_opinion disputes the deterministic verdict and was "
                            "recorded as an annotation (the verdict stands)."
                            if disputed
                            else "."
                        )
                    )
                },
                {"json": record},
            ],
        }
    except (ValueError, FileNotFoundError, OSError) as e:
        return _error(f"write_label: {e}")


def create_judge_agent(model: Any = None, system_prompt: str | None = None) -> Agent:
    """Assemble the episode-judge agent from the four labeling tools.

    Model-provider agnostic: ``model`` is any strands model object - a
    Bedrock-hosted VLM, an OpenAI-compatible local endpoint (vLLM/Ollama), or
    ``None`` for the strands default provider. No cloud dependency is
    required by the tools themselves; they only read the dataset and write
    the sidecar.

    Args:
        model: Optional strands model object forwarded to ``Agent(model=...)``.
        system_prompt: Override for :data:`JUDGE_SYSTEM_PROMPT`. The default
            carries the two-stage doctrine (deterministic verdict is
            authoritative; the judge annotates).

    Returns:
        A strands ``Agent`` wired with ``load_episode`` / ``sample_frames`` /
        ``read_predicate_verdict`` / ``write_label``.
    """
    return Agent(
        model=model,
        tools=[load_episode, sample_frames, read_predicate_verdict, write_label],
        system_prompt=JUDGE_SYSTEM_PROMPT if system_prompt is None else system_prompt,
    )
