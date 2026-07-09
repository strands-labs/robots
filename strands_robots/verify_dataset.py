"""``strands-robots verify-dataset`` - validate a recorded LeRobot dataset.

Detects the "mega-episode" corruption class: a collection run that intended to
record N distinct episodes but instead buffered every frame into a single
``episode_index=0`` episode (or whose ``meta/info.json`` count drifted from the
on-disk parquet). The parquet under ``meta/episodes/**/*.parquet`` is the ground
truth; this never trusts an agent's narration ("recorded 20/20") or in-memory
recorder bookkeeping.

Checks performed against a dataset root (the dir containing ``meta/``):
  1. parquet exists and holds at least one distinct episode;
  2. every episode has at least ``--min-frames`` frames (default 1) - flags any
     zero-length episode;
  3. ``meta/info.json`` ``total_episodes`` / ``total_frames`` (when present)
     agree with the parquet ground truth - flags metadata/parquet drift;
  4. when ``--expected N`` is given, the parquet holds exactly N episodes -
     flags the "wanted N, got M" mismatch.
  5. every per-episode video file referenced by the dataset (one MP4 per
     camera per episode, resolved from ``meta/info.json``'s ``video_path``
     template and the episode parquet's ``videos/<key>/chunk_index`` /
     ``file_index`` columns) exists on disk, is non-empty, and - when its
     frame count can be read from the container header - holds exactly the
     number of frames the parquet maps into it (the sum of the ``length`` of
     every episode packed into that file). Flags the video-modality sibling
     of mega-episode corruption: a correct episode count but missing pixels,
     whether the file is absent/empty or a truncated/partial encode with
     fewer frames than recorded (disable with ``--no-check-videos``).
  6. no ``action`` / ``observation.state`` column is identically zero across
     a multi-frame episode - the dead-control-column corruption (correct
     counts and pixels, but the proprioceptive/action signal was written as
     all zeros), read from the per-episode stats (disable with
     ``--no-check-stats``).

Usage:
    strands-robots verify-dataset /path/to/dataset
    strands-robots verify-dataset /path/to/dataset --expected 20
    python -m strands_robots verify-dataset ~/.cache/huggingface/lerobot/user/ds

Exit code is 0 when every check passes, 1 otherwise - so it drops straight into
CI as a dataset-integrity gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def verify_dataset(
    root: str | Path,
    expected: int | None = None,
    min_frames: int = 1,
    check_videos: bool = True,
    check_stats: bool = True,
) -> dict[str, Any]:
    """Verify the episode integrity of a LeRobot dataset on disk.

    Reads the canonical ``meta/episodes/**/*.parquet`` (via
    :func:`strands_robots.dataset_recorder.read_dataset_episode_indices`) and,
    when present, ``meta/info.json``, then runs the integrity checks described
    in the module docstring.

    Args:
        root: Dataset root directory (the dir that contains ``meta/``).
        expected: If given, require exactly this many distinct episodes.
        min_frames: Minimum frames every episode must contain (default 1).
        check_videos: When True (default), verify that every per-episode
            video file referenced by the dataset exists and is non-empty.
        check_stats: When True (default), flag any episode whose ``action``
            or ``observation.state`` column is identically zero across the
            whole episode (a dead control column).

    Returns:
        A report dict with:
          - ``status``: ``"success"`` if every check passed, else ``"error"``.
          - ``ok``: bool mirror of ``status``.
          - ``root``: the resolved dataset root as a string.
          - ``total_episodes`` / ``total_frames`` / ``episode_indices`` /
            ``frames_per_episode``: parquet ground truth (zeros/empties when the
            dataset is empty or unreadable).
          - ``expected``: the requested count (or ``None``).
          - ``info_total_episodes`` / ``info_total_frames``: values declared in
            ``meta/info.json`` (``None`` when the file is absent or lacks them).
          - ``video_files_checked``: number of distinct per-episode video
            files resolved and checked (``0`` when ``check_videos`` is False
            or the dataset declares no video features).
          - ``stats_vectors_checked``: number of per-episode control-feature
            stat vectors inspected for the dead-column check (``0`` when
            ``check_stats`` is False or the dataset carries no stats).
          - ``problems``: list of human-readable failure strings (empty on pass).
    """
    from strands_robots.dataset_recorder import read_dataset_episode_indices

    root_path = Path(root)
    report: dict[str, Any] = {
        "status": "error",
        "ok": False,
        "root": str(root_path),
        "total_episodes": 0,
        "total_frames": 0,
        "episode_indices": [],
        "frames_per_episode": [],
        "expected": expected,
        "info_total_episodes": None,
        "info_total_frames": None,
        "video_files_checked": 0,
        "stats_vectors_checked": 0,
        "problems": [],
    }
    problems: list[str] = report["problems"]

    if expected is not None and (not isinstance(expected, int) or expected < 0):
        problems.append(f"expected must be a non-negative int, got {expected!r}")
        return report

    # Parquet ground truth.
    try:
        info = read_dataset_episode_indices(root_path)
    except FileNotFoundError as e:
        problems.append(str(e))
        return report
    except ImportError as e:
        problems.append(str(e))
        return report

    report["total_episodes"] = info["total_episodes"]
    report["total_frames"] = info["total_frames"]
    report["episode_indices"] = info["episode_indices"]
    report["frames_per_episode"] = info["frames_per_episode"]

    # Check 1: non-empty.
    if info["total_episodes"] == 0:
        problems.append("no episodes found in parquet (dataset is empty)")

    # Check 2: every episode has >= min_frames frames. Only when per-episode
    # lengths are available (the length column is optional in some writers).
    if min_frames > 0 and info["frames_per_episode"]:
        short = [
            (ep, n)
            for ep, n in zip(info["episode_indices"], info["frames_per_episode"], strict=False)
            if n < min_frames
        ]
        if short:
            detail = ", ".join(f"episode {ep}={n} frame(s)" for ep, n in short)
            problems.append(f"{len(short)} episode(s) below min_frames={min_frames}: {detail}")

    # Check 3: meta/info.json vs parquet ground truth (drift detection).
    info_json_path = root_path / "meta" / "info.json"
    if info_json_path.is_file():
        try:
            declared = json.loads(info_json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            problems.append(f"could not read meta/info.json: {e}")
            declared = {}
        decl_eps = declared.get("total_episodes")
        decl_frames = declared.get("total_frames")
        report["info_total_episodes"] = decl_eps
        report["info_total_frames"] = decl_frames
        if isinstance(decl_eps, int) and decl_eps != info["total_episodes"]:
            problems.append(
                f"meta/info.json total_episodes={decl_eps} disagrees with parquet "
                f"({info['total_episodes']} distinct episode(s)) - metadata/parquet drift"
            )
        # Frame totals only meaningful when parquet carries per-episode lengths.
        if isinstance(decl_frames, int) and info["total_frames"] and decl_frames != info["total_frames"]:
            problems.append(
                f"meta/info.json total_frames={decl_frames} disagrees with parquet ({info['total_frames']} frame(s))"
            )

    # Check 4: exact expected episode count.
    if expected is not None and info["total_episodes"] != expected:
        problems.append(
            f"expected {expected} episode(s) but parquet holds {info['total_episodes']} "
            "- the recording did not produce the intended number of distinct episodes"
        )

    # Check 5: per-episode video files exist on disk, are non-empty, and hold
    # the frame count the parquet maps into them. A dataset can pass every count
    # check yet carry missing/empty MP4 streams, or a truncated/partial encode
    # with fewer frames than recorded (the video encoder failed mid-episode, the
    # files were partially synced, or frames were never captured) - correct
    # episode counts, missing pixels. This is the video-modality sibling of the
    # mega-episode class above.
    if check_videos:
        checked, video_problems = _verify_video_files(root_path)
        report["video_files_checked"] = checked
        problems.extend(video_problems)

    # Check 6: dead control columns. A dataset can pass every count/length
    # and video check yet carry an ``action`` (or ``observation.state``)
    # column that is identically zero across an entire multi-frame episode -
    # the proprioceptive sibling of the mega-episode and missing-video
    # classes. This happens when a writer's action keys never resolve to the
    # declared columns, so every frame's action is written as zeros while the
    # episode and frame counts stay correct. The per-episode feature stats
    # LeRobot v3 writes inline make this a cheap, decoder-independent check.
    if check_stats:
        stats_checked, stats_problems = _verify_feature_stats(root_path)
        report["stats_vectors_checked"] = stats_checked
        problems.extend(stats_problems)

    report["ok"] = not problems
    report["status"] = "success" if report["ok"] else "error"
    return report


def _video_frame_count(path: Path) -> int | None:
    """Best-effort decoded-frame count for a video file, read from the container
    header (no full decode).

    Returns the stream frame count when the container header carries it (the
    common case for the finalized MP4s LeRobot writes), or ``None`` when it
    cannot be read: PyAV (``av``) is not installed, the file is unreadable /
    corrupt, or the codec header omits ``nb_frames``. ``None`` means "cannot
    confirm", never "zero frames", so the caller only flags a genuine,
    confidently-read mismatch and never false-positives on a header that lacks a
    frame count.
    """
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return None
            n = container.streams.video[0].frames
    except (OSError, ValueError, RuntimeError):
        # Unreadable/corrupt header (incl. av.error.InvalidDataError, a
        # ValueError subclass): treat as "cannot confirm" and skip the compare.
        return None
    return int(n) if isinstance(n, int) and n > 0 else None


def _verify_video_files(root_path: Path) -> tuple[int, list[str]]:
    """Resolve and integrity-check every per-episode video file in a dataset.

    For each video feature declared in ``meta/info.json`` (``dtype == "video"``)
    and each episode, the on-disk MP4 path is resolved from the ``video_path``
    template and the episode parquet's ``videos/<key>/chunk_index`` /
    ``file_index`` columns, then checked for existence and non-zero size.
    Additionally, when a file's frame count can be read from the container
    header (via :func:`_video_frame_count`) and every episode packed into it
    carries a ``length``, the decoded frame count is compared to the sum of
    those lengths - flagging a truncated / partial encode (a present, non-empty
    file with fewer frames than recorded). This catches datasets whose episode
    counts are correct but whose pixels are missing, unwritten, or partial.

    Args:
        root_path: Dataset root directory (the dir that contains ``meta/``).

    Returns:
        A ``(checked, problems)`` tuple where ``checked`` is the number of
        distinct video files resolved and ``problems`` lists missing/empty
        files (empty when the dataset declares no video features or all files
        are present and non-empty).
    """
    info_path = root_path / "meta" / "info.json"
    if not info_path.is_file():
        return 0, []
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # An unreadable info.json is already surfaced by the info.json drift
        # check; do not double-report it here.
        return 0, []

    features = info.get("features")
    if not isinstance(features, dict):
        return 0, []
    video_keys = [key for key, feat in features.items() if isinstance(feat, dict) and feat.get("dtype") == "video"]
    if not video_keys:
        return 0, []

    template = info.get("video_path")
    if not isinstance(template, str) or not template:
        return 0, [
            f"meta/info.json declares {len(video_keys)} video feature(s) but no 'video_path' "
            "template - cannot locate the per-episode MP4 files"
        ]

    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - pyarrow ships with the lerobot extra
        return 0, []

    parquet_files = sorted((root_path / "meta" / "episodes").glob("**/*.parquet"))
    # (video_key, chunk_index, file_index) -> first referencing episode_index.
    referenced: dict[tuple[str, int, int], int] = {}
    # Same key -> total frames the parquet maps into that file (LeRobot v3 packs
    # multiple whole episodes into one shared file per camera, so the file must
    # hold exactly the sum of those episodes' ``length`` values).
    expected_frames: dict[tuple[str, int, int], int] = {}
    # Files a frame-count comparison must skip because at least one referencing
    # episode carried no ``length`` (the column is optional in some writers).
    unknown_len_files: set[tuple[str, int, int]] = set()
    keys_with_refs: set[str] = set()
    for pf in parquet_files:
        table = pq.read_table(pf)
        cols = set(table.column_names)
        data = table.to_pydict()
        episodes = data.get("episode_index", [])
        lengths = data.get("length")
        for vk in video_keys:
            ci_col = f"videos/{vk}/chunk_index"
            fi_col = f"videos/{vk}/file_index"
            if ci_col not in cols or fi_col not in cols:
                continue
            keys_with_refs.add(vk)
            chunk_idx = data[ci_col]
            file_idx = data[fi_col]
            for i in range(len(chunk_idx)):
                ci, fi = chunk_idx[i], file_idx[i]
                if ci is None or fi is None:
                    continue
                key = (vk, int(ci), int(fi))
                ep = int(episodes[i]) if i < len(episodes) and episodes[i] is not None else -1
                referenced.setdefault(key, ep)
                length = lengths[i] if lengths is not None and i < len(lengths) else None
                if length is None:
                    unknown_len_files.add(key)
                else:
                    expected_frames[key] = expected_frames.get(key, 0) + int(length)

    problems: list[str] = []
    for vk in video_keys:
        if vk not in keys_with_refs:
            problems.append(
                f"video feature '{vk}' is declared but no episode references a video file for it "
                "(missing videos/<key>/chunk_index|file_index columns)"
            )

    for (vk, ci, fi), ep in sorted(referenced.items()):
        rel = template.format(video_key=vk, chunk_index=ci, file_index=fi)
        path = root_path / rel
        if not path.is_file():
            problems.append(f"missing video file for '{vk}' (episode {ep}): {rel}")
        elif path.stat().st_size == 0:
            problems.append(f"empty video file for '{vk}' (episode {ep}): {rel}")

    # Frame-count integrity: a present, non-empty video whose decoded frame
    # count is fewer (or otherwise different) than the frames the parquet maps
    # into it is a truncated / partial encode - correct episode counts, a
    # non-empty file, but missing pixels (the encoder crashed mid-episode, the
    # file was partially synced, or the write was interrupted). This is the
    # frame-count sibling of the missing/empty-video class above. It is reported
    # only when the count can be read confidently from the container header, so
    # a codec whose header omits the frame count never yields a false positive.
    for key in sorted(expected_frames):
        if key in unknown_len_files:
            continue  # a referencing episode lacked a length: cannot compute expected
        vk, ci, fi = key
        rel = template.format(video_key=vk, chunk_index=ci, file_index=fi)
        path = root_path / rel
        if not path.is_file() or path.stat().st_size == 0:
            continue  # already reported as missing / empty above
        actual = _video_frame_count(path)
        if actual is not None and actual != expected_frames[key]:
            problems.append(
                f"video file for '{vk}' has {actual} frame(s) but the parquet maps "
                f"{expected_frames[key]} frame(s) to it: {rel} - truncated / partial "
                "encode (correct episode counts, non-empty file, but missing pixels)"
            )

    return len(referenced), problems


def _verify_feature_stats(root_path: Path) -> tuple[int, list[str]]:
    """Flag dead (identically-zero) control columns via per-episode stats.

    LeRobot v3 writes per-episode feature statistics inline in
    ``meta/episodes/**/*.parquet`` (``stats/<feature>/min`` / ``.../max`` /
    ``.../count`` ...). A recording whose ``action`` (or ``observation.state``)
    column is identically zero across an entire multi-frame episode passes every
    count, length and video check yet carries no usable control signal -- the
    proprioceptive sibling of the mega-episode and missing-video corruption
    classes. This happens when a writer's action keys never resolve to the
    declared columns (so every frame's action is written as zeros) while the
    episode and frame counts stay correct.

    For each control feature (``action`` and any ``observation.state*``) that
    carries per-episode ``min``/``max`` stats, flag every episode with at least
    two frames whose feature vector is identically zero (every component
    ``min == max == 0``). Single-frame episodes are skipped: a lone frame has
    ``min == max`` trivially, so all-zero there is not yet evidence of a dead
    column. Reading only the lightweight stats columns keeps this
    decoder-independent (no video decode, no ``data/`` scan).

    Args:
        root_path: Dataset root directory (the dir that contains ``meta/``).

    Returns:
        A ``(checked, problems)`` tuple where ``checked`` is the number of
        per-episode control-feature stat vectors inspected and ``problems``
        lists the dead-column episodes (empty when the dataset carries no stats
        or every control column varies).
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - pyarrow ships with the lerobot extra
        return 0, []

    parquet_files = sorted((root_path / "meta" / "episodes").glob("**/*.parquet"))
    if not parquet_files:
        return 0, []

    def _is_control_feature(name: str) -> bool:
        return name == "action" or name == "observation.state" or name.startswith("observation.state.")

    def _episode_count(value: Any) -> int | None:
        # The stats ``count`` column is stored as a length-1 list per episode.
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return int(value[0]) if value else None
        return int(value)

    checked = 0
    problems: list[str] = []
    for pf in parquet_files:
        table = pq.read_table(pf)
        cols = set(table.column_names)
        if "episode_index" not in cols:
            continue
        data = table.to_pydict()
        episodes = data["episode_index"]
        features = sorted(
            {
                c[len("stats/") : -len("/min")]
                for c in cols
                if c.startswith("stats/")
                and c.endswith("/min")
                and _is_control_feature(c[len("stats/") : -len("/min")])
            }
        )
        for feat in features:
            min_col = f"stats/{feat}/min"
            max_col = f"stats/{feat}/max"
            if max_col not in cols:
                continue
            mins = data[min_col]
            maxs = data[max_col]
            counts = data.get(f"stats/{feat}/count")
            for i in range(len(mins)):
                row_min = mins[i]
                row_max = maxs[i]
                if row_min is None or row_max is None:
                    continue
                checked += 1
                n = _episode_count(counts[i]) if counts is not None and i < len(counts) else None
                if n is not None and n < 2:
                    continue  # single frame: identically-zero is not yet corruption evidence
                ep = int(episodes[i]) if i < len(episodes) and episodes[i] is not None else -1
                if all(v == 0 for v in row_min) and all(v == 0 for v in row_max):
                    frames = f"{n} frame(s)" if n is not None else "all frames"
                    problems.append(
                        f"feature '{feat}' is identically zero across episode {ep} ({frames}) - "
                        "dead control column (the recorded values never change and are all zero)"
                    )
    return checked, problems


def _format_report(report: dict[str, Any]) -> str:
    """Render a report dict as a human-readable multi-line summary."""
    lines: list[str] = []
    verdict = "PASS" if report["ok"] else "FAIL"
    lines.append(f"[{verdict}] {report['root']}")
    lines.append(f"  episodes (parquet): {report['total_episodes']}")
    if report.get("video_files_checked"):
        lines.append(f"  video files checked: {report['video_files_checked']}")
    if report.get("stats_vectors_checked"):
        lines.append(f"  stat vectors checked: {report['stats_vectors_checked']}")
    if report["total_frames"]:
        lines.append(f"  frames   (parquet): {report['total_frames']}")
    if report["expected"] is not None:
        lines.append(f"  expected episodes : {report['expected']}")
    if report["info_total_episodes"] is not None:
        lines.append(f"  info.json episodes: {report['info_total_episodes']}")
    for problem in report["problems"]:
        lines.append(f"  - {problem}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on any failed check."""
    parser = argparse.ArgumentParser(
        prog="strands-robots verify-dataset",
        description="Validate the episode integrity of a recorded LeRobot dataset.",
    )
    parser.add_argument("root", help="Dataset root directory (the dir that contains meta/).")
    parser.add_argument(
        "-e",
        "--expected",
        type=int,
        default=None,
        help="Require exactly this many distinct episodes (the count you intended to record).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=1,
        help="Minimum frames every episode must contain (default: 1).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of the human-readable summary.",
    )
    parser.add_argument(
        "--no-check-videos",
        dest="check_videos",
        action="store_false",
        help="Skip per-episode video-file existence/non-empty checks.",
    )
    parser.add_argument(
        "--no-check-stats",
        dest="check_stats",
        action="store_false",
        help="Skip the dead-control-column (all-zero action/state) check.",
    )
    args = parser.parse_args(argv)

    report = verify_dataset(
        args.root,
        expected=args.expected,
        min_frames=args.min_frames,
        check_videos=args.check_videos,
        check_stats=args.check_stats,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_format_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
