"""Is a listed dataset a dataset, or the folder an aborted recording left behind?"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

__all__ = [
    "OUTSIDE_DATASET_HOME",
    "MIN_EPISODES",
    "dataset_id_error",
    "dataset_id_is_a_path",
    "dataset_verdict",
    "record_target_verdict",
]

#: One episode is a real dataset (a single demonstration you can replay or overfit on). Zero is
#: not "small", it is "nothing was ever recorded".
MIN_EPISODES = 1

#: One body for every refused path: the same whether the target exists or not.
OUTSIDE_DATASET_HOME = {
    "error": "path_outside_dataset_home",
    "detail": "paths this dashboard reads or writes must live under the dataset home ($HF_LEROBOT_HOME)",
}

#: A recording target is an ``owner/name`` id and nothing else. ``dataset_source.local_dataset_dir``
#: reads an absolute, ``./``-prefixed or slash-less id as a literal directory, so those shapes would
#: turn a dataset name into a write path; each segment starts with an alphanumeric, which also rules
#: out ``.`` and ``..``.
_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def dataset_id_is_a_path(dataset: str) -> bool:
    """Whether ``dataset_source.local_dataset_dir`` would read this id as a directory, not a Hub id.

    Absolute, ``./``-prefixed and slash-less ids are literal directories there; a ``..`` segment
    walks out of the home even in an ``owner/name`` shape. Each of those is a write path in
    disguise and gets the one fixed refusal every contained path gets.
    """
    name = (dataset or "").strip()
    if not name:
        return False
    parts = name.replace("\\", "/").split("/")
    return name.startswith(("/", "./", "~")) or len(parts) < 2 or any(seg in ("", ".", "..") for seg in parts)


def dataset_id_error(dataset: str) -> str | None:
    """Why *dataset* cannot be the repo_id of a new recording, or None when it can."""
    name = (dataset or "").strip()
    if not name:
        return None  # record_target_verdict owns the empty-name sentence
    if _DATASET_ID.fullmatch(name) is None:
        return (
            "dataset must be an 'owner/name' id (letters, digits, '.', '_', '-'; each part starting with "
            "a letter or digit) - a path, an absolute or './' name, or a bare name is not a dataset id"
        )
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def dataset_verdict(
    meta: Mapping[str, Any] | None,
    *,
    has_data_files: bool | None = None,
) -> dict[str, Any]:
    """What can be said about a listed dataset without loading it."""
    m = dict(meta or {})
    episodes = _as_int(m.get("total_episodes"))
    frames = _as_int(m.get("total_frames"))

    if not m or (episodes is None and frames is None and m.get("fps") is None):
        return {
            "usable": False,
            "reason": "unreadable_meta",
            "problem": (
                "meta/info.json could not be read, so nothing is known about this directory's "
                "contents. If a recording is running right now, wait for it to finish; otherwise "
                "the file is truncated and the recording did not survive."
            ),
        }

    if episodes is not None and episodes < MIN_EPISODES:
        return {
            "usable": False,
            "reason": "no_episodes",
            "episodes": episodes,
            "problem": (
                "0 episodes. meta/info.json is written when a recording session OPENS, before the "
                "first episode is captured, so a directory like this is what an abandoned session "
                "leaves behind - not a dataset. Record into it, or delete it."
            ),
        }

    if has_data_files is False:
        return {
            "usable": False,
            "reason": "missing_data",
            "episodes": episodes,
            "problem": (
                f"meta/info.json claims {episodes if episodes is not None else 'some'} episodes but "
                "there are no data files under data/. The metadata was written and the frames never "
                "landed (a crash mid-recording), or the data directory was moved or deleted while "
                "meta/ stayed behind. Training would fail after the setup it charges you for."
            ),
        }

    verdict: dict[str, Any] = {
        "usable": True,
        "episodes": episodes,
        "note": "read from meta/info.json only - nothing here opens a shard",
    }
    if frames == 0 and episodes:
        # Episodes recorded, no frames counted: legal-but-odd metadata rather than a refusal,
        # because the frame count is a summary and the episodes are the substance.
        verdict["warning"] = (
            f"{episodes} episode(s) but 0 frames counted in the metadata - the summary may be "
            "stale; check the episode list before a long training run."
        )
    if has_data_files is None:
        verdict["note"] += "; data/ was not checked"
    return verdict


# : A row that a recorder is writing into right now.
RECORDING_REASON = "recording_in_progress"


def _same_dataset(row: Mapping[str, Any], active: str) -> bool:
    """Is this listing row the dataset the recorder named?"""
    if not active:
        return False
    repo = str(row.get("repo_id") or "")
    if repo and repo == active:
        return True
    root = str(row.get("root") or "")
    if root:
        return root == active or root.endswith("/" + active.strip("/"))
    return False


def mark_live_recording(
    rows: list[dict[str, Any]],
    active_dataset: str | None,
    *,
    episodes_so_far: int | None = None,
) -> list[dict[str, Any]]:
    if not active_dataset:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        if not _same_dataset(row, active_dataset):
            out.append(row)
            continue
        n = episodes_so_far if isinstance(episodes_so_far, int) else None
        so_far = f"{n} episode(s) captured so far" if n is not None else "episodes are being written"
        out.append(
            {
                **row,
                "usable": False,
                "recording": True,
                "reason": RECORDING_REASON,
                "problem": (
                    f"a recording session is writing into this dataset right now - {so_far}. "
                    "Training would read a dataset that is still growing, and a replay would race the "
                    "writer. Wait for the session to close; do NOT delete the folder."
                ),
            }
        )
    return out


def record_target_verdict(
    dataset: str,
    *,
    exists: bool = False,
    has_meta: bool = False,
    episodes: int | None = None,
    non_empty: bool = False,
) -> str | None:
    """Can a NEW recording session write into this dataset name?"""
    name = (dataset or "").strip()
    if not name:
        return (
            "a dataset name is required - it becomes the dataset's repo_id "
            "(e.g. 'local/pick-place' or 'cagataydev/so101-cubes')."
        )
    if exists and has_meta:
        n = episodes if isinstance(episodes, int) else None
        if n:
            return (
                f"'{name}' already exists with {n} recorded episode(s). Recording would refuse "
                "rather than append (LeRobotDataset.create will not open an existing dataset), and "
                "overwriting it would destroy those episodes. Pick another name - or delete that "
                "dataset deliberately first."
            )
        return (
            f"'{name}' already exists on disk with no recorded episodes - what an interrupted "
            "session leaves behind. Recording still refuses to reuse the directory, so pick "
            "another name, or remove that empty dataset first."
        )
    if exists and non_empty:
        return (
            f"'{name}' resolves to a directory that already has files in it but is not a dataset "
            "(no meta/). Recording into it would fail; pick another name, or clear that directory "
            "yourself - nothing here will delete files for you."
        )
    return None
