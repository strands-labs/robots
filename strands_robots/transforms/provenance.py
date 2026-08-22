"""Provenance sidecar for generated episodes - the honesty record.

Every episode a :class:`~strands_robots.transforms.base.DatasetTransform`
writes is synthetic, and the output dataset says so on disk: a
``meta/provenance.json`` beside LeRobot's own metadata, one record per
generated episode, carrying ``synthetic=true``, the source episode it derives
from, and the transform's name and version. Training filters and evaluation
read it through :func:`load_provenance` / :func:`synthetic_episode_indices` so
generated pixels are treated honestly - silent mixing of generated and
recorded data is the failure mode this file exists to prevent.

A dataset with NO provenance file declares no synthetic episodes: that is the
ordinary state of a recorded dataset, so :func:`load_provenance` returns an
empty list for it rather than raising. A dataset the transform surface
produced always carries the file, including when every record in it was
written for a different camera count or episode subset than a reader expects -
absence means "recorded", never "unknown".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Location of the provenance sidecar, relative to a dataset root. Lives under
#: ``meta/`` beside LeRobot's own ``info.json`` so dataset tooling that copies
#: metadata copies the honesty record with it.
PROVENANCE_RELPATH = "meta/provenance.json"


def provenance_path(root: str | Path) -> Path:
    """Resolve the provenance sidecar path for a dataset root.

    Args:
        root: Dataset root directory (the dir that contains ``meta/``).

    Returns:
        The ``meta/provenance.json`` path under ``root`` (which may not exist;
        see :func:`load_provenance` for what absence means).
    """
    return Path(root) / PROVENANCE_RELPATH


def write_provenance(root: str | Path, records: list[dict[str, Any]]) -> Path:
    """Write the provenance records for a generated dataset.

    Args:
        root: Output dataset root (must already exist - the transform writes
            provenance after the dataset itself).
        records: One dict per generated episode, as assembled by
            :meth:`~strands_robots.transforms.base.DatasetTransform.transform`:
            ``episode_index``, ``synthetic``, ``source_episode_index``,
            ``source_repo_id``, ``transform``, ``transform_version``,
            ``variant``, ``prompt``, ``seed``.

    Returns:
        The path written.

    Raises:
        ValueError: ``records`` is not a list of dicts, or a record omits the
            mandatory keys (``episode_index``, ``synthetic``, ``transform``) -
            a provenance file that cannot answer "is episode N synthetic, and
            what generated it?" defeats its purpose, so it is refused rather
            than written incomplete.
    """
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError(f"write_provenance: records must be a list of dicts (got {type(records).__name__})")
    for i, record in enumerate(records):
        missing = [key for key in ("episode_index", "synthetic", "transform") if key not in record]
        if missing:
            raise ValueError(
                f"write_provenance: record {i} is missing mandatory provenance key(s) {missing} - "
                "a record that cannot say which episode is synthetic and what generated it is not provenance"
            )
    path = provenance_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "episodes": records}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_provenance(root: str | Path) -> list[dict[str, Any]]:
    """Read a dataset's per-episode provenance records.

    Args:
        root: Dataset root directory (the dir that contains ``meta/``).

    Returns:
        The list of per-episode records, empty when the dataset carries no
        provenance file - i.e. it declares no synthetic episodes, the ordinary
        state of a recorded dataset.

    Raises:
        ValueError: The file exists but cannot be parsed or does not hold the
            documented shape. A present-but-unreadable honesty record is
            corruption, not absence, so it is never read as "no synthetic
            episodes".
    """
    path = provenance_path(root)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"provenance file {path} is not valid JSON: {exc}") from exc
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, list) or any(not isinstance(r, dict) for r in episodes):
        raise ValueError(
            f'provenance file {path} does not hold the documented shape ({{"version": ..., "episodes": [record, ...]}})'
        )
    return episodes


def synthetic_episode_indices(root: str | Path) -> set[int]:
    """Episode indices a dataset declares as synthetic.

    The one-call filter for training / evaluation: everything in the returned
    set was generated by a transform; everything outside it was recorded.

    Args:
        root: Dataset root directory (the dir that contains ``meta/``).

    Returns:
        The set of ``episode_index`` values whose record carries a true
        ``synthetic`` field; empty for a dataset with no provenance file.
    """
    return {
        int(r["episode_index"]) for r in load_provenance(root) if r.get("synthetic") is True and "episode_index" in r
    }
