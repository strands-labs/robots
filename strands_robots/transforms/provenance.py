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

from strands_robots.utils import boolean_flag_error, non_negative_whole_number_error

#: Location of the provenance sidecar, relative to a dataset root. Lives under
#: ``meta/`` beside LeRobot's own ``info.json`` so dataset tooling that copies
#: metadata copies the honesty record with it.
PROVENANCE_RELPATH = "meta/provenance.json"


#: The provenance keys a reader turns into a verdict, and the shared domain each
#: is held to. ``episode_index`` is the index the reader returns and
#: ``synthetic`` is the boolean it classifies on, so both are validated rather
#: than merely required to be present - the same split
#: :func:`~strands_robots.episode_labels.record_deterministic_verdicts` makes for
#: the other per-episode sidecar, which validates the index and the verdict
#: booleans it reads and carries its descriptive keys through untouched.
#: ``transform`` stays a presence check for that reason: it labels a record, and
#: no reader branches on it.
_VERDICT_KEY_DOMAINS = {
    "episode_index": non_negative_whole_number_error,
    "synthetic": boolean_flag_error,
}

#: Keys every record must carry, whatever their type.
_MANDATORY_KEYS = ("episode_index", "synthetic", "transform")


def _record_problem(position: int, record: dict[str, Any], context: str) -> str | None:
    """Error text when a provenance record cannot answer what it is read for.

    The one owner of "is this record readable as provenance?", so
    :func:`write_provenance` and :func:`load_provenance` cannot disagree about
    which records are: a record the writer refuses is a record the reader
    refuses, and vice versa.

    Args:
        position: Index of the record within the list, for the message.
        record: The record to grade.
        context: Surface name for the message (the caller's own function).

    Returns:
        The error text, or ``None`` when the record is readable.
    """
    missing = [key for key in _MANDATORY_KEYS if key not in record]
    if missing:
        return (
            f"{context}: record {position} is missing mandatory provenance key(s) {missing} - "
            "a record that cannot say which episode is synthetic and what generated it is not provenance"
        )
    for key, domain in _VERDICT_KEY_DOMAINS.items():
        if msg := domain(record[key], f"record {position} {key!r}", context):
            return msg
    return None


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
        ValueError: ``records`` is not a list of dicts, or a record cannot
            answer what it is read for - it omits a mandatory key
            (``episode_index``, ``synthetic``, ``transform``), or its
            ``episode_index`` is not a non-negative whole number, or its
            ``synthetic`` is not a boolean. A provenance file that cannot answer
            "is episode N synthetic, and what generated it?" defeats its
            purpose, so it is refused rather than written unreadable. A
            non-boolean ``synthetic`` in particular is refused rather than
            coerced: every non-empty string is truthy and every non-zero number
            is, so guessing which of them meant "generated" is how a generated
            episode gets counted as recorded.
    """
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError(f"write_provenance: records must be a list of dicts (got {type(records).__name__})")
    written: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        if msg := _record_problem(i, record, "write_provenance"):
            raise ValueError(msg)
        # Store the verdict keys in the type the schema documents, so the file a
        # reader loads holds the same answer the writer graded. The guard
        # round-tripped both, and the caller's dict is left untouched.
        written.append(
            {**record, "episode_index": int(record["episode_index"]), "synthetic": bool(record["synthetic"])}
        )
    path = provenance_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "episodes": written}
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
        ValueError: The file exists but cannot be parsed, does not hold the
            documented shape, or holds a record that cannot answer what it is
            read for (see :func:`write_provenance` for the per-record rule -
            both ends apply the same one, so a record this refuses is a record
            the writer would have refused). A present-but-unreadable honesty
            record is corruption, not absence, so it is never read as "no
            synthetic episodes" - which is what a record whose ``synthetic``
            field is not a boolean would otherwise be read as.
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
    for i, record in enumerate(episodes):
        if msg := _record_problem(i, record, f"provenance file {path}"):
            raise ValueError(msg)
    return episodes


def synthetic_episode_indices(root: str | Path) -> set[int]:
    """Episode indices a dataset declares as synthetic.

    The one-call filter for training / evaluation: everything in the returned
    set was generated by a transform; everything outside it was recorded.

    Args:
        root: Dataset root directory (the dir that contains ``meta/``).

    Returns:
        The set of ``episode_index`` values whose record carries a true
        ``synthetic`` field; empty for a dataset with no provenance file. A
        record declaring ``synthetic=false`` is a readable answer and is simply
        outside the set; a record whose ``synthetic`` is not a boolean cannot
        answer at all and is refused by :func:`load_provenance` rather than
        landing outside it.

    Raises:
        ValueError: Propagated from :func:`load_provenance` when the file is
            present but unreadable. Never swallowed: an empty set means "this
            dataset declares no synthetic episodes", so returning it for a file
            that could not be read would be the silent mixing of generated and
            recorded data this sidecar exists to prevent.
    """
    # ``load_provenance`` round-tripped both keys through the shared domains.
    return {int(r["episode_index"]) for r in load_provenance(root) if bool(r["synthetic"])}
