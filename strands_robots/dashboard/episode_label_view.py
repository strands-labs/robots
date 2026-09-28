"""Shape an episode label sidecar into what the record screen renders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def label_view(
    document: Mapping[str, Any] | None,
    *,
    total_episodes: int | None = None,
    sidecar_error: str | None = None,
) -> dict[str, Any]:
    """What the dashboard can say about one dataset's episode labels."""
    episodes = (document or {}).get("episodes") or {}
    rows: list[dict[str, Any]] = []
    for key, rec in sorted(episodes.items(), key=lambda kv: _as_int(kv[0])):
        if not isinstance(rec, Mapping):
            continue
        det = rec.get("deterministic") if isinstance(rec.get("deterministic"), Mapping) else None
        judge = rec.get("judge") if isinstance(rec.get("judge"), Mapping) else None
        rows.append(
            {
                "episode_index": _as_int(rec.get("episode_index", key)),
                "verdict": None if det is None else ("success" if det.get("success") else "failure"),
                "steps": None if det is None else det.get("steps"),
                "quality": None if judge is None else judge.get("quality"),
                "failure_mode": None if judge is None else judge.get("failure_mode"),
                "note": None if judge is None else judge.get("note"),
                "disputes_verdict": bool(judge.get("disputes_verdict")) if judge else False,
                "model": None if judge is None else judge.get("model"),
                # The one field the UI needs to know whether a label CONTROL may be offered for this
                # row: annotate_episode refuses without a deterministic verdict, full stop.
                "annotatable": det is not None,
            }
        )

    labelled = sum(1 for r in rows if r["quality"])
    can, why = _capability(document, rows, sidecar_error)
    return {
        "benchmark": (document or {}).get("benchmark"),
        "schema_version": (document or {}).get("schema_version"),
        "episodes": rows,
        "total_episodes": total_episodes,
        "with_verdict": sum(1 for r in rows if r["annotatable"]),
        "labelled": labelled,
        "disputed": sum(1 for r in rows if r["disputes_verdict"]),
        "can_annotate": can,
        "why": why,
        "sidecar_error": sidecar_error,
    }


def _capability(
    document: Mapping[str, Any] | None,
    rows: list[dict[str, Any]],
    sidecar_error: str | None,
) -> tuple[bool, str]:
    """Whether ANY episode here can be annotated, and the sentence explaining it."""
    if sidecar_error:
        return False, (
            f"this dataset has an episode_labels.json that could not be read ({sidecar_error}), so existing labels "
            "cannot be shown and new ones would overwrite a file we do not understand"
        )
    if document is None:
        return False, (
            "no episode_labels.json in this dataset: labels start with the deterministic benchmark "
            "verdicts (record_deterministic_verdicts), and a judge annotates on top of one. A "
            "real-arm recording has no predicate verdict to annotate, so there is nothing to label "
            "yet -- this is a gap in the label rail, not a permission problem"
        )
    if not rows:
        return False, "the sidecar exists but records no episodes yet"
    if not any(r["annotatable"] for r in rows):
        return False, (
            "every episode here is missing its deterministic verdict, and annotate_episode refuses "
            "to layer a judgement on nothing -- record the benchmark verdicts first"
        )
    unjudged = [r["episode_index"] for r in rows if r["annotatable"] and not r["quality"]]
    if unjudged:
        return True, f"{len(unjudged)} episode(s) carry a verdict and are waiting for a quality grade"
    return True, "every episode with a verdict already carries a judge annotation"


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1
