#!/usr/bin/env python3
"""Episode-level labels for recorded datasets: deterministic verdicts + judge annotations.

Two-stage verdict, same doctrine as the safety and dispatch layers: the
deterministic benchmark predicates run first and are AUTHORITATIVE for what
they can measure (success / failure per the benchmark definition, from
simulator state). A judge - typically a VLM agent reading the recorded
episode - adds the labels the predicates cannot: a quality grade, a
failure-mode tag from a fixed taxonomy, a free-text note. The judge can never
overturn a deterministic verdict; it annotates on top of one. That precedence
is structural here, not advisory: :func:`annotate_episode` writes only the
``judge`` block, refuses an episode that has no deterministic verdict yet, and
records a disagreeing opinion as ``disputes_verdict`` - an annotation - while
the ``deterministic`` block stays byte-identical.

Labels are a JSON sidecar (``episode_labels.json``) at the dataset root, next
to LeRobot's own ``meta/`` / ``data/`` / ``videos/``, so downstream training
can filter episodes (e.g. train on success + high-quality only) without
rewriting the dataset. The sidecar travels with the dataset directory and dies
with it on ``overwrite=True``, which is the correct lifetime for labels about
its episodes.

Sidecar schema (``schema_version`` 1)::

    {
      "schema_version": 1,
      "benchmark": "<benchmark name the verdicts came from>",
      "episodes": {
        "<episode_index>": {
          "episode_index": 0,
          "deterministic": {          # authoritative; written only by
            "success": true,          # record_deterministic_verdicts from
            "failure": false,         # evaluate_benchmark's per-episode
            "steps": 150,             # results - never by the judge.
            "cumulative_reward": 3.2,
            "seed": 0
          },
          "judge": {                  # annotation; written only by
            "quality": "high",        # annotate_episode. One of
                                      # QUALITY_GRADES.
            "failure_mode": null,     # None or one of FAILURE_MODES. Legal
                                      # on a success too (near_miss,
                                      # wrong_but_lucky are success
                                      # annotations by design).
            "note": "...",            # free text from the judge.
            "success_opinion": true,  # the judge's own read, or None.
            "disputes_verdict": false,# True when success_opinion differs
                                      # from deterministic success. The
                                      # disagreement is recorded here, never
                                      # applied to the verdict.
            "model": "mock-judge",    # who labeled (model id / "human").
            "labeled_at": 1755590400.0  # epoch stamp (time.time()); an
                                        # absolute record another process
                                        # correlates, not a duration.
          }
        }
      }
    }

JSON object keys are strings, so ``episodes`` is keyed by ``str(episode_index)``;
every public function here takes and returns plain ``int`` indices.

Writes are two-phase (temp file + ``os.replace`` in the same directory), the
house pattern from :mod:`strands_robots.tools.harness_memory`, so a crashed
writer cannot leave a half-written sidecar. A ``schema_version`` this module
does not know is refused on read rather than misread.

See ``docs/data/episode-labels.md`` for the full schema documentation and
``examples/17_judge_recorded_episodes.py`` for the end-to-end pipeline
(record -> deterministic verdicts -> judge -> filter -> re-train).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from strands_robots.utils import boolean_flag_error, non_negative_whole_number_error

# Bump when the sidecar layout changes shape. Read paths refuse a version
# they do not know rather than guessing at its meaning.
LABEL_SCHEMA_VERSION = 1

# File name of the sidecar at the dataset root, next to meta/ and data/.
SIDECAR_FILENAME = "episode_labels.json"

# The quality vocabulary, ordered worst to best so rank comparisons are
# index comparisons. A grade outside this tuple is refused at write time -
# a free-text grade cannot be filtered on. The grade is about the EXECUTION
# visible in the recording (smoothness, directness, control), deliberately
# orthogonal to the deterministic outcome: a grade that re-derives
# success/failure carries no information where it is consulted, because
# filter_episodes already gates on the verdict. An unsteered judge collapses
# onto the outcome (measured on a graded ladder with exact ground truth),
# which is why JUDGE_SYSTEM_PROMPT states this contract where the model
# reads it.
QUALITY_GRADES = ("low", "medium", "high")

# Fixed failure-mode taxonomy - the qualities a person tags in recorded
# episodes that no simulator-state predicate can measure. Fixed so downstream
# filters match on identity rather than parsing prose; the free-text goes in
# ``note``. "near_miss" and "wrong_but_lucky" are deliberately legal on
# deterministically SUCCESSFUL episodes: they are exactly the annotations
# that make a success worth excluding from training data.
FAILURE_MODES = (
    "jerky_motion",
    "near_miss",
    "camera_occlusion",
    "wrong_but_lucky",
    "drift",
    "collision",
    "incomplete",
    "other",
)


def labels_path(root: str | Path) -> Path:
    """Path of the label sidecar for the dataset at ``root``.

    Args:
        root: Dataset root directory (the directory containing ``meta/``).

    Returns:
        ``<root>/episode_labels.json``. The file need not exist yet.
    """
    return Path(root) / SIDECAR_FILENAME


#: The label keys a reader turns into a verdict, and the domain each is held to.
#: Every one is a field a reader BRANCHES on: ``deterministic.success`` is the
#: authoritative verdict :func:`filter_episodes` gates ``require_success`` on,
#: ``judge.quality`` is the grade its quality bar ranks and
#: :func:`measure_agreement` calibrates against, ``judge.failure_mode`` is the
#: tag that measurement compares, and ``judge.disputes_verdict`` is what
#: ``exclude_disputed`` drops on. The descriptive keys (``steps``,
#: ``cumulative_reward``, ``seed``, ``note``, ``model``, ``labeled_at``,
#: ``success_opinion``) are carried through untouched: no reader branches on
#: them, so a surprising value there is a record to read, not a verdict to
#: refuse. That is the same split :func:`record_deterministic_verdicts` and
#: :func:`annotate_episode` already make on the way IN, and the one
#: :mod:`strands_robots.transforms.provenance` - the other per-episode sidecar,
#: which cites this module for it - makes at both ends.
_DETERMINISTIC_VERDICT_KEYS = {"success": boolean_flag_error, "failure": boolean_flag_error}

#: Keys a block must carry to state the thing it exists to state. A
#: ``deterministic`` block with no ``success`` cannot say what the predicates
#: measured; a ``judge`` block with no ``quality`` cannot clear a quality bar.
#: Both are read by subscript downstream, so absence is a ``KeyError`` from
#: inside a filter rather than an answer.
_REQUIRED_DETERMINISTIC_KEYS = ("success",)
_REQUIRED_JUDGE_KEYS = ("quality",)


def _record_problem(key: str, record: Any, context: str) -> str | None:
    """Error text when a sidecar record cannot answer what it is read for.

    The one owner of "is this record readable as a label?", so every reader
    reaches the same verdict about the same bytes. :func:`read_labels` calls
    it on every record, which is what makes a sidecar this module did not
    write hold to the domains its own writers enforce on the way in - and
    :func:`filter_episodes` / :func:`measure_agreement` / :func:`annotate_episode`
    inherit the check by reading through it rather than each re-deriving one.

    An unreadable record is refused rather than skipped: silently dropping it
    would make a filter's answer depend on how much of the sidecar it could
    parse, which is the same reason :func:`read_labels` refuses an unknown
    ``schema_version`` instead of projecting it onto this version's meanings.

    Args:
        key: The ``episodes`` mapping key, for the message.
        record: The record to grade.
        context: Surface name for the message (the sidecar path).

    Returns:
        The error text, or ``None`` when the record is readable.
    """
    # JSON object keys are strings, so the shared whole-number domain grades the
    # index this key SPELLS, not the key itself. The spelling has to be the
    # canonical one the writers produce (``str(index)``): measure_agreement looks
    # an episode up as ``str(int(index))``, so a non-canonical ``"00"`` is
    # invisible to it while filter_episodes would return ``int("00") == 0`` - one
    # sidecar, two answers about which episode a record is about.
    spelled = None
    if isinstance(key, str):
        try:
            spelled = int(key)
        except ValueError:
            spelled = None
    if spelled is None or str(spelled) != key:
        return (
            f"{context}: episodes key {key!r} is not the decimal spelling of a whole number. "
            "This mapping is keyed by str(episode_index), so a key a reader cannot turn back "
            "into that index selects no episode."
        )
    if msg := non_negative_whole_number_error(spelled, f"episodes key {key!r}", context):
        return msg
    if not isinstance(record, dict):
        return f"{context}: episodes[{key!r}] is {type(record).__name__}, expected a JSON object."

    if "deterministic" in record:
        block = record["deterministic"]
        if not isinstance(block, dict):
            return f"{context}: episodes[{key!r}]['deterministic'] is {type(block).__name__}, expected a JSON object."
        for required in _REQUIRED_DETERMINISTIC_KEYS:
            if required not in block:
                return (
                    f"{context}: episodes[{key!r}]['deterministic'] has no {required!r} - "
                    "a verdict block that cannot state the outcome is not a verdict."
                )
        for field, domain in _DETERMINISTIC_VERDICT_KEYS.items():
            if field in block and (
                msg := domain(block[field], f"episodes[{key!r}]['deterministic'][{field!r}]", context)
            ):
                return msg

    if "judge" in record:
        judge = record["judge"]
        if not isinstance(judge, dict):
            return f"{context}: episodes[{key!r}]['judge'] is {type(judge).__name__}, expected a JSON object."
        for required in _REQUIRED_JUDGE_KEYS:
            if required not in judge:
                return (
                    f"{context}: episodes[{key!r}]['judge'] has no {required!r} - "
                    f"an annotation with no grade cannot clear a quality bar. One of {QUALITY_GRADES}."
                )
        if judge["quality"] not in QUALITY_GRADES:
            return (
                f"{context}: episodes[{key!r}]['judge']['quality'] is {judge['quality']!r}, "
                f"which is not one of {QUALITY_GRADES}. A grade outside the vocabulary cannot be "
                "ranked against a quality bar."
            )
        mode = judge.get("failure_mode")
        if mode is not None and mode not in FAILURE_MODES:
            return (
                f"{context}: episodes[{key!r}]['judge']['failure_mode'] is {mode!r}, which is not "
                f"None or one of {FAILURE_MODES}."
            )
        if "disputes_verdict" in judge and (
            msg := boolean_flag_error(
                judge["disputes_verdict"], f"episodes[{key!r}]['judge']['disputes_verdict']", context
            )
        ):
            return msg
    return None


def _empty_document(benchmark: str) -> dict[str, Any]:
    return {"schema_version": LABEL_SCHEMA_VERSION, "benchmark": benchmark, "episodes": {}}


def _write_document(path: Path, document: dict[str, Any]) -> None:
    """Two-phase write: serialize to a temp file in the same dir, then rename.

    ``os.replace`` is atomic on POSIX and Windows for same-filesystem paths,
    so a reader never observes a truncated sidecar and a crashed writer
    leaves the previous version intact.
    """
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        # Cleanup-and-reraise: never leave the temp file behind on failure.
        # The unlink is best-effort - the exception worth the caller's
        # attention is the write failure being re-raised, not a cleanup
        # OSError on a file that may already be gone (same construct as
        # atomic_write_bytes in strands_robots.simulation.safe_output).
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def read_labels(root: str | Path) -> dict[str, Any]:
    """Read the label sidecar for the dataset at ``root``.

    Args:
        root: Dataset root directory.

    Returns:
        The full sidecar document (see the module docstring for the schema).

    Raises:
        FileNotFoundError: If no sidecar exists yet - record deterministic
            verdicts first (:func:`record_deterministic_verdicts`).
        ValueError: If the sidecar is unparseable, carries a
            ``schema_version`` this module does not know, or holds a record
            this module cannot read as a label - a verdict-bearing field
            outside its domain, or a block that cannot state the thing it
            exists to state. A future schema is
            refused rather than misread: silently projecting it onto this
            version's field meanings would hand downstream filters wrong
            answers with nothing raised, and a record whose ``success`` is the
            string ``"false"`` or whose ``quality`` is outside
            :data:`QUALITY_GRADES` does exactly that one level down. This is
            the boundary for a sidecar this module did not write, so a value
            its own writers refuse on the way in is refused on the way out.
    """
    path = labels_path(root)
    if not path.is_file():
        raise FileNotFoundError(
            f"No label sidecar at {path}. Record deterministic verdicts first "
            "(record_deterministic_verdicts) - the judge annotates on top of them."
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Label sidecar {path} is not valid JSON: {e}") from e
    if not isinstance(document, dict):
        raise ValueError(f"Label sidecar {path} must hold a JSON object, got {type(document).__name__}.")
    version = document.get("schema_version")
    if version != LABEL_SCHEMA_VERSION:
        raise ValueError(
            f"Label sidecar {path} has schema_version {version!r}; this build reads "
            f"version {LABEL_SCHEMA_VERSION}. Refusing to reinterpret an unknown schema."
        )
    episodes = document.get("episodes")
    if not isinstance(episodes, dict):
        raise ValueError(
            f"Label sidecar {path} has an 'episodes' of {type(episodes).__name__}, expected a JSON "
            "object keyed by episode index."
        )
    for key, record in episodes.items():
        if msg := _record_problem(key, record, f"Label sidecar {path}"):
            raise ValueError(msg)
    return document


def record_deterministic_verdicts(
    root: str | Path,
    episodes: list[dict[str, Any]],
    *,
    benchmark: str = "",
) -> dict[str, Any]:
    """Record the authoritative per-episode predicate verdicts for a dataset.

    This is stage one of the two-stage verdict: the input is the per-episode
    ``episodes`` list from ``evaluate_benchmark``'s JSON payload (each entry
    carrying ``episode``, ``success``, ``failure``, ``steps``,
    ``cumulative_reward``, ``seed``), produced by the benchmark's own
    predicates against simulator state. Only this function writes the
    ``deterministic`` block; the judge surface (:func:`annotate_episode`)
    cannot reach it.

    Re-recording an episode's verdict replaces the ``deterministic`` block
    (a re-evaluation supersedes the old measurement) and PRESERVES any
    existing ``judge`` block - annotations are about the recorded frames,
    which did not change.

    Args:
        root: Dataset root directory. Must exist - a verdict for a dataset
            that is not on disk labels nothing.
        episodes: Per-episode verdict dicts. Each must carry an integer
            ``episode`` index and a boolean ``success``; ``failure`` defaults
            to ``False``, ``steps`` / ``cumulative_reward`` / ``seed`` are
            carried through when present.
        benchmark: Name of the benchmark whose predicates produced the
            verdicts, stored on the document for provenance.

    Returns:
        The updated sidecar document.

    Raises:
        FileNotFoundError: If ``root`` is not an existing directory.
        ValueError: If ``episodes`` is empty or an entry is malformed (the
            message names the entry), or the existing sidecar is unreadable.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset root {root_path} is not an existing directory; nothing to label.")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(
            "record_deterministic_verdicts: episodes must be a non-empty list of per-episode "
            "verdict dicts (the 'episodes' list from evaluate_benchmark's JSON payload)."
        )

    path = labels_path(root_path)
    document = read_labels(root_path) if path.is_file() else _empty_document(benchmark)
    if benchmark:
        document["benchmark"] = benchmark

    for position, entry in enumerate(episodes):
        if not isinstance(entry, dict):
            raise ValueError(
                f"record_deterministic_verdicts: episodes[{position}] is {type(entry).__name__}, expected a dict."
            )
        index = entry.get("episode")
        if msg := non_negative_whole_number_error(index, f"episodes[{position}]['episode']", "verdict recording"):
            raise ValueError(msg)
        index = int(index)  # type: ignore[arg-type]  # guard round-tripped it
        success = entry.get("success")
        if msg := boolean_flag_error(success, f"episodes[{position}]['success']", "verdict recording"):
            raise ValueError(msg)
        failure = entry.get("failure", False)
        if msg := boolean_flag_error(failure, f"episodes[{position}]['failure']", "verdict recording"):
            raise ValueError(msg)

        deterministic: dict[str, Any] = {"success": bool(success), "failure": bool(failure)}
        for carried in ("steps", "cumulative_reward", "seed"):
            if carried in entry:
                deterministic[carried] = entry[carried]

        record = document["episodes"].setdefault(str(index), {"episode_index": index})
        record["deterministic"] = deterministic

    _write_document(path, document)
    return document


def deterministic_verdict(root: str | Path, episode: int) -> dict[str, Any]:
    """Read the authoritative predicate verdict for one episode.

    Args:
        root: Dataset root directory.
        episode: Episode index, a non-negative whole number.

    Returns:
        The episode's ``deterministic`` block.

    Raises:
        FileNotFoundError: If no sidecar exists yet.
        ValueError: If the index is unusable, or no deterministic verdict is
            recorded for that episode.
    """
    if msg := non_negative_whole_number_error(episode, "episode", "deterministic_verdict"):
        raise ValueError(msg)
    episode = int(episode)
    document = read_labels(root)
    record = document["episodes"].get(str(episode))
    if not record or "deterministic" not in record:
        raise ValueError(
            f"No deterministic verdict recorded for episode {episode}. "
            "Run evaluate_benchmark and record_deterministic_verdicts first."
        )
    return dict(record["deterministic"])


def annotate_episode(
    root: str | Path,
    episode: int,
    *,
    quality: str,
    failure_mode: str | None = None,
    note: str = "",
    success_opinion: bool | None = None,
    model: str = "",
) -> dict[str, Any]:
    """Write a judge annotation on top of an episode's deterministic verdict.

    Stage two of the two-stage verdict. This is the ONLY writer of the
    ``judge`` block and it never touches the ``deterministic`` one: a judge
    whose ``success_opinion`` contradicts the predicate verdict gets
    ``disputes_verdict: true`` recorded - an annotation a human can review -
    while the verdict stands. An episode with no deterministic verdict is
    refused outright, because an annotation layered on nothing would be a
    verdict in disguise.

    Args:
        root: Dataset root directory.
        episode: Episode index, a non-negative whole number.
        quality: One of :data:`QUALITY_GRADES`. Refused otherwise - a
            free-text grade cannot be filtered on. Grades the execution
            visible in the recording, not the outcome (see the
            :data:`QUALITY_GRADES` comment).
        failure_mode: ``None`` or one of :data:`FAILURE_MODES`. Deliberately
            legal on a deterministically successful episode (``near_miss``,
            ``wrong_but_lucky``).
        note: Free-text observation from the judge. Must be a string.
        success_opinion: The judge's own success read, or ``None`` to offer
            none. Checked on the shared boolean-flag domain when present -
            an opinion is a posture, not a quantity.
        model: Identifier of who labeled (model id, endpoint, or ``"human"``).

    Returns:
        The updated episode record (``episode_index`` / ``deterministic`` /
        ``judge``).

    Raises:
        FileNotFoundError: If no sidecar exists yet.
        ValueError: If any field is outside its domain, or the episode has no
            deterministic verdict recorded.
    """
    if msg := non_negative_whole_number_error(episode, "episode", "annotate_episode"):
        raise ValueError(msg)
    episode = int(episode)
    if quality not in QUALITY_GRADES:
        raise ValueError(f"annotate_episode: quality must be one of {QUALITY_GRADES}, got {quality!r}.")
    if failure_mode is not None and failure_mode not in FAILURE_MODES:
        raise ValueError(
            f"annotate_episode: failure_mode must be None or one of {FAILURE_MODES}, got {failure_mode!r}."
        )
    if not isinstance(note, str):
        raise ValueError(f"annotate_episode: note must be a string, got {type(note).__name__}.")
    if success_opinion is not None and (
        msg := boolean_flag_error(success_opinion, "success_opinion", "annotate_episode")
    ):
        raise ValueError(msg)
    if not isinstance(model, str):
        raise ValueError(f"annotate_episode: model must be a string, got {type(model).__name__}.")

    document = read_labels(root)
    record = document["episodes"].get(str(episode))
    if not record or "deterministic" not in record:
        raise ValueError(
            f"No deterministic verdict recorded for episode {episode}; refusing to annotate. "
            "The judge annotates ON TOP of the predicate verdict "
            "(record_deterministic_verdicts first) - it never supplies one."
        )

    verdict_success = bool(record["deterministic"]["success"])
    disputes = success_opinion is not None and bool(success_opinion) != verdict_success
    record["judge"] = {
        "quality": quality,
        "failure_mode": failure_mode,
        "note": note,
        "success_opinion": None if success_opinion is None else bool(success_opinion),
        "disputes_verdict": disputes,
        "model": model,
        # An absolute stamp another process correlates, so time.time() is the
        # right clock (see the clocks rule in AGENTS.md); nothing measures a
        # duration from it.
        "labeled_at": time.time(),
    }
    _write_document(labels_path(root), document)
    return dict(record)


def filter_episodes(
    root: str | Path,
    *,
    require_success: bool = True,
    min_quality: str = "medium",
    exclude_disputed: bool = False,
) -> list[int]:
    """Select episode indices for training from the label sidecar.

    Only labeled episodes can clear a quality bar, so episodes without a
    ``judge`` block are excluded: an unlabeled episode has no quality to
    compare, and admitting it would make the filter's answer depend on how
    much of the dataset the judge got through.

    Args:
        root: Dataset root directory.
        require_success: When ``True``, keep only episodes whose
            DETERMINISTIC verdict is success - the authoritative field, never
            the judge's opinion. Checked on the shared boolean-flag domain.
        min_quality: Lowest acceptable judge quality grade, one of
            :data:`QUALITY_GRADES`.
        exclude_disputed: When ``True``, also drop episodes whose judge
            disputed the verdict - the conservative posture when the
            judge/human agreement measurement is not in yet. Checked on the
            shared boolean-flag domain.

    Returns:
        Sorted episode indices satisfying every enabled criterion.

    Raises:
        FileNotFoundError: If no sidecar exists yet.
        ValueError: If a flag or the grade is outside its domain.
    """
    for flag, value in (("require_success", require_success), ("exclude_disputed", exclude_disputed)):
        if msg := boolean_flag_error(value, flag, "filter_episodes"):
            raise ValueError(msg)
    if min_quality not in QUALITY_GRADES:
        raise ValueError(f"filter_episodes: min_quality must be one of {QUALITY_GRADES}, got {min_quality!r}.")

    minimum_rank = QUALITY_GRADES.index(min_quality)
    document = read_labels(root)
    selected: list[int] = []
    for key, record in document["episodes"].items():
        judge = record.get("judge")
        if not judge:
            continue
        if require_success and not record.get("deterministic", {}).get("success", False):
            continue
        if QUALITY_GRADES.index(judge["quality"]) < minimum_rank:
            continue
        if exclude_disputed and judge.get("disputes_verdict", False):
            continue
        selected.append(int(key))
    return sorted(selected)


def measure_agreement(root: str | Path, human_labels: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Measure judge/human agreement on a human-labeled holdout.

    The calibration step before trusting the judge to filter training data:
    a judge whose grades do not track a human's on a small holdout should not
    be deciding what a policy trains on. The measurement ships with the
    pipeline rather than being promised - see
    ``examples/17_judge_recorded_episodes.py``.

    Args:
        root: Dataset root directory.
        human_labels: Mapping of episode index to the human's labels for the
            holdout, each a dict with ``quality`` (required, one of
            :data:`QUALITY_GRADES`) and optionally ``failure_mode``. Each key
            is an episode index on the shared non-negative-whole-number
            domain, checked like every other spelling of that quantity: a key
            that is not one selects the wrong episode to calibrate against,
            not a slower calibration.

    Returns:
        Dict with ``episodes_compared``, ``quality_agreement`` and
        ``failure_mode_agreement`` (fractions in [0, 1]; the failure-mode
        fraction is ``None`` when no holdout entry carries one), and
        ``disagreements`` - one ``{episode, field, judge, human}`` row per
        mismatch, so the calibration report names what to look at.

    Raises:
        FileNotFoundError: If no sidecar exists yet.
        ValueError: If the holdout is empty, an entry is malformed, or no
            holdout episode carries a judge annotation - agreement over
            nothing is not a measurement.
    """
    if not isinstance(human_labels, dict) or not human_labels:
        raise ValueError("measure_agreement: human_labels must be a non-empty mapping of episode index to labels.")

    document = read_labels(root)
    compared = 0
    quality_hits = 0
    mode_compared = 0
    mode_hits = 0
    disagreements: list[dict[str, Any]] = []
    for index, human in human_labels.items():
        if msg := non_negative_whole_number_error(index, "human_labels episode index", "measure_agreement"):
            raise ValueError(msg)
        if not isinstance(human, dict) or human.get("quality") not in QUALITY_GRADES:
            raise ValueError(
                f"measure_agreement: human_labels[{index}] must be a dict with 'quality' in {QUALITY_GRADES}."
            )
        record = document["episodes"].get(str(int(index)), {})
        judge = record.get("judge")
        if not judge:
            continue
        compared += 1
        if judge["quality"] == human["quality"]:
            quality_hits += 1
        else:
            disagreements.append(
                {"episode": int(index), "field": "quality", "judge": judge["quality"], "human": human["quality"]}
            )
        if "failure_mode" in human:
            mode_compared += 1
            if judge.get("failure_mode") == human["failure_mode"]:
                mode_hits += 1
            else:
                disagreements.append(
                    {
                        "episode": int(index),
                        "field": "failure_mode",
                        "judge": judge.get("failure_mode"),
                        "human": human["failure_mode"],
                    }
                )

    if compared == 0:
        raise ValueError(
            "measure_agreement: no holdout episode carries a judge annotation; "
            "agreement over nothing is not a measurement. Annotate the holdout first."
        )
    return {
        "episodes_compared": compared,
        "quality_agreement": quality_hits / compared,
        "failure_mode_agreement": (mode_hits / mode_compared) if mode_compared else None,
        "disagreements": disagreements,
    }
