#!/usr/bin/env python3
"""Strands Decider as an episode-judge backend: typed, calibrated labels per episode.

:func:`~strands_robots.tools.episode_judge.create_judge_agent` runs an agent
loop per episode. Every label that loop writes comes from a closed vocabulary
(:data:`~strands_robots.episode_labels.QUALITY_GRADES`,
:data:`~strands_robots.episode_labels.FAILURE_MODES`, a yes/no opinion), which
is exactly the shape a decision model answers: :class:`DeciderJudge` asks a
running `Strands Decider <https://github.com/strands-labs/strands-decider>`_
server (``strands-decider serve <checkpoint> --vision``) three typed questions
about one episode's sampled camera frames, in ONE request:

- ``quality``: a choice over the quality grades, phrased per the
  quality-orthogonality contract - the execution visible in the frames, never
  the outcome.
- ``failure_mode``: a choice over the failure-mode tags plus ``"none"``, which
  maps to ``None``.
- ``success``: a yes/no, written as ``success_opinion``.

The deterministic verdict is never sent, so the success opinion is an
independent read rather than an echo, and the grade has no outcome in its
input to collapse onto (the failure the quality contract names). The label is
written through :func:`~strands_robots.tools.episode_judge.write_label`, so the
verdict precedence is the one every judge gets - only the ``judge`` block is
written and a disagreeing opinion is recorded as ``disputes_verdict``.

The decider returns a calibrated probability per answer, which is what lets
this backend abstain instead of guess. An episode is written only when all
three answers are confident (see ``min_confidence``); otherwise nothing is
written and the episode is reported ``deferred`` with the probabilities, for
the agent judge or a person. A judge block cannot be half-written - ``quality``
is required, and ``failure_mode=None`` is a claim ("none observed"), not an
absence - so a deferral is per episode, not per field.

The server is reached over HTTP with the standard library, so this module adds
no dependency: the model, torch and the vision tower live in the decider's own
environment. Decoding the camera frames needs the ``lerobot`` extra, as for
any multimodal judge (:func:`~strands_robots.tools.episode_judge.sample_frames`).

Like the judge tools, :meth:`DeciderJudge.judge_episode` and
:meth:`DeciderJudge.judge_dataset` return the ``{"status", "content"}``
envelope and never raise: a run over a hundred episodes reports the one it
could not label.
"""

from __future__ import annotations

import base64
import json
import math
import numbers
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from strands_robots.episode_labels import FAILURE_MODES, QUALITY_GRADES, read_labels

# The module, not its tools: binding the @tool objects here would make each one
# a tool of this module too, to every audit that walks a module's namespace.
from strands_robots.tools import episode_judge
from strands_robots.utils import (
    boolean_flag_error,
    non_negative_whole_number_error,
    partial_construction_repr,
    positive_count_error,
    positive_finite_number_error,
    refusal_repr,
)

#: Images one ``/v1/systemone`` request may carry - a fixed per-server limit of
#: the decider's vision engine. Frames are sampled position-major over every
#: camera, so ``n_frames * cameras`` must fit inside it.
DECIDER_MAX_IMAGES = 4

# What each tag and grade means, as the decider reads it. The option sets are
# built FROM the vocabularies below, so a tag or grade added to
# strands_robots.episode_labels without a description here fails this import
# rather than being silently unaskable.
_FAILURE_MODE_DESCRIPTIONS = {
    "jerky_motion": "the arm moves in jerky, shaky or stuttering steps",
    "near_miss": "the robot almost fails: it barely clears, grazes or nearly drops something",
    "camera_occlusion": "a camera view is blocked, so the action cannot be seen",
    "wrong_but_lucky": "the robot does the wrong thing but the outcome happens to work",
    "drift": "the robot slowly drifts away from where it should be",
    "collision": "the robot hits an object or the scene",
    "incomplete": "the task is not finished when the recording ends",
    "other": "a different problem not described by the other options",
}
_QUALITY_DESCRIPTIONS = {
    "low": "jerky, hesitant, wandering or poorly controlled motion",
    "medium": "acceptable motion with some hesitation, detours or unsteadiness",
    "high": "smooth, direct, well-controlled motion",
}

#: The failure-mode options sent to the decider: every tag in
#: :data:`~strands_robots.episode_labels.FAILURE_MODES` plus ``"none"``, the
#: explicit "no failure mode observed" answer, which is written as ``None``. It
#: is an option the model must pick, never what a missing answer turns into.
FAILURE_MODE_CRITERIA: dict[str, str] = {
    "none": "no failure is visible: the motion is controlled and nothing goes wrong",
    **{mode: _FAILURE_MODE_DESCRIPTIONS[mode] for mode in FAILURE_MODES},
}

#: The quality options sent to the decider, one per grade in
#: :data:`~strands_robots.episode_labels.QUALITY_GRADES`. Execution only - the
#: deterministic verdict already carries the outcome, and a grade that
#: re-derives it carries no information where it is consulted.
QUALITY_CRITERIA: dict[str, str] = {grade: _QUALITY_DESCRIPTIONS[grade] for grade in QUALITY_GRADES}

_QUALITY_INSTRUCTIONS = (
    "Grade the EXECUTION visible in these frames - smoothness, directness, control - "
    "not whether the task succeeded. A failed attempt executed cleanly up to the point "
    "of failure can be high; a successful one with jerky or lucky motion can be low."
)
_FAILURE_MODE_INSTRUCTIONS = "Which failure mode, if any, is visible in this episode?"


def _error(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": text}]}


def _json_payload(result: dict[str, Any]) -> dict[str, Any]:
    return next((block["json"] for block in result.get("content", []) if "json" in block), {})


def _text(result: dict[str, Any]) -> str:
    return " ".join(block["text"] for block in result.get("content", []) if "text" in block)


def _probability(value: Any, where: str) -> float:
    """A response number on [0, 1], or a ValueError naming where it came from."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"decider response: {where} must be a number in [0, 1], got {refusal_repr(value)}.")
    return float(value)


def _choice_answer(answers: dict[str, Any], name: str, options: dict[str, str]) -> dict[str, Any]:
    """Read one ``choice`` answer, refusing a shape this request did not ask for."""
    answer = answers.get(name)
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ValueError(f"decider response: answers[{name!r}] is not a choice answer: {refusal_repr(answer)}.")
    choice = answer.get("choice")
    if choice not in options:
        raise ValueError(
            f"decider response: answers[{name!r}]['choice'] is {refusal_repr(choice)}, not one of {sorted(options)}."
        )
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict):
        raise ValueError(f"decider response: answers[{name!r}]['probabilities'] is not an object.")
    return {
        "choice": choice,
        "confidence": _probability(answer.get("confidence"), f"answers[{name!r}]['confidence']"),
        "probabilities": {
            str(option): _probability(p, f"answers[{name!r}]['probabilities'][{option!r}]")
            for option, p in probabilities.items()
        },
    }


def _noul_answer(answers: dict[str, Any], name: str) -> float:
    """Read one yes/no answer: the probability the statement is true."""
    answer = answers.get(name)
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        raise ValueError(f"decider response: answers[{name!r}] is not a yes/no answer: {refusal_repr(answer)}.")
    return _probability(answer.get("noul"), f"answers[{name!r}]['noul']")


class DeciderJudge:
    """Label recorded episodes with a running Strands Decider vision server.

    One HTTP request per episode carries up to :data:`DECIDER_MAX_IMAGES`
    sampled camera frames and three typed questions (quality, failure mode,
    success). Confident answers are written as the episode's judge label
    through ``write_label``; anything less is deferred, never defaulted.

    Serve the decider first, in its own environment::

        pip install "strands-decider[vision] @ git+https://github.com/strands-labs/strands-decider@3e94e9d"
        strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --vision \\
            --model-name strands-decider-2B-hobson-v19@bb282d7

    The server's model name is what the label's ``model`` field records, so
    ``--model-name`` is where to pin the checkpoint revision for provenance.

    Args:
        url: Base URL of the decider server, ``http://`` or ``https://`` with a
            host (e.g. ``"http://127.0.0.1:8000"``). ``/health`` and
            ``/v1/systemone`` are appended.
        timeout: Seconds to wait for one HTTP response, a positive finite
            number. The first image request on a cold server includes the
            vision tower's warm-up.
        n_frames: Evenly spaced frames to sample per episode, a positive count,
            or ``None`` to take as many as the image limit allows for the
            dataset's cameras (``DECIDER_MAX_IMAGES // cameras``). An explicit
            count whose ``n_frames * cameras`` exceeds the limit is refused for
            that episode rather than truncated.
        min_confidence: The decider's own confidence an answer must EXCEED to
            be written, a number in ``[0, 1)``. For a choice that is the
            server's ``confidence`` (normalised max-probability,
            ``(N * p_max - 1) / (N - 1)``, so one threshold means the same for
            3 and 9 options); for yes/no it is ``|2p - 1|``, the same formula
            at N = 2. The default 0.6 therefore defers a yes/no whose
            probability lies in ``[0.2, 0.8]``.

    Raises:
        ValueError: If any argument is outside its domain.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 120.0,
        n_frames: int | None = None,
        min_confidence: float = 0.6,
    ) -> None:
        parsed = urllib.parse.urlsplit(url) if isinstance(url, str) else None
        if parsed is None or parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                f"DeciderJudge: url must be an http:// or https:// URL with a host, got {refusal_repr(url)}."
            )
        if msg := positive_finite_number_error(timeout, "timeout", "DeciderJudge"):
            raise ValueError(msg)
        if n_frames is not None and (msg := positive_count_error(n_frames, "n_frames", "DeciderJudge")):
            raise ValueError(msg)
        if (
            isinstance(min_confidence, bool)
            or not isinstance(min_confidence, numbers.Real)
            or not math.isfinite(float(min_confidence))
            or not 0.0 <= float(min_confidence) < 1.0
        ):
            raise ValueError(
                f"DeciderJudge: min_confidence must be a number in [0, 1), got {refusal_repr(min_confidence)}. "
                "No answer's confidence can exceed 1, so 1 would defer every episode."
            )
        self.url: str = url.rstrip("/")
        self.timeout: float = float(timeout)
        self.n_frames: int | None = None if n_frames is None else int(n_frames)
        self.min_confidence: float = float(min_confidence)

    def __repr__(self) -> str:
        try:
            return (
                f"DeciderJudge(url={self.url!r}, timeout={self.timeout}, "
                f"n_frames={self.n_frames}, min_confidence={self.min_confidence})"
            )
        except AttributeError:
            return partial_construction_repr(self)

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET (no payload) or POST JSON to the server; the decoded JSON object.

        Raises:
            ConnectionError: If the server is unreachable, times out, or answers
                with an HTTP error (its ``detail`` is quoted).
            ValueError: If the body is not a JSON object.
        """
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url + path,
            data=data,
            headers={"content-type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - scheme checked
                body = response.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise ConnectionError(f"decider at {self.url}{path} answered HTTP {e.code}: {detail}") from e
        except OSError as e:  # URLError and socket timeouts are OSErrors
            raise ConnectionError(f"decider at {self.url}{path} is unreachable: {e}") from e
        try:
            decoded = json.loads(body)
        except ValueError as e:
            raise ValueError(f"decider at {self.url}{path} returned a body that is not JSON: {e}") from e
        if not isinstance(decoded, dict):
            raise ValueError(f"decider at {self.url}{path} returned {type(decoded).__name__}, expected an object.")
        return decoded

    def _unconfident(self, quality: dict[str, Any], mode: dict[str, Any], success: float) -> list[str]:
        """Names of the answers whose confidence does not exceed ``min_confidence``."""
        names = [
            name
            for name, answer in (("quality", quality), ("failure_mode", mode))
            if answer["confidence"] <= self.min_confidence
        ]
        # |2p - 1| <= c  <=>  (1 - c) / 2 <= p <= (1 + c) / 2, compared on p so
        # the band edges are the exact numbers the docstring states.
        if (1.0 - self.min_confidence) / 2.0 <= success <= (1.0 + self.min_confidence) / 2.0:
            names.append("success")
        return names

    def judge_episode(self, root: str, episode: int, *, task: str, overwrite: bool = False) -> dict[str, Any]:
        """Label one episode, or defer it, from one decider request.

        Args:
            root: Dataset root directory (the directory containing ``meta/``).
                It must carry a deterministic verdict for ``episode``.
            episode: Episode index, a non-negative whole number.
            task: The instruction the episode was recorded for, a non-empty
                string. The success question asks about this task.
            overwrite: When ``False`` an episode that already has a judge
                label (a person's, an agent's, an earlier run's) is skipped
                rather than replaced. Must be a boolean.

        Returns:
            Envelope whose JSON payload carries ``episode`` and ``outcome``:
            ``"labeled"`` (with ``label``, the written episode record),
            ``"deferred"`` (with ``deferred``, the names of the unconfident
            answers - nothing written) or ``"skipped"``; plus ``answers``
            (all three decider answers with their probabilities and
            confidences) and ``latency_s`` for labeled and deferred
            episodes. ``status`` is ``"error"`` only when the episode could
            not be judged at all.
        """
        try:
            if msg := non_negative_whole_number_error(episode, "episode", "judge_episode"):
                return _error(msg)
            if not isinstance(task, str) or not task.strip():
                return _error(f"judge_episode: task must be a non-empty string, got {refusal_repr(task)}.")
            if msg := boolean_flag_error(overwrite, "overwrite", "judge_episode"):
                return _error(msg)
            episode = int(episode)
            started = time.monotonic()

            # The verdict is read only to refuse an episode that has none
            # before a decider request is spent on it; it is never sent.
            verdict = episode_judge.read_predicate_verdict(root, episode)
            if verdict["status"] != "success":
                return verdict
            described = episode_judge.load_episode(root, episode)
            if described["status"] != "success":
                return described
            meta = _json_payload(described)
            if meta["has_judge_label"] and not overwrite:
                return {
                    "status": "success",
                    "content": [
                        {"text": f"Episode {episode} already has a judge label; skipped (overwrite=False)."},
                        {"json": {"episode": episode, "outcome": "skipped"}},
                    ],
                }

            cameras = meta["camera_keys"]
            if not cameras:
                return _error(f"judge_episode: episode {episode} has no camera features; the decider judges images.")
            n_frames = DECIDER_MAX_IMAGES // len(cameras) if self.n_frames is None else self.n_frames
            if n_frames == 0 or n_frames * len(cameras) > DECIDER_MAX_IMAGES:
                return _error(
                    f"judge_episode: {n_frames} frame(s) x {len(cameras)} camera(s) is not between 1 and "
                    f"{DECIDER_MAX_IMAGES} images, the decider's per-request limit. Pass a smaller n_frames."
                )
            sampled = episode_judge.sample_frames(root, episode, n_frames=n_frames, include_images=True)
            if sampled["status"] != "success":
                return sampled
            images = [
                base64.b64encode(block["image"]["source"]["bytes"]).decode("ascii")
                for block in sampled["content"]
                if "image" in block
            ]
            samples = _json_payload(sampled)["samples"]

            state = (
                f"Recorded robot episode. Task: {task.strip()}\n"
                f"{len(images)} camera images: {len(samples)} frames sampled evenly over the episode's "
                f"{meta['length']} frames, in time order"
                + (f", each frame shown from cameras {', '.join(cameras)} in that order" if len(cameras) > 1 else "")
                + ". Each frame is the state before its action, so the last frame can be one control "
                "step short of the state the episode ended in."
            )
            response = self._request(
                "/v1/systemone",
                {
                    "state": state,
                    "images": images,
                    "questions": {
                        "quality": {
                            "type": "choice",
                            "instructions": _QUALITY_INSTRUCTIONS,
                            "criteria": QUALITY_CRITERIA,
                        },
                        "failure_mode": {
                            "type": "choice",
                            "instructions": _FAILURE_MODE_INSTRUCTIONS,
                            "criteria": FAILURE_MODE_CRITERIA,
                        },
                        "success": {"type": "noul", "instructions": f"Was the task completed: {task.strip()}"},
                    },
                },
            )
            answers = response.get("answers")
            if not isinstance(answers, dict):
                raise ValueError(f"decider response carries no answers object: {refusal_repr(response)[:300]}")
            quality = _choice_answer(answers, "quality", QUALITY_CRITERIA)
            mode = _choice_answer(answers, "failure_mode", FAILURE_MODE_CRITERIA)
            success = _noul_answer(answers, "success")
            model = response.get("model")
            if not isinstance(model, str) or not model:
                raise ValueError(f"decider response names no model: {refusal_repr(model)}.")
            answered = {"quality": quality, "failure_mode": mode, "success": {"probability": success}}

            deferred = self._unconfident(quality, mode, success)
            if deferred:
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"Episode {episode} deferred: {', '.join(deferred)} not above confidence "
                                f"{self.min_confidence}; nothing written - leave it for the agent judge or a person."
                            )
                        },
                        {
                            "json": {
                                "episode": episode,
                                "outcome": "deferred",
                                "deferred": deferred,
                                "answers": answered,
                                "latency_s": time.monotonic() - started,
                            }
                        },
                    ],
                }

            failure_mode = None if mode["choice"] == "none" else mode["choice"]
            written = episode_judge.write_label(
                root,
                episode,
                quality=quality["choice"],
                failure_mode=failure_mode,
                note=(
                    f"strands-decider: quality={quality['choice']} (confidence {quality['confidence']:.2f}), "
                    f"failure_mode={mode['choice']} (confidence {mode['confidence']:.2f}), "
                    f"P(success)={success:.2f}"
                ),
                success_opinion=success > 0.5,
                judge_model=f"strands-decider:{model}",
            )
            if written["status"] != "success":
                return written
            return {
                "status": "success",
                "content": [
                    {"text": _text(written)},
                    {
                        "json": {
                            "episode": episode,
                            "outcome": "labeled",
                            "label": _json_payload(written),
                            "answers": answered,
                            "latency_s": time.monotonic() - started,
                        }
                    },
                ],
            }
        except (ConnectionError, ValueError) as e:
            return _error(f"judge_episode: episode {episode}: {e}")

    def judge_dataset(
        self,
        root: str,
        *,
        task: str,
        episodes: list[int] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Label every episode of a dataset, deferring the unconfident ones.

        Checks the server once first (reachable, started with ``--vision``),
        so a stopped server is one error rather than one per episode. After
        that a failing episode is reported and the run continues.

        Args:
            root: Dataset root directory (the directory containing ``meta/``).
            task: The instruction the episodes were recorded for, a non-empty
                string.
            episodes: Episode indices to judge, or ``None`` for every episode
                the label sidecar holds a deterministic verdict for (an
                episode without one cannot be annotated).
            overwrite: When ``False`` episodes that already carry a judge
                label are skipped. Must be a boolean.

        Returns:
            Envelope whose JSON payload carries ``labeled``, ``deferred``,
            ``skipped`` and ``failed`` (episode index lists) and ``episodes``
            (one payload per episode, as :meth:`judge_episode` returns, or
            ``{"episode", "outcome": "failed", "error"}``). ``status`` is
            ``"error"`` when any episode failed or the server check did;
            labels already written stay written.
        """
        try:
            health = self._request("/health")
        except (ConnectionError, ValueError) as e:
            return _error(f"judge_dataset: {e}")
        if health.get("vision") is not True:
            return _error(
                f"judge_dataset: the decider at {self.url} was not started with --vision "
                f"(/health reports vision={refusal_repr(health.get('vision'))}); it would refuse every image request."
            )
        if episodes is None:
            try:
                document = read_labels(Path(root))
            except (OSError, ValueError) as e:
                return _error(f"judge_dataset: {e}")
            episodes = sorted(int(key) for key, record in document["episodes"].items() if "deterministic" in record)
        elif not isinstance(episodes, list) or not episodes:
            return _error(f"judge_dataset: episodes must be a non-empty list or None, got {refusal_repr(episodes)}.")

        rows: list[dict[str, Any]] = []
        buckets: dict[str, list[int]] = {"labeled": [], "deferred": [], "skipped": [], "failed": []}
        for episode in episodes:
            result = self.judge_episode(root, episode, task=task, overwrite=overwrite)
            if result["status"] == "success":
                row = _json_payload(result)
            else:
                row = {"episode": episode, "outcome": "failed", "error": _text(result)}
            rows.append(row)
            buckets[row["outcome"]].append(episode)

        summary = ", ".join(f"{len(indices)} {outcome}" for outcome, indices in buckets.items())
        failures = "; ".join(row["error"] for row in rows if row["outcome"] == "failed")
        return {
            "status": "error" if buckets["failed"] else "success",
            "content": [
                {
                    "text": f"Decider judge over {len(rows)} episode(s): {summary}."
                    + (f" {failures}" if failures else "")
                },
                {"json": {**buckets, "model": health.get("model"), "episodes": rows}},
            ],
        }
