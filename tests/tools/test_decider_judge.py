"""Behavior tests for the Strands Decider episode-judge backend.

``DeciderJudge`` asks a running decider three typed questions per episode and
writes the answer through ``write_label``. These tests stand a fake decider up
on a loopback ``ThreadingHTTPServer`` (no model, no torch, nothing leaves the
host) and pin what the backend promises:

- the answers map onto the closed vocabularies (``"none"`` -> ``None``) and
  land through the same verdict-precedence path every judge uses;
- an answer inside the abstention band is deferred - nothing is written, and
  the probabilities are reported - rather than defaulted;
- a broken server, a refused request and a malformed answer are per-episode
  errors in the ``{"status", "content"}`` envelope, never a raise mid-dataset.

The dataset is a synthetic LeRobot-v3-shaped layout written with pyarrow; the
camera decode (lerobot's video stack) is stubbed at the module seam the
episode-judge tests stub, so the image path runs on every install.
"""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import strands_robots.tools.episode_judge as episode_judge
from strands_robots.episode_labels import FAILURE_MODES, QUALITY_GRADES, read_labels, record_deterministic_verdicts
from strands_robots.tools.decider_judge import (
    DECIDER_MAX_IMAGES,
    FAILURE_MODE_CRITERIA,
    QUALITY_CRITERIA,
    DeciderJudge,
)

pq = pytest.importorskip("pyarrow.parquet", reason="the synthetic dataset fixture writes parquet")
import pyarrow as pa  # noqa: E402

TASK = "sweep the arm base past 0.3 rad"
MODEL = "strands-decider-2B-hobson-v19@bb282d7"
_EPISODE_LENGTH = 12


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    return next((c["json"] for c in result.get("content", []) if "json" in c), {})


def _text(result: dict[str, Any]) -> str:
    return " ".join(c.get("text", "") for c in result.get("content", []) if "text" in c)


def _choice(choice: str, options: dict[str, str], confidence: float) -> dict[str, Any]:
    rest = (1.0 - 0.9) / (len(options) - 1)
    return {
        "type": "choice",
        "choice": choice,
        "probabilities": {o: (0.9 if o == choice else rest) for o in options},
        "confidence": confidence,
    }


def answers(
    *,
    quality: str = "high",
    quality_confidence: float = 0.9,
    mode: str = "none",
    mode_confidence: float = 0.9,
    success: float = 0.95,
) -> dict[str, Any]:
    """A well-formed /v1/systemone response body."""
    return {
        "model": MODEL,
        "answers": {
            "quality": _choice(quality, QUALITY_CRITERIA, quality_confidence),
            "failure_mode": _choice(mode, FAILURE_MODE_CRITERIA, mode_confidence),
            "success": {"type": "noul", "noul": success},
        },
        "usage": {"input_tokens": 900, "output_tokens": 3},
        "latency_ms": 12.5,
    }


class FakeDecider:
    """A loopback stand-in for ``strands-decider serve --vision``.

    ``respond`` maps the parsed request to ``(status, body)``; every request
    body is kept in ``requests`` so a test can assert what was (not) sent.
    """

    def __init__(self) -> None:
        self.vision = True
        self.requests: list[dict[str, Any]] = []
        self.respond: Callable[[dict[str, Any]], tuple[int, Any]] = lambda _request: (200, answers())
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, status: int, body: Any) -> None:
                raw = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802 - http.server's dispatch name
                self._send(200, {"status": "ok", "model": MODEL, "vision": fake.vision})

            def do_POST(self) -> None:  # noqa: N802 - http.server's dispatch name
                request = json.loads(self.rfile.read(int(self.headers["content-length"])))
                fake.requests.append(request)
                self._send(*fake.respond(request))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def decider() -> Iterator[FakeDecider]:
    fake = FakeDecider()
    yield fake
    fake.close()


def _write_dataset(root: Path, cameras: tuple[str, ...], episodes: int = 2) -> None:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    features: dict[str, Any] = {"observation.state": {"dtype": "float32", "names": ["pan"]}}
    for camera in cameras:
        features[f"observation.images.{camera}"] = {"dtype": "video"}
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 50,
                "total_episodes": episodes,
                "total_frames": episodes * _EPISODE_LENGTH,
                "features": features,
            }
        )
    )
    pq.write_table(
        pa.table({"episode_index": list(range(episodes)), "length": [_EPISODE_LENGTH] * episodes}),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    rows = [(e, f) for e in range(episodes) for f in range(_EPISODE_LENGTH)]
    pq.write_table(
        pa.table(
            {
                "episode_index": [e for e, _ in rows],
                "frame_index": [f for _, f in rows],
                "timestamp": [f / 50.0 for _, f in rows],
                "observation.state": [[f * 0.01] for _, f in rows],
            }
        ),
        root / "data" / "chunk-000" / "file-000.parquet",
    )


def _dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cameras: tuple[str, ...] = ("front",)) -> Path:
    """A two-episode dataset with verdicts: episode 0 succeeded, episode 1 failed."""
    root = tmp_path / "dataset"
    root.mkdir()
    _write_dataset(root, cameras)
    record_deterministic_verdicts(
        root,
        [{"episode": 0, "success": True, "steps": 12}, {"episode": 1, "success": False, "failure": True}],
        benchmark="so100_pan_reach",
    )

    def fake_blocks(_root: Path, episode: int, positions: list[int]) -> list[dict[str, Any]]:
        return [
            {"image": {"format": "png", "source": {"bytes": f"ep{episode}-pos{p}-{c}".encode()}}}
            for p in positions
            for c in sorted(cameras)
        ]

    monkeypatch.setattr(episode_judge, "_decoded_image_blocks", fake_blocks)
    return root


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _dataset(tmp_path, monkeypatch)


class TestTheQuestionsAreTheVocabularies:
    def test_the_options_are_exactly_the_label_vocabularies(self) -> None:
        assert tuple(QUALITY_CRITERIA) == QUALITY_GRADES
        assert set(FAILURE_MODE_CRITERIA) == {"none", *FAILURE_MODES}

    def test_one_request_carries_the_frames_and_three_typed_questions(self, decider, dataset) -> None:
        DeciderJudge(decider.url).judge_episode(str(dataset), 0, task=TASK)
        (request,) = decider.requests
        assert [base64.b64decode(i) for i in request["images"]] == [f"ep0-pos{p}-front".encode() for p in (0, 4, 7, 11)]
        questions = request["questions"]
        assert questions["quality"]["type"] == "choice"
        assert set(questions["quality"]["criteria"]) == set(QUALITY_GRADES)
        assert questions["failure_mode"]["type"] == "choice"
        assert set(questions["failure_mode"]["criteria"]) == {"none", *FAILURE_MODES}
        assert questions["success"]["type"] == "noul"
        assert TASK in questions["success"]["instructions"]

    def test_the_quality_question_grades_execution_not_outcome(self, decider, dataset) -> None:
        DeciderJudge(decider.url).judge_episode(str(dataset), 0, task=TASK)
        instructions = decider.requests[0]["questions"]["quality"]["instructions"]
        assert "EXECUTION" in instructions
        assert "not whether the task succeeded" in instructions

    def test_the_deterministic_verdict_is_never_sent(self, decider, dataset) -> None:
        DeciderJudge(decider.url).judge_episode(str(dataset), 0, task=TASK)
        sent = json.dumps(decider.requests[0]).lower()
        assert "verdict" not in sent
        assert "deterministic" not in sent

    def test_two_cameras_share_the_image_limit(self, decider, tmp_path, monkeypatch) -> None:
        root = _dataset(tmp_path, monkeypatch, cameras=("wrist", "front"))
        DeciderJudge(decider.url).judge_episode(str(root), 0, task=TASK)
        images = [base64.b64decode(i).decode() for i in decider.requests[0]["images"]]
        assert len(images) == DECIDER_MAX_IMAGES
        assert images == ["ep0-pos0-front", "ep0-pos0-wrist", "ep0-pos11-front", "ep0-pos11-wrist"]
        assert "front, wrist" in decider.requests[0]["state"]

    def test_frames_over_the_image_limit_are_refused_not_truncated(self, decider, tmp_path, monkeypatch) -> None:
        root = _dataset(tmp_path, monkeypatch, cameras=("wrist", "front"))
        result = DeciderJudge(decider.url, n_frames=3).judge_episode(str(root), 0, task=TASK)
        assert result["status"] == "error"
        assert "per-request limit" in _text(result)
        assert decider.requests == []


class TestAConfidentAnswerIsWrittenThroughThePrecedencePath:
    def test_the_label_maps_onto_the_sidecar(self, decider, dataset) -> None:
        decider.respond = lambda _r: (200, answers(quality="medium", mode="jerky_motion", success=0.9))
        result = DeciderJudge(decider.url).judge_episode(str(dataset), 0, task=TASK)
        assert result["status"] == "success", _text(result)
        assert _payload(result)["outcome"] == "labeled"
        judge = read_labels(dataset)["episodes"]["0"]["judge"]
        assert judge["quality"] == "medium"
        assert judge["failure_mode"] == "jerky_motion"
        assert judge["success_opinion"] is True
        assert judge["disputes_verdict"] is False
        assert judge["model"] == f"strands-decider:{MODEL}"

    def test_none_is_written_as_no_failure_mode(self, decider, dataset) -> None:
        DeciderJudge(decider.url).judge_episode(str(dataset), 0, task=TASK)
        assert read_labels(dataset)["episodes"]["0"]["judge"]["failure_mode"] is None

    def test_the_probabilities_are_kept_in_the_note(self, decider, dataset) -> None:
        decider.respond = lambda _r: (200, answers(success=0.93))
        DeciderJudge(decider.url).judge_episode(str(dataset), 0, task=TASK)
        assert "P(success)=0.93" in read_labels(dataset)["episodes"]["0"]["judge"]["note"]

    def test_a_contrary_opinion_is_a_dispute_and_the_verdict_stands(self, decider, dataset) -> None:
        before = read_labels(dataset)["episodes"]["0"]["deterministic"]
        decider.respond = lambda _r: (200, answers(success=0.05))
        DeciderJudge(decider.url).judge_episode(str(dataset), 0, task=TASK)
        record = read_labels(dataset)["episodes"]["0"]
        assert record["judge"]["success_opinion"] is False
        assert record["judge"]["disputes_verdict"] is True
        assert record["deterministic"] == before

    def test_an_episode_without_a_verdict_is_refused_before_a_request(self, decider, dataset) -> None:
        labels = json.loads((dataset / "episode_labels.json").read_text())
        del labels["episodes"]["1"]
        (dataset / "episode_labels.json").write_text(json.dumps(labels))
        result = DeciderJudge(decider.url).judge_episode(str(dataset), 1, task=TASK)
        assert result["status"] == "error"
        assert "No deterministic verdict" in _text(result)
        assert decider.requests == []


class TestAnUnconfidentAnswerIsDeferredNotDefaulted:
    @pytest.mark.parametrize("probability", [0.2, 0.35, 0.5, 0.65, 0.8])
    def test_a_yes_no_inside_the_band_writes_nothing(self, decider, dataset, probability) -> None:
        decider.respond = lambda _r: (200, answers(success=probability))
        result = DeciderJudge(decider.url).judge_episode(str(dataset), 0, task=TASK)
        payload = _payload(result)
        assert result["status"] == "success", _text(result)
        assert payload["outcome"] == "deferred"
        assert payload["deferred"] == ["success"]
        assert payload["answers"]["success"]["probability"] == probability
        assert "judge" not in read_labels(dataset)["episodes"]["0"]

    @pytest.mark.parametrize("probability", [0.0, 0.19, 0.81, 1.0])
    def test_a_yes_no_outside_the_band_is_written(self, decider, dataset, probability) -> None:
        decider.respond = lambda _r: (200, answers(success=probability))
        result = DeciderJudge(decider.url).judge_episode(str(dataset), 0, task=TASK)
        assert _payload(result)["outcome"] == "labeled"
        assert read_labels(dataset)["episodes"]["0"]["judge"]["success_opinion"] is (probability > 0.5)

    @pytest.mark.parametrize("field", ["quality", "failure_mode"])
    @pytest.mark.parametrize(("confidence", "outcome"), [(0.3, "deferred"), (0.6, "deferred"), (0.61, "labeled")])
    def test_a_choice_must_exceed_the_threshold(self, decider, dataset, field, confidence, outcome) -> None:
        body = answers(**{f"{'quality' if field == 'quality' else 'mode'}_confidence": confidence})
        decider.respond = lambda _r: (200, body)
        payload = _payload(DeciderJudge(decider.url).judge_episode(str(dataset), 0, task=TASK))
        assert payload["outcome"] == outcome
        assert ("judge" in read_labels(dataset)["episodes"]["0"]) is (outcome == "labeled")
        if outcome == "deferred":
            assert payload["deferred"] == [field]

    def test_the_threshold_moves_the_band(self, decider, dataset) -> None:
        decider.respond = lambda _r: (200, answers(success=0.85))
        payload = _payload(DeciderJudge(decider.url, min_confidence=0.8).judge_episode(str(dataset), 0, task=TASK))
        assert payload["outcome"] == "deferred"


class TestAFailureIsReportedNotRaised:
    def test_an_unreachable_server_is_one_error_for_the_dataset(self, decider, dataset) -> None:
        url = decider.url
        decider.close()
        result = DeciderJudge(url, timeout=2.0).judge_dataset(str(dataset), task=TASK)
        assert result["status"] == "error"
        assert "unreachable" in _text(result)
        assert "judge" not in read_labels(dataset)["episodes"]["0"]

    def test_a_server_without_vision_is_refused_up_front(self, decider, dataset) -> None:
        decider.vision = False
        result = DeciderJudge(decider.url).judge_dataset(str(dataset), task=TASK)
        assert result["status"] == "error"
        assert "--vision" in _text(result)
        assert decider.requests == []

    def test_a_refused_request_fails_its_episode_and_the_run_continues(self, decider, dataset) -> None:
        def respond(request: dict[str, Any]) -> tuple[int, Any]:
            if base64.b64decode(request["images"][0]).startswith(b"ep0"):
                return 422, {"detail": "4 images; this server takes at most 2"}
            return 200, answers(success=0.1, mode="incomplete")

        decider.respond = respond
        result = DeciderJudge(decider.url).judge_dataset(str(dataset), task=TASK)
        payload = _payload(result)
        assert result["status"] == "error"
        assert payload["failed"] == [0]
        assert payload["labeled"] == [1]
        assert "HTTP 422" in _text(result) and "at most 2" in _text(result)
        assert read_labels(dataset)["episodes"]["1"]["judge"]["failure_mode"] == "incomplete"

    @pytest.mark.parametrize(
        "body",
        [
            {**answers(), "model": ""},
            {"model": MODEL},
            {
                **answers(),
                "answers": {**answers()["answers"], "quality": _choice("great", {"great": "", "fine": ""}, 0.9)},
            },
            {**answers(), "answers": {**answers()["answers"], "success": {"type": "noul", "noul": 1.5}}},
            {**answers(), "answers": {**answers()["answers"], "success": {"type": "choice"}}},
            b"not json",
            [1, 2],
        ],
        ids=[
            "no-model",
            "no-answers",
            "grade-outside-vocabulary",
            "probability-over-one",
            "wrong-type",
            "not-json",
            "not-object",
        ],
    )
    def test_a_malformed_answer_writes_nothing(self, decider, dataset, body) -> None:
        decider.respond = lambda _r: (200, body)
        result = DeciderJudge(decider.url).judge_episode(str(dataset), 0, task=TASK)
        assert result["status"] == "error"
        assert "judge" not in read_labels(dataset)["episodes"]["0"]

    def test_a_camera_less_dataset_is_an_error_dict(self, decider, tmp_path, monkeypatch) -> None:
        root = _dataset(tmp_path, monkeypatch, cameras=())
        result = DeciderJudge(decider.url).judge_episode(str(root), 0, task=TASK)
        assert result["status"] == "error"
        assert "no camera" in _text(result)

    @pytest.mark.parametrize(
        ("kwargs", "needle"), [({"task": ""}, "task"), ({"task": TASK, "overwrite": "no"}, "overwrite")]
    )
    def test_bad_arguments_are_error_dicts(self, decider, dataset, kwargs, needle) -> None:
        result = DeciderJudge(decider.url).judge_episode(str(dataset), 0, **kwargs)
        assert result["status"] == "error"
        assert needle in _text(result)


class TestADatasetRun:
    def test_every_verdict_bearing_episode_is_judged(self, decider, dataset) -> None:
        def respond(request: dict[str, Any]) -> tuple[int, Any]:
            first = base64.b64decode(request["images"][0])
            return 200, answers(success=0.9) if first.startswith(b"ep0") else answers(success=0.5)

        decider.respond = respond
        result = DeciderJudge(decider.url).judge_dataset(str(dataset), task=TASK)
        payload = _payload(result)
        assert result["status"] == "success", _text(result)
        assert (payload["labeled"], payload["deferred"], payload["failed"]) == ([0], [1], [])
        assert payload["model"] == MODEL
        assert "1 labeled, 1 deferred" in _text(result)

    def test_an_existing_label_is_kept_unless_overwrite(self, decider, dataset) -> None:
        episode_judge.write_label(str(dataset), 0, quality="low", note="a person looked", judge_model="human")
        result = DeciderJudge(decider.url).judge_dataset(str(dataset), task=TASK, episodes=[0])
        assert _payload(result)["skipped"] == [0]
        assert read_labels(dataset)["episodes"]["0"]["judge"]["model"] == "human"
        assert decider.requests == []

        DeciderJudge(decider.url).judge_dataset(str(dataset), task=TASK, episodes=[0], overwrite=True)
        assert read_labels(dataset)["episodes"]["0"]["judge"]["model"] == f"strands-decider:{MODEL}"

    def test_an_empty_episode_list_is_refused(self, decider, dataset) -> None:
        result = DeciderJudge(decider.url).judge_dataset(str(dataset), task=TASK, episodes=[])
        assert result["status"] == "error"


class TestConstruction:
    @pytest.mark.parametrize(
        ("kwargs", "needle"),
        [
            ({"url": "ftp://host"}, "url"),
            ({"url": "http://"}, "url"),
            ({"url": 8000}, "url"),
            ({"url": "http://h", "timeout": 0}, "timeout"),
            ({"url": "http://h", "n_frames": 0}, "n_frames"),
            ({"url": "http://h", "min_confidence": 1.0}, "min_confidence"),
            ({"url": "http://h", "min_confidence": -0.1}, "min_confidence"),
            ({"url": "http://h", "min_confidence": float("nan")}, "min_confidence"),
            ({"url": "http://h", "min_confidence": True}, "min_confidence"),
        ],
    )
    def test_a_value_outside_its_domain_is_refused(self, kwargs, needle) -> None:
        with pytest.raises(ValueError, match=needle):
            DeciderJudge(**kwargs)

    def test_the_repr_names_the_server(self) -> None:
        assert "http://127.0.0.1:8000" in repr(DeciderJudge("http://127.0.0.1:8000/"))
