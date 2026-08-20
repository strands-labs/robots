"""Behavior tests for the episode-judge labeling tools.

The four tools (`load_episode`, `sample_frames`, `read_predicate_verdict`,
`write_label`) are the agent-facing surface over
:mod:`strands_robots.episode_labels`. They must return structured error dicts
rather than raise (a judge run over many episodes reports the episode it
could not read), and `write_label` must be structurally unable to overturn a
deterministic verdict - the tool-level pin of the precedence contract.

The dataset fixture is a synthetic on-disk LeRobot-v3-shaped layout written
with pyarrow, so no recording (and no shared cache) is involved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import strands_robots.tools.episode_judge as M
from strands_robots.episode_labels import deterministic_verdict, record_deterministic_verdicts

pq = pytest.importorskip("pyarrow.parquet", reason="the synthetic dataset fixture writes parquet")
import pyarrow as pa  # noqa: E402

# The tools are wrapped by the Strands @tool decorator; call the raw functions.
_load_episode = getattr(M.load_episode, "__wrapped__", None) or M.load_episode
_sample_frames = getattr(M.sample_frames, "__wrapped__", None) or M.sample_frames
_read_predicate_verdict = getattr(M.read_predicate_verdict, "__wrapped__", None) or M.read_predicate_verdict
_write_label = getattr(M.write_label, "__wrapped__", None) or M.write_label

#: Frame counts per episode in the synthetic dataset.
_EPISODE_LENGTHS = {0: 10, 1: 6}


def _json_payload(result: dict[str, Any]) -> dict[str, Any]:
    return next((c["json"] for c in result.get("content", []) if "json" in c), {})


def _text(result: dict[str, Any]) -> str:
    return " ".join(c.get("text", "") for c in result.get("content", []) if "text" in c)


def _write_dataset(root: Path) -> None:
    """Write a minimal LeRobot-v3-shaped dataset: meta/info.json + parquet."""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 50,
                "total_episodes": len(_EPISODE_LENGTHS),
                "total_frames": sum(_EPISODE_LENGTHS.values()),
                "features": {
                    "observation.state": {"dtype": "float32", "names": ["pan", "lift"]},
                    "action": {"dtype": "float32", "names": ["pan", "lift"]},
                },
            }
        )
    )

    episodes_table = pa.table(
        {
            "episode_index": list(_EPISODE_LENGTHS),
            "length": list(_EPISODE_LENGTHS.values()),
        }
    )
    pq.write_table(episodes_table, root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    episode_column: list[int] = []
    frame_column: list[int] = []
    timestamp_column: list[float] = []
    state_column: list[list[float]] = []
    for episode, length in _EPISODE_LENGTHS.items():
        for frame in range(length):
            episode_column.append(episode)
            frame_column.append(frame)
            timestamp_column.append(frame / 50.0)
            # A deliberate 0.5-per-step jump on the second dim so the motion
            # summary has a known answer.
            state_column.append([float(frame) * 0.1, float(frame) * 0.5])
    data_table = pa.table(
        {
            "episode_index": episode_column,
            "frame_index": frame_column,
            "timestamp": timestamp_column,
            "observation.state": state_column,
        }
    )
    pq.write_table(data_table, root / "data" / "chunk-000" / "file-000.parquet")


@pytest.fixture
def dataset_root(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    _write_dataset(root)
    return root


@pytest.fixture
def verdict_root(dataset_root):
    record_deterministic_verdicts(
        dataset_root,
        [
            {"episode": 0, "success": True, "failure": False, "steps": 10},
            {"episode": 1, "success": False, "failure": True, "steps": 6},
        ],
        benchmark="go2_walk_short",
    )
    return dataset_root


class TestLoadEpisode:
    def test_reports_length_features_and_label_state(self, verdict_root):
        result = _load_episode(str(verdict_root), 0)
        assert result["status"] == "success", _text(result)
        payload = _json_payload(result)
        assert payload["length"] == 10
        assert payload["total_episodes"] == 2
        assert payload["fps"] == 50
        assert payload["camera_keys"] == []
        assert payload["state_names"] == ["pan", "lift"]
        assert payload["has_deterministic_verdict"] is True
        assert payload["has_judge_label"] is False

    def test_unknown_episode_is_an_error_dict_not_a_raise(self, verdict_root):
        result = _load_episode(str(verdict_root), 9)
        assert result["status"] == "error"
        assert "Episode 9 not in dataset" in _text(result)

    def test_missing_dataset_is_an_error_dict(self, tmp_path):
        result = _load_episode(str(tmp_path / "nowhere"), 0)
        assert result["status"] == "error"
        assert "not an existing directory" in _text(result)

    def test_a_traversal_root_is_refused_before_any_read(self, dataset_root):
        result = _load_episode(str(dataset_root) + "/../dataset", 0)
        assert result["status"] == "error"


class TestSampleFrames:
    def test_samples_span_the_episode_and_summarize_motion(self, dataset_root):
        result = _sample_frames(str(dataset_root), 0, n_frames=3)
        assert result["status"] == "success", _text(result)
        payload = _json_payload(result)
        assert payload["length"] == 10
        assert [s["frame_index"] for s in payload["samples"]] == [0, 4, 9]
        assert payload["samples"][0]["state"] == [0.0, 0.0]
        # The synthetic state moves 0.5 per step on its largest dim.
        assert payload["max_state_delta"] == pytest.approx(0.5)
        # A linear ramp has zero third difference: the smoothness statistic
        # reads 0 however large the per-step velocity is (abs tolerance:
        # 0.1-per-step is not exactly representable, and the residue is
        # amplified by the 1/dt^3 scaling).
        assert payload["rms_state_jerk"] == pytest.approx(0.0, abs=1e-6)

    def test_n_frames_is_clamped_to_the_episode_length(self, dataset_root):
        result = _sample_frames(str(dataset_root), 1, n_frames=100)
        payload = _json_payload(result)
        assert len(payload["samples"]) == _EPISODE_LENGTHS[1]

    def test_domain_refusals_are_error_dicts(self, dataset_root):
        assert _sample_frames(str(dataset_root), -1)["status"] == "error"
        assert _sample_frames(str(dataset_root), 0, n_frames=0)["status"] == "error"
        assert _sample_frames(str(dataset_root), 0, include_images="yes")["status"] == "error"

    def test_images_on_a_camera_less_dataset_report_the_gap(self, dataset_root):
        # Refused from the dataset's own metadata, before any decode stack
        # (lerobot) is touched - so the diagnosis is the same on every install.
        result = _sample_frames(str(dataset_root), 0, include_images=True)
        assert result["status"] == "error"
        assert "no observation.images" in _text(result)

    def test_image_block_count_and_grouping_are_stated_in_the_text(self, dataset_root, monkeypatch):
        """A judge asked for n frames and handed n_frames x n_cameras
        unlabelled image blocks has no stated way to know adjacent blocks are
        the same timestep from different viewpoints unless the payload says
        so - the leading text block states the block count, the
        position-major grouping and the sorted camera order (PR #2486
        review). The decode itself is integration territory (lerobot video
        stack); here it is stubbed at the module seam."""
        info = json.loads((dataset_root / "meta" / "info.json").read_text())
        for camera in ("front", "overview"):
            info["features"][f"observation.images.{camera}"] = {"dtype": "video"}
        (dataset_root / "meta" / "info.json").write_text(json.dumps(info))

        def fake_blocks(root: Path, episode: int, positions: list[int]) -> list[dict[str, Any]]:
            return [
                {"image": {"format": "png", "source": {"bytes": b""}}} for _ in positions for _ in ("front", "overview")
            ]

        monkeypatch.setattr(M, "_decoded_image_blocks", fake_blocks)
        result = _sample_frames(str(dataset_root), 0, n_frames=2, include_images=True)
        assert result["status"] == "success", _text(result)
        assert (
            result["content"][0]["text"]
            == "Episode 0: sampled 2 of 10 frames; 4 image blocks, position-major, cameras sorted (front, overview)."
        )
        assert sum(1 for block in result["content"] if "image" in block) == 4

    def test_an_episode_with_no_frames_is_an_error_dict(self, dataset_root):
        result = _sample_frames(str(dataset_root), 7)
        assert result["status"] == "error"
        assert "no frames" in _text(result)


class TestSmoothnessIsCarriedByTheThirdDifference:
    """Pin the distinction between the two motion-summary fields, not their
    arithmetic: two episodes whose per-step maxima MATCH while their third
    differences differ. ``max_state_delta`` is a peak-velocity statistic
    pinned by the gross traverse - measured constant to 0.013% across a 4x
    range of true jerk on a real SO-101 recording (PR #2486 review) - so
    smoothness must be read from ``rms_state_jerk`` and only that field may
    separate these episodes.
    """

    _FPS = 50.0

    def _write_jerk_ladder(self, root: Path) -> None:
        """Three episodes, one state dim, all with per-step max delta 0.5.

        - episode 0 (smooth): linear ramp, third difference identically 0;
        - episode 1 (jerky): steps alternating 0.5 / 0.0, third difference
          alternating +/-1.0 per step (rms exactly 1.0);
        - episode 2: three frames - too short for a third difference.
        """
        (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (root / "data" / "chunk-000").mkdir(parents=True)
        (root / "meta" / "info.json").write_text(
            json.dumps(
                {
                    "fps": self._FPS,
                    "total_episodes": 3,
                    "total_frames": 23,
                    "features": {"observation.state": {"dtype": "float32", "names": ["pan"]}},
                }
            )
        )
        pq.write_table(
            pa.table({"episode_index": [0, 1, 2], "length": [10, 10, 3]}),
            root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        )
        episode_column: list[int] = []
        frame_column: list[int] = []
        timestamp_column: list[float] = []
        state_column: list[list[float]] = []
        smooth = [0.5 * i for i in range(10)]
        jerky = [0.5 * ((i + 1) // 2) for i in range(10)]  # 0, .5, .5, 1, 1, ...
        for episode, series in ((0, smooth), (1, jerky), (2, smooth[:3])):
            for frame, value in enumerate(series):
                episode_column.append(episode)
                frame_column.append(frame)
                timestamp_column.append(frame / self._FPS)
                state_column.append([value])
        pq.write_table(
            pa.table(
                {
                    "episode_index": episode_column,
                    "frame_index": frame_column,
                    "timestamp": timestamp_column,
                    "observation.state": state_column,
                }
            ),
            root / "data" / "chunk-000" / "file-000.parquet",
        )

    @pytest.fixture
    def ladder_root(self, tmp_path):
        root = tmp_path / "ladder"
        root.mkdir()
        self._write_jerk_ladder(root)
        return root

    def test_matching_maxima_differing_third_differences(self, ladder_root):
        smooth = _json_payload(_sample_frames(str(ladder_root), 0))
        jerky = _json_payload(_sample_frames(str(ladder_root), 1))

        # The first-difference statistic cannot separate the two episodes...
        assert smooth["max_state_delta"] == pytest.approx(0.5)
        assert jerky["max_state_delta"] == pytest.approx(smooth["max_state_delta"])

        # ...and the third-difference one does: 0 on the ramp, and on the
        # jerky series an rms raw third difference of exactly 1.0, scaled to
        # per-second-cubed by the recorded timestamp spacing.
        assert smooth["rms_state_jerk"] == pytest.approx(0.0)
        assert jerky["rms_state_jerk"] == pytest.approx(1.0 / (1.0 / self._FPS) ** 3)
        assert jerky["rms_state_jerk"] > smooth["rms_state_jerk"]

    def test_an_episode_too_short_for_a_third_difference_reports_none(self, ladder_root):
        payload = _json_payload(_sample_frames(str(ladder_root), 2))
        assert payload["max_state_delta"] == pytest.approx(0.5)
        assert payload["rms_state_jerk"] is None


class TestReadPredicateVerdict:
    def test_returns_the_deterministic_block(self, verdict_root):
        result = _read_predicate_verdict(str(verdict_root), 1)
        assert result["status"] == "success"
        payload = _json_payload(result)
        assert payload["success"] is False
        assert payload["failure"] is True
        assert payload["steps"] == 6

    def test_before_any_verdict_the_tool_names_the_remedy(self, dataset_root):
        result = _read_predicate_verdict(str(dataset_root), 0)
        assert result["status"] == "error"
        assert "record_deterministic_verdicts" in _text(result)


class TestWriteLabel:
    def test_annotation_round_trips(self, verdict_root):
        result = _write_label(
            str(verdict_root),
            0,
            quality="high",
            note="smooth approach",
            success_opinion=True,
            judge_model="mock-judge",
        )
        assert result["status"] == "success", _text(result)
        record = _json_payload(result)
        assert record["judge"]["quality"] == "high"
        assert record["judge"]["disputes_verdict"] is False
        assert record["judge"]["model"] == "mock-judge"

    def test_a_dispute_is_an_annotation_and_the_verdict_stands(self, verdict_root):
        """Tool-level pin of the acceptance criterion: judge disagreement is
        recorded as annotation, never as an overridden verdict."""
        before = deterministic_verdict(verdict_root, 1)
        assert before["success"] is False

        result = _write_label(str(verdict_root), 1, quality="medium", success_opinion=True)
        assert result["status"] == "success"
        assert "the verdict stands" in _text(result)
        assert _json_payload(result)["judge"]["disputes_verdict"] is True

        # The authoritative verdict on disk is untouched by the tool call.
        assert deterministic_verdict(verdict_root, 1) == before

    def test_an_episode_without_a_verdict_is_refused(self, verdict_root):
        result = _write_label(str(verdict_root), 9, quality="high")
        assert result["status"] == "error"
        assert "No deterministic verdict recorded" in _text(result)

    def test_domain_refusals_are_error_dicts(self, verdict_root):
        assert _write_label(str(verdict_root), 0, quality="excellent")["status"] == "error"
        assert _write_label(str(verdict_root), 0, quality="high", failure_mode="sloppy")["status"] == "error"
        assert _write_label(str(verdict_root), 0, quality="high", success_opinion="yes")["status"] == "error"


class TestJudgeAgentAssembly:
    def test_the_system_prompt_carries_the_doctrine(self):
        # The agent's operating contract is stated where the model reads it.
        assert "authoritative" in M.JUDGE_SYSTEM_PROMPT
        assert "never overturn" in M.JUDGE_SYSTEM_PROMPT
        for tool_name in ("read_predicate_verdict", "load_episode", "sample_frames", "write_label"):
            assert tool_name in M.JUDGE_SYSTEM_PROMPT

    def test_the_system_prompt_pins_quality_to_execution_not_outcome(self):
        """The grade is orthogonal to the verdict, and the prompt says so.

        An unsteered VLM's quality grade tracks the OUTCOME - low for every
        failure, high only for the success, measured on a graded
        five-recording ladder with exact ground truth (PR #2486 review) -
        which re-derives the verdict the judge can never overturn and
        carries no information where the grade is consulted
        (filter_episodes already gates on the deterministic verdict). The
        model reads nothing but the prompt and the tool schemas, so the
        contract has to be stated in both.
        """
        prompt = M.JUDGE_SYSTEM_PROMPT
        assert "not about the outcome" in prompt
        # Both directions of the decoupling are spelled out, because each is
        # the counterintuitive half for one verdict class.
        assert "failed episode" in prompt and "can be medium or high" in prompt
        assert "successful episode" in prompt and "can be low" in prompt
        # The tool schema is derived from the docstring's Args entry
        # (AGENTS.md convention 13), so the same contract must reach the
        # quality parameter's own description.
        quality_doc = M.write_label.tool_spec["inputSchema"]["json"]["properties"]["quality"]["description"]
        assert "not the outcome" in quality_doc
