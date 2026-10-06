"""Live Strands Decider episode judge: a real recording, a real decider, a real sidecar.

Records two MuJoCo episodes of the SO-100 mock policy with a per-episode
predicate stop - one given too few steps to reach the target (deterministic
failure), one given enough (deterministic success) - records their verdicts,
and labels both through :class:`~strands_robots.tools.decider_judge.DeciderJudge`
against a running decider. What a unit test with a fake server cannot show:
the real ``/v1/systemone`` accepts the request this backend builds from
decoded camera frames, and every answer is either written through the
precedence path or deferred - never a failure, never a changed verdict.

Gated behind ``STRANDS_DECIDER_URL`` and a server that answers ``/health``.

Run (decider in its own environment; it brings torch and the vision tower)::

    pip install "strands-decider[vision] @ git+https://github.com/strands-labs/strands-decider@3e94e9d"
    strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --vision --port 8000 \\
        --model-name strands-decider-2B-hobson-v19@bb282d7
    STRANDS_DECIDER_URL=http://127.0.0.1:8000 pytest -m decider tests_integ/tools/test_decider_judge_live.py
"""

from __future__ import annotations

import json
import os
import urllib.request

import pytest

pytestmark = pytest.mark.decider

_URL = os.getenv("STRANDS_DECIDER_URL", "")
if not _URL:
    pytest.skip("STRANDS_DECIDER_URL unset: no decider server to judge with", allow_module_level=True)
pytest.importorskip("mujoco", reason="requires the [sim-mujoco] extra to record episodes")
pytest.importorskip("lerobot", reason="requires the [lerobot] extra to record and decode camera frames")


def _health_or_skip(url: str) -> dict:
    """The server's /health payload, or a module skip when no vision decider answers."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=5) as response:  # noqa: S310 - test-owned URL
            return json.loads(response.read())
    except OSError as e:
        pytest.skip(f"no decider answering at {url}: {e}", allow_module_level=True)


_HEALTH = _health_or_skip(_URL)
if _HEALTH.get("vision") is not True:
    pytest.skip(f"the decider at {_URL} was not started with --vision", allow_module_level=True)


from strands_robots import Robot  # noqa: E402
from strands_robots.episode_labels import (  # noqa: E402
    FAILURE_MODES,
    QUALITY_GRADES,
    read_labels,
    record_deterministic_verdicts,
)
from strands_robots.tools.decider_judge import DeciderJudge  # noqa: E402

TASK = "sweep the arm base past 0.3 rad"
SUCCESS_CLAUSE = {"all": [{"predicate": "joint_above", "joint": "Rotation", "value": 0.3}]}
# The mock policy crosses 0.3 rad on step 24: 15 steps run out, 40 reach it.
BUDGETS = (15, 40)


def _record(root: str) -> list[dict]:
    sim = Robot("so100", mesh=False)
    try:
        started = sim.start_recording(
            repo_id="local/decider_judge_live", root=root, fps=50, task=TASK, cameras=["default"], overwrite=True
        )
        assert started["status"] == "success", started
        verdicts = []
        for episode, budget in enumerate(BUDGETS):
            sim.reset()
            result = sim.run_policy(
                robot_name="so100",
                policy_provider="mock",
                instruction=TASK,
                n_steps=budget,
                control_frequency=50.0,
                stop_when=SUCCESS_CLAUSE,
                seed=episode,
            )
            assert result["status"] == "success", result
            payload = next(c["json"] for c in result["content"] if "json" in c)
            verdicts.append(
                {
                    "episode": episode,
                    "success": payload["stopped_reason"] == "predicate",
                    "failure": payload["stopped_reason"] == "budget",
                }
            )
        assert sim.stop_recording()["status"] == "success"
    finally:
        sim.destroy()
    return verdicts


def test_a_recorded_dataset_is_labeled_or_deferred_and_no_verdict_moves(tmp_path) -> None:
    root = str(tmp_path / "dataset")
    verdicts = _record(root)
    assert [v["success"] for v in verdicts] == [False, True]
    record_deterministic_verdicts(root, verdicts, benchmark="so100_pan_reach")
    before = {k: r["deterministic"] for k, r in read_labels(root)["episodes"].items()}

    result = DeciderJudge(_URL).judge_dataset(root, task=TASK)
    payload = next(c["json"] for c in result["content"] if "json" in c)

    assert result["status"] == "success", result["content"][0]["text"]
    assert sorted(payload["labeled"] + payload["deferred"]) == [0, 1]
    after = read_labels(root)["episodes"]
    assert {k: r["deterministic"] for k, r in after.items()} == before
    for row in payload["episodes"]:
        assert set(row["answers"]) == {"quality", "failure_mode", "success"}
        assert 0.0 <= row["answers"]["success"]["probability"] <= 1.0
        judge = after[str(row["episode"])].get("judge")
        if row["outcome"] == "labeled":
            assert judge["quality"] in QUALITY_GRADES
            assert judge["failure_mode"] is None or judge["failure_mode"] in FAILURE_MODES
            assert judge["model"] == f"strands-decider:{_HEALTH['model']}"
        else:
            assert judge is None
