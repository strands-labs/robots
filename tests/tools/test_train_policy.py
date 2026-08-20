"""Behavior tests for the ``train_policy`` Strands ``@tool`` wrapper.

Pins the tool's action dispatch and result-shaping contract against the
dependency-free ``mock`` trainer, so the whole
``list / validate / status / train / export`` surface is exercised without
torch, lerobot, or a GPU. The tool is a thin parse-and-format layer over
:class:`~strands_robots.training.base.Trainer`; these tests assert the
observable outputs (status, human text, and the structured ``{"json": ...}``
content block) rather than internal state.

Covered contracts:

* Every action routes to the right ``Trainer`` method and shapes a canonical
  ``{status, content:[...]}`` result with structured fields nested INSIDE the
  content list (never as sibling keys of ``status``/``content``).
* ``status`` requires ``job_id``; ``train``/``validate``/``export`` require a
  data source AND ``output_dir`` before a spec is built.
* ``validate`` surfaces preflight problems; ``export`` is validate-gated and
  refuses to run without a checkpoint.
* A full ``train -> export`` round trip on the mock backend writes a checkpoint
  stub and returns a loadable artifact path.
* The tool boundary never raises: an unknown provider is reported as a
  structured error, not an exception.
* A non-error ``train`` result is not always a loadable one: a ``running`` run
  and a ``success`` run that wrote no checkpoint are both reported with the step
  that applies to them, never with the completed-run load instruction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from strands_robots.tools.train_policy import train_policy as train_policy_tool


def _unwrap(t: Any) -> Any:
    # ``strands.tools.decorator.tool`` wraps the function in a
    # DecoratedFunctionTool; reach the underlying callable. Keep several
    # attribute names so the test survives SDK churn.
    for attr in ("_tool_func", "original_function", "__wrapped__", "func"):
        target = getattr(t, attr, None)
        if callable(target):
            return target
    return t


train_policy = _unwrap(train_policy_tool)


def _json_block(result: dict[str, Any]) -> dict[str, Any]:
    """Return the single ``{"json": ...}`` payload from a tool result's content."""
    blocks = [c["json"] for c in result["content"] if "json" in c]
    assert len(blocks) == 1, f"expected exactly one json block, got {blocks}"
    return blocks[0]


def _make_dataset_root(tmp_path: Path) -> str:
    """Write the minimal LeRobotDataset v3 marker the mock validator checks."""
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
    return str(root)


def _valid_kwargs(tmp_path: Path) -> dict[str, Any]:
    return {
        "provider": "mock",
        "dataset_root": _make_dataset_root(tmp_path),
        "output_dir": str(tmp_path / "out"),
        "base_model": "mock/base",
        "steps": 5,
    }


# --------------------------------------------------------------------------
# Result-shape contract shared by every branch
# --------------------------------------------------------------------------
def _assert_canonical(result: dict[str, Any]) -> None:
    assert set(result) == {"status", "content"}, "result must have exactly status + content"
    assert result["status"] in {"success", "error"}
    assert isinstance(result["content"], list) and result["content"]
    assert all(isinstance(block, dict) for block in result["content"])


# --------------------------------------------------------------------------
# list
# --------------------------------------------------------------------------
def test_list_reports_registered_providers() -> None:
    result = train_policy(action="list")
    _assert_canonical(result)
    assert result["status"] == "success"
    text = result["content"][0]["text"]
    assert "mock" in text and "lerobot_local" in text


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------
def test_status_without_job_id_is_a_loud_error() -> None:
    result = train_policy(action="status", provider="mock")
    _assert_canonical(result)
    assert result["status"] == "error"
    assert "job_id" in result["content"][0]["text"]


def test_status_with_job_id_reports_learning_verdict() -> None:
    result = train_policy(action="status", provider="mock", job_id="mock-123")
    _assert_canonical(result)
    assert result["status"] == "success"
    payload = _json_block(result)
    assert payload["job_id"] == "mock-123"
    assert payload["provider"] == "mock"
    assert payload["status"] == "success"
    # The mock reports a "learning" verdict in its metrics.
    assert payload["metrics"]["learning"] is True


# --------------------------------------------------------------------------
# spec-required guard (train / validate / export)
# --------------------------------------------------------------------------
def test_missing_data_source_and_output_dir_is_rejected() -> None:
    result = train_policy(action="train", provider="mock")
    _assert_canonical(result)
    assert result["status"] == "error"
    assert "output_dir" in result["content"][0]["text"]


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------
def test_validate_accepts_a_launchable_spec(tmp_path: Path) -> None:
    result = train_policy(action="validate", **_valid_kwargs(tmp_path))
    _assert_canonical(result)
    assert result["status"] == "success"
    assert "valid" in result["content"][0]["text"].lower()


def test_validate_surfaces_preflight_problems(tmp_path: Path) -> None:
    kwargs = _valid_kwargs(tmp_path)
    kwargs["method"] = "nonsense-method"  # unsupported -> a preflight problem
    kwargs["base_model"] = ""  # missing required base_model -> another problem
    result = train_policy(action="validate", **kwargs)
    _assert_canonical(result)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "nonsense-method" in text
    assert "base_model" in text


# --------------------------------------------------------------------------
# train -> export round trip
# --------------------------------------------------------------------------
def test_train_writes_checkpoint_and_returns_structured_result(tmp_path: Path) -> None:
    result = train_policy(action="train", **_valid_kwargs(tmp_path))
    _assert_canonical(result)
    assert result["status"] == "success"
    payload = _json_block(result)
    assert payload["provider"] == "mock"
    assert payload["job_id"].startswith("mock-")
    ckpt = Path(payload["checkpoint_dir"])
    assert ckpt.is_dir()
    assert (ckpt / "config.json").is_file()
    assert payload["metrics"]["latest_step"] == 5


def test_export_after_train_returns_loadable_artifact(tmp_path: Path) -> None:
    kwargs = _valid_kwargs(tmp_path)
    train_policy(action="train", **kwargs)  # produces checkpoints/last
    result = train_policy(action="export", **kwargs)
    _assert_canonical(result)
    assert result["status"] == "success"
    payload = _json_block(result)
    exported = Path(payload["exported_model"])
    # Mock export is identity: the returned path is the checkpoint dir on disk.
    assert exported.is_dir()
    assert exported.name == "last"


def test_export_without_a_checkpoint_is_rejected(tmp_path: Path) -> None:
    result = train_policy(action="export", **_valid_kwargs(tmp_path))
    _assert_canonical(result)
    assert result["status"] == "error"
    assert "no checkpoint" in result["content"][0]["text"].lower()


# --------------------------------------------------------------------------
# unknown action + tool-boundary error handling
# --------------------------------------------------------------------------
def test_unknown_action_lists_valid_actions(tmp_path: Path) -> None:
    kwargs = _valid_kwargs(tmp_path)
    kwargs["action"] = "obliterate"
    result = train_policy(**kwargs)
    _assert_canonical(result)
    assert result["status"] == "error"
    assert "Unknown action" in result["content"][0]["text"]


def test_unknown_provider_is_reported_not_raised(tmp_path: Path) -> None:
    kwargs = _valid_kwargs(tmp_path)
    kwargs["provider"] = "no-such-backend"
    result = train_policy(**kwargs)
    _assert_canonical(result)
    assert result["status"] == "error"
    # create_trainer raises ValueError; the tool boundary converts it to a result.
    assert "train_policy error" in result["content"][0]["text"]


def test_export_with_invalid_spec_exports_nothing(tmp_path: Path) -> None:
    kwargs = _valid_kwargs(tmp_path)
    train_policy(action="train", **kwargs)  # a checkpoint exists...
    kwargs["method"] = "nonsense-method"  # ...but the spec is now invalid
    result = train_policy(action="export", **kwargs)
    _assert_canonical(result)
    assert result["status"] == "error"
    assert "nothing exported" in result["content"][0]["text"]


def test_train_with_invalid_spec_launches_nothing(tmp_path: Path) -> None:
    kwargs = _valid_kwargs(tmp_path)
    kwargs["steps"] = 0  # not a positive integer -> preflight problem
    result = train_policy(action="train", **kwargs)
    _assert_canonical(result)
    assert result["status"] == "error"
    assert "nothing launched" in result["content"][0]["text"]


def test_backend_training_failure_is_shaped_as_error(tmp_path: Path) -> None:
    """A spec that passes validate but whose backend train() fails must surface
    as a structured error, not a success."""
    from strands_robots.training import register_trainer
    from strands_robots.training.base import TrainResult
    from strands_robots.training.mock import MockTrainer

    class _FailingTrainer(MockTrainer):
        @property
        def provider_name(self) -> str:
            return "mock_failing"

        def validate(self, spec: Any) -> list[str]:  # passes preflight
            return []

        def train(self, spec: Any) -> TrainResult:  # but the run fails
            return TrainResult(status="error", job_id="", message="cuda OOM")

    register_trainer("mock_failing", lambda: _FailingTrainer)

    kwargs = _valid_kwargs(tmp_path)
    kwargs["provider"] = "mock_failing"
    result = train_policy(action="train", **kwargs)
    _assert_canonical(result)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "training failed" in text
    assert "cuda OOM" in text


# --------------------------------------------------------------------------
# train: a non-error result is not always a loadable one
# --------------------------------------------------------------------------
# ``train`` returns three statuses, not two. Its sibling above pins that an
# ``error`` run is never shaped as a success; these pin the other non-success
# status. A ``running`` result is the managed-job backend reporting that the job
# outlived the submitting process (``SagemakerTrainer.train`` returns it when
# its local poll budget expires - pinned by
# ``test_exhausted_poll_budget_reports_running_not_success``), and a ``success``
# result carries whatever ``latest_checkpoint`` found, which is ``None`` when
# the run wrote no discoverable checkpoint tree (``LerobotTrainer.train`` passes
# that value straight through). Neither has an artifact to load, so neither may
# be reported with the completed-run load instruction.
_RUNNING_JOB_ID = "strands-train-abc123"


def _register_trainer_returning(name: str, result: Any) -> None:
    """Register a trainer whose ``train`` returns exactly ``result``."""
    from strands_robots.training import register_trainer
    from strands_robots.training.mock import MockTrainer

    class _FixedResultTrainer(MockTrainer):
        @property
        def provider_name(self) -> str:
            return name

        def validate(self, spec: Any) -> list[str]:  # passes preflight
            return []

        def train(self, spec: Any) -> Any:
            return result

        def status(self, job_id: str) -> Any:
            return result

    register_trainer(name, lambda: _FixedResultTrainer)


def _running_result() -> Any:
    from strands_robots.training.base import TrainResult

    return TrainResult(
        status="running",
        job_id=_RUNNING_JOB_ID,
        checkpoint_dir=None,
        metrics={"sagemaker_status": "InProgress"},
        message=f"training job '{_RUNNING_JOB_ID}' is still running after the local poll budget",
    )


def _train_text(tmp_path: Path, provider: str) -> tuple[dict[str, Any], str]:
    kwargs = _valid_kwargs(tmp_path)
    kwargs["provider"] = provider
    result = train_policy(action="train", **kwargs)
    _assert_canonical(result)
    return result, result["content"][0]["text"]


def test_the_load_instruction_names_a_provider_create_policy_refuses() -> None:
    """Premise: the value an artifact-less result interpolates is a dead end.

    Passes on both sides of the fix - it is a property of ``create_policy``, and
    it is what makes naming ``checkpoint_dir`` unconditionally a defect rather
    than a wording preference.
    """
    import pytest

    from strands_robots.policies import create_policy

    with pytest.raises(ValueError, match="Unknown policy provider"):
        create_policy(str(None))


def test_a_still_running_run_is_not_offered_as_a_loadable_checkpoint(tmp_path: Path) -> None:
    _register_trainer_returning("mock_running", _running_result())
    _, text = _train_text(tmp_path, "mock_running")
    assert "create_policy(" not in text, (
        "a run that has not finished has no artifact, but the report offered one: " + text
    )
    assert "has not finished" in text


def test_a_still_running_run_names_the_poll_step_for_its_job(tmp_path: Path) -> None:
    _register_trainer_returning("mock_running_poll", _running_result())
    _, text = _train_text(tmp_path, "mock_running_poll")
    assert "action='status'" in text and _RUNNING_JOB_ID in text, (
        "the result carries a job_id this tool's status action polls; the report must name it: " + text
    )


def test_the_poll_step_a_running_run_names_is_one_this_tool_answers(tmp_path: Path) -> None:
    """Follow the offered remedy verbatim - it must not be a second dead end."""
    import re

    _register_trainer_returning("mock_running_followed", _running_result())
    _, text = _train_text(tmp_path, "mock_running_followed")
    offered = re.search(r"train_policy\(action='status', provider='([^']+)', job_id='([^']+)'\)", text)
    assert offered is not None, f"no runnable poll step in: {text}"
    polled = train_policy(action="status", provider=offered.group(1), job_id=offered.group(2))
    _assert_canonical(polled)
    assert polled["status"] == "success", polled
    assert _json_block(polled)["status"] == "running"


def test_a_success_without_a_checkpoint_is_not_offered_as_a_loadable_one(tmp_path: Path) -> None:
    from strands_robots.training.base import TrainResult

    _register_trainer_returning(
        "mock_no_ckpt",
        TrainResult(status="success", job_id="j1", checkpoint_dir=None, message="run complete"),
    )
    _, text = _train_text(tmp_path, "mock_no_ckpt")
    assert "create_policy(" not in text, "no checkpoint was written, but a load was offered: " + text
    assert "no checkpoint path" in text


def test_a_completed_run_with_a_checkpoint_still_names_the_load(tmp_path: Path) -> None:
    """Control: the reported-completed path is unchanged."""
    from strands_robots.training.base import TrainResult

    _register_trainer_returning(
        "mock_done",
        TrainResult(
            status="success",
            job_id="j2",
            checkpoint_dir="/tmp/ft/checkpoints/last/pretrained_model",
            message="run complete",
        ),
    )
    _, text = _train_text(tmp_path, "mock_done")
    assert "Load the result with: create_policy('/tmp/ft/checkpoints/last/pretrained_model')" in text


def test_the_run_status_is_reported_verbatim_whatever_the_next_step(tmp_path: Path) -> None:
    """Control: only the next-step line is state-dependent.

    The tool-level ``status`` stays ``success`` for a submitted-but-unfinished
    run - the action is documented as "validate + launch training", and the
    sibling ``status`` action likewise maps a running job onto a successful poll
    - and the run's own status travels verbatim in the json block. Changing that
    vocabulary is a separate contract; this pins that this fix does not.
    """
    _register_trainer_returning("mock_running_status", _running_result())
    result, _ = _train_text(tmp_path, "mock_running_status")
    assert result["status"] == "success"
    payload = _json_block(result)
    assert payload["status"] == "running"
    assert payload["checkpoint_dir"] is None
    assert payload["job_id"] == _RUNNING_JOB_ID
