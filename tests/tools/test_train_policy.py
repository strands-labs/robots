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
    kwargs["steps"] = 0  # steps must be > 0 -> preflight problem
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
