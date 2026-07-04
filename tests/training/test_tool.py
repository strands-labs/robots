"""Tests for the train_policy agent tool (Strands @tool wrapper)."""

import importlib
import json

import pytest

from strands_robots.training.base import TrainResult

# ``from strands_robots.tools import train_policy`` resolves via the package's
# lazy __getattr__ (tools/__init__.py) to the *tool object* (a
# DecoratedFunctionTool), which CodeQL's points-to treats as a non-callable
# class instance, flagging every ``train_policy(...)`` call. import_module
# returns the module and we bind the callable via attribute access -- the same
# idiom used by tests/tools/test_gr00t_container_hardening.py.
_tp = importlib.import_module("strands_robots.tools.train_policy")
train_policy = _tp.train_policy


def _text(res):
    return res["content"][0]["text"]


def _json(res):
    """Extract the structured ``{"json": ...}`` content block.

    The tool returns the canonical ``{status, content:[...]}`` only — structured
    fields (job_id / checkpoint_dir / metrics / exported_model) live in a json
    content block, NOT as sibling keys of the result dict.
    """
    for block in res["content"]:
        if "json" in block:
            return block["json"]
    raise AssertionError(f"no json content block in result: {res}")


@pytest.fixture
def dataset_root(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 5}))
    return str(tmp_path)


class TestActions:
    def test_list(self):
        res = train_policy(action="list")
        assert res["status"] == "success"
        for p in ("mock", "lerobot_local", "groot", "cosmos3"):
            assert p in _text(res)

    def test_validate_clean(self, dataset_root, tmp_path):
        res = train_policy(
            action="validate",
            provider="mock",
            dataset_root=dataset_root,
            base_model="m",
            output_dir=str(tmp_path / "o"),
            steps=10,
        )
        assert res["status"] == "success"
        assert "valid and launchable" in _text(res)

    def test_validate_reports_problems(self, tmp_path):
        res = train_policy(
            action="validate",
            provider="mock",
            dataset_root=str(tmp_path / "nope"),
            base_model="m",
            output_dir=str(tmp_path / "o"),
            steps=0,
        )
        assert res["status"] == "error"
        assert "validation problems" in _text(res)

    def test_missing_required_args(self):
        res = train_policy(action="train", provider="mock")
        assert res["status"] == "error"
        assert "required" in _text(res)

    def test_train_mock_full_loop(self, dataset_root, tmp_path):
        out = str(tmp_path / "out")
        res = train_policy(
            action="train",
            provider="mock",
            dataset_root=dataset_root,
            base_model="mock/base",
            output_dir=out,
            steps=50,
        )
        assert res["status"] == "success", _text(res)
        data = _json(res)
        assert data["job_id"]
        assert data["checkpoint_dir"]
        assert data["metrics"]["learning"] is True
        assert "create_policy(" in _text(res)
        # the result dict must NOT be extended beyond {status, content}
        assert set(res.keys()) == {"status", "content"}

    def test_export_after_train_succeeds(self, dataset_root, tmp_path):
        # Regression: export action used to reach a private _latest_checkpoint
        # via getattr and silently fail for providers (mock/cosmos) that lacked
        # it. It now uses the public latest_checkpoint() ABC method.
        out = str(tmp_path / "out")
        train_policy(
            action="train",
            provider="mock",
            dataset_root=dataset_root,
            base_model="mock/base",
            output_dir=out,
            steps=10,
        )
        res = train_policy(
            action="export",
            provider="mock",
            dataset_root=dataset_root,
            base_model="mock/base",
            output_dir=out,
        )
        assert res["status"] == "success", _text(res)
        assert _json(res)["exported_model"]
        assert set(res.keys()) == {"status", "content"}

    def test_export_without_checkpoint_errors(self, dataset_root, tmp_path):
        res = train_policy(
            action="export",
            provider="mock",
            dataset_root=dataset_root,
            base_model="mock/base",
            output_dir=str(tmp_path / "never"),
        )
        assert res["status"] == "error"
        assert "no checkpoint" in _text(res)

    def test_status_requires_job_id(self):
        res = train_policy(action="status", provider="mock")
        assert res["status"] == "error"
        assert "job_id" in _text(res)

    def test_status_verdict(self):
        res = train_policy(action="status", provider="mock", job_id="mock-123")
        assert res["status"] == "success"
        assert _json(res)["metrics"]["learning"] is True
        assert set(res.keys()) == {"status", "content"}

    def test_unknown_action(self, dataset_root, tmp_path):
        res = train_policy(
            action="frobnicate",
            provider="mock",
            dataset_root=dataset_root,
            base_model="m",
            output_dir=str(tmp_path / "o"),
        )
        assert res["status"] == "error"
        assert "Unknown action" in _text(res)


class TestProviderRouting:
    def test_lerobot_validate_routes_to_lerobot(self, dataset_root, tmp_path):
        # non-native policy_type -> lerobot-specific validation message
        res = train_policy(
            action="validate",
            provider="lerobot_local",
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "o"),
            steps=10,
            extra={"policy_type": "openvla"},
        )
        assert res["status"] == "error"
        assert "not LeRobot-native" in _text(res)

    def test_groot_requires_embodiment(self, dataset_root, tmp_path):
        res = train_policy(
            action="validate",
            provider="groot",
            dataset_root=dataset_root,
            base_model="nvidia/GR00T-N1.5-3B",
            output_dir=str(tmp_path / "o"),
            steps=10,
            extra={"groot_root": "/tmp"},  # missing launch script -> also errors, but embodiment first
        )
        assert res["status"] == "error"
        assert "embodiment is required" in _text(res)


class TestInputSafety:
    """Pin: agent-supplied values cannot smuggle subprocess flags or escape paths.

    ``train_policy`` is an agent ``@tool``; every value reaches a trainer's
    native config / argv-parity helper (training runs in-process now, not via
    subprocess). ``validate()`` / ``_security_problems`` must reject the
    injection vectors up front (AGENTS.md Review Learnings #92, "LLM Input
    Safety") for EVERY spec-consuming action, including export.
    """

    def test_export_rejects_unsafe_extra_key(self, dataset_root, tmp_path):
        # Regression: the export action must gate on validate() before calling
        # trainer.export(spec, ...) - it consumes the same agent-supplied spec.
        res = train_policy(
            action="export",
            provider="mock",
            dataset_root=dataset_root,
            base_model="m",
            output_dir=str(tmp_path / "o"),
            steps=10,
            extra={"--evil-flag": "x"},
        )
        assert res["status"] == "error"
        assert "not allowed" in _text(res)

    def test_export_rejects_output_dir_traversal(self, dataset_root):
        res = train_policy(
            action="export",
            provider="mock",
            dataset_root=dataset_root,
            base_model="m",
            output_dir="../../etc/evil",
            steps=10,
        )
        assert res["status"] == "error"
        # path traversal / protected-dir check fires before any export work.
        assert "validation problems" in _text(res)

    def test_extra_flag_key_rejected(self, dataset_root, tmp_path):
        res = train_policy(
            action="validate",
            provider="mock",
            dataset_root=dataset_root,
            base_model="m",
            output_dir=str(tmp_path / "o"),
            steps=10,
            extra={"--evil-flag": "x"},
        )
        assert res["status"] == "error"
        assert "not allowed" in _text(res)

    def test_base_model_leading_dash_rejected(self, dataset_root, tmp_path):
        res = train_policy(
            action="validate",
            provider="mock",
            dataset_root=dataset_root,
            base_model="--config_path=/etc/passwd",
            output_dir=str(tmp_path / "o"),
            steps=10,
        )
        assert res["status"] == "error"
        assert "must not start with '-'" in _text(res)

    def test_output_dir_protected_path_rejected(self, dataset_root):
        res = train_policy(
            action="validate",
            provider="mock",
            dataset_root=dataset_root,
            base_model="m",
            output_dir="/etc/cron.d/evil",
            steps=10,
        )
        assert res["status"] == "error"
        assert "protected system directory" in _text(res)

    def test_dataset_root_traversal_rejected(self, tmp_path):
        res = train_policy(
            action="validate",
            provider="mock",
            dataset_root="../../etc",
            base_model="m",
            output_dir=str(tmp_path / "o"),
            steps=10,
        )
        assert res["status"] == "error"
        assert "path traversal" in _text(res)

    def test_legitimate_dotted_extra_key_allowed(self, dataset_root, tmp_path):
        # lerobot dotted flags / cosmos hydra paths must still pass the allowlist.
        res = train_policy(
            action="validate",
            provider="mock",
            dataset_root=dataset_root,
            base_model="m",
            output_dir=str(tmp_path / "o"),
            steps=10,
            extra={"dataset.episodes": "1", "num_workers": "4"},
        )
        assert res["status"] == "success", _text(res)


class TestTrainErrorBoundary:
    """The ``action='train'`` error-path contract of the train_policy tool.

    train_policy is an agent ``@tool`` boundary: every failure mode must be
    reported as the canonical ``{"status": "error", "content": [...]}`` result,
    the tool must never raise past dispatch, and a failure at any stage must
    launch nothing further.
    """

    def test_train_reports_backend_validation_problems_and_launches_nothing(self, dataset_root, tmp_path):
        """A valid data source + output_dir clear the early argument guard, so
        the tool reaches the backend's own ``validate()``. A spec problem there
        (``steps <= 0``) must be reported as a validation problem and launch
        NOTHING -- no checkpoint may be written to disk."""
        out = tmp_path / "out"
        res = train_policy(
            action="train",
            provider="mock",
            dataset_root=dataset_root,
            base_model="m",
            output_dir=str(out),
            steps=0,
        )
        assert res["status"] == "error"
        assert "validation problems (nothing launched)" in _text(res)
        assert "steps must be > 0" in _text(res)
        # "launches nothing" is the behavioural claim: no checkpoint stub exists.
        assert not (out / "checkpoints" / "last").exists()

    def test_train_surfaces_runtime_train_failure_as_error(self, monkeypatch, dataset_root, tmp_path):
        """When ``validate()`` passes but the backend's ``train()`` fails at
        RUNTIME (e.g. CUDA OOM) -- distinct from a config problem -- the tool
        must surface it as a provider-tagged error, never as a success."""

        class _RuntimeFailTrainer:
            def validate(self, spec):
                return []

            def train(self, spec):
                return TrainResult(status="error", job_id="", message="CUDA out of memory")

        monkeypatch.setattr(_tp, "create_trainer", lambda provider: _RuntimeFailTrainer())
        res = train_policy(
            action="train",
            provider="mock",
            dataset_root=dataset_root,
            base_model="m",
            output_dir=str(tmp_path / "o"),
            steps=10,
        )
        assert res["status"] == "error"
        assert "training failed" in _text(res)
        assert "CUDA out of memory" in _text(res)

    def test_unexpected_internal_exception_is_caught_at_tool_boundary(self, monkeypatch, dataset_root, tmp_path):
        """An unexpected internal error (here ``create_trainer`` raising) must be
        caught and returned as a structured error -- an agent ``@tool`` must
        never let an exception propagate past dispatch and crash the agent."""

        def _boom(provider):
            raise RuntimeError("unexpected boom")

        monkeypatch.setattr(_tp, "create_trainer", _boom)
        res = train_policy(
            action="train",
            provider="mock",
            dataset_root=dataset_root,
            base_model="m",
            output_dir=str(tmp_path / "o"),
            steps=10,
        )
        assert res["status"] == "error"
        assert "train_policy error" in _text(res)
        assert "unexpected boom" in _text(res)
