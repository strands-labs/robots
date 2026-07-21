"""Mock trainer - the canonical no-dependency reference implementation.

Mirrors :class:`~strands_robots.policies.mock.MockPolicy`: zero heavy deps,
deterministic, used in tests and as the worked example of the :class:`Trainer`
contract. It does NOT launch any real training - ``train`` simulates a run by
writing a tiny checkpoint stub and a job record, so the full lifecycle
(``validate -> prepare -> train -> status -> export``) is exercisable on any
machine.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from strands_robots.training.base import Trainer, TrainResult, TrainSpec

_SUPPORTED_METHODS = {"full", "lora", "expert_only", "frozen_backbone"}


class MockTrainer(Trainer):
    """Reference :class:`Trainer` that simulates a training run (no deps)."""

    @property
    def provider_name(self) -> str:
        """Provider identity - the dependency-free ``mock`` reference trainer."""
        return "mock"

    def validate(self, spec: TrainSpec) -> list[str]:
        """Pure preflight: dataset presence, method sanity, lora/expert clash."""
        problems: list[str] = self._security_problems(spec)

        if not spec.dataset_root:
            problems.append("dataset_root is required")
        else:
            info = os.path.join(spec.dataset_root, "meta", "info.json")
            if not os.path.isfile(info):
                problems.append(f"dataset_root is not a LeRobotDataset v3 root (missing {info})")

        if not spec.base_model:
            problems.append("base_model is required")
        if not spec.output_dir:
            problems.append("output_dir is required")

        if spec.method not in _SUPPORTED_METHODS:
            problems.append(f"unsupported method '{spec.method}' (expected one of {sorted(_SUPPORTED_METHODS)})")
        if spec.method == "lora" and spec.tune.get("expert_only"):
            problems.append("lora and expert_only are mutually exclusive")

        if spec.steps <= 0:
            problems.append(f"steps must be > 0, got {spec.steps}")

        return problems

    def _job_path(self, output_dir: str) -> str:
        return os.path.join(output_dir, "mock_job.json")

    def latest_checkpoint(self, output_dir: str) -> str | None:
        """Return the simulated checkpoint dir (``checkpoints/last``), or None."""
        ckpt = os.path.join(output_dir, "checkpoints", "last")
        return ckpt if os.path.isdir(ckpt) else None

    def train(self, spec: TrainSpec) -> TrainResult:
        """Simulate a training run: write a checkpoint stub + job record."""
        problems = self.validate(spec)
        if problems:
            return TrainResult(
                status="error",
                job_id="",
                message="validation failed: " + "; ".join(problems),
            )

        self.prepare(spec)

        job_id = f"mock-{int(time.time())}"
        ckpt_dir = os.path.join(spec.output_dir, "checkpoints", "last")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Minimal checkpoint stub so export() has something to return.
        with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"provider": "mock", "base_model": spec.base_model, "steps": spec.steps},
                f,
            )

        record: dict[str, Any] = {
            "job_id": job_id,
            "status": "success",
            "checkpoint_dir": ckpt_dir,
            "metrics": {
                "latest_step": spec.steps,
                "latest_loss": 0.0,
                "learning": True,
                "liveness_ok": True,
            },
        }
        with open(self._job_path(spec.output_dir), "w", encoding="utf-8") as f:
            json.dump(record, f)

        return TrainResult(
            status="success",
            job_id=job_id,
            checkpoint_dir=ckpt_dir,
            metrics=record["metrics"],
            message=f"mock training complete ({spec.steps} steps simulated)",
        )

    def status(self, job_id: str) -> TrainResult:
        """Mock status is always 'learning' for a known job id format."""
        return TrainResult(
            status="success",
            job_id=job_id,
            metrics={"learning": True, "liveness_ok": True, "latest_loss": 0.0},
            message="mock job is (always) learning",
        )
