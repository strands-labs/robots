"""Pin the :class:`~strands_robots.training.rl.base_algo.BaseRLAlgo` lifecycle contracts.

``BaseRLAlgo.train()`` supplies the default on-policy loop
(``validate -> setup -> (collect_rollout, update)* -> save_checkpoint``) that
every RL backend inherits, and ``load_checkpoint`` supplies the shared
weight-restore path. Three inherited contracts had no coverage because the
concrete PPO/SAC suites always drive a valid spec, a positive ``log_interval``,
and a real saved checkpoint:

  * ``train`` fails closed when :meth:`validate` reports problems (never runs
    ``setup``/rollout on an unlaunchable spec).
  * ``train`` still writes exactly one final checkpoint when ``log_interval`` is
    disabled (``0``), so a run never returns ``checkpoint_dir=None``.
  * ``load_checkpoint`` raises a clear ``FileNotFoundError`` when ``policy.pt``
    is absent, rather than a bare loader error.

These mirror the ``_BareTrainer`` approach in
``tests/training/test_trainer_base_contract.py``: a minimal concrete subclass
that implements only the abstract surface so calls fall through to the base
defaults under test.
"""

from __future__ import annotations

import pytest

from strands_robots.training.base import TrainSpec
from strands_robots.training.rl.base_algo import BaseRLAlgo, RLTrainSpec


class _BareRLAlgo(BaseRLAlgo):
    """Smallest legal ``BaseRLAlgo``: stubbed hooks that record their calls.

    The hooks do no real learning; they only let the inherited ``train`` loop
    run so its checkpoint/validation branches are exercised. ``validate``
    returns whatever ``validate_returns`` is set to, so a test can force the
    fail-closed branch.
    """

    def __init__(self) -> None:
        self.validate_returns: list[str] = []
        self.calls: list[str] = []
        self.saved_iterations: list[int | None] = []

    @property
    def provider_name(self) -> str:
        return "bare_rl"

    def validate(self, spec: TrainSpec) -> list[str]:
        return list(self.validate_returns)

    def setup(self, spec: RLTrainSpec) -> None:
        self.calls.append("setup")
        self.steps_per_iter = 1

    def collect_rollout(self) -> dict[str, float]:
        self.calls.append("collect_rollout")
        return {"mean_reward": 1.0}

    def update(self) -> dict[str, float]:
        self.calls.append("update")
        return {"loss": 0.5}

    def save_checkpoint(self, output_dir: str, iteration: int | None = None) -> str:
        self.calls.append("save_checkpoint")
        self.saved_iterations.append(iteration)
        return output_dir


def test_train_fails_closed_on_validation_problems() -> None:
    """A spec that fails :meth:`validate` yields an error before any work runs.

    The default loop must return a terminal ``error`` naming the problems and
    must not proceed to ``setup``/``collect_rollout`` on an unlaunchable spec.
    """
    algo = _BareRLAlgo()
    algo.validate_returns = ["env_factory is None", "total_timesteps <= 0"]

    result = algo.train(RLTrainSpec())

    assert result.status == "error"
    assert result.job_id == ""
    assert "validation failed" in result.message
    assert "env_factory is None" in result.message
    assert "total_timesteps <= 0" in result.message
    # Fail-closed: no lifecycle hook ran once validation reported problems.
    assert algo.calls == []


def test_train_writes_a_final_checkpoint_when_log_interval_disabled() -> None:
    """With ``log_interval=0`` the loop saves no intermediate checkpoint.

    The end-of-train fallback must then still persist exactly one checkpoint at
    the final iteration, so a successful run never reports ``checkpoint_dir``
    of ``None``.
    """
    algo = _BareRLAlgo()
    # steps_per_iter is 1 (set in setup) -> num_iters == total_timesteps.
    spec = RLTrainSpec(total_timesteps=3, log_interval=0, output_dir="/tmp/rl-out")

    result = algo.train(spec)

    assert result.status == "success"
    assert result.checkpoint_dir == "/tmp/rl-out"
    # Exactly one save happened: the end-of-train fallback, at the final iter.
    assert algo.calls.count("save_checkpoint") == 1
    assert algo.saved_iterations == [3]


def test_load_checkpoint_missing_policy_pt_raises(tmp_path) -> None:
    """Loading a directory with no ``policy.pt`` fails loud with the dir named."""
    algo = _BareRLAlgo()

    with pytest.raises(FileNotFoundError, match="no policy.pt in checkpoint dir"):
        algo.load_checkpoint(str(tmp_path))
