"""Pin the :class:`~strands_robots.training.base.Trainer` base-class defaults.

Every shipped backend (mock / lerobot_local / groot / cosmos3 / rl) overrides
:meth:`Trainer.status` and :meth:`Trainer.latest_checkpoint`, so the *default*
implementations those methods provide are the contract a NEW backend inherits
when it does not override them - yet nothing exercised that fallback. These
tests define a minimal concrete ``Trainer`` (only the three abstract members)
and assert the inherited behaviour of every non-abstract default so a
regression in the base contract is caught, not just in a subclass override.
"""

from __future__ import annotations

from strands_robots.training.base import Trainer, TrainResult, TrainSpec


class _BareTrainer(Trainer):
    """Smallest legal Trainer: implements only the abstract surface.

    Deliberately overrides nothing else, so calls fall through to the
    ``Trainer`` base-class defaults under test.
    """

    @property
    def provider_name(self) -> str:
        return "bare"

    def validate(self, spec: TrainSpec) -> list[str]:
        return []

    def train(self, spec: TrainSpec) -> TrainResult:
        return TrainResult(status="success", job_id="bare-job")


def test_default_status_reports_polling_unsupported() -> None:
    """The default ``status`` returns an actionable error, not a fake verdict.

    A backend that cannot poll a detached job inherits this: the result is a
    terminal ``error`` (never a misleading ``running``/``success``), echoes the
    queried ``job_id``, and names the provider so the caller knows which
    backend declined.
    """
    result = _BareTrainer().status("job-123")

    assert result.status == "error"
    assert result.job_id == "job-123"
    assert "bare" in result.message
    assert "not supported" in result.message


def test_default_latest_checkpoint_returns_none() -> None:
    """A backend with no checkpoint-discovery layout inherits ``None``.

    ``None`` (not an empty string or a raised error) is the documented
    "no discoverable checkpoint" signal that the ``export`` action and resume
    logic branch on.
    """
    assert _BareTrainer().latest_checkpoint("/no/such/output/dir") is None


def test_default_export_returns_checkpoint_dir_unchanged() -> None:
    """HF-native backends need no conversion; the default is a passthrough."""
    trainer = _BareTrainer()
    assert trainer.export(TrainSpec(), "/tmp/ckpt/step_100") == "/tmp/ckpt/step_100"


def test_default_hardware_floor_is_single_24gb_gpu() -> None:
    """The advisory floor defaults to one 24 GB single-node GPU."""
    floor = _BareTrainer().hardware_floor

    assert floor == {"min_gpus": 1, "min_vram_gb": 24, "multinode": False}
