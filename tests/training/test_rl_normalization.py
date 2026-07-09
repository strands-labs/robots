"""Behavior contracts for the RL observation normalizer.

``EmpiricalNormalization`` (strands_robots.training.rl.normalization) documents
three behaviors beyond the running-statistics happy path that RL trainers rely
on: scale-only whitening (``center=False``), a warmup freeze after ``until``
samples, and an empty-batch no-op. Each is a small branch that is easy to break
silently, so pin them directly.

The convergence / eval-freeze happy path is covered separately in
``test_rl_ppo.py``; this module targets the contract edges. Imports are deferred
into each test (matching ``test_rl_ppo.py``) so the module collects even when
the CI torch mock stands in for a real PyTorch install.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_center_false_scales_without_subtracting_mean() -> None:
    """center=False divides by std+eps but keeps the input's origin.

    A centered pass subtracts the running mean; a scale-only pass must not, so
    the two disagree for a non-zero mean and the scale-only output equals
    ``x / (std + eps)`` exactly.
    """
    from strands_robots.training.rl.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(2, device="cpu")
    norm.train()
    norm(torch.tensor([[10.0, 20.0], [12.0, 24.0]]))
    norm.eval()  # freeze stats so update= has no effect on the assertion

    x = torch.tensor([[10.0, 20.0]])
    centered = norm(x, center=True, update=False)
    scaled = norm(x, center=False, update=False)

    assert not torch.allclose(centered, scaled)
    assert torch.allclose(scaled, x / (norm.std + norm.eps))


def test_until_freezes_statistics_after_warmup() -> None:
    """Once ``until`` samples are seen, further batches stop moving the stats.

    With ``until=5`` the third batch (count 4 -> 6) is the last that updates;
    a subsequent large batch must be ignored, leaving count and mean frozen.
    """
    from strands_robots.training.rl.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(1, device="cpu", until=5)
    norm.train()
    for _ in range(3):
        norm(torch.zeros(2, 1))  # count: 2 -> 4 -> 6 (last crosses `until`)

    count_before = int(norm.count)
    mean_before = norm.mean.clone()

    norm(torch.full((10, 1), 100.0))  # would swing mean hard if not frozen

    assert int(norm.count) == count_before
    assert torch.allclose(norm.mean, mean_before)


def test_empty_batch_update_is_noop() -> None:
    """A zero-row batch must not advance the count or divide by zero."""
    from strands_robots.training.rl.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(1, device="cpu")
    norm.train()
    norm(torch.zeros(4, 1))

    count_before = int(norm.count)
    mean_before = norm.mean.clone()

    norm.update(torch.zeros(0, 1))

    assert int(norm.count) == count_before
    assert torch.allclose(norm.mean, mean_before)
