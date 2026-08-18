"""Behavior contracts for the RL observation normalizer.

``EmpiricalNormalization`` (strands_robots.training.rl.normalization) documents
four behaviors beyond the running-statistics happy path that RL trainers rely
on: scale-only whitening (``center=False``), a warmup freeze after ``until``
samples, an empty-batch no-op, and the eval-mode freeze. Each is a small branch
that is easy to break silently, so pin them directly.

The eval-mode freeze has TWO entry points - ``forward`` and the public
``update`` - and the class documents the freeze for both. That matters because
the statistics are persistent buffers: a batch folded in after ``eval()`` is
written into the next checkpoint and changes how an exported policy whitens
every observation, with nothing raised and nothing logged. ``BaseRLAlgo.evaluate``
relies on the freeze (it flips the normalizer to ``eval()`` around a scoring
rollout and restores the prior mode afterwards), so the round trip is pinned too.

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


@pytest.mark.parametrize("route", ["forward", "update"])
def test_eval_mode_freezes_the_statistics_through_either_entry_point(route: str) -> None:
    """Neither entry point may move the running statistics in eval mode.

    The class documents the freeze without qualifying which call reaches it, so
    ``forward`` and the public ``update`` have to agree. A batch folded in after
    ``eval()`` lands in the persistent buffers and therefore in the next
    checkpoint, so a route that ignores the mode silently changes what a
    deployed policy computes.
    """
    from strands_robots.training.rl.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(3, device="cpu")
    norm.train()
    norm(torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]))

    norm.eval()
    frozen_count = int(norm.count)
    frozen_mean = norm.mean.clone()
    frozen_std = norm.std.clone()

    swing = torch.full((8, 3), 100.0)  # would move the mean hard if folded in
    if route == "forward":
        norm(swing)
    else:
        norm.update(swing)

    assert int(norm.count) == frozen_count, (
        f"{route}() in eval mode advanced the sample count {frozen_count} -> {int(norm.count)}"
    )
    assert torch.allclose(norm.mean, frozen_mean), (
        f"{route}() in eval mode moved the running mean {frozen_mean.tolist()} -> {norm.mean.tolist()}"
    )
    assert torch.allclose(norm.std, frozen_std), (
        f"{route}() in eval mode moved the running std {frozen_std.tolist()} -> {norm.std.tolist()}"
    )


def test_a_batch_folded_in_after_eval_does_not_reach_the_checkpoint() -> None:
    """The statistics a checkpoint would carry must be untouched by an eval-mode fold.

    ``PpoTrainer`` / ``FastSacTrainer`` persist the normalizer with
    ``state_dict()`` and reload it in ``BaseRLAlgo.load_checkpoint``, so any
    buffer that shifts after ``eval()`` is what the next exported policy
    whitens with.
    """
    from strands_robots.training.rl.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(3, device="cpu")
    norm.train()
    norm(torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]))
    saved = {k: v.clone() for k, v in norm.state_dict().items()}

    norm.eval()
    norm.update(torch.full((8, 3), 100.0))

    drifted = sorted(k for k, v in norm.state_dict().items() if not torch.allclose(v.float(), saved[k].float()))
    assert drifted == [], f"buffers a checkpoint would carry drifted after eval(): {drifted}"


def test_update_still_folds_the_batch_in_training_mode() -> None:
    """The freeze must be mode-scoped, not a blanket disabling of ``update``.

    Direct ``update`` calls are how a caller folds a batch without whitening it
    (``test_empty_batch_update_is_noop`` uses that route), so training-mode
    behavior has to be unchanged.
    """
    from strands_robots.training.rl.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(2, device="cpu")
    norm.train()

    norm.update(torch.tensor([[2.0, 6.0], [4.0, 10.0]]))

    assert int(norm.count) == 2
    assert torch.allclose(norm.mean, torch.tensor([3.0, 8.0]))


def test_restoring_training_mode_resumes_updating() -> None:
    """A train -> eval -> train round trip must leave ``update`` working.

    ``BaseRLAlgo.evaluate`` flips the normalizer to ``eval()`` for a scoring
    rollout and restores the previous mode afterwards, so the freeze must be
    read from the live mode rather than latched on first use.
    """
    from strands_robots.training.rl.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(1, device="cpu")
    norm.train()
    norm.update(torch.zeros(2, 1))

    norm.eval()
    norm.update(torch.full((4, 1), 50.0))  # frozen: ignored
    count_after_eval = int(norm.count)

    norm.train()
    norm.update(torch.full((3, 1), 50.0))  # training again: folded in

    assert count_after_eval == 2
    assert int(norm.count) == 5
