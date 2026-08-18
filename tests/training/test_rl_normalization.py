"""Behavior contracts for the RL observation normalizer.

``EmpiricalNormalization`` (strands_robots.training.rl.normalization) documents
five behaviors beyond the running-statistics happy path that RL trainers rely
on: scale-only whitening (``center=False``), a warmup freeze after ``until``
samples, an empty-batch no-op, the eval-mode freeze, and the ``(batch, *shape)``
input contract. Each is a small branch that is easy to break silently, so pin
them directly.

The eval-mode freeze has TWO entry points - ``forward`` and the public
``update`` - and the class documents the freeze for both. That matters because
the statistics are persistent buffers: a batch folded in after ``eval()`` is
written into the next checkpoint and changes how an exported policy whitens
every observation, with nothing raised and nothing logged. ``BaseRLAlgo.evaluate``
relies on the freeze (it flips the normalizer to ``eval()`` around a scoring
rollout and restores the prior mode afterwards), so the round trip is pinned too.

The input-shape contract has the same persistence consequence and one more:
a shape the running buffers merely *broadcast* against is silently accepted, so
a single unbatched observation - the shape ``GymSimEnv`` and the gymnasium
contract hand back - is folded as one sample per feature, collapsing every
per-feature statistic onto one pooled number. Both entry points refuse it, and
the refusal is checked before ``count`` moves so a rejected batch cannot skew
the update rate of the batches after it.

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


@pytest.mark.parametrize("route", ["forward", "update"])
def test_an_unbatched_observation_is_refused_rather_than_broadcast(route: str) -> None:
    """A ``(D,)`` observation must be refused, not read as D one-feature samples.

    The class documents ``(batch, *shape)`` in and "the same shape as ``x``" out.
    A single unbatched observation satisfies neither: it broadcasts against the
    ``(1, D)`` buffers, so ``forward`` returns ``(1, D)`` for a ``(D,)`` input,
    and ``update`` folds it as D samples of a one-feature stream - advancing
    ``count`` by D for one observation and collapsing every per-feature mean and
    std onto a single pooled number, inside buffers that persist into the
    checkpoint.

    ``GymSimEnv`` hands back exactly this shape (the gymnasium contract is a flat
    observation array), so the caller most likely to reach it is following the
    package's own adapter.
    """
    from strands_robots.training.rl.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(4, device="cpu")
    norm.train()
    flat = torch.tensor([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match=r"batch of shape \(batch, 4\)"):
        norm(flat) if route == "forward" else norm.update(flat)

    assert int(norm.count) == 0


def test_a_feature_count_mismatch_that_broadcasts_is_refused() -> None:
    """A single-column batch must not have its stats written into every feature.

    ``(3, 1)`` into a 4-feature normalizer broadcasts, so it used to return
    ``(3, 4)`` and write that one column's mean and std into all four running
    slots. Every feature then whitens by a statistic measured from a different
    signal, which is exactly the per-feature separation this module exists to
    keep.
    """
    from strands_robots.training.rl.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(4, device="cpu")
    norm.train()

    with pytest.raises(ValueError, match=r"got \(3, 1\)"):
        norm(torch.tensor([[10.0], [20.0], [30.0]]))

    assert int(norm.count) == 0
    assert torch.allclose(norm.mean, torch.zeros(4))


@pytest.mark.parametrize("bad", [(3, 3), (2, 1, 4), ()], ids=["wrong-width", "extra-dim", "scalar"])
def test_a_refused_batch_leaves_the_running_count_where_it_was(bad: tuple[int, ...]) -> None:
    """The shape is checked before ``count`` moves, not after.

    ``count`` was advanced by ``x.shape[0]`` before ``var_mean`` ran, so a batch
    that failed inside the statistics update had already moved it. Since the
    Welford step weights each batch by ``count_x / count``, that left every
    later batch folded in at the wrong rate - a permanent skew from a batch that
    was never folded at all.
    """
    from strands_robots.training.rl.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(4, device="cpu")
    norm.train()
    norm.update(torch.zeros(1, 4))
    count_before = int(norm.count)

    with pytest.raises(ValueError):
        norm.update(torch.zeros(bad))

    assert int(norm.count) == count_before


def test_the_frozen_eval_path_also_refuses_an_unbatched_observation() -> None:
    """The refusal covers the deployed-policy path, where nothing is folded.

    In eval mode ``update`` returns early, so no statistic moves - but the
    whitening arithmetic still broadcasts, and an exported policy handed a
    ``(D,)`` observation gets a ``(1, D)`` tensor back. The shape contract is a
    property of the call, not of whether the batch is folded, so it has to hold
    with the statistics frozen too.
    """
    from strands_robots.training.rl.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(4, device="cpu")
    norm.train()
    norm.update(torch.zeros(4, 4))
    norm.eval()

    with pytest.raises(ValueError):
        norm(torch.tensor([1.0, 2.0, 3.0, 4.0]), update=False)


def test_the_refusal_names_the_call_that_batches_a_single_observation() -> None:
    """The message's remedy, applied verbatim, produces per-feature statistics.

    A refusal that only reports the mismatch leaves the caller to guess whether
    a flat observation is one sample or D of them. Parsing the remedy out of the
    message and running it is what proves the advice is both present and
    correct: after ``unsqueeze(0)`` the count reflects one observation and the
    four features keep four distinct means, rather than one pooled value.
    """
    import re

    from strands_robots.training.rl.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(4, device="cpu")
    norm.train()
    flat = torch.tensor([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError) as excinfo:
        norm(flat)

    remedy = re.search(r"normalized as (x\.unsqueeze\(0\))\.", str(excinfo.value))
    assert remedy is not None, f"no remedy in: {excinfo.value}"

    out = norm(flat.unsqueeze(0))
    assert tuple(out.shape) == (1, 4)
    assert int(norm.count) == 1
    assert torch.allclose(norm.mean, flat)


@pytest.mark.parametrize(
    ("shape", "batch"),
    [(4, (2, 4)), ((2, 3), (5, 2, 3)), (1, (3, 1)), (4, (0, 4))],
    ids=["vector", "multi-dim-feature", "single-feature", "empty-batch"],
)
def test_a_correctly_batched_input_is_accepted_unchanged(shape: int | tuple[int, ...], batch: tuple[int, ...]) -> None:
    """Every shape the contract already documents keeps working.

    The guard must refuse only what broadcasting used to hide, so the accepted
    domain is pinned alongside it: a plain feature vector, a multi-dimensional
    per-feature shape, a one-feature stream (where ``(batch, 1)`` is genuinely a
    batch and not a mis-shaped ``(batch,)``), and the documented zero-row no-op.
    """
    from strands_robots.training.rl.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(shape, device="cpu")
    norm.train()

    out = norm(torch.zeros(batch))

    assert tuple(out.shape) == batch
    assert int(norm.count) == batch[0]
