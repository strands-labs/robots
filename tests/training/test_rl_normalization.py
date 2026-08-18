"""Behavior contracts for the RL observation normalizer.

``EmpiricalNormalization`` (strands_robots.training.rl.normalization) documents
five behaviors beyond the running-statistics happy path that RL trainers rely
on: scale-only whitening (``center=False``), a warmup freeze after ``until``
samples, an empty-batch no-op, the eval-mode freeze, and the batch shape both
entry points accept. Each is a small branch that is easy to break silently, so
pin them directly.

The shape contract and the empty-batch no-op are one rule at two sizes: a batch
that cannot be folded must leave the estimator untouched. ``update`` reads row 0
as the batch axis, so an unbatched ``(num_obs,)`` observation is arithmetically
indistinguishable from ``num_obs`` samples of a one-feature stream, and whitening
broadcasts rather than failing - the pooled statistics and the reshaped return
value are both silent. The buffers are persisted, so ``count`` is committed only
once the fold succeeds: an inflated count is not recoverable, since it shrinks
the blend rate for every later batch and brings the ``until`` freeze forward.

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

import re

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


# Batches a normalizer built for per-feature shape ``(4,)`` cannot fold, and why.
_UNFOLDABLE = [
    pytest.param((4,), "unbatched observation, read as 4 samples of 1 feature", id="flat"),
    pytest.param((3, 1), "one feature broadcast across all 4", id="too-few-features"),
    pytest.param((3, 3), "wrong feature count", id="wrong-features"),
    pytest.param((2, 1, 4), "an extra dimension ahead of the features", id="extra-dim"),
    pytest.param((), "no batch axis at all", id="scalar"),
]


def _tensor_of_shape(shape: tuple[int, ...]):
    """A float tensor of exactly ``shape``, with distinct ascending values."""
    rows = 1
    for dim in shape:
        rows *= dim
    return torch.arange(1.0, 1.0 + max(rows, 1)).reshape(shape)


def _warmed(width: int = 4):
    """A normalizer for per-feature shape ``(width,)`` with established statistics."""
    from strands_robots.training.rl.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(width, device="cpu")
    norm.train()
    for _ in range(6):
        norm.update(torch.arange(1.0, width + 1).repeat(4, 1))
    return norm


@pytest.mark.parametrize(("shape", "why"), _UNFOLDABLE)
@pytest.mark.parametrize("route", ["forward", "update"])
def test_a_batch_that_cannot_be_folded_leaves_the_estimator_untouched(
    shape: tuple[int, ...], why: str, route: str
) -> None:
    """A rejected batch must change no buffer, through either entry point.

    This is the empty-batch no-op generalised: ``update`` advances ``count`` by
    row 0 of whatever it is handed, so a batch it cannot meaningfully fold either
    pools the per-feature statistics into one number or fails part-way through
    with the sample count already advanced. Both outcomes are written into the
    next checkpoint, and both are silent.
    """
    norm = _warmed()
    before = {name: buf.clone() for name, buf in norm.state_dict().items()}
    x = _tensor_of_shape(shape)

    with pytest.raises(ValueError, match="expected a batched tensor"):
        norm(x) if route == "forward" else norm.update(x)

    after = norm.state_dict()
    drifted = sorted(name for name in before if not torch.equal(before[name], after[name]))
    assert drifted == [], f"{route} left {drifted} changed by a rejected batch ({why})"


@pytest.mark.parametrize(
    ("shape", "broadcast_to"),
    [
        pytest.param((4,), (1, 4), id="flat"),
        pytest.param((3, 1), (3, 4), id="too-few-features"),
        pytest.param((), (1, 4), id="scalar"),
    ],
)
def test_whitening_never_returns_a_shape_the_caller_did_not_pass(
    shape: tuple[int, ...], broadcast_to: tuple[int, ...]
) -> None:
    """``forward`` documents "same shape as ``x``"; broadcasting is the alternative.

    Whitening divides by buffers of shape ``(1, *shape)``, so these three inputs
    are broadcast up into the buffers' shape rather than failing, and the caller
    gets back a ``broadcast_to`` tensor it never passed in - no error, no log.
    Refusing is the only outcome that keeps the documented return contract.
    """
    norm = _warmed()
    x = _tensor_of_shape(shape)

    with pytest.raises(ValueError) as excinfo:
        out = norm(x, update=False)
        pytest.fail(f"whitening a {tuple(x.shape)} input returned {tuple(out.shape)}, expected a refusal")

    assert str(tuple(x.shape)) in str(excinfo.value)
    assert tuple(x.shape) != broadcast_to, "premise: the broadcast shape differs from the input"


def test_the_deployed_eval_mode_path_refuses_an_unbatched_observation() -> None:
    """An exported policy whitens with ``update=False`` in eval mode.

    ``BaseRLAlgo.evaluate`` scores a rollout on exactly this path, so it is the
    one an unbatched single observation is most likely to reach. Nothing folds
    there, which leaves the reshaped return value as the only symptom.
    """
    norm = _warmed()
    norm.eval()

    single = torch.arange(1.0, 5)
    with pytest.raises(ValueError, match="expected a batched tensor"):
        norm(single, update=False)

    batched = norm(single.unsqueeze(0), update=False)
    assert tuple(batched.shape) == (1, 4)


def test_the_refusal_names_the_required_shape_and_its_remedy_works() -> None:
    """Parse the remedy out of the message and apply it: it must fix the call.

    Asserting on wording pins the sentence; applying the remedy pins that the
    message is true. A refusal that names neither the shape wanted nor the way
    to produce it sends the caller to read the source.
    """
    norm = _warmed()
    with pytest.raises(ValueError) as excinfo:
        norm(torch.arange(1.0, 5))
    message = str(excinfo.value)

    assert "(batch, 4)" in message, message
    match = re.search(r"x\.unsqueeze\((\d+)\)", message)
    assert match, f"no applicable remedy in: {message}"

    fixed = torch.arange(1.0, 5).unsqueeze(int(match.group(1)))
    whitened = norm(fixed)
    assert tuple(whitened.shape) == (1, 4)


def test_a_fold_that_fails_on_dtype_does_not_advance_the_sample_count() -> None:
    """The count is committed after the fold, not before it.

    An integer observation tensor has an acceptable shape and still cannot be
    folded - ``torch.var_mean`` refuses it - so it reaches the arithmetic the
    shape check cannot screen. Advancing ``count`` first would leave the
    estimator claiming samples it never folded, which permanently shrinks the
    blend rate applied to every later batch.
    """
    from strands_robots.training.rl.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(4, device="cpu")
    norm.train()
    norm.update(torch.arange(1.0, 5).repeat(4, 1))
    count_before = int(norm.count)
    mean_before = norm.mean.clone()

    with pytest.raises(RuntimeError):
        norm.update(torch.arange(1, 5).repeat(4, 1))

    assert int(norm.count) == count_before
    assert torch.allclose(norm.mean, mean_before)


def test_two_folds_reproduce_the_population_statistics_of_their_union() -> None:
    """The accepted path is unchanged: chunked Welford equals the exact answer.

    Committing ``count`` after the fold instead of before it must not change any
    number, so pin the arithmetic against the population mean/std of the two
    batches taken together rather than against a recorded snapshot.
    """
    from strands_robots.training.rl.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(2, device="cpu")
    norm.train()
    first = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    second = torch.tensor([[4.0, 4.0], [6.0, 6.0]])
    norm.update(first)
    norm.update(second)

    both = torch.cat([first, second])
    assert int(norm.count) == 4
    assert torch.allclose(norm.mean, both.mean(dim=0))
    assert torch.allclose(norm.std, both.std(dim=0, unbiased=False))
