"""Running observation normalization for RL training.

``EmpiricalNormalization`` keeps a streaming (Welford) estimate of the mean and
standard deviation of an observation stream and whitens inputs with it. RL value
and policy networks train far more stably on zero-mean / unit-variance inputs
than on raw joint angles / velocities whose natural scales differ by orders of
magnitude, so on-policy algorithms (PPO here, SAC later) wrap their observation
inputs in one of these.

Adapted from the Amazon FAR Holosoma project (BSD-3-Clause,
https://github.com/amazon-far/holosoma), itself derived from the RSL-RL
``EmpiricalNormalization``. The statistics math is sim-agnostic, so it is ported
directly rather than reimplemented.
"""

from __future__ import annotations

import torch
from torch import nn


def _batched_shape_error(got: tuple[int, ...], feature_shape: tuple[int, ...]) -> str | None:
    """Return why ``got`` is not a batch of ``feature_shape``, or ``None`` if it is.

    :class:`EmpiricalNormalization` keeps one running mean/std *per feature* and
    reads a leading batch axis, so the only shape it can interpret is
    ``(batch, *feature_shape)``. Anything else is not a smaller version of that
    contract - it is a different one, and torch broadcasting hides the
    difference instead of refusing it.

    Refusing here rather than letting the arithmetic proceed is what keeps the
    class's two documented promises true. ``forward`` promises a tensor of "the
    same shape as ``x``", which broadcasting silently breaks: a single
    unbatched observation of shape ``(D,)`` comes back ``(1, D)``. And the
    statistics are *persistent buffers*, so a folded batch of the wrong shape
    is written into the next checkpoint: that same ``(D,)`` observation is read
    as ``D`` samples of a one-feature stream, advancing ``count`` by ``D`` for
    one observation and collapsing every per-feature mean and std onto a single
    pooled number - which is the whole quantity the normalizer exists to keep
    separate. Neither raises, and neither is recoverable once saved.

    The refusal also makes the class's behavior uniform. Most mis-shaped inputs
    already fail loudly, because ``var_mean`` over a ``(batch, *feature_shape)``
    mismatch that does *not* broadcast raises a ``RuntimeError``; only the
    shapes that happen to broadcast slipped through, and those raised nothing at
    all. This is checked before any buffer is touched, so a refused batch also
    leaves ``count`` where it was - previously a batch that raised inside
    ``var_mean`` had already advanced it, skewing the ``count_x / count`` update
    rate of every later batch.

    It is a module-private tuple rule rather than one of the shared domains in
    :mod:`strands_robots.utils` because those are deliberately torch-free and
    describe scalars and plain numeric vectors, while this describes a tensor's
    rank and trailing extents.

    Args:
        got: The received tensor's shape, as a tuple.
        feature_shape: The per-feature shape this normalizer was built for.

    Returns:
        ``None`` when ``got`` is ``(batch, *feature_shape)``; otherwise a
        message naming the shape read and the shape received, and - for a single
        unbatched observation - the call that batches it.
    """
    if len(got) == len(feature_shape) + 1 and got[len(got) - len(feature_shape) :] == feature_shape:
        return None
    expected = ", ".join(["batch", *(str(n) for n in feature_shape)]) + ("," if not feature_shape else "")
    remedy = " A single unbatched observation is normalized as x.unsqueeze(0)." if got == feature_shape else ""
    return (
        f"EmpiricalNormalization keeps one running mean and std per feature of shape "
        f"{feature_shape}, so both entry points read a batch of shape ({expected}); got {got}. "
        f"A different shape either broadcasts - returning a tensor that is not the caller's "
        f"shape, and pooling per-feature statistics onto one number inside buffers that "
        f"persist into the checkpoint - or fails inside the running-statistics update." + remedy
    )


class EmpiricalNormalization(nn.Module):
    """Normalize a tensor stream by its running (Welford) mean and std.

    The running statistics update only while the module is in training mode:
    :meth:`forward` additionally requires ``update=True``, and :meth:`update`
    folds a batch whenever the module is training. In eval mode the learned
    statistics are frozen at BOTH entry points, so an exported policy
    normalizes deterministically.

    Both entry points read a batch of shape ``(batch, *shape)`` and refuse
    anything else (see :func:`_batched_shape_error`). A single unbatched
    observation of shape ``shape`` - what
    :class:`~strands_robots.training.rl.gym_env.GymSimEnv` and the gymnasium
    contract hand back - has to be batched by the caller as ``x.unsqueeze(0)``.

    Args:
        shape: Per-feature shape of the observation (e.g. ``(num_obs,)``).
        device: Torch device the buffers live on.
        eps: Small constant added to the std before dividing, to bound the gain
            on near-constant features.
        until: Stop updating statistics once this many samples have been seen
            (``None`` keeps updating forever). Useful to freeze normalization
            after an initial warmup.
    """

    _mean: torch.Tensor
    _var: torch.Tensor
    _std: torch.Tensor
    count: torch.Tensor

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        device: torch.device | str | None = None,
        eps: float = 1e-2,
        until: int | None = None,
    ) -> None:
        super().__init__()
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape, device=device).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape, device=device).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape, device=device).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long, device=device))

    @property
    def _feature_shape(self) -> tuple[int, ...]:
        """Per-feature shape the running buffers were built for.

        Read off ``_mean`` rather than stored at construction so there is one
        owner of the value: the buffers are what the arithmetic broadcasts
        against, and ``load_state_dict`` can replace them.
        """
        return tuple(self._mean.shape[1:])

    @property
    def mean(self) -> torch.Tensor:
        """Current running mean, shape ``shape`` (batch dim squeezed)."""
        return self._mean.squeeze(0).clone()

    @property
    def std(self) -> torch.Tensor:
        """Current running std, shape ``shape`` (batch dim squeezed)."""
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor, center: bool = True, update: bool = True) -> torch.Tensor:
        """Whiten ``x`` with the running statistics.

        Args:
            x: Batched input, shape ``(batch, *shape)``.
            center: Subtract the running mean before scaling. Set ``False`` to
                only scale (e.g. for inputs that should keep their sign origin).
            update: Update the running statistics from this batch. Ignored
                unless the module is in training mode.

        Returns:
            The normalized tensor, same shape as ``x``.

        Raises:
            ValueError: If ``x`` is not shaped ``(batch, *shape)``. The check
                covers this entry point too, not only :meth:`update`: an
                unbatched observation broadcasts against the running buffers,
                so the returned tensor would not have the caller's shape even
                in eval mode, where nothing is folded at all.
        """
        problem = _batched_shape_error(tuple(x.shape), self._feature_shape)
        if problem is not None:
            raise ValueError(problem)
        if self.training and update:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        return x / (self._std + self.eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Fold a batch into the running mean/var.

        A no-op in eval mode, and once ``until`` samples have been seen. The
        mode check lives here rather than only in :meth:`forward` so the
        documented eval-mode freeze holds for a direct ``update`` call too:
        the statistics are persistent buffers, so folding a batch after
        ``eval()`` would be written into the next checkpoint and change how an
        exported policy whitens every observation.

        Raises:
            ValueError: If ``x`` is not shaped ``(batch, *shape)``. Checked
                before ``count`` moves, so a refused batch does not skew the
                ``count_x / count`` update rate of the batches after it.
        """
        problem = _batched_shape_error(tuple(x.shape), self._feature_shape)
        if problem is not None:
            raise ValueError(problem)
        if not self.training:
            return
        if self.until is not None and int(self.count) >= self.until:
            return
        count_x = x.shape[0]
        if count_x == 0:
            return
        self.count += count_x
        rate = count_x / float(self.count)
        var_x, mean_x = torch.var_mean(x, dim=0, unbiased=False, keepdim=True)
        delta = mean_x - self._mean
        self._mean += rate * delta
        self._var += rate * (var_x - self._var + delta * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)
