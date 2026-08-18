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


class EmpiricalNormalization(nn.Module):
    """Normalize a tensor stream by its running (Welford) mean and std.

    The running statistics update only while the module is in training mode:
    :meth:`forward` additionally requires ``update=True``, and :meth:`update`
    folds a batch whenever the module is training. In eval mode the learned
    statistics are frozen at BOTH entry points, so an exported policy
    normalizes deterministically.

    Both entry points require a batched tensor whose trailing dimensions are the
    ``shape`` this normalizer was built for, and refuse anything else by name.
    Whitening broadcasts, so an unbatched ``(num_obs,)`` observation would
    otherwise come back as ``(1, num_obs)`` - a different shape than was passed
    in - and folding one would read its features as that many samples of a
    single feature. A batch that cannot be folded leaves the estimator
    untouched: the sample count is committed only once the fold succeeds, so a
    rejected batch never advances it.

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
    def mean(self) -> torch.Tensor:
        """Current running mean, shape ``shape`` (batch dim squeezed)."""
        return self._mean.squeeze(0).clone()

    @property
    def std(self) -> torch.Tensor:
        """Current running std, shape ``shape`` (batch dim squeezed)."""
        return self._std.squeeze(0).clone()

    def _batch_shape_error(self, x: torch.Tensor, context: str) -> str | None:
        """Return why ``x`` is not a batch this normalizer accepts, else ``None``.

        One rule for both entry points, so :meth:`forward` and :meth:`update`
        cannot disagree about which inputs they accept. A batch is acceptable
        when it has a leading batch dimension and its trailing dimensions are
        the per-feature ``shape`` the buffers were built for; a zero-row batch
        qualifies and is handled as a no-op by :meth:`update`.

        Args:
            x: The tensor the caller passed in.
            context: Method name to attribute the refusal to.

        Returns:
            An actionable message naming the received shape, the required shape
            and the remedy, or ``None`` when ``x`` is acceptable.
        """
        expected = tuple(self._mean.shape[1:])
        got = tuple(x.shape)
        if x.ndim >= 1 and got[1:] == expected:
            return None
        required = f"(batch, {', '.join(map(str, expected))})" if expected else "(batch,)"
        remedy = (
            "add the leading batch dimension (x.unsqueeze(0) for a single observation)"
            if got == expected
            else f"the trailing dimensions must be the per-feature shape {expected}"
        )
        return f"EmpiricalNormalization.{context}: expected a batched tensor of shape {required}, got {got}; {remedy}."

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
            ValueError: If ``x`` is not a batch of this normalizer's per-feature
                shape. Whitening broadcasts, so returning a differently shaped
                tensor is the alternative.
        """
        problem = self._batch_shape_error(x, "forward")
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
        exported policy whitens every observation. The shape check is here for
        the same reason, and runs before the mode check so both entry points
        answer for a malformed batch whichever mode they are in.

        A rejected batch leaves every buffer untouched. ``count`` is committed
        after the fold rather than before it, so a batch that cannot be folded -
        a non-float dtype still raises out of ``torch.var_mean`` - does not
        advance the sample count. An inflated count is not recoverable: it
        shrinks ``rate`` for every later batch, brings the ``until`` freeze
        forward, and is written into the next checkpoint.

        Args:
            x: Batch to fold, shape ``(batch, *shape)``.

        Raises:
            ValueError: If ``x`` is not a batch of this normalizer's per-feature
                shape. Folding one would read its features as samples.
        """
        problem = self._batch_shape_error(x, "update")
        if problem is not None:
            raise ValueError(problem)
        if not self.training:
            return
        if self.until is not None and int(self.count) >= self.until:
            return
        count_x = x.shape[0]
        if count_x == 0:
            return
        var_x, mean_x = torch.var_mean(x, dim=0, unbiased=False, keepdim=True)
        rate = count_x / float(int(self.count) + count_x)
        delta = mean_x - self._mean
        self._mean += rate * delta
        self._var += rate * (var_x - self._var + delta * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)
        self.count += count_x
