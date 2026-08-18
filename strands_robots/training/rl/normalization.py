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
        """
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
        """
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
