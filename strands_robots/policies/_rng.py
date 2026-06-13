"""Shared RNG reseed helper for Policy providers.

#331: ``Gr00tPolicy.reset`` reseeds Python ``random``, NumPy, torch CPU + CUDA,
and toggles cuDNN determinism, while ``Cosmos3Policy.reset`` only mutated the
global NumPy RNG. Two providers conforming to the same ``Policy`` contract must
behave identically for ``set_eval_seed``-style reproducibility (#187). This
module is the single source of truth for the client-side reseed so both
providers stay in parity.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def reseed_client_rngs(seed: int | None) -> None:
    """Reseed the client-side RNGs for per-episode reproducibility.

    Seeds Python ``random``, NumPy, and (if importable) torch CPU + CUDA, and
    toggles cuDNN into deterministic mode. ``None`` is a no-op. Best-effort:
    a missing torch is skipped silently (it is an optional dependency for the
    policy providers); any other failure is logged and swallowed because
    ``reset`` is a soft reproducibility hint, not a hard requirement.

    Args:
        seed: Master per-episode seed, or ``None`` to leave RNGs untouched.
    """
    if seed is None:
        return
    try:
        import random as _random

        _random.seed(seed)

        import numpy as _np

        _np.random.seed(seed)

        try:
            import torch as _torch

            _torch.manual_seed(seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed)
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
        except ImportError:
            # torch is optional for the policy providers (mock / service-only
            # installs); no torch RNG state to seed when it is not present.
            pass
    except Exception as exc:  # noqa: BLE001 - reset is best-effort
        logger.info("reseed_client_rngs: reseed failed (seed=%r): %s", seed, exc)
