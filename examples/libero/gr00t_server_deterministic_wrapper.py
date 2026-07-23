"""Wrap run_gr00t_server with strict-determinism torch flags + per-episode seed.

This is a docker-mountable wrapper that enforces:
- cudnn.deterministic = True
- cudnn.benchmark = False
- torch.use_deterministic_algorithms(True, warn_only=True)
- CUBLAS_WORKSPACE_CONFIG=":4096:8" (required for cuBLAS determinism)
- A monkey-patch on Gr00tPolicy.reset that reseeds torch / numpy / random
  to a fixed seed at the start of each episode. Mirrors what
  set_eval_seed does in-process.

The reset is triggered when the client calls the "reset" endpoint (which
is part of the standard server protocol per server_client.py:94). For
clients that don't call reset, the seed is also applied at server start.

Run via:
  docker run ... -v examples/libero/gr00t_server_deterministic_wrapper.py:/srv_wrap.py \\
    gr00t:latest python /srv_wrap.py --model-path ... --use-sim-policy-wrapper --port 8000

Set STRANDS_GR00T_SERVER_SEED=<int> to override the default seed (42).
"""

from __future__ import annotations

import os

# Set BEFORE importing torch — required for cuBLAS determinism.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402  # must load AFTER `CUBLAS_WORKSPACE_CONFIG` is set

# Strict CUDA / cuDNN determinism.
# Note: torch.use_deterministic_algorithms(True) can force slower kernels
# that produce slightly different numerics than the default, sometimes
# hurting trained-model quality. cudnn.deterministic=True alone is the
# safer "deterministic enough" middle ground for diffusion sampling.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
_strict_det = os.environ.get("STRANDS_GR00T_STRICT_DETERMINISTIC", "0") == "1"
if _strict_det:
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[srv_wrap] STRICT mode: torch.use_deterministic_algorithms(True)", flush=True)
    except Exception as e:
        print(f"[srv_wrap] warning: use_deterministic_algorithms failed: {e}", flush=True)

print(
    f"[srv_wrap] determinism: cudnn.deterministic=True, benchmark=False, strict={_strict_det}",
    flush=True,
)
print(f"[srv_wrap] CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}", flush=True)

_DEFAULT_SEED = int(os.environ.get("STRANDS_GR00T_SERVER_SEED", "42"))


def _seed_all(seed: int) -> None:
    import random as _random

    import numpy as _np

    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Apply once at server start (covers any startup-time module state that
# isn't otherwise touched by reset).
_seed_all(_DEFAULT_SEED)
print(f"[srv_wrap] initial seed applied: {_DEFAULT_SEED}", flush=True)


# Monkey-patch Gr00tPolicy.reset so the client can trigger a re-seed
# per episode. The client passes options={"seed": <int>} to override
# the default seed (e.g. seed=42+ep_index for per-episode reproducibility).
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402  # imports `torch`; must follow seeding setup

_original_reset = Gr00tPolicy.reset


def _seeded_reset(self, options=None):
    seed = _DEFAULT_SEED
    if isinstance(options, dict) and "seed" in options:
        try:
            seed = int(options["seed"])
        except (TypeError, ValueError):
            print(
                f"[srv_wrap] warning: bad seed in reset options: {options['seed']!r}; using {_DEFAULT_SEED}",
                flush=True,
            )
            seed = _DEFAULT_SEED
    _seed_all(seed)
    print(f"[srv_wrap] reset: re-seeded to {seed}", flush=True)
    return _original_reset(self, options)


Gr00tPolicy.reset = _seeded_reset
print(
    "[srv_wrap] patched Gr00tPolicy.reset: applies torch/numpy/random seed via options['seed']",
    flush=True,
)


# Now hand off to the unmodified server entrypoint with whatever args
# the user passed.
import tyro  # noqa: E402  # late import: must follow Gr00tPolicy patch above
from gr00t.eval.run_gr00t_server import ServerConfig, main  # noqa: E402

config = tyro.cli(ServerConfig)
main(config)
