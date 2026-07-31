"""Determinism wrapper for ``gr00t.eval.run_gr00t_server`` (runs INSIDE the container).

Docker-mountable wrapper around the GR00T N1.7 server entrypoint that
enforces:

- ``cudnn.deterministic = True``
- ``cudnn.benchmark = False``
- ``torch.use_deterministic_algorithms(True, warn_only=True)`` (opt-in via
  ``STRANDS_GR00T_STRICT_DETERMINISTIC=1``)
- ``CUBLAS_WORKSPACE_CONFIG=":4096:8"`` (required for cuBLAS determinism)
- A monkey-patch on the upstream ``Gr00tPolicy.reset`` (a no-op by default)
  that reseeds torch / numpy / random at the start of each episode. The
  reset is triggered when the client calls the ``reset`` endpoint (part of
  the standard server protocol per ``server_client.py:94``); the
  client-side :meth:`strands_robots.policies.groot.policy.Gr00tPolicy.reset`
  forwards ``options={"seed": N}`` there every episode. For clients that
  never call reset, the seed is also applied once at server start.

This file executes inside the GR00T container, where ``strands-robots`` is
NOT installed - it must stay fully self-contained (stdlib at module level;
``torch`` / ``gr00t`` / ``tyro`` imported only when :func:`main` runs, so
importing this module on the host is side-effect free).

The supported way to run it is the ``gr00t_inference`` tool's
``deterministic=True`` flag, which bind-mounts this file (read-only) to
``/srv_wrap.py`` and swaps the N1.7 server entrypoint to
``python /srv_wrap.py <same args>``. The manual escape hatch is::

    WRAP=$(python -c "import strands_robots.policies.groot.server_wrapper as m; print(m.__file__)")
    docker run ... -v "$WRAP":/srv_wrap.py:ro \\
      gr00t:latest python /srv_wrap.py --model-path ... --use-sim-policy-wrapper --port 8000

Environment variables (read inside the container):

- ``STRANDS_GR00T_SERVER_SEED``: default seed applied at server start and on
  ``reset`` calls that carry no seed (default: 42).
- ``STRANDS_GR00T_STRICT_DETERMINISTIC=1``: additionally enable
  ``torch.use_deterministic_algorithms(True, warn_only=True)``. Strict mode
  can force slower kernels whose numerics differ slightly from the default,
  sometimes hurting trained-model quality; ``cudnn.deterministic=True``
  alone is the safer "deterministic enough" middle ground for diffusion
  sampling.
"""

from __future__ import annotations

import os


def main() -> None:
    """Configure determinism, patch ``Gr00tPolicy.reset``, and run the server.

    Everything happens here (not at module import) so the module can be
    imported host-side without pulling in ``torch``/``gr00t``/``tyro``,
    which only exist inside the GR00T container.
    """
    # Set BEFORE importing torch - required for cuBLAS determinism.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch  # must load AFTER `CUBLAS_WORKSPACE_CONFIG` is set

    # Strict CUDA / cuDNN determinism.
    # Note: torch.use_deterministic_algorithms(True) can force slower kernels
    # that produce slightly different numerics than the default, sometimes
    # hurting trained-model quality. cudnn.deterministic=True alone is the
    # safer "deterministic enough" middle ground for diffusion sampling.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    strict_det = os.environ.get("STRANDS_GR00T_STRICT_DETERMINISTIC", "0") == "1"
    if strict_det:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            print("[srv_wrap] STRICT mode: torch.use_deterministic_algorithms(True)", flush=True)
        except Exception as e:  # noqa: BLE001 - degrade to non-strict, never kill the server
            print(f"[srv_wrap] warning: use_deterministic_algorithms failed: {e}", flush=True)

    print(
        f"[srv_wrap] determinism: cudnn.deterministic=True, benchmark=False, strict={strict_det}",
        flush=True,
    )
    print(f"[srv_wrap] CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}", flush=True)

    default_seed = int(os.environ.get("STRANDS_GR00T_SERVER_SEED", "42"))

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
    _seed_all(default_seed)
    print(f"[srv_wrap] initial seed applied: {default_seed}", flush=True)

    # Monkey-patch Gr00tPolicy.reset so the client can trigger a re-seed
    # per episode. The client passes options={"seed": <int>} to override
    # the default seed (e.g. seed=42+ep_index for per-episode reproducibility).
    from gr00t.policy.gr00t_policy import Gr00tPolicy  # imports `torch`; must follow seeding setup

    original_reset = Gr00tPolicy.reset

    def _seeded_reset(self, options=None):
        seed = default_seed
        if isinstance(options, dict) and "seed" in options:
            try:
                seed = int(options["seed"])
            except (TypeError, ValueError):
                print(
                    f"[srv_wrap] warning: bad seed in reset options: {options['seed']!r}; using {default_seed}",
                    flush=True,
                )
                seed = default_seed
        _seed_all(seed)
        print(f"[srv_wrap] reset: re-seeded to {seed}", flush=True)
        return original_reset(self, options)

    Gr00tPolicy.reset = _seeded_reset
    print(
        "[srv_wrap] patched Gr00tPolicy.reset: applies torch/numpy/random seed via options['seed']",
        flush=True,
    )

    # Now hand off to the unmodified server entrypoint with whatever args
    # the user passed.
    import tyro  # late import: must follow Gr00tPolicy patch above
    from gr00t.eval.run_gr00t_server import ServerConfig
    from gr00t.eval.run_gr00t_server import main as server_main

    config = tyro.cli(ServerConfig)
    server_main(config)


if __name__ == "__main__":
    main()
