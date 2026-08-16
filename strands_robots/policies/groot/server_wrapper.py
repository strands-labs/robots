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
  ``reset`` calls that carry no seed (default: 42). Must be an integer in
  ``[0, 2**32 - 1]`` - the range every RNG below can be seeded with. A value
  outside it is refused at startup, because a server that could not apply the
  seed it was configured with cannot deliver the determinism it was started
  for.
- ``STRANDS_GR00T_STRICT_DETERMINISTIC=1``: additionally enable
  ``torch.use_deterministic_algorithms(True, warn_only=True)``. Strict mode
  can force slower kernels whose numerics differ slightly from the default,
  sometimes hurting trained-model quality; ``cudnn.deterministic=True``
  alone is the safer "deterministic enough" middle ground for diffusion
  sampling.
"""

from __future__ import annotations

import os

#: Largest seed this wrapper can apply to every RNG it reseeds.
#:
#: The three appliers do not share a domain. Python ``random.seed`` and
#: ``torch.manual_seed`` take far wider values - including negatives, which
#: ``manual_seed`` reduces modulo ``2**64`` - while NumPy's legacy global
#: seeder (``numpy.random.seed``) refuses anything negative, non-integral or
#: above ``2**32 - 1``. The narrowest applier therefore decides what a seed can
#: be, and it has to be checked *before the first* applier runs: the three
#: mutate process-wide state in sequence, so a value only NumPy refuses left
#: Python ``random`` reseeded with NumPy and torch untouched - an episode drawing
#: half its randomness from a fresh stream and half from the previous episode's.
#:
#: This restates the ceiling ``strands_robots.simulation.base.MAX_EVAL_SEED``
#: carries for the same destination rather than importing it: this module is
#: mounted into the GR00T container, where ``strands-robots`` is not installed,
#: so it can depend on the standard library only. The two are pinned in
#: agreement, so the duplication cannot drift into a disagreement about which
#: seeds a rollout may use.
_MAX_SEED = 2**32 - 1


def _seed_error(value: object, source: str) -> str | None:
    """Return why *value* cannot seed every RNG this wrapper reseeds.

    Args:
        value: Candidate seed, exactly as it arrived from *source*.
        source: Where the value came from, named in the message so an operator
            can tell a misconfigured environment variable from a per-episode
            ``reset`` option.

    Returns:
        The reason the value is unusable, or ``None`` when it can be applied to
        all of Python ``random``, NumPy and torch.
    """
    if isinstance(value, bool):
        # bool is an int subclass, so a bare range test reads True as a seed of
        # 1 - a seed the caller never named, applied under a success report.
        return f"{source}: seed must be an integer, not a bool (got {value!r})"
    if not isinstance(value, int):
        # No coercion: int("42") and int(3.7) would accept a value NumPy itself
        # refuses, the second one by silently truncating it to a different seed.
        return f"{source}: seed must be an integer in [0, {_MAX_SEED}], got {value!r}"
    if not 0 <= value <= _MAX_SEED:
        return f"{source}: seed must be in [0, {_MAX_SEED}], got {value!r}"
    return None


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

    configured_seed = os.environ.get("STRANDS_GR00T_SERVER_SEED", "42")
    try:
        default_seed = int(configured_seed)
    except ValueError:
        raise ValueError(
            f"STRANDS_GR00T_SERVER_SEED: seed must be an integer in [0, {_MAX_SEED}], got {configured_seed!r}"
        ) from None

    def _seed_all(seed: int, source: str) -> None:
        """Reseed Python, NumPy and torch - all of them, or none of them.

        The single point at which an unusable seed is refused, so the
        all-or-nothing property belongs to the applier rather than to each
        caller remembering to check first.

        Args:
            seed: The seed to apply to every RNG.
            source: Where the seed came from, named in the refusal.

        Raises:
            ValueError: If *seed* is outside the domain every applier shares.
                Checked, and both modules imported, before the first applier
                runs, so a refused seed and a missing NumPy both leave every
                RNG exactly as it was rather than part-way reseeded.
        """
        if error := _seed_error(seed, source):
            raise ValueError(error)

        import random as _random

        import numpy as _np

        _random.seed(seed)
        _np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Apply once at server start (covers any startup-time module state that
    # isn't otherwise touched by reset).
    _seed_all(default_seed, "STRANDS_GR00T_SERVER_SEED")
    print(f"[srv_wrap] initial seed applied: {default_seed}", flush=True)

    # Monkey-patch Gr00tPolicy.reset so the client can trigger a re-seed
    # per episode. The client passes options={"seed": <int>} to override
    # the default seed (e.g. seed=42+ep_index for per-episode reproducibility).
    from gr00t.policy.gr00t_policy import Gr00tPolicy  # imports `torch`; must follow seeding setup

    original_reset = Gr00tPolicy.reset

    def _seeded_reset(self, options=None):
        seed = default_seed
        if isinstance(options, dict) and "seed" in options:
            requested = options["seed"]
            if error := _seed_error(requested, "reset options['seed']"):
                # The configured default is already known-good, so the episode
                # still gets a complete reseed - just not the requested one,
                # and the reason says which value was dropped and why.
                print(f"[srv_wrap] warning: {error}; using {default_seed}", flush=True)
            else:
                seed = requested
        _seed_all(seed, "reset options['seed']")
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
