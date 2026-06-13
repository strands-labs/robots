"""#331: shared client-side RNG reseed helper + provider parity.

Pins that ``reseed_client_rngs`` reseeds Python ``random`` + NumPy (and torch
when present) deterministically, and that both Gr00tPolicy and Cosmos3Policy
route their reset reseed through it so they behave identically for #187
reproducibility.
"""

from __future__ import annotations

import random

import numpy as np

from strands_robots.policies._rng import reseed_client_rngs


def test_reseed_is_deterministic_across_python_and_numpy():
    reseed_client_rngs(1234)
    py_a = [random.random() for _ in range(3)]
    np_a = np.random.rand(3).tolist()

    reseed_client_rngs(1234)
    py_b = [random.random() for _ in range(3)]
    np_b = np.random.rand(3).tolist()

    assert py_a == py_b, "Python random must be reproducible after reseed"
    assert np_a == np_b, "NumPy RNG must be reproducible after reseed"


def test_reseed_none_is_noop():
    # Establish a known state, draw once, then call with None and confirm the
    # stream is NOT reset (the next draw differs from a fresh-seed draw).
    reseed_client_rngs(7)
    first = random.random()
    reseed_client_rngs(None)  # must not reset
    second = random.random()
    reseed_client_rngs(7)
    fresh = random.random()
    assert first == fresh, "reseed(7) must reproduce the first draw"
    assert second != first, "reseed(None) must be a no-op, not a reset"


def test_distinct_seeds_diverge():
    reseed_client_rngs(1)
    a = np.random.rand(4).tolist()
    reseed_client_rngs(2)
    b = np.random.rand(4).tolist()
    assert a != b


def test_both_providers_route_reset_through_shared_helper():
    """Source-level parity pin: both reset() methods call reseed_client_rngs
    so they cannot drift apart again (the #331 root cause)."""
    import inspect

    from strands_robots.policies.cosmos3 import policy as cosmos_mod
    from strands_robots.policies.groot import policy as groot_mod

    cosmos_src = inspect.getsource(cosmos_mod.Cosmos3Policy.reset)
    groot_src = inspect.getsource(groot_mod.Gr00tPolicy.reset)
    assert "reseed_client_rngs" in cosmos_src, "Cosmos3Policy.reset must use the shared reseed helper (#331)"
    assert "reseed_client_rngs" in groot_src, "Gr00tPolicy.reset must use the shared reseed helper (#331)"
    # The old global-only mutation must be gone from cosmos3.
    assert "np.random.seed(seed)" not in cosmos_src, (
        "Cosmos3Policy.reset must not reseed only the global NumPy RNG anymore (#331)"
    )
