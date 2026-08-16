"""The server-side reseed accepts exactly the seeds it can apply to every RNG.

:mod:`strands_robots.policies.groot.server_wrapper` is the container-side half
of the #187 per-episode reproducibility contract: it patches
``Gr00tPolicy.reset`` so a seed the client forwards in ``options`` reseeds the
server's RNGs before the episode's first diffusion sample. Its reseed applies
Python ``random``, then NumPy's legacy global RNG, then torch - three appliers
with three different accepted domains, run in sequence against process-wide
state.

Only the second of them bounds the value. Before the guard under test, every
seed NumPy refuses - anything negative, non-integral, or above ``2**32 - 1`` -
left the server *half* reseeded: Python ``random`` moved, NumPy and torch
untouched, and the upstream ``reset`` never reached at all because the NumPy
refusal raised out of the patched method first. The client swallows a failed
reset at ``INFO`` and reports "continuing without per-episode server-side
reseed", so the one thing nothing observed was that the server's RNGs had in
fact been moved, inconsistently. The episode then drew part of its randomness
from a fresh stream and part from the previous episode's.

What is pinned here:

* a seed no applier can honor is refused *before the first one runs*, so no RNG
  moves - at startup for ``STRANDS_GR00T_SERVER_SEED`` and per episode for
  ``reset``;
* a ``reset`` carrying an unusable seed still completes a whole reseed, on the
  already-validated configured default, and still reaches the upstream
  ``reset`` - a refused *request* is not a refused *episode*;
* ``bool`` is refused rather than read as a seed of 1, and a numeric string or
  float is refused rather than coerced into a different seed than the one
  asked for;
* the accepted domain is the one
  :func:`~strands_robots.simulation.base.randomization_seed_error` already
  enforces for a rollout seed, value for value. The wrapper restates that
  domain instead of importing it because it runs where ``strands-robots`` is
  not installed, so the agreement is asserted rather than assumed;
* the determinism configuration the wrapper exists to apply - cuDNN flags,
  ``CUBLAS_WORKSPACE_CONFIG``, the optional strict mode, the ``reset`` patch,
  and the hand-off to the unmodified server entrypoint.
"""

from __future__ import annotations

import math
import random
import sys
import types
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.groot import server_wrapper
from strands_robots.policies.groot.server_wrapper import _MAX_SEED, _seed_error
from strands_robots.simulation.base import MAX_EVAL_SEED, randomization_seed_error

#: Seeds this path cannot apply, each refused by NumPy's legacy global RNG -
#: the narrowest of the three appliers. Python ``random`` accepts most of them,
#: which is exactly why they used to leave the server half reseeded.
UNUSABLE_SEEDS: list[Any] = [
    -1,
    -5,
    2.5,
    3.0,
    True,
    False,
    "42",
    [7],
    math.nan,
    math.inf,
    _MAX_SEED + 1,
    2**64,
]

#: Seeds every applier honors, including both endpoints - ``0`` is a seed, not
#: the absence of one.
USABLE_SEEDS: list[int] = [0, 1, 7, 12345, _MAX_SEED]

#: State the RNGs are put into before each measurement. Kept out of the lists
#: above so "the state changed" is never true by coincidence.
BASELINE_SEED = 999


def _stub_module(name: str) -> Any:
    """Return a module stand-in whose attributes the caller assigns freely.

    Typed ``Any`` so a checked test body can populate it: the wrapper reaches
    these through ``sys.modules``, which is what lets a host without the GR00T
    image drive :func:`~strands_robots.policies.groot.server_wrapper.main`.
    """
    return types.ModuleType(name)


class _StubPolicy:
    """Stand-in for the upstream ``Gr00tPolicy`` whose ``reset`` is patched."""

    upstream_calls: list[Any] = []

    def reset(self, options: Any = None) -> str:
        type(self).upstream_calls.append(options)
        return "upstream-reset"


class _Container:
    """The container-side modules :func:`main` imports, captured for assertions."""

    def __init__(self) -> None:
        self.torch = _stub_module("torch")
        self.seeded: list[int] = []
        self.cuda_seeded: list[int] = []
        self.strict_calls: list[tuple[Any, ...]] = []
        self.cuda_available = False
        self.strict_raises: Exception | None = None

        self.torch.manual_seed = self.seeded.append
        self.torch.cuda = types.SimpleNamespace(
            is_available=lambda: self.cuda_available,
            manual_seed_all=self.cuda_seeded.append,
        )
        self.torch.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(deterministic=False, benchmark=True))

        def _strict(*args: Any, **kwargs: Any) -> None:
            self.strict_calls.append(args)
            if self.strict_raises is not None:
                raise self.strict_raises

        self.torch.use_deterministic_algorithms = _strict

        self.policy_cls: Any = type("Gr00tPolicy", (_StubPolicy,), {"upstream_calls": []})
        self.server_config_cls: Any = type("ServerConfig", (), {})
        self.server_configs: list[Any] = []
        self.cli_targets: list[Any] = []

    @property
    def cudnn(self) -> Any:
        return self.torch.backends.cudnn

    @property
    def patched_reset(self) -> Any:
        """The ``reset`` implementation :func:`main` installed on the policy."""
        return self.policy_cls.reset


@pytest.fixture
def container(monkeypatch: pytest.MonkeyPatch) -> _Container:
    """Install stubs for the modules that only exist inside the GR00T image.

    The wrapper imports ``torch`` / ``gr00t`` / ``tyro`` inside :func:`main`
    precisely so it stays importable on a host without them, which is what
    makes it drivable here.
    """
    stub = _Container()
    monkeypatch.setitem(sys.modules, "torch", stub.torch)

    for name in ("gr00t", "gr00t.policy", "gr00t.eval"):
        monkeypatch.setitem(sys.modules, name, _stub_module(name))

    policy_mod = _stub_module("gr00t.policy.gr00t_policy")
    policy_mod.Gr00tPolicy = stub.policy_cls
    monkeypatch.setitem(sys.modules, "gr00t.policy.gr00t_policy", policy_mod)

    server_mod = _stub_module("gr00t.eval.run_gr00t_server")
    server_mod.ServerConfig = stub.server_config_cls
    server_mod.main = stub.server_configs.append
    monkeypatch.setitem(sys.modules, "gr00t.eval.run_gr00t_server", server_mod)

    tyro = _stub_module("tyro")

    def _cli(target: Any) -> str:
        stub.cli_targets.append(target)
        return "parsed-config"

    tyro.cli = _cli
    monkeypatch.setitem(sys.modules, "tyro", tyro)

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("STRANDS_GR00T_STRICT_DETERMINISTIC", raising=False)
    monkeypatch.setenv("STRANDS_GR00T_SERVER_SEED", str(BASELINE_SEED))
    return stub


def _rng_state() -> tuple[Any, Any]:
    """Snapshot both RNGs the wrapper mutates before torch is reached."""
    return random.getstate(), np.random.get_state()


def _baseline() -> tuple[Any, Any]:
    """Seed both RNGs to a known value and return that state."""
    random.seed(BASELINE_SEED)
    np.random.seed(BASELINE_SEED)
    return _rng_state()


def _states_equal(left: tuple[Any, Any], right: tuple[Any, Any]) -> bool:
    py_ok = left[0] == right[0]
    np_ok = all(np.array_equal(a, b) for a, b in zip(left[1], right[1], strict=True))
    return py_ok and np_ok


class TestTheStartupSeedIsAppliedOrRefusedWhole:
    """``STRANDS_GR00T_SERVER_SEED`` is applied to every RNG, or to none."""

    def test_the_configured_seed_reaches_every_rng_and_the_server_starts(self, container: _Container) -> None:
        container.cuda_available = True
        server_wrapper.main()

        assert container.seeded == [BASELINE_SEED], "torch must be seeded with the configured seed"
        assert container.cuda_seeded == [BASELINE_SEED], "CUDA generators must be seeded when CUDA reports available"
        assert container.cudnn.deterministic is True, "cuDNN must be pinned deterministic"
        assert container.cudnn.benchmark is False, "the cuDNN autotuner must be off for determinism"
        assert server_wrapper.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8", (
            "cuBLAS determinism requires the workspace config, set before torch loads"
        )
        assert container.cli_targets == [container.server_config_cls], (
            "the wrapper must parse the unmodified server's own config"
        )
        assert container.server_configs == ["parsed-config"], "the unmodified server entrypoint must run"

    def test_a_preset_workspace_config_is_left_alone(self, container: _Container, monkeypatch) -> None:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        server_wrapper.main()
        assert server_wrapper.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8", (
            "an operator-supplied workspace config must win over the default"
        )

    def test_cuda_generators_are_left_alone_when_cuda_is_absent(self, container: _Container) -> None:
        container.cuda_available = False
        server_wrapper.main()
        assert container.seeded == [BASELINE_SEED]
        assert container.cuda_seeded == [], "there are no CUDA generators to seed on a CPU-only host"

    @pytest.mark.parametrize("value", ["4294967296", "-1", "abc", "3.5", "", "true"])
    def test_an_unusable_configured_seed_moves_no_rng(self, container: _Container, monkeypatch, value: str) -> None:
        """The refusal must land before the first applier, and name the variable.

        Pre-fix, ``"4294967296"`` and ``"-1"`` parsed as ints, reseeded Python
        ``random``, and only then raised out of NumPy - so the server failed to
        start *and* left the interpreter's RNG moved, with a message naming
        NumPy's range rather than the variable that carried the value.
        """
        monkeypatch.setenv("STRANDS_GR00T_SERVER_SEED", value)
        before = _baseline()

        with pytest.raises(ValueError, match="STRANDS_GR00T_SERVER_SEED"):
            server_wrapper.main()

        assert _states_equal(before, _rng_state()), "a refused startup seed must leave every RNG untouched"
        assert container.seeded == [], "torch must not be seeded when the configured seed was refused"

    def test_the_refusal_states_the_domain_it_enforces(self, container: _Container, monkeypatch) -> None:
        monkeypatch.setenv("STRANDS_GR00T_SERVER_SEED", "-7")
        with pytest.raises(ValueError) as excinfo:
            server_wrapper.main()
        assert str(_MAX_SEED) in str(excinfo.value), "the reason must state the ceiling, not just that it was exceeded"
        assert "-7" in str(excinfo.value), "the reason must quote the value it refused"


class TestStrictDeterminismIsOptIn:
    """The slower strict mode is applied only when asked for, and never fatal."""

    def test_strict_mode_is_off_by_default(self, container: _Container) -> None:
        server_wrapper.main()
        assert container.strict_calls == [], "strict algorithms must not be forced without the opt-in"

    def test_the_opt_in_enables_deterministic_algorithms(self, container: _Container, monkeypatch) -> None:
        monkeypatch.setenv("STRANDS_GR00T_STRICT_DETERMINISTIC", "1")
        server_wrapper.main()
        assert container.strict_calls == [(True,)], "the opt-in must reach torch.use_deterministic_algorithms"

    def test_a_strict_mode_failure_does_not_stop_the_server(self, container: _Container, monkeypatch) -> None:
        monkeypatch.setenv("STRANDS_GR00T_STRICT_DETERMINISTIC", "1")
        container.strict_raises = RuntimeError("no deterministic kernel")
        server_wrapper.main()
        assert container.server_configs == ["parsed-config"], (
            "strict mode is a tightening, so losing it must degrade rather than kill the server"
        )


class TestAResetSeedIsAppliedWholeOrNotAtAll:
    """The patched ``reset`` never leaves the server part-way reseeded."""

    @pytest.mark.parametrize("seed", USABLE_SEEDS)
    def test_an_accepted_seed_reaches_every_rng(self, container: _Container, seed: int) -> None:
        container.cuda_available = True
        server_wrapper.main()
        container.seeded.clear()
        container.cuda_seeded.clear()

        expected = (random.Random(seed).getstate(), None)
        result = container.patched_reset(container.policy_cls(), {"seed": seed})

        assert random.getstate() == expected[0], "Python random must be reseeded with the requested seed"
        assert container.seeded == [seed], "torch must be reseeded with the requested seed"
        assert container.cuda_seeded == [seed], "CUDA generators must be reseeded with the requested seed"
        assert container.policy_cls.upstream_calls == [{"seed": seed}], "the upstream reset must still run"
        assert result == "upstream-reset", "the patch must return what the upstream reset returned"

    @pytest.mark.parametrize("seed", UNUSABLE_SEEDS)
    def test_an_unusable_seed_still_leaves_a_whole_reseed(self, container: _Container, seed: Any) -> None:
        """A refused request must not raise, and must not half-apply.

        Pre-fix this split three ways, none of them a whole reseed: a value only
        NumPy refuses raised out of ``reset`` with Python ``random`` already
        moved, ``True`` was read as a seed of 1, and ``"42"`` was coerced to 42
        - two of the three reporting success for a seed the caller never asked
        for.
        """
        server_wrapper.main()
        container.seeded.clear()
        container.policy_cls.upstream_calls.clear()
        _baseline()

        result = container.patched_reset(container.policy_cls(), {"seed": seed})

        assert random.getstate() == random.Random(BASELINE_SEED).getstate(), (
            "the episode must be reseeded on the validated configured seed, not left part-way"
        )
        assert container.seeded == [BASELINE_SEED], "torch must be reseeded too, so all three streams agree"
        assert container.policy_cls.upstream_calls == [{"seed": seed}], (
            "a refused seed is not a refused episode - the upstream reset must still run"
        )
        assert result == "upstream-reset"

    def test_a_reset_without_a_seed_uses_the_configured_default(self, container: _Container) -> None:
        server_wrapper.main()
        container.seeded.clear()
        container.patched_reset(container.policy_cls(), None)
        assert container.seeded == [BASELINE_SEED], "a seedless reset must still reseed, on the configured default"

    def test_the_warning_names_the_dropped_value_and_the_fallback(
        self, container: _Container, capsys: pytest.CaptureFixture[str]
    ) -> None:
        server_wrapper.main()
        capsys.readouterr()
        container.patched_reset(container.policy_cls(), {"seed": -1})
        out = capsys.readouterr().out
        assert "-1" in out, "the operator must be told which value was dropped"
        assert str(BASELINE_SEED) in out, "and which seed the episode actually ran with"


class TestTheDomainMatchesTheSharedRolloutDomain:
    """The restated domain must not drift from the one it restates."""

    def test_the_ceiling_is_the_shared_rollout_ceiling(self) -> None:
        assert _MAX_SEED == MAX_EVAL_SEED, (
            "the wrapper reseeds the same legacy NumPy global RNG a rollout seed reaches, so it can "
            "honor exactly the same seeds; a divergence is a defect in whichever moved"
        )

    @pytest.mark.parametrize("seed", [*UNUSABLE_SEEDS, *USABLE_SEEDS, None])
    def test_every_value_gets_the_same_verdict_from_both(self, seed: Any) -> None:
        shared = randomization_seed_error(seed, "rollout", max_seed=MAX_EVAL_SEED, allow_none=False)
        assert (_seed_error(seed, "reset") is None) == (shared is None), (
            f"the wrapper and the shared rollout domain disagree about {seed!r}"
        )

    def test_the_wrapper_stays_importable_without_the_container(self) -> None:
        """The domain has to be reachable on a host, which is why it is restated.

        The wrapper is mounted into an image where ``strands-robots`` is absent,
        so importing the shared helper is not an option - this asserts the
        replacement is a module-level constant rather than something only the
        container can evaluate.
        """
        assert isinstance(_MAX_SEED, int)
        assert _seed_error(1, "probe") is None
        assert _seed_error(-1, "probe") is not None
