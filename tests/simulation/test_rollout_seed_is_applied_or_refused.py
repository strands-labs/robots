"""A rollout seed is either applied or refused with a reason.

``seed`` is the reproducibility contract of a policy rollout: it reseeds the
client RNGs, and each per-episode seed is forwarded to ``policy.reset`` so a
service-mode policy can reseed the process its sampler runs in. Two evals at the
same seed are supposed to replay identically.

It reached exactly one statement in :meth:`PolicyRunner.evaluate` - the
``_evaluate_with_spec`` delegation - so the plain ``success_fn`` loop below it
never read the value. Every :meth:`SimEngine.eval_policy` call lands in that loop,
because that facade exposes no ``spec`` parameter, so the whole surface accepted a
seed and discarded it: two evals at ``seed=7`` drew different actions,
``policy.reset(seed=...)`` was never called, and both runs reported
``status="success"``. Its sibling :meth:`SimEngine.run_policy` seeded correctly
one layer away, while ``eval_policy``'s own docstring advertises that the two
entry points behave the same way.

Honoring the seed is only half the contract, because an unusable value cannot be
applied at all - the seed ends at ``numpy.random.seed`` / ``default_rng``, which
accept only non-negative integers. That half had already gone three separate ways,
none of them naming the parameter:

* ``run_policy`` raised NumPy's own ``TypeError: Cannot cast scalar from
  dtype('float64') to dtype('int64')`` (and ``ValueError: Seed must be between 0
  and 2**32 - 1``) straight out of a method documented to return a structured
  ``{"status": ...}`` envelope, and bound as an agent-tool action.
* ``start_policy`` reported ``"started"`` and failed on its worker thread, where
  the raise was swallowed - the documented reason its sibling horizon knobs are
  validated synchronously, before ``executor.submit``.
* ``True`` was accepted everywhere as a silent seed of ``1``.

The domain those values need already existed:
:func:`~strands_robots.simulation.base.randomization_seed_error`, whose docstring
states the failure ("A float or string seed raises there ... so it is rejected at
the call that supplied it") and which ``randomize`` / ``set_obs_noise`` have used
on both backends since their inputs were hardened. It refused every value above
while the rollout surfaces that share its destination accepted them.

These tests pin both halves: the seed is applied on the path that dropped it, and
the one shared domain answers for it at every surface that accepts one.
"""

from __future__ import annotations

import ast
import inspect
import math
import random
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation.base import MAX_EVAL_SEED, SimEngine, randomization_seed_error
from strands_robots.simulation.policy_runner import PolicyRunner, set_eval_seed

pytest.importorskip("mujoco")

from strands_robots import Simulation  # noqa: E402  - after the mujoco probe
from strands_robots.policies import Policy  # noqa: E402

# Values no RNG can be seeded from. ``2.7`` and ``3.0`` are both refused: NumPy
# rejects a float seed whatever its fractional part, so an integral float is not
# a usable spelling here even though the whole-number domains honor one.
UNUSABLE_SEEDS: list[Any] = [
    -1,
    -100,
    2.7,
    3.0,
    True,
    False,
    math.nan,
    math.inf,
    "42",
    [1],
    {"a": 1},
]

# Integers no *rollout* seed can use, though ``randomize`` / ``set_obs_noise``
# can: those reach only ``default_rng``, which takes an integer of any width,
# while a rollout seed is also applied to the legacy NumPy global RNG, which
# refuses anything above ``MAX_EVAL_SEED``. ``1_754_000_000_000`` is a
# millisecond-epoch timestamp - the common "just use the clock" seed idiom.
SEEDS_ABOVE_THE_APPLIERS_BOUND: list[int] = [
    MAX_EVAL_SEED + 1,
    MAX_EVAL_SEED + 2,
    1_754_000_000_000,
]

# ``None`` is the documented "draw fresh entropy" spelling, not an error.
# ``MAX_EVAL_SEED`` is the accepted side of that boundary, so both sides of it
# are pinned rather than only the refused one.
USABLE_SEEDS: list[Any] = [None, 0, 7, 2**31, MAX_EVAL_SEED]

_ARM_XML = """<mujoco model="arm">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <geom type="box" size="0.05 0.05 0.1"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" limited="true" damping="2"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="30" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""


class _Jitter(Policy):
    """Draws each action from the global RNG - what a seed exists to pin.

    A deterministic policy cannot tell a seeded rollout from an unseeded one, so
    the reproducibility half of the contract is unobservable without a policy
    whose output depends on the RNG state the seed is supposed to fix.
    """

    def __init__(self) -> None:
        self.keys: list[str] = []
        self.reset_seeds: list[Any] = []
        self.drawn: list[float] = []

    @property
    def provider_name(self) -> str:
        return "jitter"

    def set_robot_state_keys(self, keys: list[str]) -> None:
        self.keys = list(keys)

    def reset(self, seed: int | None = None) -> None:
        self.reset_seeds.append(seed)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, float]]:
        value = random.random()
        self.drawn.append(value)
        return [{key: value - 0.5 for key in self.keys}]


@pytest.fixture
def arm_xml(tmp_path: Path) -> Path:
    path = tmp_path / "arm.xml"
    path.write_text(_ARM_XML, encoding="utf-8")
    return path


def _sim_and_policy(arm_xml: Path) -> tuple[Any, _Jitter]:
    sim = Simulation(backend="mujoco", tool_name="seed_test", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm", urdf_path=str(arm_xml))
    policy = _Jitter()
    policy.set_robot_state_keys(sim.robot_action_keys("arm"))
    return sim, policy


def _text(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result.get("content") or [] if isinstance(block, dict))


class TestTheSeedIsAppliedOnTheEvalPath:
    """The half that was silently dropped: eval_policy never read its seed."""

    def test_two_evals_at_the_same_seed_draw_the_same_actions(self, arm_xml: Path) -> None:
        runs = []
        for _ in range(2):
            sim, policy = _sim_and_policy(arm_xml)
            result = sim.eval_policy(
                robot_name="arm",
                policy_object=policy,
                n_episodes=2,
                max_steps=4,
                control_frequency=30.0,
                seed=7,
            )
            sim.cleanup()
            assert result["status"] == "success", _text(result)
            runs.append(list(policy.drawn))
        assert runs[0], "the policy must have been queried, or nothing is measured"
        assert runs[0] == runs[1], "a seeded eval must replay identically"

    def test_two_evals_at_different_seeds_draw_different_actions(self, arm_xml: Path) -> None:
        """Non-vacuity: the equality above is the seed's doing, not determinism."""
        drawn = []
        for seed in (7, 8):
            sim, policy = _sim_and_policy(arm_xml)
            sim.eval_policy(
                robot_name="arm",
                policy_object=policy,
                n_episodes=2,
                max_steps=4,
                control_frequency=30.0,
                seed=seed,
            )
            sim.cleanup()
            drawn.append(list(policy.drawn))
        assert drawn[0] != drawn[1]

    def test_each_episode_seed_is_forwarded_to_policy_reset(self, arm_xml: Path) -> None:
        """A service-mode policy samples in a process ``set_eval_seed`` cannot reach."""
        sim, policy = _sim_and_policy(arm_xml)
        sim.eval_policy(
            robot_name="arm",
            policy_object=policy,
            n_episodes=3,
            max_steps=3,
            control_frequency=30.0,
            seed=7,
        )
        sim.cleanup()
        assert len(policy.reset_seeds) == 3, policy.reset_seeds
        assert all(isinstance(s, int) and s >= 0 for s in policy.reset_seeds)
        assert len(set(policy.reset_seeds)) == 3, "each episode gets its own child seed"

    def test_episode_seeds_are_reproducible_across_runs(self, arm_xml: Path) -> None:
        seeds = []
        for _ in range(2):
            sim, policy = _sim_and_policy(arm_xml)
            sim.eval_policy(
                robot_name="arm",
                policy_object=policy,
                n_episodes=3,
                max_steps=3,
                control_frequency=30.0,
                seed=7,
            )
            sim.cleanup()
            seeds.append(list(policy.reset_seeds))
        assert seeds[0] == seeds[1]

    def test_an_unseeded_eval_does_not_touch_policy_reset_or_the_global_rng(self, arm_xml: Path) -> None:
        """``seed=None`` must not acquire a global RNG side effect it never had."""
        random.seed(1234)
        expected = random.getstate()
        random.seed(1234)
        sim, policy = _sim_and_policy(arm_xml)
        result = sim.eval_policy(
            robot_name="arm",
            policy_object=policy,
            n_episodes=2,
            max_steps=3,
            control_frequency=30.0,
            seed=None,
        )
        sim.cleanup()
        assert result["status"] == "success", _text(result)
        assert policy.reset_seeds == [], "no seed was supplied, so none is forwarded"
        # The rollout consumed draws from the un-reseeded stream, so the state
        # moved on rather than being reset to the start of a new one.
        assert random.getstate() != expected

    def test_the_seeded_eval_matches_its_sibling_run_policy_contract(self, arm_xml: Path) -> None:
        """``eval_policy``'s docstring promises the two entry points behave alike."""
        singles = []
        for _ in range(2):
            sim, policy = _sim_and_policy(arm_xml)
            sim.run_policy(robot_name="arm", policy_object=policy, n_steps=4, control_frequency=30.0, seed=7)
            sim.cleanup()
            singles.append(list(policy.drawn))
        assert singles[0] == singles[1]
        assert singles[0], "the policy must have been queried"


class TestEverySurfaceRefusesAnUnusableSeed:
    """The domain half, on each facade, through the structured envelope."""

    @pytest.mark.parametrize("seed", UNUSABLE_SEEDS, ids=repr)
    def test_run_policy_refuses(self, arm_xml: Path, seed: Any) -> None:
        sim, policy = _sim_and_policy(arm_xml)
        result = sim.run_policy(robot_name="arm", policy_object=policy, n_steps=4, control_frequency=30.0, seed=seed)
        sim.cleanup()
        assert result["status"] == "error"
        assert "seed must be a non-negative integer or None" in _text(result)
        assert "run_policy" in _text(result)
        assert policy.drawn == [], "a refused seed must not run the rollout"

    @pytest.mark.parametrize("seed", UNUSABLE_SEEDS, ids=repr)
    def test_eval_policy_refuses(self, arm_xml: Path, seed: Any) -> None:
        sim, policy = _sim_and_policy(arm_xml)
        result = sim.eval_policy(
            robot_name="arm",
            policy_object=policy,
            n_episodes=2,
            max_steps=3,
            control_frequency=30.0,
            seed=seed,
        )
        sim.cleanup()
        assert result["status"] == "error"
        assert "seed must be a non-negative integer or None" in _text(result)
        assert "eval_policy" in _text(result)
        assert policy.drawn == []

    @pytest.mark.parametrize("seed", UNUSABLE_SEEDS, ids=repr)
    def test_start_policy_refuses_before_submitting_to_the_executor(self, arm_xml: Path, seed: Any) -> None:
        sim, policy = _sim_and_policy(arm_xml)
        result = sim.start_policy(robot_name="arm", policy_object=policy, n_steps=4, control_frequency=30.0, seed=seed)
        sim.cleanup()
        assert result["status"] == "error"
        assert "seed must be a non-negative integer or None" in _text(result)
        # The false "started" is the whole point of a synchronous check.
        assert "started" not in _text(result).lower()

    @pytest.mark.parametrize("seed", USABLE_SEEDS, ids=repr)
    def test_a_usable_seed_is_accepted_everywhere(self, arm_xml: Path, seed: Any) -> None:
        """Over-reach control: the guard must not refuse a seed that works."""
        sim, policy = _sim_and_policy(arm_xml)
        result = sim.run_policy(robot_name="arm", policy_object=policy, n_steps=3, control_frequency=30.0, seed=seed)
        sim.cleanup()
        assert result["status"] == "success", _text(result)

        sim, policy = _sim_and_policy(arm_xml)
        result = sim.eval_policy(
            robot_name="arm",
            policy_object=policy,
            n_episodes=1,
            max_steps=3,
            control_frequency=30.0,
            seed=seed,
        )
        sim.cleanup()
        assert result["status"] == "success", _text(result)


class TestNothingRaisesPastTheEnvelope:
    """The refusal replaces NumPy's own message, which named neither the
    parameter nor the method."""

    @pytest.mark.parametrize("seed", UNUSABLE_SEEDS, ids=repr)
    def test_no_bare_numpy_error_escapes(self, arm_xml: Path, seed: Any) -> None:
        sim, policy = _sim_and_policy(arm_xml)
        try:
            result = sim.run_policy(
                robot_name="arm", policy_object=policy, n_steps=4, control_frequency=30.0, seed=seed
            )
        finally:
            sim.cleanup()
        text = _text(result)
        assert "Cannot cast scalar" not in text
        assert "2**32" not in text
        assert "supported seed types" not in text


class TestThePolicyRunnerLayerEnforcesItToo:
    """``PolicyRunner`` is drivable directly; a direct caller has no envelope."""

    @pytest.mark.parametrize("seed", UNUSABLE_SEEDS, ids=repr)
    def test_run_raises_a_named_value_error(self, arm_xml: Path, seed: Any) -> None:
        sim, policy = _sim_and_policy(arm_xml)
        try:
            with pytest.raises(ValueError, match="seed must be a non-negative integer or None"):
                PolicyRunner(sim).run("arm", policy, n_steps=3, control_frequency=30.0, seed=seed)
        finally:
            sim.cleanup()

    @pytest.mark.parametrize("seed", UNUSABLE_SEEDS, ids=repr)
    def test_evaluate_raises_a_named_value_error(self, arm_xml: Path, seed: Any) -> None:
        sim, policy = _sim_and_policy(arm_xml)
        try:
            with pytest.raises(ValueError, match="seed must be a non-negative integer or None"):
                PolicyRunner(sim).evaluate("arm", policy, n_episodes=1, max_steps=3, control_frequency=30.0, seed=seed)
        finally:
            sim.cleanup()


class TestTheDomainIsShared:
    """One rule, so a seed refused for ``randomize`` cannot be accepted for the
    rollout whose reproducibility it is supposed to pin."""

    @pytest.mark.parametrize("seed", UNUSABLE_SEEDS + USABLE_SEEDS + SEEDS_ABOVE_THE_APPLIERS_BOUND, ids=repr)
    def test_every_seed_randomize_refuses_is_refused_by_a_rollout_too(self, arm_xml: Path, seed: Any) -> None:
        """An implication, not an equivalence - the rollout domain is narrower.

        The two families share the non-negative-integer rule, so nothing
        ``randomize`` refuses may be accepted for the rollout whose
        reproducibility it pins. The converse does not hold: the rollout applier
        adds the legacy NumPy global RNG, so it refuses a high integer
        ``randomize`` can honor. That direction is pinned below, with its reason,
        rather than smoothed away.
        """
        randomize_refuses = randomization_seed_error(seed, "randomize") is not None
        sim, policy = _sim_and_policy(arm_xml)
        result = sim.run_policy(robot_name="arm", policy_object=policy, n_steps=3, control_frequency=30.0, seed=seed)
        sim.cleanup()
        rollout_refuses = result["status"] == "error"
        if randomize_refuses:
            assert rollout_refuses, (
                f"randomize refuses seed={seed!r} but the rollout accepted it - the shared rule is not shared"
            )

    @pytest.mark.parametrize("seed", SEEDS_ABOVE_THE_APPLIERS_BOUND, ids=repr)
    def test_the_narrower_rollout_domain_is_the_appliers_and_is_documented(self, seed: int) -> None:
        """The one divergence: a width ``default_rng`` honors and the legacy
        global RNG does not.

        Refusing these for ``randomize`` too would remove a capability that
        works, so the bound is carried per destination instead of narrowing the
        shared rule.
        """
        assert randomization_seed_error(seed, "randomize") is None
        assert randomization_seed_error(seed, "run_policy", max_seed=MAX_EVAL_SEED) is not None
        envelope = SimEngine._validate_seed(seed, "run_policy")
        assert envelope is not None
        assert f"[0, {MAX_EVAL_SEED}]" in _text(envelope)

    def test_the_envelope_binding_carries_the_shared_reason_verbatim(self) -> None:
        for seed in UNUSABLE_SEEDS:
            reason = randomization_seed_error(seed, "run_policy")
            envelope = SimEngine._validate_seed(seed, "run_policy")
            assert envelope is not None
            assert _text(envelope) == reason


class TestTheCeilingIsTheOneItsApplierCanHonor:
    """A seed above ``MAX_EVAL_SEED`` is refused, not applied and then raised.

    ``set_eval_seed`` reseeds the legacy NumPy global RNG as well as
    ``default_rng``, and ``numpy.random.seed`` refuses anything above
    ``2**32 - 1``. An accepted domain wider than that reintroduces both failure
    modes this file exists to close, on exactly the range it does not cover:
    ``eval_policy`` raised NumPy's bare ``ValueError`` out of a method
    documented to return an envelope, and ``start_policy`` reported "started"
    and died on its worker thread. A millisecond-epoch timestamp - a common
    seed idiom - lands in that range.
    """

    @pytest.mark.parametrize("seed", SEEDS_ABOVE_THE_APPLIERS_BOUND, ids=repr)
    def test_run_policy_refuses(self, arm_xml: Path, seed: int) -> None:
        sim, policy = _sim_and_policy(arm_xml)
        result = sim.run_policy(robot_name="arm", policy_object=policy, n_steps=4, control_frequency=30.0, seed=seed)
        sim.cleanup()
        assert result["status"] == "error"
        assert f"seed must be an integer in [0, {MAX_EVAL_SEED}]" in _text(result)
        assert policy.drawn == [], "a refused seed must not run the rollout"

    @pytest.mark.parametrize("seed", SEEDS_ABOVE_THE_APPLIERS_BOUND, ids=repr)
    def test_eval_policy_refuses_instead_of_raising_numpys_message(self, arm_xml: Path, seed: int) -> None:
        sim, policy = _sim_and_policy(arm_xml)
        try:
            result = sim.eval_policy(
                robot_name="arm",
                policy_object=policy,
                n_episodes=1,
                max_steps=3,
                control_frequency=30.0,
                seed=seed,
            )
        finally:
            sim.cleanup()
        assert result["status"] == "error"
        text = _text(result)
        assert f"seed must be an integer in [0, {MAX_EVAL_SEED}]" in text
        assert "eval_policy" in text
        # NumPy's own wording named neither the parameter nor the method.
        assert "Seed must be between" not in text
        assert policy.drawn == []

    @pytest.mark.parametrize("seed", SEEDS_ABOVE_THE_APPLIERS_BOUND, ids=repr)
    def test_start_policy_refuses_without_reporting_started(self, arm_xml: Path, seed: int) -> None:
        """The false "started" is why this check is synchronous."""
        sim, policy = _sim_and_policy(arm_xml)
        result = sim.start_policy(robot_name="arm", policy_object=policy, n_steps=4, control_frequency=30.0, seed=seed)
        sim.cleanup()
        assert result["status"] == "error"
        assert f"seed must be an integer in [0, {MAX_EVAL_SEED}]" in _text(result)
        assert "started" not in _text(result).lower()

    @pytest.mark.parametrize("seed", SEEDS_ABOVE_THE_APPLIERS_BOUND, ids=repr)
    def test_evaluate_benchmark_refuses(self, arm_xml: Path, seed: int) -> None:
        sim, policy = _sim_and_policy(arm_xml)
        result = sim.evaluate_benchmark(benchmark_name="whatever", robot_name="arm", policy_object=policy, seed=seed)
        sim.cleanup()
        assert result["status"] == "error"
        assert f"seed must be an integer in [0, {MAX_EVAL_SEED}]" in _text(result)

    @pytest.mark.parametrize("seed", SEEDS_ABOVE_THE_APPLIERS_BOUND, ids=repr)
    def test_the_runner_layer_raises_a_named_error(self, arm_xml: Path, seed: int) -> None:
        sim, policy = _sim_and_policy(arm_xml)
        try:
            with pytest.raises(ValueError, match=r"seed must be an integer in \[0, 4294967295\]"):
                PolicyRunner(sim).run("arm", policy, n_steps=3, control_frequency=30.0, seed=seed)
            with pytest.raises(ValueError, match=r"seed must be an integer in \[0, 4294967295\]"):
                PolicyRunner(sim).evaluate("arm", policy, n_episodes=1, max_steps=3, control_frequency=30.0, seed=seed)
        finally:
            sim.cleanup()

    @pytest.mark.parametrize("seed", SEEDS_ABOVE_THE_APPLIERS_BOUND, ids=repr)
    def test_the_applier_itself_names_the_bound(self, seed: int) -> None:
        """``set_eval_seed`` is public API and documented for direct callers, so
        the rule is enforced where it is owned - not only at the facades."""
        with pytest.raises(ValueError, match=r"set_eval_seed: seed must be an integer in \[0, 4294967295\]"):
            set_eval_seed(seed)

    def test_the_accepted_side_of_the_boundary_still_runs(self, arm_xml: Path) -> None:
        """Over-reach control: the largest seed the applier honors is usable."""
        set_eval_seed(MAX_EVAL_SEED)
        sim, policy = _sim_and_policy(arm_xml)
        result = sim.run_policy(
            robot_name="arm", policy_object=policy, n_steps=3, control_frequency=30.0, seed=MAX_EVAL_SEED
        )
        sim.cleanup()
        assert result["status"] == "success", _text(result)
        assert policy.drawn, "the policy must have been queried"

    def test_the_bound_is_the_appliers_own_and_not_a_chosen_number(self) -> None:
        """Premise: ``MAX_EVAL_SEED`` is exactly where NumPy's legacy global RNG
        stops, so the domain tracks the applier rather than a literal."""
        np = pytest.importorskip("numpy")
        np.random.seed(MAX_EVAL_SEED)
        with pytest.raises(ValueError):
            np.random.seed(MAX_EVAL_SEED + 1)
        # ``default_rng`` - the randomization families' only destination - is
        # wider, which is why the bound is per-caller and not in the shared rule.
        np.random.default_rng(MAX_EVAL_SEED + 1)


class TestNoRolloutSurfaceCanShipWithoutTheGuard:
    """Structural: the facades and the runner layer all reach the one domain.

    AST-based, so a surface that stops calling the guard - or a new rollout
    surface that never starts - fails here rather than silently re-acquiring its
    own accepted domain, the way ``eval_policy`` and ``start_policy`` each did.
    """

    _GUARD_CALLS = {"_validate_seed", "randomization_seed_error"}
    # A surface may also reach the domain by delegating to the one guarded
    # funnel: ``SimEngine.start_policy`` is literally ``return
    # self.run_policy(...)``. A backend that overrides ``start_policy`` to
    # submit to an executor instead does NOT delegate, and is pinned separately
    # below to guard directly - that is exactly the surface where the seed used
    # to fail on a worker thread after a false "started".
    #
    # ``_drive_rollout`` is the same funnel one hop further in: it is literally
    # ``return super().run_policy(...)``, and exists because the blocking entry
    # and the executor worker must reach that funnel without both of them
    # claiming the robot (see ``MuJoCoSimEngine._announce_rollout``). A surface
    # reaching it reaches every guard the funnel applies.
    _DELEGATES_TO = {"run_policy", "_drive_rollout"}

    @staticmethod
    def _seed_taking_methods(module_path: Path) -> dict[str, set[str]]:
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        found: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                continue
            args = [a.arg for a in node.args.args + node.args.kwonlyargs]
            if "seed" not in args:
                continue
            called: set[str] = set()
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                func = call.func
                if isinstance(func, ast.Name):
                    called.add(func.id)
                elif isinstance(func, ast.Attribute):
                    called.add(func.attr)
            found[node.name] = called
        return found

    @staticmethod
    def _rollout_modules() -> list[Path]:
        base_dir = Path(inspect.getfile(SimEngine)).parent
        paths = [base_dir / "base.py", base_dir / "policy_runner.py", base_dir / "mujoco" / "simulation.py"]
        for path in paths:
            assert path.is_file(), path
        return paths

    def test_every_seed_taking_rollout_surface_reaches_the_shared_domain(self) -> None:
        # ``reset`` is a Policy-protocol hook, not a rollout entry point that
        # owns the caller's seed; ``set_eval_seed`` IS the applier the domain
        # protects, and ``generate_heightfield`` seeds a terrain generator whose
        # own entry point validates it.
        exempt = {"reset", "set_eval_seed", "generate_heightfield"}
        checked = 0
        for module_path in self._rollout_modules():
            for name, called in self._seed_taking_methods(module_path).items():
                if name in exempt:
                    continue
                assert (self._GUARD_CALLS | self._DELEGATES_TO) & called, (
                    f"{module_path.name}:{name}() accepts a seed and reaches no shared "
                    f"seed domain (called: {sorted(called)})"
                )
                checked += 1
        assert checked >= 5, f"expected the known rollout surfaces, found {checked}"

    def test_a_backend_that_does_not_delegate_guards_the_seed_itself(self) -> None:
        """The delegation clause must not excuse an override that submits work.

        ``MuJoCoSimEngine.start_policy`` runs the rollout on its executor rather
        than calling ``run_policy``, so a refusal reached from inside the future
        arrives after the caller has already been told "started". It has to hold
        the domain itself.
        """
        base_dir = Path(inspect.getfile(SimEngine)).parent
        mujoco_sim = base_dir / "mujoco" / "simulation.py"
        methods = self._seed_taking_methods(mujoco_sim)
        assert "start_policy" in methods, sorted(methods)
        called = methods["start_policy"]
        assert "run_policy" not in called, "this override does not delegate"
        assert self._GUARD_CALLS & called, sorted(called)

    def test_the_known_surfaces_are_the_ones_scanned(self) -> None:
        """Non-vacuity: a scan root resolving elsewhere must fail, not pass."""
        names: set[str] = set()
        for module_path in self._rollout_modules():
            names |= set(self._seed_taking_methods(module_path))
        assert {"run_policy", "eval_policy", "start_policy", "evaluate_benchmark", "run", "evaluate"} <= names, names

    def test_the_scanner_detects_a_surface_that_drops_the_guard(self, tmp_path: Path) -> None:
        """A guard that silently matched nothing would look like a clean sweep."""
        planted = tmp_path / "planted.py"
        planted.write_text(
            "def run_policy(self, robot_name, seed=None):\n    return {'status': 'success'}\n",
            encoding="utf-8",
        )
        found = self._seed_taking_methods(planted)
        assert "run_policy" in found
        assert not (self._GUARD_CALLS & found["run_policy"])
