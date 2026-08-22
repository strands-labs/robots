"""Tests for ``strands_robots.policies.base.Policy`` ABC contract.

Covers both ``get_actions_sync`` dispatch paths -- the 'no loop' fast path and
the 'already-in-a-running-loop' offload -- and pins that the offload is resolved
by ``strands_robots._async_utils``, the one owner of that rule, rather than by a
private per-call executor.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
from typing import Any

import pytest

from strands_robots._async_utils import _resolve_coroutine
from strands_robots.policies import base as policy_base
from strands_robots.policies.base import ChunkedPolicy, Policy, resolve_chunk_length
from strands_robots.policies.mock import MockPolicy


class _IdentityPolicy(Policy):
    """Minimal concrete Policy for testing Policy ABC's sync wrapper."""

    def __init__(self) -> None:
        self._keys = ["j0"]

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [{"j0": 0.1}, {"j0": 0.2}]

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._keys = list(robot_state_keys)

    @property
    def provider_name(self) -> str:
        return "identity"


def test_get_actions_sync_outside_event_loop_uses_asyncio_run():
    p = _IdentityPolicy()
    actions = p.get_actions_sync({"observation.state": [0.0]}, instruction="hi")
    assert actions == [{"j0": 0.1}, {"j0": 0.2}]


def test_get_actions_sync_inside_event_loop_returns_actions_instead_of_raising():
    """From inside a running loop the wrapper must offload, not raise 'already in a loop'."""
    p = _IdentityPolicy()

    async def inner():
        # Calling the sync wrapper here forces the offload branch
        return p.get_actions_sync({"observation.state": [0.0]}, instruction="hi")

    actions = asyncio.run(inner())
    assert actions == [{"j0": 0.1}, {"j0": 0.2}]


def test_provider_name_and_state_keys():
    p = _IdentityPolicy()
    assert p.provider_name == "identity"
    p.set_robot_state_keys(["a", "b", "c"])
    assert p._keys == ["a", "b", "c"]


def test_requires_images_default_is_true():
    """The base ABC defaults requires_images=True; subclasses opt out."""
    p = _IdentityPolicy()
    assert p.requires_images is True


def test_reset_default_is_noop():
    """Default reset() returns None and must be safe to call without a seed."""
    p = _IdentityPolicy()
    assert p.reset() is None
    assert p.reset(seed=42) is None


def _documented_well_known_keys() -> list[str]:
    """The well-known goal keys ``Policy.get_actions`` documents, from its docstring."""
    doc = Policy.get_actions.__doc__ or ""
    start = doc.find("are **well-known**")
    end = doc.find("Providers MUST ignore unknown")
    if start == -1 or end == -1 or end <= start:
        return []
    return re.findall(r"^\s*-\s+``([A-Za-z_][A-Za-z0-9_]*)\s*:", doc[start:end], re.MULTILINE)


# One representative value per well-known goal key, on that key's documented
# domain. Keyed by name so the round-trip below covers whatever
# ``Policy.get_actions`` currently documents rather than a copy of the list that
# can fall behind it -- this smoke test asserted only the first three for as long
# as ``target_velocity`` was missing from the ABC, so it agreed with the omission
# instead of catching it. A key added to the contract with no sample here fails
# the completeness assertion rather than being silently skipped.
_WELL_KNOWN_KWARG_SAMPLES: dict[str, Any] = {
    "target_pose": [0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
    "target_joints": {"j0": 0.1, "j1": -0.2},
    "target_velocity": [0.5, 0.0, 0.0],
    "world_update": None,
}


def test_well_known_kwargs_are_accepted_by_contract():
    """Every well-known goal key the ABC documents must round-trip through a provider.

    Non-VLA providers receive goals via the well-known ``**kwargs`` keys, and the
    Policy contract requires ``get_actions`` to ignore unknown kwargs rather than
    raising, so callers can pass shared keys across providers without coupling to
    a backend.

    The key set is read from the contract itself. The exact vocabulary and its
    parity with shipped provider code are pinned in
    ``test_well_known_goal_kwargs_have_one_definition.py``; this is the
    behavioural half -- that passing all of them at once is actually accepted.
    """
    documented = set(_documented_well_known_keys())
    assert documented, "parsed no well-known goal keys from the Policy.get_actions contract"
    unsampled = sorted(documented - set(_WELL_KNOWN_KWARG_SAMPLES))
    assert not unsampled, (
        f"Policy.get_actions documents well-known goal keys with no sample value "
        f"here: {unsampled}. Add one on that key's documented domain so this "
        "round-trip covers the whole contract."
    )

    p = MockPolicy()
    p.set_robot_state_keys(["j0", "j1"])
    obs = {"observation.state": [0.0, 0.0]}

    # All well-known kwargs together must round-trip cleanly through the sync
    # wrapper -- this is the smoke test that pins the documented API surface for
    # non-VLA providers.
    actions = p.get_actions_sync(
        obs,
        instruction="",
        **{key: _WELL_KNOWN_KWARG_SAMPLES[key] for key in sorted(documented)},
    )
    assert isinstance(actions, list) and actions, "Policy must return a non-empty action list"
    assert all(isinstance(a, dict) for a in actions)


def test_non_vla_providers_can_skip_camera_rendering():
    """``requires_images=False`` is the opt-out for joint-state-only
    providers (MockPolicy, planners, MPC, scripted)."""
    assert MockPolicy().requires_images is False


# Providers that opt into the non-VLA path and inherit the documented
# "ignore unknown ``**kwargs`` rather than raising" contract from the
# Policy ABC. As CuroboPolicy / MoveIt2Policy land via #305 / #306 they
# extend this list rather than re-asserting the same contract locally.
_NON_VLA_PROVIDER_FACTORIES: list[Any] = [
    pytest.param(lambda: MockPolicy(), id="mock"),
    # pytest.param(lambda: CuroboPolicy(...), id="curobo"),     # PR #306
    # pytest.param(lambda: MoveIt2Policy(...), id="moveit2"),   # PR #305
]


@pytest.mark.parametrize("provider_factory", _NON_VLA_PROVIDER_FACTORIES)
def test_unknown_kwargs_are_silently_ignored(provider_factory):
    """Regression pin for the cross-provider contract documented in the
    Policy ABC module docstring: ``get_actions(**kwargs)`` MUST silently
    ignore kwargs it does not recognise rather than raising ``TypeError``.

    A made-up kwarg no provider knows about (``some_future_kwarg``) must
    round-trip cleanly through ``get_actions_sync`` -- this fails on any
    future provider whose ``get_actions`` signature drops ``**kwargs``
    entirely (e.g. ``def get_actions(self, obs, instruction, target_pose=None)``),
    which would otherwise be silently masked by the sync wrapper's own
    ``**kwargs`` passthrough.

    Centralising here means #305 / #306 inherit the contract automatically
    instead of each PR re-asserting it locally."""
    p = provider_factory()
    p.set_robot_state_keys(["j0", "j1"])
    obs = {"observation.state": [0.0, 0.0]}

    actions = p.get_actions_sync(
        obs,
        instruction="",
        target_pose=[0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
        some_future_kwarg="opaque",
    )
    assert isinstance(actions, list) and actions, (
        "Policy must return a non-empty action list even when passed an "
        "unknown kwarg; the contract is to ignore, not raise."
    )


def test_get_actions_docstring_pins_value_convention():
    """The Policy.get_actions ``Returns:`` docstring MUST pin the per-tick
    action value convention: python ``float`` / ``list[float]``, never a raw
    ``np.ndarray``. This is the contract C2 makes explicit so providers and
    consumers agree on the value type regardless of compute backend.

    Fails on the pre-C2 docstring, which only described the dict *shape* and
    left the value type unspecified -- the ambiguity that let providers leak
    ``np.ndarray`` into action dicts."""
    doc = inspect.getdoc(Policy.get_actions) or ""
    assert "float" in doc and "list[float]" in doc, (
        "get_actions docstring must state values are python float or list[float]"
    )
    assert "np.ndarray" in doc, "get_actions docstring must explicitly forbid returning raw np.ndarray"


def test_policy_class_docstring_references_value_convention():
    """The Policy class docstring MUST reference the action value convention
    so implementers see it before reading the method, satisfying C2's
    class-level note acceptance criterion."""
    doc = inspect.getdoc(Policy) or ""
    assert "value convention" in doc.lower() and "np.ndarray" in doc, (
        "Policy class docstring must reference the per-tick action value "
        "convention and that values are not raw np.ndarray"
    )


def test_mock_policy_action_values_are_json_native_floats():
    """MockPolicy is the canonical reference for the value convention: every
    action value must be a python ``float`` (not ``np.ndarray`` / numpy
    scalar), so the action list is JSON-serializable as-is. Pins the
    behavioural half of the C2 contract against the documented reference."""
    p = MockPolicy()
    p.set_robot_state_keys(["j0", "j1", "j2"])
    actions = p.get_actions_sync({"observation.state": [0.0, 0.0, 0.0]}, instruction="")
    assert actions, "MockPolicy must return a non-empty action list"
    for tick in actions:
        for key, value in tick.items():
            assert type(value) is float, (
                f"action value for {key!r} must be a python float per the "
                f"documented convention, got {type(value).__name__}"
            )
    # JSON round-trip is the canonical proof of native-value compliance.
    json.dumps(actions)


class TestControlFrequency:
    """``Policy.set_control_frequency`` / ``control_frequency`` contract.

    The control rate is how a latency-sensitive provider converts wall-clock
    inference latency into a count of consumed action steps (RTC). The runtime
    sets it before the loop; until then it is ``None`` so providers can detect
    "not told yet" rather than silently assuming a rate.
    """

    def test_default_control_frequency_is_none(self):
        assert _IdentityPolicy().control_frequency is None

    def test_set_control_frequency_sets_attribute(self):
        p = _IdentityPolicy()
        p.set_control_frequency(90.0)
        assert p.control_frequency == 90.0

    def test_set_control_frequency_coerces_to_float(self):
        p = _IdentityPolicy()
        p.set_control_frequency(50)
        assert isinstance(p.control_frequency, float)
        assert p.control_frequency == 50.0

    @pytest.mark.parametrize("bad", [0, -1, -30.0])
    def test_set_control_frequency_rejects_non_positive(self, bad):
        p = _IdentityPolicy()
        with pytest.raises(ValueError, match="control_frequency must be > 0"):
            p.set_control_frequency(bad)

    def test_control_frequency_is_per_instance(self):
        a, b = _IdentityPolicy(), _IdentityPolicy()
        a.set_control_frequency(120.0)
        assert a.control_frequency == 120.0
        assert b.control_frequency is None


class TestRTCObservedDelay:
    """``Policy.set_rtc_observed_delay`` / ``rtc_observed_delay_steps`` contract.

    The runtime supplies the EXACT number of control steps that elapse during
    inference so latency-sensitive providers slice the chunk seam by a known
    integer rather than a non-reproducible wall-clock estimate.
    """

    def test_default_observed_delay_is_none(self):
        assert _IdentityPolicy().rtc_observed_delay_steps is None

    def test_set_observed_delay_sets_attribute(self):
        p = _IdentityPolicy()
        p.set_rtc_observed_delay(3)
        assert p.rtc_observed_delay_steps == 3

    def test_set_observed_delay_zero_is_honoured_not_treated_as_none(self):
        # 0 (synchronous loop: world paused during inference) is a real value,
        # distinct from None (no count supplied -> wall-clock fallback).
        p = _IdentityPolicy()
        p.set_rtc_observed_delay(0)
        assert p.rtc_observed_delay_steps == 0

    def test_set_observed_delay_none_clears_override(self):
        p = _IdentityPolicy()
        p.set_rtc_observed_delay(5)
        p.set_rtc_observed_delay(None)
        assert p.rtc_observed_delay_steps is None

    def test_set_observed_delay_refuses_a_bool_rather_than_counting_it_as_one(self):
        """A ``bool`` is not a step count, and coercing it fabricated one.

        This previously asserted the coercion (``True`` -> ``1``) on the grounds
        that ``bool`` is an ``int`` subclass. That is true of the type but not of
        the quantity: the count is the offset at which the provider slices the
        next action chunk, so a silent ``1`` is a seam the caller never asked
        for, not a harmless default. Being an ``int`` subclass is precisely why
        the old bare ``steps < 0`` test could not see it.
        """
        p = _IdentityPolicy()
        with pytest.raises(ValueError, match="rtc_observed_delay_steps must be a non-negative integer"):
            p.set_rtc_observed_delay(True)
        assert p.rtc_observed_delay_steps is None

    def test_set_observed_delay_stores_a_true_int(self):
        p = _IdentityPolicy()
        p.set_rtc_observed_delay(4)
        assert p.rtc_observed_delay_steps == 4
        assert isinstance(p.rtc_observed_delay_steps, int)

    def test_set_observed_delay_rejects_negative(self):
        p = _IdentityPolicy()
        with pytest.raises(ValueError, match="rtc_observed_delay_steps"):
            p.set_rtc_observed_delay(-1)

    def test_observed_delay_is_per_instance(self):
        a, b = _IdentityPolicy(), _IdentityPolicy()
        a.set_rtc_observed_delay(7)
        assert a.rtc_observed_delay_steps == 7
        assert b.rtc_observed_delay_steps is None


class _ChunkedPolicy(_IdentityPolicy):
    """Open-loop chunked policy: emits an N-step chunk, no cross-chunk state."""

    actions_per_step = 8
    supports_rtc = False


class _RTCPolicy(_IdentityPolicy):
    """RTC policy: owns its re-query interval and blends chunk seams itself."""

    supports_rtc = True

    @property
    def execution_horizon(self) -> int:
        return 5


class _BadHorizonPolicy(_IdentityPolicy):
    """A policy that declares a non-numeric ``actions_per_step``."""

    actions_per_step = "not-a-number"  # type: ignore[assignment]


class TestExecutionHorizon:
    """``execution_horizon`` is the single source of truth for the re-query interval."""

    def test_defaults_to_one_when_actions_per_step_undeclared(self):
        # A single-step policy (MockPolicy, classical planners) declares nothing.
        assert _IdentityPolicy().execution_horizon == 1

    def test_derives_from_actions_per_step_for_chunked_policy(self):
        assert _ChunkedPolicy().execution_horizon == 8

    def test_non_numeric_actions_per_step_falls_back_to_one(self):
        # A misconfigured chunk length must degrade to single-step, not crash
        # the re-query loop that reads this property every tick.
        assert _BadHorizonPolicy().execution_horizon == 1

    def test_non_positive_actions_per_step_is_clamped_to_one(self):
        class _Zero(_IdentityPolicy):
            actions_per_step = 0

        assert _Zero().execution_horizon == 1


class TestIsChunkEmitting:
    """``is_chunk_emitting`` is derived from the re-query interval, not provider name."""

    def test_single_step_policy_is_not_chunk_emitting(self):
        assert _IdentityPolicy().is_chunk_emitting() is False

    def test_open_loop_chunked_policy_is_chunk_emitting(self):
        assert _ChunkedPolicy().is_chunk_emitting() is True

    def test_rtc_policy_reporting_horizon_gt_one_is_chunk_emitting(self):
        assert _RTCPolicy().is_chunk_emitting() is True


class TestChunkedPolicyProtocol:
    """The runtime-checkable protocol lets a consumer branch on chunk metadata."""

    def test_policy_exposing_chunk_metadata_matches_protocol(self):
        assert isinstance(_ChunkedPolicy(), ChunkedPolicy)

    def test_policy_without_chunk_metadata_does_not_match(self):
        # _IdentityPolicy declares neither actions_per_step nor supports_rtc.
        assert not isinstance(_IdentityPolicy(), ChunkedPolicy)


class TestResolveChunkLength:
    """The single re-query rule every chunk consumer must apply identically."""

    def test_single_step_policy_honours_requested_action_horizon(self):
        # execution_horizon == 1, so the caller's action_horizon wins.
        assert resolve_chunk_length(_IdentityPolicy(), action_horizon=4) == 4

    def test_open_loop_chunk_is_not_truncated_below_trained_length(self):
        # A smaller action_horizon must NOT drop the trained chunk tail.
        assert resolve_chunk_length(_ChunkedPolicy(), action_horizon=4) == 8

    def test_open_loop_chunk_extends_to_requested_horizon(self):
        # A larger action_horizon is honoured for a non-RTC policy.
        assert resolve_chunk_length(_ChunkedPolicy(), action_horizon=16) == 16

    def test_rtc_policy_interval_is_not_overridden_by_action_horizon(self):
        # RTC owns its interval; stretching it would empty the blended tail.
        assert resolve_chunk_length(_RTCPolicy(), action_horizon=20) == 5

    def test_action_horizon_is_clamped_to_at_least_one(self):
        assert resolve_chunk_length(_IdentityPolicy(), action_horizon=0) == 1
        assert resolve_chunk_length(_IdentityPolicy(), action_horizon=-3) == 1

    def test_duck_typed_object_falls_back_to_actions_per_step(self):
        # A non-Policy object without execution_horizon is sized by its raw
        # actions_per_step attribute.
        class _DuckChunk:
            actions_per_step = 6

        assert resolve_chunk_length(_DuckChunk(), action_horizon=2) == 6

    def test_duck_typed_object_without_chunk_metadata_is_single_action(self):
        class _Bare:
            pass

        assert resolve_chunk_length(_Bare(), action_horizon=3) == 3

    def test_none_duck_typed_horizon_degrades_to_single_action(self):
        class _DuckNone:
            actions_per_step = None

        assert resolve_chunk_length(_DuckNone(), action_horizon=4) == 4

    def test_non_numeric_duck_typed_horizon_degrades_to_single_action(self):
        # A garbage chunk length must not crash the consumer's sizing call.
        class _DuckGarbage:
            actions_per_step = "not-a-number"

        assert resolve_chunk_length(_DuckGarbage(), action_horizon=4) == 4

    def test_non_positive_duck_typed_horizon_is_clamped_to_one(self):
        class _DuckNegative:
            actions_per_step = -2

        assert resolve_chunk_length(_DuckNegative(), action_horizon=3) == 3


class TestPreflightDefault:
    """The default preflight hook is a cheap no-op that never rejects config."""

    def test_default_preflight_accepts_any_observation_keys(self):
        # A provider that does not override preflight must not block construction.
        assert _IdentityPolicy.preflight({"observation.state", "front"}) is None


class _RaisingPolicy(_IdentityPolicy):
    """A policy whose inference fails, to pin that the wrapper is transparent to it."""

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        raise RuntimeError("inference exploded")


class _EchoKwargsPolicy(_IdentityPolicy):
    """A policy that returns whatever kwargs reached it, to pin verbatim forwarding."""

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [{"obs": observation_dict, "instruction": instruction, "kwargs": kwargs}]


class TestSyncWrapperResolvesThroughTheSharedOwner:
    """``get_actions_sync`` delegates resolution instead of re-deriving it.

    ``strands_robots._async_utils`` owns "resolve a policy coroutine in a sync
    context", and its offload branch submits to ONE module-level worker that is
    reused across calls -- the comment on that executor says why. Every
    in-process rollout path resolves through it.

    The offload branch is the one a documented caller lands in: this wrapper is
    advertised as safe to call from an event loop or a notebook, and a notebook
    cell body always runs inside a running loop. Re-deriving the branch with a
    private ``ThreadPoolExecutor`` in a ``with`` block therefore starts and
    joins one OS thread per call at control rate. Because the block joins before
    returning, the live thread COUNT never grows, so nothing observing thread
    counts can see it -- which is why the pin below counts ``Thread.start``
    instead.
    """

    @staticmethod
    def _count_started_threads(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        """Count every OS thread started from here on."""
        counter = {"n": 0}
        original = threading.Thread.start

        def counting_start(self: threading.Thread) -> None:
            counter["n"] += 1
            original(self)

        monkeypatch.setattr(threading.Thread, "start", counting_start)
        return counter

    def test_a_call_inside_a_running_loop_starts_no_thread_of_its_own(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The offload reuses the shared worker, so N calls start 0 threads."""
        policy = _IdentityPolicy()
        calls = 24

        async def drive() -> int:
            # Warm the shared worker first, so what is measured afterwards is
            # per-call thread creation rather than the one-off worker start.
            policy.get_actions_sync({}, "")
            counter = self._count_started_threads(monkeypatch)
            for _ in range(calls):
                policy.get_actions_sync({}, "")
            return counter["n"]

        started = asyncio.run(drive())
        assert started == 0, (
            f"{calls} calls to get_actions_sync inside a running loop started {started} OS "
            f"threads; the shared resolver in strands_robots._async_utils reuses one worker, "
            f"so a warm wrapper must start none"
        )

    def test_the_wrapper_resolves_through_the_shared_resolver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolution is delegated to the owner, not re-implemented locally."""
        seen: list[object] = []
        # Read through getattr and install with raising=False so a tree whose
        # wrapper does not consult the owner fails on the observable (nothing
        # recorded) rather than on the name being absent.
        real = getattr(policy_base, "_resolve_coroutine", _resolve_coroutine)

        def spy(coro_or_result: Any) -> Any:
            seen.append(coro_or_result)
            return real(coro_or_result)

        monkeypatch.setattr(policy_base, "_resolve_coroutine", spy, raising=False)
        actions = _IdentityPolicy().get_actions_sync({}, "hi")

        assert actions == [{"j0": 0.1}, {"j0": 0.2}]
        assert len(seen) == 1, (
            "get_actions_sync did not resolve through strands_robots._async_utils._resolve_coroutine; "
            "a second implementation of the same rule is what lets the wrapper and the runner "
            "disagree about which branch a caller lands in"
        )

    def test_the_wrapper_builds_no_executor_of_its_own(self) -> None:
        """One owner for the offload, so there is one executor to reason about."""
        source = inspect.getsource(Policy.get_actions_sync)
        assert "ThreadPoolExecutor" not in source, (
            "get_actions_sync constructs its own executor; the offload belongs to "
            "strands_robots._async_utils, whose worker is created once and reused"
        )

    def test_both_branches_agree_with_the_resolver_the_runner_uses(self) -> None:
        """The wrapper and the in-process rollout path resolve to the same actions."""
        policy = _IdentityPolicy()
        expected = [{"j0": 0.1}, {"j0": 0.2}]

        via_wrapper_no_loop = policy.get_actions_sync({}, "")
        via_runner_no_loop = _resolve_coroutine(policy.get_actions({}, ""))
        assert via_wrapper_no_loop == expected
        assert via_runner_no_loop == expected

        async def inside_a_loop() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            return (
                policy.get_actions_sync({}, ""),
                _resolve_coroutine(policy.get_actions({}, "")),
            )

        via_wrapper, via_runner = asyncio.run(inside_a_loop())
        assert via_wrapper == expected
        assert via_runner == expected

    @pytest.mark.parametrize("inside_a_loop", [False, True], ids=["no-loop", "running-loop"])
    def test_an_inference_failure_reaches_the_caller_unchanged(self, inside_a_loop: bool) -> None:
        """The wrapper is transparent to the policy's own exception on both branches."""
        policy = _RaisingPolicy()

        def call() -> list[dict[str, Any]]:
            return policy.get_actions_sync({}, "")

        if not inside_a_loop:
            with pytest.raises(RuntimeError, match="inference exploded"):
                call()
            return

        async def inner() -> list[dict[str, Any]]:
            return call()

        with pytest.raises(RuntimeError, match="inference exploded"):
            asyncio.run(inner())

    @pytest.mark.parametrize("inside_a_loop", [False, True], ids=["no-loop", "running-loop"])
    def test_the_observation_instruction_and_kwargs_are_forwarded_verbatim(self, inside_a_loop: bool) -> None:
        """Delegating resolution must not touch what the policy is handed."""
        policy = _EchoKwargsPolicy()
        obs = {"observation.state": [0.1, 0.2]}
        goal = [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]

        def call() -> list[dict[str, Any]]:
            return policy.get_actions_sync(obs, "pick it up", target_pose=goal, world_update=None)

        if inside_a_loop:

            async def inner() -> list[dict[str, Any]]:
                return call()

            got = asyncio.run(inner())
        else:
            got = call()

        assert got == [
            {
                "obs": obs,
                "instruction": "pick it up",
                "kwargs": {"target_pose": goal, "world_update": None},
            }
        ]
