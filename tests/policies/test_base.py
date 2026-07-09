"""Tests for ``strands_robots.policies.base.Policy`` ABC contract.

Covers the ``get_actions_sync`` event-loop dispatch paths: the 'no loop'
fast path and the 'already-in-event-loop' ThreadPoolExecutor fallback.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

import pytest

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


def test_get_actions_sync_inside_event_loop_uses_threadpool():
    """When called from within a running event loop, the sync wrapper must
    off-load to a thread pool instead of raising 'already in a loop'."""
    p = _IdentityPolicy()

    async def inner():
        # Calling the sync wrapper here forces the thread-pool branch
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


def test_well_known_kwargs_are_accepted_by_contract():
    """Non-VLA providers receive goals via ``**kwargs`` (target_pose,
    target_joints, world_update). The Policy contract requires get_actions
    to ignore unknown kwargs rather than raising, so callers can pass
    shared keys across providers without coupling to a backend."""
    p = MockPolicy()
    p.set_robot_state_keys(["j0", "j1"])
    obs = {"observation.state": [0.0, 0.0]}

    # All three well-known kwargs together must round-trip cleanly through
    # the sync wrapper -- this is the smoke test that pins the documented
    # API surface for non-VLA providers.
    actions = p.get_actions_sync(
        obs,
        instruction="",
        target_pose=[0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
        target_joints={"j0": 0.1, "j1": -0.2},
        world_update=None,
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
        with pytest.raises(ValueError, match="must be positive"):
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

    def test_set_observed_delay_coerces_to_int(self):
        p = _IdentityPolicy()
        p.set_rtc_observed_delay(True)  # bool is an int subclass
        assert p.rtc_observed_delay_steps == 1
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
