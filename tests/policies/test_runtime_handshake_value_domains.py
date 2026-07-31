"""A policy must refuse a runtime handshake value it cannot honor.

A runtime tells a policy two things before it drives it: the control rate of the
executing loop (:meth:`~strands_robots.policies.base.Policy.set_control_frequency`)
and how many control steps elapse while one inference is in flight
(:meth:`~strands_robots.policies.base.Policy.set_rtc_observed_delay`). Both are
stored verbatim and read much later, inside a provider's Real-Time Chunking
path, so a value the setter accepts but the provider cannot use is not
discovered where the caller can act on it.

Both setters used to guard with a bare comparison (``hz <= 0`` /
``steps < 0``), which is the ordering :func:`strands_robots.utils.positive_finite_number_error`
documents as the one that lets ``nan`` through - ``nan`` compares ``False`` to
everything - and which lets ``bool`` through because it is an ``int`` subclass.
The observable results were a stored ``nan``/``inf`` rate that raised out of the
``int()`` in the delay estimator on the *second* inference of a rollout, and a
``True`` that installed a silent 1 Hz clock (or a 1-step chunk seam).

These tests pin the domains, the agreement between the policy-side rate guard
and the runner-side guard for the same quantity, and the agreement between the
in-process API and the same call arriving over the wire.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import pytest

from strands_robots.inference import protocol
from strands_robots.inference.server import PolicyServer
from strands_robots.policies import MockPolicy
from strands_robots.policies.composite import CompositePolicy
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.simulation.base import SimEngine
from strands_robots.utils import non_negative_count_error

# Rates no loop can run at, and the reason each one used to survive.
UNUSABLE_RATES: list[Any] = [
    0,  # already refused before this change
    -1,
    -30.0,
    True,  # int subclass: a silent 1 Hz
    float("nan"),  # nan <= 0 is False
    float("inf"),  # inf <= 0 is False; the period collapses to 0
    float("-inf"),
    "50",  # leaked a TypeError from the comparison itself
    None,
    [50],
]
USABLE_RATES: list[Any] = [1, 30, 50, 30.0, 62.5, 240.0]

# Step counts that are not an offset into an action chunk.
UNUSABLE_STEPS: list[Any] = [-1, -5, True, False, 2.7, 3.0, float("nan"), float("inf"), "3", [2], {}]
USABLE_STEPS: list[Any] = [0, 1, 3, 120]


def _verdict(fn: Any) -> str:
    """Classify one call as ``refused`` (the documented ValueError) or otherwise.

    Any other exception type is reported by name so a test can distinguish a
    refusal that names the parameter from a comparison or conversion internal
    leaking out of a method documented to raise ``ValueError``. Catching
    ``BaseException`` would fold an interrupted run into a verdict string, so
    only ``Exception`` is collected.
    """
    try:
        fn()
    except ValueError:
        return "refused"
    except Exception as exc:  # noqa: BLE001 - the type IS the finding
        return f"leaked {type(exc).__name__}"
    return "accepted"


class TestControlFrequencyDomain:
    """``set_control_frequency`` accepts exactly the rates a loop can run at."""

    @pytest.mark.parametrize("bad", UNUSABLE_RATES)
    def test_an_unusable_rate_is_refused_where_it_arrives(self, bad):
        p = MockPolicy()
        assert _verdict(lambda: p.set_control_frequency(bad)) == "refused"
        # Refused means not stored: a later read cannot pick up the bad value.
        assert p.control_frequency is None

    @pytest.mark.parametrize("bad", UNUSABLE_RATES)
    def test_the_refusal_names_the_parameter_and_the_value(self, bad):
        p = MockPolicy()
        with pytest.raises(ValueError) as excinfo:
            p.set_control_frequency(bad)
        message = str(excinfo.value)
        assert "set_control_frequency" in message
        assert "control_frequency" in message
        assert repr(bad) in message

    @pytest.mark.parametrize("hz", USABLE_RATES)
    def test_a_usable_rate_is_stored_as_a_float(self, hz):
        p = MockPolicy()
        p.set_control_frequency(hz)
        assert isinstance(p.control_frequency, float)
        assert p.control_frequency == float(hz)

    @pytest.mark.parametrize("hz", USABLE_RATES)
    def test_every_accepted_rate_converts_into_a_step_count(self, hz):
        """The property the domain exists for: an accepted rate is usable downstream.

        The provider multiplies a measured latency by this rate and takes
        ``int()`` of it. A stored ``nan``/``inf`` raised there instead - and not
        on the first inference, because the estimator returns ``0`` until it has
        a latency sample, so the failure landed mid-rollout.
        """
        policy = LerobotLocalPolicy.__new__(LerobotLocalPolicy)
        policy._rtc_latency_history = deque([0.05])
        p = MockPolicy()
        p.set_control_frequency(hz)
        assert p.control_frequency is not None
        delay = policy._estimate_inference_delay(fps=p.control_frequency)
        assert isinstance(delay, int)
        assert delay >= 0

    def test_a_stored_non_finite_rate_would_break_the_estimator(self):
        """Why the rate is checked at the setter rather than at the read.

        Pinning the downstream failure keeps the justification honest: if this
        ever stops raising, the guard's rationale needs revisiting rather than
        silently becoming decoration.
        """
        policy = LerobotLocalPolicy.__new__(LerobotLocalPolicy)
        policy._rtc_latency_history = deque([0.05])
        with pytest.raises(ValueError):
            policy._estimate_inference_delay(fps=float("nan"))
        with pytest.raises(OverflowError):
            policy._estimate_inference_delay(fps=float("inf"))


class TestObservedDelayDomain:
    """``set_rtc_observed_delay`` accepts exactly the counts a chunk can be sliced by."""

    @pytest.mark.parametrize("bad", UNUSABLE_STEPS)
    def test_an_unusable_step_count_is_refused(self, bad):
        p = MockPolicy()
        assert _verdict(lambda: p.set_rtc_observed_delay(bad)) == "refused"
        assert p.rtc_observed_delay_steps is None

    @pytest.mark.parametrize("bad", UNUSABLE_STEPS)
    def test_the_refusal_names_the_parameter_and_the_value(self, bad):
        p = MockPolicy()
        with pytest.raises(ValueError) as excinfo:
            p.set_rtc_observed_delay(bad)
        message = str(excinfo.value)
        assert "set_rtc_observed_delay" in message
        assert "rtc_observed_delay_steps" in message
        assert repr(bad) in message

    @pytest.mark.parametrize("steps", USABLE_STEPS)
    def test_a_usable_step_count_is_stored_verbatim(self, steps):
        p = MockPolicy()
        p.set_rtc_observed_delay(steps)
        assert p.rtc_observed_delay_steps == steps
        assert isinstance(p.rtc_observed_delay_steps, int)

    def test_zero_is_the_dominant_case_and_stays_usable(self):
        """A synchronous eval loop pauses the world, so exactly zero steps elapse.

        This is why the count has its own non-negative domain instead of reusing
        the positive-count one: refusing ``0`` would refuse the common path.
        """
        p = MockPolicy()
        p.set_rtc_observed_delay(0)
        assert p.rtc_observed_delay_steps == 0

    def test_none_clears_the_override(self):
        p = MockPolicy()
        p.set_rtc_observed_delay(7)
        p.set_rtc_observed_delay(None)
        assert p.rtc_observed_delay_steps is None


class TestPolicyAndRunnerAgreeOnTheRateDomain:
    """The same control rate cannot be refused by one layer and accepted by the other.

    ``SimEngine._validate_positive_frequency`` guards the rate the rollout loop
    runs at; ``Policy.set_control_frequency`` receives that same rate one call
    later. A rate the runner refuses must not be settable on a policy, and a
    rate the runner accepts must not be refused by one.
    """

    @pytest.mark.parametrize("value", UNUSABLE_RATES + USABLE_RATES)
    def test_the_two_guards_return_the_same_verdict(self, value):
        runner_refuses = SimEngine._validate_positive_frequency(value, "run_policy") is not None
        policy_refuses = _verdict(lambda: MockPolicy().set_control_frequency(value)) == "refused"
        assert policy_refuses == runner_refuses, (
            f"verdicts differ for {value!r}: runner_refuses={runner_refuses}, policy_refuses={policy_refuses}"
        )


class _RecordingPolicy(MockPolicy):
    """A policy that records whether inference was reached."""

    def __init__(self) -> None:
        super().__init__()
        self.inference_calls: list[str] = []

    def get_actions_sync(self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any) -> Any:
        self.inference_calls.append(instruction)
        return super().get_actions_sync(observation_dict, instruction, **kwargs)


class TestTheWireCannotAcceptWhatTheLocalApiRefuses:
    """A remote caller reaches the same accepted domain as an in-process one.

    ``PolicyServer`` applies both handshake values to the wrapped policy before
    inference. It used to coerce the rate with ``float(...)`` first, which
    re-admitted the two values the policy-side guard exists to refuse.
    """

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "50"])
    def test_the_wire_can_really_deliver_these_values(self, value):
        """Executable premise: the transport is JSON, which round-trips these.

        ``json`` emits the non-standard ``NaN``/``Infinity`` tokens by default
        and reads them back, and a JSON ``true`` decodes to a Python ``bool``,
        so none of these is hypothetical for a third-party client.
        """
        decoded = protocol.loads(protocol.dumps({"hz": value}))["hz"]
        if isinstance(value, float) and math.isnan(value):
            assert isinstance(decoded, float) and math.isnan(decoded)
        else:
            assert decoded == value
            assert type(decoded) is type(value)

    @pytest.mark.parametrize("value", UNUSABLE_RATES + USABLE_RATES)
    def test_the_rate_domain_is_the_same_over_the_wire(self, value):
        local = _verdict(lambda: MockPolicy().set_control_frequency(value))
        server = PolicyServer(policy=MockPolicy())
        wire = _verdict(
            lambda: server._dispatch({"type": protocol.MSG_SET_CONTROL_FREQUENCY, "hz": value}),
        )
        assert wire == local, f"verdicts differ for {value!r}: local={local}, wire={wire}"

    @pytest.mark.parametrize("value", UNUSABLE_STEPS)
    def test_an_unusable_step_count_from_the_wire_never_reaches_inference(self, value):
        policy = _RecordingPolicy()
        server = PolicyServer(policy=policy)
        with pytest.raises(ValueError):
            server._dispatch(
                {
                    "type": protocol.MSG_GET_ACTIONS,
                    "observation": {},
                    "instruction": "pick up the cube",
                    "rtc_observed_delay_steps": value,
                }
            )
        assert policy.inference_calls == []
        assert policy.rtc_observed_delay_steps is None

    def test_a_usable_step_count_from_the_wire_still_reaches_inference(self):
        policy = _RecordingPolicy()
        server = PolicyServer(policy=policy)
        reply = server._dispatch(
            {
                "type": protocol.MSG_GET_ACTIONS,
                "observation": {},
                "instruction": "pick up the cube",
                "rtc_observed_delay_steps": 0,
            }
        )
        assert reply["type"] == protocol.MSG_ACTIONS
        assert policy.inference_calls == ["pick up the cube"]
        assert policy.rtc_observed_delay_steps == 0


class TestForwardingWrappersRefuseBeforeForwarding:
    """A wrapper that fans a handshake value out to children refuses first.

    ``CompositePolicy`` calls the base setter before forwarding, so a refused
    value must leave no child half-updated.
    """

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "50"])
    def test_an_unusable_rate_leaves_both_children_untouched(self, bad):
        lower, upper = MockPolicy(), MockPolicy()
        composite = CompositePolicy(lower, upper)
        with pytest.raises(ValueError):
            composite.set_control_frequency(bad)
        assert composite.control_frequency is None
        assert lower.control_frequency is None
        assert upper.control_frequency is None

    @pytest.mark.parametrize("bad", [True, 2.7, float("nan"), "3"])
    def test_an_unusable_step_count_leaves_both_children_untouched(self, bad):
        lower, upper = MockPolicy(), MockPolicy()
        composite = CompositePolicy(lower, upper)
        with pytest.raises(ValueError):
            composite.set_rtc_observed_delay(bad)
        assert composite.rtc_observed_delay_steps is None
        assert lower.rtc_observed_delay_steps is None
        assert upper.rtc_observed_delay_steps is None


class TestNonNegativeCountDomain:
    """The shared domain itself, independent of any caller."""

    @pytest.mark.parametrize("good", [0, 1, 2, 1000])
    def test_a_non_negative_int_is_accepted(self, good):
        assert non_negative_count_error(good, "steps", "ctx") is None

    @pytest.mark.parametrize("bad", [-1, True, False, 0.0, 2.7, float("nan"), "0", None, [0]])
    def test_everything_else_is_refused(self, bad):
        error = non_negative_count_error(bad, "steps", "ctx")
        assert error is not None
        assert error == f"ctx: steps must be a non-negative integer, got {bad!r}."

    def test_zero_separates_it_from_the_positive_count_domain(self):
        """The one difference from ``positive_count_error``, pinned deliberately."""
        from strands_robots.utils import positive_count_error

        assert non_negative_count_error(0, "steps", "ctx") is None
        assert positive_count_error(0, "steps", "ctx") is not None
