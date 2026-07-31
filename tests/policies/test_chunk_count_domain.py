"""A per-inference chunk count the consumer cannot execute is refused, not floored.

``actions_per_step`` (how many actions of one inference chunk a consumer executes
before re-querying) and ``actions_per_chunk`` (how many the provider emits) were
stored verbatim by both LeRobot providers and only ever reconciled on the read
path, where :attr:`~strands_robots.policies.base.Policy.execution_horizon`
resolves them through ``max(1, int(...))``. That floor converts a count no
consumer can execute into ``1`` and reports success.

For ``lerobot_local`` the floor is not merely lossy, it is worse than the
default. ``_auto_detect_actions_per_step`` treats any value other than the
default ``1`` as a horizon the caller pinned deliberately ("caller pinned an
explicit horizon - never override it") and returns without adopting
``config.n_action_steps``. So on a checkpoint trained to replay a 100-action
chunk open-loop, ``actions_per_step=0`` skipped that adoption AND floored the
horizon to ``1`` - re-querying the model every step, the out-of-distribution
operation the auto-detection exists to prevent - while the default ``1`` would
have executed the trained 100. It also flipped
:meth:`~strands_robots.policies.base.Policy.is_chunk_emitting` to ``False``,
silently disabling the async-RTC latency masking for a chunk-emitting model.

``rtc_execution_horizon`` is the same count once more. With Real-Time Chunking
active it *is* the re-query interval, replacing ``actions_per_step`` in the very
same property, and it was stored verbatim. ``0`` was the sharpest spelling: it is
falsy, so the property fell through to the full trained chunk - the open-loop
replay its own docstring says collapses RTC - while ``_init_rtc`` skipped
adopting the checkpoint's ``rtc_config.execution_horizon`` because ``0`` is not
``None``, and the model was handed the ``0`` verbatim. Consumer and model
disagreed about the horizon for every unusable value, which is exactly the
agreement RTC's cross-chunk blend rests on.

These tests pin:

* the degraded regime is unreachable, against a measured control showing what
  the default does with the same checkpoint;
* the RTC re-query horizon shares that domain, and an accepted one is the same
  number the consumer executes and the model is told;
* every value outside the domain is refused by both providers, with parity
  between them apart from the one documented asymmetry (``None`` is
  ``lerobot_async``'s "use ``actions_per_chunk``" and not a count at all);
* the refusal precedes the checkpoint download, in the constructor and in
  ``preflight`` (which the rollout entry points run first, so the same mistake
  surfaces as a structured error rather than a raise);
* ``actions_per_chunk`` is refused too - it is the default for
  ``actions_per_step``, so an unchecked one reopens the same path;
* the duck-typed floor in ``resolve_chunk_length`` is deliberately untouched.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from strands_robots.policies import MockPolicy, preflight_policy
from strands_robots.policies.base import resolve_chunk_length
from strands_robots.policies.lerobot_async.policy import LerobotAsyncPolicy
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

# Values no chunk consumer can execute, or cannot execute as written. Each is a
# plausible way to spell an intent the loop has no way to honor.
REJECTED: list[Any] = [0, -5, False, True, 2.7, "4", float("nan"), float("inf"), [4]]

# The trained chunk length of the checkpoint the fixtures below emulate.
TRAINED_CHUNK = 100

# The RTC re-query horizon that same checkpoint declares in its ``rtc_config``.
# Deliberately unequal to TRAINED_CHUNK: the RTC horizon exists to re-query
# mid-chunk, so a fixture where they matched could not tell the two apart.
MODEL_RTC_HORIZON = 10

# Annotated ``Any`` so mypy does not narrow the splat against the mixed-type
# signature; the two required identifiers are strings.
ASYNC_REQUIRED: dict[str, Any] = {"policy_type": "act", "pretrained_name_or_path": "acme/act-cube"}


class _ChunkTrainedConfig:
    """An ACT-shaped config: a 100-action chunk trained for open-loop replay."""

    temporal_ensemble_coeff = None
    n_obs_steps = 1
    n_action_steps = TRAINED_CHUNK


class _ChunkTrainedModel:
    config = _ChunkTrainedConfig()


class _RtcConfig:
    """The RTC block a flow-matching checkpoint (Pi0, SmolVLA) carries."""

    enabled = True
    execution_horizon = MODEL_RTC_HORIZON
    max_guidance_weight = 10.0


class _RtcTrainedConfig(_ChunkTrainedConfig):
    """The same trained chunk, on a checkpoint that also enables RTC."""

    rtc_config = _RtcConfig()


class _RtcTrainedModel:
    config = _RtcTrainedConfig()

    def predict_action_chunk(self, *args: Any, **kwargs: Any) -> Any:
        """Present so ``_init_rtc`` sees a flow-matching policy; never called."""
        raise AssertionError("inference is out of scope for these tests")


def _local(**config: Any) -> LerobotLocalPolicy:
    """Build a ``lerobot_local`` policy with the model load stubbed out."""
    with patch.object(LerobotLocalPolicy, "_load_model"):
        return LerobotLocalPolicy(pretrained_name_or_path="acme/act-cube", **config)


def _regime(policy: LerobotLocalPolicy) -> tuple[Any, int, bool]:
    """Resolve the inference regime a loaded chunk-trained checkpoint yields."""
    policy._policy = _ChunkTrainedModel()
    policy._auto_detect_actions_per_step()
    return policy.actions_per_step, policy.execution_horizon, policy.is_chunk_emitting()


def _rtc_horizons(**config: Any) -> tuple[int, Any]:
    """Resolve (what the consumer executes, what the model is told) under RTC.

    Loads the RTC-enabled checkpoint emulation so ``_init_rtc`` runs its
    adoption branch. The second element is the value
    :mod:`strands_robots.policies.lerobot_local.policy` forwards to the model as
    ``rtc_kwargs["execution_horizon"]``; RTC's cross-chunk blend is only correct
    while the two agree, so the tests below compare them rather than reading
    either alone.
    """
    policy = _local(**config)
    policy._policy = _RtcTrainedModel()
    policy._loaded = True
    policy._init_rtc()
    assert policy.supports_rtc, "fixture must reach the RTC path for these assertions to mean anything"
    return policy.execution_horizon, policy._rtc_execution_horizon


class TestTheDegradedRegimeIsUnreachable:
    """The count that discarded the trained chunk is refused up front."""

    def test_the_default_adopts_the_checkpoints_trained_chunk(self) -> None:
        """Control: left at the default, the trained 100-action chunk is executed whole."""
        actions_per_step, horizon, chunk_emitting = _regime(_local())

        assert actions_per_step == TRAINED_CHUNK
        assert horizon == TRAINED_CHUNK
        assert chunk_emitting is True

    def test_a_pinned_count_is_still_honored_verbatim(self) -> None:
        """Control: an explicit, executable count overrides the auto-detection."""
        actions_per_step, horizon, chunk_emitting = _regime(_local(actions_per_step=8))

        assert actions_per_step == 8
        assert horizon == 8
        assert chunk_emitting is True

    def test_a_zero_count_cannot_discard_the_trained_chunk(self) -> None:
        """``0`` is refused rather than skipping the adoption and flooring to 1.

        Pre-fix this constructed, and ``_regime`` then reported
        ``(0, 1, False)`` - no adoption of the trained 100, a re-query every
        step, and chunk emission undetected - all under a successful call.
        """
        with pytest.raises(ValueError, match="actions_per_step must be a positive integer"):
            _local(actions_per_step=0)

    def test_a_fractional_count_is_not_truncated(self) -> None:
        """``2.7`` is refused rather than silently executing 2 actions per chunk."""
        with pytest.raises(ValueError, match="actions_per_step must be a positive integer"):
            _local(actions_per_step=2.7)

    def test_a_non_numeric_count_is_refused_where_it_arrives(self) -> None:
        """A value ``int()`` cannot read is refused at the constructor.

        ``execution_horizon`` is a property read by ``resolve_chunk_length``
        inside the rollout loop, so on the read path these surfaced as a bare
        ``TypeError``/``ValueError``/``OverflowError`` mid-rollout.
        """
        for value in (None, [4], float("nan"), float("inf")):
            with pytest.raises(ValueError, match="actions_per_step must be a positive integer"):
                _local(actions_per_step=value)


class TestTheRefusalPrecedesTheCheckpointLoad:
    """Nothing is downloaded for a configuration that cannot run."""

    def test_the_constructor_refuses_before_loading_the_model(self) -> None:
        loads: list[int] = []

        def record(self: LerobotLocalPolicy) -> None:
            loads.append(1)

        with patch.object(LerobotLocalPolicy, "_load_model", record):
            with pytest.raises(ValueError, match="actions_per_step"):
                LerobotLocalPolicy(pretrained_name_or_path="acme/act-cube", actions_per_step=0)

        assert loads == [], "the checkpoint load must not be reached for a refused count"

    def test_preflight_refuses_the_count_with_no_embodiment_configured(self) -> None:
        """The check runs before ``preflight``'s embodiment early-return.

        The rollout entry points run ``preflight`` before ``create_policy``, so
        this is what turns the same mistake into a structured error instead of a
        raise. Scoping it behind an embodiment would leave every configuration
        that declares none unguarded on that path.
        """
        with pytest.raises(ValueError, match="actions_per_step must be a positive integer"):
            LerobotLocalPolicy.preflight(set(), actions_per_step=0)

    def test_the_provider_preflight_entry_point_refuses_it(self) -> None:
        """Reached through the same function the rollout entry points call."""
        with pytest.raises(ValueError, match="actions_per_step must be a positive integer"):
            preflight_policy("lerobot_local", {"joint_1", "front"}, actions_per_step=-5)

    def test_preflight_passes_an_executable_count(self) -> None:
        preflight_policy("lerobot_local", {"joint_1", "front"}, actions_per_step=30)


class TestBothProvidersShareTheDomain:
    """The same chunk count cannot be refused locally and accepted remotely."""

    @pytest.mark.parametrize("value", REJECTED, ids=repr)
    def test_lerobot_local_refuses_it(self, value: Any) -> None:
        with pytest.raises(ValueError, match="actions_per_step must be a positive integer"):
            _local(actions_per_step=value)

    @pytest.mark.parametrize("value", REJECTED, ids=repr)
    def test_lerobot_async_refuses_it(self, value: Any) -> None:
        with pytest.raises(ValueError, match="actions_per_step must be a positive integer"):
            LerobotAsyncPolicy(**ASYNC_REQUIRED, actions_per_step=value)

    @pytest.mark.parametrize("value", REJECTED, ids=repr)
    def test_the_two_providers_reach_the_same_verdict(self, value: Any) -> None:
        """Parity, so the domains cannot drift apart again."""

        def verdict(build: Any) -> str:
            try:
                build()
            except ValueError as exc:
                return "refused" if "must be a positive integer" in str(exc) else "other-error"
            return "accepted"

        local = verdict(lambda: _local(actions_per_step=value))
        remote = verdict(lambda: LerobotAsyncPolicy(**ASYNC_REQUIRED, actions_per_step=value))
        assert local == remote, f"verdicts differ for actions_per_step={value!r}"

    @pytest.mark.parametrize("value", [1, 8, 30, TRAINED_CHUNK])
    def test_both_providers_accept_an_executable_count(self, value: int) -> None:
        assert _local(actions_per_step=value).actions_per_step == value
        assert LerobotAsyncPolicy(**ASYNC_REQUIRED, actions_per_step=value).actions_per_step == value


class TestTheAsyncChunkLength:
    """``actions_per_chunk`` is the default for ``actions_per_step``, so it is checked too."""

    @pytest.mark.parametrize("value", REJECTED, ids=repr)
    def test_an_unusable_chunk_length_cannot_supply_the_step_count(self, value: Any) -> None:
        """Otherwise omitting ``actions_per_step`` reopens the same path."""
        with pytest.raises(ValueError, match="actions_per_chunk must be a positive integer"):
            LerobotAsyncPolicy(**ASYNC_REQUIRED, actions_per_chunk=value)

    def test_omitting_the_step_count_still_adopts_the_chunk_length(self) -> None:
        """``None`` is this provider's documented default, not a count."""
        policy = LerobotAsyncPolicy(**ASYNC_REQUIRED, actions_per_chunk=32, actions_per_step=None)

        assert policy.actions_per_step == 32
        assert policy.execution_horizon == 32

    def test_lerobot_local_has_no_such_default_and_refuses_none(self) -> None:
        """The one asymmetry, and it follows from the two signatures.

        ``lerobot_local`` declares ``actions_per_step: int = 1``, so ``None`` is
        not a spelling of its default and there is no chunk length to fall back
        to.
        """
        with pytest.raises(ValueError, match="actions_per_step must be a positive integer"):
            _local(actions_per_step=None)


class TestTheRtcReQueryHorizon:
    """``rtc_execution_horizon`` replaces the step count, so it shares its domain."""

    def test_omitting_it_adopts_the_checkpoints_own_rtc_horizon(self) -> None:
        """Control, and the premise the refusals below are measured against."""
        consumer, model = _rtc_horizons()

        assert consumer == MODEL_RTC_HORIZON
        assert model == MODEL_RTC_HORIZON

    def test_a_supplied_horizon_overrides_the_checkpoints(self) -> None:
        """Control: an executable override is honored, not merely tolerated."""
        consumer, model = _rtc_horizons(rtc_execution_horizon=8)

        assert consumer == 8
        assert model == 8

    def test_a_zero_horizon_cannot_collapse_rtc_to_open_loop_replay(self) -> None:
        """``0`` is refused rather than silently disabling the feature it configures.

        Pre-fix this constructed. ``0`` is falsy, so ``execution_horizon`` fell
        through to ``actions_per_step`` and the consumer executed the whole
        trained chunk before re-querying - the open-loop replay that keeps the
        blended tail empty - while ``_init_rtc`` skipped adopting the
        checkpoint's own horizon (``0`` is not ``None``) and the model was told
        ``0``. Measured on this fixture: consumer 100, model 0, against 10/10
        for the same call with the parameter omitted.
        """
        with pytest.raises(ValueError, match="rtc_execution_horizon must be a positive integer"):
            _local(rtc_execution_horizon=0)

    @pytest.mark.parametrize("value", REJECTED, ids=repr)
    def test_an_unusable_horizon_is_refused_where_it_arrives(self, value: Any) -> None:
        """Not on the read path, which is a property read inside the rollout loop.

        ``nan``, ``inf`` and a list surfaced there as a bare
        ``ValueError``/``OverflowError``/``TypeError`` mid-rollout; the rest were
        floored or truncated into a horizon the caller never asked for.
        """
        with pytest.raises(ValueError, match="rtc_execution_horizon must be a positive integer"):
            _local(rtc_execution_horizon=value)

    def test_none_is_the_adopt_request_and_not_a_count(self) -> None:
        """The documented default: take the checkpoint's own horizon.

        Unlike ``actions_per_step``, whose signature declares ``int = 1``, this
        parameter declares ``int | None = None``, so ``None`` must stay valid.
        """
        assert _local(rtc_execution_horizon=None)._rtc_execution_horizon is None

    def test_the_refusal_precedes_the_checkpoint_load(self) -> None:
        loads: list[int] = []

        def record(self: LerobotLocalPolicy) -> None:
            loads.append(1)

        with patch.object(LerobotLocalPolicy, "_load_model", record):
            with pytest.raises(ValueError, match="rtc_execution_horizon"):
                LerobotLocalPolicy(pretrained_name_or_path="acme/pi0-cube", rtc_execution_horizon=0)

        assert loads == [], "the checkpoint load must not be reached for a refused horizon"

    def test_preflight_refuses_it_with_no_embodiment_configured(self) -> None:
        """RTC is decided by the model config, so this cannot be embodiment-scoped."""
        with pytest.raises(ValueError, match="rtc_execution_horizon must be a positive integer"):
            LerobotLocalPolicy.preflight(set(), rtc_execution_horizon=0)

    def test_the_provider_preflight_entry_point_refuses_it(self) -> None:
        """The structured-error path the rollout entry points run first."""
        with pytest.raises(ValueError, match="rtc_execution_horizon must be a positive integer"):
            preflight_policy("lerobot_local", {"joint_1", "front"}, rtc_execution_horizon=-5)

    def test_preflight_passes_an_executable_horizon_and_the_adopt_request(self) -> None:
        preflight_policy("lerobot_local", {"joint_1", "front"}, rtc_execution_horizon=10)
        preflight_policy("lerobot_local", {"joint_1", "front"}, rtc_execution_horizon=None)

    @pytest.mark.parametrize("value", REJECTED, ids=repr)
    def test_it_shares_the_verdict_with_the_step_count_it_replaces(self, value: Any) -> None:
        """Parity: one horizon cannot be refused under one name and accepted under the other."""

        def verdict(build: Any) -> str:
            try:
                build()
            except ValueError as exc:
                return "refused" if "must be a positive integer" in str(exc) else "other-error"
            return "accepted"

        as_step_count = verdict(lambda: _local(actions_per_step=value))
        as_rtc_horizon = verdict(lambda: _local(rtc_execution_horizon=value))
        assert as_step_count == as_rtc_horizon, f"verdicts differ for horizon={value!r}"

    @pytest.mark.parametrize("value", [1, 8, MODEL_RTC_HORIZON, TRAINED_CHUNK])
    def test_an_accepted_horizon_is_the_one_both_sides_use(self, value: int) -> None:
        """The invariant the unusable values broke, pinned for the whole domain."""
        consumer, model = _rtc_horizons(rtc_execution_horizon=value)

        assert consumer == value
        assert model == value


class TestTheDuckTypedFloorIsUntouched:
    """The read-path floor stays: it is the default for a non-provider chunk source."""

    def test_resolve_chunk_length_still_floors_an_unvalidated_source(self) -> None:
        """A duck-typed object never passed through a provider constructor.

        ``resolve_chunk_length`` reads ``execution_horizon`` off anything that
        offers one, so its floor is the only thing standing between an
        unvalidated chunk source and a zero-length slice. Narrowing it is a
        separate contract (see ``test_chunked_policy_contract``); this fix
        checks the counts where a caller supplies them instead.
        """

        class _Unvalidated:
            actions_per_step = 0

        # Not a Policy by declared type - that is exactly the caller this
        # floor exists for; it only needs an ``actions_per_step`` attribute.
        assert resolve_chunk_length(_Unvalidated(), action_horizon=0) == 1  # type: ignore[arg-type]

    def test_a_single_step_policy_is_unaffected(self) -> None:
        assert resolve_chunk_length(MockPolicy(), action_horizon=8) == 8
