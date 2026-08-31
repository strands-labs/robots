"""Episodic behaviors in :class:`MicroduckPolicyBundle` auto-return on their own.

Pollen's reference ``scripts/infer_policy.py`` has a second, orthogonal
switching mode alongside the walking/standing velocity gate: **timed episodic
behaviors**. ``trigger_behavior("kick_left")`` arms a short skill with a fixed
duration; ``update_behavior(dt)`` decrements ``behavior_time_left`` each tick
and ``_end_behavior`` reverts to the default when it reaches zero.

Without this the bundle exposes a switching FSM that only ever enters a state.
A caller who wanted a walk -> kick -> walk sequence had ``switch("kick_left")``,
which stayed on ``kick_left`` forever, feeding the walking/standing observation
contract to an episodic policy that expects to end within ~1.2s. This test
holds the new API to the reference FSM's contract: only declared skills can be
triggered; a running episodic behavior blocks another (matches
``_end_behavior`` gating); the timer counts down at exactly the control rate
the runtime told the bundle; the last tick of the episode still executes the
episodic skill; and the auto-return goes to ``default_skill``, which is
refused if it names an episodic skill (an auto-return that re-arms its own
timer never terminates).

The control rate is read off :attr:`Policy.control_frequency`, the same seam
every other consumer uses; a bundle asked to run an episodic behavior without
being told its clock refuses loudly rather than assuming 50Hz, because an
assumed rate mis-times every kick at any other loop rate.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from strands_robots.policies.microduck import MicroduckPolicy, MicroduckPolicyBundle
from tests.policies.microduck.test_microduck_policy import _obs_dict, _StubSession

#: Skills the bundle holds. ``walk`` and ``stand`` are the non-episodic
#: defaults; the three ``kick_*`` / ``roulade`` names match the upstream
#: reference and each pair with a duration in seconds.
DEFAULT_MOVE = "walk"
DEFAULT_IDLE = "stand"
KICK_LEFT = "kick_left"
KICK_RIGHT = "kick_right"
ROULADE = "roulade"

#: The rate every runtime tick advances at. 50Hz is the shipped default and
#: makes a 1.2s kick land on 60 ticks exactly - a whole-integer count the
#: revert-tick assertion can name without epsilon slop.
CONTROL_HZ = 50.0
KICK_DURATION_S = 1.2
KICK_STEPS = 60  # 1.2 s @ 50 Hz


def _policy() -> MicroduckPolicy:
    """A real policy over an injected stub session - no onnxruntime needed."""
    return MicroduckPolicy(session=_StubSession())


def _bundle(
    *,
    episodic_skills: dict[str, float] | None = None,
    default_skill: str | None = None,
    active: str | None = None,
    set_hz: bool = True,
) -> MicroduckPolicyBundle:
    """A four-skill bundle over stubs, matching the upstream skill set.

    Args:
        episodic_skills: Passed through to the constructor. When ``None``, the
            episodic path is entirely off - :meth:`trigger` refuses, the timer
            never runs, :meth:`reset` leaves the active skill alone, and the
            shape of the previous velocity-gated bundle is unchanged.
        default_skill: The skill the bundle reverts to when an episode ends.
            ``None`` uses the initially active skill.
        active: Initial active skill. Defaults to ``walk`` so the auto-return
            fallback is a stable non-episodic anchor.
        set_hz: Whether to call :meth:`set_control_frequency`. Off only for the
            one test that holds the bundle to a refusal when the clock is
            missing.
    """
    bundle = MicroduckPolicyBundle(
        {
            DEFAULT_MOVE: _policy(),
            DEFAULT_IDLE: _policy(),
            KICK_LEFT: _policy(),
            KICK_RIGHT: _policy(),
            ROULADE: _policy(),
        },
        active=active or DEFAULT_MOVE,
        episodic_skills=episodic_skills,
        default_skill=default_skill,
    )
    if set_hz:
        bundle.set_control_frequency(CONTROL_HZ)
    return bundle


def _tick(bundle: MicroduckPolicyBundle, **kwargs: Any) -> None:
    """One tick, without asserting shape - the test cares about ``active``."""
    asyncio.run(bundle.get_actions(_obs_dict(), "", **kwargs))


class TestTriggerIsGatedByTheDeclaredEpisodicSkillSet:
    """A caller who never declared an episodic skill cannot trigger one.

    ``trigger`` is the ``trigger_behavior`` seam from ``infer_policy.py``. The
    upstream FSM refuses a name it never registered; the bundle inherits the
    same rule off the ``episodic_skills`` mapping, which also holds the
    per-skill duration the timer counts down.
    """

    def test_a_bundle_with_no_episodic_skills_refuses_every_trigger(self) -> None:
        bundle = _bundle()
        with pytest.raises(ValueError, match="is not a declared episodic skill"):
            bundle.trigger(KICK_LEFT)

    def test_a_bundle_names_the_declared_set_when_it_refuses_an_unknown(self) -> None:
        bundle = _bundle(episodic_skills={KICK_LEFT: KICK_DURATION_S})
        with pytest.raises(ValueError, match=r"have \['kick_left'\]"):
            bundle.trigger(KICK_RIGHT)

    def test_an_episodic_skill_that_names_no_held_policy_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="names no held skill"):
            MicroduckPolicyBundle(
                {DEFAULT_MOVE: _policy(), DEFAULT_IDLE: _policy()},
                episodic_skills={"phantom_skill": KICK_DURATION_S},
            )

    @pytest.mark.parametrize("bad_duration", [0.0, -0.5, float("nan"), float("inf")])
    def test_a_duration_that_is_not_a_positive_finite_number_is_refused(self, bad_duration: float) -> None:
        with pytest.raises(ValueError):
            MicroduckPolicyBundle(
                {DEFAULT_MOVE: _policy(), KICK_LEFT: _policy()},
                episodic_skills={KICK_LEFT: bad_duration},
            )


class TestDefaultSkillIsHeldAndCannotBeEpisodic:
    """The revert target is validated at construction, same seam every other key is.

    ``default_skill`` is where the bundle lands when a timer expires. Two rules
    hold: it must name a held policy (like ``active`` and the two gate keys),
    and it cannot itself be an episodic skill - an auto-return into an episodic
    skill would re-arm the timer the tick it fires and never terminate.
    """

    def test_a_default_skill_that_names_no_held_policy_is_refused(self) -> None:
        with pytest.raises(ValueError, match="default_skill='ghost'"):
            MicroduckPolicyBundle(
                {DEFAULT_MOVE: _policy(), DEFAULT_IDLE: _policy()},
                default_skill="ghost",
            )

    def test_a_default_skill_that_is_itself_episodic_is_refused(self) -> None:
        with pytest.raises(ValueError, match="is itself an episodic skill"):
            MicroduckPolicyBundle(
                {DEFAULT_MOVE: _policy(), KICK_LEFT: _policy()},
                episodic_skills={KICK_LEFT: KICK_DURATION_S},
                default_skill=KICK_LEFT,
            )


class TestTheTimerCountsDownAtTheControlRate:
    """The bundle uses :attr:`control_frequency` as the tick rate, no assumption.

    At 50 Hz a 1.2 s kick is 60 ticks. The 60th tick still executes the
    episodic skill; the 61st runs ``default_skill``. That off-by-one holds
    because :meth:`get_actions` decrements the timer AFTER delegating to the
    active child - a caller reading the last frame of the episode sees the
    episodic policy's output, not the next skill's.
    """

    def test_a_bundle_with_no_clock_refuses_when_the_first_tick_would_decrement(self) -> None:
        bundle = _bundle(
            episodic_skills={KICK_LEFT: KICK_DURATION_S},
            default_skill=DEFAULT_MOVE,
            set_hz=False,
        )
        bundle.trigger(KICK_LEFT)
        with pytest.raises(RuntimeError, match="requires control_frequency"):
            _tick(bundle)

    def test_the_last_tick_of_the_episode_runs_the_episodic_skill(self) -> None:
        bundle = _bundle(
            episodic_skills={KICK_LEFT: KICK_DURATION_S},
            default_skill=DEFAULT_MOVE,
        )
        bundle.trigger(KICK_LEFT)
        for _ in range(KICK_STEPS - 1):
            _tick(bundle)
            assert bundle.active == KICK_LEFT
            assert bundle.episodic_active == KICK_LEFT
        # The 60th tick still runs the kick policy; the timer expires AFTER.
        _tick(bundle)
        assert bundle.episodic_active is None
        assert bundle.active == DEFAULT_MOVE

    def test_the_first_tick_after_expiry_runs_the_default_skill(self) -> None:
        bundle = _bundle(
            episodic_skills={KICK_LEFT: KICK_DURATION_S},
            default_skill=DEFAULT_MOVE,
        )
        bundle.trigger(KICK_LEFT)
        for _ in range(KICK_STEPS):
            _tick(bundle)
        _tick(bundle)
        assert bundle.active == DEFAULT_MOVE
        assert bundle.episodic_active is None


class TestOnlyOneEpisodicBehaviorAtATime:
    """Trigger during an active episode refuses, mirroring ``_end_behavior``.

    The upstream FSM does not queue behaviors - a running one wins. A second
    ``trigger`` must be refused explicitly (naming the running one and its
    remaining time) so the caller either waits or overrides via
    :meth:`switch`, which cancels.
    """

    def test_triggering_a_second_episodic_skill_mid_episode_is_refused(self) -> None:
        bundle = _bundle(
            episodic_skills={KICK_LEFT: KICK_DURATION_S, KICK_RIGHT: KICK_DURATION_S},
            default_skill=DEFAULT_MOVE,
        )
        bundle.trigger(KICK_LEFT)
        _tick(bundle)  # one tick in, ~1.18s left
        with pytest.raises(ValueError, match="'kick_left' is already running"):
            bundle.trigger(KICK_RIGHT)

    def test_switch_cancels_a_running_episode_and_frees_the_next_trigger(self) -> None:
        bundle = _bundle(
            episodic_skills={KICK_LEFT: KICK_DURATION_S, KICK_RIGHT: KICK_DURATION_S},
            default_skill=DEFAULT_MOVE,
        )
        bundle.trigger(KICK_LEFT)
        _tick(bundle)
        bundle.switch(DEFAULT_MOVE)
        assert bundle.episodic_active is None
        assert bundle.active == DEFAULT_MOVE
        # A trigger now succeeds without complaint about kick_left running.
        bundle.trigger(KICK_RIGHT)
        assert bundle.episodic_active == KICK_RIGHT
        assert bundle.active == KICK_RIGHT


class TestTheVelocityGateAndEpisodicBehaviorDoNotFightEachOther:
    """During an episode the velocity gate is inhibited.

    Otherwise a caller who commanded twist=[0.3, 0, 0] during a kick would see
    the bundle yank ``walk`` back mid-kick every tick - the walking/standing
    gate would override the episodic FSM. The reference ``infer_policy.py``
    resolves this by treating a running behavior as top-priority; the bundle
    does the same by skipping ``_auto_switch`` while ``episodic_active`` is
    non-``None``.
    """

    def test_a_running_kick_is_not_preempted_by_a_move_command(self) -> None:
        bundle = MicroduckPolicyBundle(
            {DEFAULT_MOVE: _policy(), DEFAULT_IDLE: _policy(), KICK_LEFT: _policy()},
            active=DEFAULT_IDLE,
            switch_on_velocity=0.05,
            move_key=DEFAULT_MOVE,
            idle_key=DEFAULT_IDLE,
            episodic_skills={KICK_LEFT: KICK_DURATION_S},
            default_skill=DEFAULT_IDLE,
        )
        bundle.set_control_frequency(CONTROL_HZ)
        bundle.trigger(KICK_LEFT)
        _tick(bundle, target_velocity=[0.3, 0.0, 0.0])
        assert bundle.active == KICK_LEFT

    def test_the_velocity_gate_resumes_after_the_episode_ends(self) -> None:
        bundle = MicroduckPolicyBundle(
            {DEFAULT_MOVE: _policy(), DEFAULT_IDLE: _policy(), KICK_LEFT: _policy()},
            active=DEFAULT_IDLE,
            switch_on_velocity=0.05,
            move_key=DEFAULT_MOVE,
            idle_key=DEFAULT_IDLE,
            episodic_skills={KICK_LEFT: KICK_DURATION_S},
            default_skill=DEFAULT_IDLE,
        )
        bundle.set_control_frequency(CONTROL_HZ)
        bundle.trigger(KICK_LEFT)
        for _ in range(KICK_STEPS):
            _tick(bundle, target_velocity=[0.3, 0.0, 0.0])
        _tick(bundle, target_velocity=[0.3, 0.0, 0.0])
        assert bundle.active == DEFAULT_MOVE
        assert bundle.episodic_active is None


class TestResetClearsTheEpisodicFSM:
    """``reset`` is the per-rollout seam; it must not leave the timer armed."""

    def test_an_episodic_bundle_normalises_the_active_skill_with_no_episode_running(self) -> None:
        """Declaring an episodic skill is what makes ``reset`` normalise ``active``.

        The runtime calls this between episodes, so a bundle that can kick starts
        every episode from ``default_skill`` whether or not the previous episode
        ended mid-behaviour. Gating the normalisation on a *running* episode
        instead would leave the two cases disagreeing for no reason a caller can
        see.
        """
        bundle = _bundle(
            episodic_skills={KICK_LEFT: KICK_DURATION_S},
            default_skill=DEFAULT_IDLE,
        )
        bundle.switch(DEFAULT_MOVE)
        assert bundle.episodic_active is None
        bundle.reset()
        assert bundle.active == DEFAULT_IDLE

    def test_reset_mid_episode_clears_the_episode_and_returns_to_default(self) -> None:
        bundle = _bundle(
            episodic_skills={KICK_LEFT: KICK_DURATION_S},
            default_skill=DEFAULT_MOVE,
        )
        bundle.trigger(KICK_LEFT)
        _tick(bundle)
        bundle.reset()
        assert bundle.episodic_active is None
        assert bundle.active == DEFAULT_MOVE


class TestABundleWithNoEpisodicSkillsIsUntouched:
    """The compatibility claim: declaring no episodic skill changes nothing.

    ``reset`` is the seam where that claim is easiest to break, because the
    episodic FSM and the active skill are cleared together and only one of them
    belongs to the new feature. The rollout calls ``policy.reset(seed=...)``
    once per episode, so a bundle that normalised the active skill regardless of
    whether it declared an episodic behaviour would discard an explicit
    :meth:`switch` at every episode boundary of a multi-episode run - a caller
    who never asked for a timer would find episode two running a skill they had
    switched away from, with nothing reported.
    """

    def test_a_switched_skill_survives_a_reset_when_no_episodic_skill_is_declared(self) -> None:
        bundle = _bundle(active=DEFAULT_MOVE)
        bundle.switch(DEFAULT_IDLE)
        bundle.reset()
        assert bundle.active == DEFAULT_IDLE

    def test_the_velocity_gate_does_not_change_that(self) -> None:
        """The gate reads ``active``; it must not be handed a reverted one."""
        bundle = MicroduckPolicyBundle(
            {DEFAULT_MOVE: _policy(), DEFAULT_IDLE: _policy()},
            active=DEFAULT_MOVE,
            switch_on_velocity=0.1,
        )
        bundle.set_control_frequency(CONTROL_HZ)
        bundle.switch(DEFAULT_IDLE)
        bundle.reset(seed=7)
        assert bundle.active == DEFAULT_IDLE

    def test_reset_still_reaches_every_child(self) -> None:
        """The gate must scope the episodic block, not the child reset.

        ``MicroduckPolicy.reset`` clears ``_last_action``, so a child that has
        run carries one until the bundle forwards the reset. Asserting only that
        the next tick returns actions is too weak - it does that either way,
        because ``_ensure_config`` rebuilds lazily.
        """
        walk = _policy()
        bundle = MicroduckPolicyBundle({DEFAULT_MOVE: walk, DEFAULT_IDLE: _policy()}, active=DEFAULT_MOVE)
        bundle.set_control_frequency(CONTROL_HZ)
        _tick(bundle)
        assert walk._last_action is not None
        bundle.reset(seed=7)
        assert walk._last_action is None
