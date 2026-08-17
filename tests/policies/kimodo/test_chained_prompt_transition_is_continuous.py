"""Chaining prompts through KimodoPolicy stays continuous at the segment seam.

Changing the prompt mid-rollout is the documented way to drive a long-horizon
sequence: each new prompt samples the next segment and the stream continues.
Kimodo samples every motion from its own canonical start pose, though, so a
fresh segment's first frame bears no relation to the pose the previous segment
left the robot in. Emitted unmodified it steps every joint at once - a reference
no tracker can follow, reported as a successful rollout.

The policy therefore eases a new segment off the pose last commanded across
``config.transition_frames`` native frames, the same transition length Kimodo's
own sampler applies to a multi-prompt sequence.

The stub agent below returns a DIFFERENT motion per prompt, with a deliberate
pose offset between them: a stub that returned the same frames for every prompt
could not observe a seam at all.

``native_fps == tracker_fps`` throughout so one native frame is one emitted
frame and the transition window is countable directly; the upsample path is
covered separately.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from strands_robots.policies.kimodo import KIMODO_G1_JOINTS, KimodoConfig, KimodoPolicy

_NUM_JOINTS = len(KIMODO_G1_JOINTS)
_ROOT = 7

# Two prompts whose motions sit far apart in joint space, so the seam between
# them is large enough to measure and not an artefact of rounding.
_WALK = "a person walking forward with confident strides"
_REACH = "a person crouching down to pick an object off the floor"


class _TwoMotionAgent:
    """Return a distinct, slowly-varying motion per prompt.

    Each motion oscillates gently about its own centre, so within a segment the
    per-frame change is small while the offset BETWEEN the two centres is large.
    That separation is what makes a seam step distinguishable from ordinary
    motion.
    """

    #: Joint-space centre of each motion, in radians.
    CENTRES = {_WALK: 0.10, _REACH: -1.40}

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def sample(self, prompt, num_frames, diffusion_steps, guidance_scale, seed):
        self.prompts.append(prompt)
        centre = self.CENTRES[prompt]
        out = np.zeros((num_frames, _ROOT + _NUM_JOINTS), dtype=np.float32)
        out[:, 3] = 1.0  # identity quaternion, wxyz
        # Amplitude and rate chosen so one frame's motion is roughly a tenth of
        # the gap between the two centres, the ratio real Kimodo motions show.
        phase = np.linspace(0.0, 4.0 * np.pi, num_frames, dtype=np.float32)
        for j in range(_NUM_JOINTS):
            out[:, _ROOT + j] = centre + 0.30 * np.sin(phase + 0.1 * j)
        return out


def _start_pose(prompt: str) -> np.ndarray:
    """The joint pose a motion begins in, per the sampler.

    Uses a throwaway agent: sampling from the agent under observation would
    register as a sampler run and corrupt the call-count assertions.
    """
    return _TwoMotionAgent().sample(prompt, 40, 100, 7.5, None)[0, _ROOT:].astype(np.float64)


def _seam_gap(last_commanded: np.ndarray, next_prompt: str) -> float:
    """The pose gap a seam has to absorb.

    Measured against the pose actually last commanded rather than the previous
    motion's start pose: a segment ends wherever its own motion took it, so the
    gap at a seam is generally wider than the distance between two start poses.
    """
    return float(np.abs(_start_pose(next_prompt) - last_commanded).max())


def _policy(**overrides):
    agent = _TwoMotionAgent()
    policy = KimodoPolicy(
        config=KimodoConfig(num_frames=40, native_fps=30, tracker_fps=30, **overrides),
        motion_agent=agent,
    )
    policy.set_robot_state_keys(list(KIMODO_G1_JOINTS))
    return policy, agent


def _joint_vector(action: dict[str, float]) -> np.ndarray:
    return np.asarray([action[name] for name in KIMODO_G1_JOINTS], dtype=np.float64)


def _drive(policy, prompts_per_tick) -> np.ndarray:
    """Run one tick per entry and return the commanded joint vectors."""
    return np.asarray([_joint_vector(asyncio.run(policy.get_actions({}, p))[0]) for p in prompts_per_tick])


def _per_tick_steps(commands: np.ndarray) -> np.ndarray:
    """Largest single-joint change between consecutive ticks."""
    return np.abs(np.diff(commands, axis=0)).max(axis=1)


def test_a_prompt_change_spreads_the_pose_offset_instead_of_stepping_it() -> None:
    """The seam must carry a fraction of the pose gap, not the whole of it.

    Two motions start in unrelated poses, so switching between them has a pose
    gap to absorb. The guarantee is that the gap is spread across
    ``transition_frames`` rather than commanded in a single tick, which bounds
    the seam step at roughly ``gap / (transition_frames + 1)``.

    Asserting instead that the seam falls inside the sampled motion's own
    per-tick envelope would be a weaker claim than it looks: whether that holds
    depends on how fast the particular motions move relative to the gap between
    them, so it can pass on one corpus and fail on another for no change in
    behaviour. The bound below is a property of the transition itself.

    Deliberately runs on the default configuration and compares two measured
    quantities, so it reports the size of the step rather than the absence of a
    setting.
    """
    policy, _ = _policy()
    segment = 15
    commands = _drive(policy, [_WALK] * segment + [_REACH] * segment)

    gap = _seam_gap(commands[segment - 1], _REACH)
    seam = _per_tick_steps(commands)[segment - 1]

    assert seam <= gap / 2.0, (
        f"changing the prompt stepped a joint {seam:.4f} rad in a single tick out of "
        f"a {gap:.4f} rad pose gap ({100 * seam / gap:.0f}% of it): a fresh sample is "
        "being emitted from its own start pose instead of being eased off the pose "
        "last commanded"
    )


def test_the_seam_shrinks_as_the_transition_lengthens() -> None:
    """``transition_frames`` is the knob that spreads the offset, so it must bite."""
    seams = {}
    for frames in (2, 5, 20):
        policy, _ = _policy(transition_frames=frames)
        segment = 25
        commands = _drive(policy, [_WALK] * segment + [_REACH] * segment)
        seams[frames] = _per_tick_steps(commands)[segment - 1]

    assert seams[2] > seams[5] > seams[20], (
        f"a longer transition must spread the same pose offset over more ticks, so each step is smaller; got {seams}"
    )


def test_the_segment_returns_to_the_sampler_s_own_frames_after_the_transition() -> None:
    """Only the transition window is touched - the motion is not rescaled.

    Easing must absorb the starting offset and then get out of the way, so the
    frames after the window are exactly what the sampler produced. Anything else
    would distort the motion the caller asked for.
    """
    transition = 5
    segment = 30
    chained, _ = _policy(transition_frames=transition)
    chained_cmds = _drive(chained, [_WALK] * segment + [_REACH] * segment)

    # The same segment sampled with nothing before it: the sampler's own frames.
    virgin, _ = _policy(transition_frames=transition)
    virgin_cmds = _drive(virgin, [_REACH] * segment)

    eased = chained_cmds[segment:]
    np.testing.assert_allclose(
        eased[transition:],
        virgin_cmds[transition:],
        atol=1e-6,
        err_msg="frames past the transition window must be the sampler's own",
    )
    assert not np.allclose(eased[0], virgin_cmds[0], atol=1e-3), (
        "the first frame of the second segment should have been pulled toward the "
        "pose last commanded, not emitted at the motion's own start pose"
    )


def test_the_transition_preserves_the_motion_s_own_velocity() -> None:
    """The ease decays a pose offset; it must not stall the motion.

    Blending toward a static pose would flatten the first frames to near-zero
    velocity. Decaying the offset instead keeps each frame's own advance, which
    is what makes the transition look like motion rather than a pause.
    """
    transition = 8
    segment = 30
    chained, _ = _policy(transition_frames=transition)
    chained_cmds = _drive(chained, [_WALK] * segment + [_REACH] * segment)
    virgin, _ = _policy(transition_frames=transition)
    virgin_cmds = _drive(virgin, [_REACH] * segment)

    eased_speed = _per_tick_steps(chained_cmds[segment : segment + transition])
    own_speed = _per_tick_steps(virgin_cmds[:transition])
    assert eased_speed.min() > 0.2 * own_speed.min(), (
        "the eased frames advance far more slowly than the motion itself, so the "
        f"transition is stalling rather than decaying an offset: {eased_speed.min():.5f} "
        f"vs {own_speed.min():.5f} rad/tick"
    )


def test_chaining_many_segments_eases_every_seam_not_just_the_first() -> None:
    """A long-horizon chain must be continuous at every seam, not just the first."""
    policy, agent = _policy()
    segment = 12
    chain = [_WALK, _REACH, _WALK, _REACH, _WALK]
    commands = _drive(policy, [p for p in chain for _ in range(segment)])

    steps = _per_tick_steps(commands)

    assert len(agent.prompts) == len(chain), "each prompt change should sample once"
    for index, next_prompt in enumerate(chain[1:], start=1):
        seam = index * segment - 1
        gap = _seam_gap(commands[seam], next_prompt)
        assert steps[seam] <= gap / 2.0, (
            f"seam {index} stepped {steps[seam]:.4f} rad of a {gap:.4f} rad pose gap "
            f"({100 * steps[seam] / gap:.0f}% of it), so it is not being eased"
        )


# --------------------------------------------------------------------------
# Controls: behaviour that must NOT change.
# --------------------------------------------------------------------------
def test_the_first_segment_of_a_rollout_is_the_sampler_s_own_motion() -> None:
    """With nothing commanded yet there is no seam, so nothing is eased."""
    policy, _ = _policy()
    segment = 20
    commands = _drive(policy, [_WALK] * segment)

    expected = _TwoMotionAgent().sample(_WALK, 40, 100, 7.5, None)[:segment, _ROOT:]
    np.testing.assert_allclose(
        commands,
        expected.astype(np.float64),
        atol=1e-6,
        err_msg="a single-prompt rollout must be untouched by the transition logic",
    )


def test_an_episode_boundary_opens_on_the_motion_s_own_start_pose() -> None:
    """``reset`` ends the episode, so the next one is not eased onto its last pose.

    Episodes are independent. Forgetting the commanded pose is not enough on its
    own: a buffer already eased onto it would still be replayed from frame 0,
    opening the new episode on a transition built for the previous one.
    """
    policy, _ = _policy()
    _drive(policy, [_WALK] * 15 + [_REACH] * 5)  # chain, so the buffer is eased
    policy.reset()
    reopened = _drive(policy, [_REACH] * 5)

    virgin, _ = _policy()
    expected = _drive(virgin, [_REACH] * 5)
    np.testing.assert_array_equal(
        reopened,
        expected,
        err_msg="a new episode must open on the motion's own start pose",
    )


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, float("nan")])
def test_transition_frames_outside_the_domain_is_refused(bad: object) -> None:
    """Kimodo's sampler refuses ``num_transition_frames < 1``; so does this knob."""
    with pytest.raises(ValueError, match="transition_frames"):
        KimodoConfig(transition_frames=bad)  # type: ignore[arg-type]


def test_transition_frames_reaches_the_config_as_a_constructor_override() -> None:
    """The knob is a real constructor parameter, like every advertised field."""
    policy, _ = _policy()
    assert policy.config.transition_frames == 5, "default matches Kimodo's own length"
    assert KimodoPolicy(motion_agent=_TwoMotionAgent(), transition_frames=11).config.transition_frames == 11
