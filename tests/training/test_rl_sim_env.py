"""Unit tests for ``SimEnv`` - the single-environment RL wrapper over a ``SimEngine``.

CPU-only, fake engine. Pins the construction-time validation contract (obs keys,
reward terms, substeps, action-dim inference / requirement) and the reset
lifecycle hooks (custom ``reset_fn``, stateful reward-term ``reset``), plus the
``close`` no-op contract that keeps ``SimEnv`` interface-compatible with
``VecSimEnv`` / ``GymSimEnv``. These are behaviors the RL trainers rely on: a
typo in obs keys or a missing action dim must fail loudly at construction, not
mid-rollout.
"""

from __future__ import annotations

from typing import cast

import pytest

torch = pytest.importorskip("torch")

from strands_robots.simulation.base import SimEngine  # noqa: E402 - after torch importorskip
from strands_robots.training.rl import SimEnv  # noqa: E402 - after torch importorskip


class _OneJointEngine:
    """Minimal fake engine with a single robot ``fake`` and one joint ``J``."""

    def list_robots(self) -> list[str]:
        return ["fake"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return ["J"]

    def robot_action_keys(self, robot_name: str) -> list[str]:
        # These fakes are duck-typed rather than ``SimEngine`` subclasses, so
        # they do not inherit the default that mirrors the joint names. This
        # robot's one joint is its one actuator, so the two vocabularies agree -
        # which is the shape ``SimEnv`` sizes its action head from.
        return ["J"]

    def reset(self) -> dict:
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images: bool = False) -> dict:
        return {"J": 0.0, "J.vel": 1.0}

    def send_action(self, action, robot_name=None, n_substeps: int = 1) -> dict:
        return {"status": "success"}


class _NoRobotEngine(_OneJointEngine):
    """Fake engine with no registered robots (``list_robots`` is empty)."""

    def list_robots(self) -> list[str]:
        return []


def _engine(obj: object) -> SimEngine:
    """Present a duck-typed fake as a ``SimEngine`` for the ``SimEnv`` constructor.

    ``SimEngine`` is a nominal ABC, but ``SimEnv`` only touches the handful of
    methods the fakes implement (list_robots / robot_joint_names /
    robot_action_keys / reset / get_observation / send_action), so the cast is
    safe for these unit tests.
    """
    return cast(SimEngine, obj)


def test_rejects_empty_actor_obs_keys() -> None:
    with pytest.raises(ValueError, match="actor_obs_keys"):
        SimEnv(_engine(_OneJointEngine()), actor_obs_keys=[], reward_terms=[lambda e: 1.0], action_dim=1)


def test_rejects_empty_reward_terms() -> None:
    with pytest.raises(ValueError, match="reward_terms"):
        SimEnv(_engine(_OneJointEngine()), actor_obs_keys=["J"], reward_terms=[], action_dim=1)


def test_rejects_nonpositive_n_substeps() -> None:
    with pytest.raises(ValueError, match="n_substeps"):
        SimEnv(
            _engine(_OneJointEngine()),
            actor_obs_keys=["J"],
            reward_terms=[lambda e: 1.0],
            action_dim=1,
            n_substeps=0,
        )


def test_infers_action_dim_from_robot_action_keys() -> None:
    # No action_dim given -> derived from the robot's action-key count, which
    # send_action binds a vector against (one actuator -> 1).
    env = SimEnv(_engine(_OneJointEngine()), actor_obs_keys=["J"], reward_terms=[lambda e: 1.0])
    assert env.num_actions == 1


def test_requires_action_dim_when_no_robot() -> None:
    with pytest.raises(ValueError, match="action_dim must be given"):
        SimEnv(_engine(_NoRobotEngine()), actor_obs_keys=["J"], reward_terms=[lambda e: 1.0])


def test_reset_fn_invoked_instead_of_engine_reset() -> None:
    calls: dict[str, int] = {"reset_fn": 0, "engine_reset": 0}

    class _TrackingEngine(_OneJointEngine):
        def reset(self) -> dict:
            calls["engine_reset"] += 1
            return {"status": "success"}

    def reset_fn(engine: SimEngine) -> None:
        calls["reset_fn"] += 1

    env = SimEnv(
        _engine(_TrackingEngine()),
        actor_obs_keys=["J"],
        reward_terms=[lambda e: 1.0],
        action_dim=1,
        reset_fn=reset_fn,
    )
    env.reset()
    assert calls["reset_fn"] == 1
    # Custom reset_fn takes over: the engine's own reset() must not be called.
    assert calls["engine_reset"] == 0


def test_stateful_reward_term_reset_called_on_reset() -> None:
    class _StatefulTerm:
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

        def __call__(self, engine: SimEngine) -> float:
            return 1.0

    term = _StatefulTerm()
    env = SimEnv(_engine(_OneJointEngine()), actor_obs_keys=["J"], reward_terms=[term], action_dim=1)
    # Construction does not reset the term; the first reset() does.
    assert term.reset_calls == 0
    env.reset()
    assert term.reset_calls == 1


def test_close_is_noop() -> None:
    env = SimEnv(_engine(_OneJointEngine()), actor_obs_keys=["J"], reward_terms=[lambda e: 1.0], action_dim=1)
    # close() owns no resources (engine lifecycle is the caller's); it must not raise.
    env.close()


# --- termination vs truncation classification (the SAC/PPO bootstrap contract) ---
#
# ``step`` MUST report a time-out (episode-length limit) and a genuine success
# terminal as DISTINCT events: a time-out is a truncation whose successor value
# is still bootstrapped, while a success is a terminal that stops the value
# backup. Collapsing the two -- surfacing a time-out as ``terminated`` -- silently
# breaks the off-policy target (the FastTD3 truncation-bootstrap bug, fixed
# upstream Jun 2025). These pin that the flags never collapse.


def test_step_timeout_is_truncation_not_terminal() -> None:
    """A time-out sets done=1 but reports time_out (bootstrappable), NOT terminated."""
    # max_episode_steps=1 -> the first step is a time-out; no success_fn -> never a terminal.
    env = SimEnv(
        _engine(_OneJointEngine()),
        actor_obs_keys=["J"],
        reward_terms=[lambda e: 1.0],
        action_dim=1,
        max_episode_steps=1,
    )
    env.reset()
    _, _, done, info = env.step(torch.zeros(1, 1))
    assert float(done.reshape(-1)[0]) == 1.0  # episode ends (the caller resets)
    assert info["time_out"] is True  # ...but as a truncation -> bootstrap the value
    assert info["terminated"] is False  # a time-out is NOT a terminal state


def test_step_success_is_terminal_not_truncation() -> None:
    """A success_fn hit sets terminated (no bootstrap), NOT time_out; disproves always-False."""
    # success on the first step, with head-room before the time-out limit so the
    # two conditions are unambiguously separable.
    env = SimEnv(
        _engine(_OneJointEngine()),
        actor_obs_keys=["J"],
        reward_terms=[lambda e: 1.0],
        action_dim=1,
        max_episode_steps=99,
        success_fn=lambda e: True,
    )
    env.reset()
    _, _, done, info = env.step(torch.zeros(1, 1))
    assert float(done.reshape(-1)[0]) == 1.0
    assert info["terminated"] is True  # genuine terminal -> stop the value backup
    assert info["time_out"] is False  # not a truncation


def test_step_truncation_boundary_is_exact() -> None:
    """time_out fires exactly at step == max_episode_steps, not the step before."""
    env = SimEnv(
        _engine(_OneJointEngine()),
        actor_obs_keys=["J"],
        reward_terms=[lambda e: 1.0],
        action_dim=1,
        max_episode_steps=2,
    )
    env.reset()
    _, _, done1, info1 = env.step(torch.zeros(1, 1))
    assert float(done1.reshape(-1)[0]) == 0.0  # step 1 of 2 -> not yet a time-out
    assert info1["time_out"] is False
    _, _, done2, info2 = env.step(torch.zeros(1, 1))
    assert float(done2.reshape(-1)[0]) == 1.0  # step 2 == limit -> time-out
    assert info2["time_out"] is True
