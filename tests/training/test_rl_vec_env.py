"""Unit tests for ``VecSimEnv`` - the N-env vectorized wrapper over ``SimEnv``.

CPU-only, fake engine. Pins: batch shapes (N, D), per-env independent reset,
autoreset + terminal_obs capture (load-bearing for GAE bootstrap), homogeneity
guard, num_envs=1 degenerate path, and executor lifecycle.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from strands_robots.training.rl import SimEnv, VecSimEnv  # noqa: E402 - after torch importorskip


class _CountdownEngine:
    """Fake engine whose joint ``J`` counts steps; lets us script per-env dones.

    ``J`` starts at 0 and increments by 1 each step. With max_episode_steps=K
    the SimEnv times out after K steps -> done. Each engine instance is
    independent, so different envs advance independently.
    """

    def __init__(self) -> None:
        self._j = 0.0

    def list_robots(self) -> list[str]:
        return ["fake"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return ["J"]

    def reset(self) -> dict:
        self._j = 0.0
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images: bool = False) -> dict:
        return {"J": self._j, "J.vel": 1.0}

    def send_action(self, action, robot_name=None, n_substeps: int = 1) -> dict:
        self._j += 1.0
        return {"status": "success"}


def _factory(max_steps=3):  # type: ignore[no-untyped-def]
    def make():  # type: ignore[no-untyped-def]
        return SimEnv(
            _CountdownEngine(),
            actor_obs_keys=["J", "J.vel"],
            reward_terms=[lambda e: 1.0],
            action_dim=1,
            max_episode_steps=max_steps,
        )

    return make


def test_vec_reset_shapes() -> None:
    vec = VecSimEnv(_factory(), num_envs=4)
    assert vec.num_envs == 4
    assert vec.num_actor_obs == 2
    assert vec.num_actions == 1
    obs = vec.reset()
    assert obs["actor_obs"].shape == (4, 2)
    assert obs["critic_obs"].shape == (4, 2)
    vec.close()


def test_vec_step_shapes() -> None:
    vec = VecSimEnv(_factory(), num_envs=4)
    vec.reset()
    actions = torch.zeros(4, 1)
    obs, rewards, dones, infos = vec.step(actions)
    assert obs["actor_obs"].shape == (4, 2)
    assert rewards.shape == (4,)
    assert dones.shape == (4,)
    assert len(infos) == 4
    vec.close()


def test_autoreset_and_terminal_obs_capture() -> None:
    """On done, terminal_obs is captured and obs is the fresh post-reset obs."""
    vec = VecSimEnv(_factory(max_steps=2), num_envs=2)
    vec.reset()  # J=0 in both
    a = torch.zeros(2, 1)
    # Step 1: J -> 1, not done (max_steps=2).
    _o, _r, dones, infos = vec.step(a)
    assert not bool(dones.any())
    assert all("terminal_obs" not in inf for inf in infos)
    # Step 2: J -> 2, time-out done for BOTH envs.
    obs, _r, dones, infos = vec.step(a)
    assert bool(dones.all()), "both envs should time out at step 2"
    for inf in infos:
        assert "terminal_obs" in inf, "terminal obs must be captured on done"
        assert inf["time_out"] is True
        assert inf["terminated"] is False
        # terminal obs J should be 2 (pre-reset), fresh obs should be 0.
        assert float(inf["terminal_obs"]["actor_obs"][0, 0].item()) == pytest.approx(2.0)
    # Returned obs is the fresh (post-reset) observation: J back to 0.
    assert float(obs["actor_obs"][0, 0].item()) == pytest.approx(0.0)
    vec.close()


def test_per_env_independent_dones() -> None:
    """Envs with different max_steps must finish independently (no cross-contamination)."""
    # Build envs with staggered horizons by alternating factories.
    envs_specs = [2, 5]
    idx = {"i": 0}

    def make():  # type: ignore[no-untyped-def]
        ms = envs_specs[idx["i"] % len(envs_specs)]
        idx["i"] += 1
        return SimEnv(
            _CountdownEngine(),
            actor_obs_keys=["J", "J.vel"],
            reward_terms=[lambda e: 1.0],
            action_dim=1,
            max_episode_steps=ms,
        )

    vec = VecSimEnv(make, num_envs=2)
    vec.reset()
    a = torch.zeros(2, 1)
    vec.step(a)  # J=1
    _o, _r, dones, _infos = vec.step(a)  # J=2: env0 (ms=2) done, env1 (ms=5) not
    assert bool(dones[0].item()) is True
    assert bool(dones[1].item()) is False
    vec.close()


def test_num_envs_one_uses_no_executor() -> None:
    vec = VecSimEnv(_factory(), num_envs=1)
    assert vec._executor is None  # degenerate path: no thread pool
    obs = vec.reset()
    assert obs["actor_obs"].shape == (1, 2)
    vec.close()


def test_rejects_bad_num_envs() -> None:
    # Matched on the parameter the refusal names, not on wording, so the shared
    # count domain can own the message (see test_vec_env_counts_take_the_shared_domain).
    with pytest.raises(ValueError, match="num_envs"):
        VecSimEnv(_factory(), num_envs=0)


def test_rejects_heterogeneous_envs() -> None:
    """Sub-envs with mismatched dims must be rejected at construction."""
    flip = {"n": 0}

    def make():  # type: ignore[no-untyped-def]
        # First env has 2 actor obs, second has 1 -> mismatch.
        keys = ["J", "J.vel"] if flip["n"] == 0 else ["J"]
        flip["n"] += 1
        return SimEnv(
            _CountdownEngine(),
            actor_obs_keys=keys,
            reward_terms=[lambda e: 1.0],
            action_dim=1,
            max_episode_steps=3,
        )

    with pytest.raises(ValueError, match="differ from env 0"):
        VecSimEnv(make, num_envs=2)


def test_step_rejects_wrong_action_batch() -> None:
    vec = VecSimEnv(_factory(), num_envs=3)
    vec.reset()
    with pytest.raises(ValueError, match="!= num_envs"):
        vec.step(torch.zeros(2, 1))  # batch 2 != 3
    vec.close()
