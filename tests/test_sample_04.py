"""Tests for Sample 04 — Gymnasium Training.

Tests cover:
- Custom reward functions (pure numpy, no GPU/sim needed)
- Gym basics demo (demo mode fallback)
- PPO trainer config (mocked, no GPU needed)
- Evaluate script structure
"""

# Skip if samples/ directory not present (requires PR #13)
import os as _os
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

if not _os.path.isdir(_os.path.join(_os.path.dirname(__file__), "..", "samples")):
    __import__("pytest").skip("Requires PR #13 (samples)", allow_module_level=True)


# ── Reward function imports ──────────────────────────────────────────────
# The sample files import strands_robots.envs.StrandsSimEnv at module scope
# with a try/except fallback.  We need to import from the sample modules
# which use `from strands_robots.envs import StrandsSimEnv`.


@pytest.fixture(autouse=True)
def _isolate():
    """Remove sample modules from sys.modules between tests."""
    keys_before = set(sys.modules.keys())
    yield
    for key in set(sys.modules.keys()) - keys_before:
        if "samples" in key or "sample_04" in key:
            sys.modules.pop(key, None)


# ---------------------------------------------------------------------------
# Helper: load sample module from file path
# ---------------------------------------------------------------------------
def _load_sample(filename: str):
    """Import a sample script by file path, handling import errors gracefully."""
    import importlib.util

    path = f"samples/04_gymnasium_training/{filename}"
    spec = importlib.util.spec_from_file_location(f"sample_04_{filename[:-3]}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ===========================================================================
# custom_reward.py — Pure numpy, fully CPU-testable
# ===========================================================================


class TestReachTargetReward:
    """Test reach_target_reward function."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load_sample("custom_reward.py")
        self.fn = self.mod.reach_target_reward

    def test_near_target_higher_reward(self):
        """Closer to target → higher reward."""
        obs_near = np.array([0.95, 0.48, 0.32, 0, 0, 0, 0.1, 0.1, 0.05, 0, 0, 0])
        obs_far = np.array([0.0, 0.0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        action = np.zeros(6)
        info = {}

        r_near = self.fn(obs_near, action, info)
        r_far = self.fn(obs_far, action, info)
        assert r_near > r_far

    def test_at_target_bonus(self):
        """Being within 10cm of target gives bonus ≥ 10."""
        obs_at = np.array([1.0, 0.5, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        r = self.fn(obs_at, np.zeros(6), {})
        assert r >= 10.0

    def test_far_from_target_negative(self):
        """Far from target → negative reward."""
        obs_far = np.array([10.0, 10.0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        r = self.fn(obs_far, np.zeros(6), {})
        assert r < 0

    def test_returns_float(self):
        """Always returns a float."""
        obs = np.array([0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        r = self.fn(obs, np.zeros(6), {})
        assert isinstance(r, float)


class TestSmoothMotionReward:
    """Test smooth_motion_reward function."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load_sample("custom_reward.py")
        self.fn = self.mod.smooth_motion_reward

    def test_slow_motion_higher_reward(self):
        """Low velocities → higher reward than high velocities."""
        obs_slow = np.array([0.5, 0.5, 0.4, 0, 0, 0, 0.1, 0.1, 0.05, 0.08, 0.06, 0.09])
        obs_fast = np.array([0.5, 0.5, 0.4, 0, 0, 0, 2.5, 3.0, 1.5, 2.0, 1.8, 2.2])
        action = np.array([0.1, -0.2, 0.3, 0.0, 0.1, -0.1])
        info = {}

        r_slow = self.fn(obs_slow, action, info)
        r_fast = self.fn(obs_fast, action, info)
        assert r_slow > r_fast

    def test_zero_velocity_base_reward(self):
        """Zero velocity and zero action → base reward (0.1)."""
        obs_zero = np.zeros(12)
        r = self.fn(obs_zero, np.zeros(6), {})
        assert r == pytest.approx(0.1, abs=0.001)

    def test_large_action_penalized(self):
        """Large actions are penalized more than small actions."""
        obs = np.zeros(12)
        small_action = np.ones(6) * 0.01
        large_action = np.ones(6) * 10.0

        r_small = self.fn(obs, small_action, {})
        r_large = self.fn(obs, large_action, {})
        assert r_small > r_large

    def test_short_obs_fallback(self):
        """Works with short observation vectors (uses fallback)."""
        obs_short = np.array([0.5, 0.5, 0.4, 0.1, 0.1, 0.05])
        r = self.fn(obs_short, np.zeros(6), {})
        assert isinstance(r, float)


class TestStayUprightReward:
    """Test stay_upright_reward function."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load_sample("custom_reward.py")
        self.fn = self.mod.stay_upright_reward

    def test_fallen_large_penalty(self):
        """z < 0.3 → -10.0 penalty."""
        obs_fallen = np.array([0.5, 0.5, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        r = self.fn(obs_fallen, np.zeros(6), {})
        assert r == -10.0

    def test_upright_positive(self):
        """z ≥ 0.3 → positive reward."""
        obs_up = np.array([0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        r = self.fn(obs_up, np.zeros(6), {})
        assert r > 0

    def test_optimal_height_bonus(self):
        """Optimal height (0.5) gets maximum bonus."""
        obs_optimal = np.array([0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        obs_high = np.array([0.5, 0.5, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        r_optimal = self.fn(obs_optimal, np.zeros(6), {})
        r_high = self.fn(obs_high, np.zeros(6), {})
        assert r_optimal >= r_high

    def test_short_obs(self):
        """Works with short observation vector."""
        obs_short = np.array([0.5, 0.5])
        r = self.fn(obs_short, np.zeros(6), {})
        assert isinstance(r, float)


class TestCombinedReward:
    """Test combined_reward function."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _load_sample("custom_reward.py")
        self.fn = self.mod.combined_reward

    def test_returns_float(self):
        """Combined reward returns a float."""
        obs = np.array([0.5, 0.5, 0.5, 0, 0, 0, 0.1, 0.1, 0.05, 0, 0, 0])
        r = self.fn(obs, np.zeros(6), {})
        assert isinstance(r, float)

    def test_at_target_upright_slow_best(self):
        """Best case: at target, upright, slow motion → highest reward."""
        obs_best = np.array([1.0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        obs_worst = np.array([10.0, 10.0, 0.1, 0, 0, 0, 5, 5, 5, 5, 5, 5])
        action = np.zeros(6)

        r_best = self.fn(obs_best, action, {})
        r_worst = self.fn(obs_worst, action, {})
        assert r_best > r_worst

    def test_weighted_sum_structure(self):
        """Combined is a weighted sum of the 3 components."""
        obs = np.array([0.5, 0.5, 0.5, 0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0])
        action = np.zeros(6)
        info = {}

        r_combined = self.fn(obs, action, info)
        r_target = self.mod.reach_target_reward(obs, action, info)
        r_smooth = self.mod.smooth_motion_reward(obs, action, info)
        r_upright = self.mod.stay_upright_reward(obs, action, info)

        expected = 1.0 * r_target + 0.3 * r_smooth + 0.5 * r_upright
        assert r_combined == pytest.approx(expected, abs=0.001)


class TestDemoFunctions:
    """Test demo functions run without errors in demo mode."""

    def test_demo_reward_functions_runs(self, capsys):
        """demo_custom_reward_functions completes without error."""
        mod = _load_sample("custom_reward.py")
        mod.demo_custom_reward_functions()
        captured = capsys.readouterr()
        assert "Custom Reward Functions" in captured.out

    def test_demo_env_demo_mode(self, capsys):
        """demo_custom_reward_with_env runs in demo mode when StrandsSimEnv not available."""
        mod = _load_sample("custom_reward.py")
        mod.STRANDS_AVAILABLE = False
        mod.demo_custom_reward_with_env()
        captured = capsys.readouterr()
        assert "Demo mode" in captured.out


class TestGymBasics:
    """Test gym_basics.py demo mode."""

    def test_demo_mode_runs(self, capsys):
        """demo_gym_basics runs in demo mode."""
        mod = _load_sample("gym_basics.py")
        mod.STRANDS_AVAILABLE = False
        mod.demo_gym_basics()
        captured = capsys.readouterr()
        assert "Demo mode" in captured.out
        assert "Statistics" in captured.out

    def test_print_header(self, capsys):
        """print_header formats correctly."""
        mod = _load_sample("gym_basics.py")
        mod.print_header("Test Title")
        captured = capsys.readouterr()
        assert "Test Title" in captured.out
        assert "=" * 60 in captured.out


class TestTrainPPO:
    """Test train_ppo.py structure (mocked, no GPU)."""

    def test_check_dependencies_without_trainer(self):
        """check_dependencies returns False when trainer not available."""
        mod = _load_sample("train_ppo.py")
        mod.TRAINER_AVAILABLE = False
        assert mod.check_dependencies() is False

    def test_print_header(self, capsys):
        """print_header formats correctly."""
        mod = _load_sample("train_ppo.py")
        mod.print_header("PPO Training")
        captured = capsys.readouterr()
        assert "PPO Training" in captured.out


class TestEvaluate:
    """Test evaluate.py structure (import guards)."""

    def test_evaluate_function_exists(self):
        """evaluate module has expected functions."""
        # evaluate.py imports from strands_robots directly (no try/except)
        # so we mock it
        mock_envs = types.ModuleType("strands_robots")
        mock_envs.StrandsSimEnv = MagicMock

        with patch.dict(
            "sys.modules",
            {
                "strands_robots": mock_envs,
                "strands_robots.envs": mock_envs,
            },
        ):
            mod = _load_sample("evaluate.py")
            assert hasattr(mod, "evaluate")
            assert hasattr(mod, "load_model")

    def test_evaluate_collects_stats(self):
        """evaluate() collects per-episode statistics."""
        mock_envs = types.ModuleType("strands_robots")
        mock_envs.StrandsSimEnv = MagicMock

        with patch.dict(
            "sys.modules",
            {
                "strands_robots": mock_envs,
                "strands_robots.envs": mock_envs,
            },
        ):
            mod = _load_sample("evaluate.py")

            # Mock model and env
            mock_model = MagicMock()
            mock_model.predict.return_value = (np.zeros(6), None)

            mock_env = MagicMock()
            mock_env.reset.return_value = (np.zeros(18), {})
            # Episode of 5 steps then terminate
            mock_env.step.side_effect = [
                (np.zeros(18), 1.0, False, False, {}),
                (np.zeros(18), 0.5, False, False, {}),
                (np.zeros(18), 0.8, False, False, {}),
                (np.zeros(18), 0.3, False, False, {}),
                (np.zeros(18), 0.2, True, False, {"success": True}),
            ] * 10  # repeat for 10 episodes

            result = mod.evaluate(mock_model, mock_env, num_episodes=2)
            assert "mean_reward" in result
            assert "std_reward" in result
            assert result["num_episodes"] == 2
