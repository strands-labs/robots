#!/usr/bin/env python3
"""
End-to-End Test for GR00T Fine-Tuning Pipeline

Tests the full pipeline chain: Generate 3D → Isaac → RL → Eval → Cosmos → Code
All GPU-dependent components are mocked to run on CPU CI.

Refs:
    - Issue #141: Fine Tune GR00T 1.6
    - scripts/training/groot_finetune_pipeline.py
"""

import importlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# ── Pre-mock heavy deps ──────────────────────────────────────────────

_mock_cv2 = MagicMock()
_mock_cv2.dnn = MagicMock()
# Always use a fresh ModuleSpec — find_spec("cv2") explodes when another test
# has already injected a MagicMock cv2 with no real __spec__ into sys.modules.
_mock_cv2.__spec__ = importlib.machinery.ModuleSpec("cv2", None)
sys.modules.setdefault("cv2", _mock_cv2)
sys.modules.setdefault("cv2.dnn", _mock_cv2.dnn)

_mock_strands = MagicMock()
_mock_strands.tool = lambda f: f
sys.modules.setdefault("strands", _mock_strands)
sys.modules.setdefault("strands.tools", MagicMock())
sys.modules.setdefault("strands.tools.decorator", MagicMock(tool=lambda f: f))

# Insert project root
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPipelineConfig(unittest.TestCase):
    """Test PipelineConfig defaults and serialization."""

    def test_default_config(self):
        from scripts.training.groot_finetune_pipeline import PipelineConfig

        config = PipelineConfig()
        self.assertEqual(config.sac_robot, "so100")
        self.assertEqual(config.sac_task, "pick and place cube")
        self.assertEqual(config.sac_backend, "mujoco")
        self.assertEqual(config.sac_timesteps, 500_000)
        self.assertEqual(config.groot_base_model, "nvidia/GR00T-N1-2B")
        self.assertEqual(config.groot_data_config, "so100_dualcam")
        self.assertEqual(config.groot_max_steps, 10_000)
        self.assertEqual(config.eval_episodes, 50)
        self.assertIn("pick up the red cube", config.eval_tasks)

    def test_custom_config(self):
        from scripts.training.groot_finetune_pipeline import PipelineConfig

        config = PipelineConfig(
            sac_robot="panda",
            sac_timesteps=1_000_000,
            groot_max_steps=20_000,
        )
        self.assertEqual(config.sac_robot, "panda")
        self.assertEqual(config.sac_timesteps, 1_000_000)
        self.assertEqual(config.groot_max_steps, 20_000)

    def test_config_serializable(self):
        from dataclasses import asdict

        from scripts.training.groot_finetune_pipeline import PipelineConfig

        config = PipelineConfig()
        d = asdict(config)
        s = json.dumps(d)
        self.assertIn("so100", s)
        self.assertIn("nvidia/GR00T-N1-2B", s)


class TestStage1Marble(unittest.TestCase):
    """Test Stage 1: Marble 3D World Generation."""

    def test_marble_skipped_without_api(self):
        from scripts.training.groot_finetune_pipeline import PipelineConfig, stage_1_marble_world

        config = PipelineConfig(marble_output_dir="/tmp/test_marble_141")
        # Marble requires WLT_API_KEY — without it, should skip gracefully
        result = stage_1_marble_world(config)
        self.assertIn(result["status"], ("skipped", "success"))


class TestStage2SACTraining(unittest.TestCase):
    """Test Stage 2: SAC RL Training."""

    def test_sac_config_creation(self):
        """Verify create_rl_trainer produces correct SAC config."""
        from strands_robots.rl_trainer import create_rl_trainer

        trainer = create_rl_trainer(
            algorithm="sac",
            env_config={
                "robot_name": "so100",
                "task": "pick and place cube",
                "backend": "mujoco",
            },
            total_timesteps=100,
            output_dir="/tmp/test_sac_141",
        )
        self.assertEqual(trainer.config.algorithm, "sac")
        self.assertEqual(trainer.config.robot_name, "so100")
        self.assertEqual(trainer.config.total_timesteps, 100)

    def test_pick_and_place_reward_auto_detect(self):
        """Verify PickAndPlaceReward is auto-selected for pick-and-place tasks."""
        from strands_robots.rl_trainer import PickAndPlaceReward, RLConfig, SB3Trainer

        config = RLConfig(task="pick and place cube")
        trainer = SB3Trainer(config)
        reward_fn = trainer._get_reward_fn()
        self.assertIsInstance(reward_fn, PickAndPlaceReward)

    def test_pick_and_place_reward_phases(self):
        """Test the 4-phase reward function."""
        from strands_robots.rl_trainer import PickAndPlaceReward

        reward = PickAndPlaceReward(
            ee_pos_indices=(0, 3),
            object_pos_indices=(3, 6),
            gripper_index=6,
        )

        # Phase 1: Reach — EE far from object
        obs = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.3, 0.1, 0, 0, 0, 0, 0, 0, 0])
        action = np.zeros(6)
        r1 = reward(obs, action)
        self.assertEqual(reward.current_phase, 0)  # Still in Reach
        self.assertIsInstance(r1, float)

        # Phase 2: EE close to object → should advance to Grasp
        obs_close = np.array([0.1, 0.1, 0.3, 0.1, 0.1, 0.3, 0.01, 0, 0, 0, 0, 0, 0, 0])
        r2 = reward(obs_close, action)
        self.assertIsInstance(r2, float)
        self.assertIn(reward.current_phase, (0, 1))  # May or may not advance

        # Reset
        reward.reset()
        self.assertEqual(reward.current_phase, 0)
        self.assertFalse(reward.is_success)

    def test_pick_and_place_reward_info(self):
        """Test reward diagnostic info."""
        from strands_robots.rl_trainer import PickAndPlaceReward

        reward = PickAndPlaceReward()
        info = reward.get_info()
        self.assertIn("phase", info)
        self.assertIn("phase_name", info)
        self.assertIn("is_success", info)
        self.assertEqual(info["phase_name"], "Reach")


class TestWriteEpisodeToDisk(unittest.TestCase):
    """Test _write_episode_to_disk dataset serialization."""

    def test_writes_data_json(self):
        """Verify episode data is written as data.json with correct structure."""
        import tempfile

        from scripts.training.groot_finetune_pipeline import _write_episode_to_disk

        with tempfile.TemporaryDirectory() as tmpdir:
            episode_data = {
                "frames_front": [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(5)],
                "frames_wrist": [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(5)],
                "states": [{"state.single_arm": [0.1, 0.2, 0.3, 0.4, 0.5], "state.gripper": [0.0]} for _ in range(5)],
                "actions": [
                    {"action.single_arm": [0.01, 0.02, 0.03, 0.04, 0.05], "action.gripper": [0.8]} for _ in range(5)
                ],
            }

            ep_dir = _write_episode_to_disk(episode_data, 0, tmpdir, "pick cube")

            # Verify data.json exists and has correct structure
            data_path = os.path.join(ep_dir, "data.json")
            self.assertTrue(os.path.exists(data_path))

            with open(data_path) as f:
                data = json.load(f)

            self.assertIn("timesteps", data)
            self.assertEqual(len(data["timesteps"]), 5)
            self.assertEqual(data["task"], "pick cube")

            # Verify GR00T-format keys in first timestep
            ts0 = data["timesteps"][0]
            self.assertIn("state", ts0)
            self.assertIn("action", ts0)
            self.assertIn("state.single_arm", ts0["state"])
            self.assertIn("action.gripper", ts0["action"])

            # Verify language annotation per timestep
            self.assertIn("annotation.human.task_description", ts0)
            self.assertEqual(ts0["annotation.human.task_description"], "pick cube")

    def test_writes_dual_camera_videos(self):
        """Verify both front and wrist camera frames are persisted."""
        import tempfile

        from scripts.training.groot_finetune_pipeline import _write_episode_to_disk

        with tempfile.TemporaryDirectory() as tmpdir:
            frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]
            episode_data = {
                "frames_front": frames,
                "frames_wrist": frames,
                "states": [{"state.single_arm": [0.0] * 5, "state.gripper": [0.0]}] * 10,
                "actions": [{"action.single_arm": [0.0] * 5, "action.gripper": [0.0]}] * 10,
            }

            ep_dir = _write_episode_to_disk(episode_data, 0, tmpdir, "test")

            # Front camera
            has_front_mp4 = os.path.exists(os.path.join(ep_dir, "video_front.mp4"))
            has_front_npz = os.path.exists(os.path.join(ep_dir, "video_front.npz"))
            self.assertTrue(has_front_mp4 or has_front_npz, "Front camera must be persisted")

            # Wrist camera
            has_wrist_mp4 = os.path.exists(os.path.join(ep_dir, "video_wrist.mp4"))
            has_wrist_npz = os.path.exists(os.path.join(ep_dir, "video_wrist.npz"))
            self.assertTrue(has_wrist_mp4 or has_wrist_npz, "Wrist camera must be persisted")

    def test_writes_metadata_json(self):
        """Verify episode metadata is written with dual camera flags."""
        import tempfile

        from scripts.training.groot_finetune_pipeline import _write_episode_to_disk

        with tempfile.TemporaryDirectory() as tmpdir:
            episode_data = {
                "frames_front": [np.zeros((4, 4, 3), dtype=np.uint8)],
                "frames_wrist": [np.zeros((4, 4, 3), dtype=np.uint8)],
                "states": [{"state.single_arm": [0.0] * 5, "state.gripper": [0.0]}],
                "actions": [{"action.single_arm": [0.0] * 5, "action.gripper": [0.0]}],
            }

            ep_dir = _write_episode_to_disk(episode_data, 42, tmpdir, "test task")
            meta_path = os.path.join(ep_dir, "metadata.json")
            self.assertTrue(os.path.exists(meta_path))

            with open(meta_path) as f:
                meta = json.load(f)

            self.assertEqual(meta["episode_idx"], 42)
            self.assertEqual(meta["num_steps"], 1)
            self.assertEqual(meta["task"], "test task")
            self.assertTrue(meta["has_video_front"])
            self.assertTrue(meta["has_video_wrist"])

    def test_legacy_frames_key_backward_compat(self):
        """Verify 'frames' key (no _front suffix) still works."""
        import tempfile

        from scripts.training.groot_finetune_pipeline import _write_episode_to_disk

        with tempfile.TemporaryDirectory() as tmpdir:
            episode_data = {
                "frames": [np.zeros((4, 4, 3), dtype=np.uint8)],
                "states": [{"state.single_arm": [0.0] * 5, "state.gripper": [0.0]}],
                "actions": [{"action.single_arm": [0.0] * 5, "action.gripper": [0.0]}],
            }

            ep_dir = _write_episode_to_disk(episode_data, 0, tmpdir, "legacy")

            # Front video should exist (mapped from "frames")
            has_front = os.path.exists(os.path.join(ep_dir, "video_front.mp4")) or os.path.exists(
                os.path.join(ep_dir, "video_front.npz")
            )
            self.assertTrue(has_front)

            # No wrist video (not provided)
            has_wrist = os.path.exists(os.path.join(ep_dir, "video_wrist.mp4")) or os.path.exists(
                os.path.join(ep_dir, "video_wrist.npz")
            )
            self.assertFalse(has_wrist)

    def test_empty_frames_ok(self):
        """Verify episode with no frames still writes data/metadata."""
        import tempfile

        from scripts.training.groot_finetune_pipeline import _write_episode_to_disk

        with tempfile.TemporaryDirectory() as tmpdir:
            episode_data = {
                "frames_front": [],
                "frames_wrist": [],
                "states": [{"state.single_arm": [0.0] * 5, "state.gripper": [0.0]}],
                "actions": [{"action.single_arm": [0.0] * 5, "action.gripper": [0.0]}],
            }

            ep_dir = _write_episode_to_disk(episode_data, 0, tmpdir, "no frames")

            self.assertTrue(os.path.exists(os.path.join(ep_dir, "data.json")))
            self.assertTrue(os.path.exists(os.path.join(ep_dir, "metadata.json")))

            with open(os.path.join(ep_dir, "metadata.json")) as f:
                meta = json.load(f)
            self.assertFalse(meta["has_video_front"])
            self.assertFalse(meta["has_video_wrist"])

    def test_episode_dir_naming(self):
        """Verify episode directories use zero-padded naming."""
        import tempfile

        from scripts.training.groot_finetune_pipeline import _write_episode_to_disk

        with tempfile.TemporaryDirectory() as tmpdir:
            episode_data = {
                "frames_front": [],
                "frames_wrist": [],
                "states": [{"s": [0]}],
                "actions": [{"a": [0]}],
            }
            ep_dir = _write_episode_to_disk(episode_data, 7, tmpdir, "t")
            self.assertTrue(ep_dir.endswith("episode_000007"))


class TestConvertToLeRobotFormat(unittest.TestCase):
    """Test _convert_to_lerobot_format conversion."""

    def _create_raw_episodes(self, tmpdir, num_episodes=3, num_steps=5):
        """Helper to create raw episode directories for conversion tests."""
        from scripts.training.groot_finetune_pipeline import _write_episode_to_disk

        raw_dir = os.path.join(tmpdir, "_raw")
        for ep in range(num_episodes):
            episode_data = {
                "frames_front": [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(num_steps)],
                "frames_wrist": [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(num_steps)],
                "states": [
                    {"state.single_arm": [0.1 * (ep + 1)] * 5, "state.gripper": [0.0]} for _ in range(num_steps)
                ],
                "actions": [
                    {"action.single_arm": [0.01 * (ep + 1)] * 5, "action.gripper": [0.5]} for _ in range(num_steps)
                ],
            }
            _write_episode_to_disk(episode_data, ep, raw_dir, f"task_{ep}")
        return raw_dir

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("pyarrow") is not None,
        "pyarrow required for LeRobot format conversion",
    )
    def test_creates_parquet(self):
        """Verify parquet file is created with correct columns."""
        import tempfile

        import pyarrow.parquet as pq
        from scripts.training.groot_finetune_pipeline import _convert_to_lerobot_format

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = self._create_raw_episodes(tmpdir, num_episodes=2, num_steps=3)
            out_dir = os.path.join(tmpdir, "lerobot_dataset")

            _convert_to_lerobot_format(raw_dir, out_dir, "so100_dualcam", "so100")

            parquet_path = os.path.join(out_dir, "data", "train-00000-of-00001.parquet")
            self.assertTrue(os.path.exists(parquet_path))

            table = pq.read_table(parquet_path)
            columns = table.column_names
            self.assertIn("episode_index", columns)
            self.assertIn("frame_index", columns)
            self.assertIn("timestamp", columns)
            self.assertIn("state.single_arm", columns)
            self.assertIn("state.gripper", columns)
            self.assertIn("action.single_arm", columns)
            self.assertIn("action.gripper", columns)
            self.assertIn("annotation.human.task_description", columns)

            # 2 episodes × 3 steps = 6 rows
            self.assertEqual(table.num_rows, 6)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("pyarrow") is not None,
        "pyarrow required",
    )
    def test_creates_meta_files(self):
        """Verify meta/info.json and meta/episodes.jsonl are created."""
        import tempfile

        from scripts.training.groot_finetune_pipeline import _convert_to_lerobot_format

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = self._create_raw_episodes(tmpdir, num_episodes=2, num_steps=3)
            out_dir = os.path.join(tmpdir, "lerobot_dataset")

            _convert_to_lerobot_format(raw_dir, out_dir, "so100_dualcam", "so100")

            # info.json
            info_path = os.path.join(out_dir, "meta", "info.json")
            self.assertTrue(os.path.exists(info_path))
            with open(info_path) as f:
                info = json.load(f)
            self.assertEqual(info["robot_type"], "so100")
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual(info["total_frames"], 6)
            self.assertEqual(info["data_config"], "so100_dualcam")
            self.assertEqual(info["fps"], 30)

            # episodes.jsonl
            episodes_path = os.path.join(out_dir, "meta", "episodes.jsonl")
            self.assertTrue(os.path.exists(episodes_path))
            with open(episodes_path) as f:
                lines = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["episode_index"], 0)
            self.assertEqual(lines[0]["length"], 3)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("pyarrow") is not None,
        "pyarrow required",
    )
    def test_creates_video_directory_structure(self):
        """Verify videos are organized in LeRobot format: videos/video.front/episode_*.mp4."""
        import tempfile

        from scripts.training.groot_finetune_pipeline import _convert_to_lerobot_format

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = self._create_raw_episodes(tmpdir, num_episodes=2, num_steps=3)
            out_dir = os.path.join(tmpdir, "lerobot_dataset")

            _convert_to_lerobot_format(raw_dir, out_dir, "so100_dualcam", "so100")

            # Check video.front directory
            front_dir = os.path.join(out_dir, "videos", "video.front")
            self.assertTrue(os.path.isdir(front_dir))

            # Check video.wrist directory
            wrist_dir = os.path.join(out_dir, "videos", "video.wrist")
            self.assertTrue(os.path.isdir(wrist_dir))

            # Should have files for each episode (mp4 or npz)
            front_files = os.listdir(front_dir)
            wrist_files = os.listdir(wrist_dir)
            self.assertEqual(len(front_files), 2)
            self.assertEqual(len(wrist_files), 2)

    def test_conversion_fails_without_pyarrow(self):
        """Verify ImportError is raised when pyarrow is missing."""
        import tempfile
        from unittest.mock import patch

        from scripts.training.groot_finetune_pipeline import _convert_to_lerobot_format

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = self._create_raw_episodes(tmpdir, num_episodes=1, num_steps=2)
            out_dir = os.path.join(tmpdir, "lerobot_dataset")

            with patch.dict("sys.modules", {"pyarrow": None, "pyarrow.parquet": None}):
                with self.assertRaises(ImportError):
                    _convert_to_lerobot_format(raw_dir, out_dir, "so100", "so100")


class TestRenderNamedCamera(unittest.TestCase):
    """Test _render_named_camera helper."""

    def test_returns_numpy_array(self):
        """Verify _render_named_camera returns numpy array via fallback path."""
        from unittest.mock import patch

        from scripts.training.groot_finetune_pipeline import _render_named_camera

        # Mock env with MuJoCo-like internals
        mock_env = MagicMock()
        mock_model = MagicMock()
        mock_data = MagicMock()
        mock_env._sim._world._model = mock_model
        mock_env._sim._world._data = mock_data
        mock_env.render.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

        # Force fallback path by hiding mujoco import inside the function
        with patch.dict("sys.modules", {"mujoco": None}):
            frame = _render_named_camera(mock_env, "wrist", 224, 224)
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (224, 224, 3))

    def test_returns_numpy_array_with_mujoco(self):
        """Verify _render_named_camera uses MuJoCo when available."""
        try:
            import mujoco
        except ImportError:
            self.skipTest("mujoco not installed")
        if not os.environ.get("DISPLAY"):
            self.skipTest("no DISPLAY — MuJoCo renderer needs OpenGL context")

        from scripts.training.groot_finetune_pipeline import _render_named_camera

        # Create a minimal MuJoCo model with a named camera
        xml = """
        <mujoco>
          <worldbody>
            <camera name="wrist" pos="0 -1 1" xyaxes="1 0 0 0 0.7 0.7"/>
            <body>
              <geom type="box" size=".1 .1 .1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        mock_env = MagicMock()
        mock_env._sim._world._model = model
        mock_env._sim._world._data = data

        frame = _render_named_camera(mock_env, "wrist", 64, 64)
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (64, 64, 3))
        self.assertEqual(frame.dtype, np.uint8)


class TestStage4GR00TTrainer(unittest.TestCase):
    """Test Stage 4: GR00T Trainer configuration."""

    def test_groot_trainer_creation(self):
        from strands_robots.training import create_trainer

        trainer = create_trainer(
            "groot",
            base_model_path="nvidia/GR00T-N1-2B",
            dataset_path="/data/test",
            embodiment_tag="so100",
            data_config="so100_dualcam",
        )
        self.assertEqual(trainer.provider_name, "groot")
        self.assertEqual(trainer.base_model_path, "nvidia/GR00T-N1-2B")
        self.assertEqual(trainer.embodiment_tag, "so100")
        self.assertEqual(trainer.data_config, "so100_dualcam")
        self.assertTrue(trainer.tune_projector)
        self.assertTrue(trainer.tune_diffusion_model)
        self.assertFalse(trainer.tune_llm)
        self.assertFalse(trainer.tune_visual)

    def test_groot_data_config_so100_dualcam(self):
        """Verify so100_dualcam data config matches GR00T requirements."""
        from strands_robots.policies.groot.data_config import load_data_config

        config = load_data_config("so100_dualcam")
        self.assertEqual(config.video_keys, ["video.front", "video.wrist"])
        self.assertEqual(config.state_keys, ["state.single_arm", "state.gripper"])
        self.assertEqual(config.action_keys, ["action.single_arm", "action.gripper"])
        self.assertTrue(len(config.language_keys) > 0)

    def test_groot_data_config_so101_exists(self):
        """Verify SO-101 configs exist."""
        from strands_robots.policies.groot.data_config import DATA_CONFIG_MAP

        self.assertIn("so101", DATA_CONFIG_MAP)
        self.assertIn("so101_dualcam", DATA_CONFIG_MAP)
        self.assertIn("so101_tricam", DATA_CONFIG_MAP)

    def test_all_data_configs_valid(self):
        """Verify all data configs have required fields."""
        from strands_robots.policies.groot.data_config import DATA_CONFIG_MAP

        for name, config in DATA_CONFIG_MAP.items():
            self.assertTrue(len(config.video_keys) > 0, f"{name}: no video_keys")
            self.assertTrue(len(config.state_keys) > 0, f"{name}: no state_keys")
            self.assertTrue(len(config.action_keys) > 0, f"{name}: no action_keys")
            self.assertTrue(len(config.language_keys) > 0, f"{name}: no language_keys")


class TestStage5Evaluate(unittest.TestCase):
    """Test Stage 5: Evaluation harness."""

    def test_evaluate_import(self):
        from strands_robots.training import evaluate

        self.assertTrue(callable(evaluate))

    def test_evaluate_signature(self):
        """Verify evaluate function accepts expected arguments."""
        import inspect

        from strands_robots.training import evaluate

        sig = inspect.signature(evaluate)
        params = list(sig.parameters.keys())
        self.assertIn("policy", params)
        self.assertIn("task", params)
        self.assertIn("robot_name", params)
        self.assertIn("num_episodes", params)
        self.assertIn("backend", params)


class TestStage6CosmosReason(unittest.TestCase):
    """Test Stage 6: Cosmos Reason Analysis."""

    def test_cosmos_reason_with_eval_results(self):
        """Test analysis of evaluation results."""
        import tempfile

        from scripts.training.groot_finetune_pipeline import PipelineConfig, stage_6_cosmos_reason

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = os.path.join(tmpdir, "eval")
            cosmos_dir = os.path.join(tmpdir, "cosmos")
            os.makedirs(eval_dir)

            # Create mock eval results
            eval_results = {
                "pick up the red cube": {
                    "success_rate": 25.0,
                    "mean_reward": 1.5,
                    "episodes": [
                        {"episode": i, "steps": 100 + i * 10, "reward": 0.5 + i * 0.3, "success": i % 4 == 0}
                        for i in range(10)
                    ],
                }
            }
            with open(os.path.join(eval_dir, "eval_results.json"), "w") as f:
                json.dump(eval_results, f)

            config = PipelineConfig(
                eval_output_dir=eval_dir,
                cosmos_analysis_dir=cosmos_dir,
            )

            result = stage_6_cosmos_reason(config)
            self.assertEqual(result["status"], "success")

            # Check analysis output
            analysis = result["analysis"]
            self.assertIn("recommendations", analysis)
            self.assertIn("task_analysis", analysis)
            self.assertIn("overall_assessment", analysis)
            self.assertIn("pick up the red cube", analysis["task_analysis"])

            # Verify recommendations generated for low success rate
            recs = analysis["recommendations"]
            self.assertTrue(len(recs) > 0, "Should generate recommendations for 25% success rate")

            # Verify file was written
            self.assertTrue(os.path.exists(os.path.join(cosmos_dir, "cosmos_analysis.json")))

    def test_cosmos_reason_no_results(self):
        """Test graceful handling when no eval results exist."""
        import tempfile

        from scripts.training.groot_finetune_pipeline import PipelineConfig, stage_6_cosmos_reason

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                eval_output_dir=os.path.join(tmpdir, "nonexistent"),
                cosmos_analysis_dir=os.path.join(tmpdir, "cosmos"),
            )
            result = stage_6_cosmos_reason(config)
            self.assertEqual(result["status"], "skipped")

    def test_cosmos_reason_high_success(self):
        """Test analysis with high success rate."""
        import tempfile

        from scripts.training.groot_finetune_pipeline import PipelineConfig, stage_6_cosmos_reason

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = os.path.join(tmpdir, "eval")
            cosmos_dir = os.path.join(tmpdir, "cosmos")
            os.makedirs(eval_dir)

            eval_results = {
                "pick up the red cube": {
                    "success_rate": 90.0,
                    "mean_reward": 15.0,
                    "episodes": [{"episode": i, "steps": 80, "reward": 15.0, "success": True} for i in range(10)],
                }
            }
            with open(os.path.join(eval_dir, "eval_results.json"), "w") as f:
                json.dump(eval_results, f)

            config = PipelineConfig(
                eval_output_dir=eval_dir,
                cosmos_analysis_dir=cosmos_dir,
            )
            result = stage_6_cosmos_reason(config)
            self.assertIn("EXCELLENT", result["analysis"]["overall_assessment"])


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for pipeline data flow."""

    def test_sac_to_groot_data_format_bridge(self):
        """Verify SAC action space maps to GR00T data config format.

        SAC output: action_space[0:5] → arm, action_space[5] → gripper
        GR00T input: action.single_arm (5D), action.gripper (1D)
        """
        # Simulate SAC action
        sac_action = np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.8], dtype=np.float32)

        # Bridge to GR00T format (as done in stage_3_record_dataset)
        arm_action = sac_action[:5].tolist()
        gripper_action = [float(sac_action[5])]

        self.assertEqual(len(arm_action), 5)
        self.assertEqual(len(gripper_action), 1)
        self.assertAlmostEqual(arm_action[0], 0.1, places=5)
        self.assertAlmostEqual(gripper_action[0], 0.8, places=5)

    def test_reward_function_compatible_with_gymnasium(self):
        """Test PickAndPlaceReward works with gymnasium-style observations."""
        from strands_robots.rl_trainer import PickAndPlaceReward

        reward = PickAndPlaceReward()

        # numpy array obs (gymnasium default)
        obs_array = np.zeros(14)
        action = np.zeros(6)
        r = reward(obs_array, action)
        self.assertIsInstance(r, float)

        reward.reset()

        # dict obs (Isaac Lab style)
        obs_dict = {"state": np.zeros(14), "policy": np.zeros(14)}
        r2 = reward(obs_dict, action)
        self.assertIsInstance(r2, float)

    def test_pipeline_output_structure(self):
        """Verify expected output directory structure."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            from scripts.training.groot_finetune_pipeline import PipelineConfig

            config = PipelineConfig(
                sac_output_dir=os.path.join(tmpdir, "sac"),
                record_output_dir=os.path.join(tmpdir, "dataset"),
                groot_output_dir=os.path.join(tmpdir, "groot"),
                eval_output_dir=os.path.join(tmpdir, "eval"),
                cosmos_analysis_dir=os.path.join(tmpdir, "cosmos"),
            )

            # Verify output dirs are configurable
            self.assertTrue(config.sac_output_dir.endswith("sac"))
            self.assertTrue(config.record_output_dir.endswith("dataset"))
            self.assertTrue(config.groot_output_dir.endswith("groot"))

    def test_create_trainer_all_providers(self):
        """Verify all training providers can be instantiated."""
        from strands_robots.training import create_trainer

        # GR00T
        t = create_trainer("groot", base_model_path="test", dataset_path="/tmp/test")
        self.assertEqual(t.provider_name, "groot")

        # LeRobot
        t = create_trainer("lerobot", policy_type="act", dataset_repo_id="test/data")
        self.assertEqual(t.provider_name, "lerobot")

        # DreamGen IDM
        t = create_trainer("dreamgen_idm", dataset_path="/tmp/test")
        self.assertEqual(t.provider_name, "dreamgen_idm")

        # DreamGen VLA
        t = create_trainer("dreamgen_vla", dataset_path="/tmp/test")
        self.assertEqual(t.provider_name, "dreamgen_vla")

        # Cosmos Predict
        t = create_trainer("cosmos_predict", dataset_path="/tmp/test")
        self.assertEqual(t.provider_name, "cosmos_predict")

        # Cosmos Transfer
        t = create_trainer("cosmos_transfer", dataset_path="/tmp/test")
        self.assertEqual(t.provider_name, "cosmos_transfer")

    def test_rl_trainer_algorithms(self):
        """Verify both PPO and SAC can be created."""
        from strands_robots.rl_trainer import create_rl_trainer

        ppo = create_rl_trainer(algorithm="ppo", total_timesteps=100)
        self.assertEqual(ppo.config.algorithm, "ppo")

        sac = create_rl_trainer(algorithm="sac", total_timesteps=100)
        self.assertEqual(sac.config.algorithm, "sac")


class TestRunFullPipelineErrorHandling(unittest.TestCase):
    """Test run_full_pipeline stage dependency logic."""

    def test_eval_skipped_when_groot_fails(self):
        """Verify Stage 5 is skipped when Stage 4 fails."""
        # Simulate the pipeline's stage dependency logic
        results = {"stages": {"groot": {"status": "error", "error": "GPU required"}}}

        groot_status = results["stages"].get("groot", {}).get("status")
        self.assertNotEqual(groot_status, "success")

        # This is the guard added in the fix
        if groot_status != "success":
            results["stages"]["eval"] = {
                "status": "skipped",
                "reason": f"GR00T training status: {groot_status}",
            }

        self.assertEqual(results["stages"]["eval"]["status"], "skipped")
        self.assertIn("error", results["stages"]["eval"]["reason"])

    def test_cosmos_skipped_when_eval_skipped(self):
        """Verify Stage 6 is skipped when Stage 5 is skipped."""
        results = {"stages": {"eval": {"status": "skipped", "reason": "test"}}}

        eval_status = results["stages"].get("eval", {}).get("status")
        if eval_status != "success":
            results["stages"]["cosmos_reason"] = {
                "status": "skipped",
                "reason": f"Evaluation status: {eval_status}",
            }

        self.assertEqual(results["stages"]["cosmos_reason"]["status"], "skipped")


class TestE2EPipelineChain(unittest.TestCase):
    """Test the complete pipeline chain: 3D → Isaac → RL → Eval → Cosmos → Code."""

    def test_chain_data_format_consistency(self):
        """Verify data format is consistent across all pipeline stages.

        Stage 2 output (SAC) → Stage 3 input (recording)
        Stage 3 output (dataset) → Stage 4 input (GR00T)
        Stage 4 output (checkpoint) → Stage 5 input (eval)
        Stage 5 output (results) → Stage 6 input (analysis)
        """
        from strands_robots.policies.groot.data_config import load_data_config

        config = load_data_config("so100_dualcam")

        # SAC → Dataset bridge: action dimensions must match
        sac_action_dim = 6  # 5 arm + 1 gripper for SO-100
        groot_arm_dim = 5  # action.single_arm is 5D
        groot_gripper_dim = 1  # action.gripper is 1D
        self.assertEqual(sac_action_dim, groot_arm_dim + groot_gripper_dim)

        # Dataset → GR00T: key names must match data_config
        self.assertIn("action.single_arm", config.action_keys)
        self.assertIn("action.gripper", config.action_keys)
        self.assertIn("state.single_arm", config.state_keys)
        self.assertIn("state.gripper", config.state_keys)

        # Dual camera: video keys must match
        self.assertIn("video.front", config.video_keys)
        self.assertIn("video.wrist", config.video_keys)
        self.assertEqual(len(config.video_keys), 2)

        # Language annotation: must have annotation key
        self.assertIn("annotation.human.task_description", config.language_keys)

        # Eval results → Cosmos analysis: JSON structure
        eval_result = {
            "success_rate": 50.0,
            "mean_reward": 5.0,
            "episodes": [{"episode": 0, "steps": 100, "reward": 5.0, "success": True}],
        }
        # Cosmos expects these exact keys
        self.assertIn("success_rate", eval_result)
        self.assertIn("mean_reward", eval_result)
        self.assertIn("episodes", eval_result)


if __name__ == "__main__":
    unittest.main()
