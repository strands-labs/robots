"""Tests for strands_robots/dreamgen/__init__.py — DreamGen Neural Trajectory Pipeline.

Coverage target: ~85%+ of dreamgen/__init__.py (567 lines).
All tests CPU-only with mocked heavy dependencies (torch, transformers, cv2, decord,
cosmos_transfer).

Test organization:
1. SyntaxValidation — file parses correctly
2. DataclassTests — NeuralTrajectory, DreamGenConfig defaults
3. PipelineInit — DreamGenPipeline creation from config and kwargs
4. Stage1 — finetune_video_model (cosmos_transfer skip, normal models)
5. Stage2 — generate_videos (triple-nested loop, cosmos_transfer delegation)
6. Stage3 — extract_actions (IDM, latent, unknown method)
7. Stage4 — create_dataset (raw format, empty trajectories)
8. FullPipeline — run_full_pipeline orchestration
9. LoadVideoFrames — cv2/decord fallback
10. CosmosTransfer — _generate_via_cosmos_transfer with mocked pipeline
"""

import ast
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    _has_torch = True
except ImportError:
    _has_torch = False


# ───────────────────────────── Syntax Validation ─────────────────────────────


class TestSyntaxValidation:
    """Verify source file parses without errors."""

    def test_dreamgen_parses(self):
        """dreamgen/__init__.py is valid Python."""
        src = Path(__file__).resolve().parent.parent / "strands_robots" / "dreamgen" / "__init__.py"
        source = src.read_text()
        tree = ast.parse(source)
        assert tree is not None

    def test_exports(self):
        """Module exports the expected symbols."""
        from strands_robots.dreamgen import __all__

        assert "DreamGenPipeline" in __all__
        assert "DreamGenConfig" in __all__
        assert "NeuralTrajectory" in __all__


# ───────────────────────────── Dataclass Tests ─────────────────────────────


class TestNeuralTrajectory:
    """Tests for the NeuralTrajectory dataclass."""

    def test_defaults(self):
        """NeuralTrajectory has correct defaults."""
        from strands_robots.dreamgen import NeuralTrajectory

        t = NeuralTrajectory(
            frames=np.zeros((10, 480, 640, 3), dtype=np.uint8),
            actions=np.zeros((9, 6), dtype=np.float32),
            instruction="pick up the cup",
        )
        assert t.action_type == "idm"
        assert t.metadata == {}
        assert t.instruction == "pick up the cup"
        assert t.frames.shape == (10, 480, 640, 3)
        assert t.actions.shape == (9, 6)

    def test_custom_action_type(self):
        """NeuralTrajectory accepts custom action_type."""
        from strands_robots.dreamgen import NeuralTrajectory

        t = NeuralTrajectory(
            frames=np.zeros((5, 64, 64, 3), dtype=np.uint8),
            actions=np.zeros((4, 8), dtype=np.float32),
            instruction="test",
            action_type="latent",
            metadata={"source": "lapa"},
        )
        assert t.action_type == "latent"
        assert t.metadata["source"] == "lapa"

    def test_metadata_isolation(self):
        """Each NeuralTrajectory gets its own metadata dict."""
        from strands_robots.dreamgen import NeuralTrajectory

        t1 = NeuralTrajectory(frames=np.zeros(1), actions=np.zeros(1), instruction="a")
        t2 = NeuralTrajectory(frames=np.zeros(1), actions=np.zeros(1), instruction="b")
        t1.metadata["key"] = "value"
        assert "key" not in t2.metadata


class TestDreamGenConfig:
    """Tests for the DreamGenConfig dataclass."""

    def test_defaults(self):
        """DreamGenConfig has sensible defaults."""
        from strands_robots.dreamgen import DreamGenConfig

        c = DreamGenConfig()
        assert c.video_model == "wan2.1"
        assert c.lora_rank == 4
        assert c.lora_alpha == 4
        assert c.video_finetune_epochs == 100
        assert c.video_finetune_batch_size == 32
        assert c.video_finetune_lr == 1e-4
        assert c.idm_checkpoint == ""
        assert c.idm_action_horizon == 16
        assert c.idm_sliding_window is True
        assert c.num_inference_steps == 50
        assert c.video_length == 81
        assert c.video_resolution == (480, 640)
        assert c.embodiment_tag == "new_embodiment"
        assert c.data_config == "so100"
        assert c.num_gpus == 1

    def test_custom_values(self):
        """DreamGenConfig accepts custom values."""
        from strands_robots.dreamgen import DreamGenConfig

        c = DreamGenConfig(
            video_model="cogvideox",
            lora_rank=16,
            idm_checkpoint="my/checkpoint",
            embodiment_tag="panda",
        )
        assert c.video_model == "cogvideox"
        assert c.lora_rank == 16
        assert c.idm_checkpoint == "my/checkpoint"
        assert c.embodiment_tag == "panda"


# ───────────────────────────── Pipeline Init ─────────────────────────────


class TestPipelineInit:
    """Tests for DreamGenPipeline initialization."""

    def test_init_with_config(self):
        """Pipeline accepts explicit DreamGenConfig."""
        from strands_robots.dreamgen import DreamGenConfig, DreamGenPipeline

        cfg = DreamGenConfig(video_model="cosmos", embodiment_tag="g1")
        p = DreamGenPipeline(config=cfg)
        assert p.config.video_model == "cosmos"
        assert p.config.embodiment_tag == "g1"
        assert p._video_model is None
        assert p._idm_model is None

    def test_init_with_kwargs(self):
        """Pipeline creates DreamGenConfig from kwargs."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline(video_model="hunyuan", lora_rank=8, data_config="panda")
        assert p.config.video_model == "hunyuan"
        assert p.config.lora_rank == 8
        assert p.config.data_config == "panda"

    def test_init_unknown_kwargs_ignored(self):
        """Pipeline ignores kwargs not in DreamGenConfig fields."""
        from strands_robots.dreamgen import DreamGenPipeline

        # Should not raise
        p = DreamGenPipeline(video_model="wan2.1", unknown_field="hello", xyz=42)
        assert p.config.video_model == "wan2.1"

    def test_init_default_config(self):
        """Pipeline with no args gets default DreamGenConfig."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()
        assert p.config.video_model == "wan2.1"
        assert p.config.embodiment_tag == "new_embodiment"


# ───────────────────────────── Stage 1: finetune_video_model ─────────────────


class TestFinetuneVideoModel:
    """Tests for Stage 1: finetune_video_model."""

    def test_cosmos_transfer_skips(self):
        """cosmos_transfer model skips fine-tuning entirely."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline(video_model="cosmos_transfer")
        result = p.finetune_video_model(dataset_path="/data/test")
        assert result["status"] == "skipped"
        assert result["model"] == "cosmos_transfer"
        assert result["stage"] == "finetune_video_model"
        assert "ControlNet" in result["reason"]

    def test_wan21_returns_ready(self):
        """wan2.1 model returns ready status with correct command."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline(video_model="wan2.1")
        result = p.finetune_video_model(dataset_path="/data/test", output_dir="/out")
        assert result["status"] == "ready"
        assert result["model"] == "wan2.1"
        assert result["dataset"] == "/data/test"
        assert result["output_dir"] == "/out"
        assert "finetune_wan" in result["command"]

    def test_cogvideox_command(self):
        """cogvideox model gets correct command."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline(video_model="cogvideox")
        result = p.finetune_video_model(dataset_path="/data")
        assert "finetune_cogvideo" in result["command"]

    def test_cosmos_command(self):
        """cosmos model gets correct command."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline(video_model="cosmos")
        result = p.finetune_video_model(dataset_path="/data")
        assert "finetune_cosmos" in result["command"]

    def test_hunyuan_command(self):
        """hunyuan model gets correct command."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline(video_model="hunyuan")
        result = p.finetune_video_model(dataset_path="/data")
        assert "finetune_hunyuan" in result["command"]

    def test_unknown_model_returns_unknown_command(self):
        """Unknown video model returns 'unknown' command string."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline(video_model="my_custom_model")
        result = p.finetune_video_model(dataset_path="/data")
        assert result["command"] == "unknown"  # .get with default returns "unknown"

    def test_sets_video_model_path(self):
        """finetune_video_model sets config.video_model_path."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline(video_model="wan2.1")
        assert p.config.video_model_path is None
        p.finetune_video_model(dataset_path="/data", output_dir="/my/output")
        assert p.config.video_model_path == "/my/output"

    def test_default_output_dir(self):
        """finetune_video_model uses default output_dir."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline(video_model="wan2.1")
        result = p.finetune_video_model(dataset_path="/data")
        assert result["output_dir"] == "./video_model_finetuned"


# ───────────────────────────── Stage 2: generate_videos ─────────────────


class TestGenerateVideos:
    """Tests for Stage 2: generate_videos."""

    def test_video_count_formula(self):
        """Generated video count = frames × instructions × num_per_prompt."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline(video_model="wan2.1")
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
        instructions = ["pick", "place"]
        result = p.generate_videos(frames, instructions, num_per_prompt=5)
        assert len(result) == 3 * 2 * 5  # 30 videos

    def test_single_frame_single_instruction(self):
        """1 frame × 1 instruction × 1 per_prompt = 1 video."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        result = p.generate_videos(frames, ["go"], num_per_prompt=1)
        assert len(result) == 1
        assert result[0]["instruction"] == "go"
        assert result[0]["frame_idx"] == 0

    def test_video_metadata_structure(self):
        """Each video dict has expected keys."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        result = p.generate_videos(frames, ["test"], num_per_prompt=1)
        v = result[0]
        assert "video_id" in v
        assert "video_path" in v
        assert "instruction" in v
        assert "frame_idx" in v
        assert "initial_frame_shape" in v
        assert v["initial_frame_shape"] == (480, 640, 3)

    def test_video_ids_unique(self):
        """All video IDs are unique."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
        result = p.generate_videos(frames, ["a", "b"], num_per_prompt=3)
        ids = [v["video_id"] for v in result]
        assert len(ids) == len(set(ids))

    def test_output_dir_created(self):
        """generate_videos creates output directory."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "gen_videos")
            p.generate_videos(
                [np.zeros((64, 64, 3), dtype=np.uint8)],
                ["test"],
                num_per_prompt=1,
                output_dir=out,
            )
            assert os.path.isdir(out)

    def test_empty_frames_produces_empty(self):
        """Empty frames list produces no videos."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()
        result = p.generate_videos([], ["test"], num_per_prompt=5)
        assert len(result) == 0

    def test_empty_instructions_produces_empty(self):
        """Empty instructions list produces no videos."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()
        result = p.generate_videos([np.zeros((64, 64, 3), dtype=np.uint8)], [], num_per_prompt=5)
        assert len(result) == 0

    def test_cosmos_transfer_delegates(self):
        """cosmos_transfer model delegates to _generate_via_cosmos_transfer."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline(video_model="cosmos_transfer")
        with patch.object(p, "_generate_via_cosmos_transfer", return_value=[{"mock": True}]) as mock_ct:
            result = p.generate_videos(
                [np.zeros((64, 64, 3), dtype=np.uint8)],
                ["test"],
                num_per_prompt=1,
            )
            mock_ct.assert_called_once()
            assert result == [{"mock": True}]


# ───────────────────────────── Stage 3: extract_actions ─────────────────


class TestExtractActions:
    """Tests for Stage 3: extract_actions."""

    def test_unknown_method_raises(self):
        """Unknown extraction method raises ValueError."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()
        with pytest.raises(ValueError, match="Unknown action extraction method"):
            p.extract_actions([], method="bad_method")

    def test_idm_delegates(self):
        """method='idm' delegates to _extract_idm_actions."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()
        with patch.object(p, "_extract_idm_actions", return_value=[]) as mock_idm:
            p.extract_actions([{"video_path": "v.mp4", "instruction": "t"}], method="idm")
            mock_idm.assert_called_once()

    def test_latent_delegates(self):
        """method='latent' delegates to _extract_latent_actions."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()
        with patch.object(p, "_extract_latent_actions", return_value=[]) as mock_lat:
            p.extract_actions([{"video_path": "v.mp4", "instruction": "t"}], method="latent")
            mock_lat.assert_called_once()


class TestExtractIdmActions:
    """Tests for _extract_idm_actions (IDM model loading + extraction)."""

    def test_model_load_failure_returns_placeholders(self):
        """IDM load failure returns placeholder trajectories."""
        from strands_robots.dreamgen import DreamGenPipeline, NeuralTrajectory

        p = DreamGenPipeline(idm_checkpoint="bad/path")
        videos = [
            {"video_path": "v1.mp4", "instruction": "pick"},
            {"video_path": "v2.mp4", "instruction": "place"},
        ]
        # Mock torch to raise ImportError
        with patch.dict(sys.modules, {"torch": None}):
            # Force re-evaluation — _idm_model is None so it tries loading
            p._idm_model = None
            result = p._extract_idm_actions(videos, "/tmp/out")

        assert len(result) == 2
        for traj in result:
            assert isinstance(traj, NeuralTrajectory)
            assert traj.action_type == "idm"
            assert traj.frames.shape == (10, 480, 640, 3)
            assert traj.actions.shape == (9, 6)

    def test_placeholder_instructions_match(self):
        """Placeholder trajectories preserve original instructions."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()
        videos = [
            {"video_path": "v.mp4", "instruction": "grasp the red block"},
        ]
        with patch.dict(sys.modules, {"torch": None}):
            p._idm_model = None
            result = p._extract_idm_actions(videos, "/tmp/out")
        assert result[0].instruction == "grasp the red block"
        assert result[0].metadata == videos[0]


class TestExtractLatentActions:
    """Tests for _extract_latent_actions (LAPA placeholder)."""

    def test_returns_latent_trajectories(self):
        """Latent extraction returns placeholder trajectories with action_dim=8."""
        from strands_robots.dreamgen import DreamGenPipeline, NeuralTrajectory

        p = DreamGenPipeline()
        videos = [
            {"video_path": "v1.mp4", "instruction": "push"},
            {"video_path": "v2.mp4", "instruction": "pull"},
            {"video_path": "v3.mp4", "instruction": "lift"},
        ]
        result = p._extract_latent_actions(videos, "/tmp/out")
        assert len(result) == 3
        for traj in result:
            assert isinstance(traj, NeuralTrajectory)
            assert traj.action_type == "latent"
            assert traj.actions.shape == (9, 8)  # LAPA codebook size 8
            assert traj.frames.shape == (10, 480, 640, 3)

    def test_preserves_instructions(self):
        """Latent trajectories preserve instructions from video metadata."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()
        videos = [{"video_path": "v.mp4", "instruction": "wave hello"}]
        result = p._extract_latent_actions(videos, "/tmp/out")
        assert result[0].instruction == "wave hello"


# ───────────────────────────── Stage 4: create_dataset ─────────────────


class TestCreateDataset:
    """Tests for Stage 4: create_dataset."""

    def test_raw_format_saves_files(self):
        """format='raw' saves frames.npy, actions.npy, instruction.txt."""
        from strands_robots.dreamgen import DreamGenPipeline, NeuralTrajectory

        p = DreamGenPipeline()
        trajs = [
            NeuralTrajectory(
                frames=np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8),
                actions=np.random.randn(9, 6).astype(np.float32),
                instruction="pick up the cup",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            p.create_dataset(trajs, output_path=tmpdir, format="raw")
            traj_dir = os.path.join(tmpdir, "trajectory_00000")
            assert os.path.isdir(traj_dir)
            assert os.path.isfile(os.path.join(traj_dir, "frames.npy"))
            assert os.path.isfile(os.path.join(traj_dir, "actions.npy"))
            assert os.path.isfile(os.path.join(traj_dir, "instruction.txt"))

            # Verify content
            frames = np.load(os.path.join(traj_dir, "frames.npy"))
            assert frames.shape == (10, 64, 64, 3)
            with open(os.path.join(traj_dir, "instruction.txt")) as f:
                assert f.read() == "pick up the cup"

    def test_raw_format_multiple_trajectories(self):
        """Multiple trajectories create numbered subdirectories."""
        from strands_robots.dreamgen import DreamGenPipeline, NeuralTrajectory

        p = DreamGenPipeline()
        trajs = [
            NeuralTrajectory(
                frames=np.zeros((5, 32, 32, 3), dtype=np.uint8),
                actions=np.zeros((4, 6), dtype=np.float32),
                instruction=f"instruction_{i}",
            )
            for i in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = p.create_dataset(trajs, output_path=tmpdir, format="raw")
            assert result["num_trajectories"] == 3
            assert result["total_frames"] == 15  # 5 * 3
            assert result["action_type"] == "idm"
            for i in range(3):
                assert os.path.isdir(os.path.join(tmpdir, f"trajectory_{i:05d}"))

    def test_result_metadata(self):
        """create_dataset returns correct metadata dict."""
        from strands_robots.dreamgen import DreamGenPipeline, NeuralTrajectory

        p = DreamGenPipeline()
        trajs = [
            NeuralTrajectory(
                frames=np.zeros((8, 32, 32, 3), dtype=np.uint8),
                actions=np.zeros((7, 6), dtype=np.float32),
                instruction="test",
                action_type="latent",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = p.create_dataset(trajs, output_path=tmpdir, format="lerobot")
            assert result["format"] == "lerobot"
            assert result["num_trajectories"] == 1
            assert result["total_frames"] == 8
            assert result["action_type"] == "latent"
            assert result["output_path"] == tmpdir

    def test_empty_trajectories(self):
        """Empty trajectory list produces valid result."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = p.create_dataset([], output_path=tmpdir, format="raw")
            assert result["num_trajectories"] == 0
            assert result["total_frames"] == 0
            assert result["action_type"] == "unknown"

    def test_creates_output_directory(self):
        """create_dataset creates output directory if missing."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "nested", "output")
            p.create_dataset([], output_path=out, format="raw")
            assert os.path.isdir(out)


# ───────────────────────────── Full Pipeline ─────────────────────────────


class TestRunFullPipeline:
    """Tests for run_full_pipeline orchestration."""

    def test_orchestration_calls_all_stages(self):
        """run_full_pipeline calls all 4 stages in order."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline(video_model="wan2.1")

        with (
            patch.object(p, "finetune_video_model", return_value={"status": "ready"}) as s1,
            patch.object(p, "generate_videos", return_value=[{"video_path": "v.mp4", "instruction": "pick"}]) as s2,
            patch.object(p, "extract_actions", return_value=[]) as s3,
            patch.object(
                p,
                "create_dataset",
                return_value={
                    "num_trajectories": 0,
                    "total_frames": 0,
                    "action_type": "unknown",
                    "output_path": "/out",
                    "format": "lerobot",
                },
            ) as s4,
        ):

            frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
            result = p.run_full_pipeline(
                robot_dataset_path="/data",
                initial_frames=frames,
                instructions=["pick"],
                num_per_prompt=10,
                output_dir="/output",
            )

            s1.assert_called_once()
            s2.assert_called_once()
            s3.assert_called_once()
            s4.assert_called_once()

            assert "stage1" in result
            assert "stage2" in result
            assert "stage3" in result
            assert "stage4" in result

    def test_subdirectory_propagation(self):
        """Each stage gets a subdirectory of output_dir."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline(video_model="wan2.1")

        with (
            patch.object(p, "finetune_video_model", return_value={"status": "ready"}) as s1,
            patch.object(p, "generate_videos", return_value=[]) as s2,
            patch.object(p, "extract_actions", return_value=[]),
            patch.object(
                p,
                "create_dataset",
                return_value={
                    "num_trajectories": 0,
                    "total_frames": 0,
                    "action_type": "unknown",
                    "output_path": "",
                    "format": "lerobot",
                },
            ),
        ):

            p.run_full_pipeline(
                robot_dataset_path="/data",
                initial_frames=[np.zeros((64, 64, 3))],
                instructions=["go"],
                output_dir="/base",
            )

            # Verify subdirectory paths
            s1_call = s1.call_args
            assert "video_model" in s1_call.kwargs.get("output_dir", s1_call[1].get("output_dir", ""))
            s2_call = s2.call_args
            assert "generated_videos" in s2_call.kwargs.get("output_dir", s2_call[1].get("output_dir", ""))

    def test_action_method_passthrough(self):
        """action_method is passed through to extract_actions."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()

        with (
            patch.object(p, "finetune_video_model", return_value={}),
            patch.object(p, "generate_videos", return_value=[]),
            patch.object(p, "extract_actions", return_value=[]) as s3,
            patch.object(
                p,
                "create_dataset",
                return_value={
                    "num_trajectories": 0,
                    "total_frames": 0,
                    "action_type": "unknown",
                    "output_path": "",
                    "format": "lerobot",
                },
            ),
        ):

            p.run_full_pipeline(
                robot_dataset_path="/data",
                initial_frames=[np.zeros((64, 64, 3))],
                instructions=["go"],
                action_method="latent",
            )

            s3_call = s3.call_args
            assert s3_call.kwargs.get("method") == "latent" or s3_call[1].get("method") == "latent"


# ───────────────────────────── _load_video_frames ─────────────────────────────


class TestLoadVideoFrames:
    """Tests for _load_video_frames cv2/decord fallback."""

    def test_cv2_success(self):
        """Uses cv2.VideoCapture when cv2 is available."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()

        mock_cap = MagicMock()
        # Return 3 frames then stop
        mock_cap.isOpened.side_effect = [True, True, True, True]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [
            (True, frame.copy()),
            (True, frame.copy()),
            (True, frame.copy()),
            (False, None),
        ]

        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.cvtColor.side_effect = lambda f, _: f  # passthrough
        mock_cv2.COLOR_BGR2RGB = 4

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = p._load_video_frames("/path/to/video.mp4")

        assert result.shape == (3, 480, 640, 3)
        mock_cap.release.assert_called_once()

    def test_cv2_unavailable_falls_back_to_decord(self):
        """Falls back to decord when cv2 is not available."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()

        mock_vr = MagicMock()
        mock_array = MagicMock()
        mock_array.asnumpy.return_value = np.zeros((5, 480, 640, 3), dtype=np.uint8)
        mock_vr.__getitem__ = MagicMock(return_value=mock_array)

        mock_decord = MagicMock()
        mock_decord.VideoReader.return_value = mock_vr

        # Block cv2 import, provide decord
        with patch.dict(sys.modules, {"cv2": None, "decord": mock_decord}):
            result = p._load_video_frames("/path/to/video.mp4")

        assert result.shape == (5, 480, 640, 3)


# ───────────────────────────── Cosmos Transfer Integration ─────────────────


class TestCosmosTransferGeneration:
    """Tests for _generate_via_cosmos_transfer."""

    def test_with_sim_videos(self):
        """Cosmos transfer uses provided sim_videos."""

        mock_pipeline = MagicMock()
        mock_pipeline.transfer_video.return_value = {"status": "transferred"}

        mock_config_cls = MagicMock()
        mock_pipeline_cls = MagicMock(return_value=mock_pipeline)

        with patch.dict(
            sys.modules,
            {
                "strands_robots.cosmos_transfer": MagicMock(
                    CosmosTransferPipeline=mock_pipeline_cls,
                    CosmosTransferConfig=mock_config_cls,
                ),
            },
        ):
            # Need to reload or import fresh
            from strands_robots.dreamgen import DreamGenPipeline as DP

            p = DP(video_model="cosmos_transfer")

            frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
            result = p._generate_via_cosmos_transfer(
                initial_frames=frames,
                instructions=["pick up"],
                num_per_prompt=2,
                output_dir="/tmp/out",
                sim_videos=["/sim/v1.mp4", "/sim/v2.mp4"],
            )

        # 2 sim_videos × 2 per_prompt = 4
        assert len(result) == 4
        assert all("cosmos_" in r["video_id"] for r in result)
        assert all(r["control_type"] == "depth" for r in result)

    def test_no_sim_videos_fallback(self):
        """Missing sim_videos triggers frame-based fallback."""

        mock_pipeline = MagicMock()
        mock_pipeline.transfer_video.return_value = {"status": "ok"}
        mock_config_cls = MagicMock()
        mock_pipeline_cls = MagicMock(return_value=mock_pipeline)

        with patch.dict(
            sys.modules,
            {
                "strands_robots.cosmos_transfer": MagicMock(
                    CosmosTransferPipeline=mock_pipeline_cls,
                    CosmosTransferConfig=mock_config_cls,
                ),
            },
        ):
            from strands_robots.dreamgen import DreamGenPipeline as DP

            p = DP(video_model="cosmos_transfer")

            frames = [
                np.zeros((64, 64, 3), dtype=np.uint8),
                np.zeros((64, 64, 3), dtype=np.uint8),
            ]
            with tempfile.TemporaryDirectory() as tmpdir:
                result = p._generate_via_cosmos_transfer(
                    initial_frames=frames,
                    instructions=["test"],
                    num_per_prompt=1,
                    output_dir=tmpdir,
                    # No sim_videos kwarg
                )

            # 2 fallback frames × 1 per_prompt = 2
            assert len(result) == 2


# ───────────────────────────── IDM with Pre-loaded Model ─────────────────


class TestIdmWithModel:
    """Tests for _extract_idm_actions when IDM model loads successfully."""

    @pytest.mark.skipif(not _has_torch, reason="torch required")
    def test_sliding_window_extraction(self):
        """IDM extraction uses sliding window over frame pairs."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline(idm_checkpoint="nvidia/test-idm")

        # Pre-set the model so it doesn't try to load
        mock_model = MagicMock()
        action_pred = np.zeros((1, 16, 6), dtype=np.float32)
        mock_outputs = {"action_pred": MagicMock()}
        mock_outputs["action_pred"].cpu.return_value.numpy.return_value = action_pred
        mock_model.get_action.return_value = mock_outputs
        p._idm_model = mock_model

        # Mock _load_video_frames to return 5 frames
        frames = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
        with patch.object(p, "_load_video_frames", return_value=frames):
            videos = [{"video_path": "v.mp4", "instruction": "pick"}]
            result = p._extract_idm_actions(videos, "/tmp/out")

        assert len(result) == 1
        assert result[0].action_type == "idm"
        # 5 frames → 4 frame pairs → 4 actions
        assert result[0].actions.shape[0] == 4
        assert mock_model.get_action.call_count == 4

    def test_video_load_failure_skips(self):
        """Failed video loading skips that video."""
        from strands_robots.dreamgen import DreamGenPipeline

        p = DreamGenPipeline()
        p._idm_model = MagicMock()

        with patch.object(p, "_load_video_frames", side_effect=Exception("corrupt file")):
            videos = [{"video_path": "bad.mp4", "instruction": "test"}]
            result = p._extract_idm_actions(videos, "/tmp/out")

        assert len(result) == 0  # Skipped due to load failure
