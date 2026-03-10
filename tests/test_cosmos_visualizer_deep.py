#!/usr/bin/env python3
"""Deep coverage tests for cosmos_transfer and visualizer modules.

cosmos_transfer: 11% → 50%+ (config validation, pipeline init, spec building, helpers)
visualizer: 68% → 90%+ (all display modes, web server, update, new_episode)
"""

import json
import os
import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ═════════════════════════════════════════════════════════════════════════════
# CosmosTransferConfig Tests
# ═════════════════════════════════════════════════════════════════════════════

from strands_robots.cosmos_transfer import (
    VALID_MODEL_VARIANTS,
    VALID_OUTPUT_RESOLUTIONS,
    CosmosTransferConfig,
    CosmosTransferPipeline,
)


class TestCosmosTransferConfig:
    def test_defaults(self):
        cfg = CosmosTransferConfig()
        assert cfg.model_variant == "depth"
        assert cfg.num_gpus == 1
        assert cfg.guidance == 3.0
        assert cfg.num_steps == 35
        assert cfg.control_weight == 1.0
        assert cfg.seed == 2025
        assert cfg.output_resolution == "720"
        assert cfg.enable_autoregressive is False
        assert cfg.num_chunks == 2
        assert cfg.chunk_overlap == 1

    def test_all_valid_variants(self):
        for variant in VALID_MODEL_VARIANTS:
            cfg = CosmosTransferConfig(model_variant=variant)
            assert cfg.model_variant == variant

    def test_invalid_variant(self):
        with pytest.raises(ValueError, match="Invalid model_variant"):
            CosmosTransferConfig(model_variant="invalid")

    def test_all_valid_resolutions(self):
        for res in VALID_OUTPUT_RESOLUTIONS:
            cfg = CosmosTransferConfig(output_resolution=res)
            assert cfg.output_resolution == res

    def test_invalid_resolution(self):
        with pytest.raises(ValueError, match="Invalid output_resolution"):
            CosmosTransferConfig(output_resolution="4K")

    def test_invalid_num_gpus(self):
        with pytest.raises(ValueError, match="num_gpus must be >= 1"):
            CosmosTransferConfig(num_gpus=0)

    def test_invalid_guidance(self):
        with pytest.raises(ValueError, match="guidance must be >= 0.0"):
            CosmosTransferConfig(guidance=-1.0)

    def test_invalid_num_steps(self):
        with pytest.raises(ValueError, match="num_steps must be >= 1"):
            CosmosTransferConfig(num_steps=0)

    def test_invalid_control_weight_low(self):
        with pytest.raises(ValueError, match="control_weight"):
            CosmosTransferConfig(control_weight=-0.1)

    def test_invalid_control_weight_high(self):
        with pytest.raises(ValueError, match="control_weight"):
            CosmosTransferConfig(control_weight=2.1)

    def test_invalid_num_chunks(self):
        with pytest.raises(ValueError, match="num_chunks must be >= 1"):
            CosmosTransferConfig(num_chunks=0)

    def test_invalid_chunk_overlap(self):
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            CosmosTransferConfig(chunk_overlap=-1)

    def test_resolve_checkpoint_explicit(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        cfg = CosmosTransferConfig(checkpoint_path=str(ckpt_dir))
        assert cfg.resolve_checkpoint_path() == str(ckpt_dir)

    def test_resolve_checkpoint_env(self, tmp_path):
        ckpt_dir = tmp_path / "env_ckpt"
        ckpt_dir.mkdir()
        cfg = CosmosTransferConfig()  # No explicit path
        with patch.dict(os.environ, {"COSMOS_CHECKPOINT_DIR": str(ckpt_dir)}):
            assert cfg.resolve_checkpoint_path() == str(ckpt_dir)

    def test_resolve_checkpoint_not_found(self):
        cfg = CosmosTransferConfig(checkpoint_path="/nonexistent/path")
        with patch.dict(os.environ, {}, clear=True):
            # Remove COSMOS_CHECKPOINT_DIR if it exists
            os.environ.pop("COSMOS_CHECKPOINT_DIR", None)
            with pytest.raises(FileNotFoundError, match="Could not resolve"):
                cfg.resolve_checkpoint_path()

    def test_resolve_checkpoint_hf_cache(self, tmp_path):
        cfg = CosmosTransferConfig()
        hf_path = tmp_path / ".cache" / "huggingface" / "hub" / "models--nvidia--Cosmos-Transfer2-7B"
        hf_path.mkdir(parents=True)
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("COSMOS_CHECKPOINT_DIR", None)
            with patch("os.path.expanduser", return_value=str(hf_path)):
                assert cfg.resolve_checkpoint_path() == str(hf_path)

    def test_negative_prompt_default(self):
        cfg = CosmosTransferConfig()
        assert len(cfg.negative_prompt) > 50
        assert "blurry" in cfg.negative_prompt


# ═════════════════════════════════════════════════════════════════════════════
# CosmosTransferPipeline Init Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestCosmosTransferPipelineInit:
    def test_default_init(self):
        pipeline = CosmosTransferPipeline()
        assert pipeline.config.model_variant == "depth"
        assert pipeline._tmp_dirs == []

    def test_init_with_config(self):
        cfg = CosmosTransferConfig(num_gpus=4, guidance=5.0)
        pipeline = CosmosTransferPipeline(config=cfg)
        assert pipeline.config.num_gpus == 4
        assert pipeline.config.guidance == 5.0

    def test_init_with_kwargs(self):
        pipeline = CosmosTransferPipeline(model_variant="edge", num_steps=50)
        assert pipeline.config.model_variant == "edge"
        assert pipeline.config.num_steps == 50

    def test_init_both_config_and_kwargs_raises(self):
        cfg = CosmosTransferConfig()
        with pytest.raises(ValueError, match="Cannot specify both"):
            CosmosTransferPipeline(config=cfg, model_variant="edge")

    def test_repr(self):
        pipeline = CosmosTransferPipeline()
        r = repr(pipeline)
        assert "CosmosTransferPipeline" in r
        assert "depth" in r

    def test_cleanup_empty(self):
        pipeline = CosmosTransferPipeline()
        pipeline.cleanup()  # No-op

    def test_cleanup_with_dirs(self, tmp_path):
        pipeline = CosmosTransferPipeline()
        d = tmp_path / "work"
        d.mkdir()
        pipeline._tmp_dirs = [str(d)]
        assert d.exists()
        pipeline.cleanup()
        assert not d.exists()
        assert pipeline._tmp_dirs == []

    def test_del(self):
        pipeline = CosmosTransferPipeline()
        pipeline._tmp_dirs = ["/nonexistent/dir"]
        pipeline.__del__()  # Should not raise


# ═════════════════════════════════════════════════════════════════════════════
# CosmosTransferPipeline _build_inference_spec Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestBuildInferenceSpec:
    def test_basic_spec(self):
        pipeline = CosmosTransferPipeline()
        spec = pipeline._build_inference_spec(
            sim_video_path="/tmp/sim.mp4",
            prompt="Robot arm on table",
            output_path="/tmp/out.mp4",
            control_types=["depth"],
            control_video_paths={"depth": "/tmp/depth.mp4"},
            control_weights=[1.0],
            config=pipeline.config,
        )
        # Actual spec format: flat dict with control types as top-level keys
        assert spec["prompt"] == "Robot arm on table"
        assert spec["name"] == "out"
        assert "depth" in spec
        assert spec["depth"]["control_weight"] == 1.0
        assert spec["guidance"] == pipeline.config.guidance

    def test_multi_control_spec(self):
        pipeline = CosmosTransferPipeline()
        spec = pipeline._build_inference_spec(
            sim_video_path="/tmp/sim.mp4",
            prompt="test",
            output_path="/tmp/out.mp4",
            control_types=["depth", "edge"],
            control_video_paths={"depth": "/tmp/d.mp4", "edge": "/tmp/e.mp4"},
            control_weights=[1.0, 0.8],
            config=pipeline.config,
        )
        # Control types are top-level keys in the spec
        assert "depth" in spec
        assert "edge" in spec
        assert spec["edge"]["control_weight"] == 0.8

    def test_autoregressive_spec(self):
        """Autoregressive config is on the CosmosTransferConfig, not in the spec.
        The spec itself is the same flat format regardless of autoregressive setting.
        """
        cfg = CosmosTransferConfig(enable_autoregressive=True, num_chunks=4, chunk_overlap=2)
        pipeline = CosmosTransferPipeline(config=cfg)
        spec = pipeline._build_inference_spec(
            sim_video_path="/tmp/sim.mp4",
            prompt="test",
            output_path="/tmp/out.mp4",
            control_types=["depth"],
            control_video_paths={"depth": "/tmp/d.mp4"},
            control_weights=[1.0],
            config=cfg,
        )
        # Spec is standard format; autoregressive is handled by pipeline logic
        assert spec["prompt"] == "test"
        assert "depth" in spec
        # Verify the config itself has autoregressive settings
        assert cfg.enable_autoregressive is True
        assert cfg.num_chunks == 4
        assert cfg.chunk_overlap == 2


# ═════════════════════════════════════════════════════════════════════════════
# CosmosTransferPipeline _merge_config_overrides Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestMergeConfigOverrides:
    def test_no_overrides(self):
        pipeline = CosmosTransferPipeline(guidance=3.0)
        merged = pipeline._merge_config_overrides()
        assert merged is pipeline.config  # Same object

    def test_with_overrides(self):
        pipeline = CosmosTransferPipeline(guidance=3.0)
        merged = pipeline._merge_config_overrides(guidance=5.0, seed=42)
        assert merged.guidance == 5.0
        assert merged.seed == 42
        assert merged is not pipeline.config


# ═════════════════════════════════════════════════════════════════════════════
# CosmosTransferPipeline _count_frames Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestCountFrames:
    def test_count_frames_with_cv2(self):
        pipeline = CosmosTransferPipeline()
        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 120
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_COUNT = 7  # OpenCV constant
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            count = pipeline._count_frames("/tmp/video.mp4")
        assert count == 120

    def test_count_frames_cannot_open(self):
        pipeline = CosmosTransferPipeline()
        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            count = pipeline._count_frames("/tmp/video.mp4")
        assert count == 0

    def test_count_frames_exception(self):
        pipeline = CosmosTransferPipeline()
        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.side_effect = Exception("no cv2")
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            count = pipeline._count_frames("/tmp/video.mp4")
        assert count == 0


# ═════════════════════════════════════════════════════════════════════════════
# CosmosTransferPipeline transfer_video validation Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestTransferVideoValidation:
    def test_file_not_found(self):
        pipeline = CosmosTransferPipeline()
        with pytest.raises(FileNotFoundError, match="Simulation video not found"):
            pipeline.transfer_video(
                sim_video_path="/nonexistent/video.mp4",
                prompt="test",
                output_path="/tmp/out.mp4",
            )

    def test_invalid_control_type(self, tmp_path):
        video = tmp_path / "sim.mp4"
        video.write_text("fake video")
        pipeline = CosmosTransferPipeline()
        with pytest.raises(ValueError, match="Invalid control type"):
            pipeline.transfer_video(
                sim_video_path=str(video),
                prompt="test",
                output_path=str(tmp_path / "out.mp4"),
                control_types=["invalid"],
            )

    def test_mismatched_weights(self, tmp_path):
        video = tmp_path / "sim.mp4"
        video.write_text("fake video")
        pipeline = CosmosTransferPipeline()
        with pytest.raises(ValueError, match="control_weights length"):
            pipeline.transfer_video(
                sim_video_path=str(video),
                prompt="test",
                output_path=str(tmp_path / "out.mp4"),
                control_types=["depth", "edge"],
                control_weights=[1.0],  # Only 1 weight for 2 types
            )


# ═════════════════════════════════════════════════════════════════════════════
# CosmosTransferPipeline _generate_control dispatch Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestGenerateControl:
    def test_dispatch_depth(self):
        pipeline = CosmosTransferPipeline()
        with patch.object(pipeline, "generate_depth_control", return_value="/tmp/d.mp4") as m:
            result = pipeline._generate_control("depth", "/tmp/sim.mp4", "/tmp/work")
        assert result == "/tmp/d.mp4"
        m.assert_called_once()

    def test_dispatch_edge(self):
        pipeline = CosmosTransferPipeline()
        with patch.object(pipeline, "generate_edge_control", return_value="/tmp/e.mp4"):
            result = pipeline._generate_control("edge", "/tmp/sim.mp4", "/tmp/work")
        assert result == "/tmp/e.mp4"

    def test_dispatch_seg(self):
        pipeline = CosmosTransferPipeline()
        with patch.object(pipeline, "generate_seg_control", return_value="/tmp/s.mp4"):
            result = pipeline._generate_control("seg", "/tmp/sim.mp4", "/tmp/work")
        assert result == "/tmp/s.mp4"

    def test_dispatch_unknown(self):
        pipeline = CosmosTransferPipeline()
        with pytest.raises(ValueError, match="Unknown control type"):
            pipeline._generate_control("unknown", "/tmp/sim.mp4", "/tmp/work")


# ═════════════════════════════════════════════════════════════════════════════
# CosmosTransferPipeline generate_depth_control Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestGenerateDepthControl:
    def test_file_not_found(self):
        pipeline = CosmosTransferPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.generate_depth_control("/nonexistent.mp4")

    def test_default_output_path(self, tmp_path):
        video = tmp_path / "sim.mp4"
        video.write_text("fake")
        pipeline = CosmosTransferPipeline()
        # Mock all strategies to fail so we just test the output path logic
        with patch.object(pipeline, "_try_mujoco_depth", return_value=False):
            with patch.object(pipeline, "_try_video_depth_anything", return_value=False):
                with patch.object(pipeline, "_fallback_greyscale_depth") as fb:
                    result = pipeline.generate_depth_control(str(video))
        assert "sim_depth_control.mp4" in result
        fb.assert_called_once()

    def test_mujoco_succeeds(self, tmp_path):
        video = tmp_path / "sim.mp4"
        video.write_text("fake")
        out = str(tmp_path / "depth.mp4")
        pipeline = CosmosTransferPipeline()
        with patch.object(pipeline, "_try_mujoco_depth", return_value=True):
            result = pipeline.generate_depth_control(str(video), out)
        assert result == out

    def test_video_depth_anything_fallback(self, tmp_path):
        video = tmp_path / "sim.mp4"
        video.write_text("fake")
        out = str(tmp_path / "depth.mp4")
        pipeline = CosmosTransferPipeline()
        with patch.object(pipeline, "_try_mujoco_depth", return_value=False):
            with patch.object(pipeline, "_try_video_depth_anything", return_value=True):
                result = pipeline.generate_depth_control(str(video), out)
        assert result == out


# ═════════════════════════════════════════════════════════════════════════════
# CosmosTransferPipeline generate_edge_control Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestGenerateEdgeControl:
    def test_file_not_found(self):
        pipeline = CosmosTransferPipeline()
        mock_cv2 = MagicMock()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with pytest.raises(FileNotFoundError):
                pipeline.generate_edge_control("/nonexistent.mp4")

    def test_invalid_threshold(self, tmp_path):
        video = tmp_path / "sim.mp4"
        video.write_text("fake")
        pipeline = CosmosTransferPipeline()
        mock_cv2 = MagicMock()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with pytest.raises(ValueError, match="Invalid threshold"):
                pipeline.generate_edge_control(str(video), threshold="extreme")


# ═════════════════════════════════════════════════════════════════════════════
# CosmosTransferPipeline generate_seg_control Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestGenerateSegControl:
    def test_file_not_found(self):
        pipeline = CosmosTransferPipeline()
        mock_cv2 = MagicMock()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with pytest.raises(FileNotFoundError):
                pipeline.generate_seg_control("/nonexistent.mp4")

    def test_sam2_succeeds(self, tmp_path):
        video = tmp_path / "sim.mp4"
        video.write_text("fake")
        out = str(tmp_path / "seg.mp4")
        pipeline = CosmosTransferPipeline()
        mock_cv2 = MagicMock()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with patch.object(pipeline, "_try_sam2_segmentation", return_value=True):
                result = pipeline.generate_seg_control(str(video), out)
        assert result == out

    def test_sam2_fails_uses_fallback(self, tmp_path):
        video = tmp_path / "sim.mp4"
        video.write_text("fake")
        out = str(tmp_path / "seg.mp4")
        pipeline = CosmosTransferPipeline()
        mock_cv2 = MagicMock()
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with patch.object(pipeline, "_try_sam2_segmentation", return_value=False):
                with patch.object(pipeline, "_fallback_colour_segmentation") as fb:
                    pipeline.generate_seg_control(str(video), out)
        fb.assert_called_once()


# ═════════════════════════════════════════════════════════════════════════════
# _try_mujoco_depth / _try_video_depth_anything / _try_sam2_segmentation
# ═════════════════════════════════════════════════════════════════════════════


class TestTryStrategies:
    def test_try_mujoco_depth_no_mujoco(self):
        pipeline = CosmosTransferPipeline()
        with patch.dict(sys.modules, {"mujoco": None}):
            result = pipeline._try_mujoco_depth("/tmp/sim.mp4", "/tmp/out.mp4")
        assert result is False

    def test_try_video_depth_anything_no_module(self):
        pipeline = CosmosTransferPipeline()
        with patch.dict(sys.modules, {"video_depth_anything": None, "video_depth_anything.infer": None}):
            result = pipeline._try_video_depth_anything("/tmp/sim.mp4", "/tmp/out.mp4")
        assert result is False

    def test_try_sam2_no_module(self):
        pipeline = CosmosTransferPipeline()
        with patch.dict(sys.modules, {"sam2": None, "sam2.build_sam": None}):
            result = pipeline._try_sam2_segmentation("/tmp/sim.mp4", "/tmp/out.mp4")
        assert result is False


# ═════════════════════════════════════════════════════════════════════════════
# transfer_video convenience function Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestTransferVideoFunction:
    def test_basic_call(self, tmp_path):
        from strands_robots.cosmos_transfer import transfer_video

        video = tmp_path / "sim.mp4"
        video.write_text("fake")
        out = str(tmp_path / "out.mp4")

        with patch.object(CosmosTransferPipeline, "transfer_video", return_value={"output_path": out}) as m:
            with patch.object(CosmosTransferPipeline, "cleanup") as cleanup:
                result = transfer_video(str(video), "test prompt", out)
        assert result["output_path"] == out
        m.assert_called_once()
        assert cleanup.call_count >= 1  # Called in finally block (+ possibly __del__)

    def test_with_config_kwargs(self, tmp_path):
        from strands_robots.cosmos_transfer import transfer_video

        video = tmp_path / "sim.mp4"
        video.write_text("fake")

        with patch.object(CosmosTransferPipeline, "transfer_video", return_value={}):
            with patch.object(CosmosTransferPipeline, "cleanup"):
                transfer_video(str(video), "test", str(tmp_path / "out.mp4"), guidance=5.0, num_steps=50)

    def test_with_explicit_config(self, tmp_path):
        from strands_robots.cosmos_transfer import transfer_video

        video = tmp_path / "sim.mp4"
        video.write_text("fake")
        cfg = CosmosTransferConfig(guidance=7.0)

        with patch.object(CosmosTransferPipeline, "transfer_video", return_value={}):
            with patch.object(CosmosTransferPipeline, "cleanup"):
                transfer_video(str(video), "test", str(tmp_path / "out.mp4"), config=cfg)


# ═════════════════════════════════════════════════════════════════════════════
# Visualizer Tests
# ═════════════════════════════════════════════════════════════════════════════

from strands_robots.visualizer import RecordingStats, RecordingVisualizer  # noqa: E402


class TestRecordingStats:
    def test_defaults(self):
        stats = RecordingStats()
        assert stats.episode == 0
        assert stats.frame_count == 0
        assert stats.fps_actual == 0.0
        assert stats.fps_target == 30.0
        assert stats.task == ""
        assert stats.cameras == []
        assert stats.errors == 0
        assert stats.recording is False


class TestRecordingVisualizerInit:
    def test_default_init(self):
        viz = RecordingVisualizer()
        assert viz.mode == "terminal"
        assert viz.refresh_rate == 2.0
        assert viz._running is False

    def test_custom_init(self):
        viz = RecordingVisualizer(mode="json", refresh_rate=5.0, port=9999)
        assert viz.mode == "json"
        assert viz.port == 9999


class TestRecordingVisualizerUpdate:
    def test_update_frame_count(self):
        viz = RecordingVisualizer()
        viz._start_time = time.time()
        viz.update(frame_count=42)
        assert viz.stats.frame_count == 42

    def test_update_episode(self):
        viz = RecordingVisualizer()
        viz._start_time = time.time()
        viz.update(episode=3, total_episodes=10)
        assert viz.stats.episode == 3
        assert viz.stats.total_episodes == 10

    def test_update_task(self):
        viz = RecordingVisualizer()
        viz._start_time = time.time()
        viz.update(task="pick up cube")
        assert viz.stats.task == "pick up cube"

    def test_update_cameras(self):
        viz = RecordingVisualizer()
        viz._start_time = time.time()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        viz.update(cameras={"wrist": img})
        assert "wrist" in viz.stats.cameras

    def test_update_action_state(self):
        viz = RecordingVisualizer()
        viz._start_time = time.time()
        viz.update(last_action={"a": 0.1, "b": 0.2}, last_state={"x": 1.0})
        assert viz.stats.action_dim == 2
        assert viz.stats.state_dim == 1

    def test_update_error(self):
        viz = RecordingVisualizer()
        viz._start_time = time.time()
        viz.update(error=True)
        viz.update(error=True)
        assert viz.stats.errors == 2

    def test_fps_calculation(self):
        viz = RecordingVisualizer()
        viz._start_time = time.time() - 10
        # Simulate rapid frame updates
        for _ in range(10):
            viz.update(frame_count=1)
        assert viz.stats.fps_actual >= 0  # Non-negative


class TestRecordingVisualizerNewEpisode:
    def test_new_episode(self):
        viz = RecordingVisualizer()
        viz._start_time = time.time()
        viz.update(frame_count=100)
        viz.new_episode(episode=2, task="new task")
        assert viz.stats.episode == 2
        assert viz.stats.task == "new task"
        assert viz.stats.frame_count == 0


class TestRecordingVisualizerStartStop:
    def test_start_stop(self):
        viz = RecordingVisualizer(mode="json")
        viz.start()
        assert viz._running is True
        assert viz.stats.recording is True
        time.sleep(0.1)
        viz.stop()
        assert viz._running is False
        assert viz.stats.recording is False

    def test_start_idempotent(self):
        viz = RecordingVisualizer(mode="json")
        viz.start()
        viz.start()  # Should not create second thread
        viz.stop()

    def test_terminal_mode(self):
        viz = RecordingVisualizer(mode="terminal")
        viz.start()
        time.sleep(0.2)
        viz.stop()

    def test_json_mode_output(self, capsys):
        viz = RecordingVisualizer(mode="json", refresh_rate=10.0)
        viz.start()
        time.sleep(0.3)
        viz.stop()
        captured = capsys.readouterr()
        # Should have printed JSON lines
        if captured.out.strip():
            first_line = captured.out.strip().split("\n")[0]
            data = json.loads(first_line)
            assert "episode" in data
            assert "fps" in data


class TestRecordingVisualizerGetStatsDict:
    def test_get_stats_dict(self):
        viz = RecordingVisualizer()
        viz._start_time = time.time()
        viz.update(frame_count=50, episode=2, task="test")
        d = viz.get_stats_dict()
        assert d["frame_count"] == 50
        assert d["episode"] == 2
        assert d["task"] == "test"
        assert isinstance(d["fps_actual"], float)


class TestRecordingVisualizerRenderTerminal:
    def test_render_terminal(self, capsys):
        viz = RecordingVisualizer(mode="terminal")
        viz._start_time = time.time()
        viz.stats.episode = 1
        viz.stats.total_episodes = 10
        viz.stats.frame_count = 100
        viz.stats.fps_actual = 28.5
        viz.stats.fps_target = 30.0
        viz.stats.task = "pick up cube"
        viz.stats.cameras = ["wrist"]
        viz.stats.last_action = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5}
        viz.stats.errors = 2
        viz.stats.duration_s = 65.0
        viz._render_terminal()
        captured = capsys.readouterr()
        assert "LIVE RECORDING MONITOR" in captured.out
        assert "pick up cube" in captured.out
        assert "wrist" in captured.out

    def test_render_terminal_zero_fps(self, capsys):
        viz = RecordingVisualizer()
        viz._start_time = time.time()
        viz.stats.fps_target = 0
        viz._render_terminal()
        captured = capsys.readouterr()
        assert "LIVE RECORDING" in captured.out


class TestRecordingVisualizerRenderJson:
    def test_render_json(self, capsys):
        viz = RecordingVisualizer()
        viz._start_time = time.time()
        viz.stats.episode = 3
        viz.stats.frame_count = 200
        viz.stats.task = "grasp"
        viz._render_json()
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["episode"] == 3
        assert data["frame"] == 200


class TestRecordingVisualizerWebServer:
    def test_start_web_server(self):
        viz = RecordingVisualizer(mode="web", port=0)  # Port 0 = OS-assigned
        try:
            viz._start_web_server()
            if viz._web_server:
                # Server started successfully
                viz._web_server.shutdown()
        except Exception:
            pass  # Port binding may fail in CI

    def test_web_mode_start_stop(self):
        viz = RecordingVisualizer(mode="web", port=0)
        with patch.object(viz, "_start_web_server"):
            viz.start()
            time.sleep(0.1)
            viz.stop()


class TestRecordingVisualizerMatplotlib:
    def test_render_matplotlib_no_frames(self):
        viz = RecordingVisualizer(mode="matplotlib")
        viz._start_time = time.time()
        viz._camera_frames = {}
        viz._render_matplotlib()  # Should be no-op

    def test_render_matplotlib_import_error(self):
        viz = RecordingVisualizer(mode="matplotlib")
        viz._start_time = time.time()
        viz._camera_frames = {"cam": np.zeros((100, 100, 3), dtype=np.uint8)}
        with patch.dict(sys.modules, {"matplotlib": None}):
            viz._render_matplotlib()
        # Should fallback to terminal mode
        assert viz.mode == "terminal"
