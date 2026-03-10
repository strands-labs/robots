"""Tests for training pipeline, dataset recording, GPU backends, and utilities."""


class TestDatasetRecorder:
    """Test dataset recording in LeRobot format."""

    def test_import_dataset_recorder(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        assert DatasetRecorder is not None

    def test_recorder_init(self):
        """DatasetRecorder should be instantiable with basic params."""
        from strands_robots.dataset_recorder import DatasetRecorder

        # Just verify class exists and has __init__
        assert callable(DatasetRecorder)


class TestTrainingPipeline:
    """Test the training module."""

    def test_import_trainers(self):
        from strands_robots.training import TrainConfig, Trainer

        assert TrainConfig is not None
        assert Trainer is not None

    def test_trainer_is_abstract(self):
        # Trainer should be abstract (ABC)
        import inspect

        from strands_robots.training import Trainer

        assert inspect.isabstract(Trainer) or hasattr(Trainer, "train")

    def test_concrete_trainers_exist(self):
        from strands_robots.training import (
            DreamgenIdmTrainer,
            DreamgenVlaTrainer,
            Gr00tTrainer,
            LerobotTrainer,
        )

        assert Gr00tTrainer is not None
        assert LerobotTrainer is not None
        assert DreamgenIdmTrainer is not None
        assert DreamgenVlaTrainer is not None

    def test_evaluate_function(self):
        from strands_robots.training import evaluate

        assert callable(evaluate)


class TestKinematics:
    """Test kinematics module."""

    def test_import_kinematics(self):
        from strands_robots import kinematics

        assert kinematics is not None


class TestProcessor:
    """Test observation processor."""

    def test_import_processor(self):
        from strands_robots.processor import ProcessedPolicy, ProcessorBridge

        assert ProcessorBridge is not None
        assert ProcessedPolicy is not None

    def test_create_processor_bridge(self):
        from strands_robots.processor import create_processor_bridge

        assert callable(create_processor_bridge)


class TestRobometer:
    """Test robometer benchmarking."""

    def test_import_robometer_benchmark(self):
        from strands_robots.robometer import RobometerBenchmark

        assert RobometerBenchmark is not None

    def test_import_robometer_trainer(self):
        from strands_robots.robometer import RobometerTrainer

        assert RobometerTrainer is not None

    def test_reward_function(self):
        from strands_robots.robometer import robometer_reward_fn

        assert callable(robometer_reward_fn)


class TestMotionLibrary:
    """Test motion library."""

    def test_import_motion_library(self):
        from strands_robots.motion_library import MotionLibrary

        assert MotionLibrary is not None


class TestVisualizer:
    """Test visualization module."""

    def test_import_visualizer(self):
        from strands_robots.visualizer import RecordingStats, RecordingVisualizer

        assert RecordingVisualizer is not None
        assert RecordingStats is not None


class TestVideo:
    """Test video recording."""

    def test_import_video_encoder(self):
        from strands_robots.video import VideoEncoder

        assert VideoEncoder is not None

    def test_encode_frames_function(self):
        from strands_robots.video import encode_frames

        assert callable(encode_frames)

    def test_video_info_function(self):
        from strands_robots.video import get_video_info

        assert callable(get_video_info)


class TestIsaacBackend:
    """Test Isaac Sim/Lab backend imports."""

    def test_import_isaac_package(self):
        import strands_robots.isaac

        assert strands_robots.isaac is not None

    def test_isaac_modules_exist(self):
        from strands_robots.isaac import isaac_gym_env, isaac_sim_backend

        assert isaac_sim_backend is not None
        assert isaac_gym_env is not None

    def test_isaac_lab_modules(self):
        from strands_robots.isaac import isaac_lab_env, isaac_lab_trainer

        assert isaac_lab_env is not None
        assert isaac_lab_trainer is not None


class TestNewtonBackend:
    """Test Newton differentiable sim backend."""

    def test_import_newton_package(self):
        import strands_robots.newton

        assert strands_robots.newton is not None

    def test_newton_modules(self):
        from strands_robots.newton import newton_backend, newton_gym_env

        assert newton_backend is not None
        assert newton_gym_env is not None


class TestTelemetry:
    """Test telemetry module."""

    def test_import_telemetry(self):
        import strands_robots.telemetry

        assert strands_robots.telemetry is not None

    def test_telemetry_stream(self):
        from strands_robots.telemetry.stream import TelemetryStream

        assert TelemetryStream is not None


class TestCosmosTransfer:
    """Test Cosmos sim-to-real transfer."""

    def test_import_cosmos_transfer(self):
        import strands_robots.cosmos_transfer

        assert strands_robots.cosmos_transfer is not None


class TestStereo:
    """Test stereo depth module."""

    def test_import_stereo(self):
        import strands_robots.stereo

        assert strands_robots.stereo is not None
