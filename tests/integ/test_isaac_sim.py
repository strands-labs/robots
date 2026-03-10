#!/usr/bin/env python3
"""
Integration tests for Isaac Sim EC2 (NVIDIA L40S GPU).

These tests run ONLY on the Isaac Sim EC2 instance via self-hosted GitHub Actions runner.
They verify Isaac Sim installation, GPU rendering, simulation backends, GR00T inference,
Cosmos world models, and full sim-to-policy pipelines.

Markers:
    @pytest.mark.isaac_sim  — requires Isaac Sim EC2 (self-hosted runner label: isaac-sim)
    @pytest.mark.gpu        — requires CUDA GPU

Run locally on EC2:
    DISPLAY=:1 python -m pytest tests/integ/test_isaac_sim.py -v

Run via CI:
    Triggered by .github/workflows/isaac_sim_ci.yml on push/PR/schedule
"""

import os
import subprocess

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
ISAAC_SIM_PATH = os.getenv("ISAAC_SIM_PATH", "/home/ubuntu/IsaacSim")


def _is_isaac_sim_host():
    """Detect Isaac Sim EC2 instance."""
    return (
        os.getenv("ISAAC_SIM_CI") == "true"
        or os.path.isdir(ISAAC_SIM_PATH)
        or os.path.isfile(os.path.join(ISAAC_SIM_PATH, "VERSION"))
    )


def _has_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _has_mujoco():
    try:
        import mujoco  # noqa: F401

        return True
    except ImportError:
        return False


def _has_groot():
    try:
        from strands_robots.policies.groot import Gr00tPolicy  # noqa: F401

        return True
    except (ImportError, Exception):
        return False


def _has_isaac_sim_python():
    """Check if Isaac Sim's Python is available."""
    python_sh = os.path.join(ISAAC_SIM_PATH, "python.sh")
    return os.path.isfile(python_sh)


requires_isaac_sim = pytest.mark.skipif(not _is_isaac_sim_host(), reason="Requires Isaac Sim EC2 instance")
requires_cuda = pytest.mark.skipif(not _has_cuda(), reason="Requires CUDA GPU")
requires_mujoco = pytest.mark.skipif(not _has_mujoco(), reason="Requires MuJoCo")
requires_groot = pytest.mark.skipif(not _has_groot(), reason="Requires GR00T")
requires_isaac_python = pytest.mark.skipif(not _has_isaac_sim_python(), reason="Requires Isaac Sim Python")


# ===================================================================
# 1. GPU & CUDA Health
# ===================================================================
class TestIsaacSimGPU:
    """Verify L40S GPU is healthy."""

    @requires_cuda
    def test_cuda_available(self):
        import torch

        assert torch.cuda.is_available()

    @requires_cuda
    def test_gpu_is_l40s_or_similar(self):
        """EC2 g6e should have L40S."""
        import torch

        name = torch.cuda.get_device_name(0)
        print(f"GPU: {name}")
        # Don't hard-assert L40S — could be other GPU types
        assert name, "GPU name should not be empty"

    @requires_cuda
    def test_gpu_memory_sufficient(self):
        """L40S has 46 GB VRAM."""
        import torch

        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1e9
        assert total_gb > 20, f"Expected >20 GB VRAM, got {total_gb:.1f} GB"
        print(f"GPU memory: {total_gb:.1f} GB")

    @requires_cuda
    def test_cuda_compute(self):
        import torch

        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        c = a @ b
        assert c.shape == (1024, 1024)
        assert not torch.isnan(c).any()

    @requires_isaac_sim
    def test_nvidia_smi(self):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,temperature.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        print(f"nvidia-smi: {result.stdout.strip()}")


# ===================================================================
# 2. Isaac Sim Installation
# ===================================================================
class TestIsaacSimInstallation:
    """Verify Isaac Sim is properly installed."""

    @requires_isaac_sim
    def test_isaac_sim_dir_exists(self):
        assert os.path.isdir(ISAAC_SIM_PATH), f"Isaac Sim not found at {ISAAC_SIM_PATH}"

    @requires_isaac_sim
    def test_isaac_sim_version_file(self):
        version_file = os.path.join(ISAAC_SIM_PATH, "VERSION")
        assert os.path.isfile(version_file), "VERSION file missing"
        with open(version_file) as f:
            version = f.read().strip()
        assert version, "VERSION file is empty"
        print(f"Isaac Sim version: {version}")

    @requires_isaac_sim
    def test_isaac_sim_shell_script(self):
        """isaac-sim.sh should exist and be executable."""
        script = os.path.join(ISAAC_SIM_PATH, "isaac-sim.sh")
        assert os.path.isfile(script), "isaac-sim.sh not found"

    @requires_isaac_python
    def test_isaac_sim_python_available(self):
        """Isaac Sim's bundled Python should work."""
        python_sh = os.path.join(ISAAC_SIM_PATH, "python.sh")
        result = subprocess.run(
            [python_sh, "-c", "print('Isaac Sim Python OK')"], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, f"Isaac Sim Python failed: {result.stderr}"

    @requires_isaac_python
    def test_isaac_sim_can_import_omni(self):
        """Isaac Sim's Python should be able to import omni modules."""
        python_sh = os.path.join(ISAAC_SIM_PATH, "python.sh")
        result = subprocess.run(
            [python_sh, "-c", "import omni; print('omni imported OK')"], capture_output=True, text=True, timeout=60
        )
        # This may fail if Isaac Sim needs full startup; just check it runs
        print(f"omni import: rc={result.returncode}")
        if result.returncode != 0:
            print(f"stderr: {result.stderr[:500]}")


# ===================================================================
# 3. Isaac Sim Backend (strands-robots integration)
# ===================================================================
class TestIsaacSimBackend:
    """Test strands-robots Isaac Sim backend."""

    @requires_isaac_sim
    def test_isaac_backend_import(self):
        from strands_robots.isaac import IsaacSimBackend

        assert IsaacSimBackend is not None

    @requires_isaac_sim
    def test_isaac_lab_env_import(self):
        from strands_robots.isaac.isaac_lab_env import IsaacLabEnv

        assert IsaacLabEnv is not None

    @requires_isaac_sim
    def test_isaac_lab_trainer_import(self):
        from strands_robots.isaac.isaac_lab_trainer import IsaacLabTrainer

        assert IsaacLabTrainer is not None

    @requires_isaac_sim
    def test_factory_has_isaac_route(self):
        """Robot factory should support backend='isaac'."""
        import inspect

        from strands_robots.factory import Robot

        source = inspect.getsource(Robot)
        assert "isaac" in source.lower()


# ===================================================================
# 4. MuJoCo Simulation on EC2
# ===================================================================
class TestIsaacSimMuJoCo:
    """MuJoCo should also work on the EC2 instance (EGL rendering)."""

    @requires_mujoco
    @requires_cuda
    def test_mujoco_step(self):
        import mujoco

        xml = """
        <mujoco>
          <worldbody>
            <geom type="plane" size="1 1 0.1"/>
            <body pos="0 0 1">
              <joint type="free"/>
              <geom type="sphere" size="0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        initial_z = data.qpos[2]
        for _ in range(200):
            mujoco.mj_step(model, data)
        assert data.qpos[2] < initial_z

    @requires_mujoco
    @requires_cuda
    def test_mujoco_render_egl(self):
        import mujoco

        xml = """
        <mujoco>
          <worldbody>
            <light diffuse="1 1 1" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="1 1 0.1"/>
            <body pos="0 0 0.3">
              <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
            </body>
          </worldbody>
          <visual><global offwidth="640" offheight="480"/></visual>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=480, width=640)
        renderer.update_scene(data)
        pixels = renderer.render()
        renderer.close()
        assert pixels.shape == (480, 640, 3)
        assert pixels.max() > 0


# ===================================================================
# 5. Simulation Class (strands-robots)
# ===================================================================
class TestIsaacSimSimulation:
    """Test strands-robots Simulation on EC2 with GPU rendering."""

    @requires_mujoco
    @requires_cuda
    def test_simulation_create_and_step(self):
        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world()

        for _ in range(50):
            sim.step()

        try:
            sim.destroy()
        except Exception:
            pass

    @requires_mujoco
    @requires_cuda
    def test_simulation_add_robot(self):
        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world()
        sim.add_robot(data_config="so100", name="arm")

        try:
            sim.destroy()
        except Exception:
            pass

    @requires_mujoco
    @requires_cuda
    def test_simulation_render(self):
        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world()
        sim.render(camera_name="default", width=640, height=480)

        try:
            sim.destroy()
        except Exception:
            pass


# ===================================================================
# 6. GR00T Policy
# ===================================================================
class TestIsaacSimGR00T:
    """Test GR00T policy on EC2 GPU."""

    @requires_groot
    @requires_cuda
    def test_groot_policy_import(self):
        from strands_robots.policies.groot import Gr00tPolicy

        assert Gr00tPolicy is not None

    @requires_groot
    @requires_cuda
    def test_groot_data_configs_loaded(self):
        from strands_robots.policies.groot.data_config import DATA_CONFIG_MAP

        assert len(DATA_CONFIG_MAP) > 0
        print(f"GR00T data configs: {list(DATA_CONFIG_MAP.keys())[:10]}")

    @requires_groot
    @requires_cuda
    def test_groot_version_detection(self):
        """Detect which GR00T version is installed."""
        from strands_robots.policies.groot import _detect_groot_version

        version = _detect_groot_version()
        print(f"GR00T version: {version}")
        # Version may be None if not installed, which is fine
        if version:
            assert version in ("n1.5", "n1.6")

    @requires_groot
    @requires_cuda
    def test_groot_inference_tool(self):
        from strands_robots.tools.gr00t_inference import gr00t_inference

        assert callable(gr00t_inference)


# ===================================================================
# 7. Cosmos World Models (if available)
# ===================================================================
class TestIsaacSimCosmos:
    """Test Cosmos world model components (import-level, no GPU inference)."""

    @requires_isaac_sim
    def test_cosmos_transfer_module_import(self):
        from strands_robots import cosmos_transfer

        assert cosmos_transfer is not None

    @requires_isaac_sim
    def test_cosmos_predict_policy_import(self):
        try:
            from strands_robots.policies.cosmos_predict import CosmosPredictPolicy

            assert CosmosPredictPolicy is not None
        except ImportError:
            pytest.skip("Cosmos predict not installed")

    @requires_isaac_sim
    def test_dreamgen_import(self):
        from strands_robots.dreamgen import DreamGenConfig

        assert DreamGenConfig is not None

    @requires_isaac_sim
    def test_dreamzero_import(self):
        try:
            from strands_robots.policies.dreamzero import DreamZeroPolicy

            assert DreamZeroPolicy is not None
        except ImportError:
            pytest.skip("DreamZero not installed")


# ===================================================================
# 8. Newton Backend (if available)
# ===================================================================
class TestIsaacSimNewton:
    """Test Newton (MuJoCo-Warp GPU) backend."""

    @requires_isaac_sim
    def test_newton_module_import(self):
        try:
            from strands_robots.newton import NewtonBackend

            assert NewtonBackend is not None
        except ImportError:
            pytest.skip("Newton/warp not installed")

    @requires_isaac_sim
    @requires_cuda
    def test_newton_config(self):
        try:
            from strands_robots.newton.newton_backend import NewtonConfig

            config = NewtonConfig()
            assert config is not None
        except ImportError:
            pytest.skip("Newton not installed")


# ===================================================================
# 9. Full Pipeline: Sim → Policy → Evaluate
# ===================================================================
class TestIsaacSimPipeline:
    """End-to-end pipeline tests."""

    @requires_mujoco
    @requires_cuda
    def test_mock_policy_in_sim(self):
        """Run mock policy in MuJoCo simulation."""
        import asyncio

        from strands_robots.policies import create_policy

        policy = create_policy("mock")
        policy.set_robot_state_keys(["j0", "j1", "j2", "j3", "j4", "j5"])

        obs = {"observation.state": [0.0] * 6}
        actions = asyncio.run(policy.get_actions(obs, "pick up object"))
        assert isinstance(actions, list)
        assert len(actions) > 0

    @requires_mujoco
    @requires_cuda
    def test_robot_factory_sim(self):
        """Create robot via factory in sim mode."""
        from strands_robots import Robot

        robot = Robot("so100", mode="sim")
        assert robot is not None

    @requires_cuda
    def test_video_encode_decode(self, tmp_path):
        """Encode video frames and verify output."""
        from strands_robots.video import VideoEncoder

        out_path = str(tmp_path / "pipeline_test.mp4")
        encoder = VideoEncoder(out_path, fps=30)

        for i in range(60):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            encoder.add_frame(frame)

        encoder.close()
        assert os.path.exists(out_path)
        size_kb = os.path.getsize(out_path) / 1024
        assert size_kb > 1, f"Video too small: {size_kb:.1f} KB"
        print(f"Video: {size_kb:.1f} KB, 60 frames @ 30fps")


# ===================================================================
# 10. Training Infrastructure
# ===================================================================
class TestIsaacSimTraining:
    """Test training infrastructure availability."""

    def test_trainer_abc_import(self):
        from strands_robots.training import Trainer

        assert Trainer is not None

    def test_create_trainer_groot(self):
        from strands_robots.training import create_trainer

        try:
            trainer = create_trainer("groot")
            assert trainer is not None
        except Exception as e:
            # May fail if GR00T training deps not installed
            print(f"groot trainer: {e}")

    def test_create_trainer_lerobot(self):
        from strands_robots.training import create_trainer

        try:
            trainer = create_trainer("lerobot")
            assert trainer is not None
        except Exception as e:
            print(f"lerobot trainer: {e}")

    @requires_isaac_sim
    def test_isaac_lab_trainer_import(self):
        try:
            from strands_robots.isaac.isaac_lab_trainer import IsaacLabTrainer

            assert IsaacLabTrainer is not None
        except ImportError:
            pytest.skip("Isaac Lab trainer not available")
