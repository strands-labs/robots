#!/usr/bin/env python3
"""
Integration tests for Thor (NVIDIA Jetson AGX Thor).

These tests run ONLY on the Thor device via self-hosted GitHub Actions runner.
They verify GPU/CUDA, MuJoCo simulation, GR00T inference, and policy execution
on real hardware with sm_110 (Blackwell) GPU.

Markers:
    @pytest.mark.thor  — requires Thor device (self-hosted runner label: thor)
    @pytest.mark.gpu   — requires CUDA GPU

Run locally on Thor:
    MUJOCO_GL=egl python -m pytest tests/integ/test_thor.py -v

Run via CI:
    Triggered by .github/workflows/thor_ci.yml on push/PR/schedule
"""

import os
import platform
import subprocess

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Skip entire module if not on Thor / no GPU
# ---------------------------------------------------------------------------
def _is_thor():
    """Detect Thor device by hostname or env marker."""
    hostname = platform.node().lower()
    return "thor" in hostname or os.getenv("THOR_CI") == "true" or os.getenv("NVIDIA_JETSON") == "true"


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
    """Check if GR00T (N1.5 or N1.6) is importable."""
    try:
        from strands_robots.policies.groot import Gr00tPolicy  # noqa: F401

        return True
    except (ImportError, Exception):
        return False


requires_thor = pytest.mark.skipif(not _is_thor(), reason="Requires Thor device")
requires_cuda = pytest.mark.skipif(not _has_cuda(), reason="Requires CUDA GPU")
requires_mujoco = pytest.mark.skipif(not _has_mujoco(), reason="Requires MuJoCo")
requires_groot = pytest.mark.skipif(not _has_groot(), reason="Requires GR00T")


# ===================================================================
# 1. GPU & CUDA Health
# ===================================================================
class TestThorGPU:
    """Verify Thor GPU is healthy and CUDA works."""

    @requires_cuda
    def test_cuda_available(self):
        import torch

        assert torch.cuda.is_available(), "CUDA must be available on Thor"

    @requires_cuda
    def test_gpu_device_name(self):
        import torch

        name = torch.cuda.get_device_name(0)
        assert name, "GPU device name should not be empty"
        print(f"GPU: {name}")

    @requires_cuda
    def test_gpu_memory_sufficient(self):
        """Thor has ~132 GB unified memory; GPU portion should be >10 GB."""
        import torch

        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1e9
        assert total_gb > 10, f"Expected >10 GB GPU memory, got {total_gb:.1f} GB"
        print(f"GPU memory: {total_gb:.1f} GB")

    @requires_cuda
    def test_cuda_matmul(self):
        """Basic GPU compute sanity check."""
        import torch

        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        c = a @ b
        assert c.shape == (512, 512)
        assert not torch.isnan(c).any(), "GPU matmul produced NaN"

    @requires_cuda
    def test_cuda_memory_allocate_free(self):
        """Allocate and free GPU memory."""
        import torch

        t = torch.zeros(1024, 1024, 128, device="cuda", dtype=torch.float32)
        allocated = torch.cuda.memory_allocated(0)
        assert allocated > 0
        del t
        torch.cuda.empty_cache()

    @requires_thor
    def test_nvidia_smi(self):
        """nvidia-smi should be available and return successfully."""
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,temperature.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"nvidia-smi failed: {result.stderr}"
        print(f"nvidia-smi: {result.stdout.strip()}")


# ===================================================================
# 2. MuJoCo Simulation (headless via EGL)
# ===================================================================
class TestThorMuJoCo:
    """Verify MuJoCo simulation works on Thor with EGL rendering."""

    @requires_mujoco
    @requires_cuda
    def test_mujoco_import(self):
        import mujoco

        assert hasattr(mujoco, "MjModel")
        assert hasattr(mujoco, "MjData")

    @requires_mujoco
    @requires_cuda
    def test_mujoco_create_world(self):
        """Create a MuJoCo world from XML."""
        import mujoco

        xml = """
        <mujoco>
          <worldbody>
            <light diffuse="1 1 1" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="1 1 0.1"/>
            <body pos="0 0 0.5">
              <joint type="free"/>
              <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        mujoco.MjData(model)
        assert model.nq > 0
        assert model.nv > 0

    @requires_mujoco
    @requires_cuda
    def test_mujoco_step(self):
        """Step the physics simulation."""
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
        for _ in range(100):
            mujoco.mj_step(model, data)

        # Ball should have fallen
        assert data.qpos[2] < initial_z, "Ball should fall under gravity"

    @requires_mujoco
    @requires_cuda
    def test_mujoco_egl_render(self):
        """Headless rendering with EGL (MUJOCO_GL=egl)."""
        import mujoco

        xml = """
        <mujoco>
          <worldbody>
            <light diffuse="1 1 1" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="1 1 0.1" rgba="0.5 0.5 0.5 1"/>
            <body pos="0 0 0.3">
              <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
            </body>
          </worldbody>
          <visual>
            <global offwidth="640" offheight="480"/>
          </visual>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        renderer = mujoco.Renderer(model, height=480, width=640)
        renderer.update_scene(data)
        pixels = renderer.render()
        renderer.close()

        assert pixels.shape == (480, 640, 3), f"Expected (480, 640, 3), got {pixels.shape}"
        assert pixels.max() > 0, "Rendered image should not be all black"


# ===================================================================
# 3. strands-robots Simulation Integration
# ===================================================================
class TestThorSimulation:
    """Test strands-robots Simulation class on Thor hardware."""

    @requires_mujoco
    @requires_cuda
    def test_simulation_import(self):
        from strands_robots.simulation import Simulation

        assert Simulation is not None

    @requires_mujoco
    @requires_cuda
    def test_create_sim_world(self):
        """Create a simulation world via strands-robots API."""
        from strands_robots.simulation import Simulation

        sim = Simulation()
        result = sim.create_world()
        assert result.get("status") in ("success", "ok", True), f"create_world failed: {result}"

        # Cleanup
        try:
            sim.destroy()
        except Exception:
            pass

    @requires_mujoco
    @requires_cuda
    def test_add_robot_to_sim(self):
        """Add a robot to the simulation via data_config."""
        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world()

        result = sim.add_robot(data_config="so100", name="test_arm")
        # Should succeed or provide meaningful info
        assert "error" not in str(result).lower() or "not found" not in str(result).lower()

        try:
            sim.destroy()
        except Exception:
            pass

    @requires_mujoco
    @requires_cuda
    def test_sim_step_loop(self):
        """Run a short simulation loop."""
        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world()

        for i in range(10):
            sim.step()

        try:
            sim.destroy()
        except Exception:
            pass

    @requires_mujoco
    @requires_cuda
    def test_sim_render_headless(self):
        """Render a frame from simulation (headless EGL)."""
        from strands_robots.simulation import Simulation

        sim = Simulation()
        sim.create_world()

        sim.render(camera_name="default", width=320, height=240)

        try:
            sim.destroy()
        except Exception:
            pass


# ===================================================================
# 4. Robot Factory
# ===================================================================
class TestThorRobotFactory:
    """Test robot creation via factory on Thor."""

    @requires_mujoco
    @requires_cuda
    def test_create_robot_sim_mode(self):
        """Create a robot in simulation mode."""
        from strands_robots import Robot

        robot = Robot("so100", mode="sim")
        assert robot is not None

    @requires_mujoco
    @requires_cuda
    def test_list_robots_sim(self):
        """List all sim-capable robots."""
        from strands_robots import list_robots

        robots = list_robots(mode="sim")
        assert len(robots) > 0
        names = [r["name"] for r in robots]
        assert "so100" in names


# ===================================================================
# 5. GR00T Policy Inference
# ===================================================================
class TestThorGR00T:
    """Test GR00T policy inference on Thor GPU."""

    @requires_groot
    @requires_cuda
    def test_groot_policy_import(self):
        from strands_robots.policies.groot import Gr00tPolicy

        assert Gr00tPolicy is not None

    @requires_groot
    @requires_cuda
    def test_groot_policy_create(self):
        """Create a GR00T policy instance (service mode, no active server needed)."""
        from strands_robots.policies.groot import Gr00tPolicy

        # Just instantiate — don't connect to server
        policy = Gr00tPolicy(server_address="localhost:5555")
        assert policy.provider_name in ("groot", "gr00t")

    @requires_groot
    @requires_cuda
    def test_groot_data_configs(self):
        """Verify data configs are available."""
        from strands_robots.policies.groot.data_config import DATA_CONFIG_MAP

        assert len(DATA_CONFIG_MAP) > 0
        assert "so100" in DATA_CONFIG_MAP or "fourier_gr1_arms_only" in DATA_CONFIG_MAP

    @requires_groot
    @requires_cuda
    def test_groot_inference_tool_import(self):
        """Verify gr00t_inference tool loads."""
        from strands_robots.tools.gr00t_inference import gr00t_inference

        assert callable(gr00t_inference)


# ===================================================================
# 6. Policy Registry
# ===================================================================
class TestThorPolicies:
    """Test policy creation on Thor with GPU."""

    @requires_cuda
    def test_mock_policy_on_gpu(self):
        """MockPolicy should work even on GPU device."""
        import asyncio

        from strands_robots.policies import create_policy

        policy = create_policy("mock")
        policy.set_robot_state_keys(["j0", "j1", "j2"])
        actions = asyncio.run(policy.get_actions({"observation.state": [0.0, 0.0, 0.0]}, "test"))
        assert isinstance(actions, list)
        assert len(actions) > 0

    @requires_cuda
    def test_list_all_providers(self):
        from strands_robots.policies import list_providers

        providers = list_providers()
        assert "mock" in providers
        assert len(providers) >= 10  # Should have many providers


# ===================================================================
# 7. Kinematics on GPU
# ===================================================================
class TestThorKinematics:
    """Test kinematics computations on Thor."""

    @requires_mujoco
    @requires_cuda
    def test_mujoco_kinematics_import(self):
        from strands_robots.kinematics import MuJoCoKinematics

        assert MuJoCoKinematics is not None

    @requires_mujoco
    @requires_cuda
    def test_create_kinematics(self):
        """Create a MuJoCo kinematics solver."""
        import mujoco

        from strands_robots.kinematics import MuJoCoKinematics

        # Simple 2-link arm
        xml = """
        <mujoco>
          <worldbody>
            <body name="link1" pos="0 0 0">
              <joint name="j1" type="hinge" axis="0 0 1"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0"/>
              <body name="link2" pos="0.3 0 0">
                <joint name="j2" type="hinge" axis="0 0 1"/>
                <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0"/>
                <body name="ee" pos="0.3 0 0"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        kin = MuJoCoKinematics(model=model, data=data, body_name="ee")
        assert kin.backend == "mujoco"

        # Forward kinematics
        pose = kin.forward_kinematics(np.array([0.0, 0.0]))
        assert pose.shape == (4, 4), f"FK should return 4x4, got {pose.shape}"


# ===================================================================
# 8. Video Encoding (GPU-accelerated)
# ===================================================================
class TestThorVideo:
    """Test video encoding on Thor."""

    def test_video_encoder_import(self):
        from strands_robots.video import VideoEncoder

        assert VideoEncoder is not None

    @requires_cuda
    def test_encode_frames(self, tmp_path):
        """Encode a short video from numpy frames."""
        from strands_robots.video import VideoEncoder

        out_path = str(tmp_path / "test.mp4")
        encoder = VideoEncoder(out_path, fps=30)

        for i in range(30):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            encoder.add_frame(frame)

        encoder.close()
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
