#!/usr/bin/env python3
"""Tests for Sample 03: Build a World — 3D Environment Generation.

All tests use mocks so they run on CPU without MuJoCo or a GPU.

Coverage:
    - build_world.py  : world creation, robot/object/camera addition, rendering
    - domain_randomization.py : base scene + N randomized variants
    - marble_world.py : preset listing, pipeline config, generation, composition
"""

from __future__ import annotations

# Skip if samples/ directory not present (requires PR #13)
import os as _os

import pytest as _pytest_guard

if not _os.path.isdir(_os.path.join(_os.path.dirname(__file__), "..", "samples")):
    _pytest_guard.skip("Requires PR #13 (samples)", allow_module_level=True)


import importlib
import json
import os
import sys
from pathlib import Path
from unittest import mock

# ── Helpers ──────────────────────────────────────────────────────────

# Fake PNG bytes (1×1 white pixel PNG)
_FAKE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _ok(text: str = "ok") -> dict:
    """Shortcut for a successful result dict."""
    return {"status": "success", "content": [{"text": text}]}


def _ok_image(text: str = "rendered") -> dict:
    """Successful render result with fake PNG bytes."""
    return {
        "status": "success",
        "content": [
            {"text": text},
            {"image": {"format": "png", "source": {"bytes": _FAKE_PNG}}},
        ],
    }


def _err(text: str = "error") -> dict:
    return {"status": "error", "content": [{"text": text}]}


class MockSimulation:
    """Minimal mock of strands_robots.simulation.Simulation.

    Implements every method that the sample scripts call.
    Records all calls for assertion.
    """

    def __init__(self, **kwargs):
        self.calls: list[tuple[str, dict]] = []

    def _record(self, method_name: str, **kwargs):
        self.calls.append((method_name, kwargs))

    # World management
    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        self._record("create_world", timestep=timestep, gravity=gravity)
        return _ok("🌍 Simulation world created\n⚙️ Timestep: 0.002s (500Hz)")

    def destroy(self):
        self._record("destroy")
        return _ok("🗑️ World destroyed.")

    # Robot management
    def add_robot(self, name="robot_0", urdf_path=None, data_config=None, position=None, orientation=None):
        self._record("add_robot", name=name, data_config=data_config, position=position)
        return _ok(f"🤖 Robot '{name}' added to simulation\n📍 Position: {position}")

    # Object management
    def add_object(
        self,
        name="",
        shape="box",
        position=None,
        orientation=None,
        size=None,
        color=None,
        mass=0.1,
        is_static=False,
        mesh_path=None,
    ):
        self._record("add_object", name=name, shape=shape, position=position)
        return _ok(f"📦 '{name}' added: {shape}")

    # Camera management
    def add_camera(self, name="cam", position=None, target=None, fov=60.0, width=640, height=480):
        self._record("add_camera", name=name, position=position)
        return _ok(f"📷 Camera '{name}' added")

    # Simulation control
    def step(self, n_steps=1):
        self._record("step", n_steps=n_steps)
        return _ok(f"⏩ +{n_steps} steps")

    # Rendering
    def render(self, camera_name="default", width=None, height=None):
        self._record("render", camera_name=camera_name, width=width, height=height)
        return _ok_image(f"📸 {width}x{height} from '{camera_name}'")

    # Domain randomization
    def randomize(
        self,
        randomize_colors=True,
        randomize_lighting=True,
        randomize_physics=False,
        randomize_positions=False,
        position_noise=0.02,
        color_range=(0.1, 1.0),
        friction_range=(0.5, 1.5),
        mass_range=(0.8, 1.2),
        seed=None,
    ):
        self._record(
            "randomize",
            seed=seed,
            randomize_colors=randomize_colors,
            randomize_physics=randomize_physics,
            randomize_positions=randomize_positions,
        )
        lines = ["🎲 Domain Randomization applied:"]
        if randomize_colors:
            lines.append("🎨 Colors: randomized")
        if randomize_lighting:
            lines.append("💡 Lighting: randomized")
        if randomize_physics:
            lines.append("⚙️ Physics: randomized")
        if randomize_positions:
            lines.append("📍 Positions: randomized")
        return _ok("\n".join(lines))

    # State
    def get_state(self):
        self._record("get_state")
        return _ok("🌍 Simulation State\n" "🕐 t=0.4000s (step 200)\n" "🤖 Robots: 1 | 📦 Objects: 5 | 📷 Cameras: 3")


# =====================================================================
# Tests for build_world.py
# =====================================================================


class TestBuildWorld:
    """Tests for samples/03_build_a_world/build_world.py"""

    def _import_build_world(self):
        """Import the module (works even when strands_robots isn't installed)."""
        sample_dir = Path(__file__).resolve().parent.parent / "samples" / "03_build_a_world"
        if str(sample_dir) not in sys.path:
            sys.path.insert(0, str(sample_dir))
        import build_world

        importlib.reload(build_world)  # ensure fresh state
        return build_world

    def test_build_world_creates_scene(self):
        """build_world() calls create_world, add_robot, add_object, add_camera, step."""
        bw = self._import_build_world()
        sim = MockSimulation()
        result_sim = bw.build_world(sim=sim)

        assert result_sim is sim

        call_names = [c[0] for c in sim.calls]
        assert "create_world" in call_names
        assert "add_robot" in call_names
        assert call_names.count("add_object") == 5
        assert call_names.count("add_camera") == 3
        assert "step" in call_names

    def test_build_world_robot_config(self):
        """Robot is added with data_config='so100'."""
        bw = self._import_build_world()
        sim = MockSimulation()
        bw.build_world(sim=sim)

        robot_calls = [c for c in sim.calls if c[0] == "add_robot"]
        assert len(robot_calls) == 1
        assert robot_calls[0][1]["data_config"] == "so100"
        assert robot_calls[0][1]["name"] == "arm"

    def test_build_world_objects(self):
        """All 5 objects have distinct names and valid shapes."""
        bw = self._import_build_world()
        sim = MockSimulation()
        bw.build_world(sim=sim)

        obj_calls = [c for c in sim.calls if c[0] == "add_object"]
        names = [c[1]["name"] for c in obj_calls]
        assert len(set(names)) == 5, "Object names must be unique"

        valid_shapes = {"box", "sphere", "cylinder", "capsule", "mesh", "plane"}
        for c in obj_calls:
            assert c[1]["shape"] in valid_shapes

    def test_build_world_cameras(self):
        """Three cameras: front, side, top."""
        bw = self._import_build_world()
        sim = MockSimulation()
        bw.build_world(sim=sim)

        cam_calls = [c for c in sim.calls if c[0] == "add_camera"]
        cam_names = {c[1]["name"] for c in cam_calls}
        assert cam_names == {"front", "side", "top"}

    def test_render_cameras_saves_files(self, tmp_path):
        """render_cameras() saves PNG files to the output directory."""
        bw = self._import_build_world()
        sim = MockSimulation()

        saved = bw.render_cameras(sim, cameras=bw.CAMERAS, out_dir=str(tmp_path))

        assert len(saved) == 3
        for path in saved:
            assert os.path.exists(path)
            assert path.endswith(".png")

    def test_render_cameras_handles_failure(self, tmp_path):
        """If render returns an error, the file is not created."""
        bw = self._import_build_world()

        class FailSim(MockSimulation):
            def render(self, **kw):
                return _err("render failed")

        sim = FailSim()
        saved = bw.render_cameras(sim, cameras=bw.CAMERAS, out_dir=str(tmp_path))
        assert len(saved) == 0

    def test_scene_objects_constant(self):
        """SCENE_OBJECTS is a list of 5 dicts with required keys."""
        bw = self._import_build_world()
        assert len(bw.SCENE_OBJECTS) == 5
        for obj in bw.SCENE_OBJECTS:
            assert "name" in obj
            assert "shape" in obj
            assert "position" in obj
            assert "size" in obj
            assert "color" in obj

    def test_physics_step_count(self):
        """build_world steps physics to let objects settle."""
        bw = self._import_build_world()
        sim = MockSimulation()
        bw.build_world(sim=sim)

        step_calls = [c for c in sim.calls if c[0] == "step"]
        assert len(step_calls) >= 1
        total_steps = sum(c[1].get("n_steps", 1) for c in step_calls)
        assert total_steps >= 100, "Should step enough for physics to settle"


# =====================================================================
# Tests for domain_randomization.py
# =====================================================================


class TestDomainRandomization:
    """Tests for samples/03_build_a_world/domain_randomization.py"""

    def _import_module(self):
        sample_dir = Path(__file__).resolve().parent.parent / "samples" / "03_build_a_world"
        if str(sample_dir) not in sys.path:
            sys.path.insert(0, str(sample_dir))
        import domain_randomization

        importlib.reload(domain_randomization)
        return domain_randomization

    def test_create_base_scene(self):
        """create_base_scene() builds a world with robot + objects + camera."""
        dr = self._import_module()
        sim = MockSimulation()
        result_sim = dr.create_base_scene(sim=sim)

        assert result_sim is sim
        call_names = [c[0] for c in sim.calls]
        assert "create_world" in call_names
        assert "add_robot" in call_names
        assert call_names.count("add_object") == len(dr.BASE_OBJECTS)
        assert "add_camera" in call_names

    def test_randomize_and_render_calls_randomize(self, tmp_path):
        """randomize_and_render() calls sim.randomize() with correct seed."""
        dr = self._import_module()
        sim = MockSimulation()
        dr.create_base_scene(sim=sim)
        sim.calls.clear()  # reset to isolate randomize calls

        # Temporarily override OUT_DIR
        old_out = dr.OUT_DIR
        dr.OUT_DIR = str(tmp_path)
        try:
            info = dr.randomize_and_render(sim, variant_index=3, seed=45)
        finally:
            dr.OUT_DIR = old_out

        assert info["saved"] is True
        assert info["seed"] == 45
        assert info["variant"] == 3
        assert info["filename"] == "randomized_003.png"

        rand_calls = [c for c in sim.calls if c[0] == "randomize"]
        assert len(rand_calls) == 1
        assert rand_calls[0][1]["seed"] == 45

    def test_randomize_physics_every_third(self, tmp_path):
        """Physics randomization is applied on every 3rd variant (3,6,9…)."""
        dr = self._import_module()
        sim = MockSimulation()
        dr.create_base_scene(sim=sim)

        old_out = dr.OUT_DIR
        dr.OUT_DIR = str(tmp_path)
        try:
            for i in range(1, 11):
                sim.calls.clear()
                dr.randomize_and_render(sim, i, seed=42 + i)
                rand_call = [c for c in sim.calls if c[0] == "randomize"][0]
                expected_physics = i % 3 == 0
                assert (
                    rand_call[1]["randomize_physics"] == expected_physics
                ), f"Variant {i}: physics should be {expected_physics}"
        finally:
            dr.OUT_DIR = old_out

    def test_randomize_positions_every_second(self, tmp_path):
        """Position randomization is applied on every 2nd variant (2,4,6…)."""
        dr = self._import_module()
        sim = MockSimulation()
        dr.create_base_scene(sim=sim)

        old_out = dr.OUT_DIR
        dr.OUT_DIR = str(tmp_path)
        try:
            for i in range(1, 11):
                sim.calls.clear()
                dr.randomize_and_render(sim, i, seed=42 + i)
                rand_call = [c for c in sim.calls if c[0] == "randomize"][0]
                expected_pos = i % 2 == 0
                assert (
                    rand_call[1]["randomize_positions"] == expected_pos
                ), f"Variant {i}: positions should be {expected_pos}"
        finally:
            dr.OUT_DIR = old_out

    def test_num_variants_constant(self):
        """NUM_VARIANTS should be 10."""
        dr = self._import_module()
        assert dr.NUM_VARIANTS == 10

    def test_base_objects_have_required_keys(self):
        """Every base object has name, shape, position, color."""
        dr = self._import_module()
        for obj in dr.BASE_OBJECTS:
            for key in ("name", "shape", "position", "color"):
                assert key in obj, f"Missing '{key}' in object {obj.get('name', '?')}"


# =====================================================================
# Tests for marble_world.py
# =====================================================================


class TestMarbleWorld:
    """Tests for samples/03_build_a_world/marble_world.py"""

    def _import_module(self):
        sample_dir = Path(__file__).resolve().parent.parent / "samples" / "03_build_a_world"
        if str(sample_dir) not in sys.path:
            sys.path.insert(0, str(sample_dir))
        import marble_world

        importlib.reload(marble_world)
        return marble_world

    def test_list_presets_with_marble(self):
        """list_presets() returns the MARBLE_PRESETS dict when available."""
        mw = self._import_module()

        # Create a fake MARBLE_PRESETS dict
        fake_presets = {
            "kitchen": {
                "prompt": "A modern kitchen with wooden countertops",
                "description": "Kitchen manipulation",
                "category": "indoor",
            },
            "workshop": {
                "prompt": "A workshop table with tools",
                "description": "Workshop assembly",
                "category": "indoor",
            },
        }

        with mock.patch.dict(
            "sys.modules",
            {
                "strands_robots": mock.MagicMock(),
                "strands_robots.marble": mock.MagicMock(MARBLE_PRESETS=fake_presets),
            },
        ):
            # Re-import to pick up the patched module
            importlib.reload(mw)
            result = mw.list_presets()
            assert result is not None
            assert "kitchen" in result
            assert "workshop" in result

    def test_list_presets_without_marble(self):
        """list_presets() returns None when marble is not installed."""
        mw = self._import_module()

        # Force ImportError for strands_robots.marble
        with mock.patch.dict(
            "sys.modules",
            {
                "strands_robots.marble": None,
            },
        ):
            importlib.reload(mw)
            result = mw.list_presets()
            assert result is None

    def test_configure_pipeline(self):
        """configure_pipeline() creates a MarblePipeline with MarbleConfig."""
        mw = self._import_module()

        mock_config_cls = mock.MagicMock()
        mock_config_instance = mock.MagicMock()
        mock_config_instance.model = "Marble 0.1-mini"
        mock_config_instance.output_format = "ply"
        mock_config_instance.robot = "so101"
        mock_config_instance.seed = 42
        mock_config_cls.return_value = mock_config_instance

        mock_pipeline_cls = mock.MagicMock()
        mock_pipeline_instance = mock.MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline_instance

        mock_marble = mock.MagicMock(
            MarbleConfig=mock_config_cls,
            MarblePipeline=mock_pipeline_cls,
        )

        with mock.patch.dict(
            "sys.modules",
            {
                "strands_robots": mock.MagicMock(),
                "strands_robots.marble": mock_marble,
            },
        ):
            importlib.reload(mw)
            pipeline, config = mw.configure_pipeline()

        mock_config_cls.assert_called_once_with(
            model="Marble 0.1-mini",
            output_format="ply",
            robot="so101",
        )
        mock_pipeline_cls.assert_called_once_with(mock_config_instance)

    def test_main_no_api_key(self, monkeypatch, capsys):
        """main() in exploration mode (no API key) prints preset info."""
        mw = self._import_module()

        monkeypatch.delenv("WLT_API_KEY", raising=False)
        monkeypatch.delenv("MARBLE_API_KEY", raising=False)

        fake_presets = {
            "kitchen": {
                "prompt": "A modern kitchen",
                "description": "Kitchen",
                "category": "indoor",
            },
        }
        mock_config_cls = mock.MagicMock()
        mock_config_instance = mock.MagicMock()
        mock_config_instance.model = "Marble 0.1-mini"
        mock_config_instance.output_format = "ply"
        mock_config_instance.robot = "so101"
        mock_config_instance.seed = 42
        mock_config_cls.return_value = mock_config_instance

        mock_marble = mock.MagicMock(
            MARBLE_PRESETS=fake_presets,
            MarbleConfig=mock_config_cls,
            MarblePipeline=mock.MagicMock(),
        )

        with mock.patch.dict(
            "sys.modules",
            {
                "strands_robots": mock.MagicMock(),
                "strands_robots.marble": mock_marble,
            },
        ):
            importlib.reload(mw)
            mw.main()

        captured = capsys.readouterr()
        assert (
            "Skipped" in captured.out or "no API key" in captured.out.lower() or "exploration" in captured.out.lower()
        )

    def test_generate_scene_success(self, tmp_path):
        """generate_scene() saves metadata JSON on success."""
        mw = self._import_module()

        mock_scene = mock.MagicMock()
        mock_scene.world_id = "wld_abc123"
        mock_scene.caption = "A kitchen scene"
        mock_scene.glb_path = "/tmp/scene.glb"
        mock_scene.spz_path = None
        mock_scene.ply_path = "/tmp/scene.ply"
        mock_scene.scene_id = "scn_xyz"
        mock_scene.world_marble_url = "https://marble.worldlabs.ai/w/abc123"
        mock_scene.metadata = {"placeholder": False}

        mock_pipeline = mock.MagicMock()
        mock_pipeline.generate_world.return_value = [mock_scene]

        old_out = mw.OUT_DIR
        mw.OUT_DIR = str(tmp_path)
        try:
            result = mw.generate_scene(mock_pipeline, prompt="test kitchen")
        finally:
            mw.OUT_DIR = old_out

        assert result is mock_scene
        info_path = tmp_path / "scene_info.json"
        assert info_path.exists()

        with open(info_path) as fh:
            info = json.load(fh)
        assert info["world_id"] == "wld_abc123"
        assert info["placeholder"] is False

    def test_generate_scene_failure(self):
        """generate_scene() returns None when the API call fails."""
        mw = self._import_module()

        mock_pipeline = mock.MagicMock()
        mock_pipeline.generate_world.side_effect = RuntimeError("API error")

        result = mw.generate_scene(mock_pipeline, prompt="fail test")
        assert result is None

    def test_compose_with_robot_skips_placeholder(self):
        """compose_with_robot() skips when scene is a placeholder."""
        mw = self._import_module()

        mock_scene = mock.MagicMock()
        mock_scene.glb_path = "/tmp/scene.glb"
        mock_scene.metadata = {"placeholder": True}

        mock_pipeline = mock.MagicMock()
        result = mw.compose_with_robot(mock_pipeline, mock_scene)
        assert result is None
        mock_pipeline.compose_scene.assert_not_called()

    def test_default_prompt_not_empty(self):
        """DEFAULT_PROMPT is a non-empty string."""
        mw = self._import_module()
        assert isinstance(mw.DEFAULT_PROMPT, str)
        assert len(mw.DEFAULT_PROMPT) > 20


# =====================================================================
# Integration-style tests (still mocked, but exercise main())
# =====================================================================


class TestIntegration:
    """End-to-end tests that call main() with mocked Simulation."""

    def _import_build_world(self):
        sample_dir = Path(__file__).resolve().parent.parent / "samples" / "03_build_a_world"
        if str(sample_dir) not in sys.path:
            sys.path.insert(0, str(sample_dir))
        import build_world

        importlib.reload(build_world)
        return build_world

    def _import_domain_rand(self):
        sample_dir = Path(__file__).resolve().parent.parent / "samples" / "03_build_a_world"
        if str(sample_dir) not in sys.path:
            sys.path.insert(0, str(sample_dir))
        import domain_randomization

        importlib.reload(domain_randomization)
        return domain_randomization

    def test_build_world_main(self, tmp_path):
        """build_world.build_world() + render_cameras() completes under mock."""
        bw = self._import_build_world()

        sim = MockSimulation()
        bw.build_world(sim=sim)
        saved = bw.render_cameras(sim, out_dir=str(tmp_path))
        assert len(saved) == 3
        for p in saved:
            assert os.path.exists(p)

    def test_domain_randomization_main(self, tmp_path):
        """domain_randomization produces baseline + N variant files."""
        dr = self._import_domain_rand()

        sim = MockSimulation()
        dr.create_base_scene(sim=sim)

        old_out = dr.OUT_DIR
        dr.OUT_DIR = str(tmp_path)
        try:
            # Render baseline
            baseline_result = sim.render(
                camera_name=dr.CAMERA_NAME,
                width=dr.RENDER_WIDTH,
                height=dr.RENDER_HEIGHT,
            )
            assert baseline_result["status"] == "success"

            # Render variants
            for i in range(1, dr.NUM_VARIANTS + 1):
                info = dr.randomize_and_render(sim, i, seed=42 + i)
                assert info["saved"] is True
        finally:
            dr.OUT_DIR = old_out

        # Verify all variant files exist
        for i in range(1, dr.NUM_VARIANTS + 1):
            assert (tmp_path / f"randomized_{i:03d}.png").exists()


# =====================================================================
# API fidelity tests — verify we use the correct Simulation signatures
# =====================================================================


class TestAPIFidelity:
    """Verify that sample code matches the actual Simulation API."""

    def test_create_world_signature(self):
        """create_world() accepts timestep, gravity, ground_plane."""
        sim = MockSimulation()
        result = sim.create_world(timestep=0.002, gravity=[0, 0, -9.81])
        assert result["status"] == "success"

    def test_add_robot_signature(self):
        """add_robot() accepts name, data_config, position."""
        sim = MockSimulation()
        result = sim.add_robot(name="arm", data_config="so100", position=[0, 0, 0])
        assert result["status"] == "success"

    def test_add_object_signature(self):
        """add_object() accepts name, shape, position, size, color, mass."""
        sim = MockSimulation()
        result = sim.add_object(
            name="test_obj",
            shape="box",
            position=[0.3, 0, 0.5],
            size=[0.05, 0.05, 0.05],
            color=[1, 0, 0, 1],
            mass=0.1,
        )
        assert result["status"] == "success"

    def test_add_camera_signature(self):
        """add_camera() accepts name, position, target, fov, width, height."""
        sim = MockSimulation()
        result = sim.add_camera(
            name="test_cam",
            position=[1, 1, 1],
            target=[0, 0, 0],
            fov=60.0,
            width=640,
            height=480,
        )
        assert result["status"] == "success"

    def test_randomize_signature(self):
        """randomize() accepts all documented parameters."""
        sim = MockSimulation()
        result = sim.randomize(
            randomize_colors=True,
            randomize_lighting=True,
            randomize_physics=True,
            randomize_positions=True,
            position_noise=0.02,
            color_range=(0.1, 1.0),
            friction_range=(0.5, 1.5),
            mass_range=(0.8, 1.2),
            seed=42,
        )
        assert result["status"] == "success"

    def test_render_signature(self):
        """render() accepts camera_name, width, height."""
        sim = MockSimulation()
        result = sim.render(camera_name="default", width=640, height=480)
        assert result["status"] == "success"
        # Should contain image content
        has_image = any("image" in item for item in result.get("content", []))
        assert has_image

    def test_step_signature(self):
        """step() accepts n_steps."""
        sim = MockSimulation()
        result = sim.step(n_steps=100)
        assert result["status"] == "success"

    def test_get_state_signature(self):
        """get_state() returns status dict."""
        sim = MockSimulation()
        result = sim.get_state()
        assert result["status"] == "success"

    def test_destroy_signature(self):
        """destroy() returns status dict."""
        sim = MockSimulation()
        result = sim.destroy()
        assert result["status"] == "success"
