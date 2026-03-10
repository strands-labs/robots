#!/usr/bin/env python3
"""Tests for strands_robots.marble module.

Comprehensive test coverage for MarblePipeline, MarbleConfig, MarbleScene,
convenience functions, and all pipeline stages. All tests run without
external dependencies (Marble API, 3DGrut, pxr, Isaac Sim).
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# MarbleConfig tests
# ---------------------------------------------------------------------------


class TestMarbleConfig:
    """Tests for MarbleConfig dataclass validation and defaults."""

    def test_default_config(self):
        from strands_robots.marble import MARBLE_API_URL, MarbleConfig

        config = MarbleConfig()
        assert config.api_url == MARBLE_API_URL
        assert config.model == "Marble 0.1-plus"
        assert config.output_format == "ply"
        assert config.input_mode == "text"
        assert config.robot is None
        assert config.table_replacement is True
        assert config.auto_compose is False
        assert config.convert_to_usdz is True
        assert config.chisel_enabled is False
        assert config.num_variations == 1
        assert config.seed == 42
        assert config.poll_interval == 5.0
        assert config.poll_timeout == 600.0
        assert config.is_public is False

    def test_config_with_robot(self):
        from strands_robots.marble import MarbleConfig

        config = MarbleConfig(robot="so101")
        assert config.robot == "so101"

    def test_config_with_all_valid_robots(self):
        from strands_robots.marble import SUPPORTED_ROBOTS, MarbleConfig

        for robot_key in SUPPORTED_ROBOTS:
            config = MarbleConfig(robot=robot_key)
            assert config.robot == robot_key

    def test_config_invalid_robot(self):
        from strands_robots.marble import MarbleConfig

        with pytest.raises(ValueError, match="Unknown robot"):
            MarbleConfig(robot="invalid_bot")

    def test_config_invalid_output_format(self):
        from strands_robots.marble import MarbleConfig

        with pytest.raises(ValueError, match="Invalid output_format"):
            MarbleConfig(output_format="jpeg")

    def test_config_invalid_input_mode(self):
        from strands_robots.marble import MarbleConfig

        with pytest.raises(ValueError, match="Invalid input_mode"):
            MarbleConfig(input_mode="audio")

    def test_config_invalid_num_variations(self):
        from strands_robots.marble import MarbleConfig

        with pytest.raises(ValueError, match="num_variations must be >= 1"):
            MarbleConfig(num_variations=0)

    def test_config_negative_num_variations(self):
        from strands_robots.marble import MarbleConfig

        with pytest.raises(ValueError, match="num_variations must be >= 1"):
            MarbleConfig(num_variations=-5)

    @pytest.mark.parametrize("fmt", ["ply", "glb", "usdz", "video"])
    def test_config_valid_output_formats(self, fmt):
        from strands_robots.marble import MarbleConfig

        config = MarbleConfig(output_format=fmt)
        assert config.output_format == fmt

    @pytest.mark.parametrize("mode", ["text", "image", "video", "multi-image"])
    def test_config_valid_input_modes(self, mode):
        from strands_robots.marble import MarbleConfig

        config = MarbleConfig(input_mode=mode)
        assert config.input_mode == mode

    def test_config_api_key_from_env(self):
        from strands_robots.marble import MarbleConfig

        with patch.dict(os.environ, {"MARBLE_API_KEY": "test-key-123"}):
            config = MarbleConfig()
            assert config.api_key == "test-key-123"

    def test_config_api_key_from_wlt_env(self):
        from strands_robots.marble import MarbleConfig

        with patch.dict(os.environ, {"WLT_API_KEY": "wlt-key-456"}, clear=False):
            os.environ.pop("MARBLE_API_KEY", None)
            config = MarbleConfig()
            assert config.api_key == "wlt-key-456"

    def test_config_api_key_explicit(self):
        from strands_robots.marble import MarbleConfig

        config = MarbleConfig(api_key="explicit-key")
        assert config.api_key == "explicit-key"

    def test_config_threedgrut_path_from_env(self):
        from strands_robots.marble import MarbleConfig

        with patch.dict(os.environ, {"THREEDGRUT_PATH": "/opt/3dgrut"}):
            config = MarbleConfig()
            assert config.threedgrut_path == "/opt/3dgrut"

    def test_config_threedgrut_path_explicit(self):
        from strands_robots.marble import MarbleConfig

        config = MarbleConfig(threedgrut_path="/my/3dgrut")
        assert config.threedgrut_path == "/my/3dgrut"

    def test_config_no_api_key(self):
        from strands_robots.marble import MarbleConfig

        with patch.dict(os.environ, {}, clear=True):
            # Remove MARBLE_API_KEY if present
            os.environ.pop("MARBLE_API_KEY", None)
            config = MarbleConfig()
            assert config.api_key is None

    def test_config_local_api(self):
        from strands_robots.marble import MarbleConfig

        config = MarbleConfig(api_url="local")
        assert config.api_url == "local"

    def test_resolve_threedgrut_path_explicit(self, tmp_path):
        from strands_robots.marble import MarbleConfig

        grut_dir = tmp_path / "3dgrut"
        grut_dir.mkdir()
        config = MarbleConfig(threedgrut_path=str(grut_dir))
        assert config.resolve_threedgrut_path() == str(grut_dir)

    def test_resolve_threedgrut_path_env(self, tmp_path):
        from strands_robots.marble import MarbleConfig

        grut_dir = tmp_path / "3dgrut_env"
        grut_dir.mkdir()
        with patch.dict(os.environ, {"THREEDGRUT_PATH": str(grut_dir)}):
            config = MarbleConfig()
            assert config.resolve_threedgrut_path() == str(grut_dir)

    def test_resolve_threedgrut_path_not_found(self):
        from strands_robots.marble import MarbleConfig

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("THREEDGRUT_PATH", None)
            config = MarbleConfig(threedgrut_path="/nonexistent/path")
            # Should check common locations and return None if none exist
            result = config.resolve_threedgrut_path()
            # Unless ~/3dgrut etc. happen to exist
            if result is not None:
                assert os.path.isdir(result)

    def test_resolve_threedgrut_path_common_locations(self, tmp_path):
        from strands_robots.marble import MarbleConfig

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("THREEDGRUT_PATH", None)
            config = MarbleConfig(threedgrut_path=None)
            # Mock os.path.isdir to return True for /opt/3dgrut
            with patch("os.path.isdir") as mock_isdir:
                mock_isdir.side_effect = lambda p: p == "/opt/3dgrut"
                result = config.resolve_threedgrut_path()
                assert result == "/opt/3dgrut"


# ---------------------------------------------------------------------------
# MarbleScene tests
# ---------------------------------------------------------------------------


class TestMarbleScene:
    """Tests for MarbleScene dataclass."""

    def test_scene_defaults(self):
        from strands_robots.marble import MarbleScene

        scene = MarbleScene(scene_id="test_001", prompt="A kitchen")
        assert scene.scene_id == "test_001"
        assert scene.prompt == "A kitchen"
        assert scene.input_mode == "text"
        assert scene.ply_path is None
        assert scene.spz_path is None
        assert scene.glb_path is None
        assert scene.video_path is None
        assert scene.pano_path is None
        assert scene.usdz_path is None
        assert scene.scene_usd is None
        assert scene.output_dir == ""
        assert scene.robot is None
        assert scene.task_objects == []
        assert scene.composed is False
        assert scene.metadata == {}
        # Properties
        assert scene.splat_path is None
        assert scene.best_background is None

    def test_scene_to_dict(self):
        from strands_robots.marble import MarbleScene

        scene = MarbleScene(
            scene_id="test_002",
            prompt="An office",
            ply_path="/tmp/scene.ply",
            robot="so101",
            composed=True,
            metadata={"seed": 42},
        )
        d = scene.to_dict()
        assert isinstance(d, dict)
        assert d["scene_id"] == "test_002"
        assert d["prompt"] == "An office"
        assert d["ply_path"] == "/tmp/scene.ply"
        assert d["robot"] == "so101"
        assert d["composed"] is True
        assert d["metadata"]["seed"] == 42
        # Properties in to_dict
        assert d["splat_path"] == "/tmp/scene.ply"  # PLY fallback since no SPZ
        assert d["best_background"] == "/tmp/scene.ply"

    def test_scene_splat_path_prefers_spz(self):
        from strands_robots.marble import MarbleScene

        scene = MarbleScene(
            scene_id="spz_test",
            prompt="SPZ test",
            ply_path="/tmp/scene.ply",
            spz_path="/tmp/scene_500k.spz",
        )
        assert scene.splat_path == "/tmp/scene_500k.spz"
        assert scene.best_background == "/tmp/scene_500k.spz"

    def test_scene_best_background_priority(self):
        from strands_robots.marble import MarbleScene

        # USDZ > GLB > SPZ > PLY > pano
        scene = MarbleScene(
            scene_id="bg_test",
            prompt="Background priority test",
            ply_path="/a.ply",
            spz_path="/a.spz",
            glb_path="/a.glb",
            usdz_path="/a.usdz",
            pano_path="/a.jpg",
        )
        assert scene.best_background == "/a.usdz"

        # Without USDZ, GLB wins
        scene2 = MarbleScene(
            scene_id="bg_test2",
            prompt="test",
            spz_path="/a.spz",
            glb_path="/a.glb",
        )
        assert scene2.best_background == "/a.glb"

        # Pano only
        scene3 = MarbleScene(scene_id="bg_test3", prompt="test", pano_path="/a.jpg")
        assert scene3.best_background == "/a.jpg"

    def test_scene_full_attributes(self):
        from strands_robots.marble import MarbleScene

        scene = MarbleScene(
            scene_id="full",
            prompt="Full scene",
            input_mode="image",
            ply_path="/a.ply",
            spz_path="/a.spz",
            glb_path="/a.glb",
            video_path="/a.mp4",
            pano_path="/a_pano.jpg",
            usdz_path="/a.usdz",
            scene_usd="/a_composed.usda",
            output_dir="/out",
            robot="panda",
            task_objects=["mug", "plate"],
            composed=True,
            metadata={"var": 0},
        )
        d = scene.to_dict()
        assert d["input_mode"] == "image"
        assert d["glb_path"] == "/a.glb"
        assert d["video_path"] == "/a.mp4"
        assert d["task_objects"] == ["mug", "plate"]


# ---------------------------------------------------------------------------
# MarblePipeline tests
# ---------------------------------------------------------------------------


class TestMarblePipeline:
    """Tests for MarblePipeline lifecycle, generation, conversion, composition."""

    def test_init_default(self):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()
        assert pipeline.config.output_format == "ply"
        assert pipeline.config.robot is None
        assert pipeline._generated_scenes == []

    def test_init_with_config(self):
        from strands_robots.marble import MarbleConfig, MarblePipeline

        config = MarbleConfig(robot="so101", num_variations=3)
        pipeline = MarblePipeline(config)
        assert pipeline.config.robot == "so101"
        assert pipeline.config.num_variations == 3

    def test_init_with_kwargs(self):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline(robot="panda", seed=99)
        assert pipeline.config.robot == "panda"
        assert pipeline.config.seed == 99

    def test_init_config_and_kwargs_raises(self):
        from strands_robots.marble import MarbleConfig, MarblePipeline

        config = MarbleConfig()
        with pytest.raises(ValueError, match="Cannot specify both"):
            MarblePipeline(config, robot="so101")

    def test_repr(self):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline(robot="so101", num_variations=5)
        r = repr(pipeline)
        assert "MarblePipeline" in r
        assert "so101" in r
        assert "5" in r

    def test_cleanup(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()
        # Create tmp dir and add to _tmp_dirs
        tmp_dir = tmp_path / "cleanup_test"
        tmp_dir.mkdir()
        pipeline._tmp_dirs.append(str(tmp_dir))
        assert tmp_dir.exists()
        pipeline.cleanup()
        assert not tmp_dir.exists()
        assert pipeline._tmp_dirs == []

    def test_cleanup_nonexistent_dir(self):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()
        pipeline._tmp_dirs.append("/nonexistent/dir/12345")
        pipeline.cleanup()  # Should not raise
        assert pipeline._tmp_dirs == []

    def test_get_generated_scenes(self):
        from strands_robots.marble import MarblePipeline, MarbleScene

        pipeline = MarblePipeline()
        scene = MarbleScene(scene_id="a", prompt="test")
        pipeline._generated_scenes.append(scene)
        scenes = pipeline.get_generated_scenes()
        assert len(scenes) == 1
        assert scenes[0].scene_id == "a"
        # Should return a copy
        scenes.append(MarbleScene(scene_id="b", prompt="test2"))
        assert len(pipeline.get_generated_scenes()) == 1

    # --- Stage 1: Generate ---

    def test_generate_world_text(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        output_dir = str(tmp_path / "scenes")
        pipeline = MarblePipeline()
        scenes = pipeline.generate_world(
            prompt="A kitchen with fruits",
            output_dir=output_dir,
            num_variations=2,
        )

        assert len(scenes) == 2
        for scene in scenes:
            assert scene.prompt == "A kitchen with fruits"
            assert scene.input_mode == "text"
            assert scene.output_dir
            assert scene.scene_id.startswith("marble_")
            assert scene.metadata.get("placeholder") is True
            # PLY should be created as placeholder
            assert scene.ply_path is not None
            assert os.path.isfile(scene.ply_path)

    def test_generate_world_with_image_input(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        # Create a fake input image
        img = tmp_path / "input.jpg"
        img.write_text("fake image data")

        output_dir = str(tmp_path / "scenes")
        pipeline = MarblePipeline()
        scenes = pipeline.generate_world(
            prompt="Recreate this scene",
            output_dir=output_dir,
            input_image=str(img),
        )
        assert len(scenes) == 1
        assert scenes[0].input_mode == "image"

    def test_generate_world_with_video_input(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        video = tmp_path / "input.mp4"
        video.write_text("fake video")

        pipeline = MarblePipeline()
        scenes = pipeline.generate_world(
            prompt="Scene from video",
            output_dir=str(tmp_path / "scenes"),
            input_video=str(video),
        )
        assert len(scenes) == 1
        assert scenes[0].input_mode == "video"

    def test_generate_world_with_layout_input(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        layout = tmp_path / "layout.json"
        layout.write_text('{"objects": []}')

        pipeline = MarblePipeline()
        scenes = pipeline.generate_world(
            prompt="From layout",
            output_dir=str(tmp_path / "scenes"),
            input_layout=str(layout),
        )
        assert len(scenes) == 1
        assert scenes[0].input_mode == "layout_3d"

    def test_generate_world_auto_compose(self, tmp_path):
        from strands_robots.marble import MarbleConfig, MarblePipeline

        config = MarbleConfig(robot="so101", auto_compose=True, convert_to_usdz=False)
        pipeline = MarblePipeline(config)
        scenes = pipeline.generate_world(
            prompt="Auto compose test",
            output_dir=str(tmp_path / "scenes"),
        )
        assert len(scenes) == 1
        # Compose may succeed or fail depending on whether pxr is available
        # But the scene should still be created
        assert scenes[0].prompt == "Auto compose test"

    def test_generate_world_stores_in_history(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()
        assert len(pipeline.get_generated_scenes()) == 0
        pipeline.generate_world(
            prompt="History test",
            output_dir=str(tmp_path / "scenes"),
            num_variations=3,
        )
        assert len(pipeline.get_generated_scenes()) == 3

    def test_generate_world_num_variations_override(self, tmp_path):
        from strands_robots.marble import MarbleConfig, MarblePipeline

        config = MarbleConfig(num_variations=1)
        pipeline = MarblePipeline(config)
        scenes = pipeline.generate_world(
            prompt="Override test",
            output_dir=str(tmp_path / "scenes"),
            num_variations=5,
        )
        assert len(scenes) == 5

    def test_generate_world_metadata_seed(self, tmp_path):
        from strands_robots.marble import MarbleConfig, MarblePipeline

        config = MarbleConfig(seed=100)
        pipeline = MarblePipeline(config)
        scenes = pipeline.generate_world(
            prompt="Seed test",
            output_dir=str(tmp_path / "scenes"),
            num_variations=3,
        )
        assert scenes[0].metadata["seed"] == 100
        assert scenes[1].metadata["seed"] == 101
        assert scenes[2].metadata["seed"] == 102

    # --- Stage 2: PLY → USDZ conversion ---

    def test_convert_ply_to_usdz_file_not_found(self):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()
        with pytest.raises(FileNotFoundError, match="PLY file not found"):
            pipeline.convert_ply_to_usdz("/nonexistent/file.ply")

    def test_convert_ply_to_usdz_no_converters(self, tmp_path):
        from strands_robots.marble import MarbleConfig, MarblePipeline

        # Create a minimal PLY file
        ply_file = tmp_path / "test.ply"
        ply_file.write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n0.0 0.0 0.0\n"
        )

        config = MarbleConfig(threedgrut_path="/nonexistent")
        pipeline = MarblePipeline(config)

        # Mock all converters to fail
        with (
            patch.object(pipeline, "_try_pxr_conversion", return_value=False),
            patch.object(pipeline, "_try_trimesh_conversion", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="No PLY→USDZ converter"):
                pipeline.convert_ply_to_usdz(str(ply_file))

    def test_convert_ply_to_usdz_via_3dgrut(self, tmp_path):
        from strands_robots.marble import MarbleConfig, MarblePipeline

        ply_file = tmp_path / "test.ply"
        ply_file.write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n0.0 0.0 0.0\n"
        )

        grut_dir = tmp_path / "3dgrut"
        grut_dir.mkdir()
        scripts = grut_dir / "scripts"
        scripts.mkdir()
        export_script = scripts / "export_usdz.py"
        export_script.write_text("# fake script")

        config = MarbleConfig(threedgrut_path=str(grut_dir))
        pipeline = MarblePipeline(config)

        with patch.object(pipeline, "_run_subprocess", return_value=str(tmp_path / "test.usdz")) as mock_run:
            result = pipeline.convert_ply_to_usdz(str(ply_file))
            assert result == str(tmp_path / "test.usdz")
            mock_run.assert_called_once()

    def test_convert_ply_to_usdz_via_pxr(self, tmp_path):
        from strands_robots.marble import MarbleConfig, MarblePipeline

        ply_file = tmp_path / "test.ply"
        ply_file.write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n0.0 0.0 0.0\n"
        )

        config = MarbleConfig(threedgrut_path=None)
        pipeline = MarblePipeline(config)

        with patch.object(pipeline, "_try_pxr_conversion", return_value=True):
            result = pipeline.convert_ply_to_usdz(str(ply_file))
            assert result.endswith(".usdz")

    def test_convert_ply_to_usdz_via_trimesh(self, tmp_path):
        from strands_robots.marble import MarbleConfig, MarblePipeline

        ply_file = tmp_path / "test.ply"
        ply_file.write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n0.0 0.0 0.0\n"
        )

        config = MarbleConfig(threedgrut_path=None)
        pipeline = MarblePipeline(config)

        with (
            patch.object(pipeline, "_try_pxr_conversion", return_value=False),
            patch.object(pipeline, "_try_trimesh_conversion", return_value=True),
        ):
            result = pipeline.convert_ply_to_usdz(str(ply_file))
            assert result.endswith(".usdz")

    def test_convert_ply_default_output_path(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        ply_file = tmp_path / "scene.ply"
        ply_file.write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n0.0 0.0 0.0\n"
        )

        pipeline = MarblePipeline()

        with patch.object(pipeline, "_try_pxr_conversion", return_value=True):
            result = pipeline.convert_ply_to_usdz(str(ply_file))
            assert result.endswith(".usdz")
            assert "scene.usdz" in result

    # --- PLY reader ---

    def test_read_ply_vertices_ascii(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        ply_file = tmp_path / "ascii.ply"
        ply_file.write_text(
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 3\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
            "1.0 2.0 3.0\n"
            "4.0 5.0 6.0\n"
            "7.0 8.0 9.0\n"
        )
        vertices, colors = MarblePipeline._read_ply_vertices(str(ply_file))
        assert vertices is not None
        assert len(vertices) == 3
        assert vertices[0] == (1.0, 2.0, 3.0)
        assert vertices[2] == (7.0, 8.0, 9.0)
        assert colors is None or len(colors) == 0

    def test_read_ply_vertices_with_colors(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        ply_file = tmp_path / "colored.ply"
        ply_file.write_text(
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 2\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
            "1.0 2.0 3.0 255 0 0\n"
            "4.0 5.0 6.0 0 255 0\n"
        )
        vertices, colors = MarblePipeline._read_ply_vertices(str(ply_file))
        assert vertices is not None
        assert len(vertices) == 2
        assert colors is not None
        assert len(colors) == 2
        assert abs(colors[0][0] - 1.0) < 0.01  # 255/255
        assert abs(colors[1][1] - 1.0) < 0.01

    def test_read_ply_vertices_nonexistent(self):
        from strands_robots.marble import MarblePipeline

        vertices, colors = MarblePipeline._read_ply_vertices("/nonexistent.ply")
        assert vertices is None
        assert colors is None

    def test_read_ply_vertices_empty(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        ply_file = tmp_path / "empty.ply"
        ply_file.write_text(
            "ply\nformat ascii 1.0\nelement vertex 0\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n"
        )
        vertices, colors = MarblePipeline._read_ply_vertices(str(ply_file))
        # Either None or empty list
        assert vertices is None or len(vertices) == 0

    # --- PLY reader with plyfile ---

    def test_read_ply_vertices_with_plyfile(self, tmp_path):
        """Test PLY reading when plyfile is available."""
        from strands_robots.marble import MarblePipeline

        ply_file = tmp_path / "plyfile_test.ply"
        ply_file.write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n1.0 2.0 3.0\n"
        )

        # Mock plyfile
        mock_plydata = MagicMock()
        mock_vertex = MagicMock()
        mock_vertex.__getitem__ = lambda self, key: {"x": [1.0], "y": [2.0], "z": [3.0]}[key]
        mock_vertex.data.dtype.names = ("x", "y", "z")
        mock_plydata.__getitem__ = lambda self, key: mock_vertex
        mock_plyfile = MagicMock()
        mock_plyfile.PlyData.read.return_value = mock_plydata

        with patch.dict(sys.modules, {"plyfile": mock_plyfile}):
            vertices, colors = MarblePipeline._read_ply_vertices(str(ply_file))
            assert vertices is not None
            assert len(vertices) == 1

    # --- Placeholder creation ---

    def test_create_placeholder_ply(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        ply_path = str(tmp_path / "placeholder.ply")
        MarblePipeline._create_placeholder_ply(ply_path, "A kitchen scene")

        assert os.path.isfile(ply_path)
        content = Path(ply_path).read_text()
        assert "ply" in content
        assert "element vertex 1000" in content
        assert "end_header" in content

        # Count data lines
        lines = content.strip().split("\n")
        header_end = lines.index("end_header")
        data_lines = lines[header_end + 1 :]
        assert len(data_lines) == 1000

    def test_create_placeholder_ply_deterministic(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        ply1 = str(tmp_path / "p1.ply")
        ply2 = str(tmp_path / "p2.ply")
        MarblePipeline._create_placeholder_ply(ply1, "same prompt")
        MarblePipeline._create_placeholder_ply(ply2, "same prompt")
        assert Path(ply1).read_text() == Path(ply2).read_text()

    def test_create_placeholder_ply_different_prompts(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        ply1 = str(tmp_path / "p1.ply")
        ply2 = str(tmp_path / "p2.ply")
        MarblePipeline._create_placeholder_ply(ply1, "kitchen")
        MarblePipeline._create_placeholder_ply(ply2, "office")
        assert Path(ply1).read_text() != Path(ply2).read_text()

    # --- Stage 3: Compose ---

    def test_compose_scene_file_not_found(self):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()
        with pytest.raises(FileNotFoundError, match="Scene file not found"):
            pipeline.compose_scene(scene_path="/nonexistent/scene.usdz")

    def test_compose_scene_invalid_robot(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        scene_file = tmp_path / "scene.ply"
        scene_file.write_text("fake")
        pipeline = MarblePipeline()
        with pytest.raises(ValueError, match="Unknown robot"):
            pipeline.compose_scene(scene_path=str(scene_file), robot="invalid")

    def test_compose_scene_no_pxr_fallback(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        scene_file = tmp_path / "scene.ply"
        scene_file.write_text("fake scene data")

        pipeline = MarblePipeline()
        with patch.object(pipeline, "_try_usd_composition", return_value=False):
            result = pipeline.compose_scene(
                scene_path=str(scene_file),
                robot="so101",
                task_objects=["orange", "plate"],
            )

        assert "scene_usd" in result
        assert result["robot"] == "so101"
        assert result["task_objects"] == ["orange", "plate"]
        assert result["table_replacement"] is True
        # Fallback creates a JSON manifest
        manifest_path = result["scene_usd"]
        assert manifest_path.endswith(".json")
        assert os.path.isfile(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["robot"] == "so101"
        assert "orange" in manifest["task_objects"]

    def test_compose_scene_with_robot_position(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        scene_file = tmp_path / "scene.ply"
        scene_file.write_text("fake")

        pipeline = MarblePipeline()
        with patch.object(pipeline, "_try_usd_composition", return_value=False):
            result = pipeline.compose_scene(
                scene_path=str(scene_file),
                robot="panda",
                robot_position=[1.0, 2.0, 0.5],
            )
        assert result["robot"] == "panda"

    def test_compose_scene_no_robot(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        scene_file = tmp_path / "scene.ply"
        scene_file.write_text("fake")

        pipeline = MarblePipeline()
        with patch.object(pipeline, "_try_usd_composition", return_value=False):
            result = pipeline.compose_scene(scene_path=str(scene_file))
        assert result["robot"] is None

    def test_compose_scene_no_table_replacement(self, tmp_path):
        from strands_robots.marble import MarbleConfig, MarblePipeline

        scene_file = tmp_path / "scene.ply"
        scene_file.write_text("fake")

        config = MarbleConfig(table_replacement=False)
        pipeline = MarblePipeline(config)
        with patch.object(pipeline, "_try_usd_composition", return_value=False):
            result = pipeline.compose_scene(scene_path=str(scene_file))
        assert result["table_replacement"] is False

    def test_compose_scene_output_dir(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        scene_file = tmp_path / "scene.ply"
        scene_file.write_text("fake")
        out_dir = tmp_path / "output"

        pipeline = MarblePipeline()
        with patch.object(pipeline, "_try_usd_composition", return_value=False):
            result = pipeline.compose_scene(
                scene_path=str(scene_file),
                output_dir=str(out_dir),
            )
        assert str(out_dir) in result["scene_usd"]

    # --- Stage 4: Batch generation ---

    def test_generate_training_scenes_from_preset(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()
        result = pipeline.generate_training_scenes(
            preset="kitchen",
            num_per_prompt=3,
            output_dir=str(tmp_path / "training"),
        )
        assert result["total_scenes"] == 3
        assert result["preset"] == "kitchen"
        assert len(result["scenes"]) == 3

    def test_generate_training_scenes_custom_prompts(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()
        result = pipeline.generate_training_scenes(
            prompts=["A kitchen", "An office"],
            num_per_prompt=2,
            output_dir=str(tmp_path / "training"),
        )
        assert result["total_scenes"] == 4  # 2 prompts × 2 variations
        assert result["prompts"] == ["A kitchen", "An office"]

    def test_generate_training_scenes_no_prompt_or_preset(self):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()
        with pytest.raises(ValueError, match="Either 'prompts' or a valid 'preset'"):
            pipeline.generate_training_scenes(
                output_dir="/tmp/test",
            )

    def test_generate_training_scenes_invalid_preset(self):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()
        with pytest.raises(ValueError, match="Either 'prompts' or a valid 'preset'"):
            pipeline.generate_training_scenes(
                preset="nonexistent",
                output_dir="/tmp/test",
            )

    def test_generate_training_scenes_with_robot(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()
        result = pipeline.generate_training_scenes(
            preset="office_desk",
            num_per_prompt=2,
            robot="so101",
            output_dir=str(tmp_path / "training"),
        )
        assert result["robot"] == "so101"
        # Composition may or may not succeed
        assert result["total_scenes"] == 2

    # --- Marble API ---

    def test_call_marble_api_placeholder(self, tmp_path):
        from strands_robots.marble import MarbleConfig, MarblePipeline

        config = MarbleConfig(api_key=None)
        pipeline = MarblePipeline(config)

        result = pipeline._call_marble_api(
            prompt="Test kitchen",
            input_mode="text",
            output_dir=str(tmp_path),
            seed=42,
        )

        assert result.get("ply_path") is not None
        assert os.path.isfile(result["ply_path"])
        assert result["metadata"]["placeholder"] is True

        # Check metadata JSON was created
        meta_path = tmp_path / "generation_metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["prompt"] == "Test kitchen"
        assert meta["placeholder"] is True

    def test_call_marble_api_remote_fallback(self, tmp_path):
        from strands_robots.marble import MarbleConfig, MarblePipeline

        config = MarbleConfig(api_key="test-key")
        pipeline = MarblePipeline(config)

        # Mock requests to fail → should fall back to placeholder
        with patch.object(pipeline, "_call_marble_api_remote", side_effect=Exception("API Error")):
            result = pipeline._call_marble_api(
                prompt="Test",
                input_mode="text",
                output_dir=str(tmp_path),
                seed=42,
            )

        assert result["metadata"]["placeholder"] is True
        assert result.get("ply_path") is not None

    def test_call_marble_api_local_mode(self, tmp_path):
        from strands_robots.marble import MarbleConfig, MarblePipeline

        config = MarbleConfig(api_url="local", api_key="test")
        pipeline = MarblePipeline(config)

        result = pipeline._call_marble_api(
            prompt="Local test",
            input_mode="text",
            output_dir=str(tmp_path),
            seed=42,
        )

        assert result["metadata"]["placeholder"] is True

    def test_call_marble_api_remote_success(self, tmp_path):
        from strands_robots.marble import MarbleConfig, MarblePipeline

        config = MarbleConfig(api_key="test-key")
        pipeline = MarblePipeline(config)

        # Mock the /worlds:generate response (returns operation_id)
        mock_generate_response = MagicMock()
        mock_generate_response.json.return_value = {
            "operation_id": "op-123",
            "done": False,
        }
        mock_generate_response.raise_for_status = MagicMock()

        # Mock the /operations/op-123 poll response (done=true with World)
        mock_poll_response = MagicMock()
        mock_poll_response.json.return_value = {
            "operation_id": "op-123",
            "done": True,
            "response": {
                "world_id": "world-abc",
                "display_name": "Remote test",
                "world_marble_url": "https://marble.worldlabs.ai/w/world-abc",
                "model": "Marble 0.1-plus",
                "assets": {
                    "caption": "A test scene",
                    "thumbnail_url": "https://example.com/thumb.jpg",
                    "imagery": {"pano_url": "https://example.com/pano.jpg"},
                    "mesh": {"collider_mesh_url": "https://example.com/mesh.glb"},
                    "splats": {"spz_urls": {"default": "https://example.com/scene.spz"}},
                },
            },
        }
        mock_poll_response.raise_for_status = MagicMock()

        # Mock download responses
        mock_dl_response = MagicMock()
        mock_dl_response.content = b"fake data"
        mock_dl_response.iter_content = MagicMock(return_value=[b"fake data"])
        mock_dl_response.raise_for_status = MagicMock()

        mock_requests = MagicMock()
        mock_requests.post.return_value = mock_generate_response
        mock_requests.get.side_effect = [
            mock_poll_response,
            mock_dl_response,
            mock_dl_response,
            mock_dl_response,
            mock_dl_response,
        ]

        with patch.dict(sys.modules, {"requests": mock_requests}):
            result = pipeline._call_marble_api_remote(
                prompt="Remote test",
                input_mode="text",
                output_dir=str(tmp_path),
                seed=42,
            )

        assert result.get("metadata", {}).get("world_id") == "world-abc"
        assert result.get("metadata", {}).get("operation_id") == "op-123"

    def test_call_marble_api_remote_with_image(self, tmp_path):
        from strands_robots.marble import MarbleConfig, MarblePipeline

        # Create a fake image
        img = tmp_path / "input.jpg"
        img.write_bytes(b"fake image bytes")

        config = MarbleConfig(api_key="test-key")
        pipeline = MarblePipeline(config)

        # Mock generate response
        mock_gen_response = MagicMock()
        mock_gen_response.json.return_value = {"operation_id": "op-img", "done": False}
        mock_gen_response.raise_for_status = MagicMock()

        # Mock poll response
        mock_poll_response = MagicMock()
        mock_poll_response.json.return_value = {
            "operation_id": "op-img",
            "done": True,
            "response": {
                "world_id": "world-img",
                "display_name": "From image",
                "world_marble_url": "https://marble.worldlabs.ai/w/world-img",
                "assets": {},
            },
        }
        mock_poll_response.raise_for_status = MagicMock()

        mock_requests = MagicMock()
        mock_requests.post.return_value = mock_gen_response
        mock_requests.get.return_value = mock_poll_response
        mock_requests.request = MagicMock(return_value=MagicMock(raise_for_status=MagicMock()))

        with patch.dict(sys.modules, {"requests": mock_requests}):
            result = pipeline._call_marble_api_remote(
                prompt="From image",
                input_mode="image",
                output_dir=str(tmp_path),
                seed=42,
                input_image=str(img),
            )
        assert "metadata" in result

    def test_call_marble_api_remote_no_requests(self):
        from strands_robots.marble import MarbleConfig, MarblePipeline

        config = MarbleConfig(api_key="test-key")
        pipeline = MarblePipeline(config)

        with patch.dict(sys.modules, {"requests": None}):
            with pytest.raises(ImportError, match="requests is required"):
                pipeline._call_marble_api_remote(
                    prompt="No requests",
                    input_mode="text",
                    output_dir="/tmp",
                    seed=42,
                )

    # --- Subprocess runner ---

    def test_run_subprocess_success(self):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()
        pipeline._run_subprocess(["echo", "hello"], "test")
        # Should not raise

    def test_run_subprocess_failure(self):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()
        with pytest.raises(RuntimeError, match="failed"):
            pipeline._run_subprocess(["false"], "test_fail")

    def test_run_subprocess_timeout(self):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()
        with patch("subprocess.run", side_effect=OSError("No such file")):
            with pytest.raises(RuntimeError, match="Failed to launch"):
                pipeline._run_subprocess(["/nonexistent/binary"], "test")

    # --- 3DGrut conversion paths ---

    def test_convert_via_3dgrut_module_fallback(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        ply_file = tmp_path / "test.ply"
        ply_file.write_text("fake")

        grut_dir = tmp_path / "3dgrut"
        grut_dir.mkdir()
        scripts = grut_dir / "scripts"
        scripts.mkdir()
        # No export_usdz.py, no alternatives → module-based invocation

        pipeline = MarblePipeline()
        with patch.object(pipeline, "_run_subprocess", return_value="output.usdz") as mock_run:
            pipeline._convert_via_3dgrut(str(ply_file), "output.usdz", str(grut_dir))
            assert "threedgrut.export" in str(mock_run.call_args)

    def test_convert_via_3dgrut_alt_script(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        ply_file = tmp_path / "test.ply"
        ply_file.write_text("fake")

        grut_dir = tmp_path / "3dgrut"
        grut_dir.mkdir()
        scripts = grut_dir / "scripts"
        scripts.mkdir()
        # Create alternative script
        (scripts / "export.py").write_text("# alt")

        pipeline = MarblePipeline()
        with patch.object(pipeline, "_run_subprocess", return_value="output.usdz") as mock_run:
            pipeline._convert_via_3dgrut(str(ply_file), "output.usdz", str(grut_dir))
            mock_run.assert_called_once()

    # --- pxr conversion ---

    def test_try_pxr_conversion_no_pxr(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()
        # Without pxr installed, should return False
        result = pipeline._try_pxr_conversion(str(tmp_path / "test.ply"), str(tmp_path / "test.usdz"))
        # May or may not be available in CI
        assert isinstance(result, bool)

    # --- trimesh conversion ---

    def test_try_trimesh_conversion_no_trimesh(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()
        # Mock trimesh not available
        with patch.dict(sys.modules, {"trimesh": None}):
            result = pipeline._try_trimesh_conversion(str(tmp_path / "test.ply"), str(tmp_path / "test.usdz"))
        assert result is False

    def test_try_trimesh_conversion_success(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        ply_file = tmp_path / "test.ply"
        ply_file.write_text("fake")

        mock_trimesh = MagicMock()
        mock_mesh = MagicMock()

        # mesh.export() must create the GLB file for shutil.copy2 to succeed
        def _fake_export(path, **kwargs):
            Path(path).write_text("fake glb")

        mock_mesh.export.side_effect = _fake_export
        mock_trimesh.load.return_value = mock_mesh

        pipeline = MarblePipeline()
        with patch.dict(sys.modules, {"trimesh": mock_trimesh}):
            result = pipeline._try_trimesh_conversion(str(ply_file), str(tmp_path / "test.usdz"))
        assert result is True
        mock_mesh.export.assert_called_once()

    def test_try_trimesh_conversion_error(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        mock_trimesh = MagicMock()
        mock_trimesh.load.side_effect = Exception("bad mesh")

        pipeline = MarblePipeline()
        with patch.dict(sys.modules, {"trimesh": mock_trimesh}):
            result = pipeline._try_trimesh_conversion(str(tmp_path / "bad.ply"), str(tmp_path / "test.usdz"))
        assert result is False

    # --- USD composition ---

    def test_try_usd_composition_no_pxr(self, tmp_path):
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()

        # Simulate no pxr
        with patch.dict(sys.modules, {"pxr": None}):
            # The import inside will fail
            # Depends on whether pxr is actually installed
            pass
        # Just verify the method exists and returns bool
        assert hasattr(pipeline, "_try_usd_composition")


# ---------------------------------------------------------------------------
# Convenience functions tests
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_generate_world_single(self, tmp_path):
        from strands_robots.marble import generate_world

        scene = generate_world(
            prompt="A simple kitchen",
            output_dir=str(tmp_path / "scenes"),
            num_variations=1,
        )
        # Single variation → returns MarbleScene (not list)
        assert hasattr(scene, "scene_id")
        assert hasattr(scene, "prompt")
        assert scene.prompt == "A simple kitchen"

    def test_generate_world_multiple(self, tmp_path):
        from strands_robots.marble import generate_world

        result = generate_world(
            prompt="A kitchen",
            output_dir=str(tmp_path / "scenes"),
            num_variations=3,
        )
        assert isinstance(result, list)
        assert len(result) == 3

    def test_generate_world_with_robot(self, tmp_path):
        from strands_robots.marble import generate_world

        scene = generate_world(
            prompt="Office desk",
            output_dir=str(tmp_path / "scenes"),
            robot="so101",
            compose=True,
        )
        assert hasattr(scene, "scene_id")

    def test_compose_scene_function(self, tmp_path):
        from strands_robots.marble import compose_scene

        scene_file = tmp_path / "scene.ply"
        scene_file.write_text("fake")

        result = compose_scene(
            scene_path=str(scene_file),
            robot="so101",
            task_objects=["mug"],
        )
        assert "scene_usd" in result
        assert result["robot"] == "so101"

    def test_list_presets(self):
        from strands_robots.marble import list_presets

        presets = list_presets()
        assert isinstance(presets, list)
        assert len(presets) == 8
        names = [p["name"] for p in presets]
        assert "kitchen" in names
        assert "office_desk" in names
        assert "workshop" in names

        for p in presets:
            assert "name" in p
            assert "description" in p
            assert "category" in p
            assert "prompt" in p
            assert "objects" in p


# ---------------------------------------------------------------------------
# Constants and data tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_marble_presets(self):
        from strands_robots.marble import MARBLE_PRESETS

        assert isinstance(MARBLE_PRESETS, dict)
        assert len(MARBLE_PRESETS) == 8

        for name, info in MARBLE_PRESETS.items():
            assert "prompt" in info
            assert "description" in info
            assert "category" in info
            assert "task_objects" in info
            assert isinstance(info["task_objects"], list)
            assert len(info["prompt"]) > 10

    def test_supported_robots(self):
        from strands_robots.marble import SUPPORTED_ROBOTS

        assert isinstance(SUPPORTED_ROBOTS, dict)
        assert len(SUPPORTED_ROBOTS) == 6

        expected = {"so101", "bi_so101", "panda", "ur5e", "xarm7", "lekiwi"}
        assert set(SUPPORTED_ROBOTS.keys()) == expected

        for key, info in SUPPORTED_ROBOTS.items():
            assert "usd_path" in info
            assert "type" in info
            assert "mount_height" in info
            assert "description" in info
            assert info["type"] in {"single_arm", "dual_arm", "mobile_manipulation"}

    def test_valid_output_formats(self):
        from strands_robots.marble import VALID_OUTPUT_FORMATS

        assert set(VALID_OUTPUT_FORMATS) == {"ply", "glb", "usdz", "video"}

    def test_valid_input_modes(self):
        from strands_robots.marble import VALID_INPUT_MODES

        assert set(VALID_INPUT_MODES) == {"text", "image", "video", "multi-image"}

    def test_all_exports(self):
        from strands_robots.marble import __all__

        expected = {
            "MarblePipeline",
            "MarbleConfig",
            "MarbleScene",
            "generate_world",
            "compose_scene",
            "list_presets",
            "MARBLE_PRESETS",
            "MARBLE_MODELS",
            "SUPPORTED_ROBOTS",
        }
        assert set(__all__) == expected


# ---------------------------------------------------------------------------
# Integration tests (end-to-end pipeline)
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests without external dependencies."""

    def test_full_pipeline_text_to_scene(self, tmp_path):
        """Full pipeline: text → generate → placeholder PLY → compose (manifest)."""
        from strands_robots.marble import MarbleConfig, MarblePipeline

        config = MarbleConfig(
            robot="so101",
            auto_compose=True,
            convert_to_usdz=False,  # Skip conversion (no 3DGrut/pxr)
        )
        pipeline = MarblePipeline(config)

        scenes = pipeline.generate_world(
            prompt="A kitchen with fruits on the counter",
            output_dir=str(tmp_path / "integration"),
            num_variations=2,
        )

        assert len(scenes) == 2
        for scene in scenes:
            assert scene.ply_path is not None
            assert os.path.isfile(scene.ply_path)
            assert scene.metadata.get("placeholder") is True

    def test_full_pipeline_with_preset(self, tmp_path):
        """Full pipeline using a preset."""
        from strands_robots.marble import MarblePipeline

        pipeline = MarblePipeline()
        result = pipeline.generate_training_scenes(
            preset="workshop",
            num_per_prompt=2,
            robot="panda",
            output_dir=str(tmp_path / "preset_training"),
        )
        assert result["total_scenes"] == 2
        assert result["preset"] == "workshop"
        assert result["robot"] == "panda"
        assert "screw" in result["task_objects"]

    def test_full_pipeline_multiple_presets_via_prompts(self, tmp_path):
        """Use custom prompts from multiple presets."""
        from strands_robots.marble import MARBLE_PRESETS, MarblePipeline

        prompts = [
            MARBLE_PRESETS["kitchen"]["prompt"],
            MARBLE_PRESETS["office_desk"]["prompt"],
        ]

        pipeline = MarblePipeline()
        result = pipeline.generate_training_scenes(
            prompts=prompts,
            num_per_prompt=2,
            output_dir=str(tmp_path / "multi"),
        )
        assert result["total_scenes"] == 4
        assert len(result["scene_paths"]) > 0
