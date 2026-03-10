#!/usr/bin/env python3
"""Tests for strands_robots.tools.marble_tool.

Tests the Strands agent tool wrapper around MarblePipeline.
Uses the standard strands mock pattern (lambda decorator).
"""

import os
import sys
import types
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Mock strands SDK before importing marble_tool
# ---------------------------------------------------------------------------

_mock_strands = types.ModuleType("strands")
_mock_strands.tool = lambda f: f  # type: ignore
sys.modules.setdefault("strands", _mock_strands)


# ---------------------------------------------------------------------------
# marble_tool tests
# ---------------------------------------------------------------------------


class TestMarbleToolPresets:
    """Test the 'presets' action."""

    def test_presets_action(self):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(action="presets")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Marble Scene Presets" in text
        assert "kitchen" in text
        assert "office_desk" in text
        assert "workshop" in text
        assert "8 presets" in text

    def test_presets_contains_all_eight(self):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(action="presets")
        text = result["content"][0]["text"]
        expected = [
            "kitchen",
            "office_desk",
            "workshop",
            "living_room",
            "warehouse",
            "lab_bench",
            "outdoor_garden",
            "restaurant",
        ]
        for name in expected:
            assert name in text


class TestMarbleToolRobots:
    """Test the 'robots' action."""

    def test_robots_action(self):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(action="robots")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Supported Robots" in text
        assert "so101" in text
        assert "panda" in text
        assert "lekiwi" in text

    def test_robots_count(self):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(action="robots")
        text = result["content"][0]["text"]
        assert "6 robots" in text


class TestMarbleToolInfo:
    """Test the 'info' action."""

    def test_info_default(self):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(action="info")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Marble 3D World Generation" in text
        assert "Pipeline" in text

    def test_info_preset(self):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(action="info", preset="kitchen")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "kitchen" in text.lower()
        assert "orange" in text

    def test_info_unknown_preset(self):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(action="info", preset="nonexistent")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Unknown preset" in text or "nonexistent" in text


class TestMarbleToolGenerate:
    """Test the 'generate' action."""

    def test_generate_with_prompt(self, tmp_path):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(
            action="generate",
            prompt="A kitchen with fruits",
            output_dir=str(tmp_path / "gen"),
            num_variations=1,
        )
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "World Generation Complete" in text
        assert "Kitchen" in text or "kitchen" in text.lower()

    def test_generate_with_preset(self, tmp_path):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(
            action="generate",
            preset="workshop",
            output_dir=str(tmp_path / "gen"),
        )
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "World Generation Complete" in text

    def test_generate_no_prompt_or_preset(self):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(action="generate")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "prompt" in text.lower() or "preset" in text.lower()

    def test_generate_multiple_variations(self, tmp_path):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(
            action="generate",
            prompt="Test scene",
            output_dir=str(tmp_path / "gen"),
            num_variations=3,
        )
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "3" in text  # Variations count

    def test_generate_with_robot(self, tmp_path):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(
            action="generate",
            prompt="Kitchen scene",
            robot="so101",
            output_dir=str(tmp_path / "gen"),
        )
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "so101" in text

    def test_generate_with_task_objects(self, tmp_path):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(
            action="generate",
            prompt="Kitchen",
            task_objects="orange,mug,plate",
            robot="so101",
            output_dir=str(tmp_path / "gen"),
        )
        assert result["status"] == "success"

    def test_generate_placeholder_warning(self, tmp_path):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(
            action="generate",
            prompt="Test",
            output_dir=str(tmp_path / "gen"),
        )
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Placeholder" in text or "MARBLE_API_KEY" in text

    def test_generate_returns_scenes(self, tmp_path):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(
            action="generate",
            prompt="Test",
            output_dir=str(tmp_path / "gen"),
            num_variations=2,
        )
        assert "scenes" in result
        assert len(result["scenes"]) == 2
        for scene in result["scenes"]:
            assert "scene_id" in scene
            assert "prompt" in scene


class TestMarbleToolCompose:
    """Test the 'compose' action."""

    def test_compose_no_scene_path(self):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(action="compose")
        assert result["status"] == "error"
        assert "scene_path" in result["content"][0]["text"]

    def test_compose_no_robot(self):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(action="compose", scene_path="/tmp/fake.ply")
        assert result["status"] == "error"
        assert "robot" in result["content"][0]["text"]

    def test_compose_success(self, tmp_path):
        from strands_robots.tools.marble_tool import marble_tool

        scene_file = tmp_path / "scene.ply"
        scene_file.write_text("fake ply data")

        result = marble_tool(
            action="compose",
            scene_path=str(scene_file),
            robot="so101",
            task_objects="orange,plate",
        )
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Scene Composed" in text
        assert "so101" in text

    def test_compose_with_position(self, tmp_path):
        from strands_robots.tools.marble_tool import marble_tool

        scene_file = tmp_path / "scene.ply"
        scene_file.write_text("fake")

        result = marble_tool(
            action="compose",
            scene_path=str(scene_file),
            robot="panda",
            robot_position="[1.0, 2.0, 0.5]",
        )
        assert result["status"] == "success"


class TestMarbleToolBatch:
    """Test the 'batch' action."""

    def test_batch_with_preset(self, tmp_path):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(
            action="batch",
            preset="kitchen",
            num_per_prompt=3,
            output_dir=str(tmp_path / "batch"),
        )
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Batch Training" in text
        assert "3" in text

    def test_batch_with_custom_prompts(self, tmp_path):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(
            action="batch",
            prompt="A kitchen|An office",  # pipe-separated
            num_per_prompt=2,
            output_dir=str(tmp_path / "batch"),
        )
        assert result["status"] == "success"

    def test_batch_with_robot(self, tmp_path):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(
            action="batch",
            preset="office_desk",
            num_per_prompt=2,
            robot="so101",
            output_dir=str(tmp_path / "batch"),
        )
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "so101" in text


class TestMarbleToolConvert:
    """Test the 'convert' action."""

    def test_convert_no_ply_path(self):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(action="convert")
        assert result["status"] == "error"
        assert "ply_path" in result["content"][0]["text"]

    def test_convert_file_not_found(self):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(action="convert", ply_path="/nonexistent/file.ply")
        assert result["status"] == "error"

    def test_convert_success(self, tmp_path):
        from strands_robots.marble import MarblePipeline
        from strands_robots.tools.marble_tool import marble_tool

        ply_file = tmp_path / "test.ply"
        ply_file.write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n0.0 0.0 0.0\n"
        )

        # Mock the pipeline's convert method to succeed
        with patch.object(MarblePipeline, "convert_ply_to_usdz", return_value=str(tmp_path / "test.usdz")):
            result = marble_tool(action="convert", ply_path=str(ply_file))
            assert result["status"] == "success"
            text = result["content"][0]["text"]
            assert "Conversion Complete" in text


class TestMarbleToolUnknown:
    """Test unknown action handling."""

    def test_unknown_action(self):
        from strands_robots.tools.marble_tool import marble_tool

        result = marble_tool(action="destroy_world")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "Unknown action" in text or "destroy_world" in text


class TestMarbleToolGlobalPipeline:
    """Test the global pipeline caching."""

    def test_get_pipeline_creates_once(self):
        from strands_robots.tools import marble_tool as mt_module

        # Reset global
        mt_module._pipeline = None

        p1 = mt_module._get_pipeline(robot="so101")
        p2 = mt_module._get_pipeline()  # Should return cached
        assert p1 is p2

        # Cleanup
        mt_module._pipeline = None

    def test_get_pipeline_with_kwargs(self):
        from strands_robots.tools import marble_tool as mt_module

        mt_module._pipeline = None

        p = mt_module._get_pipeline(robot="panda", seed=99)
        assert p.config.robot == "panda"
        assert p.config.seed == 99

        mt_module._pipeline = None


# ══════════════════════════════════════════════════════════════════════════════
# list_worlds action
# ══════════════════════════════════════════════════════════════════════════════


class TestMarbleToolListWorlds:
    """Test the 'list_worlds' action."""

    def test_list_worlds_no_api_key(self):
        from strands_robots.tools import marble_tool as mt_module

        mt_module._pipeline = None
        old_env = os.environ.get("WLT_API_KEY")
        os.environ.pop("WLT_API_KEY", None)
        os.environ.pop("MARBLE_API_KEY", None)

        result = mt_module.marble_tool(action="list_worlds")
        assert result["status"] == "error"
        assert "API_KEY" in result["content"][0]["text"]

        mt_module._pipeline = None
        if old_env:
            os.environ["WLT_API_KEY"] = old_env

    def test_list_worlds_success(self):
        from strands_robots.tools import marble_tool as mt_module

        mt_module._pipeline = None

        mock_pipeline = MagicMock()
        mock_pipeline.config.api_key = "test_key_123"
        mock_pipeline.list_worlds.return_value = {
            "worlds": [
                {
                    "world_id": "w1",
                    "display_name": "Kitchen Scene",
                    "model": "Marble 0.1-plus",
                    "world_marble_url": "https://marble.com/w1",
                },
                {
                    "world_id": "w2",
                    "display_name": "Office Desk",
                    "model": "Marble 0.1-plus",
                    "world_marble_url": "https://marble.com/w2",
                },
            ],
        }

        with patch.object(mt_module, "_get_pipeline", return_value=mock_pipeline):
            result = mt_module.marble_tool(action="list_worlds")

        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "2 results" in text
        assert "Kitchen Scene" in text
        assert "Office Desk" in text
        assert "w1" in text

        mt_module._pipeline = None

    def test_list_worlds_with_filters(self):
        from strands_robots.tools import marble_tool as mt_module

        mt_module._pipeline = None

        mock_pipeline = MagicMock()
        mock_pipeline.config.api_key = "key"
        mock_pipeline.list_worlds.return_value = {"worlds": []}

        with patch.object(mt_module, "_get_pipeline", return_value=mock_pipeline):
            result = mt_module.marble_tool(
                action="list_worlds",
                tags="kitchen,robot",
                is_public="true",
                status="completed",
            )

        assert result["status"] == "success"
        call_kw = mock_pipeline.list_worlds.call_args[1]
        assert call_kw["tags"] == ["kitchen", "robot"]
        assert call_kw["is_public"] is True
        assert call_kw["status"] == "completed"

        mt_module._pipeline = None

    def test_list_worlds_next_page_token(self):
        from strands_robots.tools import marble_tool as mt_module

        mt_module._pipeline = None

        mock_pipeline = MagicMock()
        mock_pipeline.config.api_key = "key"
        mock_pipeline.list_worlds.return_value = {
            "worlds": [{"world_id": "w1", "display_name": "W1", "model": "M", "world_marble_url": ""}],
            "next_page_token": "abc123",
        }

        with patch.object(mt_module, "_get_pipeline", return_value=mock_pipeline):
            result = mt_module.marble_tool(action="list_worlds")

        assert result["status"] == "success"
        assert "More results" in result["content"][0]["text"]

        mt_module._pipeline = None

    def test_list_worlds_is_public_false(self):
        from strands_robots.tools import marble_tool as mt_module

        mt_module._pipeline = None

        mock_pipeline = MagicMock()
        mock_pipeline.config.api_key = "key"
        mock_pipeline.list_worlds.return_value = {"worlds": []}

        with patch.object(mt_module, "_get_pipeline", return_value=mock_pipeline):
            mt_module.marble_tool(action="list_worlds", is_public="false")

        call_kw = mock_pipeline.list_worlds.call_args[1]
        assert call_kw["is_public"] is False

        mt_module._pipeline = None


# ══════════════════════════════════════════════════════════════════════════════
# get_world action
# ══════════════════════════════════════════════════════════════════════════════


class TestMarbleToolGetWorld:
    """Test the 'get_world' action."""

    def test_get_world_missing_id(self):
        from strands_robots.tools import marble_tool as mt_module

        result = mt_module.marble_tool(action="get_world")
        assert result["status"] == "error"
        assert "world_id" in result["content"][0]["text"].lower()

    def test_get_world_no_api_key(self):
        from strands_robots.tools import marble_tool as mt_module

        mt_module._pipeline = None
        old_env = os.environ.get("WLT_API_KEY")
        os.environ.pop("WLT_API_KEY", None)
        os.environ.pop("MARBLE_API_KEY", None)

        result = mt_module.marble_tool(action="get_world", world_id="w123")
        assert result["status"] == "error"
        assert "API_KEY" in result["content"][0]["text"]

        mt_module._pipeline = None
        if old_env:
            os.environ["WLT_API_KEY"] = old_env

    def test_get_world_success(self):
        from strands_robots.tools import marble_tool as mt_module

        mt_module._pipeline = None

        mock_pipeline = MagicMock()
        mock_pipeline.config.api_key = "test_key_123"
        mock_pipeline.get_world.return_value = {
            "world_id": "w123",
            "display_name": "Kitchen Counter",
            "model": "Marble 0.1-plus",
            "world_marble_url": "https://marble.com/w123",
            "created_at": "2026-03-04T00:00:00Z",
            "assets": {
                "caption": "A modern kitchen scene",
                "thumbnail_url": "https://img.com/thumb.jpg",
                "imagery": {"pano_url": "https://img.com/pano.jpg"},
                "mesh": {"collider_mesh_url": "https://img.com/mesh.glb"},
                "splats": {"spz_urls": {"high": "https://s.com/h.spz", "low": "https://s.com/l.spz"}},
            },
            "tags": ["kitchen", "indoor"],
        }

        with patch.object(mt_module, "_get_pipeline", return_value=mock_pipeline):
            result = mt_module.marble_tool(action="get_world", world_id="w123")

        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Kitchen Counter" in text
        assert "w123" in text
        assert "kitchen" in text
        assert "Pano:" in text
        assert "Collider Mesh:" in text
        assert "SPZ Splats: 2 file(s)" in text
        assert "caption" in text.lower()

        mt_module._pipeline = None

    def test_get_world_minimal_assets(self):
        from strands_robots.tools import marble_tool as mt_module

        mt_module._pipeline = None

        mock_pipeline = MagicMock()
        mock_pipeline.config.api_key = "key"
        mock_pipeline.get_world.return_value = {
            "world_id": "w456",
            "display_name": "Empty Scene",
            "assets": {},
        }

        with patch.object(mt_module, "_get_pipeline", return_value=mock_pipeline):
            result = mt_module.marble_tool(action="get_world", world_id="w456")

        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Empty Scene" in text
        # Shouldn't have optional fields
        assert "SPZ" not in text
        assert "Pano:" not in text

        mt_module._pipeline = None


# ══════════════════════════════════════════════════════════════════════════════
# generate action — uncovered branches
# ══════════════════════════════════════════════════════════════════════════════


class TestMarbleToolGenerateExtra:
    """Cover generate action branches: video input, image input, composition, scene properties."""

    def test_generate_with_video_input(self, tmp_path):
        from strands_robots.tools import marble_tool as mt_module

        mt_module._pipeline = None

        # Create a fake video file
        vid = tmp_path / "test.mp4"
        vid.write_text("fake")

        mock_scene = MagicMock()
        mock_scene.scene_id = "s1"
        mock_scene.world_id = "w1"
        mock_scene.world_marble_url = "https://marble.com/w1"
        mock_scene.caption = "A video-derived scene"
        mock_scene.spz_path = "/out/s1.spz"
        mock_scene.ply_path = None
        mock_scene.glb_path = None
        mock_scene.pano_path = "/out/pano.jpg"
        mock_scene.usdz_path = None
        mock_scene.scene_usd = None
        mock_scene.pano_url = "https://img.com/pano"
        mock_scene.spz_urls = {}
        mock_scene.metadata = {}
        mock_scene.best_background = None

        mock_pipeline = MagicMock()
        mock_pipeline.config.auto_compose = True
        mock_pipeline.generate_world.return_value = [mock_scene]

        with patch.object(mt_module, "_get_pipeline", return_value=mock_pipeline):
            result = mt_module.marble_tool(
                action="generate",
                prompt="kitchen",
                input_video=str(vid),
                output_dir=str(tmp_path),
            )

        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "s1" in text

        mt_module._pipeline = None

    def test_generate_with_image_url(self, tmp_path):
        from strands_robots.tools import marble_tool as mt_module

        mt_module._pipeline = None

        mock_scene = MagicMock()
        mock_scene.scene_id = "s2"
        mock_scene.world_id = None
        mock_scene.world_marble_url = None
        mock_scene.caption = None
        mock_scene.spz_path = None
        mock_scene.ply_path = None
        mock_scene.glb_path = "/out/mesh.glb"
        mock_scene.pano_path = None
        mock_scene.usdz_path = "/out/scene.usdz"
        mock_scene.scene_usd = "/out/scene.usd"
        mock_scene.pano_url = None
        mock_scene.spz_urls = {"hd": "https://s.com/hd.spz"}
        mock_scene.metadata = {"placeholder": True}
        mock_scene.best_background = None

        mock_pipeline = MagicMock()
        mock_pipeline.config.auto_compose = True
        mock_pipeline.generate_world.return_value = [mock_scene]

        with patch.object(mt_module, "_get_pipeline", return_value=mock_pipeline):
            result = mt_module.marble_tool(
                action="generate",
                prompt="office",
                input_image="https://example.com/photo.jpg",
                output_dir=str(tmp_path),
            )

        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "GLB" in text
        assert "USDZ" in text
        assert "Composed USD" in text
        assert "placeholder" in text.lower()
        assert "SPZ splats" in text

        mt_module._pipeline = None

    def test_generate_with_composition(self, tmp_path):
        from strands_robots.tools import marble_tool as mt_module

        mt_module._pipeline = None

        mock_scene = MagicMock()
        mock_scene.scene_id = "s3"
        mock_scene.world_id = None
        mock_scene.world_marble_url = None
        mock_scene.caption = None
        mock_scene.spz_path = None
        mock_scene.ply_path = "/out/s3.ply"
        mock_scene.glb_path = None
        mock_scene.pano_path = None
        mock_scene.usdz_path = None
        mock_scene.scene_usd = None
        mock_scene.pano_url = None
        mock_scene.spz_urls = {}
        mock_scene.metadata = {}
        mock_scene.best_background = "/out/bg.jpg"
        mock_scene.composed = False

        mock_pipeline = MagicMock()
        mock_pipeline.config.auto_compose = False  # Manual composition
        mock_pipeline.generate_world.return_value = [mock_scene]
        mock_pipeline.compose_scene.return_value = {"scene_usd": "/out/composed.usd"}

        with patch.object(mt_module, "_get_pipeline", return_value=mock_pipeline):
            result = mt_module.marble_tool(
                action="generate",
                prompt="workshop",
                robot="so101",
                task_objects="hammer, screwdriver",
                output_dir=str(tmp_path),
            )

        assert result["status"] == "success"
        mock_pipeline.compose_scene.assert_called_once()
        compose_kw = mock_pipeline.compose_scene.call_args[1]
        assert compose_kw["robot"] == "so101"
        assert compose_kw["task_objects"] == ["hammer", "screwdriver"]

        mt_module._pipeline = None

    def test_generate_composition_failure_graceful(self, tmp_path):
        from strands_robots.tools import marble_tool as mt_module

        mt_module._pipeline = None

        mock_scene = MagicMock()
        mock_scene.scene_id = "s4"
        mock_scene.world_id = None
        mock_scene.world_marble_url = None
        mock_scene.caption = None
        mock_scene.spz_path = None
        mock_scene.ply_path = None
        mock_scene.glb_path = None
        mock_scene.pano_path = None
        mock_scene.usdz_path = None
        mock_scene.scene_usd = None
        mock_scene.pano_url = None
        mock_scene.spz_urls = {}
        mock_scene.metadata = {}
        mock_scene.best_background = "/out/bg.jpg"
        mock_scene.composed = False

        mock_pipeline = MagicMock()
        mock_pipeline.config.auto_compose = False
        mock_pipeline.generate_world.return_value = [mock_scene]
        mock_pipeline.compose_scene.side_effect = Exception("Composition error")

        with patch.object(mt_module, "_get_pipeline", return_value=mock_pipeline):
            result = mt_module.marble_tool(
                action="generate",
                prompt="lab",
                robot="panda",
                output_dir=str(tmp_path),
            )

        # Should still succeed despite composition failure
        assert result["status"] == "success"

        mt_module._pipeline = None
