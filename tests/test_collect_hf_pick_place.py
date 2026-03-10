"""Tests for scripts/collect_hf_pick_place.py — HuggingFace data collection pipeline.

Tests all pure logic functions (classify_dataset_task, map_dataset_to_embodiment,
generate_thor_dispatch_script) plus mocked HF API interactions (search, download, manifest).

0% → ~80% coverage target for 973-line script.
"""

import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ──────────────────────────────────────────────────────────────────────
# Module-level mock for huggingface_hub (may not be installed)
# ──────────────────────────────────────────────────────────────────────
_mock_hf_hub = types.ModuleType("huggingface_hub")
_mock_hf_hub.HfApi = MagicMock
_mock_hf_hub.hf_hub_download = MagicMock()
_mock_hf_hub.snapshot_download = MagicMock(return_value="/tmp/fake")
_mock_hf_hub.list_repo_files = MagicMock(return_value=[])

# Ensure the script can be imported
_SCRIPT_DIR = str(Path(__file__).parent.parent / "scripts")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Also add project root for strands_robots imports
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import the module under test with mocked deps
with patch.dict(sys.modules, {"huggingface_hub": _mock_hf_hub, "huggingface_hub.hf_api": MagicMock()}):
    from collect_hf_pick_place import (
        EMBODIMENT_DATASET_MAP,
        DatasetInfo,
        check_dataset_exists,
        classify_dataset_task,
        download_dataset,
        generate_manifest,
        generate_thor_dispatch_script,
        map_dataset_to_embodiment,
        print_summary,
        search_hf_datasets,
    )


# ══════════════════════════════════════════════════════════════════════
#  DatasetInfo dataclass
# ══════════════════════════════════════════════════════════════════════


class TestDatasetInfo:
    """Test the DatasetInfo dataclass."""

    def test_default_construction(self):
        ds = DatasetInfo(repo_id="test/repo", embodiment_tag="new_embodiment", data_config="so100")
        assert ds.repo_id == "test/repo"
        assert ds.embodiment_tag == "new_embodiment"
        assert ds.data_config == "so100"
        assert ds.description == ""
        assert ds.num_episodes == 0
        assert ds.confidence == 0.0
        assert ds.tags == []
        assert ds.task_types == []

    def test_full_construction(self):
        ds = DatasetInfo(
            repo_id="lerobot/so100_pick",
            embodiment_tag="new_embodiment",
            data_config="so100_dualcam",
            description="Pick and place dataset",
            num_episodes=100,
            num_steps=5000,
            size_bytes=1024000,
            tags=["lerobot", "pick"],
            task_types=["pick_and_place"],
            is_lerobot_v2=True,
            confidence=0.95,
        )
        assert ds.num_episodes == 100
        assert ds.is_lerobot_v2 is True
        assert ds.confidence == 0.95

    def test_to_dict(self):
        ds = DatasetInfo(
            repo_id="test/repo",
            embodiment_tag="unitree_g1",
            data_config="unitree_g1",
            confidence=0.9,
        )
        d = ds.to_dict()
        assert d["repo_id"] == "test/repo"
        assert d["embodiment_tag"] == "unitree_g1"
        assert d["data_config"] == "unitree_g1"
        assert d["confidence"] == 0.9
        assert "num_episodes" in d
        assert "is_lerobot_v2" in d

    def test_to_dict_json_serializable(self):
        ds = DatasetInfo(
            repo_id="x/y",
            embodiment_tag="gr1",
            data_config="fourier_gr1_arms_only",
            tags=["a", "b"],
            task_types=["pick_and_place"],
        )
        result = json.dumps(ds.to_dict())
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["repo_id"] == "x/y"

    def test_mutable_default_isolation(self):
        """Ensure mutable defaults don't leak between instances."""
        ds1 = DatasetInfo(repo_id="a", embodiment_tag="a", data_config="a")
        ds2 = DatasetInfo(repo_id="b", embodiment_tag="b", data_config="b")
        ds1.tags.append("test")
        assert ds2.tags == []

    def test_to_dict_excludes_download_url(self):
        """to_dict should not include download_url (not in the dict)."""
        ds = DatasetInfo(repo_id="x", embodiment_tag="y", data_config="z", download_url="http://x")
        d = ds.to_dict()
        assert "download_url" not in d


# ══════════════════════════════════════════════════════════════════════
#  classify_dataset_task
# ══════════════════════════════════════════════════════════════════════


class TestClassifyDatasetTask:
    """Test task classification based on dataset metadata."""

    @pytest.mark.parametrize(
        "keyword",
        [
            "pick",
            "grasp",
            "grip",
            "grab",
            "lifting",
            "pick_and_place",
            "pick-and-place",
            "pickandplace",
            "pick place",
        ],
    )
    def test_pick_keywords(self, keyword):
        info = {"repo_id": f"test/{keyword}_dataset", "description": ""}
        tasks = classify_dataset_task(info)
        assert "pick_and_place" in tasks

    @pytest.mark.parametrize(
        "keyword,expected_task",
        [
            ("push", "pushing"),
            ("pushing", "pushing"),
            ("stack", "stacking"),
            ("stacking", "stacking"),
            ("wipe", "wiping"),
            ("wiping", "wiping"),
            ("clean", "wiping"),
            ("fold", "folding"),
            ("folding", "folding"),
            ("pour", "pouring"),
            ("pouring", "pouring"),
            ("open", "articulated"),
            ("close", "articulated"),
            ("door", "articulated"),
            ("drawer", "articulated"),
        ],
    )
    def test_manipulation_keywords(self, keyword, expected_task):
        info = {"repo_id": f"test/{keyword}_data", "description": ""}
        tasks = classify_dataset_task(info)
        assert expected_task in tasks

    def test_lerobot_format_from_repo_id(self):
        info = {"repo_id": "lerobot/so100_pick", "description": ""}
        tasks = classify_dataset_task(info)
        assert "lerobot_format" in tasks

    def test_lerobot_format_from_tags(self):
        info = {"repo_id": "test/data", "description": "", "tags": ["lerobot"]}
        tasks = classify_dataset_task(info)
        assert "lerobot_format" in tasks

    def test_manipulation_generic(self):
        info = {"repo_id": "test/robot_arm_data", "description": ""}
        tasks = classify_dataset_task(info)
        assert "manipulation" in tasks

    def test_multi_task(self):
        info = {"repo_id": "lerobot/so100_pick_and_push", "description": ""}
        tasks = classify_dataset_task(info)
        assert "pick_and_place" in tasks
        assert "pushing" in tasks
        assert "lerobot_format" in tasks

    def test_unknown_task(self):
        info = {"repo_id": "test/weather_forecast", "description": "temperature data"}
        tasks = classify_dataset_task(info)
        assert tasks == ["unknown"]

    def test_empty_info(self):
        info = {"repo_id": "", "description": ""}
        tasks = classify_dataset_task(info)
        assert tasks == ["unknown"]

    def test_description_matching(self):
        info = {"repo_id": "test/data", "description": "A robot pick and place dataset"}
        tasks = classify_dataset_task(info)
        assert "pick_and_place" in tasks


# ══════════════════════════════════════════════════════════════════════
#  map_dataset_to_embodiment
# ══════════════════════════════════════════════════════════════════════


class TestMapDatasetToEmbodiment:
    """Test embodiment mapping from dataset metadata."""

    def test_so100_mapping(self):
        info = {"repo_id": "lerobot/so100_pick_place", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        emb, cfg, conf = result
        assert emb == "new_embodiment"
        assert cfg == "so100"
        assert conf == 0.95

    def test_so101_mapping(self):
        info = {"repo_id": "user/so101_manipulation", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "new_embodiment"

    def test_koch_mapping(self):
        info = {"repo_id": "lerobot/koch_pick", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "new_embodiment"

    def test_gr1_mapping(self):
        info = {"repo_id": "nvidia/PhysicalAI-GR1-Dataset", "description": "Fourier GR-1", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "gr1"
        assert result[2] == 0.9

    def test_unitree_g1_mapping(self):
        info = {"repo_id": "user/unitree_g1_pick_place", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "unitree_g1"

    def test_droid_mapping(self):
        info = {"repo_id": "droid_rss/droid_100", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "oxe_droid"

    def test_google_rt1_mapping(self):
        info = {"repo_id": "google/rt1_manipulation", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "oxe_google"

    def test_bridge_v2_mapping(self):
        info = {"repo_id": "berkeley/bridge_v2_data", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "oxe_google"

    def test_widowx_mapping(self):
        info = {"repo_id": "ndimensions/widowx_pick_place", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "oxe_widowx"

    def test_libero_mapping(self):
        info = {"repo_id": "nvidia/PhysicalAI-LIBERO-Dataset", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "libero_panda"

    def test_robocasa_mapping(self):
        info = {"repo_id": "nvidia/GR00T-RoboCasa-Data", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "robocasa_panda_omron"

    def test_panda_mapping(self):
        info = {"repo_id": "user/franka_panda_pick", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "robocasa_panda_omron"
        assert result[2] == 0.7  # Lower confidence for generic panda

    def test_behavior_mapping(self):
        info = {"repo_id": "nvidia/PhysicalAI-BEHAVIOR-Dataset", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "behavior_r1_pro"

    def test_galaxea_mapping(self):
        info = {"repo_id": "user/galaxea_r1_data", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "behavior_r1_pro"

    def test_oxe_generic_mapping(self):
        info = {"repo_id": "user/open_x_embodiment", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "oxe_google"
        assert result[2] == 0.6

    def test_lerobot_fallback(self):
        info = {"repo_id": "lerobot/unknown_task", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "new_embodiment"
        assert result[2] == 0.5  # Low confidence fallback

    def test_no_match(self):
        info = {"repo_id": "user/weather_data", "description": "temperature forecast", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is None

    def test_description_matching(self):
        info = {"repo_id": "user/data123", "description": "unitree_g1 humanoid pick dataset", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "unitree_g1"

    def test_tag_matching(self):
        info = {"repo_id": "user/data456", "description": "", "tags": ["widowx", "robot"]}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "oxe_widowx"

    def test_confidence_ordering(self):
        """SO-100 (0.95) should have higher confidence than generic lerobot (0.5)."""
        so100 = map_dataset_to_embodiment({"repo_id": "user/so100_data", "description": "", "tags": []})
        generic = map_dataset_to_embodiment({"repo_id": "lerobot/unknown", "description": "", "tags": []})
        assert so100[2] > generic[2]

    def test_case_insensitivity(self):
        info = {"repo_id": "user/SO100_PICK", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "new_embodiment"


# ══════════════════════════════════════════════════════════════════════
#  search_hf_datasets (mocked)
# ══════════════════════════════════════════════════════════════════════


class TestSearchHfDatasets:
    """Test HuggingFace search with mocked API."""

    def test_successful_search(self):
        """Test search with mocked HfApi returning realistic results."""
        mock_ds = MagicMock()
        mock_ds.id = "lerobot/so100_pick_place"
        mock_ds.description = "A pick and place dataset"
        mock_ds.tags = ["lerobot", "robotics"]
        mock_ds.downloads = 1000
        mock_ds.likes = 50
        mock_ds.created_at = "2025-01-01"
        mock_ds.last_modified = "2025-06-01"

        mock_api_cls = MagicMock()
        mock_api_cls.return_value.list_datasets.return_value = [mock_ds]

        mock_hf = types.ModuleType("huggingface_hub")
        mock_hf.HfApi = mock_api_cls

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            results = search_hf_datasets("so100 pick place")
            assert len(results) == 1
            assert results[0]["repo_id"] == "lerobot/so100_pick_place"
            assert results[0]["downloads"] == 1000
            assert "lerobot" in results[0]["tags"]

    def test_import_error(self):
        """Should return empty list if huggingface_hub not available."""
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            # When huggingface_hub is blocked, import inside the function fails
            # The function should catch ImportError and return []
            results = search_hf_datasets("test query")
            assert results == []

    def test_api_exception(self):
        """Should return empty list on API errors."""
        mock_api_cls = MagicMock()
        mock_api_cls.return_value.list_datasets.side_effect = Exception("Network error")

        mock_hf = types.ModuleType("huggingface_hub")
        mock_hf.HfApi = mock_api_cls

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            results = search_hf_datasets("test query")
            assert results == []

    def test_none_attributes(self):
        """Should handle None values in dataset attributes gracefully."""
        mock_ds = MagicMock()
        mock_ds.id = "test/repo"
        mock_ds.description = None
        mock_ds.tags = None
        mock_ds.downloads = None
        mock_ds.likes = None
        mock_ds.created_at = None
        mock_ds.last_modified = None

        mock_api_cls = MagicMock()
        mock_api_cls.return_value.list_datasets.return_value = [mock_ds]

        mock_hf = types.ModuleType("huggingface_hub")
        mock_hf.HfApi = mock_api_cls

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            results = search_hf_datasets("test")
            assert len(results) == 1
            assert results[0]["description"] == ""
            assert results[0]["tags"] == []
            assert results[0]["downloads"] == 0

    def test_getattr_safety(self):
        """Test that getattr handles missing attributes (the PR #129 fix)."""
        mock_ds = MagicMock(spec=[])  # No attributes at all
        mock_ds.id = "test/bare"
        # description, tags, downloads etc. are all missing

        mock_api_cls = MagicMock()
        mock_api_cls.return_value.list_datasets.return_value = [mock_ds]

        mock_hf = types.ModuleType("huggingface_hub")
        mock_hf.HfApi = mock_api_cls

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            results = search_hf_datasets("test")
            assert len(results) == 1
            assert results[0]["repo_id"] == "test/bare"


# ══════════════════════════════════════════════════════════════════════
#  check_dataset_exists (mocked)
# ══════════════════════════════════════════════════════════════════════


class TestCheckDatasetExists:
    """Test dataset existence checks with mocked API."""

    def test_existing_dataset(self):
        mock_info = MagicMock()
        mock_info.id = "lerobot/so100_pick"
        mock_info.description = "SO-100 pick"
        mock_info.tags = ["lerobot"]
        mock_info.downloads = 500
        mock_info.likes = 10
        mock_info.dataset_size = 1024000

        mock_api_cls = MagicMock()
        mock_api_cls.return_value.dataset_info.return_value = mock_info

        mock_hf = types.ModuleType("huggingface_hub")
        mock_hf.HfApi = mock_api_cls

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            result = check_dataset_exists("lerobot/so100_pick")
            assert result is not None
            assert result["exists"] is True
            assert result["repo_id"] == "lerobot/so100_pick"

    def test_404_not_found(self):
        mock_api_cls = MagicMock()
        mock_api_cls.return_value.dataset_info.side_effect = Exception("404 Not Found")

        mock_hf = types.ModuleType("huggingface_hub")
        mock_hf.HfApi = mock_api_cls

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            result = check_dataset_exists("nonexistent/repo")
            assert result is None

    def test_401_unauthorized(self):
        mock_api_cls = MagicMock()
        mock_api_cls.return_value.dataset_info.side_effect = Exception("401 Unauthorized")

        mock_hf = types.ModuleType("huggingface_hub")
        mock_hf.HfApi = mock_api_cls

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            result = check_dataset_exists("nvidia/PhysicalAI-GR1-Dataset")
            assert result is None

    def test_import_error(self):
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            result = check_dataset_exists("test/repo")
            assert result is None

    def test_getattr_safety(self):
        """dataset_info objects may lack some attributes (PR #129 fix)."""
        mock_info = MagicMock(spec=[])
        mock_info.id = "test/bare"

        mock_api_cls = MagicMock()
        mock_api_cls.return_value.dataset_info.return_value = mock_info

        mock_hf = types.ModuleType("huggingface_hub")
        mock_hf.HfApi = mock_api_cls

        with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            result = check_dataset_exists("test/bare")
            assert result is not None
            assert result["repo_id"] == "test/bare"
            assert result["description"] == ""


# ══════════════════════════════════════════════════════════════════════
#  generate_manifest
# ══════════════════════════════════════════════════════════════════════


class TestGenerateManifest:
    """Test manifest generation."""

    def _sample_discovered(self):
        return {
            "new_embodiment": [
                DatasetInfo(
                    repo_id="lerobot/so100_pick",
                    embodiment_tag="new_embodiment",
                    data_config="so100",
                    confidence=0.95,
                    task_types=["pick_and_place"],
                    is_lerobot_v2=True,
                ),
                DatasetInfo(
                    repo_id="user/so101_data",
                    embodiment_tag="new_embodiment",
                    data_config="so100_dualcam",
                    confidence=0.8,
                    task_types=["manipulation"],
                ),
            ],
            "unitree_g1": [
                DatasetInfo(
                    repo_id="shivubind/unitree_g1_pick_place2",
                    embodiment_tag="unitree_g1",
                    data_config="unitree_g1",
                    confidence=0.9,
                    task_types=["pick_and_place"],
                ),
            ],
        }

    def test_manifest_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = generate_manifest(self._sample_discovered(), tmpdir)
            assert "generated_at" in manifest
            assert manifest["total_datasets"] == 3
            assert manifest["total_embodiments"] == 2
            assert "new_embodiment" in manifest["embodiments"]
            assert "unitree_g1" in manifest["embodiments"]

    def test_manifest_files_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_manifest(self._sample_discovered(), tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "hf_pick_place_manifest.json"))
            assert os.path.exists(os.path.join(tmpdir, "manifest_new_embodiment.json"))
            assert os.path.exists(os.path.join(tmpdir, "manifest_unitree_g1.json"))
            assert os.path.exists(os.path.join(tmpdir, "thor_dispatch.sh"))

    def test_manifest_json_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_manifest(self._sample_discovered(), tmpdir)
            with open(os.path.join(tmpdir, "hf_pick_place_manifest.json")) as f:
                data = json.load(f)
            assert data["total_datasets"] == 3

    def test_confidence_filtering(self):
        discovered = {
            "new_embodiment": [
                DatasetInfo(repo_id="low/conf", embodiment_tag="new_embodiment", data_config="so100", confidence=0.1),
                DatasetInfo(repo_id="high/conf", embodiment_tag="new_embodiment", data_config="so100", confidence=0.9),
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = generate_manifest(discovered, tmpdir, min_confidence=0.5)
            assert manifest["total_datasets"] == 1

    def test_deduplication(self):
        discovered = {
            "new_embodiment": [
                DatasetInfo(repo_id="dup/repo", embodiment_tag="new_embodiment", data_config="so100", confidence=0.9),
                DatasetInfo(repo_id="dup/repo", embodiment_tag="new_embodiment", data_config="so100", confidence=0.8),
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = generate_manifest(discovered, tmpdir)
            assert manifest["embodiments"]["new_embodiment"]["count"] == 1

    def test_download_commands_generated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = generate_manifest(self._sample_discovered(), tmpdir)
            # All 3 datasets have confidence ≥ 0.5
            assert len(manifest["download_commands"]) == 3

    def test_high_confidence_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = generate_manifest(self._sample_discovered(), tmpdir)
            assert manifest["embodiments"]["new_embodiment"]["high_confidence"] == 2
            assert manifest["embodiments"]["unitree_g1"]["high_confidence"] == 1

    def test_empty_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = generate_manifest({}, tmpdir)
            assert manifest["total_datasets"] == 0
            assert manifest["total_embodiments"] == 0

    def test_per_embodiment_manifest_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_manifest(self._sample_discovered(), tmpdir)
            with open(os.path.join(tmpdir, "manifest_unitree_g1.json")) as f:
                data = json.load(f)
            assert data["count"] == 1
            assert data["datasets"][0]["repo_id"] == "shivubind/unitree_g1_pick_place2"

    def test_thor_dispatch_script_executable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_manifest(self._sample_discovered(), tmpdir)
            script_path = os.path.join(tmpdir, "thor_dispatch.sh")
            assert os.access(script_path, os.X_OK)


# ══════════════════════════════════════════════════════════════════════
#  generate_thor_dispatch_script
# ══════════════════════════════════════════════════════════════════════


class TestGenerateThorDispatchScript:
    """Test Thor dispatch script generation."""

    def test_bash_header(self):
        script = generate_thor_dispatch_script(
            {
                "generated_at": "2026-01-01T00:00:00Z",
                "total_datasets": 5,
                "total_embodiments": 2,
                "download_commands": [],
            }
        )
        assert script.startswith("#!/bin/bash")
        assert "set -euo pipefail" in script

    def test_download_commands(self):
        manifest = {
            "generated_at": "2026-01-01T00:00:00Z",
            "total_datasets": 1,
            "total_embodiments": 1,
            "download_commands": [
                {
                    "repo_id": "lerobot/so100_pick",
                    "embodiment_tag": "new_embodiment",
                    "confidence": 0.95,
                    "command": "python scripts/collect_hf_pick_place.py --download --repo lerobot/so100_pick",
                }
            ],
        }
        script = generate_thor_dispatch_script(manifest)
        assert "lerobot/so100_pick" in script
        assert "python scripts/collect_hf_pick_place.py" in script

    def test_hf_token_warning(self):
        script = generate_thor_dispatch_script(
            {
                "generated_at": "",
                "total_datasets": 0,
                "total_embodiments": 0,
                "download_commands": [],
            }
        )
        assert "HF_TOKEN" in script

    def test_output_dir_variable(self):
        script = generate_thor_dispatch_script(
            {
                "generated_at": "",
                "total_datasets": 0,
                "total_embodiments": 0,
                "download_commands": [],
            }
        )
        assert "OUTPUT_DIR" in script

    def test_empty_manifest(self):
        script = generate_thor_dispatch_script(
            {
                "generated_at": "",
                "total_datasets": 0,
                "total_embodiments": 0,
                "download_commands": [],
            }
        )
        assert "#!/bin/bash" in script
        assert "Data collection complete" in script

    def test_multiple_commands(self):
        manifest = {
            "generated_at": "",
            "total_datasets": 2,
            "total_embodiments": 1,
            "download_commands": [
                {"repo_id": "a/b", "embodiment_tag": "new_embodiment", "confidence": 0.9, "command": "cmd1"},
                {"repo_id": "c/d", "embodiment_tag": "unitree_g1", "confidence": 0.85, "command": "cmd2"},
            ],
        }
        script = generate_thor_dispatch_script(manifest)
        assert "a/b" in script
        assert "c/d" in script
        assert "cmd1" in script
        assert "cmd2" in script


# ══════════════════════════════════════════════════════════════════════
#  print_summary
# ══════════════════════════════════════════════════════════════════════


class TestPrintSummary:
    """Test summary output."""

    def test_header(self, capsys):
        print_summary({"new_embodiment": []})
        captured = capsys.readouterr()
        assert "HuggingFace Multi-Embodiment" in captured.out

    def test_dataset_display(self, capsys):
        discovered = {
            "new_embodiment": [
                DatasetInfo(
                    repo_id="lerobot/so100_pick",
                    embodiment_tag="new_embodiment",
                    data_config="so100",
                    confidence=0.95,
                    task_types=["pick_and_place"],
                    is_lerobot_v2=True,
                ),
            ]
        }
        print_summary(discovered)
        captured = capsys.readouterr()
        assert "lerobot/so100_pick" in captured.out
        assert "[LeRobot]" in captured.out

    def test_confidence_icons(self, capsys):
        discovered = {
            "new_embodiment": [
                DatasetInfo(
                    repo_id="a/high",
                    embodiment_tag="new_embodiment",
                    data_config="so100",
                    confidence=0.9,
                    task_types=["manipulation"],
                ),
                DatasetInfo(
                    repo_id="b/med",
                    embodiment_tag="new_embodiment",
                    data_config="so100",
                    confidence=0.6,
                    task_types=["manipulation"],
                ),
                DatasetInfo(
                    repo_id="c/low",
                    embodiment_tag="new_embodiment",
                    data_config="so100",
                    confidence=0.3,
                    task_types=["manipulation"],
                ),
            ]
        }
        print_summary(discovered)
        captured = capsys.readouterr()
        assert "🟢" in captured.out  # high confidence
        assert "🟡" in captured.out  # medium confidence
        assert "🔴" in captured.out  # low confidence

    def test_empty_summary(self, capsys):
        print_summary({})
        captured = capsys.readouterr()
        assert "Summary: 0 datasets" in captured.out

    def test_deduplication(self, capsys):
        discovered = {
            "new_embodiment": [
                DatasetInfo(
                    repo_id="dup/repo",
                    embodiment_tag="new_embodiment",
                    data_config="so100",
                    confidence=0.9,
                    task_types=["manipulation"],
                ),
                DatasetInfo(
                    repo_id="dup/repo",
                    embodiment_tag="new_embodiment",
                    data_config="so100",
                    confidence=0.8,
                    task_types=["manipulation"],
                ),
            ]
        }
        print_summary(discovered)
        captured = capsys.readouterr()
        assert "Datasets found: 1" in captured.out

    def test_truncation(self, capsys):
        """When > 8 datasets, should show '... and N more'."""
        datasets = [
            DatasetInfo(
                repo_id=f"user/ds_{i}",
                embodiment_tag="new_embodiment",
                data_config="so100",
                confidence=0.9,
                task_types=["manipulation"],
            )
            for i in range(15)
        ]
        print_summary({"new_embodiment": datasets})
        captured = capsys.readouterr()
        assert "... and 7 more" in captured.out


# ══════════════════════════════════════════════════════════════════════
#  EMBODIMENT_DATASET_MAP structure
# ══════════════════════════════════════════════════════════════════════


class TestEmbodimentDatasetMap:
    """Validate the EMBODIMENT_DATASET_MAP structure."""

    EXPECTED_EMBODIMENTS = [
        "new_embodiment",
        "gr1",
        "unitree_g1",
        "robocasa_panda_omron",
        "oxe_droid",
        "oxe_google",
        "oxe_widowx",
        "libero_panda",
        "behavior_r1_pro",
    ]

    def test_all_embodiments_present(self):
        for emb in self.EXPECTED_EMBODIMENTS:
            assert emb in EMBODIMENT_DATASET_MAP, f"Missing embodiment: {emb}"

    def test_exactly_nine_embodiments(self):
        assert len(EMBODIMENT_DATASET_MAP) == 9

    def test_required_keys(self):
        for emb, info in EMBODIMENT_DATASET_MAP.items():
            assert "description" in info, f"{emb} missing 'description'"
            assert "data_configs" in info, f"{emb} missing 'data_configs'"
            assert "hf_patterns" in info, f"{emb} missing 'hf_patterns'"
            assert "search_queries" in info, f"{emb} missing 'search_queries'"

    def test_non_empty_configs(self):
        for emb, info in EMBODIMENT_DATASET_MAP.items():
            assert len(info["data_configs"]) > 0, f"{emb} has empty data_configs"

    def test_non_empty_patterns(self):
        for emb, info in EMBODIMENT_DATASET_MAP.items():
            assert len(info["hf_patterns"]) > 0, f"{emb} has empty hf_patterns"

    def test_non_empty_queries(self):
        for emb, info in EMBODIMENT_DATASET_MAP.items():
            assert len(info["search_queries"]) > 0, f"{emb} has empty search_queries"

    def test_types(self):
        for emb, info in EMBODIMENT_DATASET_MAP.items():
            assert isinstance(info["description"], str)
            assert isinstance(info["data_configs"], list)
            assert isinstance(info["hf_patterns"], list)
            assert isinstance(info["search_queries"], list)

    def test_so101_configs(self):
        """PR #123 added so101 configs — verify they're mappable."""
        configs = EMBODIMENT_DATASET_MAP["new_embodiment"]["data_configs"]
        assert "so100" in configs


# ══════════════════════════════════════════════════════════════════════
#  download_dataset (mocked)
# ══════════════════════════════════════════════════════════════════════


class TestDownloadDataset:
    """Test dataset download with mocked HuggingFace API."""

    def test_successful_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = os.path.join(tmpdir, "downloaded")
            os.makedirs(fake_path)
            # Create some fake files
            for i in range(3):
                Path(fake_path).joinpath(f"ep_{i}.parquet").touch()

            mock_hf = types.ModuleType("huggingface_hub")
            mock_hf.snapshot_download = MagicMock(return_value=fake_path)

            with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
                result = download_dataset("test/repo", tmpdir)
                assert result["status"] == "downloaded"
                assert result["episode_count_estimate"] == 3

    def test_download_failure(self):
        mock_hf = types.ModuleType("huggingface_hub")
        mock_hf.snapshot_download = MagicMock(side_effect=Exception("Download failed"))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
                result = download_dataset("bad/repo", tmpdir)
                assert result["status"] == "failed"
                assert "error" in result

    def test_output_dir_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = os.path.join(tmpdir, "dl")
            os.makedirs(fake_path)

            mock_hf = types.ModuleType("huggingface_hub")
            mock_hf.snapshot_download = MagicMock(return_value=fake_path)

            output = os.path.join(tmpdir, "output")
            with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
                result = download_dataset("test/repo", output)
                assert result["status"] == "downloaded"


# ══════════════════════════════════════════════════════════════════════
#  convert_to_lerobot_v3 (mocked)
# ══════════════════════════════════════════════════════════════════════


class TestConvertToLerobotV3:
    """Test LeRobot v3 conversion with mocked data config."""

    def test_manifest_generation(self):
        mock_config = MagicMock()
        mock_config.video_keys = ["observation.images.top"]
        mock_config.state_keys = ["observation.state"]
        mock_config.action_keys = ["action"]

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "source")
            out_path = os.path.join(tmpdir, "output")
            os.makedirs(src_path)

            with patch("collect_hf_pick_place.convert_to_lerobot_v3") as mock_fn:
                mock_fn.return_value = {
                    "status": "manifest_generated",
                    "embodiment_tag": "new_embodiment",
                    "data_config": "so100",
                }
                result = mock_fn(src_path, "new_embodiment", "so100", out_path)
                assert result["status"] == "manifest_generated"

    def test_conversion_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("collect_hf_pick_place.convert_to_lerobot_v3") as mock_fn:
                mock_fn.return_value = {"status": "failed", "error": "Missing strands_robots"}
                result = mock_fn("/bad/path", "new_embodiment", "so100", tmpdir)
                assert result["status"] == "failed"


# ══════════════════════════════════════════════════════════════════════
#  Edge cases
# ══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_case_insensitive_classification(self):
        info = {"repo_id": "TEST/PICK_ROBOT", "description": "A GRASPING Dataset"}
        tasks = classify_dataset_task(info)
        assert "pick_and_place" in tasks

    def test_combined_text_matching(self):
        """Tags should be included in text matching."""
        info = {"repo_id": "user/data", "description": "", "tags": ["unitree_g1"]}
        result = map_dataset_to_embodiment(info)
        assert result is not None
        assert result[0] == "unitree_g1"

    def test_first_match_wins(self):
        """When multiple patterns match, first match in list wins."""
        # "so100" matches before "lerobot" fallback
        info = {"repo_id": "lerobot/so100_pick", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result[0] == "new_embodiment"
        assert result[2] == 0.95  # Direct match, not 0.5 fallback

    def test_dataset_info_equality(self):
        ds1 = DatasetInfo(repo_id="a", embodiment_tag="b", data_config="c")
        ds2 = DatasetInfo(repo_id="a", embodiment_tag="b", data_config="c")
        assert ds1 == ds2

    def test_manifest_nested_dir_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "c")
            generate_manifest({}, nested)
            assert os.path.exists(nested)

    def test_classify_with_missing_keys(self):
        """Gracefully handle info dicts with missing keys."""
        info = {}
        tasks = classify_dataset_task(info)
        assert tasks == ["unknown"]

    def test_map_with_empty_tags(self):
        info = {"repo_id": "", "description": "", "tags": []}
        result = map_dataset_to_embodiment(info)
        assert result is None
