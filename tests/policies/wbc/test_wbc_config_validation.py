"""Fail-fast validation and file-loading contracts for :class:`WBCConfig`.

WBC drives a walking humanoid, so a config paired with the wrong checkpoint (a
bad dimension, a truncated per-joint vector) would destabilise the robot at
runtime. :meth:`WBCConfig.__post_init__` therefore rejects impossible
dimensions at construction rather than warn-and-continue, and
:meth:`WBCConfig.from_file` surfaces malformed/unsupported files as a loud
``ValueError`` instead of returning a half-built config. These tests pin those
contracts through the public surface (the constructor and ``from_file``) so a
regression that silently accepts a broken config is caught.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strands_robots.policies.wbc import WBCConfig


class TestDimensionFailFast:
    """__post_init__ rejects sub-minimal dimensions with an actionable message."""

    def test_num_actions_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"num_actions must be >= 1"):
            WBCConfig(policy_path="p.onnx", num_actions=0)

    def test_obs_history_len_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"obs_history_len must be >= 1"):
            WBCConfig(policy_path="p.onnx", obs_history_len=0)

    def test_single_obs_dim_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"single_obs_dim must be >= 1"):
            WBCConfig(policy_path="p.onnx", single_obs_dim=0)


class TestFromFileErrorPaths:
    """from_file distinguishes not-found, malformed, and unsupported inputs."""

    def test_malformed_json_raises_value_error_naming_the_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "wbc.json"
        bad.write_text("{not valid json")
        with pytest.raises(ValueError, match=r"is not valid JSON"):
            WBCConfig.from_file(bad)

    def test_unsupported_extension_is_rejected(self, tmp_path: Path) -> None:
        cfg = tmp_path / "wbc.txt"
        cfg.write_text("policy_path: p.onnx")
        with pytest.raises(ValueError, match=r"unsupported extension '\.txt'"):
            WBCConfig.from_file(cfg)

    def test_non_mapping_document_is_rejected(self, tmp_path: Path) -> None:
        # A JSON list parses fine but is not a config mapping; reject it loudly
        # rather than crash later on attribute access.
        cfg = tmp_path / "wbc.json"
        cfg.write_text(json.dumps(["not", "a", "mapping"]))
        with pytest.raises(ValueError, match=r"must contain a mapping"):
            WBCConfig.from_file(cfg)

    def test_valid_json_round_trips_through_from_file(self, tmp_path: Path) -> None:
        cfg = tmp_path / "wbc.json"
        cfg.write_text(json.dumps({"policy_path": "g1.onnx", "num_actions": 15}))
        loaded = WBCConfig.from_file(cfg)
        assert loaded.policy_path == "g1.onnx"
        assert loaded.num_actions == 15
