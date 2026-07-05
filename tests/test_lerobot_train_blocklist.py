"""Regression tests for lerobot_train extra_flags security blocklist."""

from __future__ import annotations

import pytest

from strands_robots.tools.lerobot_train import _validate_extra_flags


class TestValidateExtraFlags:
    """Pin the blocklist contract: dangerous flags raise, benign flags pass."""

    @pytest.mark.parametrize("key", [
        "output_dir",
        "--output_dir",
        "+output_dir",
        "~output_dir",
        "++output_dir",
    ])
    def test_output_dir_all_hydra_forms_blocked(self, key):
        err = _validate_extra_flags({key: "/tmp/evil"})
        assert err is not None
        assert "blocked" in err

    @pytest.mark.parametrize("key", [
        "config_path",
        "--config_path",
        "+config_path",
    ])
    def test_config_path_blocked(self, key):
        err = _validate_extra_flags({key: "/tmp/malicious.yaml"})
        assert err is not None

    @pytest.mark.parametrize("key", [
        "wandb.enable",
        "--wandb.enable",
        "+wandb.enable",
        "wandb.project",
        "wandb.entity",
        "wandb.api_key",
    ])
    def test_wandb_flags_blocked(self, key):
        err = _validate_extra_flags({key: "true"})
        assert err is not None

    @pytest.mark.parametrize("key", [
        "dataset.root",
        "--dataset.root",
        "policy.pretrained_path",
        "--policy.pretrained_path",
    ])
    def test_data_and_model_paths_blocked(self, key):
        err = _validate_extra_flags({key: "/etc/shadow"})
        assert err is not None

    @pytest.mark.parametrize("key", [
        "push_to_hub",
        "policy.push_to_hub",
        "hub_repo_id",
    ])
    def test_hub_push_flags_blocked(self, key):
        err = _validate_extra_flags({key: "attacker/repo"})
        assert err is not None

    def test_benign_flags_pass(self):
        assert _validate_extra_flags({"lr": "1e-4"}) is None
        assert _validate_extra_flags({"--batch_size": "32"}) is None
        assert _validate_extra_flags({"training.num_workers": "4"}) is None

    def test_multiple_flags_first_blocked_wins(self):
        err = _validate_extra_flags({"lr": "1e-4", "output_dir": "/tmp"})
        assert err is not None
        assert "output_dir" in err

    def test_env_override_blocklist(self, monkeypatch):
        monkeypatch.setenv("STRANDS_TRAIN_EXTRA_FLAGS_BLOCKLIST", "custom_flag,another")
        err = _validate_extra_flags({"custom_flag": "val"})
        assert err is not None
        # Default blocklist item should now pass
        assert _validate_extra_flags({"output_dir": "/tmp"}) is None

    def test_empty_dict_passes(self):
        assert _validate_extra_flags({}) is None
