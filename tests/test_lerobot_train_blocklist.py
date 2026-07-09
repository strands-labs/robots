"""Regression tests for lerobot_train extra_flags security blocklist + HIL gate."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from strands_robots.tools.lerobot_train import (
    _approve_response,
    _gate_extra_flags,
    _normalize_hydra_key,
    _validate_extra_flags,
)


class TestValidateExtraFlags:
    """Pin the blocklist contract: dangerous flags detected, benign flags pass."""

    @pytest.mark.parametrize(
        "key",
        [
            "output_dir",
            "--output_dir",
            "+output_dir",
            "~output_dir",
            "++output_dir",
        ],
    )
    def test_output_dir_all_hydra_forms_blocked(self, key):
        blocked = _validate_extra_flags({key: "/tmp/evil"})
        assert len(blocked) == 1
        assert blocked[0][1] == "output_dir"

    @pytest.mark.parametrize(
        "key",
        [
            "config_path",
            "--config_path",
            "+config_path",
        ],
    )
    def test_config_path_blocked(self, key):
        blocked = _validate_extra_flags({key: "/tmp/malicious.yaml"})
        assert len(blocked) == 1

    @pytest.mark.parametrize(
        "key",
        [
            "wandb.enable",
            "--wandb.enable",
            "+wandb.enable",
            "wandb.project",
            "wandb.entity",
            "wandb.api_key",
        ],
    )
    def test_wandb_flags_blocked(self, key):
        blocked = _validate_extra_flags({key: "true"})
        assert len(blocked) == 1

    @pytest.mark.parametrize(
        "key",
        [
            "dataset.root",
            "--dataset.root",
            "policy.pretrained_path",
            "--policy.pretrained_path",
        ],
    )
    def test_data_and_model_paths_blocked(self, key):
        blocked = _validate_extra_flags({key: "/etc/shadow"})
        assert len(blocked) == 1

    @pytest.mark.parametrize(
        "key",
        [
            "push_to_hub",
            "policy.push_to_hub",
            "hub_repo_id",
        ],
    )
    def test_hub_push_flags_blocked(self, key):
        blocked = _validate_extra_flags({key: "attacker/repo"})
        assert len(blocked) == 1

    def test_benign_flags_pass(self):
        assert _validate_extra_flags({"lr": "1e-4"}) == []
        assert _validate_extra_flags({"--batch_size": "32"}) == []
        assert _validate_extra_flags({"training.num_workers": "4"}) == []

    def test_multiple_flags_all_blocked_returned(self):
        blocked = _validate_extra_flags({"lr": "1e-4", "output_dir": "/tmp", "wandb.enable": "true"})
        assert len(blocked) == 2
        norms = {b[1] for b in blocked}
        assert norms == {"output_dir", "wandb.enable"}

    def test_empty_dict_passes(self):
        assert _validate_extra_flags({}) == []


class TestNormalizeHydraKey:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("output_dir", "output_dir"),
            ("--output_dir", "output_dir"),
            ("+output_dir", "output_dir"),
            ("~output_dir", "output_dir"),
            ("++output_dir", "output_dir"),
        ],
    )
    def test_strips_prefixes(self, raw, expected):
        assert _normalize_hydra_key(raw) == expected


class TestGateExtraFlags:
    """Pin the HIL gate contract: allowlist, bypass, interrupt, decline."""

    def test_benign_flags_pass(self):
        assert _gate_extra_flags({"lr": "1e-4"}, None) is None

    def test_blocked_flag_no_context_returns_error(self):
        result = _gate_extra_flags({"output_dir": "/tmp"}, None)
        assert result is not None
        assert result["status"] == "error"
        assert "approval" in result["content"][0]["text"].lower()

    def test_allowlist_skips_gate(self, monkeypatch):
        monkeypatch.setenv("STRANDS_TRAIN_EXTRA_FLAGS_ALLOW", "output_dir")
        assert _gate_extra_flags({"output_dir": "/tmp"}, None) is None

    def test_allowlist_partial(self, monkeypatch):
        """Allowlist covers one flag but not the other."""
        monkeypatch.setenv("STRANDS_TRAIN_EXTRA_FLAGS_ALLOW", "output_dir")
        result = _gate_extra_flags({"output_dir": "/tmp", "wandb.enable": "true"}, None)
        assert result is not None
        assert result["status"] == "error"

    def test_bypass_consent_allows(self, monkeypatch):
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")
        assert _gate_extra_flags({"output_dir": "/tmp"}, None) is None

    def test_interrupt_approved(self):
        ctx = MagicMock()
        ctx.interrupt.return_value = "y"
        assert _gate_extra_flags({"output_dir": "/tmp"}, ctx) is None
        ctx.interrupt.assert_called_once()
        reason = ctx.interrupt.call_args[1]["reason"]
        assert reason["action"] == "train"
        assert "output_dir" in str(reason["blocked_flags"])

    def test_interrupt_declined(self):
        ctx = MagicMock()
        ctx.interrupt.return_value = "no"
        result = _gate_extra_flags({"output_dir": "/tmp"}, ctx)
        assert result is not None
        assert result["status"] == "error"
        assert "declined" in result["content"][0]["text"]

    def test_interrupt_runtime_error_fails_closed(self):
        ctx = MagicMock()
        ctx.interrupt.side_effect = RuntimeError("no agent loop")
        result = _gate_extra_flags({"output_dir": "/tmp"}, ctx)
        assert result is not None
        assert result["status"] == "error"

    @pytest.mark.parametrize("response", ["y", "Y", "yes", "YES", "approve", "Approved"])
    def test_approve_response_affirmative(self, response):
        assert _approve_response(response) is True

    @pytest.mark.parametrize("response", ["n", "no", "nope", "", 42, None])
    def test_approve_response_negative(self, response):
        assert _approve_response(response) is False
