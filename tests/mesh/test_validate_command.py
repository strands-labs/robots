"""Tests for :mod:`strands_robots.mesh.security` payload validation.

After the Zenoh-native refactor, ``security.py`` only owns payload
semantics -- wire authentication is the transport's job. The tests below
exercise the action allowlist, per-action bounds (instruction length,
duration, step counts), the ``policy_host`` / ``model_path`` /
``policy_type`` allowlists, and the composite ``server_address`` check.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh import security as sec

# --- action allowlist --------------------------------------------------


class TestActionAllowlist:
    def test_default_action_is_status(self):
        out = sec.validate_command({})
        assert out["action"] == "status"

    def test_status_passes(self):
        assert sec.validate_command({"action": "status"})["action"] == "status"

    def test_resume_passes(self):
        assert sec.validate_command({"action": "resume"})["action"] == "resume"

    def test_unknown_action_rejected(self):
        with pytest.raises(sec.ValidationError, match="unknown action"):
            sec.validate_command({"action": "rm_minus_rf"})

    def test_non_string_action_rejected(self):
        with pytest.raises(sec.ValidationError, match="action must be a string"):
            sec.validate_command({"action": 42})

    def test_non_dict_command_rejected(self):
        with pytest.raises(sec.ValidationError, match="command must be a dict"):
            sec.validate_command("status")  # type: ignore[arg-type]


# --- execute / start: instruction bounds + policy_host -----------------


class TestExecuteStart:
    def test_execute_requires_instruction(self):
        with pytest.raises(sec.ValidationError, match="non-empty `instruction`"):
            sec.validate_command({"action": "execute"})

    def test_execute_empty_instruction_rejected(self):
        with pytest.raises(sec.ValidationError, match="non-empty"):
            sec.validate_command({"action": "execute", "instruction": "   "})

    def test_execute_oversize_instruction_rejected(self):
        with pytest.raises(sec.ValidationError, match="exceeds"):
            sec.validate_command(
                {
                    "action": "execute",
                    "instruction": "x" * (sec.MAX_INSTRUCTION_LEN + 1),
                }
            )

    def test_execute_default_policy_host_is_localhost(self):
        out = sec.validate_command({"action": "execute", "instruction": "go", "policy_provider": "mock"})
        assert out["policy_host"] == "localhost"
        assert out["duration"] == 30.0
        assert out["policy_provider"] == "mock"

    def test_execute_unknown_policy_host_rejected(self):
        with pytest.raises(sec.ValidationError, match="policy_host"):
            sec.validate_command(
                {
                    "action": "execute",
                    "instruction": "go",
                    "policy_host": "evil.example.com",
                }
            )

    def test_execute_duration_negative_rejected(self):
        with pytest.raises(sec.ValidationError, match="duration"):
            sec.validate_command({"action": "execute", "instruction": "go", "duration": -1.0})

    def test_execute_duration_oversize_rejected(self):
        with pytest.raises(sec.ValidationError, match="duration"):
            sec.validate_command(
                {
                    "action": "execute",
                    "instruction": "go",
                    "duration": sec.MAX_DURATION_S + 1.0,
                }
            )

    def test_execute_policy_port_bounds(self):
        out = sec.validate_command(
            {"action": "execute", "instruction": "go", "policy_provider": "mock", "policy_port": 8080}
        )
        assert out["policy_port"] == 8080

        with pytest.raises(sec.ValidationError, match="policy_port"):
            sec.validate_command(
                {"action": "execute", "instruction": "go", "policy_provider": "mock", "policy_port": 70000}
            )


# --- HF repo / local model path / policy_type --------------------------


class TestModelPathGating:
    def test_pretrained_name_default_allowed(self):
        out = sec.validate_command(
            {
                "action": "execute",
                "instruction": "go",
                "policy_provider": "mock",
                "pretrained_name_or_path": "nvidia/some-model",
            }
        )
        assert out["pretrained_name_or_path"] == "nvidia/some-model"

    def test_pretrained_name_unknown_org_rejected(self):
        with pytest.raises(sec.ValidationError, match="pretrained_name"):
            sec.validate_command(
                {
                    "action": "execute",
                    "instruction": "go",
                    "pretrained_name_or_path": "evil-corp/backdoor",
                }
            )

    def test_pretrained_name_traversal_rejected(self):
        with pytest.raises(sec.ValidationError):
            sec.validate_command(
                {
                    "action": "execute",
                    "instruction": "go",
                    "pretrained_name_or_path": "nvidia/../../etc/passwd",
                }
            )

    def test_model_path_local_traversal_rejected(self):
        with pytest.raises(sec.ValidationError, match="model_path"):
            sec.validate_command(
                {
                    "action": "execute",
                    "instruction": "go",
                    "model_path": "/tmp/../etc/shadow",
                }
            )

    def test_model_path_shell_meta_rejected(self):
        with pytest.raises(sec.ValidationError):
            sec.validate_command(
                {
                    "action": "execute",
                    "instruction": "go",
                    "model_path": "/tmp/foo;rm -rf /",
                }
            )

    def test_policy_type_unknown_rejected(self):
        with pytest.raises(sec.ValidationError, match="policy_type"):
            sec.validate_command(
                {
                    "action": "execute",
                    "instruction": "go",
                    "policy_type": "evil_type",
                }
            )

    def test_policy_type_extends_via_env(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_POLICY_TYPE_ALLOW", "my_custom_type")
        out = sec.validate_command(
            {
                "action": "execute",
                "instruction": "go",
                "policy_provider": "mock",
                "policy_type": "my_custom_type",
            }
        )
        assert out["policy_type"] == "my_custom_type"

    def test_policy_provider_default_is_mock(self):
        # Security hardening: policy_provider is now REQUIRED on execute/start.
        # The old silent default of 'mock' was a security boundary footgun
        # (a peer that forgot the field was indistinguishable from one that
        # legitimately wanted mock). Renamed for clarity? Keep the test name
        # for git-history continuity; the assertion changes.
        with pytest.raises(sec.ValidationError, match="policy_provider is required"):
            sec.validate_command({"action": "execute", "instruction": "go"})

    def test_policy_provider_unknown_rejected(self):
        with pytest.raises(sec.ValidationError, match="policy_provider"):
            sec.validate_command(
                {
                    "action": "execute",
                    "instruction": "go",
                    "policy_provider": "evil_provider",
                }
            )

    def test_server_address_loopback_allowed(self):
        out = sec.validate_command(
            {
                "action": "execute",
                "instruction": "go",
                "policy_provider": "mock",
                "server_address": "tcp://localhost:5555",
            }
        )
        assert out["server_address"] == "tcp://localhost:5555"

    def test_server_address_external_rejected(self):
        with pytest.raises(sec.ValidationError, match="server_address"):
            sec.validate_command(
                {
                    "action": "execute",
                    "instruction": "go",
                    "policy_provider": "mock",
                    "server_address": "tcp://evil.example.com:5555",
                }
            )

    def test_policy_host_extends_via_env(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_POLICY_HOST_ALLOW", "10.0.0.0/24")
        out = sec.validate_command(
            {
                "action": "execute",
                "instruction": "go",
                "policy_provider": "mock",
                "policy_host": "10.0.0.42",
            }
        )
        assert out["policy_host"] == "10.0.0.42"


# --- step / teleop_receive ---------------------------------------------


class TestStepAndTeleop:
    def test_step_default_is_one(self):
        out = sec.validate_command({"action": "step"})
        assert out["steps"] == 1

    def test_step_bounds(self):
        out = sec.validate_command({"action": "step", "steps": 100})
        assert out["steps"] == 100

        with pytest.raises(sec.ValidationError, match="steps"):
            sec.validate_command({"action": "step", "steps": 0})

        with pytest.raises(sec.ValidationError, match="steps"):
            sec.validate_command({"action": "step", "steps": 100_000})

    def test_step_non_numeric_rejected(self):
        with pytest.raises(sec.ValidationError):
            sec.validate_command({"action": "step", "steps": "five"})

    def test_teleop_receive_requires_source(self):
        with pytest.raises(sec.ValidationError, match="source_peer_id"):
            sec.validate_command({"action": "teleop_receive"})

    def test_teleop_receive_with_source_passes(self):
        out = sec.validate_command({"action": "teleop_receive", "source_peer_id": "operator-7"})
        assert out["source_peer_id"] == "operator-7"


# --- safe-host / safe-model helpers ------------------------------------


class TestSafeHostAndModel:
    def test_safe_policy_host_loopback(self):
        assert sec.is_safe_policy_host("localhost") is True
        assert sec.is_safe_policy_host("127.0.0.1") is True
        assert sec.is_safe_policy_host("::1") is True

    def test_safe_policy_host_rejects_external(self):
        assert sec.is_safe_policy_host("8.8.8.8") is False
        assert sec.is_safe_policy_host("evil.example.com") is False

    def test_safe_policy_host_rejects_invalid(self):
        assert sec.is_safe_policy_host("") is False
        assert sec.is_safe_policy_host(42) is False  # type: ignore[arg-type]
        assert sec.is_safe_policy_host(None) is False  # type: ignore[arg-type]

    def test_safe_policy_host_cidr_extension(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_POLICY_HOST_ALLOW", "192.168.0.0/16")
        assert sec.is_safe_policy_host("192.168.42.7") is True
        assert sec.is_safe_policy_host("10.0.0.7") is False

    def test_safe_model_path_basic(self):
        assert sec.is_safe_model_path("nvidia/some-model") is True
        assert sec.is_safe_model_path("/local/path/to/model") is True

    def test_safe_model_path_rejects_traversal(self):
        assert sec.is_safe_model_path("../etc/passwd") is False
        assert sec.is_safe_model_path("a/b/../c") is False
        assert sec.is_safe_model_path("/tmp/../etc/shadow") is False

    def test_safe_model_path_rejects_shell_meta(self):
        for bad in [
            "model;rm -rf /",
            "model | nc evil",
            "model$(whoami)",
            "model\nfoo",
            "model\x00",
        ]:
            assert sec.is_safe_model_path(bad) is False, bad

    def test_safe_model_path_hf_only_org_gate(self):
        assert sec.is_safe_model_path("nvidia/some-model", hf_only=True) is True
        assert sec.is_safe_model_path("evil-corp/backdoor", hf_only=True) is False

    def test_safe_model_path_hf_only_rejects_local(self):
        assert sec.is_safe_model_path("/local/path", hf_only=True) is False

    def test_safe_policy_type_builtins(self):
        for pt in ("mock", "groot", "lerobot", "act", "diffusion"):
            assert sec.is_safe_policy_type(pt) is True

    def test_safe_policy_type_rejects_unknown(self):
        assert sec.is_safe_policy_type("evil_type") is False

    def test_safe_server_address_strips_scheme_and_port(self):
        assert sec.is_safe_server_address("tcp://localhost:5555") is True
        assert sec.is_safe_server_address("zmq://127.0.0.1:9000") is True
        assert sec.is_safe_server_address("tcp://evil.example.com:5555") is False

    def test_safe_server_address_accepts_bracketed_ipv6_loopback(self):
        assert sec.is_safe_server_address("[::1]") is True
        assert sec.is_safe_server_address("[::1]:8080") is True

    def test_safe_server_address_rejects_unallowed_ipv6(self):
        assert sec.is_safe_server_address("[2001:db8::1]") is False


class TestValidateCommandResume:
    """Pin the prior fix review fix: validate_command must bound resume.override_code.

    Before the prior fix, ``validate_command`` had no ``resume`` clause -- a peer
    sending ``{"action": "resume", "override_code": <non-string>}`` would
    pass validation and reach ``Mesh._resume_lockout`` where ``.strip()``
    raises ``AttributeError`` on a list/dict, surfacing as a generic
    dispatch error rather than a clean ValidationError.
    """

    def test_resume_string_override_passes(self):
        cmd = {"action": "resume", "override_code": "valid-secret"}
        out = sec.validate_command(cmd)
        assert out["override_code"] == "valid-secret"

    def test_resume_empty_override_passes(self):
        # An empty string is the sentinel for "no override supplied" --
        # validation must let it through; ``_resume_lockout`` then rejects.
        cmd = {"action": "resume", "override_code": ""}
        out = sec.validate_command(cmd)
        assert out["override_code"] == ""

    def test_resume_missing_override_passes_with_default(self):
        # Missing key is also the no-code sentinel.
        cmd = {"action": "resume"}
        out = sec.validate_command(cmd)
        assert out["override_code"] == ""

    def test_resume_list_override_rejected(self):
        cmd = {"action": "resume", "override_code": ["x"]}
        with pytest.raises(sec.ValidationError, match="must be a string"):
            sec.validate_command(cmd)

    def test_resume_dict_override_rejected(self):
        cmd = {"action": "resume", "override_code": {"k": "v"}}
        with pytest.raises(sec.ValidationError, match="must be a string"):
            sec.validate_command(cmd)

    def test_resume_int_override_rejected(self):
        cmd = {"action": "resume", "override_code": 12345}
        with pytest.raises(sec.ValidationError, match="must be a string"):
            sec.validate_command(cmd)

    def test_resume_oversized_override_rejected(self):
        cmd = {"action": "resume", "override_code": "x" * 257}
        with pytest.raises(sec.ValidationError, match="too long"):
            sec.validate_command(cmd)

    def test_resume_at_limit_override_passes(self):
        cmd = {"action": "resume", "override_code": "x" * 256}
        out = sec.validate_command(cmd)
        assert out["override_code"] == "x" * 256


# === validate_command strips unknown top-level keys ===


class TestValidateCommandKeyAllowlist:
    """Defence-in-depth: the validator returns only validated fields.
    the prior implementation the validator did `out = dict(cmd)` and preserved every
    unknown key. Today's _dispatch only consumes a known whitelist so
    the gap was not exploitable, but a future handler that did `**cmd`
    would silently pick up attacker-controlled values.
    """

    def test_unknown_keys_are_stripped(self):
        out = sec.validate_command(
            {
                "action": "execute",
                "instruction": "do thing",
                "policy_provider": "mock",
                # Attacker-controlled extras:
                "trust_remote_code": True,
                "policy_url": "http://evil.example.com/payload.bin",
                "extra_kwargs": {"shell": "rm -rf /"},
            }
        )
        for forbidden in ("trust_remote_code", "policy_url", "extra_kwargs"):
            assert forbidden not in out, f"{forbidden!r} leaked through validator"

    def test_validated_fields_pass_through(self):
        out = sec.validate_command(
            {
                "action": "execute",
                "instruction": "go pick up the block",
                "policy_provider": "mock",
                "policy_host": "127.0.0.1",
                "duration": 30,
            }
        )
        assert out["action"] == "execute"
        assert out["instruction"] == "go pick up the block"
        assert out["policy_provider"] == "mock"
        assert out["policy_host"] == "127.0.0.1"
        assert out["duration"] == 30.0

    def test_turn_id_and_sender_id_pass_through(self):
        # Wire-routing fields used by _on_cmd / _on_response correlation
        out = sec.validate_command(
            {
                "action": "status",
                "turn_id": "abc-123",
                "sender_id": "operator-1",
            }
        )
        assert out["turn_id"] == "abc-123"
        assert out["sender_id"] == "operator-1"
