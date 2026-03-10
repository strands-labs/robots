"""Tests for strands_robots.policy_resolver — the smart string→(provider, kwargs) engine.

Validates all resolution paths:
  1. Server addresses (ws://, wss://, host:port, grpc://, zmq://)
  2. Shorthand names (mock, groot, dreamzero, ...)
  3. HuggingFace model IDs (org/model → provider routing)
  4. Model ID overrides (nvidia/* disambiguation)
  5. Bare provider names
  6. Fallback behavior
  7. Mapping table consistency (structural regression guards)
"""

import pytest

from strands_robots.policy_resolver import (
    _HF_ORG_TO_PROVIDER,
    _MODEL_ID_OVERRIDES,
    _SHORTHAND_TO_PROVIDER,
    resolve_policy,
)

# ═══════════════════════════════════════════════════════════════════
# 1. WebSocket resolution → dreamzero
# ═══════════════════════════════════════════════════════════════════


class TestWebSocketResolution:
    """ws:// and wss:// always route to dreamzero."""

    def test_ws_with_host_and_port(self):
        provider, kwargs = resolve_policy("ws://gpu-server:9000")
        assert provider == "dreamzero"
        assert kwargs["host"] == "gpu-server"
        assert kwargs["port"] == 9000

    def test_wss_with_host_and_port(self):
        provider, kwargs = resolve_policy("wss://secure.host:4433")
        assert provider == "dreamzero"
        assert kwargs["host"] == "secure.host"
        assert kwargs["port"] == 4433

    def test_ws_without_port_defaults_to_8000(self):
        provider, kwargs = resolve_policy("ws://gpu-server")
        assert provider == "dreamzero"
        assert kwargs["host"] == "gpu-server"
        assert kwargs["port"] == 8000

    def test_wss_without_port_defaults_to_8000(self):
        provider, kwargs = resolve_policy("wss://gpu-server")
        assert provider == "dreamzero"
        assert kwargs["host"] == "gpu-server"
        assert kwargs["port"] == 8000

    def test_ws_with_ip_address(self):
        provider, kwargs = resolve_policy("ws://192.168.1.100:5555")
        assert provider == "dreamzero"
        assert kwargs["host"] == "192.168.1.100"
        assert kwargs["port"] == 5555

    def test_ws_localhost(self):
        provider, kwargs = resolve_policy("ws://localhost:9000")
        assert provider == "dreamzero"
        assert kwargs["host"] == "localhost"
        assert kwargs["port"] == 9000

    def test_ws_extra_kwargs_merged(self):
        provider, kwargs = resolve_policy("ws://gpu:9000", timeout=30)
        assert provider == "dreamzero"
        assert kwargs["host"] == "gpu"
        assert kwargs["port"] == 9000
        assert kwargs["timeout"] == 30


# ═══════════════════════════════════════════════════════════════════
# 2. Host:port resolution → lerobot_async (gRPC)
# ═══════════════════════════════════════════════════════════════════


class TestHostPortResolution:
    """host:port (without protocol) routes to lerobot_async gRPC."""

    def test_localhost_port(self):
        provider, kwargs = resolve_policy("localhost:8080")
        assert provider == "lerobot_async"
        assert kwargs["server_address"] == "localhost:8080"

    def test_ip_port(self):
        provider, kwargs = resolve_policy("192.168.1.50:5000")
        assert provider == "lerobot_async"
        assert kwargs["server_address"] == "192.168.1.50:5000"

    def test_hostname_port(self):
        provider, kwargs = resolve_policy("my-gpu-server:50051")
        assert provider == "lerobot_async"
        assert kwargs["server_address"] == "my-gpu-server:50051"

    def test_extra_kwargs_merged(self):
        provider, kwargs = resolve_policy("host:8080", batch_size=4)
        assert provider == "lerobot_async"
        assert kwargs["batch_size"] == 4


# ═══════════════════════════════════════════════════════════════════
# 3. gRPC prefix resolution → lerobot_async
# ═══════════════════════════════════════════════════════════════════


class TestGrpcResolution:
    """grpc:// prefix routes to lerobot_async with address stripped."""

    def test_grpc_basic(self):
        provider, kwargs = resolve_policy("grpc://localhost:50051")
        assert provider == "lerobot_async"
        assert kwargs["server_address"] == "localhost:50051"

    def test_grpc_hostname(self):
        provider, kwargs = resolve_policy("grpc://jetson:8080")
        assert provider == "lerobot_async"
        assert kwargs["server_address"] == "jetson:8080"

    def test_grpc_strips_prefix(self):
        provider, kwargs = resolve_policy("grpc://192.168.1.1:5000")
        assert "grpc://" not in kwargs["server_address"]


# ═══════════════════════════════════════════════════════════════════
# 4. ZMQ prefix resolution → groot
# ═══════════════════════════════════════════════════════════════════


class TestZmqResolution:
    """zmq:// prefix routes to groot with host/port parsed."""

    def test_zmq_with_host_port(self):
        provider, kwargs = resolve_policy("zmq://jetson:5555")
        assert provider == "groot"
        assert kwargs["host"] == "jetson"
        assert kwargs["port"] == 5555

    def test_zmq_with_ip(self):
        provider, kwargs = resolve_policy("zmq://10.0.0.5:5556")
        assert provider == "groot"
        assert kwargs["host"] == "10.0.0.5"
        assert kwargs["port"] == 5556


# ═══════════════════════════════════════════════════════════════════
# 5. Shorthand names
# ═══════════════════════════════════════════════════════════════════


class TestShorthandResolution:
    """Shorthand names resolve via _SHORTHAND_TO_PROVIDER table."""

    @pytest.mark.parametrize(
        "shorthand, expected_provider",
        [
            ("mock", "mock"),
            ("random", "mock"),
            ("test", "mock"),
            ("groot", "groot"),
            ("dreamgen", "dreamgen"),
            ("dreamzero", "dreamzero"),
            ("cosmos", "cosmos_predict"),
            ("cosmos_predict", "cosmos_predict"),
            ("go1", "go1"),
        ],
    )
    def test_all_shorthands(self, shorthand, expected_provider):
        provider, kwargs = resolve_policy(shorthand)
        assert provider == expected_provider

    def test_shorthand_case_insensitive(self):
        provider, _ = resolve_policy("MOCK")
        assert provider == "mock"

    def test_shorthand_mixed_case(self):
        provider, _ = resolve_policy("GrOoT")
        assert provider == "groot"

    def test_shorthand_returns_empty_kwargs(self):
        _, kwargs = resolve_policy("mock")
        assert kwargs == {}

    def test_shorthand_table_complete(self):
        """Every entry in _SHORTHAND_TO_PROVIDER is accounted for in tests."""
        tested = {"mock", "random", "test", "groot", "dreamgen", "dreamzero", "cosmos", "cosmos_predict", "go1"}
        assert set(_SHORTHAND_TO_PROVIDER.keys()) == tested


# ═══════════════════════════════════════════════════════════════════
# 6. Model ID overrides (nvidia/* disambiguation)
# ═══════════════════════════════════════════════════════════════════


class TestModelIdOverrides:
    """_MODEL_ID_OVERRIDES take precedence over org-based routing."""

    @pytest.mark.parametrize(
        "model_id, expected_provider",
        [
            ("nvidia/gr00t", "groot"),
            ("nvidia/groot", "groot"),
            ("nvidia/alpamayo", "alpamayo"),
            ("nvidia/dreamzero", "dreamzero"),
            ("nvidia/cosmos-predict", "cosmos_predict"),
            ("nvidia/cosmos-predict2", "cosmos_predict"),
            ("nvidia/cosmos-predict2.5-2b", "cosmos_predict"),
            ("nvidia/cosmos-predict2.5-14b", "cosmos_predict"),
            ("nvidia/cosmos-policy", "cosmos_predict"),
            ("agibot-world/GO-1", "go1"),
            ("agibot-world/GO-1-Air", "go1"),
        ],
    )
    def test_all_overrides(self, model_id, expected_provider):
        provider, kwargs = resolve_policy(model_id)
        assert provider == expected_provider
        assert kwargs["pretrained_name_or_path"] == model_id

    def test_override_precedence_over_org(self):
        """nvidia/alpamayo must route to alpamayo, NOT groot (the nvidia org default)."""
        provider, _ = resolve_policy("nvidia/alpamayo")
        assert provider == "alpamayo"
        assert provider != "groot"  # the nvidia org default

    def test_override_case_insensitive_match(self):
        """Override matching is case-insensitive via .lower()."""
        provider, _ = resolve_policy("NVIDIA/GR00T")
        assert provider == "groot"

    def test_override_prefix_matching(self):
        """Overrides use startswith, so nvidia/cosmos-predict2.5-2b-ft also works."""
        provider, _ = resolve_policy("nvidia/cosmos-predict2.5-2b-custom-finetune")
        assert provider == "cosmos_predict"

    def test_override_table_complete(self):
        """Every entry in _MODEL_ID_OVERRIDES is accounted for in tests."""
        tested_prefixes = {
            "nvidia/gr00t",
            "nvidia/groot",
            "nvidia/alpamayo",
            "nvidia/dreamzero",
            "nvidia/cosmos-predict",
            "nvidia/cosmos-predict2",
            "nvidia/cosmos-predict2.5-2b",
            "nvidia/cosmos-predict2.5-14b",
            "nvidia/cosmos-policy",
            "agibot-world/go-1",
            "agibot-world/go-1-air",
        }
        assert set(_MODEL_ID_OVERRIDES.keys()) == tested_prefixes


# ═══════════════════════════════════════════════════════════════════
# 7. HuggingFace org → provider routing
# ═══════════════════════════════════════════════════════════════════


class TestHfOrgRouting:
    """org/model strings route via _HF_ORG_TO_PROVIDER."""

    @pytest.mark.parametrize(
        "model_id, expected_provider",
        [
            ("lerobot/act_aloha_sim_transfer_cube_human", "lerobot_local"),
            ("openvla/openvla-7b", "openvla"),
            ("microsoft/magma-8b", "magma"),
            ("internrobotics/internvla-2b", "internvla"),
            ("robotics-diffusion-transformer/rdt-1b", "rdt"),
            ("unitreerobotics/unifolm-g1", "unifolm"),
            ("baai/robobrain-7b", "robobrain"),
            ("nvidia/some-new-groot-model", "groot"),  # fallback nvidia org
            ("cogact/cogact-base", "cogact"),
            ("dream-org/dreamgen-v2", "dreamgen"),
            ("agibot-world/GO-1", "go1"),
        ],
    )
    def test_all_orgs(self, model_id, expected_provider):
        provider, _ = resolve_policy(model_id)
        assert provider == expected_provider

    def test_org_table_complete(self):
        """Every entry in _HF_ORG_TO_PROVIDER is accounted for in tests."""
        tested_orgs = {
            "lerobot",
            "openvla",
            "microsoft",
            "internrobotics",
            "robotics-diffusion-transformer",
            "unitreerobotics",
            "baai",
            "nvidia",
            "cogact",
            "dream-org",
            "agibot-world",
        }
        assert set(_HF_ORG_TO_PROVIDER.keys()) == tested_orgs

    def test_nvidia_without_override_falls_to_groot(self):
        """nvidia/some-random-model → groot (the nvidia org default)."""
        provider, _ = resolve_policy("nvidia/some-totally-new-model")
        assert provider == "groot"


# ═══════════════════════════════════════════════════════════════════
# 8. HuggingFace kwarg routing (model_id vs pretrained_name_or_path)
# ═══════════════════════════════════════════════════════════════════


class TestHfKwargRouting:
    """Provider-specific kwarg keys are set correctly."""

    def test_lerobot_uses_pretrained_name(self):
        _, kwargs = resolve_policy("lerobot/act_aloha_sim")
        assert "pretrained_name_or_path" in kwargs
        assert "model_id" not in kwargs

    @pytest.mark.parametrize(
        "model_id",
        [
            "openvla/openvla-7b",
            "microsoft/magma-8b",
            "internrobotics/internvla-2b",
            "robotics-diffusion-transformer/rdt-1b",
            "unitreerobotics/unifolm-g1",
            "baai/robobrain-7b",
            "cogact/cogact-base",
            "agibot-world/some-go1-variant",
        ],
    )
    def test_vla_providers_use_model_id(self, model_id):
        _, kwargs = resolve_policy(model_id)
        assert "model_id" in kwargs
        assert kwargs["model_id"] == model_id

    def test_groot_uses_pretrained_name(self):
        """GR00T (via nvidia org) uses pretrained_name_or_path."""
        _, kwargs = resolve_policy("nvidia/some-groot-model")
        assert "pretrained_name_or_path" in kwargs

    def test_dreamgen_uses_pretrained_name(self):
        _, kwargs = resolve_policy("dream-org/dreamgen-v2")
        assert "pretrained_name_or_path" in kwargs


# ═══════════════════════════════════════════════════════════════════
# 9. Unknown HuggingFace org → lerobot_local fallback
# ═══════════════════════════════════════════════════════════════════


class TestHfUnknownOrg:
    """Unknown orgs with / fall back to lerobot_local."""

    def test_unknown_org_defaults_to_lerobot_local(self):
        provider, kwargs = resolve_policy("someuser/my-custom-policy")
        assert provider == "lerobot_local"
        assert kwargs["pretrained_name_or_path"] == "someuser/my-custom-policy"

    def test_unknown_org_preserves_model_id(self):
        _, kwargs = resolve_policy("random-org/random-model-v3")
        assert kwargs["pretrained_name_or_path"] == "random-org/random-model-v3"

    def test_unknown_org_extra_kwargs(self):
        _, kwargs = resolve_policy("custom/model", device="cuda")
        assert kwargs["device"] == "cuda"


# ═══════════════════════════════════════════════════════════════════
# 10. Bare provider name resolution
# ═══════════════════════════════════════════════════════════════════


class TestBareProviderName:
    """Registered provider names (without / or :) resolve directly."""

    def test_bare_mock_resolves(self):
        """'mock' is both a shorthand AND a provider — shorthand takes priority."""
        provider, _ = resolve_policy("mock")
        assert provider == "mock"

    def test_bare_provider_returns_lowercase(self):
        provider, _ = resolve_policy("mock")
        assert provider == provider.lower()


# ═══════════════════════════════════════════════════════════════════
# 11. Fallback resolution
# ═══════════════════════════════════════════════════════════════════


class TestFallbackResolution:
    """Completely unknown strings fall back to lerobot_local."""

    def test_unknown_string_falls_back(self):
        provider, kwargs = resolve_policy("/path/to/local/checkpoint")
        # Contains / so goes through HF path, unknown org → lerobot_local
        assert provider == "lerobot_local"

    def test_local_model_path(self):
        provider, kwargs = resolve_policy("some-local-model")
        # No / or : — if not a registered provider, falls through to fallback
        # May resolve via bare provider or fallback depending on registration
        assert provider is not None  # should never error


# ═══════════════════════════════════════════════════════════════════
# 12. Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and input validation."""

    def test_whitespace_stripped(self):
        provider, _ = resolve_policy("  mock  ")
        assert provider == "mock"

    def test_return_type_is_tuple(self):
        result = resolve_policy("mock")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_provider_is_string(self):
        provider, _ = resolve_policy("mock")
        assert isinstance(provider, str)

    def test_kwargs_is_dict(self):
        _, kwargs = resolve_policy("mock")
        assert isinstance(kwargs, dict)

    def test_extra_kwargs_override(self):
        """Extra kwargs passed to resolve_policy should appear in result."""
        _, kwargs = resolve_policy("lerobot/act_aloha_sim", num_steps=100)
        assert kwargs["num_steps"] == 100

    def test_extra_kwargs_override_server_address(self):
        """Extra kwargs can override parsed values."""
        _, kwargs = resolve_policy("ws://gpu:9000", port=1234)
        # extra_kwargs update happens after parse, so port should be 1234
        assert kwargs["port"] == 1234


# ═══════════════════════════════════════════════════════════════════
# 13. Mapping table structural consistency
# ═══════════════════════════════════════════════════════════════════


class TestMappingConsistency:
    """Structural invariants on the mapping tables."""

    def test_all_shorthand_values_are_strings(self):
        for key, val in _SHORTHAND_TO_PROVIDER.items():
            assert isinstance(key, str)
            assert isinstance(val, str)

    def test_all_org_values_are_strings(self):
        for key, val in _HF_ORG_TO_PROVIDER.items():
            assert isinstance(key, str)
            assert isinstance(val, str)

    def test_all_override_values_are_strings(self):
        for key, val in _MODEL_ID_OVERRIDES.items():
            assert isinstance(key, str)
            assert isinstance(val, str)

    def test_all_override_keys_have_slash(self):
        """Overrides should be org/model format."""
        for key in _MODEL_ID_OVERRIDES:
            assert "/" in key, f"Override key '{key}' missing org/ prefix"

    def test_no_duplicate_shorthand_providers_overlap_with_orgs(self):
        """Shorthands shouldn't collide with org names (to avoid confusion)."""
        for shorthand in _SHORTHAND_TO_PROVIDER:
            # A shorthand like "lerobot" would cause confusion with the org
            if shorthand in _HF_ORG_TO_PROVIDER:
                # This is only a problem if they map to DIFFERENT providers
                assert _SHORTHAND_TO_PROVIDER[shorthand] == _HF_ORG_TO_PROVIDER[shorthand], (
                    f"Shorthand '{shorthand}' maps to '{_SHORTHAND_TO_PROVIDER[shorthand]}' "
                    f"but org maps to '{_HF_ORG_TO_PROVIDER[shorthand]}'"
                )


# ═══════════════════════════════════════════════════════════════════
# 14. Docstring examples (smoke test)
# ═══════════════════════════════════════════════════════════════════


class TestDocstringExamples:
    """Every example from the resolve_policy docstring must work."""

    def test_lerobot_act_aloha(self):
        provider, kwargs = resolve_policy("lerobot/act_aloha_sim_transfer_cube_human")
        assert provider == "lerobot_local"
        assert kwargs["pretrained_name_or_path"] == "lerobot/act_aloha_sim_transfer_cube_human"

    def test_mock(self):
        provider, kwargs = resolve_policy("mock")
        assert provider == "mock"
        assert kwargs == {}

    def test_localhost_8080(self):
        provider, kwargs = resolve_policy("localhost:8080")
        assert provider == "lerobot_async"
        assert kwargs["server_address"] == "localhost:8080"

    def test_ws_gpu_server(self):
        provider, kwargs = resolve_policy("ws://gpu-server:9000")
        assert provider == "dreamzero"
        assert kwargs["host"] == "gpu-server"
        assert kwargs["port"] == 9000

    def test_openvla(self):
        provider, kwargs = resolve_policy("openvla/openvla-7b")
        assert provider == "openvla"
        assert kwargs["model_id"] == "openvla/openvla-7b"
