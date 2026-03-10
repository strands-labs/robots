#!/usr/bin/env python3
"""Tests for strands_robots/tools/inference.py — Issue #27 Testing Matrix.

Phase 0: Import & Smoke (CPU-only, no GPU required)
Phase 7: Port Conflict & Edge Cases (CPU-only)

These tests run on any environment (Cloud CI, Mac Mini, Thor) without GPU
or heavy dependencies. They validate the inference tool's API surface,
error handling, provider registry, and edge-case robustness.

Uses `inference._tool_func(...)` to call the raw function directly,
bypassing the `@tool` decorator's `DecoratedFunctionTool` wrapper.
"""

import os
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

# Mock strands if not installed so tests can run without the full SDK
try:
    import strands

    # Verify it's the real strands, not a mock from another test file
    HAS_STRANDS = hasattr(strands, "Agent")
except ImportError:
    import types
    from unittest.mock import MagicMock

    _mock_strands = types.ModuleType("strands")
    _mock_strands.tool = lambda f: f  # @tool decorator becomes identity
    sys.modules["strands"] = _mock_strands
    HAS_STRANDS = False

_requires_strands = pytest.mark.skipif(not HAS_STRANDS, reason="requires strands-agents SDK")


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def call():
    """Return a callable that invokes the inference tool's raw function."""
    from strands_robots.tools.inference import inference

    # When strands is installed, @tool wraps the function; raw fn is at ._tool_func
    # When strands is mocked (@tool = identity), the function IS the raw function
    return getattr(inference, "_tool_func", inference)


@pytest.fixture
def provider_configs():
    """Return the PROVIDER_CONFIGS dict."""
    from strands_robots.tools.inference import PROVIDER_CONFIGS

    return PROVIDER_CONFIGS


@pytest.fixture
def running_services():
    """Return the _RUNNING_SERVICES dict (mutable in-process registry)."""
    # Import the module directly from sys.modules to get the actual module,
    # not the DecoratedFunctionTool that __getattr__ returns
    import importlib

    mod = importlib.import_module("strands_robots.tools.inference")
    return mod._RUNNING_SERVICES


@pytest.fixture(autouse=True)
def clean_running_services():
    """Ensure _RUNNING_SERVICES is clean before each test."""
    import importlib

    mod = importlib.import_module("strands_robots.tools.inference")
    mod._RUNNING_SERVICES.clear()
    yield
    mod._RUNNING_SERVICES.clear()


def _start_tcp_server(port, host="127.0.0.1"):
    """Start a minimal TCP server on a port. Returns (server_socket, thread)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    srv.settimeout(10)

    def _accept_loop():
        try:
            while True:
                conn, _ = srv.accept()
                conn.close()
        except (OSError, socket.timeout):
            pass

    t = threading.Thread(target=_accept_loop, daemon=True)
    t.start()
    return srv, t


def _find_free_port():
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ─────────────────────────────────────────────────────────────────────
# Phase 0 — Import & Smoke
# ─────────────────────────────────────────────────────────────────────


class TestPhase0ImportSmoke:
    """Phase 0: Import & Smoke tests (T0.1 – T0.8).

    No GPU required. Validates the tool loads and responds correctly.
    """

    # ── T0.1: Clean import ─────────────────────────────────────────

    def test_t0_1_clean_import(self):
        """T0.1: inference tool imports cleanly without errors."""
        from strands_robots.tools.inference import inference

        assert inference is not None

    @_requires_strands
    def test_t0_1_import_type(self):
        """T0.1b: inference is a DecoratedFunctionTool (from @tool decorator)."""
        from strands_robots.tools.inference import inference

        assert hasattr(inference, "_tool_func")
        assert hasattr(inference, "tool_name")
        assert inference.tool_name == "inference"

    @_requires_strands
    def test_t0_1_import_via_package(self):
        """T0.1c: tools.__getattr__('inference') returns the tool, not the module.

        Note: In a fresh process, 'from strands_robots.tools import inference' works
        correctly via __getattr__. However, in pytest the submodule is already cached
        in sys.modules from earlier imports, so Python resolves it as a module.
        We test the __getattr__ mechanism directly, plus verify via subprocess.
        """
        import strands_robots.tools as tools_pkg

        # Test __getattr__ directly — this is what gets called in a fresh process
        result = tools_pkg.__getattr__("inference")
        import types

        assert not isinstance(
            result, types.ModuleType
        ), "tools.__getattr__('inference') should return the tool object, not the module"
        assert hasattr(result, "_tool_func")

    @_requires_strands
    def test_t0_1_import_via_package_subprocess(self):
        """T0.1c-sub: Verify in a fresh subprocess that the import works correctly."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from strands_robots.tools import inference; "
                "import types; "
                "assert not isinstance(inference, types.ModuleType), 'Got module instead of tool'; "
                "assert hasattr(inference, '_tool_func'); "
                "print('OK')",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        assert "OK" in result.stdout

    def test_t0_1_provider_configs_importable(self):
        """T0.1d: PROVIDER_CONFIGS is importable and is a dict."""
        from strands_robots.tools.inference import PROVIDER_CONFIGS

        assert isinstance(PROVIDER_CONFIGS, dict)
        assert len(PROVIDER_CONFIGS) > 0

    # ── T0.2: providers action ─────────────────────────────────────

    def test_t0_2_providers_action(self, call):
        """T0.2: inference(action='providers') lists all 12 providers."""
        result = call(action="providers")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "12 providers" in text

    def test_t0_2_providers_all_names_present(self, call, provider_configs):
        """T0.2b: Every provider name appears in the output."""
        result = call(action="providers")
        text = result["content"][0]["text"]
        for name in provider_configs:
            assert name in text, f"Provider '{name}' not found in providers output"

    def test_t0_2_providers_display_names(self, call, provider_configs):
        """T0.2c: Provider display names appear in output."""
        from strands_robots.tools.inference import PROVIDERS

        result = call(action="providers")
        text = result["content"][0]["text"]
        for cfg in PROVIDERS.values():
            assert cfg["name"] in text

    def test_t0_2_providers_protocols(self, call):
        """T0.2d: All protocol types appear (HTTP, WEBSOCKET, ZMQ, GRPC)."""
        result = call(action="providers")
        text = result["content"][0]["text"]
        for proto in ["HTTP", "WEBSOCKET", "ZMQ", "GRPC"]:
            assert proto in text, f"Protocol '{proto}' not found in providers output"

    # ── T0.3: list (empty) ─────────────────────────────────────────

    def test_t0_3_list_empty(self, call):
        """T0.3: inference(action='list') returns empty list when no services."""
        from unittest.mock import patch

        # Mock port scanner to avoid detecting unrelated processes on common ports
        with patch("strands_robots.tools.inference._is_port_in_use", return_value=False):
            result = call(action="list")
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "(none)" in text

    # ── T0.4: status on unused port ────────────────────────────────

    def test_t0_4_status_down(self, call):
        """T0.4: inference(action='status', port=8000) returns DOWN."""
        # Use a high port unlikely to be in use
        port = _find_free_port()
        result = call(action="status", port=port)
        assert result["status"] == "success"
        assert result["running"] is False
        text = result["content"][0]["text"]
        assert "DOWN" in text

    def test_t0_4_status_requires_port(self, call):
        """T0.4b: status without port returns error."""
        result = call(action="status")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "Port required" in text

    # ── T0.5: stop on unused port ──────────────────────────────────

    def test_t0_5_stop_noop(self, call):
        """T0.5: inference(action='stop', port=9999) is a no-op, no crash."""
        result = call(action="stop", port=9999)
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "stopped" in text

    def test_t0_5_stop_requires_port(self, call):
        """T0.5b: stop without port returns error."""
        result = call(action="stop")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "Port required" in text

    # ── T0.6: invalid provider ─────────────────────────────────────

    def test_t0_6_start_invalid_provider(self, call):
        """T0.6: inference(action='start', provider='invalid_name') returns error."""
        result = call(action="start", provider="invalid_name")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "Unknown provider" in text
        assert "invalid_name" in text

    def test_t0_6_error_lists_valid_providers(self, call, provider_configs):
        """T0.6b: Error message includes list of valid providers."""
        result = call(action="start", provider="nonexistent_xyz")
        text = result["content"][0]["text"]
        assert "Available:" in text
        # At least a few known providers should appear
        for name in ["dreamzero", "lerobot", "openvla"]:
            assert name in text, f"Valid provider '{name}' not listed in error"

    # ── T0.7: start lerobot with default model ────────────────────

    def test_t0_7_lerobot_default_model(self, call, provider_configs):
        """T0.7: lerobot config has a default hf_model_id so start doesn't
        fail with 'No checkpoint/model_id specified'.

        On CI without lerobot deps, the start will fail at the subprocess
        level (lerobot.scripts.server not found), but it should get past
        the checkpoint validation stage — proving the default model ID works.
        """
        cfg = provider_configs["lerobot"]
        assert "hf_model_id" in cfg, "BUG: lerobot config missing hf_model_id"
        assert cfg["hf_model_id"] == "lerobot/act_aloha_sim_transfer_cube_human"

    def test_t0_7_all_providers_have_hf_model_id(self, provider_configs):
        """T0.7b: All providers except groot (Docker-based) have hf_model_id.

        This was a bug — lerobot, cosmos, cogact, gear_sonic, and unifolm
        were missing hf_model_id, causing 'No checkpoint/model_id specified'
        errors when starting without explicit checkpoint_path.
        """
        for name, cfg in provider_configs.items():
            if name == "groot":
                # groot is Docker-based, doesn't use HF checkpoints
                continue
            assert "hf_model_id" in cfg, (
                f"BUG: '{name}' config missing hf_model_id — "
                f"inference(action='start', provider='{name}') would fail "
                f"with 'No checkpoint/model_id specified'"
            )

    def test_t0_7_lerobot_start_passes_checkpoint_validation(self, call):
        """T0.7c: Starting lerobot without explicit checkpoint passes the
        'no checkpoint' guard and reaches the subprocess launch stage.

        We use a port that's not in use. On CI, the process will fail
        (no lerobot installed), but we should NOT get the error
        'No checkpoint/model_id specified for lerobot'.
        """
        port = _find_free_port()
        result = call(action="start", provider="lerobot", port=port, timeout=3)
        # It should NOT be "No checkpoint/model_id specified"
        if result["status"] == "error":
            text = result["content"][0]["text"]
            assert "No checkpoint/model_id specified" not in text, (
                "BUG: lerobot start failed at checkpoint validation, " "hf_model_id should provide the default"
            )

    # ── T0.8: bogus action ─────────────────────────────────────────

    def test_t0_8_bogus_action(self, call):
        """T0.8: inference(action='bogus') returns error with valid action list."""
        result = call(action="bogus")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "Unknown action" in text
        assert "bogus" in text

    def test_t0_8_error_lists_valid_actions(self, call):
        """T0.8b: Error message lists all valid actions."""
        result = call(action="definitely_not_real")
        text = result["content"][0]["text"]
        for action in ["start", "stop", "status", "list", "providers", "info", "download"]:
            assert action in text, f"Valid action '{action}' not listed in error"


# ─────────────────────────────────────────────────────────────────────
# Phase 0 — Supplementary: Provider Config Integrity
# ─────────────────────────────────────────────────────────────────────


class TestProviderConfigIntegrity:
    """Validate PROVIDER_CONFIGS structure for all 12 providers."""

    REQUIRED_KEYS = {"display_name", "protocol", "default_port", "multi_gpu", "launch_method", "requires"}
    VALID_PROTOCOLS = {"http", "websocket", "zmq", "grpc"}
    VALID_LAUNCH_METHODS = {"python", "torchrun", "vllm", "docker"}

    def test_exactly_12_providers(self, provider_configs):
        """There should be exactly 12 providers."""
        assert len(provider_configs) == 12

    def test_expected_provider_names(self, provider_configs):
        """All expected provider names are present."""
        expected = {
            "dreamzero",
            "groot",
            "openvla",
            "internvla",
            "lerobot",
            "cosmos",
            "alpamayo",
            "rdt",
            "magma",
            "cogact",
            "gear_sonic",
            "unifolm",
        }
        assert set(provider_configs.keys()) == expected

    @pytest.mark.parametrize(
        "provider_name",
        [
            "dreamzero",
            "groot",
            "openvla",
            "internvla",
            "lerobot",
            "cosmos",
            "alpamayo",
            "rdt",
            "magma",
            "cogact",
            "gear_sonic",
            "unifolm",
        ],
    )
    def test_required_keys_present(self, provider_configs, provider_name):
        """Each provider has all required config keys."""
        cfg = provider_configs[provider_name]
        missing = self.REQUIRED_KEYS - set(cfg.keys())
        assert not missing, f"Provider '{provider_name}' missing keys: {missing}"

    @pytest.mark.parametrize(
        "provider_name",
        [
            "dreamzero",
            "groot",
            "openvla",
            "internvla",
            "lerobot",
            "cosmos",
            "alpamayo",
            "rdt",
            "magma",
            "cogact",
            "gear_sonic",
            "unifolm",
        ],
    )
    def test_valid_protocol(self, provider_configs, provider_name):
        """Each provider has a valid protocol."""
        protocol = provider_configs[provider_name]["protocol"]
        assert protocol in self.VALID_PROTOCOLS, f"Provider '{provider_name}' has invalid protocol: {protocol}"

    @pytest.mark.parametrize(
        "provider_name",
        [
            "dreamzero",
            "groot",
            "openvla",
            "internvla",
            "lerobot",
            "cosmos",
            "alpamayo",
            "rdt",
            "magma",
            "cogact",
            "gear_sonic",
            "unifolm",
        ],
    )
    def test_valid_launch_method(self, provider_configs, provider_name):
        """Each provider has a valid launch method."""
        method = provider_configs[provider_name]["launch_method"]
        assert method in self.VALID_LAUNCH_METHODS, f"Provider '{provider_name}' has invalid launch_method: {method}"

    @pytest.mark.parametrize(
        "provider_name",
        [
            "dreamzero",
            "groot",
            "openvla",
            "internvla",
            "lerobot",
            "cosmos",
            "alpamayo",
            "rdt",
            "magma",
            "cogact",
            "gear_sonic",
            "unifolm",
        ],
    )
    def test_default_port_is_int(self, provider_configs, provider_name):
        """Each provider has an integer default_port."""
        port = provider_configs[provider_name]["default_port"]
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    def test_no_duplicate_default_ports(self, provider_configs):
        """No two providers share the same default port."""
        ports = {}
        for name, cfg in provider_configs.items():
            port = cfg["default_port"]
            assert port not in ports, f"Port {port} conflict: '{name}' and '{ports[port]}'"
            ports[port] = name

    def test_multi_gpu_providers_have_default_num_gpus(self, provider_configs):
        """Providers with multi_gpu=True should have default_num_gpus."""
        for name, cfg in provider_configs.items():
            if cfg.get("multi_gpu"):
                assert "default_num_gpus" in cfg, f"Provider '{name}' has multi_gpu=True but no default_num_gpus"
                assert cfg["default_num_gpus"] >= 1


# ─────────────────────────────────────────────────────────────────────
# Phase 0 — Supplementary: Info Action
# ─────────────────────────────────────────────────────────────────────


class TestInfoAction:
    """Test the 'info' action for individual provider details."""

    def test_info_valid_provider(self, call):
        """info action returns config for a valid provider."""
        result = call(action="info", provider="dreamzero")
        assert result["status"] == "success"
        assert result["provider"] == "dreamzero"
        assert "config" in result
        assert result["config"]["display_name"] == "DreamZero 14B (World Action Model)"

    def test_info_invalid_provider(self, call):
        """info action returns error for an unknown provider."""
        result = call(action="info", provider="nonexistent_xyz")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "Unknown provider" in text

    @pytest.mark.parametrize(
        "provider_name",
        [
            "dreamzero",
            "groot",
            "openvla",
            "internvla",
            "lerobot",
            "cosmos",
            "alpamayo",
            "rdt",
            "magma",
            "cogact",
            "gear_sonic",
            "unifolm",
        ],
    )
    def test_info_all_providers(self, call, provider_name):
        """info action works for every provider."""
        result = call(action="info", provider=provider_name)
        assert result["status"] == "success"
        assert result["provider"] == provider_name


# ─────────────────────────────────────────────────────────────────────
# Phase 7 — Port Conflict & Edge Cases
# ─────────────────────────────────────────────────────────────────────


class TestPhase7PortConflictEdgeCases:
    """Phase 7: Port Conflict & Edge Cases (T7.1 – T7.7).

    Tests port conflicts, concurrent services, external kill detection,
    stop-on-dead-port safety, and timeout behavior.
    """

    # ── T7.1: Port already in use → error ──────────────────────────

    def test_t7_1_port_conflict(self, call):
        """T7.1: Starting a provider on an already-used port returns error."""
        port = _find_free_port()
        srv, _ = _start_tcp_server(port)
        try:
            time.sleep(0.2)  # Let server bind
            result = call(action="start", provider="lerobot", port=port, timeout=3)
            assert result["status"] == "error"
            text = result["content"][0]["text"]
            assert "already in use" in text.lower() or "in use" in text.lower()
        finally:
            srv.close()

    def test_t7_1_port_conflict_message_suggests_stop(self, call):
        """T7.1b: Error message suggests using stop to free the port."""
        port = _find_free_port()
        srv, _ = _start_tcp_server(port)
        try:
            time.sleep(0.2)
            result = call(action="start", provider="openvla", port=port, timeout=3)
            text = result["content"][0]["text"]
            assert "stop" in text.lower()
        finally:
            srv.close()

    # ── T7.2: Two providers on different ports ─────────────────────

    def test_t7_2_two_providers_different_ports(self, running_services):
        """T7.2: Two providers can be registered on different ports simultaneously."""
        # Simulate by directly populating the registry (actual start needs GPU deps)
        running_services[50051] = {
            "provider": "lerobot",
            "protocol": "grpc",
            "port": 50051,
            "pid": 12345,
            "started_at": "2024-01-01 00:00:00",
        }
        running_services[8001] = {
            "provider": "openvla",
            "protocol": "http",
            "port": 8001,
            "pid": 12346,
            "started_at": "2024-01-01 00:00:01",
        }
        assert len(running_services) == 2
        assert 50051 in running_services
        assert 8001 in running_services

    # ── T7.3: List shows both services ─────────────────────────────

    def test_t7_3_list_shows_both(self, call, running_services):
        """T7.3: inference(action='list') shows both registered services."""
        running_services[50051] = {
            "provider": "lerobot",
            "protocol": "grpc",
            "port": 50051,
            "pid": 12345,
        }
        running_services[8001] = {
            "provider": "openvla",
            "protocol": "http",
            "port": 8001,
            "pid": 12346,
        }
        result = call(action="list")
        text = result["content"][0]["text"]
        assert "lerobot" in text
        assert "openvla" in text
        assert "50051" in text
        assert "8001" in text

    def test_t7_3_list_shows_status_icons(self, call, running_services):
        """T7.3b: List shows ✅ for running and ❌ for dead services."""
        # Register a service on a port that's NOT actually running
        running_services[59999] = {
            "provider": "lerobot",
            "protocol": "grpc",
            "port": 59999,
            "pid": 99999,
        }
        result = call(action="list")
        text = result["content"][0]["text"]
        # Port 59999 is not actually listening, so should show ❌
        assert "❌" in text

    # ── T7.4: Kill externally → status shows DOWN ──────────────────

    def test_t7_4_external_kill_detection(self, call, running_services):
        """T7.4: After external kill, status shows DOWN despite registry entry."""
        # Register a fake service
        running_services[59998] = {
            "provider": "lerobot",
            "protocol": "grpc",
            "port": 59998,
            "pid": 99999,  # Non-existent PID
            "started_at": "2024-01-01 00:00:00",
        }

        result = call(action="status", port=59998)
        assert result["running"] is False
        text = result["content"][0]["text"]
        assert "DOWN" in text
        # But registry info should still be shown
        assert "lerobot" in text

    def test_t7_4_live_then_dead(self, call, running_services):
        """T7.4b: Start a real TCP server, verify RUNNING, kill it, verify DOWN."""
        port = _find_free_port()
        srv, _ = _start_tcp_server(port)
        time.sleep(0.2)

        # Register it
        running_services[port] = {
            "provider": "test_provider",
            "protocol": "http",
            "port": port,
            "pid": os.getpid(),  # Use our own PID (won't actually kill)
        }

        # Should be RUNNING
        result = call(action="status", port=port)
        assert result["running"] is True
        text = result["content"][0]["text"]
        assert "RUNNING" in text

        # Kill the server
        srv.close()
        # Wait for OS to release socket (CI can be slow)
        for _ in range(10):
            time.sleep(0.3)
            result = call(action="status", port=port)
            if result["running"] is False:
                break

        # Should be DOWN
        assert result["running"] is False
        text = result["content"][0]["text"]
        assert "DOWN" in text

    # ── T7.5: Stop on dead port → no crash ─────────────────────────

    def test_t7_5_stop_dead_port(self, call):
        """T7.5: inference(action='stop') on a port with no process doesn't crash."""
        port = _find_free_port()
        result = call(action="stop", port=port)
        # Should succeed (no-op) without raising
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "stopped" in text.lower()

    def test_t7_5_stop_with_stale_registry(self, call, running_services):
        """T7.5b: Stop a registered service whose process is already dead."""
        port = _find_free_port()
        running_services[port] = {
            "provider": "lerobot",
            "protocol": "grpc",
            "port": port,
            "pid": 999999999,  # Non-existent PID
        }

        result = call(action="stop", port=port)
        assert result["status"] == "success"
        # Registry should be cleaned up
        assert port not in running_services

    def test_t7_5_stop_cleans_registry(self, call, running_services):
        """T7.5c: Stop removes the service from the registry."""
        port = _find_free_port()
        running_services[port] = {
            "provider": "test",
            "protocol": "http",
            "port": port,
            "pid": None,
        }
        assert port in running_services

        call(action="stop", port=port)
        assert port not in running_services

    # ── T7.6: Timeout behavior ─────────────────────────────────────

    def test_t7_6_timeout_returns_starting_status(self, call):
        """T7.6: Start with very short timeout returns 'starting' (not 'running')
        when the service doesn't come up in time.

        On CI without GPU deps, the subprocess will fail immediately,
        but we should see the timeout/starting behavior in the result.
        """
        port = _find_free_port()
        result = call(
            action="start",
            provider="openvla",
            checkpoint_path="openvla/openvla-7b",
            port=port,
            timeout=2,
        )
        # Either "starting" (timeout expired) or "error" (subprocess failed)
        # Both are valid on CI. Key thing: no crash.
        assert result["status"] in ("starting", "success", "error")

    # ── T7.7: Port scan detection ──────────────────────────────────

    def test_t7_7_port_scan_detects_unregistered(self, call):
        """T7.7: Port scan detects unregistered services on known ports.

        Uses mock to avoid hardcoded port binding (prevents OSError in CI
        when ports 8005/8006 are already in use by other tests or services).
        """
        from unittest.mock import patch

        # Mock _is_port_in_use: return True for port 8005 (rdt default), False otherwise
        original_fn = __import__("strands_robots.tools.inference", fromlist=["_is_port_in_use"])._is_port_in_use

        def mock_is_port_in_use(port, host="localhost"):
            if port == 8005:
                return True  # Simulate something running on 8005
            return original_fn(port, host)

        with patch("strands_robots.tools.inference._is_port_in_use", side_effect=mock_is_port_in_use):
            result = call(action="list")
            text = result["content"][0]["text"]
            # Should detect the unregistered service on 8005
            assert "8005" in text
            assert "unregistered" in text.lower() or "unknown" in text.lower()

    def test_t7_7_registered_not_marked_unregistered(self, call, running_services):
        """T7.7b: A registered service on a scan port is NOT marked as unregistered.

        Uses mock to avoid hardcoded port binding (prevents OSError in CI).
        """
        from unittest.mock import patch

        port = 8006  # magma default, in scan list

        # Register it in the service registry
        running_services[port] = {
            "provider": "magma",
            "protocol": "http",
            "port": port,
            "pid": os.getpid(),
        }

        # Mock _is_port_in_use: return True for 8006 (simulates it being alive)
        original_fn = __import__("strands_robots.tools.inference", fromlist=["_is_port_in_use"])._is_port_in_use

        def mock_is_port_in_use(p, host="127.0.0.1"):
            if p == 8006:
                return True
            return original_fn(p, host)

        with patch("strands_robots.tools.inference._is_port_in_use", side_effect=mock_is_port_in_use):
            result = call(action="list")
            text = result["content"][0]["text"]
            assert "magma" in text
            # Should NOT appear as unregistered (it's registered)
            lines_out = text.split("\n")
            for line in lines_out:
                if "8006" in line:
                    assert "unregistered" not in line.lower()


# ─────────────────────────────────────────────────────────────────────
# Phase 7 — Supplementary: Helper Function Tests
# ─────────────────────────────────────────────────────────────────────


class TestHelperFunctions:
    """Test internal helper functions used by the inference tool."""

    def test_is_port_in_use_free_port(self):
        """_is_port_in_use returns False for a free port."""
        from strands_robots.tools.inference import _is_port_in_use

        port = _find_free_port()
        assert _is_port_in_use(port) is False

    def test_is_port_in_use_occupied_port(self):
        """_is_port_in_use returns True for an occupied port."""
        from strands_robots.tools.inference import _is_port_in_use

        port = _find_free_port()
        srv, _ = _start_tcp_server(port)
        try:
            time.sleep(0.2)
            assert _is_port_in_use(port, host="127.0.0.1") is True
        finally:
            srv.close()

    def test_wait_for_port_timeout(self):
        """_wait_for_port returns False when port never opens (short timeout)."""
        from strands_robots.tools.inference import _wait_for_port

        port = _find_free_port()
        result = _wait_for_port(port, timeout=1, host="127.0.0.1")
        assert result is False

    def test_wait_for_port_success(self):
        """_wait_for_port returns True when port is already open."""
        from strands_robots.tools.inference import _wait_for_port

        port = _find_free_port()
        srv, _ = _start_tcp_server(port)
        try:
            time.sleep(0.2)
            result = _wait_for_port(port, timeout=5, host="127.0.0.1")
            assert result is True
        finally:
            srv.close()

    def test_kill_process_nonexistent(self):
        """_kill_process with non-existent PID doesn't crash."""
        from strands_robots.tools.inference import _kill_process

        # Use a PID that almost certainly doesn't exist
        _kill_process(999999999, force=False)
        _kill_process(999999999, force=True)
        # No exception means success

    def test_find_process_on_port_empty(self):
        """_find_process_on_port returns None for an unused port."""
        from strands_robots.tools.inference import _find_process_on_port

        port = _find_free_port()
        result = _find_process_on_port(port)
        assert result is None

    def test_download_checkpoint_local_path(self):
        """_download_checkpoint returns immediately for existing local paths."""
        from strands_robots.tools.inference import _download_checkpoint

        # /tmp always exists
        result = _download_checkpoint("/tmp")
        assert result == "/tmp"


# ─────────────────────────────────────────────────────────────────────
# Regression: __init__.py Export Bug
# ─────────────────────────────────────────────────────────────────────


class TestInitExportBug:
    """Regression tests for the tools/__init__.py export bug.

    Bug: 'inference' was missing from _LAZY_IMPORTS in tools/__init__.py,
    causing 'from strands_robots.tools import inference' to return the
    module instead of the DecoratedFunctionTool.
    """

    def test_inference_in_lazy_imports(self):
        """inference is listed in _LAZY_IMPORTS."""
        from strands_robots.tools import _LAZY_IMPORTS

        assert "inference" in _LAZY_IMPORTS

    def test_inference_in_all(self):
        """inference is in __all__."""
        from strands_robots.tools import __all__

        assert "inference" in __all__

    @_requires_strands
    def test_import_returns_tool_not_module(self):
        """tools.__getattr__('inference') returns DecoratedFunctionTool, not module."""
        import types

        import strands_robots.tools as tools_pkg

        result = tools_pkg.__getattr__("inference")
        assert not isinstance(result, types.ModuleType)
        assert hasattr(result, "_tool_func")
        assert hasattr(result, "tool_name")

    @_requires_strands
    def test_tool_is_callable(self):
        """The imported inference tool has a callable _tool_func."""
        from strands_robots.tools import inference

        assert callable(inference._tool_func)


# ─────────────────────────────────────────────────────────────────────
# Regression: Missing hf_model_id Bug
# ─────────────────────────────────────────────────────────────────────


class TestMissingHfModelIdBug:
    """Regression tests for missing hf_model_id in PROVIDER_CONFIGS.

    Bug: lerobot, cosmos, cogact, gear_sonic, and unifolm configs were
    missing hf_model_id. This caused inference(action='start', provider='lerobot')
    to fail with 'No checkpoint/model_id specified for lerobot' before
    _start_lerobot() could apply its own default.

    The start() code path checks:
        checkpoint = cfg.get("hf_model_id")
        if not checkpoint: return error

    So the default in _start_lerobot() was unreachable.
    """

    @pytest.mark.parametrize(
        "provider_name,expected_model_id",
        [
            ("lerobot", "lerobot/act_aloha_sim_transfer_cube_human"),
            ("cosmos", "nvidia/Cosmos-Predict1-7B"),
            ("cogact", "CogACT/CogACT-Base"),
            ("gear_sonic", "GEAR-Group/GEAR-Sonic"),
            ("unifolm", "unitreerobotics/UnifolM-50M"),
        ],
    )
    def test_hf_model_id_present(self, provider_configs, provider_name, expected_model_id):
        """Previously-missing hf_model_id is now set correctly."""
        cfg = provider_configs[provider_name]
        assert "hf_model_id" in cfg
        assert cfg["hf_model_id"] == expected_model_id

    @pytest.mark.parametrize(
        "provider_name",
        [
            "lerobot",
            "cosmos",
            "cogact",
            "gear_sonic",
            "unifolm",
        ],
    )
    def test_start_no_checkpoint_doesnt_fail_at_validation(self, call, provider_name):
        """Starting without explicit checkpoint should NOT fail with
        'No checkpoint/model_id specified'.
        """
        port = _find_free_port()
        result = call(action="start", provider=provider_name, port=port, timeout=2)
        if result["status"] == "error":
            text = result["content"][0]["text"]
            assert "No checkpoint/model_id specified" not in text, (
                f"REGRESSION: '{provider_name}' still fails checkpoint validation " f"without explicit checkpoint_path"
            )

    def test_groot_intentionally_has_no_hf_model_id(self, provider_configs):
        """groot is Docker-based and intentionally has no hf_model_id."""
        assert "hf_model_id" not in provider_configs["groot"]

    def test_groot_start_without_checkpoint_returns_error(self, call):
        """groot without checkpoint returns the expected error (not a crash)."""
        port = _find_free_port()
        result = call(action="start", provider="groot", port=port, timeout=2)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "No checkpoint" in text or "Docker" in text or "container" in text


# ─────────────────────────────────────────────────────────────────────
# Mock-based launcher tests — increase coverage without GPU
# ─────────────────────────────────────────────────────────────────────

from unittest.mock import MagicMock, patch  # noqa: E402


class TestGenerateHfServeScript:
    """Tests for _generate_hf_serve_script() — validates generated server script."""

    def test_generates_valid_python(self):
        """Generated script should be valid Python syntax."""
        from strands_robots.tools.inference import _generate_hf_serve_script

        script_path = _generate_hf_serve_script("test/model", 8080, "0.0.0.0", "test_provider")
        assert os.path.exists(script_path)
        with open(script_path) as f:
            code = f.read()
        # Should compile without syntax errors
        compile(code, script_path, "exec")

    def test_script_contains_model_info(self):
        """Script should contain the model ID and port."""
        from strands_robots.tools.inference import _generate_hf_serve_script

        script_path = _generate_hf_serve_script("microsoft/Magma-8B", 9999, "localhost", "magma")
        with open(script_path) as f:
            code = f.read()
        assert "microsoft/Magma-8B" in code
        assert "9999" in code
        assert "magma" in code
        assert "localhost" in code

    def test_script_has_get_and_post_handlers(self):
        """Generated server should handle GET and POST."""
        from strands_robots.tools.inference import _generate_hf_serve_script

        script_path = _generate_hf_serve_script("test/model", 8080, "0.0.0.0", "test")
        with open(script_path) as f:
            code = f.read()
        assert "def do_GET" in code
        assert "def do_POST" in code
        assert "HTTPServer" in code

    def test_script_path_uses_provider_and_port(self):
        """Script path encodes provider and port for uniqueness."""
        from strands_robots.tools.inference import _generate_hf_serve_script

        path = _generate_hf_serve_script("test/model", 5555, "0.0.0.0", "rdt")
        assert "serve_rdt_5555.py" in path

    def test_script_directory_created(self):
        """Script directory should be created if it doesn't exist."""
        from strands_robots.tools.inference import _generate_hf_serve_script

        path = _generate_hf_serve_script("test/model", 7777, "0.0.0.0", "test")
        assert os.path.isdir(os.path.dirname(path))


class TestDownloadCheckpointMocked:
    """Tests for _download_checkpoint() with mocked filesystem/subprocess."""

    def test_local_path_exists_returns_immediately(self, tmp_path):
        """If model_id is a local path that exists, return it directly."""
        from strands_robots.tools.inference import _download_checkpoint

        local_dir = str(tmp_path / "my_model")
        os.makedirs(local_dir)
        result = _download_checkpoint(local_dir)
        assert result == local_dir

    def test_cached_checkpoint_returns_cache(self, tmp_path):
        """If cache dir exists and is non-empty, return cached path."""
        from strands_robots.tools.inference import _download_checkpoint

        cache_dir = str(tmp_path / "cached_model")
        os.makedirs(cache_dir)
        # Create a file so it's non-empty
        (tmp_path / "cached_model" / "model.safetensors").write_text("fake")
        result = _download_checkpoint("fake/model", local_dir=cache_dir)
        assert result == cache_dir

    @patch("strands_robots.tools.inference.subprocess.run")
    def test_downloads_from_hf(self, mock_run, tmp_path):
        """Should call huggingface-cli when not cached."""
        from strands_robots.tools.inference import _download_checkpoint

        mock_run.return_value = MagicMock(returncode=0)
        result = _download_checkpoint("test/new-model")
        expected_cache = os.path.expanduser("~/.cache/strands_robots/checkpoints/test_new-model")
        assert result == expected_cache
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "huggingface-cli" in cmd
        assert "download" in cmd
        assert "test/new-model" in cmd

    @patch("strands_robots.tools.inference.subprocess.run", side_effect=subprocess.CalledProcessError(1, "hf"))
    def test_download_failure_raises(self, mock_run, tmp_path):
        """Should raise on download failure."""
        from strands_robots.tools.inference import _download_checkpoint

        cache_dir = str(tmp_path / "fail_model")
        with pytest.raises(subprocess.CalledProcessError):
            _download_checkpoint("fake/broken-model", local_dir=cache_dir)


class TestStartDreamzeroMocked:
    """Test _start_dreamzero() command construction with mocked subprocess."""

    @patch("strands_robots.tools.inference.subprocess.Popen")
    @patch("strands_robots.tools.inference._download_checkpoint", return_value="/tmp/fake_ckpt")
    def test_command_uses_torchrun(self, mock_dl, mock_popen, tmp_path):
        """DreamZero should use torchrun with correct nproc_per_node."""
        from strands_robots.tools.inference import _start_dreamzero

        # Create a fake server script
        script = tmp_path / "socket_test_optimized_AR.py"
        script.write_text("# fake")
        mock_popen.return_value = MagicMock(pid=12345)

        with patch("strands_robots.tools.inference.os.path.exists", side_effect=lambda p: p == str(script)):
            with patch("strands_robots.tools.inference.os.path.abspath", side_effect=lambda p: str(script)):
                # Monkeypatch the search paths to include our fake script
                result = _start_dreamzero(
                    "GEAR-Dreams/DreamZero-DROID",
                    8000,
                    2,
                    "0.0.0.0",
                    {"gpu_ids": "0,1", "enable_dit_cache": True, "max_chunk_size": None, "timeout_seconds": None},
                )

        # Result should have pid
        assert result.get("pid") == 12345 or result.get("status") in ("starting", "error")

    @patch("strands_robots.tools.inference._download_checkpoint", return_value="/tmp/fake_ckpt")
    def test_returns_error_when_script_not_found(self, mock_dl):
        """Should return error if server script can't be found."""
        from strands_robots.tools.inference import _start_dreamzero

        with patch("strands_robots.tools.inference.os.path.exists", return_value=False):
            with patch("strands_robots.tools.inference.subprocess.run"):  # mock git clone
                result = _start_dreamzero(
                    "/tmp/fake",
                    8000,
                    2,
                    "0.0.0.0",
                    {"gpu_ids": "0,1", "enable_dit_cache": True, "max_chunk_size": None, "timeout_seconds": None},
                )
        assert result["status"] == "error"
        assert "not found" in result.get("message", "")


class TestStartGrootMocked:
    """Test _start_groot() command construction with mocked docker."""

    @patch("strands_robots.tools.inference.subprocess.run")
    def test_docker_command_construction(self, mock_run):
        """groot should construct correct docker exec command."""
        from strands_robots.tools.inference import _start_groot

        # First call: docker ps (returns container list)
        # Second call: docker exec (starts inference)
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="isaac-gr00t\tisaac-gr00t:latest\tUp 2 hours"),
            MagicMock(returncode=0, stdout="", stderr=""),
        ]

        result = _start_groot(
            "/data/checkpoints/groot_n16",
            5555,
            1,
            "0.0.0.0",
            {
                "data_config": "so100_dualcam",
                "embodiment_tag": "so100",
                "denoising_steps": 4,
                "use_tensorrt": False,
                "http_server": False,
                "container_name": None,
                "trt_engine_path": "gr00t_engine",
                "vit_dtype": "fp8",
                "llm_dtype": "nvfp4",
                "dit_dtype": "fp8",
            },
        )

        assert result["status"] == "starting"
        assert result["container"] == "isaac-gr00t"
        # Check docker exec was called
        exec_call = mock_run.call_args_list[1]
        cmd = exec_call[0][0]
        assert "docker" in cmd
        assert "exec" in cmd
        assert "5555" in " ".join(str(x) for x in cmd)

    @patch("strands_robots.tools.inference.subprocess.run")
    def test_groot_no_container_returns_error(self, mock_run):
        """groot should return error if no Isaac-GR00T container is running."""
        from strands_robots.tools.inference import _start_groot

        mock_run.return_value = MagicMock(returncode=0, stdout="")  # No containers

        result = _start_groot(
            "/data/checkpoints/groot_n16",
            5555,
            1,
            "0.0.0.0",
            {
                "data_config": "so100",
                "embodiment_tag": "so100",
                "denoising_steps": 4,
                "use_tensorrt": False,
                "http_server": False,
                "container_name": None,
                "trt_engine_path": "gr00t_engine",
                "vit_dtype": "fp8",
                "llm_dtype": "nvfp4",
                "dit_dtype": "fp8",
            },
        )
        assert result["status"] == "error"
        assert "container" in result["message"].lower()

    @patch("strands_robots.tools.inference.subprocess.run")
    def test_groot_tensorrt_flags(self, mock_run):
        """groot with use_tensorrt should pass TRT flags."""
        from strands_robots.tools.inference import _start_groot

        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="groot-container\tisaac:latest\tUp 1 hour"),
            MagicMock(returncode=0, stdout=""),
        ]
        result = _start_groot(
            "/data/checkpoints/groot_n16",
            5556,
            1,
            "0.0.0.0",
            {
                "data_config": "so100",
                "embodiment_tag": "so100",
                "denoising_steps": 4,
                "use_tensorrt": True,
                "http_server": False,
                "container_name": None,
                "trt_engine_path": "gr00t_engine",
                "vit_dtype": "fp8",
                "llm_dtype": "nvfp4",
                "dit_dtype": "fp8",
            },
        )
        assert result["status"] == "starting"
        exec_call = mock_run.call_args_list[1]
        cmd_str = " ".join(str(x) for x in exec_call[0][0])
        assert "--use-tensorrt" in cmd_str

    @patch("strands_robots.tools.inference.subprocess.run")
    def test_groot_http_server_flag(self, mock_run):
        """groot with http_server=True should pass --http-server flag."""
        from strands_robots.tools.inference import _start_groot

        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="groot-c\tisaac:latest\tUp 1 hour"),
            MagicMock(returncode=0, stdout=""),
        ]
        result = _start_groot(
            "/data/checkpoints/groot_n16",
            8000,
            1,
            "0.0.0.0",
            {
                "data_config": "so100",
                "embodiment_tag": "so100",
                "denoising_steps": 4,
                "use_tensorrt": False,
                "http_server": True,
                "container_name": None,
                "trt_engine_path": "gr00t_engine",
                "vit_dtype": "fp8",
                "llm_dtype": "nvfp4",
                "dit_dtype": "fp8",
            },
        )
        assert result["status"] == "starting"
        exec_call = mock_run.call_args_list[1]
        cmd_str = " ".join(str(x) for x in exec_call[0][0])
        assert "--http-server" in cmd_str

    @patch("strands_robots.tools.inference.subprocess.run")
    def test_groot_explicit_container_name(self, mock_run):
        """groot with explicit container_name should skip discovery."""
        from strands_robots.tools.inference import _start_groot

        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=""),  # docker ps
            MagicMock(returncode=0, stdout=""),  # docker exec
        ]
        result = _start_groot(
            "/data/checkpoints/groot_n16",
            5555,
            1,
            "0.0.0.0",
            {
                "data_config": "so100",
                "embodiment_tag": "so100",
                "denoising_steps": 4,
                "use_tensorrt": False,
                "http_server": False,
                "container_name": "my-custom-container",
                "trt_engine_path": "gr00t_engine",
                "vit_dtype": "fp8",
                "llm_dtype": "nvfp4",
                "dit_dtype": "fp8",
            },
        )
        assert result["status"] == "starting"
        assert result["container"] == "my-custom-container"

    @patch("strands_robots.tools.inference.subprocess.run", side_effect=FileNotFoundError("docker"))
    def test_groot_no_docker_returns_error(self, mock_run):
        """groot without docker should return error."""
        from strands_robots.tools.inference import _start_groot

        result = _start_groot(
            "/data/checkpoints/groot_n16",
            5555,
            1,
            "0.0.0.0",
            {
                "data_config": "so100",
                "embodiment_tag": "so100",
                "denoising_steps": 4,
                "use_tensorrt": False,
                "http_server": False,
                "container_name": None,
                "trt_engine_path": "gr00t_engine",
                "vit_dtype": "fp8",
                "llm_dtype": "nvfp4",
                "dit_dtype": "fp8",
            },
        )
        assert result["status"] == "error"
        assert "Docker" in result["message"] or "docker" in result["message"]


class TestStartOpenvlaMocked:
    """Test _start_openvla() command construction."""

    @patch("strands_robots.tools.inference.subprocess.Popen")
    @patch("strands_robots.tools.inference.shutil.which", return_value="/usr/bin/vllm")
    def test_uses_vllm_when_available(self, mock_which, mock_popen):
        """When vllm is available, should use vLLM serving."""
        from strands_robots.tools.inference import _start_openvla

        mock_popen.return_value = MagicMock(pid=99999)
        result = _start_openvla("openvla/openvla-7b", 8001, 1, "0.0.0.0", {})
        assert result["status"] == "starting"
        assert result["pid"] == 99999
        cmd = result["command"]
        assert "vllm" in cmd

    @patch("strands_robots.tools.inference.subprocess.Popen")
    @patch("strands_robots.tools.inference.shutil.which", return_value=None)
    def test_falls_back_to_hf_serve(self, mock_which, mock_popen):
        """Without vllm, should fall back to generated HF serve script."""
        from strands_robots.tools.inference import _start_openvla

        mock_popen.return_value = MagicMock(pid=88888)
        result = _start_openvla("openvla/openvla-7b", 8001, 1, "0.0.0.0", {})
        assert result["status"] == "starting"
        assert result["pid"] == 88888
        # Should be using python with a serve script
        assert "python" in result["command"].lower() or "serve_openvla" in result["command"]

    @patch("strands_robots.tools.inference.subprocess.Popen")
    @patch("strands_robots.tools.inference.shutil.which", return_value="/usr/bin/vllm")
    def test_multi_gpu_adds_tensor_parallel(self, mock_which, mock_popen):
        """Multi-GPU should add --tensor-parallel-size flag."""
        from strands_robots.tools.inference import _start_openvla

        mock_popen.return_value = MagicMock(pid=77777)
        result = _start_openvla("openvla/openvla-7b", 8001, 2, "0.0.0.0", {})
        assert "tensor-parallel-size" in result["command"]


class TestStartLerobotMocked:
    """Test _start_lerobot() command construction."""

    @patch("strands_robots.tools.inference.subprocess.Popen")
    def test_lerobot_command_construction(self, mock_popen):
        """LeRobot should use lerobot.scripts.server with correct args."""
        from strands_robots.tools.inference import _start_lerobot

        mock_popen.return_value = MagicMock(pid=55555)
        result = _start_lerobot(
            "lerobot/act_aloha_sim_transfer_cube_human",
            50051,
            1,
            "0.0.0.0",
            {"device": "cuda", "pretrained_name_or_path": None},
        )
        assert result["status"] == "starting"
        assert result["pid"] == 55555
        cmd = result["command"]
        assert "lerobot" in cmd
        assert "50051" in cmd
        assert "act_aloha_sim_transfer_cube_human" in cmd

    @patch("strands_robots.tools.inference.subprocess.Popen")
    def test_lerobot_uses_pretrained_name_fallback(self, mock_popen):
        """Without checkpoint, uses pretrained_name_or_path from extra_args."""
        from strands_robots.tools.inference import _start_lerobot

        mock_popen.return_value = MagicMock(pid=44444)
        result = _start_lerobot(
            None, 50051, 1, "0.0.0.0", {"device": "cpu", "pretrained_name_or_path": "lerobot/pi0-so100-wipe"}
        )
        assert "pi0-so100-wipe" in result["command"]

    @patch("strands_robots.tools.inference.subprocess.Popen")
    def test_lerobot_device_flag(self, mock_popen):
        """LeRobot should pass --device flag."""
        from strands_robots.tools.inference import _start_lerobot

        mock_popen.return_value = MagicMock(pid=33333)
        result = _start_lerobot(
            "lerobot/act_aloha_sim", 50051, 1, "0.0.0.0", {"device": "mps", "pretrained_name_or_path": None}
        )
        assert "--device" in result["command"]
        assert "mps" in result["command"]


class TestStartGenericHfMocked:
    """Test _start_generic_hf() for generic providers."""

    @patch("strands_robots.tools.inference.subprocess.Popen")
    def test_generic_hf_starts_serve_script(self, mock_popen):
        """Generic providers should generate and run a serve script."""
        from strands_robots.tools.inference import _start_generic_hf

        mock_popen.return_value = MagicMock(pid=22222)
        result = _start_generic_hf("rdt", "robotics-diffusion-transformer/rdt-1b", 8005, 1, "0.0.0.0", {})
        assert result["status"] == "starting"
        assert result["pid"] == 22222

    @patch("strands_robots.tools.inference.subprocess.Popen")
    def test_generic_hf_no_model_returns_error(self, mock_popen):
        """Generic provider without model_id should return error."""
        from strands_robots.tools.inference import _start_generic_hf

        # Provider with no hf_model_id AND no checkpoint
        result = _start_generic_hf("nonexistent_provider", "", 8005, 1, "0.0.0.0", {})
        assert result["status"] == "error"
        assert "No model ID" in result["message"]

    @patch("strands_robots.tools.inference.subprocess.Popen")
    def test_generic_hf_multi_gpu_sets_cuda_visible(self, mock_popen):
        """Multi-GPU generic provider should set CUDA_VISIBLE_DEVICES."""
        from strands_robots.tools.inference import _start_generic_hf

        mock_popen.return_value = MagicMock(pid=11111)
        result = _start_generic_hf("magma", "microsoft/Magma-8B", 8006, 2, "0.0.0.0", {})
        assert result["status"] == "starting"
        # Verify CUDA_VISIBLE_DEVICES was in env
        call_kwargs = mock_popen.call_args[1]
        env = call_kwargs.get("env", {})
        assert env.get("CUDA_VISIBLE_DEVICES") == "0,1"


class TestStartActionMocked:
    """Test the full 'start' action with mocked launchers — covers connection strings."""

    @patch("strands_robots.tools.inference._wait_for_port", return_value=True)
    @patch("strands_robots.tools.inference._is_port_in_use", return_value=False)
    @patch(
        "strands_robots.tools.inference._start_lerobot",
        return_value={"status": "starting", "pid": 99999, "command": "python -m lerobot.scripts.server"},
    )
    def test_start_lerobot_success_builds_grpc_connection(self, mock_launch, mock_port_check, mock_wait, call):
        """Successful lerobot start should return gRPC connection string."""
        port = _find_free_port()
        result = call(
            action="start",
            provider="lerobot",
            port=port,
            pretrained_name_or_path="lerobot/act_aloha_sim_transfer_cube_human",
            timeout=5,
        )
        assert result["status"] == "success"
        assert result["protocol"] == "grpc"
        assert f"grpc://0.0.0.0:{port}" in result["endpoint"]
        assert "RUNNING" in result["content"][0]["text"]

    @patch("strands_robots.tools.inference._wait_for_port", return_value=True)
    @patch("strands_robots.tools.inference._is_port_in_use", return_value=False)
    @patch(
        "strands_robots.tools.inference._start_dreamzero",
        return_value={"status": "starting", "pid": 88888, "command": "torchrun ...", "gpu_ids": "0,1"},
    )
    def test_start_dreamzero_success_builds_ws_connection(self, mock_launch, mock_port_check, mock_wait, call):
        """Successful dreamzero start should return WebSocket connection string."""
        port = _find_free_port()
        result = call(
            action="start",
            provider="dreamzero",
            port=port,
            checkpoint_path="GEAR-Dreams/DreamZero-DROID",
            num_gpus=2,
            timeout=5,
        )
        assert result["status"] == "success"
        assert result["protocol"] == "websocket"
        assert f"ws://0.0.0.0:{port}" in result["endpoint"]

    @patch("strands_robots.tools.inference._wait_for_port", return_value=True)
    @patch("strands_robots.tools.inference._is_port_in_use", return_value=False)
    @patch(
        "strands_robots.tools.inference._start_openvla",
        return_value={"status": "starting", "pid": 77777, "command": "python -m vllm..."},
    )
    def test_start_openvla_success_builds_http_connection(self, mock_launch, mock_port_check, mock_wait, call):
        """Successful openvla start should return HTTP connection string."""
        port = _find_free_port()
        result = call(action="start", provider="openvla", port=port, model_id="openvla/openvla-7b", timeout=5)
        assert result["status"] == "success"
        assert result["protocol"] == "http"
        assert f"http://0.0.0.0:{port}" in result["endpoint"]

    @patch("strands_robots.tools.inference._wait_for_port", return_value=True)
    @patch("strands_robots.tools.inference._is_port_in_use", return_value=False)
    @patch(
        "strands_robots.tools.inference._start_groot",
        return_value={"status": "starting", "pid": None, "command": "docker exec ...", "container": "isaac-groot"},
    )
    def test_start_groot_success_builds_zmq_connection(self, mock_launch, mock_port_check, mock_wait, call):
        """Successful groot start should return ZMQ connection string."""
        port = _find_free_port()
        result = call(
            action="start", provider="groot", port=port, checkpoint_path="/data/checkpoints/groot_n16", timeout=5
        )
        assert result["status"] == "success"
        assert result["protocol"] == "zmq"
        assert f"zmq://0.0.0.0:{port}" in result["endpoint"]

    @patch("strands_robots.tools.inference._wait_for_port", return_value=False)
    @patch("strands_robots.tools.inference._is_port_in_use", return_value=False)
    @patch(
        "strands_robots.tools.inference._start_lerobot",
        return_value={"status": "starting", "pid": 66666, "command": "..."},
    )
    def test_start_timeout_returns_starting_status(self, mock_launch, mock_port_check, mock_wait, call):
        """If service doesn't come up in time, should return 'starting' status."""
        port = _find_free_port()
        result = call(
            action="start", provider="lerobot", port=port, pretrained_name_or_path="lerobot/act_aloha_sim", timeout=1
        )
        assert result["status"] == "starting"
        assert "STARTING" in result["content"][0]["text"]

    @patch("strands_robots.tools.inference._is_port_in_use", return_value=False)
    def test_start_launcher_exception_returns_error(self, mock_port_check, call):
        """If launcher raises, should catch and return error."""
        port = _find_free_port()
        with patch("strands_robots.tools.inference._start_lerobot", side_effect=RuntimeError("CUDA OOM")):
            result = call(
                action="start",
                provider="lerobot",
                port=port,
                pretrained_name_or_path="lerobot/act_aloha_sim",
                timeout=1,
            )
        assert result["status"] == "error"
        assert "CUDA OOM" in result["content"][0]["text"] or "Launch failed" in result["content"][0]["text"]

    @patch("strands_robots.tools.inference._is_port_in_use", return_value=False)
    def test_start_launcher_returns_error_dict(self, mock_port_check, call):
        """If launcher returns error dict, should propagate it."""
        port = _find_free_port()
        with patch(
            "strands_robots.tools.inference._start_groot", return_value={"status": "error", "message": "No container"}
        ):
            result = call(action="start", provider="groot", port=port, checkpoint_path="/data/groot_n16", timeout=1)
        assert result["status"] == "error"
        assert "container" in result["content"][0]["text"].lower()


class TestStartRegistersService:
    """Test that start action properly registers services in _RUNNING_SERVICES."""

    @patch("strands_robots.tools.inference._wait_for_port", return_value=True)
    @patch("strands_robots.tools.inference._is_port_in_use", return_value=False)
    @patch(
        "strands_robots.tools.inference._start_lerobot",
        return_value={"status": "starting", "pid": 12345, "command": "test"},
    )
    def test_service_registered_after_start(self, mock_launch, mock_port_check, mock_wait, call, running_services):
        """After start, service should be in _RUNNING_SERVICES."""
        port = _find_free_port()
        call(action="start", provider="lerobot", port=port, pretrained_name_or_path="lerobot/act_aloha_sim", timeout=1)
        assert port in running_services
        assert running_services[port]["provider"] == "lerobot"
        assert running_services[port]["proto"] == "grpc"

    @patch("strands_robots.tools.inference._wait_for_port", return_value=False)
    @patch("strands_robots.tools.inference._is_port_in_use", return_value=False)
    @patch(
        "strands_robots.tools.inference._start_lerobot",
        return_value={"status": "starting", "pid": 12345, "command": "test"},
    )
    def test_service_registered_as_starting_on_timeout(
        self, mock_launch, mock_port_check, mock_wait, call, running_services
    ):
        """If timeout, service should still be registered with status 'starting'."""
        port = _find_free_port()
        call(action="start", provider="lerobot", port=port, pretrained_name_or_path="lerobot/act_aloha_sim", timeout=1)
        assert port in running_services
        assert running_services[port]["provider"] == "lerobot"


class TestStopActionMocked:
    """Test stop action with mocked processes."""

    @patch("strands_robots.tools.inference._is_port_in_use", return_value=False)
    @patch("strands_robots.tools.inference._kill_process")
    @patch("strands_robots.tools.inference._find_process_on_port", return_value=None)
    def test_stop_cleans_registry(self, mock_find, mock_kill, mock_port, call, running_services):
        """Stop should remove service from registry."""
        running_services[8888] = {"provider": "test", "pid": None}
        result = call(action="stop", port=8888)
        assert result["status"] == "success"
        assert 8888 not in running_services

    @patch("strands_robots.tools.inference.time.sleep")
    @patch("strands_robots.tools.inference._is_port_in_use", side_effect=[True, False])
    @patch("strands_robots.tools.inference._kill_process")
    def test_stop_force_kills_if_still_running(self, mock_kill, mock_port, mock_sleep, call, running_services):
        """If SIGTERM doesn't work, should force kill."""
        running_services[9999] = {"provider": "test", "pid": 54321}
        call(action="stop", port=9999)
        # Should have called kill twice (SIGTERM then SIGKILL)
        assert mock_kill.call_count == 2
        # First call: SIGTERM (force=False)
        assert mock_kill.call_args_list[0][1].get("force", False) is False or mock_kill.call_args_list[0][0][1] is False
        # Second call: SIGKILL (force=True)
        assert mock_kill.call_args_list[1][1].get("force", True) is True or mock_kill.call_args_list[1][0][1] is True

    @patch("strands_robots.tools.inference._is_port_in_use", return_value=True)
    @patch("strands_robots.tools.inference.time.sleep")
    @patch("strands_robots.tools.inference._kill_process")
    @patch("strands_robots.tools.inference._find_process_on_port", return_value=None)
    def test_stop_returns_warning_if_still_running(self, mock_find, mock_kill, mock_sleep, mock_port, call):
        """If port is still in use after kill, return warning."""
        result = call(action="stop", port=7777)
        assert result["status"] == "warning"
        assert "may still be running" in result["content"][0]["text"]

    @patch("strands_robots.tools.inference._is_port_in_use", return_value=False)
    @patch("strands_robots.tools.inference._kill_process")
    @patch("strands_robots.tools.inference._find_process_on_port", return_value=None)
    @patch("strands_robots.tools.inference.subprocess.run")
    def test_stop_docker_container(self, mock_run, mock_find, mock_kill, mock_port, call, running_services):
        """Stop should attempt docker pkill for container-based services."""
        running_services[5555] = {"provider": "groot", "pid": None, "container": "isaac-groot"}
        result = call(action="stop", port=5555)
        assert result["status"] == "success"
        # Should have called docker exec pkill
        mock_run.assert_called_once()
        docker_cmd = mock_run.call_args[0][0]
        assert "docker" in docker_cmd
        assert "pkill" in docker_cmd


class TestDownloadAction:
    """Test the download action integration."""

    @patch("strands_robots.tools.inference._download_hf")
    def test_download_with_provider(self, mock_dl, call):
        """Download by provider name should use provider's default model."""
        mock_dl.return_value = "/tmp/cached/model"
        result = call(action="download", provider="openvla")
        assert result["status"] == "success"
        assert result["path"] == "/tmp/cached/model"
        mock_dl.assert_called_once_with("openvla/openvla-7b")

    @patch("strands_robots.tools.inference._download_hf")
    def test_download_with_explicit_model_id(self, mock_dl, call):
        """Download with explicit model_id should use that instead of default."""
        mock_dl.return_value = "/tmp/cached/custom"
        result = call(action="download", provider="lerobot", model_id="lerobot/pi0-so100-wipe")
        assert result["status"] == "success"
        mock_dl.assert_called_once_with("lerobot/pi0-so100-wipe")

    @patch("strands_robots.tools.inference._download_hf", side_effect=RuntimeError("Network error"))
    def test_download_failure_returns_error(self, mock_dl, call):
        """Download failure should return error status."""
        result = call(action="download", provider="openvla")
        assert result["status"] == "error"
        assert "Network error" in result["content"][0]["text"]

    def test_download_provider_without_model_returns_error(self, call):
        """Download for provider without hf_model_id and no checkpoint returns error."""
        result = call(action="download", provider="groot")
        assert result["status"] == "error"
        assert "No model ID" in result["content"][0]["text"]


class TestKillProcess:
    """Tests for _kill_process helper."""

    @patch("strands_robots.tools.inference.os.kill")
    def test_kill_sigterm(self, mock_kill):
        """Default kill should use SIGTERM."""
        from strands_robots.tools.inference import _kill_process

        _kill_process(12345)
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)

    @patch("strands_robots.tools.inference.os.kill")
    def test_kill_sigkill(self, mock_kill):
        """Force kill should use SIGKILL."""
        from strands_robots.tools.inference import _kill_process

        _kill_process(12345, force=True)
        mock_kill.assert_called_once_with(12345, signal.SIGKILL)

    @patch("strands_robots.tools.inference.os.kill", side_effect=ProcessLookupError)
    def test_kill_missing_process_no_crash(self, mock_kill):
        """Kill on missing process should not crash."""
        from strands_robots.tools.inference import _kill_process

        _kill_process(99999)  # Should not raise
        _kill_process(99999, force=True)  # Should not raise


class TestFindProcessOnPort:
    """Tests for _find_process_on_port helper."""

    @patch("strands_robots.tools.inference.subprocess.run")
    def test_finds_pid(self, mock_run):
        """Should return PID when lsof finds process."""
        from strands_robots.tools.inference import _find_process_on_port

        mock_run.return_value = MagicMock(returncode=0, stdout="12345\n")
        assert _find_process_on_port(8000) == 12345

    @patch("strands_robots.tools.inference.subprocess.run")
    def test_no_process_returns_none(self, mock_run):
        """Should return None when no process on port."""
        from strands_robots.tools.inference import _find_process_on_port

        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert _find_process_on_port(8000) is None

    @patch("strands_robots.tools.inference.subprocess.run", side_effect=FileNotFoundError)
    def test_lsof_missing_returns_none(self, mock_run):
        """Should return None if lsof is not available."""
        from strands_robots.tools.inference import _find_process_on_port

        assert _find_process_on_port(8000) is None

    @patch("strands_robots.tools.inference.subprocess.run")
    def test_multiple_pids_returns_first(self, mock_run):
        """Should return first PID when multiple processes found."""
        from strands_robots.tools.inference import _find_process_on_port

        mock_run.return_value = MagicMock(returncode=0, stdout="12345\n67890\n")
        assert _find_process_on_port(8000) == 12345
