"""Tests for strands_robots/tools/gr00t_inference.py — GR00T Docker service management.

All Docker and network operations are mocked. CPU-only.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock strands if not installed so tests can run without the full SDK
try:
    import strands

    HAS_STRANDS = hasattr(strands, "Agent")
except ImportError:
    import types

    _mock_strands = types.ModuleType("strands")
    _mock_strands.tool = lambda f: f  # @tool decorator becomes identity
    sys.modules["strands"] = _mock_strands
    HAS_STRANDS = False

from strands_robots.tools.gr00t_inference import (
    _check_service_status,
    _find_gr00t_containers,
    _is_service_running,
    _list_running_services,
    _start_service,
    _stop_service,
    gr00t_inference,
)

# When @tool is mocked, the function is raw (no _tool_func wrapper)
if not HAS_STRANDS:
    _call_gr00t = gr00t_inference
else:
    _call_gr00t = getattr(gr00t_inference, "_tool_func", gr00t_inference)

_requires = pytest.mark.skipif(not HAS_STRANDS, reason="requires strands-agents SDK")


# ═════════════════════════════════════════════════════════════════════════════
# Action Dispatch Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestActionDispatch:
    """Test top-level action routing."""

    def test_unknown_action(self):
        result = gr00t_inference(action="bogus")
        assert result["status"] == "error"
        assert "Unknown action" in result["message"]

    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    def test_find_containers_action(self, mock_find):
        mock_find.return_value = {"status": "success", "containers": []}
        gr00t_inference(action="find_containers")
        mock_find.assert_called_once()

    @patch("strands_robots.tools.gr00t_inference._list_running_services")
    def test_list_action(self, mock_list):
        mock_list.return_value = {"status": "success", "services": []}
        gr00t_inference(action="list")
        mock_list.assert_called_once()

    def test_status_no_port(self):
        result = gr00t_inference(action="status")
        assert result["status"] == "error"
        assert "Port" in result["message"]

    @patch("strands_robots.tools.gr00t_inference._check_service_status")
    def test_status_with_port(self, mock_status):
        mock_status.return_value = {"status": "success", "port": 5555}
        gr00t_inference(action="status", port=5555)
        mock_status.assert_called_once_with(5555)

    def test_stop_no_port(self):
        result = gr00t_inference(action="stop")
        assert result["status"] == "error"
        assert "Port" in result["message"]

    @patch("strands_robots.tools.gr00t_inference._stop_service")
    def test_stop_with_port(self, mock_stop):
        mock_stop.return_value = {"status": "success", "port": 5555}
        gr00t_inference(action="stop", port=5555)
        mock_stop.assert_called_once_with(5555)

    def test_start_no_checkpoint(self):
        result = gr00t_inference(action="start")
        assert result["status"] == "error"
        assert "Checkpoint" in result["message"]

    @patch("strands_robots.tools.gr00t_inference._start_service")
    def test_start_default_zmq_port(self, mock_start):
        mock_start.return_value = {"status": "success", "port": 5555}
        gr00t_inference(action="start", checkpoint_path="/data/model")
        # Default port for ZMQ is 5555
        args = mock_start.call_args
        assert args.kwargs["port"] == 5555

    @patch("strands_robots.tools.gr00t_inference._start_service")
    def test_start_default_http_port(self, mock_start):
        mock_start.return_value = {"status": "success", "port": 8000}
        gr00t_inference(action="start", checkpoint_path="/data/model", http_server=True)
        args = mock_start.call_args
        assert args.kwargs["port"] == 8000

    def test_restart_no_checkpoint(self):
        result = gr00t_inference(action="restart")
        assert result["status"] == "error"
        assert "Checkpoint" in result["message"]

    def test_restart_no_port(self):
        result = gr00t_inference(action="restart", checkpoint_path="/data/model")
        assert result["status"] == "error"
        assert "port" in result["message"].lower()

    @patch("strands_robots.tools.gr00t_inference._start_service")
    @patch("strands_robots.tools.gr00t_inference._stop_service")
    @patch("time.sleep")
    def test_restart_stops_then_starts(self, mock_sleep, mock_stop, mock_start):
        mock_stop.return_value = {"status": "success"}
        mock_start.return_value = {"status": "success", "port": 5555}
        gr00t_inference(action="restart", checkpoint_path="/data/model", port=5555)
        mock_stop.assert_called_once_with(5555)
        mock_start.assert_called_once()
        mock_sleep.assert_called_once_with(2)


# ═════════════════════════════════════════════════════════════════════════════
# _find_gr00t_containers Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestFindContainers:
    """Test Docker container discovery."""

    @patch("subprocess.run")
    def test_no_containers(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        result = _find_gr00t_containers()
        assert result["status"] == "success"
        assert result["containers"] == []

    @patch("subprocess.run")
    def test_gr00t_container_found(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="my-container\tisaac-gr00t:latest\tUp 2 hours\t0.0.0.0:5555->5555/tcp",
            returncode=0,
        )
        result = _find_gr00t_containers()
        assert result["status"] == "success"
        assert len(result["containers"]) == 1
        assert result["containers"][0]["name"] == "my-container"
        assert result["containers"][0]["image"] == "isaac-gr00t:latest"

    @patch("subprocess.run")
    def test_non_gr00t_container_filtered(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="postgres\tpostgres:16\tUp 5 hours\t5432/tcp",
            returncode=0,
        )
        result = _find_gr00t_containers()
        assert result["status"] == "success"
        assert len(result["containers"]) == 0

    @patch("subprocess.run")
    def test_docker_error(self, mock_run):
        from subprocess import CalledProcessError

        mock_run.side_effect = CalledProcessError(1, "docker")
        result = _find_gr00t_containers()
        assert result["status"] == "error"

    @patch("subprocess.run")
    def test_multiple_containers_mixed(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout=(
                "gr00t-1\tisaac-gr00t:v2\tUp 1 hour\t5555/tcp\n"
                "my-app\tnginx:latest\tUp 3 hours\t80/tcp\n"
                "jetson-thing\tisaac-jetson:latest\tExited\t"
            ),
            returncode=0,
        )
        result = _find_gr00t_containers()
        assert result["status"] == "success"
        # isaac-gr00t:v2 matches, jetson-thing has "isaac" in image but "jetson" in name
        assert len(result["containers"]) >= 1

    @patch("subprocess.run")
    def test_malformed_docker_output(self, mock_run):
        mock_run.return_value = MagicMock(stdout="incomplete_line", returncode=0)
        result = _find_gr00t_containers()
        assert result["status"] == "success"
        # Line has no tabs, so parts < 3
        assert len(result["containers"]) == 0


# ═════════════════════════════════════════════════════════════════════════════
# _is_service_running Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestIsServiceRunning:
    """Test port connectivity check."""

    @patch("socket.socket")
    def test_port_open(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.connect_ex.return_value = 0
        assert _is_service_running(5555) is True
        mock_sock.close.assert_called_once()

    @patch("socket.socket")
    def test_port_closed(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.connect_ex.return_value = 1
        assert _is_service_running(5555) is False

    @patch("socket.socket")
    def test_socket_exception(self, mock_socket_cls):
        mock_socket_cls.side_effect = OSError("Network error")
        assert _is_service_running(5555) is False


# ═════════════════════════════════════════════════════════════════════════════
# _list_running_services Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestListRunningServices:
    """Test service discovery."""

    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    def test_no_services(self, mock_running):
        mock_running.return_value = False
        result = _list_running_services()
        assert result["status"] == "success"
        assert result["services"] == []

    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    def test_zmq_service_found(self, mock_running):
        # Only port 5555 is running (ZMQ range)
        mock_running.side_effect = lambda p: p == 5555
        result = _list_running_services()
        assert result["status"] == "success"
        assert len(result["services"]) == 1
        assert result["services"][0]["port"] == 5555
        assert result["services"][0]["protocol"] == "ZMQ"

    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    def test_http_service_found(self, mock_running):
        mock_running.side_effect = lambda p: p == 8000
        result = _list_running_services()
        assert result["status"] == "success"
        assert len(result["services"]) == 1
        assert result["services"][0]["port"] == 8000
        assert result["services"][0]["protocol"] == "HTTP"

    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    def test_multiple_services(self, mock_running):
        mock_running.side_effect = lambda p: p in (5555, 8000)
        result = _list_running_services()
        assert result["status"] == "success"
        assert len(result["services"]) == 2


# ═════════════════════════════════════════════════════════════════════════════
# _check_service_status Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestCheckServiceStatus:
    """Test individual service status."""

    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    def test_running_zmq(self, mock_running):
        mock_running.return_value = True
        result = _check_service_status(5555)
        assert result["status"] == "success"
        assert result["service_status"] == "running"
        assert result["protocol"] == "ZMQ"

    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    def test_running_http(self, mock_running):
        mock_running.return_value = True
        result = _check_service_status(8000)
        assert result["status"] == "success"
        assert result["protocol"] == "HTTP"

    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    def test_not_running(self, mock_running):
        mock_running.return_value = False
        result = _check_service_status(5555)
        assert result["status"] == "error"
        assert result["service_status"] == "not_running"


# ═════════════════════════════════════════════════════════════════════════════
# _stop_service Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestStopService:
    """Test service stop logic."""

    @patch("subprocess.run")
    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    def test_stop_in_container(self, mock_find, mock_run):
        mock_find.return_value = {
            "status": "success",
            "containers": [{"name": "gr00t-1", "image": "isaac-gr00t:latest", "status": "Up 1h", "ports": ""}],
        }
        # pgrep returns pid, kill succeeds, second pgrep returns empty
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="12345\n"),  # pgrep finds pid
            MagicMock(returncode=0),  # kill -TERM
            MagicMock(returncode=1, stdout=""),  # second pgrep — process gone
        ]
        result = _stop_service(5555)
        assert result["status"] == "success"
        assert "gr00t-1" in result["container"]

    @patch("time.sleep")
    @patch("subprocess.run")
    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    def test_stop_force_kill(self, mock_find, mock_run, mock_sleep):
        mock_find.return_value = {
            "status": "success",
            "containers": [{"name": "gr00t-1", "image": "isaac-gr00t:latest", "status": "Up 1h", "ports": ""}],
        }
        # pgrep returns pid, kill -TERM, second pgrep still finds it, kill -KILL
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="12345\n"),
            MagicMock(returncode=0),
            MagicMock(returncode=0, stdout="12345\n"),  # still running
            MagicMock(returncode=0),  # kill -KILL
        ]
        result = _stop_service(5555)
        assert result["status"] == "success"

    @patch("subprocess.run")
    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    def test_stop_fallback_host(self, mock_find, mock_run):
        """No container matches → fall back to host lsof/kill."""
        mock_find.return_value = {
            "status": "success",
            "containers": [],  # no containers
        }
        # lsof finds pid on host
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="54321\n"),  # lsof
            MagicMock(returncode=0),  # kill -TERM
            MagicMock(returncode=1, stdout=""),  # second lsof — gone
        ]
        result = _stop_service(5555)
        assert result["status"] == "success"

    @patch("subprocess.run")
    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    def test_stop_no_service(self, mock_find, mock_run):
        mock_find.return_value = {"status": "success", "containers": []}
        mock_run.return_value = MagicMock(returncode=1, stdout="")  # lsof finds nothing
        result = _stop_service(5555)
        assert result["status"] == "success"
        assert "No service" in result["message"]

    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    def test_stop_exception(self, mock_find):
        mock_find.side_effect = Exception("Docker not available")
        result = _stop_service(5555)
        assert result["status"] == "error"


# ═════════════════════════════════════════════════════════════════════════════
# _start_service Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestStartService:
    """Test service start logic."""

    @patch("time.sleep")
    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    @patch("subprocess.run")
    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    def test_start_zmq_success(self, mock_find, mock_run, mock_running, mock_sleep):
        mock_find.return_value = {
            "status": "success",
            "containers": [{"name": "gr00t-1", "image": "isaac-gr00t:latest", "status": "Up 1h", "ports": ""}],
        }
        mock_run.return_value = MagicMock(returncode=0)
        mock_running.return_value = True

        result = _start_service(
            checkpoint_path="/data/model",
            port=5555,
            data_config="so100_dualcam",
            embodiment_tag="gr1",
            denoising_steps=4,
            host="0.0.0.0",
            container_name=None,
            policy_name="test_policy",
            timeout=5,
            use_tensorrt=False,
            trt_engine_path="engine",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            http_server=False,
            api_token=None,
        )
        assert result["status"] == "success"
        assert result["port"] == 5555
        assert result["protocol"] == "ZMQ"
        assert result["data_config"] == "so100_dualcam"

    @patch("time.sleep")
    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    @patch("subprocess.run")
    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    def test_start_http_success(self, mock_find, mock_run, mock_running, mock_sleep):
        mock_find.return_value = {
            "status": "success",
            "containers": [{"name": "gr00t-1", "image": "isaac-gr00t:latest", "status": "Up", "ports": ""}],
        }
        mock_run.return_value = MagicMock(returncode=0)
        mock_running.return_value = True

        result = _start_service(
            checkpoint_path="/data/model",
            port=8000,
            data_config="so100",
            embodiment_tag="so100",
            denoising_steps=4,
            host="0.0.0.0",
            container_name=None,
            policy_name=None,
            timeout=5,
            use_tensorrt=False,
            trt_engine_path="engine",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            http_server=True,
            api_token=None,
        )
        assert result["status"] == "success"
        assert result["protocol"] == "HTTP"
        assert "endpoint" in result

    @patch("time.sleep")
    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    @patch("subprocess.run")
    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    def test_start_with_tensorrt(self, mock_find, mock_run, mock_running, mock_sleep):
        mock_find.return_value = {
            "status": "success",
            "containers": [{"name": "gr00t-1", "image": "isaac-gr00t:latest", "status": "Up", "ports": ""}],
        }
        mock_run.return_value = MagicMock(returncode=0)
        mock_running.return_value = True

        result = _start_service(
            checkpoint_path="/data/model",
            port=5555,
            data_config="so100",
            embodiment_tag="so100",
            denoising_steps=4,
            host="0.0.0.0",
            container_name=None,
            policy_name=None,
            timeout=5,
            use_tensorrt=True,
            trt_engine_path="gr00t_engine",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            http_server=False,
            api_token=None,
        )
        assert result["status"] == "success"
        assert "tensorrt" in result
        assert result["tensorrt"]["enabled"] is True

    @patch("time.sleep")
    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    @patch("subprocess.run")
    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    def test_start_timeout(self, mock_find, mock_run, mock_running, mock_sleep):
        mock_find.return_value = {
            "status": "success",
            "containers": [{"name": "gr00t-1", "image": "isaac-gr00t:latest", "status": "Up", "ports": ""}],
        }
        mock_run.return_value = MagicMock(returncode=0)
        mock_running.return_value = False  # never starts

        # Make time.time() advance so the timeout loop exits
        time_counter = [0.0]

        def advancing_time():
            time_counter[0] += 2.0
            return time_counter[0]

        with patch("time.time", side_effect=advancing_time):
            result = _start_service(
                checkpoint_path="/data/model",
                port=5555,
                data_config="so100",
                embodiment_tag="so100",
                denoising_steps=4,
                host="0.0.0.0",
                container_name=None,
                policy_name=None,
                timeout=3,
                use_tensorrt=False,
                trt_engine_path="engine",
                vit_dtype="fp8",
                llm_dtype="nvfp4",
                dit_dtype="fp8",
                http_server=False,
                api_token=None,
            )
        assert result["status"] == "error"
        assert "failed to start" in result["message"].lower()

    @patch("strands_robots.tools.gr00t_inference._find_gr00t_containers")
    def test_start_no_containers(self, mock_find):
        mock_find.return_value = {
            "status": "success",
            "containers": [],
        }
        result = _start_service(
            checkpoint_path="/data/model",
            port=5555,
            data_config="so100",
            embodiment_tag="so100",
            denoising_steps=4,
            host="0.0.0.0",
            container_name=None,
            policy_name=None,
            timeout=5,
            use_tensorrt=False,
            trt_engine_path="engine",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            http_server=False,
            api_token=None,
        )
        assert result["status"] == "error"
        assert "No running" in result["message"]

    @patch("time.sleep")
    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    @patch("subprocess.run")
    def test_start_explicit_container(self, mock_run, mock_running, mock_sleep):
        mock_run.return_value = MagicMock(returncode=0)
        mock_running.return_value = True

        result = _start_service(
            checkpoint_path="/data/model",
            port=5555,
            data_config="so100",
            embodiment_tag="so100",
            denoising_steps=4,
            host="0.0.0.0",
            container_name="my-explicit-container",
            policy_name=None,
            timeout=5,
            use_tensorrt=False,
            trt_engine_path="engine",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            http_server=False,
            api_token=None,
        )
        assert result["status"] == "success"
        # Verify the explicit container was used in the docker command
        cmd_args = mock_run.call_args[0][0]
        assert "my-explicit-container" in cmd_args

    @patch("time.sleep")
    @patch("strands_robots.tools.gr00t_inference._is_service_running")
    @patch("subprocess.run")
    def test_start_with_api_token(self, mock_run, mock_running, mock_sleep):
        mock_run.return_value = MagicMock(returncode=0)
        mock_running.return_value = True

        result = _start_service(
            checkpoint_path="/data/model",
            port=5555,
            data_config="so100",
            embodiment_tag="so100",
            denoising_steps=4,
            host="0.0.0.0",
            container_name="gr00t-1",
            policy_name=None,
            timeout=5,
            use_tensorrt=False,
            trt_engine_path="engine",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            http_server=False,
            api_token="secret123",
        )
        assert result["status"] == "success"
        cmd_args = mock_run.call_args[0][0]
        assert "--api-token" in cmd_args
        assert "secret123" in cmd_args

    @patch("subprocess.run")
    def test_start_docker_exec_failure(self, mock_run):
        from subprocess import CalledProcessError

        mock_run.side_effect = CalledProcessError(1, "docker exec", stderr="container stopped")

        result = _start_service(
            checkpoint_path="/data/model",
            port=5555,
            data_config="so100",
            embodiment_tag="so100",
            denoising_steps=4,
            host="0.0.0.0",
            container_name="gr00t-1",
            policy_name=None,
            timeout=5,
            use_tensorrt=False,
            trt_engine_path="engine",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            http_server=False,
            api_token=None,
        )
        assert result["status"] == "error"
