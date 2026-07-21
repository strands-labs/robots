"""Behavioural tests for the ``gr00t_inference`` service-start and dispatch paths.

These exercise the structured-error contract an LLM relies on when driving the
tool without docker actually present:

* ``action="start"`` with ``container_name=None`` must resolve a running GR00T
  container first and degrade to a structured error (never crash) when none is
  up or when container discovery itself errored.
* A failed ``docker exec`` launch surfaces as a structured error, not a raised
  ``CalledProcessError``.
* ``_is_service_running`` is a best-effort TCP probe that returns ``False`` on
  any socket failure rather than propagating.
* The container-build actions (``build_image`` / ``start_container`` /
  ``lifecycle``) gate the operator-resolved image through the allowlist at the
  dispatch boundary, so an off-allowlist ``STRANDS_GR00T_IMAGE`` fails closed
  before any docker call.
* ``action="download_checkpoint"`` forwards its resolved arguments to the
  download helper once ``hf_repo`` is supplied.

Nothing here touches docker: ``subprocess.run``, container discovery, the
socket, and the download helper are all mocked.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.tools.gr00t_inference import (
    _build_image,
    _download_checkpoint,
    _is_service_running,
    _start_service,
    gr00t_inference,
)

_MODULE = "strands_robots.tools.gr00t_inference"


class TestStartServiceContainerResolution:
    """``action="start"`` must locate a running container before exec."""

    def test_no_running_container_returns_structured_error(self):
        discovered = {
            "status": "success",
            "containers": [{"name": "gr00t-old", "status": "Exited (0) 1 hour ago"}],
        }
        with patch(f"{_MODULE}._find_gr00t_containers", return_value=discovered):
            result = gr00t_inference(action="start", checkpoint_path="/data/ckpt")
        assert result["status"] == "error"
        assert "No running GR00T containers" in result["message"]

    def test_container_discovery_error_is_propagated(self):
        discovery_error = {"status": "error", "message": "docker binary not found"}
        with patch(f"{_MODULE}._find_gr00t_containers", return_value=discovery_error):
            result = gr00t_inference(action="start", checkpoint_path="/data/ckpt")
        assert result["status"] == "error"
        assert result["message"] == "docker binary not found"

    def test_failed_docker_launch_returns_structured_error(self):
        running = {
            "status": "success",
            "containers": [{"name": "gr00t", "status": "Up 2 minutes"}],
        }
        launch_failure = subprocess.CalledProcessError(1, "docker", stderr="exec format error")
        with (
            patch(f"{_MODULE}._find_gr00t_containers", return_value=running),
            patch(f"{_MODULE}.subprocess.run", side_effect=launch_failure),
        ):
            result = gr00t_inference(action="start", checkpoint_path="/data/ckpt")
        assert result["status"] == "error"
        assert "Failed to start service" in result["message"]
        assert "exec format error" in result["message"]


class TestServiceProbe:
    """``_is_service_running`` is a best-effort TCP probe."""

    def test_true_when_port_accepts_connection(self):
        sock = MagicMock()
        sock.connect_ex.return_value = 0
        with patch(f"{_MODULE}.socket.socket", return_value=sock):
            assert _is_service_running(5555) is True

    def test_false_when_connection_refused(self):
        sock = MagicMock()
        sock.connect_ex.return_value = 111
        with patch(f"{_MODULE}.socket.socket", return_value=sock):
            assert _is_service_running(5555) is False

    def test_false_on_socket_failure(self):
        with patch(f"{_MODULE}.socket.socket", side_effect=OSError("network unreachable")):
            assert _is_service_running(5555) is False


class TestDispatchImageAllowlistGate:
    """Build/start/lifecycle reject an off-allowlist operator image up front."""

    @pytest.mark.parametrize(
        ("action", "extra"),
        [
            ("build_image", {}),
            ("start_container", {}),
            ("lifecycle", {"lifecycle": "full"}),
        ],
    )
    def test_off_allowlist_image_fails_closed(self, monkeypatch, action, extra):
        monkeypatch.setenv("STRANDS_GR00T_IMAGE", "evil/image:tag")
        result = gr00t_inference(action=action, **extra)
        assert result["status"] == "error"
        assert "allowlist" in result["message"].lower()


class TestDownloadCheckpointDispatch:
    """``action="download_checkpoint"`` forwards to the download helper."""

    def test_forwards_resolved_arguments(self):
        sentinel = {"status": "success", "skipped": False, "message": "downloaded"}
        with patch(f"{_MODULE}._download_checkpoint", return_value=sentinel) as mock_dl:
            result = gr00t_inference(
                action="download_checkpoint",
                hf_repo="nvidia/GR00T-N1.5-3B",
                hf_subfolder="ckpt",
            )
        assert result is sentinel
        mock_dl.assert_called_once()
        assert mock_dl.call_args.kwargs["hf_repo"] == "nvidia/GR00T-N1.5-3B"
        assert mock_dl.call_args.kwargs["hf_subfolder"] == "ckpt"


class TestStartServiceResponseSurface:
    """A successful start advertises the transport-specific fields it enabled.

    An operator/LLM reading the start response relies on it to reflect exactly
    what was launched: when TensorRT is on, the compiled-engine config is echoed
    back; when the HTTP transport is on, the REST endpoint URL is surfaced so the
    caller can POST observations without reconstructing ``host:port/act`` itself.
    """

    @patch(f"{_MODULE}._is_service_running", return_value=True)
    @patch(f"{_MODULE}.subprocess.run")
    def test_tensorrt_and_http_fields_surface_in_response(self, mock_run, _mock_is_running):
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        result = _start_service(
            checkpoint_path="/data/checkpoints/so100",
            port=8000,
            data_config="so100",
            embodiment_tag="so100",
            denoising_steps=4,
            host="127.0.0.1",
            container_name="gr00t",
            policy_name=None,
            timeout=2,
            use_tensorrt=True,
            trt_engine_path="/engines/so100.plan",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            http_server=True,
            api_token=None,
            protocol="n1.5",
            use_sim_policy_wrapper=False,
        )
        assert result["status"] == "success"
        # HTTP transport surfaces a ready-to-POST endpoint URL.
        assert result["protocol"] == "HTTP"
        assert result["endpoint"] == "http://127.0.0.1:8000/act"
        # TensorRT surfaces the exact compiled-engine config that was requested.
        assert result["tensorrt"] == {
            "enabled": True,
            "engine_path": "/engines/so100.plan",
            "vit_dtype": "fp8",
            "llm_dtype": "nvfp4",
            "dit_dtype": "fp8",
        }

    @patch(f"{_MODULE}._is_service_running", return_value=True)
    @patch(f"{_MODULE}.subprocess.run")
    def test_zmq_start_omits_endpoint_and_tensorrt(self, mock_run, _mock_is_running):
        """Fields for transports/accelerators that were NOT enabled must be absent."""
        mock_run.return_value.stdout = ""
        result = _start_service(
            checkpoint_path="/cp",
            port=5555,
            data_config="so100",
            embodiment_tag="so100",
            denoising_steps=4,
            host="127.0.0.1",
            container_name="gr00t",
            policy_name=None,
            timeout=2,
            use_tensorrt=False,
            trt_engine_path="x",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            http_server=False,
            api_token=None,
            protocol="n1.5",
            use_sim_policy_wrapper=False,
        )
        assert result["status"] == "success"
        assert "endpoint" not in result
        assert "tensorrt" not in result


class TestBuildImageGuards:
    """``_build_image`` re-asserts URL/tag guards before touching git/bash.

    The dispatch boundary already validates these, but the private helper is
    reachable by operators/tests, so it fails closed on a malformed ref rather
    than passing attacker-influenced input into a ``git clone`` subprocess.
    """

    def test_malformed_tag_rejected_before_any_subprocess(self):
        with patch(f"{_MODULE}.subprocess.run") as mock_run:
            result = _build_image(
                repo_url="https://github.com/NVIDIA/Isaac-GR00T",
                repo_tag="; rm -rf /",
                image_name="isaac-gr00t:local",
                force=False,
            )
        assert result["status"] == "error"
        assert "not a valid git ref" in result["message"]
        mock_run.assert_not_called()


class TestDownloadCheckpointErrors:
    """``_download_checkpoint`` degrades HF failures into a structured error."""

    def test_snapshot_download_failure_returns_structured_error(self, tmp_path):
        fake_hub = MagicMock()
        fake_hub.snapshot_download.side_effect = RuntimeError("401 gated repo")
        with (
            patch(f"{_MODULE}._checkpoints_dir", return_value=tmp_path),
            patch(f"{_MODULE}.require_optional", return_value=fake_hub),
        ):
            result = _download_checkpoint(
                hf_repo="nvidia/GR00T-N1.5-3B",
                hf_subfolder=None,
                hf_local_dir=None,
                hf_token=None,
                force=False,
            )
        assert result["status"] == "error"
        assert "Failed to download" in result["message"]
        assert "401 gated repo" in result["message"]


class TestStartServicePolling:
    """The start loop polls until the service is reachable, then degrades cleanly."""

    @patch(f"{_MODULE}.time.sleep")
    @patch(f"{_MODULE}._is_service_running", side_effect=[False, True])
    @patch(f"{_MODULE}.subprocess.run")
    def test_retries_until_service_ready(self, mock_run, _mock_probe, mock_sleep):
        mock_run.return_value.stdout = ""
        result = _start_service(
            checkpoint_path="/cp",
            port=5555,
            data_config="so100",
            embodiment_tag="so100",
            denoising_steps=4,
            host="127.0.0.1",
            container_name="gr00t",
            policy_name=None,
            timeout=5,
            use_tensorrt=False,
            trt_engine_path="x",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            http_server=False,
            api_token=None,
            protocol="n1.5",
            use_sim_policy_wrapper=False,
        )
        assert result["status"] == "success"
        # The probe returned False once, so the loop slept before re-probing.
        mock_sleep.assert_called_once_with(1)

    @patch(f"{_MODULE}.subprocess.run", side_effect=ValueError("bad argv encoding"))
    def test_unexpected_exception_returns_structured_error(self, _mock_run):
        result = _start_service(
            checkpoint_path="/cp",
            port=5555,
            data_config="so100",
            embodiment_tag="so100",
            denoising_steps=4,
            host="127.0.0.1",
            container_name="gr00t",
            policy_name=None,
            timeout=2,
            use_tensorrt=False,
            trt_engine_path="x",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            http_server=False,
            api_token=None,
            protocol="n1.5",
            use_sim_policy_wrapper=False,
        )
        assert result["status"] == "error"
        assert "Unexpected error" in result["message"]
        assert "bad argv encoding" in result["message"]
