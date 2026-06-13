"""Tests for the ``gr00t_inference`` tool's command builder.

Establishes the testing pattern for shell-out tools in this repo: mock
``subprocess.run`` and ``_is_service_running`` so the start path returns
without actually invoking docker. Tests assert on the exact ``argv`` the
tool would have ``docker exec``'d.

Coverage:

* N1.5 default - legacy ``inference_service.py`` entrypoint with
  ``--data-config`` + ``--denoising-steps``.
* N1.6 - same entrypoint as N1.5 (the wire format diverged but the
  server CLI didn't).
* N1.7 - new ``python -m gr00t.eval.run_gr00t_server`` entrypoint with
  no ``--data-config`` / ``--denoising-steps`` and an optional
  ``--use-sim-policy-wrapper``.
* Unknown protocol → structured error.
* Optional flags (TensorRT, ``--http-server``, ``--api-token``) carry
  through to every protocol.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from strands_robots.tools.gr00t_inference import (
    _build_inference_command,
    _start_service,
    gr00t_inference,
)


def _common_kwargs(**overrides: Any) -> dict[str, Any]:
    """Default args for ``_build_inference_command`` - keeps each test focused."""
    base: dict[str, Any] = {
        "container_name": "gr00t",
        "checkpoint_path": "/data/checkpoints/model",
        "port": 5555,
        "host": "0.0.0.0",
        "data_config": "libero_panda",
        "embodiment_tag": "libero_sim",
        "denoising_steps": 4,
        "http_server": False,
        "use_tensorrt": False,
        "trt_engine_path": "gr00t_engine",
        "vit_dtype": "fp8",
        "llm_dtype": "nvfp4",
        "dit_dtype": "fp8",
        "api_token": None,
        "protocol": "n1.5",
        "use_sim_policy_wrapper": False,
    }
    base.update(overrides)
    return base


# Command builder


class TestBuildInferenceCommand:
    def test_n15_legacy_entrypoint(self):
        cmd = _build_inference_command(**_common_kwargs(protocol="n1.5"))
        # Legacy script must be the python target.
        assert "/opt/Isaac-GR00T/scripts/inference_service.py" in cmd
        # Includes the deprecated-in-n1.7 flags.
        assert "--data-config" in cmd
        assert "libero_panda" in cmd
        assert "--denoising-steps" in cmd
        assert "4" in cmd
        # Does NOT include n1.7-only flags.
        assert "-m" not in cmd
        assert "gr00t.eval.run_gr00t_server" not in cmd
        assert "--use-sim-policy-wrapper" not in cmd

    def test_n16_uses_legacy_entrypoint(self):
        """N1.6 still uses inference_service.py - only the wire format diverged."""
        cmd = _build_inference_command(**_common_kwargs(protocol="n1.6"))
        assert "/opt/Isaac-GR00T/scripts/inference_service.py" in cmd
        assert "--data-config" in cmd
        assert "--denoising-steps" in cmd

    def test_n17_module_entrypoint(self):
        cmd = _build_inference_command(**_common_kwargs(protocol="n1.7"))
        # New entrypoint: python -m gr00t.eval.run_gr00t_server
        assert "-m" in cmd
        assert "gr00t.eval.run_gr00t_server" in cmd
        # Legacy script must NOT be invoked under n1.7.
        assert "/opt/Isaac-GR00T/scripts/inference_service.py" not in cmd
        # Flags removed from n1.7's CLI surface (server reads from checkpoint).
        assert "--data-config" not in cmd
        assert "--denoising-steps" not in cmd
        # Sim policy wrapper is opt-in.
        assert "--use-sim-policy-wrapper" not in cmd
        # Regression: the n1.7 entrypoint does NOT take ``--server``.
        # Passing it makes ``tyro`` reject the invocation with
        # "Unrecognized options: --server" and the inference process
        # exits before binding the port. See discussion on #148-F3.
        assert "--server" not in cmd

    def test_n15_keeps_server_flag(self):
        """Conversely, the legacy entrypoint *does* require ``--server``."""
        cmd = _build_inference_command(**_common_kwargs(protocol="n1.5"))
        assert "--server" in cmd
        cmd = _build_inference_command(**_common_kwargs(protocol="n1.6"))
        assert "--server" in cmd

    def test_n17_with_sim_policy_wrapper(self):
        cmd = _build_inference_command(**_common_kwargs(protocol="n1.7", use_sim_policy_wrapper=True))
        assert "--use-sim-policy-wrapper" in cmd

    def test_n17_ignores_use_sim_policy_wrapper_under_n15(self):
        """The flag doesn't exist on the legacy entrypoint - silently dropped."""
        cmd = _build_inference_command(**_common_kwargs(protocol="n1.5", use_sim_policy_wrapper=True))
        assert "--use-sim-policy-wrapper" not in cmd

    def test_shared_required_flags_present_on_all_protocols(self):
        for protocol in ("n1.5", "n1.6", "n1.7"):
            cmd = _build_inference_command(**_common_kwargs(protocol=protocol))
            # ``--server`` is N1.5/N1.6-only — see test_n17_module_entrypoint
            # for the regression detail.
            if protocol in ("n1.5", "n1.6"):
                assert "--server" in cmd, protocol
            else:
                assert "--server" not in cmd, protocol
            assert "--model-path" in cmd, protocol
            assert "/data/checkpoints/model" in cmd, protocol
            assert "--port" in cmd, protocol
            assert "5555" in cmd, protocol
            assert "--host" in cmd, protocol
            assert "0.0.0.0" in cmd, protocol
            assert "--embodiment-tag" in cmd, protocol
            assert "libero_sim" in cmd, protocol

    def test_http_server_flag_carries_across_protocols(self):
        for protocol in ("n1.5", "n1.7"):
            cmd = _build_inference_command(**_common_kwargs(protocol=protocol, http_server=True))
            assert "--http-server" in cmd, protocol

    def test_tensorrt_flags_carry_across_protocols(self):
        for protocol in ("n1.5", "n1.7"):
            cmd = _build_inference_command(
                **_common_kwargs(
                    protocol=protocol,
                    use_tensorrt=True,
                    trt_engine_path="/engines/x",
                    vit_dtype="fp16",
                    llm_dtype="fp8",
                    dit_dtype="fp16",
                )
            )
            assert "--use-tensorrt" in cmd, protocol
            assert "/engines/x" in cmd, protocol
            assert "fp16" in cmd, protocol
            assert "fp8" in cmd, protocol

    def test_api_token_carries_across_protocols(self):
        for protocol in ("n1.5", "n1.7"):
            cmd = _build_inference_command(**_common_kwargs(protocol=protocol, api_token="sek"))
            assert "--api-token" in cmd
            assert "sek" in cmd

    def test_api_token_omitted_when_none(self):
        cmd = _build_inference_command(**_common_kwargs(api_token=None))
        assert "--api-token" not in cmd


# Top-level dispatcher


class TestProtocolValidation:
    def test_unknown_protocol_returns_structured_error(self):
        result = gr00t_inference(action="start", checkpoint_path="/cp", protocol="n2.0")
        assert result["status"] == "error"
        assert "Unknown protocol" in result["message"]
        # Error must enumerate the valid set so the caller can fix the call.
        assert "n1.5" in result["message"]
        assert "n1.7" in result["message"]


# _start_service end-to-end with subprocess mocked


class TestStartServiceEndToEnd:
    @patch("strands_robots.tools.gr00t_inference._is_service_running", return_value=True)
    @patch("strands_robots.tools.gr00t_inference.subprocess.run")
    def test_n17_start_succeeds_and_reports_server_protocol(self, mock_run, _mock_is_running):
        """Full start path with protocol='n1.7' must invoke run_gr00t_server."""
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        result = _start_service(
            checkpoint_path="/data/checkpoints/libero_spatial",
            port=8000,
            data_config="libero_panda",
            embodiment_tag="libero_sim",
            denoising_steps=4,
            host="0.0.0.0",
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
            protocol="n1.7",
            use_sim_policy_wrapper=True,
        )
        assert result["status"] == "success"
        assert result["server_protocol"] == "n1.7"
        assert result["use_sim_policy_wrapper"] is True
        # The legacy-only fields must NOT appear in the n1.7 response.
        assert "data_config" not in result
        assert "denoising_steps" not in result

        # Verify the actual subprocess call.
        argv = mock_run.call_args.args[0]
        assert "gr00t.eval.run_gr00t_server" in argv
        assert "--use-sim-policy-wrapper" in argv
        assert "--data-config" not in argv

    @patch("strands_robots.tools.gr00t_inference._is_service_running", return_value=True)
    @patch("strands_robots.tools.gr00t_inference.subprocess.run")
    def test_n15_start_preserves_legacy_response_fields(self, mock_run, _mock_is_running):
        """Default protocol path must keep the existing response shape."""
        mock_run.return_value.stdout = ""
        result = _start_service(
            checkpoint_path="/cp",
            port=5555,
            data_config="so100",
            embodiment_tag="so100",
            denoising_steps=4,
            host="0.0.0.0",
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
        assert result["server_protocol"] == "n1.5"
        assert result["data_config"] == "so100"
        assert result["denoising_steps"] == 4
        # Wire protocol stays "ZMQ" / "HTTP" (back-compat with pre-fix callers).
        assert result["protocol"] == "ZMQ"

    @patch("strands_robots.tools.gr00t_inference._is_service_running", return_value=False)
    @patch("strands_robots.tools.gr00t_inference.subprocess.run")
    def test_timeout_returns_error(self, mock_run, _mock_is_running):
        mock_run.return_value.stdout = ""
        result = _start_service(
            checkpoint_path="/cp",
            port=5555,
            data_config="so100",
            embodiment_tag="so100",
            denoising_steps=4,
            host="0.0.0.0",
            container_name="gr00t",
            policy_name=None,
            timeout=0,  # don't actually sleep
            use_tensorrt=False,
            trt_engine_path="x",
            vit_dtype="fp8",
            llm_dtype="nvfp4",
            dit_dtype="fp8",
            http_server=False,
            api_token=None,
            protocol="n1.7",
            use_sim_policy_wrapper=False,
        )
        assert result["status"] == "error"
        assert "failed to start" in result["message"]


@pytest.mark.parametrize("protocol", ["n1.5", "n1.6", "n1.7"])
class TestSignatureBackCompat:
    """Existing callers that don't pass ``protocol=`` must continue to work
    (default = ``n1.5``)."""

    def test_default_protocol_is_n15(self, protocol):
        # Indirect: passing nothing should match an explicit n1.5 build.
        if protocol != "n1.5":
            return
        cmd_default = _build_inference_command(**_common_kwargs(protocol="n1.5"))
        assert "/opt/Isaac-GR00T/scripts/inference_service.py" in cmd_default


# Container lifecycle (#148-F3 wider)
#
# Each new action is idempotent and uses subprocess.run to talk to docker /
# git. Tests mock subprocess.run + huggingface_hub so nothing actually
# runs - we assert on the captured argv and on the structured response
# dict's idempotency markers (``skipped`` true/false).


from unittest.mock import MagicMock  # noqa: E402

from strands_robots.tools.gr00t_inference import (  # noqa: E402
    _DEFAULT_REPO_URL,
    _build_image,
    _container_state,
    _download_checkpoint,
    _image_exists,
    _lifecycle,
    _remove_container,
    _start_container,
)

# Helpers


def _docker_inspect_returncode(rc: int):
    """Return a MagicMock that mimics ``subprocess.run`` returning ``rc``."""
    m = MagicMock()
    m.returncode = rc
    m.stdout = "running\n" if rc == 0 else ""
    m.stderr = ""
    return m


def _patch_subprocess_run(side_effect=None):
    """Patch the module-level ``subprocess.run`` used by the lifecycle helpers."""
    return patch("strands_robots.tools.gr00t_inference.subprocess.run", side_effect=side_effect)


# _image_exists / _container_state primitives


class TestImageAndContainerProbes:
    def test_image_exists_returns_true_on_zero_rc(self):
        with _patch_subprocess_run(side_effect=lambda *a, **kw: _docker_inspect_returncode(0)):
            assert _image_exists("gr00t:latest") is True

    def test_image_exists_returns_false_on_nonzero_rc(self):
        with _patch_subprocess_run(side_effect=lambda *a, **kw: _docker_inspect_returncode(1)):
            assert _image_exists("gr00t:latest") is False

    def test_image_exists_handles_missing_docker_binary(self):
        with _patch_subprocess_run(side_effect=FileNotFoundError("docker")):
            assert _image_exists("gr00t:latest") is False

    def test_container_state_running(self):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "running\n"
        with _patch_subprocess_run(side_effect=lambda *a, **kw: result):
            assert _container_state("gr00t") == "running"

    def test_container_state_absent(self):
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        with _patch_subprocess_run(side_effect=lambda *a, **kw: result):
            assert _container_state("ghost") == "absent"


# build_image


class TestBuildImage:
    # repo_url is operator-resolved + allowlisted; the canonical default is
    # the only URL these tests may use. source_dir is gone -- the clone dest is
    # fixed to _isaac_gr00t_dir(), patched here onto tmp_path.
    _OK_URL = _DEFAULT_REPO_URL

    def test_skips_when_image_exists_and_not_force(self):
        """Idempotency: existing image short-circuits to success without
        touching git or docker."""
        with patch(
            "strands_robots.tools.gr00t_inference._image_exists",
            return_value=True,
        ):
            result = _build_image(
                repo_url=self._OK_URL,
                repo_tag="n1.7-release",
                image_name="gr00t:latest",
                force=False,
            )
        assert result["status"] == "success"
        assert result["skipped"] is True
        assert "already exists" in result["message"]

    def test_force_rebuilds_even_when_image_exists(self, tmp_path):
        """force=True must run the full clone + build path regardless."""
        runs: list[list[str]] = []

        def fake_run(cmd, *a, **kw):
            runs.append(list(cmd))
            return MagicMock(stdout="", stderr="", returncode=0)

        src = tmp_path / "Isaac-GR00T"
        (src / "docker").mkdir(parents=True)
        (src / "docker" / "build.sh").write_text("#!/bin/bash\necho ok")
        (src / ".git").mkdir()  # so the update branch is taken
        with (
            patch("strands_robots.tools.gr00t_inference._image_exists", return_value=True),
            patch("strands_robots.tools.gr00t_inference._isaac_gr00t_dir", return_value=src),
            patch("strands_robots.tools.gr00t_inference.subprocess.run", side_effect=fake_run),
        ):
            _build_image(
                repo_url=self._OK_URL,
                repo_tag="n1.7-release",
                image_name="gr00t:latest",
                force=True,
            )

        # Verify a build.sh invocation was attempted.
        assert any("bash" in cmd[0] and "build.sh" in cmd[1] for cmd in runs)

    def test_clone_path_used_when_dest_missing(self, tmp_path):
        runs: list[list[str]] = []

        def fake_run(cmd, *a, **kw):
            runs.append(list(cmd))
            # Simulate the clone creating dest/.git and dest/docker/build.sh
            # so the build step finds them.
            if cmd[:2] == ["git", "clone"]:
                dest = Path(cmd[-1])
                (dest / "docker").mkdir(parents=True, exist_ok=True)
                (dest / "docker" / "build.sh").write_text("#!/bin/bash\necho ok")
                (dest / ".git").mkdir(exist_ok=True)
            return MagicMock(stdout="", stderr="", returncode=0)

        src = tmp_path / "fresh-clone"
        with (
            patch("strands_robots.tools.gr00t_inference._image_exists", return_value=False),
            patch("strands_robots.tools.gr00t_inference._isaac_gr00t_dir", return_value=src),
            patch("strands_robots.tools.gr00t_inference.subprocess.run", side_effect=fake_run),
        ):
            result = _build_image(
                repo_url=self._OK_URL,
                repo_tag="n1.7-release",
                image_name="gr00t:latest",
                force=False,
            )
        assert result["status"] == "success"
        # Must include a `git clone --depth 1 --branch <tag>` invocation.
        assert any("clone" in cmd and "--branch" in cmd and "n1.7-release" in cmd for cmd in runs)

    def test_build_failure_propagates_stderr(self, tmp_path):
        src = tmp_path / "Isaac-GR00T"
        (src / "docker").mkdir(parents=True)
        (src / "docker" / "build.sh").write_text("")
        (src / ".git").mkdir()

        def fake_run(cmd, *a, **kw):
            if "build.sh" in " ".join(cmd):
                raise subprocess.CalledProcessError(1, cmd, stderr="boom")
            return MagicMock(stdout="", stderr="", returncode=0)

        with (
            patch("strands_robots.tools.gr00t_inference._image_exists", return_value=False),
            patch("strands_robots.tools.gr00t_inference._isaac_gr00t_dir", return_value=src),
            patch("strands_robots.tools.gr00t_inference.subprocess.run", side_effect=fake_run),
        ):
            result = _build_image(
                repo_url=self._OK_URL,
                repo_tag="n1.7-release",
                image_name="gr00t:latest",
                force=False,
            )
        assert result["status"] == "error"
        assert "boom" in result["message"]


# download_checkpoint


class TestDownloadCheckpoint:
    def test_skips_when_local_dir_populated_and_not_force(self, tmp_path):
        local = tmp_path / "ckpt"
        local.mkdir()
        (local / "config.json").write_text("{}")  # populated
        result = _download_checkpoint(
            hf_repo="nvidia/foo",
            hf_subfolder=None,
            hf_local_dir=str(local),
            hf_token=None,
            force=False,
        )
        assert result["status"] == "success"
        assert result["skipped"] is True

    def test_force_redownloads(self, tmp_path):
        local = tmp_path / "ckpt"
        local.mkdir()
        (local / "config.json").write_text("{}")

        fake_hub = MagicMock()
        with patch("strands_robots.tools.gr00t_inference.require_optional", return_value=fake_hub):
            result = _download_checkpoint(
                hf_repo="nvidia/foo",
                hf_subfolder="bar",
                hf_local_dir=str(local),
                hf_token=None,
                force=True,
            )
        assert result["status"] == "success"
        assert result["skipped"] is False
        fake_hub.snapshot_download.assert_called_once()
        # allow_patterns should be ['bar/*'] when subfolder set.
        kwargs = fake_hub.snapshot_download.call_args.kwargs
        assert kwargs["allow_patterns"] == ["bar/*"]
        assert kwargs["repo_id"] == "nvidia/foo"

    def test_token_resolution_from_kwarg(self, tmp_path):
        fake_hub = MagicMock()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
            with patch(
                "strands_robots.tools.gr00t_inference.require_optional",
                return_value=fake_hub,
            ):
                _download_checkpoint(
                    hf_repo="r",
                    hf_subfolder=None,
                    hf_local_dir=str(tmp_path / "new"),
                    hf_token="explicit-token",
                    force=False,
                )
        kwargs = fake_hub.snapshot_download.call_args.kwargs
        assert kwargs["token"] == "explicit-token"

    def test_token_falls_back_to_env(self, tmp_path):
        fake_hub = MagicMock()
        with patch.dict(os.environ, {"HF_TOKEN": "env-token"}, clear=False):
            with patch(
                "strands_robots.tools.gr00t_inference.require_optional",
                return_value=fake_hub,
            ):
                _download_checkpoint(
                    hf_repo="r",
                    hf_subfolder=None,
                    hf_local_dir=str(tmp_path / "new"),
                    hf_token=None,
                    force=False,
                )
        assert fake_hub.snapshot_download.call_args.kwargs["token"] == "env-token"

    def test_huggingface_hub_missing_returns_structured_error(self, tmp_path):
        with patch(
            "strands_robots.tools.gr00t_inference.require_optional",
            side_effect=ImportError("'huggingface_hub' is required"),
        ):
            result = _download_checkpoint(
                hf_repo="r",
                hf_subfolder=None,
                hf_local_dir=str(tmp_path / "new"),
                hf_token=None,
                force=False,
            )
        assert result["status"] == "error"
        assert "huggingface_hub" in result["message"]


# start_container


class TestStartContainer:
    def test_skips_when_already_running_and_not_force(self):
        with patch(
            "strands_robots.tools.gr00t_inference._container_state",
            return_value="running",
        ):
            result = _start_container(
                image_name="gr00t:latest",
                container_name="gr00t",
                port=8000,
                volumes=None,
                hf_token=None,
                container_command="tail -f /dev/null",
                hf_local_dir=None,
                force=False,
            )
        assert result["status"] == "success"
        assert result["skipped"] is True

    def test_recreates_when_force(self):
        runs: list[list[str]] = []

        def fake_run(cmd, *a, **kw):
            runs.append(list(cmd))
            return MagicMock(stdout="", stderr="", returncode=0)

        with patch(
            "strands_robots.tools.gr00t_inference._container_state",
            side_effect=["running", "running", "absent"],  # first probe → running
        ):
            with patch("strands_robots.tools.gr00t_inference.subprocess.run", side_effect=fake_run):
                _start_container(
                    image_name="gr00t:latest",
                    container_name="gr00t",
                    port=8000,
                    volumes=None,
                    hf_token=None,
                    container_command="tail -f /dev/null",
                    hf_local_dir=None,
                    force=True,
                )
        # Must have issued a `docker rm -f gr00t` and then `docker run -d ... gr00t:latest tail -f /dev/null`.
        assert any(c[:3] == ["docker", "rm", "-f"] for c in runs)
        run_cmds = [c for c in runs if c[:2] == ["docker", "run"]]
        assert run_cmds
        run_cmd = run_cmds[0]
        assert "--gpus" in run_cmd and "all" in run_cmd
        assert "--ipc=host" in run_cmd
        assert "--name" in run_cmd and "gr00t" in run_cmd
        assert "8000:8000" in run_cmd

    def test_volumes_default_includes_checkpoints_and_hf_cache(self):
        runs: list[list[str]] = []

        def fake_run(cmd, *a, **kw):
            runs.append(list(cmd))
            return MagicMock(stdout="", stderr="", returncode=0)

        with patch(
            "strands_robots.tools.gr00t_inference._container_state",
            return_value="absent",
        ):
            with patch("strands_robots.tools.gr00t_inference.subprocess.run", side_effect=fake_run):
                result = _start_container(
                    image_name="gr00t:latest",
                    container_name="gr00t",
                    port=8000,
                    volumes=None,
                    hf_token=None,
                    container_command="tail -f /dev/null",
                    hf_local_dir="/data/cp",
                    force=False,
                )
        assert result["status"] == "success"
        # Default volumes include the checkpoint mount and the HF cache mount.
        argv = next(c for c in runs if c[:2] == ["docker", "run"])
        joined = " ".join(argv)
        assert "/data/cp:/data/checkpoints" in joined
        assert "huggingface" in joined  # HF cache path

    def test_token_propagated_as_env(self):
        runs: list[list[str]] = []

        def fake_run(cmd, *a, **kw):
            runs.append(list(cmd))
            return MagicMock(stdout="", stderr="", returncode=0)

        with patch(
            "strands_robots.tools.gr00t_inference._container_state",
            return_value="absent",
        ):
            with patch("strands_robots.tools.gr00t_inference.subprocess.run", side_effect=fake_run):
                _start_container(
                    image_name="gr00t:latest",
                    container_name="gr00t",
                    port=8000,
                    volumes={"/cp": "/data/checkpoints"},
                    hf_token="abc123",
                    container_command="tail -f /dev/null",
                    hf_local_dir=None,
                    force=False,
                )
        argv = next(c for c in runs if c[:2] == ["docker", "run"])
        # -e HF_TOKEN=abc123 must appear in the run command.
        assert any(e == "HF_TOKEN=abc123" for e in argv)

    def test_unhealthy_state_without_force_errors(self):
        with patch(
            "strands_robots.tools.gr00t_inference._container_state",
            return_value="exited",
        ):
            result = _start_container(
                image_name="gr00t:latest",
                container_name="gr00t",
                port=8000,
                volumes=None,
                hf_token=None,
                container_command="tail -f /dev/null",
                hf_local_dir=None,
                force=False,
            )
        assert result["status"] == "error"
        assert "force=True" in result["message"]


# remove_container


class TestRemoveContainer:
    def test_absent_is_idempotent_success(self):
        with patch(
            "strands_robots.tools.gr00t_inference._container_state",
            return_value="absent",
        ):
            result = _remove_container(name="ghost", remove_volumes=False)
        assert result["status"] == "success"
        assert result["skipped"] is True

    def test_running_container_removed(self):
        runs: list[list[str]] = []

        def fake_run(cmd, *a, **kw):
            runs.append(list(cmd))
            return MagicMock(stdout="", stderr="", returncode=0)

        with patch(
            "strands_robots.tools.gr00t_inference._container_state",
            return_value="running",
        ):
            with patch("strands_robots.tools.gr00t_inference.subprocess.run", side_effect=fake_run):
                result = _remove_container(name="gr00t", remove_volumes=True)
        assert result["status"] == "success"
        assert ["docker", "rm", "-f", "-v", "gr00t"] in runs


# lifecycle


class TestLifecycle:
    def test_unknown_phase_errors(self):
        # Going through the dispatcher because lifecycle="weird" should be
        # rejected by _lifecycle, not by the @tool wrapper.
        result = _lifecycle(
            phase="weird",
            **_lifecycle_default_kwargs(),
        )
        assert result["status"] == "error"
        assert "weird" in result["message"]

    def test_full_requires_hf_repo(self):
        result = _lifecycle(
            phase="full",
            **{**_lifecycle_default_kwargs(), "hf_repo": None},
        )
        assert result["status"] == "error"
        assert "hf_repo" in result["message"]

    def test_full_chains_all_steps_in_order(self, tmp_path):
        """phase='full' must call build → download → start_container → start in order."""
        order: list[str] = []

        def fake_build(**kw):
            order.append("build_image")
            return {"status": "success", "image_name": kw["image_name"], "skipped": False}

        def fake_download(**kw):
            order.append("download_checkpoint")
            return {
                "status": "success",
                "local_dir": str(tmp_path / "cp"),
                "skipped": False,
                "hf_repo": kw["hf_repo"],
            }

        def fake_start_container(**kw):
            order.append("start_container")
            return {
                "status": "success",
                "container_name": kw["container_name"] or "gr00t",
                "skipped": False,
            }

        def fake_start_service(**kw):
            order.append("start")
            return {"status": "success", "message": "service up"}

        with patch("strands_robots.tools.gr00t_inference._build_image", side_effect=fake_build):
            with patch("strands_robots.tools.gr00t_inference._download_checkpoint", side_effect=fake_download):
                with patch(
                    "strands_robots.tools.gr00t_inference._start_container",
                    side_effect=fake_start_container,
                ):
                    with patch(
                        "strands_robots.tools.gr00t_inference._start_service",
                        side_effect=fake_start_service,
                    ):
                        result = _lifecycle(
                            phase="full",
                            **{
                                **_lifecycle_default_kwargs(),
                                "hf_repo": "nvidia/foo",
                                "hf_subfolder": "libero_spatial",
                            },
                        )
        assert order == ["build_image", "download_checkpoint", "start_container", "start"]
        assert result["status"] == "success"
        # Steps array must include each sub-step's result for the agent to inspect.
        assert [s["step"] for s in result["steps"]] == [
            "build_image",
            "download_checkpoint",
            "start_container",
            "start",
        ]

    def test_full_aborts_on_build_failure(self):
        with patch(
            "strands_robots.tools.gr00t_inference._build_image",
            return_value={"status": "error", "message": "no docker"},
        ):
            result = _lifecycle(
                phase="full",
                **{**_lifecycle_default_kwargs(), "hf_repo": "nvidia/foo"},
            )
        assert result["status"] == "error"
        assert "build_image failed" in result["message"]
        # Only the failing step should be in the trail; downstream not attempted.
        assert [s["step"] for s in result["steps"]] == ["build_image"]

    def test_full_auto_resolves_in_container_checkpoint_path_from_subfolder(self, tmp_path):
        captured: dict[str, Any] = {}

        def fake_start_service(**kw):
            captured.update(kw)
            return {"status": "success", "message": "ok"}

        with patch(
            "strands_robots.tools.gr00t_inference._build_image",
            return_value={"status": "success", "image_name": "x", "skipped": True},
        ):
            with patch(
                "strands_robots.tools.gr00t_inference._download_checkpoint",
                return_value={
                    "status": "success",
                    "local_dir": str(tmp_path / "cp"),
                    "skipped": False,
                    "hf_repo": "nvidia/foo",
                },
            ):
                with patch(
                    "strands_robots.tools.gr00t_inference._start_container",
                    return_value={"status": "success", "container_name": "gr00t", "skipped": False},
                ):
                    with patch(
                        "strands_robots.tools.gr00t_inference._start_service",
                        side_effect=fake_start_service,
                    ):
                        _lifecycle(
                            phase="full",
                            **{
                                **_lifecycle_default_kwargs(),
                                "hf_repo": "nvidia/foo",
                                "hf_subfolder": "libero_spatial",
                                "checkpoint_path": None,  # let the lifecycle resolve it
                            },
                        )
        assert captured["checkpoint_path"] == "/data/checkpoints/libero_spatial"

    def test_teardown_removes_container(self):
        with patch(
            "strands_robots.tools.gr00t_inference._remove_container",
            return_value={"status": "success", "skipped": False, "message": "removed"},
        ) as mock_rm:
            result = _lifecycle(
                phase="teardown",
                **{**_lifecycle_default_kwargs(), "container_name": "gr00t", "remove_volumes": True},
            )
        assert result["status"] == "success"
        mock_rm.assert_called_once_with(name="gr00t", remove_volumes=True)


def _lifecycle_default_kwargs() -> dict[str, Any]:
    """Minimal kwargs to invoke _lifecycle in tests."""
    return {
        "repo_url": _DEFAULT_REPO_URL,
        "repo_tag": "n1.7-release",
        "image_name": "gr00t:latest",
        "hf_repo": "nvidia/foo",
        "hf_subfolder": None,
        "hf_local_dir": None,
        "hf_token": None,
        "container_name": None,
        "volumes": None,
        "container_command": "tail -f /dev/null",
        "remove_volumes": False,
        "force": False,
        "checkpoint_path": "/data/checkpoints/m",
        "policy_name": None,
        "port": 8000,
        "data_config": "libero_panda",
        "embodiment_tag": "libero_sim",
        "denoising_steps": 4,
        "host": "0.0.0.0",
        "timeout": 1,
        "use_tensorrt": False,
        "trt_engine_path": "x",
        "vit_dtype": "fp8",
        "llm_dtype": "nvfp4",
        "dit_dtype": "fp8",
        "http_server": False,
        "api_token": None,
        "protocol": "n1.7",
        "use_sim_policy_wrapper": True,
    }


# Top-level dispatcher reaches the new actions


class TestActionDispatch:
    """Verify the @tool wrapper routes the new ``action=`` values correctly."""

    def test_build_image_dispatched(self):
        with patch(
            "strands_robots.tools.gr00t_inference._build_image",
            return_value={"status": "success", "skipped": True, "message": "ok"},
        ) as mock:
            # image_name is no longer an agent parameter; the default
            # operator image (gr00t:latest) is allowlisted.
            result = gr00t_inference(action="build_image")
        assert result["status"] == "success"
        mock.assert_called_once()

    def test_download_checkpoint_requires_hf_repo(self):
        result = gr00t_inference(action="download_checkpoint")
        assert result["status"] == "error"
        assert "hf_repo" in result["message"]

    def test_start_container_dispatched(self):
        with patch(
            "strands_robots.tools.gr00t_inference._start_container",
            return_value={"status": "success", "skipped": True, "message": "ok"},
        ) as mock:
            result = gr00t_inference(action="start_container", port=8000)
        assert result["status"] == "success"
        mock.assert_called_once()

    def test_lifecycle_dispatched(self):
        with patch(
            "strands_robots.tools.gr00t_inference._lifecycle",
            return_value={"status": "success", "phase": "full", "steps": [], "message": "ok"},
        ) as mock:
            gr00t_inference(action="lifecycle", lifecycle="full", hf_repo="nvidia/foo")
        mock.assert_called_once()
        # The wrapper must forward the lifecycle phase as `phase=`.
        kwargs = mock.call_args.kwargs
        assert kwargs["phase"] == "full"
