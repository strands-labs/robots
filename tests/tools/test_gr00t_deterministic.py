"""Tests for the ``deterministic=True`` GR00T lifecycle flag (#1790).

The flag mounts the packaged determinism wrapper
(``strands_robots/policies/groot/server_wrapper.py``) read-only into the
container and swaps the N1.7 server entrypoint to ``python /srv_wrap.py``,
so server-side per-episode reseeding works without hand-written
``docker -v`` plumbing.

Pinned here:

* ``deterministic=False`` (default) is byte-identical to prior behavior -
  both the ``docker exec`` command and the ``docker run`` argv.
* ``deterministic=True`` adds ONLY the wrapper mount (read-only, fixed
  container path, library-resolved source) and nothing caller-controlled.
* The volume-safety guard rejects everything it rejected before, with the
  flag set.
* ``deterministic=True`` requires ``protocol="n1.7"`` - other protocols
  fail closed instead of silently dropping the flag.
* The flag threads through the full lifecycle chain
  (``start_container`` -> ``start``) with forward-all-advertised-kwargs
  discipline.
* The packaged wrapper is import-safe on the host (no torch/gr00t at
  module import) and self-contained (no strands_robots imports - it runs
  alone inside the container).
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

gi = importlib.import_module("strands_robots.tools.gr00t_inference")
from strands_robots.tools.gr00t_inference import (  # noqa: E402
    _build_inference_command,
    _start_container,
    gr00t_inference,
)


def _cmd_kwargs(**overrides: Any) -> dict[str, Any]:
    """Default args for ``_build_inference_command``."""
    base: dict[str, Any] = {
        "container_name": "gr00t",
        "checkpoint_path": "/data/checkpoints/model",
        "port": 8000,
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
        "protocol": "n1.7",
        "use_sim_policy_wrapper": True,
    }
    base.update(overrides)
    return base


def _container_kwargs(**overrides: Any) -> dict[str, Any]:
    """Default args for ``_start_container``."""
    base: dict[str, Any] = {
        "image_name": "gr00t:latest",
        "container_name": "gr00t",
        "port": 8000,
        "volumes": None,
        "hf_token": None,
        "container_command": "tail -f /dev/null",
        "hf_local_dir": "/data/cp",
        "force": True,
    }
    base.update(overrides)
    return base


def _capture_docker_run(**kwargs: Any) -> tuple[dict[str, Any], list[str]]:
    """Run ``_start_container`` with docker mocked; return (result, docker-run argv)."""
    runs: list[list[str]] = []

    def fake_run(cmd, *a, **kw):
        runs.append(list(cmd))
        return MagicMock(stdout="", stderr="", returncode=0)

    with (
        patch.object(gi, "_container_state", return_value="absent"),
        patch.object(gi.subprocess, "run", side_effect=fake_run),
    ):
        result = _start_container(**_container_kwargs(**kwargs))
    argv = next((c for c in runs if c[:2] == ["docker", "run"]), [])
    return result, argv


# --- command builder: entrypoint swap ----------------------------------


class TestDeterministicEntrypoint:
    def test_deterministic_swaps_n17_entrypoint_to_wrapper(self):
        cmd = _build_inference_command(**_cmd_kwargs(deterministic=True))
        joined = " ".join(cmd)
        assert "/srv_wrap.py" in joined
        # The bare module entrypoint must be gone - the wrapper hands off to it.
        assert "gr00t.eval.run_gr00t_server" not in joined
        assert "-m" not in cmd
        # Same args carry through unchanged.
        assert "--model-path" in cmd
        assert "/data/checkpoints/model" in cmd
        assert "--use-sim-policy-wrapper" in cmd
        assert "--embodiment-tag" in cmd
        # The wrapper's output is routed to the container's log stream so
        # `docker logs` shows the determinism banner (docker exec -d
        # detaches from the logging driver).
        script = next(tok for tok in cmd if "/srv_wrap.py" in tok)
        assert script.startswith("exec python /srv_wrap.py")
        assert "/proc/1/fd/1" in script

    def test_default_false_is_byte_identical_to_prior_behavior(self):
        """Omitting the kwarg and passing deterministic=False emit the same argv."""
        default_cmd = _build_inference_command(**_cmd_kwargs())
        explicit_cmd = _build_inference_command(**_cmd_kwargs(deterministic=False))
        assert default_cmd == explicit_cmd
        assert "/srv_wrap.py" not in " ".join(default_cmd)
        assert "gr00t.eval.run_gr00t_server" in default_cmd


# --- dispatch boundary: protocol gate ----------------------------------


class TestProtocolGate:
    @pytest.mark.parametrize("protocol", ["n1.5", "n1.6"])
    def test_deterministic_rejected_for_legacy_protocols(self, protocol):
        """No silent drop: the wrapper only exists for the N1.7 entrypoint."""
        result = gr00t_inference(
            action="start",
            checkpoint_path="/data/checkpoints/m",
            protocol=protocol,
            deterministic=True,
        )
        assert result["status"] == "error"
        assert "n1.7" in result["message"]

    def test_deterministic_forwarded_on_start_container(self):
        with patch(
            "strands_robots.tools.gr00t_inference._start_container",
            return_value={"status": "success", "skipped": True, "message": "ok"},
        ) as mock:
            result = gr00t_inference(
                action="start_container",
                port=8000,
                protocol="n1.7",
                deterministic=True,
            )
        assert result["status"] == "success"
        assert mock.call_args.kwargs["deterministic"] is True

    def test_deterministic_forwarded_through_lifecycle(self):
        with patch(
            "strands_robots.tools.gr00t_inference._lifecycle",
            return_value={"status": "success", "phase": "full", "steps": [], "message": "ok"},
        ) as mock:
            gr00t_inference(
                action="lifecycle",
                lifecycle="full",
                hf_repo="nvidia/foo",
                protocol="n1.7",
                deterministic=True,
            )
        assert mock.call_args.kwargs["deterministic"] is True

    def test_lifecycle_threads_deterministic_to_container_and_start(self, tmp_path):
        """`lifecycle="full"` forwards the flag to BOTH sub-steps that need it."""
        with (
            patch.object(gi, "_build_image", return_value={"status": "success"}),
            patch.object(
                gi,
                "_download_checkpoint",
                return_value={"status": "success", "local_dir": str(tmp_path)},
            ),
            patch.object(
                gi,
                "_start_container",
                return_value={"status": "success", "container_name": "gr00t"},
            ) as mock_container,
            patch.object(gi, "_start_service", return_value={"status": "success", "message": "up"}) as mock_start,
        ):
            result = gr00t_inference(
                action="lifecycle",
                lifecycle="full",
                hf_repo="nvidia/GR00T-N1.7-LIBERO",
                hf_subfolder="libero_10",
                protocol="n1.7",
                use_sim_policy_wrapper=True,
                deterministic=True,
            )
        assert result["status"] == "success"
        assert mock_container.call_args.kwargs["deterministic"] is True
        assert mock_start.call_args.kwargs["deterministic"] is True


# --- _start_container: curated wrapper mount ---------------------------


class TestWrapperMount:
    def test_deterministic_adds_only_the_readonly_wrapper_mount(self, monkeypatch):
        """The flag's whole docker-run delta is one ro mount of the packaged file."""
        for var in gi._DETERMINISTIC_ENV_PASSTHROUGH:
            monkeypatch.delenv(var, raising=False)

        _, base_argv = _capture_docker_run(deterministic=False)
        result, det_argv = _capture_docker_run(deterministic=True)

        assert result["status"] == "success"
        wrapper = gi._server_wrapper_host_path()
        expected_mount = f"{wrapper}:/srv_wrap.py:ro"
        delta = [tok for tok in det_argv if tok not in base_argv]
        assert delta == [expected_mount], f"deterministic=True changed more than the wrapper mount: {delta}"
        # The mount is read-only and its source is the packaged wrapper.
        assert expected_mount.endswith(":ro")
        assert result["deterministic"] is True
        assert result["deterministic_wrapper"] == {
            "host_path": str(wrapper),
            "container_path": "/srv_wrap.py",
            "read_only": True,
        }

    def test_default_docker_run_argv_unchanged(self, monkeypatch):
        """deterministic omitted vs =False: byte-identical docker run argv."""
        for var in gi._DETERMINISTIC_ENV_PASSTHROUGH:
            monkeypatch.delenv(var, raising=False)
        _, omitted_argv = _capture_docker_run()
        _, explicit_argv = _capture_docker_run(deterministic=False)
        assert omitted_argv == explicit_argv
        assert not any("/srv_wrap.py" in tok for tok in omitted_argv)

    def test_determinism_env_vars_forwarded_into_container(self, monkeypatch):
        monkeypatch.setenv("STRANDS_GR00T_SERVER_SEED", "7")
        monkeypatch.setenv("STRANDS_GR00T_STRICT_DETERMINISTIC", "1")
        _, argv = _capture_docker_run(deterministic=True)
        joined = " ".join(argv)
        assert "-e STRANDS_GR00T_SERVER_SEED=7" in joined
        assert "-e STRANDS_GR00T_STRICT_DETERMINISTIC=1" in joined

    def test_determinism_env_vars_not_forwarded_by_default(self, monkeypatch):
        monkeypatch.setenv("STRANDS_GR00T_SERVER_SEED", "7")
        _, argv = _capture_docker_run(deterministic=False)
        assert "STRANDS_GR00T_SERVER_SEED=7" not in " ".join(argv)

    def test_missing_wrapper_file_fails_closed_before_docker(self):
        with (
            patch.object(gi, "_server_wrapper_host_path", return_value=Path("/nonexistent/server_wrapper.py")),
            patch.object(gi, "_container_state", return_value="absent"),
            patch.object(gi.subprocess, "run", side_effect=AssertionError("docker must NOT be called")),
        ):
            result = _start_container(**_container_kwargs(deterministic=True))
        assert result["status"] == "error"
        assert "server_wrapper.py" in result["message"]

    def test_running_container_without_wrapper_mount_fails_fast(self):
        """Idempotent skip must not hand back a container that can't serve /srv_wrap.py."""
        with (
            patch.object(gi, "_container_state", return_value="running"),
            patch.object(gi, "_container_has_wrapper_mount", return_value=False),
            patch.object(gi.subprocess, "run", side_effect=AssertionError("docker run must NOT be called")),
        ):
            result = _start_container(**_container_kwargs(deterministic=True, force=False))
        assert result["status"] == "error"
        assert "force=True" in result["message"]

    def test_running_container_with_wrapper_mount_skips(self):
        with (
            patch.object(gi, "_container_state", return_value="running"),
            patch.object(gi, "_container_has_wrapper_mount", return_value=True),
            patch.object(gi.subprocess, "run", side_effect=AssertionError("docker run must NOT be called")),
        ):
            result = _start_container(**_container_kwargs(deterministic=True, force=False))
        assert result["status"] == "success"
        assert result["skipped"] is True


# --- volume-safety guard: unchanged rejection surface ------------------


class TestVolumeSafetyUnchanged:
    @pytest.mark.parametrize(
        "bad_volumes",
        [
            {"/": "/host"},
            {"/etc": "/host_etc"},
            {"/var/run/docker.sock": "/var/run/docker.sock"},
            {"/root/.ssh": "/keys"},
        ],
    )
    def test_blocked_mounts_still_rejected_with_deterministic(self, bad_volumes):
        with (
            patch.object(gi, "_container_state", return_value="absent"),
            patch.object(gi.subprocess, "run", side_effect=AssertionError("docker run must NOT be called")),
        ):
            result = _start_container(**_container_kwargs(volumes=bad_volumes, deterministic=True))
        assert result["status"] == "error"

    def test_agent_still_cannot_supply_volumes(self):
        """The tool signature must not have grown a volume/command parameter."""
        import inspect

        params = inspect.signature(inspect.unwrap(gr00t_inference)).parameters
        assert "volumes" not in params
        assert "container_command" not in params
        assert "image_name" not in params
        assert "deterministic" in params


# --- packaged wrapper hygiene ------------------------------------------


class TestPackagedWrapper:
    def test_wrapper_ships_in_the_package(self):
        wrapper = gi._server_wrapper_host_path()
        assert wrapper.is_file()
        assert wrapper.parts[-4:] == ("strands_robots", "policies", "groot", "server_wrapper.py")

    def test_wrapper_is_import_safe_on_the_host(self):
        """Importing the module must not pull torch/gr00t/tyro or start a server."""
        mod = importlib.import_module("strands_robots.policies.groot.server_wrapper")
        assert callable(mod.main)

    def test_wrapper_is_self_contained(self):
        """The wrapper runs alone inside the container - no strands_robots imports."""
        tree = ast.parse(gi._server_wrapper_host_path().read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith("strands_robots"), (
                    f"server_wrapper.py imports {name!r} but must stay self-contained: "
                    "it is mounted alone into the GR00T container where strands_robots "
                    "is not installed."
                )

    def test_example_shim_reexports_main(self):
        """Back-compat: the old examples/ path still exposes main()."""
        import runpy

        example = Path(__file__).resolve().parents[2] / "examples" / "libero" / "gr00t_server_deterministic_wrapper.py"
        ns = runpy.run_path(str(example))
        from strands_robots.policies.groot.server_wrapper import main

        assert ns["main"] is main
