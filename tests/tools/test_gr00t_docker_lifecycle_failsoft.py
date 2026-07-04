"""Pin: gr00t_inference docker lifecycle fails soft and surfaces subprocess errors.

The container helpers must behave predictably at the two failure boundaries an
operator actually hits:

  1. Docker is not installed / the daemon socket is unreachable. Probing state
     must NOT raise -- ``_container_state`` reports ``"absent"`` so
     ``lifecycle="teardown"`` and idempotent ``start`` remain callable on a
     host without docker.

  2. A ``docker run`` / ``docker rm`` / service-scan command fails. The helper
     must convert the failure into the ``{"status": "error", "message": ...}``
     tool-result contract -- never leak a raw ``CalledProcessError`` past the
     dispatch boundary and crash the agent.

These paths were previously unexercised; each test drives one real failure
branch through mocked subprocess boundaries (no docker daemon required).
"""

from __future__ import annotations

import importlib
import subprocess
from unittest.mock import MagicMock, patch

# ``from strands_robots.tools import gr00t_inference`` resolves via the package's
# lazy __getattr__ to the *tool function*, not the module -- import_module
# returns the module object so the private helpers + ``gi.subprocess`` are
# patchable. This mirrors the idiom in test_gr00t_container_hardening.py.
gi = importlib.import_module("strands_robots.tools.gr00t_inference")


# --- _container_state: fails soft when docker is absent ----------------------


def test_container_state_absent_when_docker_binary_missing():
    """No docker binary (FileNotFoundError) -> 'absent', not a crash."""
    with patch.object(gi.subprocess, "run", side_effect=FileNotFoundError("docker")):
        assert gi._container_state("gr00t") == "absent"


def test_container_state_absent_on_os_error():
    """A daemon-socket OSError while probing -> 'absent', not a crash."""
    with patch.object(gi.subprocess, "run", side_effect=OSError("socket")):
        assert gi._container_state("gr00t") == "absent"


def test_container_state_absent_on_nonzero_returncode():
    """`docker inspect` on an unknown container exits non-zero -> 'absent'."""
    with patch.object(gi.subprocess, "run", return_value=MagicMock(returncode=1, stdout="", stderr="No such object")):
        assert gi._container_state("gr00t") == "absent"


# --- _start_container: surfaces a docker run failure -------------------------


def test_start_container_surfaces_docker_run_failure():
    """A failing `docker run` becomes an error tool-result, not an exception."""
    boom = subprocess.CalledProcessError(returncode=125, cmd=["docker", "run"], stderr="port already allocated")
    with (
        patch.object(gi, "_container_state", return_value="absent"),
        patch.object(gi.subprocess, "run", side_effect=boom),
    ):
        result = gi._start_container(
            image_name="gr00t:latest",
            container_name="gr00t",
            port=5555,
            volumes=None,
            hf_token=None,
            container_command="tail -f /dev/null",
            hf_local_dir="/data/cp",
            force=True,
        )
    assert result["status"] == "error"
    assert "docker run failed" in result["message"]
    assert "port already allocated" in result["message"]


# --- _remove_container: surfaces a docker rm failure -------------------------


def test_remove_container_surfaces_docker_rm_failure():
    """A present container whose `docker rm -f` fails -> error tool-result."""
    boom = subprocess.CalledProcessError(returncode=1, cmd=["docker", "rm"], stderr="permission denied")
    with (
        patch.object(gi, "_container_state", return_value="running"),
        patch.object(gi.subprocess, "run", side_effect=boom),
    ):
        result = gi._remove_container(name="gr00t", remove_volumes=False)
    assert result["status"] == "error"
    assert "docker rm failed" in result["message"]
    assert "permission denied" in result["message"]


# --- service scans: convert unexpected errors to the tool-result contract ----


def test_list_running_services_surfaces_unexpected_error():
    """An unexpected error while probing ports -> error tool-result."""
    with patch.object(gi, "_is_service_running", side_effect=RuntimeError("probe blew up")):
        result = gi._list_running_services()
    assert result["status"] == "error"
    assert "Failed to list services" in result["message"]


def test_stop_service_surfaces_unexpected_error():
    """An unexpected error while enumerating containers -> error tool-result."""
    with patch.object(gi, "_find_gr00t_containers", side_effect=RuntimeError("daemon gone")):
        result = gi._stop_service(5555)
    assert result["status"] == "error"
    assert "Failed to stop service" in result["message"]
