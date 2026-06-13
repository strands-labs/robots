"""Tests for the ``strands-robots doctor`` diagnostic command."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


class TestDoctorChecks:
    """Unit tests for individual doctor check functions."""

    def test_check_python_version_passes(self) -> None:
        from strands_robots.doctor import check_python_version

        result = check_python_version()
        # We are running on Python 3.12+, so it should pass
        assert "PASS" in result

    def test_check_strands_robots_version_passes(self) -> None:
        from strands_robots.doctor import check_strands_robots_version

        result = check_strands_robots_version()
        assert "PASS" in result

    def test_check_mujoco_passes(self) -> None:
        from strands_robots.doctor import check_mujoco

        result = check_mujoco()
        assert "PASS" in result

    def test_check_mujoco_gl_with_egl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_mujoco_gl

        monkeypatch.setenv("MUJOCO_GL", "egl")
        result = check_mujoco_gl()
        assert "PASS" in result

    def test_check_mujoco_gl_no_display(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_mujoco_gl

        monkeypatch.delenv("MUJOCO_GL", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        result = check_mujoco_gl()
        assert "FAIL" in result

    def test_check_cuda_returns_string(self) -> None:
        from strands_robots.doctor import check_cuda

        result = check_cuda()
        # Should be one of PASS, WARN, or FAIL — never crash
        assert any(x in result for x in ("PASS", "WARN", "FAIL"))

    def test_check_hf_auth_with_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_hf_auth

        monkeypatch.setenv("HF_TOKEN", "hf_test_token")
        result = check_hf_auth()
        assert "PASS" in result

    def test_check_hf_auth_without_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_hf_auth

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        # This might pass if ~/.cache/huggingface/token exists, or warn otherwise
        result = check_hf_auth()
        assert any(x in result for x in ("PASS", "WARN"))

    def test_check_strands_agents(self) -> None:
        from strands_robots.doctor import check_strands_agents

        result = check_strands_agents()
        assert "PASS" in result

    def test_check_mesh(self) -> None:
        from strands_robots.doctor import check_mesh

        result = check_mesh()
        # Either passes (zenoh installed) or warns (not installed)
        assert any(x in result for x in ("PASS", "WARN"))

    def test_check_serial_permissions_linux(self) -> None:
        from strands_robots.doctor import check_serial_permissions

        result = check_serial_permissions()
        # Should not crash regardless of platform
        assert any(x in result for x in ("PASS", "WARN", "FAIL", "SKIP"))


class TestDoctorCLI:
    """Integration tests for the doctor CLI entry point."""

    def test_module_invocation(self) -> None:
        """``python -m strands_robots doctor`` runs without crashing."""
        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        env["STRANDS_MESH"] = "false"
        env["NO_COLOR"] = "1"
        result = subprocess.run(
            [sys.executable, "-m", "strands_robots", "doctor"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        # Should complete (exit 0 or 1 depending on env)
        assert result.returncode in (0, 1)
        assert "strands-robots doctor" in result.stdout

    def test_unknown_command(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "strands_robots", "nonexistent"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1
        assert "Unknown command" in result.stdout

    def test_no_command(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "strands_robots"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1
        assert "Usage" in result.stdout


class TestRunDoctor:
    """Integration test for the full run_doctor() pipeline."""

    def test_run_doctor_returns_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import run_doctor

        monkeypatch.setenv("MUJOCO_GL", "egl")
        monkeypatch.setenv("STRANDS_MESH", "false")
        monkeypatch.setenv("NO_COLOR", "1")
        exit_code = run_doctor()
        assert isinstance(exit_code, int)
        assert exit_code in (0, 1)
