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
        # Should be one of PASS, WARN, or FAIL - never crash
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

    def test_run_doctor_returns_1_on_failure_without_color(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failing check must yield exit 1 even when color is disabled.

        Regression: ``run_doctor`` previously detected failures by looking for
        the red ANSI escape code in each check's output. Under ``NO_COLOR`` /
        ``TERM=dumb`` (typical in CI) the color helpers emit plain text with no
        escape, so a genuine ``FAIL`` was silently ignored and the command
        exited 0 while printing "All checks passed". This made ``doctor``
        useless as a scripted setup gate. The exit code must reflect failures
        regardless of color support.
        """
        from strands_robots import doctor

        # Disable color the same way NO_COLOR / TERM=dumb would at import time.
        monkeypatch.setattr(doctor, "_NO_COLOR", True)
        # Force one check to fail deterministically, independent of host setup.
        monkeypatch.setattr(doctor, "check_sim_smoke", lambda: doctor._fail("forced failure"))
        # Sanity: with color disabled the failure line carries no ANSI escape.
        failure_line = doctor.check_sim_smoke()
        assert "\033[31m" not in failure_line
        assert "  FAIL  " in failure_line

        exit_code = doctor.run_doctor()
        assert exit_code == 1


class TestDoctorDegradedPaths:
    """Diagnostic checks must report the correct status under degraded or
    failing environments - a missing optional dep, an unusable GPU, locked-down
    serial ports, or a broken sim. These paths are exactly what ``doctor`` exists
    to surface, so each must produce its documented PASS/WARN/FAIL/SKIP outcome
    rather than crashing.
    """

    @staticmethod
    def _block_imports(monkeypatch: pytest.MonkeyPatch, *blocked: str) -> None:
        """Simulate the named modules being uninstalled.

        Blocks both ``import <name>`` and ``importlib.import_module(<name>)``:
        the ``__import__`` hook covers the statement form, while evicting any
        cached entry from ``sys.modules`` forces ``import_module`` (which
        returns a live cached module without re-invoking ``__import__``) to
        re-import and hit the hook too.
        """
        import builtins
        import sys

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name in blocked or any(name.startswith(f"{b}.") for b in blocked):
                raise ImportError(f"blocked for test: {name}")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        for name in list(sys.modules):
            if name in blocked or any(name.startswith(f"{b}.") for b in blocked):
                monkeypatch.delitem(sys.modules, name, raising=False)

        monkeypatch.setattr(builtins, "__import__", fake_import)

    def test_strands_robots_not_importable_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_strands_robots_version

        self._block_imports(monkeypatch, "strands_robots")
        result = check_strands_robots_version()
        assert "  FAIL  " in result

    def test_mujoco_not_installed_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_mujoco

        self._block_imports(monkeypatch, "mujoco")
        result = check_mujoco()
        assert "  FAIL  " in result

    def test_strands_agents_not_importable_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_strands_agents

        self._block_imports(monkeypatch, "strands")
        result = check_strands_agents()
        assert "  FAIL  " in result

    def test_mesh_zenoh_missing_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_mesh

        self._block_imports(monkeypatch, "zenoh")
        result = check_mesh()
        assert "  WARN  " in result

    def test_lerobot_missing_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_lerobot

        self._block_imports(monkeypatch, "lerobot")
        result = check_lerobot()
        assert "  WARN  " in result

    def test_mujoco_gl_glfw_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_mujoco_gl

        monkeypatch.setenv("MUJOCO_GL", "glfw")
        result = check_mujoco_gl()
        assert "  WARN  " in result

    def test_mujoco_gl_unset_with_display_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_mujoco_gl

        monkeypatch.delenv("MUJOCO_GL", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")
        result = check_mujoco_gl()
        assert "  PASS  " in result

    def test_cuda_available_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        from strands_robots.doctor import check_cuda

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda _idx=0: "FakeGPU")
        result = check_cuda()
        assert "  PASS  " in result
        assert "FakeGPU" in result

    def test_cuda_build_but_unavailable_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        from strands_robots.doctor import check_cuda

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.version, "cuda", "12.0", raising=False)
        result = check_cuda()
        assert "  WARN  " in result
        assert "12.0" in result

    def test_cuda_torch_missing_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_cuda

        self._block_imports(monkeypatch, "torch")
        result = check_cuda()
        assert "  WARN  " in result

    def test_cuda_cpu_only_build_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A CPU-only torch build (no ``torch.version.cuda``) is diagnosed as a
        WARN pointing at the CUDA-enabled install, distinct from a CUDA build
        whose runtime is merely unavailable."""
        import torch

        from strands_robots.doctor import check_cuda

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.version, "cuda", None, raising=False)
        result = check_cuda()
        assert "  WARN  " in result
        assert "CPU-only" in result

    def test_sim_smoke_empty_observation_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import strands_robots
        from strands_robots.doctor import check_sim_smoke

        class _EmptyObsRobot:
            def __init__(self, *_a: object, **_k: object) -> None:
                pass

            def step(self) -> None:
                pass

            def get_observation(self, _name: str) -> dict:
                return {}

        monkeypatch.setattr(strands_robots, "Robot", _EmptyObsRobot)
        result = check_sim_smoke()
        assert "  FAIL  " in result

    def test_sim_smoke_exception_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import strands_robots
        from strands_robots.doctor import check_sim_smoke

        def _boom(*_a: object, **_k: object) -> object:
            raise RuntimeError("mujoco exploded")

        monkeypatch.setattr(strands_robots, "Robot", _boom)
        result = check_sim_smoke()
        assert "  FAIL  " in result
        assert "mujoco exploded" in result

    def test_python_below_minimum_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import collections
        import sys

        from strands_robots.doctor import check_python_version

        # version_info is a structseq: it compares as a tuple AND exposes
        # ``.major/.minor/.micro``. A bare tuple would satisfy the comparison
        # but break the attribute access in the message, so mirror both.
        vinfo = collections.namedtuple("vinfo", "major minor micro releaselevel serial")
        monkeypatch.setattr(sys, "version_info", vinfo(3, 11, 0, "final", 0))
        result = check_python_version()
        assert "  FAIL  " in result

    def test_hf_auth_no_token_anywhere_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pathlib import Path

        from strands_robots.doctor import check_hf_auth

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        # Force the cached-token path to look absent so the warn branch is taken
        # regardless of the host's ~/.cache/huggingface state.
        monkeypatch.setattr(Path, "exists", lambda _self: False)
        result = check_hf_auth()
        assert "  WARN  " in result

    def test_run_doctor_handles_check_raising(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A check that raises unexpectedly must be reported as a failure (and
        force exit 1), never abort the whole diagnostic run."""
        from strands_robots import doctor

        def _raises() -> str:
            raise RuntimeError("kaboom")

        monkeypatch.setattr(doctor, "_NO_COLOR", True)
        monkeypatch.setattr(doctor, "check_mesh", _raises)
        exit_code = doctor.run_doctor()
        assert exit_code == 1


class TestDoctorSerialPermissions:
    """``check_serial_permissions`` must distinguish: non-Linux (skip), missing
    dialout group (skip), user not in dialout (fail), and user in dialout with /
    without accessible devices (fail / pass)."""

    @staticmethod
    def _fake_group(members: list[str], gid: int = 20) -> object:
        import types

        return types.SimpleNamespace(gr_mem=members, gr_gid=gid)

    def test_non_linux_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import platform

        from strands_robots.doctor import check_serial_permissions

        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        result = check_serial_permissions()
        assert "  SKIP  " in result

    def test_no_dialout_group_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import grp
        import platform

        from strands_robots.doctor import check_serial_permissions

        monkeypatch.setattr(platform, "system", lambda: "Linux")

        def _no_group(_name: str) -> object:
            raise KeyError(_name)

        monkeypatch.setattr(grp, "getgrnam", _no_group)
        result = check_serial_permissions()
        assert "  SKIP  " in result

    def test_not_in_dialout_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import grp
        import os
        import platform

        from strands_robots.doctor import check_serial_permissions

        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setenv("USER", "nobody")
        monkeypatch.setattr(grp, "getgrnam", lambda _n: self._fake_group(["someone_else"], gid=20))
        monkeypatch.setattr(os, "getgroups", lambda: [1000])
        result = check_serial_permissions()
        assert "  FAIL  " in result

    def test_in_dialout_no_devices_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import grp
        import platform
        from pathlib import Path

        from strands_robots.doctor import check_serial_permissions

        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setenv("USER", "robotuser")
        monkeypatch.setattr(grp, "getgrnam", lambda _n: self._fake_group(["robotuser"], gid=20))
        monkeypatch.setattr(Path, "glob", lambda _self, _pat: iter([]))
        result = check_serial_permissions()
        assert "  PASS  " in result

    def test_in_dialout_accessible_device_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """User in dialout with a readable/writable serial device passes and
        names the device - the fully-healthy hardware verdict."""
        import grp
        import os
        import platform
        from pathlib import Path

        from strands_robots.doctor import check_serial_permissions

        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setenv("USER", "robotuser")
        monkeypatch.setattr(grp, "getgrnam", lambda _n: self._fake_group(["robotuser"], gid=20))
        monkeypatch.setattr(Path, "glob", lambda self, pat: iter([Path("/dev/ttyACM0")]) if "ACM" in pat else iter([]))
        monkeypatch.setattr(os, "access", lambda _p, _m: True)
        result = check_serial_permissions()
        assert "  PASS  " in result
        assert "/dev/ttyACM0" in result

    def test_in_dialout_device_not_accessible_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import grp
        import os
        import platform
        from pathlib import Path

        from strands_robots.doctor import check_serial_permissions

        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setenv("USER", "robotuser")
        monkeypatch.setattr(grp, "getgrnam", lambda _n: self._fake_group(["robotuser"], gid=20))
        monkeypatch.setattr(Path, "glob", lambda self, pat: iter([Path("/dev/ttyACM0")]) if "ACM" in pat else iter([]))
        monkeypatch.setattr(os, "access", lambda _p, _m: False)
        result = check_serial_permissions()
        assert "  FAIL  " in result

    def test_effective_group_probe_survives_oserror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If ``os.getgroups`` raises, the effective-group probe must degrade to
        False rather than crash - the dialout membership check still decides."""
        import grp
        import os
        import platform
        from pathlib import Path

        from strands_robots.doctor import check_serial_permissions

        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setenv("USER", "robotuser")
        monkeypatch.setattr(grp, "getgrnam", lambda _n: self._fake_group(["robotuser"], gid=20))

        def _boom() -> list[int]:
            raise OSError("getgroups failed")

        monkeypatch.setattr(os, "getgroups", _boom)
        monkeypatch.setattr(Path, "glob", lambda _self, _pat: iter([]))
        result = check_serial_permissions()
        # User is a listed member, so this still passes despite the probe error.
        assert "  PASS  " in result


class TestDoctorVersionResolution:
    """Version checks must report the real installed version, not a placeholder.

    Regression: ``check_strands_robots_version``/``check_strands_agents`` read a
    module-level ``__version__`` that neither package defines, so they emitted
    ``strands-robots unknown`` / ``strands-agents ?`` even when the package was
    correctly installed - making the first diagnostic a new user runs useless.
    """

    def test_strands_robots_version_matches_metadata(self) -> None:
        from importlib.metadata import version

        from strands_robots.doctor import check_strands_robots_version

        result = check_strands_robots_version()
        assert "PASS" in result
        assert version("strands-robots") in result
        assert "unknown" not in result

    def test_strands_agents_version_matches_metadata(self) -> None:
        from importlib.metadata import version

        from strands_robots.doctor import check_strands_agents

        result = check_strands_agents()
        assert "PASS" in result
        assert version("strands-agents") in result
        assert "strands-agents ?" not in result

    def test_resolve_version_prefers_distribution_metadata(self) -> None:
        from importlib.metadata import version

        from strands_robots.doctor import _resolve_version

        assert _resolve_version("strands_robots", "strands-robots") == version("strands-robots")

    def test_resolve_version_unknown_when_absent(self) -> None:
        from strands_robots.doctor import _resolve_version

        # Neither an installed distribution nor an importable module exists.
        assert _resolve_version("strands_robots_nope_xyz", "strands-robots-nope-xyz") == "unknown"

    def test_resolve_version_falls_back_to_module_attribute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When distribution metadata is absent but the module is importable, the
        version comes from the module's ``__version__`` attribute (source-tree
        install path)."""
        import importlib
        import importlib.metadata
        import types

        from strands_robots.doctor import _resolve_version

        def _no_metadata(_dist: str) -> str:
            raise importlib.metadata.PackageNotFoundError(_dist)

        fake_module = types.SimpleNamespace(__version__="9.9.9-src")
        monkeypatch.setattr(importlib.metadata, "version", _no_metadata)
        monkeypatch.setattr(importlib, "import_module", lambda _name: fake_module)
        assert _resolve_version("some_module", "some-dist") == "9.9.9-src"
