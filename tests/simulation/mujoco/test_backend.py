"""Unit tests for mujoco/backend.py - GL backend auto-configuration."""

from __future__ import annotations

import builtins
import importlib.util
import io
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from strands_robots.simulation.mujoco import backend as backend_mod


@pytest.fixture
def restore_env(monkeypatch):
    """Isolate MUJOCO_GL / DISPLAY / glvnd EGL vendor env per test."""
    for var in (
        "MUJOCO_GL",
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        "__EGL_VENDOR_LIBRARY_DIRS",
    ):
        monkeypatch.delenv(var, raising=False)
    yield monkeypatch


class TestIsHeadless:
    """``_is_headless`` only returns True on Linux with no display server."""

    def test_non_linux_is_not_headless(self, restore_env):
        with patch.object(sys, "platform", "darwin"):
            assert backend_mod._is_headless() is False

    def test_linux_with_display_not_headless(self, restore_env):
        restore_env.setenv("DISPLAY", ":0")
        with patch.object(sys, "platform", "linux"):
            assert backend_mod._is_headless() is False

    def test_linux_with_wayland_not_headless(self, restore_env):
        restore_env.setenv("WAYLAND_DISPLAY", "wayland-0")
        with patch.object(sys, "platform", "linux"):
            assert backend_mod._is_headless() is False

    def test_linux_no_display_is_headless(self, restore_env):
        with patch.object(sys, "platform", "linux"):
            assert backend_mod._is_headless() is True


class TestConfigureGLBackend:
    """``_configure_gl_backend`` respects MUJOCO_GL and probes EGL then OSMesa."""

    def test_respects_user_mujoco_gl(self, restore_env):
        restore_env.setenv("MUJOCO_GL", "glfw")
        backend_mod._configure_gl_backend()
        # Value unchanged.
        assert os.environ["MUJOCO_GL"] == "glfw"

    def test_noop_on_non_headless(self, restore_env):
        with patch.object(sys, "platform", "darwin"):
            backend_mod._configure_gl_backend()
        # Nothing was set.
        assert "MUJOCO_GL" not in os.environ

    def test_headless_picks_egl_when_available(self, restore_env):
        with (
            patch.object(sys, "platform", "linux"),
            patch("strands_robots.simulation.mujoco.backend.ctypes.cdll.LoadLibrary") as load,
        ):
            load.side_effect = [None]
            try:
                backend_mod._configure_gl_backend()
                assert os.environ.get("MUJOCO_GL") == "egl"
                load.assert_called_once()
            finally:
                # explicit teardown - monkeypatch.delenv only covers vars it had seen at yield time
                os.environ.pop("MUJOCO_GL", None)

    def test_headless_falls_back_to_osmesa(self, restore_env):
        with (
            patch.object(sys, "platform", "linux"),
            patch("strands_robots.simulation.mujoco.backend.ctypes.cdll.LoadLibrary") as load,
        ):
            load.side_effect = [OSError("no libEGL"), None]
            try:
                backend_mod._configure_gl_backend()
                assert os.environ.get("MUJOCO_GL") == "osmesa"
                assert load.call_count == 2
            finally:
                os.environ.pop("MUJOCO_GL", None)

    def test_headless_without_any_gl_warns(self, restore_env, caplog):
        import logging

        with (
            patch.object(sys, "platform", "linux"),
            patch("strands_robots.simulation.mujoco.backend.ctypes.cdll.LoadLibrary") as load,
        ):
            load.side_effect = OSError("no GL")
            with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.mujoco.backend"):
                backend_mod._configure_gl_backend()
            # MUJOCO_GL stays unset.
            assert "MUJOCO_GL" not in os.environ
            # Warning text lists both libraries.
            assert any("EGL" in rec.message and "OSMesa" in rec.message for rec in caplog.records)


class TestCanRender:
    """``_can_render`` caches the probe result and short-circuits on headless+no-GL."""

    def _clear_cache(self):
        backend_mod._rendering_available = None

    def test_returns_cached_value(self):
        self._clear_cache()
        backend_mod._rendering_available = True
        assert backend_mod._can_render() is True

        backend_mod._rendering_available = False
        assert backend_mod._can_render() is False
        self._clear_cache()

    def test_headless_without_mujoco_gl_short_circuits(self, restore_env):
        """Probe must NOT run when headless+no-GL - otherwise GLFW SIGABRTs."""
        self._clear_cache()
        with patch.object(sys, "platform", "linux"):
            # No DISPLAY, no MUJOCO_GL.
            assert backend_mod._can_render() is False
        # Cached result remembers the negative.
        assert backend_mod._rendering_available is False
        self._clear_cache()


@pytest.mark.skipif(
    not importlib.util.find_spec("mujoco"),
    reason="mujoco not installed",
)
class TestEnsureMujoco:
    """``_ensure_mujoco`` returns a module-like object with MjModel/MjData."""

    def test_returns_module(self):
        mj = backend_mod._ensure_mujoco()
        # Smoke: these attributes must exist on the real module.
        assert hasattr(mj, "MjModel")
        assert hasattr(mj, "MjData")
        assert hasattr(mj, "mj_step")

    def test_is_cached(self):
        first = backend_mod._ensure_mujoco()
        second = backend_mod._ensure_mujoco()
        assert first is second


class TestCanRenderProbeOutcomes:
    """``_can_render`` interprets the subprocess probe result correctly.

    Thor's EGL probe always succeeds, so the failure/timeout branches are
    never exercised by integration runs. These tests drive each branch by
    faking ``subprocess.run`` while keeping MUJOCO_GL set so the headless
    short-circuit does not fire.
    """

    def _clear_cache(self):
        backend_mod._rendering_available = None

    def test_probe_success_marks_available(self, restore_env):
        self._clear_cache()
        restore_env.setenv("MUJOCO_GL", "egl")
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
        with patch.object(backend_mod.subprocess, "run", return_value=completed) as run:
            assert backend_mod._can_render() is True
            run.assert_called_once()
        # Result is cached so a second call does not re-probe.
        with patch.object(backend_mod.subprocess, "run") as run2:
            assert backend_mod._can_render() is True
            run2.assert_not_called()
        self._clear_cache()

    def test_probe_failure_marks_unavailable(self, restore_env, caplog):
        import logging

        self._clear_cache()
        restore_env.setenv("MUJOCO_GL", "egl")
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b"libEGL: failed to load driver"
        )
        with patch.object(backend_mod.subprocess, "run", return_value=completed):
            with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.mujoco.backend"):
                assert backend_mod._can_render() is False
        assert any("probe failed" in rec.message for rec in caplog.records)
        self._clear_cache()

    def test_probe_failure_truncates_long_stderr(self, restore_env, caplog):
        import logging

        self._clear_cache()
        restore_env.setenv("MUJOCO_GL", "egl")
        long_err = ("E" * 500).encode()
        completed = subprocess.CompletedProcess(args=[], returncode=2, stdout=b"", stderr=long_err)
        with patch.object(backend_mod.subprocess, "run", return_value=completed):
            with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.mujoco.backend"):
                assert backend_mod._can_render() is False
        # The truncation marker is appended once stderr exceeds 200 chars.
        assert any("..." in rec.message for rec in caplog.records)
        self._clear_cache()

    def test_probe_timeout_marks_unavailable(self, restore_env, caplog):
        import logging

        self._clear_cache()
        restore_env.setenv("MUJOCO_GL", "egl")
        with patch.object(
            backend_mod.subprocess, "run", side_effect=subprocess.TimeoutExpired(cmd="probe", timeout=10)
        ):
            with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.mujoco.backend"):
                assert backend_mod._can_render() is False
        assert any("timed out" in rec.message for rec in caplog.records)
        self._clear_cache()

    def test_probe_oserror_marks_unavailable(self, restore_env):
        self._clear_cache()
        restore_env.setenv("MUJOCO_GL", "egl")
        with patch.object(backend_mod.subprocess, "run", side_effect=OSError("exec failed")):
            assert backend_mod._can_render() is False
        assert backend_mod._rendering_available is False
        self._clear_cache()


class TestEnsureMujocoViewer:
    """``_ensure_mujoco`` loads the interactive viewer only when not headless.

    Headless CI/Thor runs skip the viewer; the viewer import branch is only
    reached on a desktop session. These tests drive both the success and the
    ImportError fallback without depending on a display server.
    """

    def _save_globals(self):
        return backend_mod._mujoco, backend_mod._mujoco_viewer

    def _restore_globals(self, saved):
        backend_mod._mujoco, backend_mod._mujoco_viewer = saved

    @pytest.mark.skipif(
        not importlib.util.find_spec("mujoco"),
        reason="mujoco not installed",
    )
    def test_viewer_loaded_when_not_headless(self):
        # _mujoco preset so the module import is skipped; only the viewer
        # branch runs. Forcing not-headless lets the real mujoco.viewer load.
        saved = self._save_globals()
        try:
            sentinel_mj = object()
            backend_mod._mujoco = sentinel_mj
            backend_mod._mujoco_viewer = None
            with patch.object(backend_mod, "_is_headless", return_value=False):
                result = backend_mod._ensure_mujoco()
            assert result is sentinel_mj
            # The interactive viewer submodule was bound.
            assert backend_mod._mujoco_viewer is not None
        finally:
            self._restore_globals(saved)

    def test_viewer_import_error_is_swallowed(self):
        saved = self._save_globals()
        try:
            sentinel_mj = object()
            backend_mod._mujoco = sentinel_mj
            backend_mod._mujoco_viewer = None
            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "mujoco.viewer":
                    raise ImportError("no viewer (headless build)")
                return real_import(name, *args, **kwargs)

            with (
                patch.object(backend_mod, "_is_headless", return_value=False),
                patch.object(builtins, "__import__", side_effect=fake_import),
            ):
                result = backend_mod._ensure_mujoco()
            # Import failure must not propagate; viewer stays unset.
            assert result is sentinel_mj
            assert backend_mod._mujoco_viewer is None
        finally:
            self._restore_globals(saved)


class TestMujocoNoiseFilter:
    """Regression tests for the benign-attach-noise filter.

    ``filter_mujoco_attach_noise`` and ``_is_mujoco_noise`` live in
    ``strands_robots.simulation.mujoco.backend`` and depend on stdlib modules
    (``contextlib``, ``os``, ``re``, ``tempfile``, ``threading``, ``warnings``)
    imported at module top. These tests pin the matching contract and the
    fd-2 capture round trip so the import wiring cannot silently regress.
    """

    @pytest.mark.parametrize(
        "line",
        [
            "Attach conflict when attaching 'scene' to 'strands_sim'",
            "timestep: parent has 0.002, child has 0.005, keeping parent value",
            "iterations: parent has 100, child has 10, keeping parent value",
            "WARNING: parent has 0.002 child has 0.005 keeping parent value",
        ],
    )
    def test_benign_lines_match(self, line):
        assert backend_mod._is_mujoco_noise(line) is True

    @pytest.mark.parametrize(
        "line",
        [
            "",
            "   ",
            "Segmentation fault in mj_step",
            "RuntimeError: actuator index out of range",
        ],
    )
    def test_real_errors_do_not_match(self, line):
        assert backend_mod._is_mujoco_noise(line) is False

    def test_verbose_env_disables_filtering(self, monkeypatch, capfd):
        """STRANDS_ROBOTS_VERBOSE_MUJOCO=1 must pass all stderr through."""
        monkeypatch.setenv("STRANDS_ROBOTS_VERBOSE_MUJOCO", "1")
        with backend_mod.filter_mujoco_attach_noise():
            print("Attach conflict when attaching scene", file=sys.stderr)
        captured = capfd.readouterr()
        assert "Attach conflict" in captured.err

    def test_filter_drops_noise_keeps_real_errors(self, monkeypatch):
        """End-to-end fd-2 capture: benign lines dropped, real lines kept.

        Uses a real OS-level pipe as stderr so the fd-dup path actually runs
        (capfd/capsys replace sys.stderr without a dup-able fileno, exercising
        only the fallback branch).
        """
        monkeypatch.delenv("STRANDS_ROBOTS_VERBOSE_MUJOCO", raising=False)
        r_fd, w_fd = os.pipe()
        saved_stderr = sys.stderr
        sys.stderr = os.fdopen(w_fd, "w")
        try:
            with backend_mod.filter_mujoco_attach_noise():
                sys.stderr.write("Attach conflict when attaching scene\n")
                sys.stderr.write("FATAL: actuator index out of range\n")
                sys.stderr.flush()
        finally:
            sys.stderr.close()
            sys.stderr = saved_stderr
        output = os.read(r_fd, 65536).decode()
        os.close(r_fd)
        assert "Attach conflict" not in output
        assert "FATAL: actuator index out of range" in output

    def test_passthrough_when_stderr_has_no_dupable_fileno(self, monkeypatch):
        """Replaced stderr without a real fd degrades to a transparent passthrough.

        Under pytest capsys, Jupyter, or any wrapper that swaps ``sys.stderr``
        for an object without a dup-able file descriptor, the fd-2 capture
        cannot run. The filter must still yield and never raise; with no fd
        capture there is nothing to scrub, so every line reaches the
        replacement stream unfiltered.
        """
        monkeypatch.delenv("STRANDS_ROBOTS_VERBOSE_MUJOCO", raising=False)
        buf = io.StringIO()  # .fileno() raises io.UnsupportedOperation
        monkeypatch.setattr(sys, "stderr", buf)

        with backend_mod.filter_mujoco_attach_noise():
            print("Attach conflict when attaching scene", file=sys.stderr)
            print("FATAL: real error", file=sys.stderr)

        out = buf.getvalue()
        assert "Attach conflict" in out
        assert "FATAL: real error" in out

    def test_detached_stderr_on_teardown_does_not_mask_body(self, monkeypatch):
        """A stderr whose flush/write fail during teardown is swallowed.

        The fd-2 capture path runs (the stream exposes a real fileno), but by
        the time the captured output is replayed the stream is detached, so its
        ``flush`` and ``write`` raise. The filter must absorb those failures
        rather than let a dead stderr propagate out of and mask the protected
        body.
        """
        monkeypatch.delenv("STRANDS_ROBOTS_VERBOSE_MUJOCO", raising=False)
        r_fd, w_fd = os.pipe()

        class _DetachedStderr:
            """Real fileno (capture runs) but flush/write fail (detached)."""

            def fileno(self):
                return w_fd

            def flush(self):
                raise ValueError("stderr detached")

            def write(self, _s):
                raise ValueError("stderr detached")

        saved_stderr = sys.stderr
        monkeypatch.setattr(sys, "stderr", _DetachedStderr())
        try:
            # Must not raise even though every flush/write on stderr fails.
            with backend_mod.filter_mujoco_attach_noise():
                # Emit a real (kept) line on fd 2 so the replay path attempts a
                # write to the detached stderr.
                os.write(w_fd, b"FATAL: actuator index out of range\n")
        finally:
            monkeypatch.setattr(sys, "stderr", saved_stderr)
            os.close(r_fd)
            os.close(w_fd)


class TestCaptureStderrFd:
    """Contract for ``capture_stderr_fd`` - the fd-2 stderr capture helper.

    ``capture_stderr_fd`` redirects the underlying file descriptor 2 (not just
    ``sys.stderr`` the Python object) so writes from C extensions such as
    MuJoCo's OpenGL backend land in the yielded box. Rendering exercises it on
    a GPU host, but the utility's contract - real capture, and a transparent
    passthrough when there is no dup-able fd - must hold on any host, GPU or
    not. These tests pin that contract without requiring a renderer.
    """

    def test_captures_c_level_fd2_writes(self):
        """A raw ``os.write`` to fd 2 inside the block lands in the box.

        Uses a real OS pipe as stderr so the fd-dup capture path actually runs
        (capsys/StringIO have no dup-able fileno and hit the passthrough branch
        instead). After the block the box holds exactly what was written at the
        C level, proving the descriptor - not just ``sys.stderr`` - was
        redirected.
        """
        r_fd, w_fd = os.pipe()
        saved_stderr = sys.stderr
        sys.stderr = os.fdopen(w_fd, "w")
        try:
            with backend_mod.capture_stderr_fd() as box:
                os.write(sys.stderr.fileno(), b"ARB_clip_control unavailable\n")
        finally:
            sys.stderr.close()
            sys.stderr = saved_stderr
            os.close(r_fd)
        assert box[0] == "ARB_clip_control unavailable\n"

    def test_passthrough_when_stderr_has_no_dupable_fileno(self, monkeypatch):
        """No dup-able fd (capsys/Jupyter) degrades to a transparent passthrough.

        When ``sys.stderr`` is an object without a real file descriptor the
        capture cannot run. The helper must still yield a box, never raise, and
        leave the box empty; writes reach the replacement stream unfiltered.
        """
        buf = io.StringIO()  # .fileno() raises io.UnsupportedOperation
        monkeypatch.setattr(sys, "stderr", buf)

        with backend_mod.capture_stderr_fd() as box:
            print("passthrough line", file=sys.stderr)

        assert box == [""]
        assert "passthrough line" in buf.getvalue()

    def test_flush_failure_on_teardown_does_not_mask_body(self, monkeypatch):
        """A stderr whose ``flush`` fails during teardown is swallowed.

        The capture path runs (the stream exposes a real fileno), but by
        teardown the stream is detached so ``flush`` raises. The helper must
        absorb that failure and still restore fd 2 and expose the captured
        text, rather than let a dead stderr propagate out of the block.
        """
        r_fd, w_fd = os.pipe()

        class _FlushFailsStderr:
            def fileno(self):
                return w_fd

            def flush(self):
                raise ValueError("stderr detached")

        saved_stderr = sys.stderr
        monkeypatch.setattr(sys, "stderr", _FlushFailsStderr())
        try:
            with backend_mod.capture_stderr_fd() as box:
                os.write(w_fd, b"captured despite flush failure\n")
        finally:
            monkeypatch.setattr(sys, "stderr", saved_stderr)
            os.close(r_fd)
            os.close(w_fd)
        assert box[0] == "captured despite flush failure\n"
