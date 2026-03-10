"""Tests for headless MuJoCo rendering auto-configuration.

Tests _is_headless() and _configure_gl_backend() functions in simulation.py
that auto-detect headless environments and configure MUJOCO_GL before import.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock strands SDK + heavy deps at module level (standard pattern)
# ---------------------------------------------------------------------------
_mock_strands = MagicMock()
_mock_strands.tool = lambda f: f
_mock_strands.tools = MagicMock()
_mock_strands.tools.tools = MagicMock()
_mock_strands.types = MagicMock()
_mock_strands.types._events = MagicMock()
_mock_strands.types.tools = MagicMock()

_mock_np = MagicMock()
_mock_pil = MagicMock()

_MOCK_MODULES = {
    "strands": _mock_strands,
    "strands.tools": _mock_strands.tools,
    "strands.tools.tools": _mock_strands.tools.tools,
    "strands.types": _mock_strands.types,
    "strands.types._events": _mock_strands.types._events,
    "strands.types.tools": _mock_strands.types.tools,
    "numpy": _mock_np,
    "mujoco": MagicMock(),
    "mujoco.viewer": MagicMock(),
    "PIL": _mock_pil,
    "PIL.Image": MagicMock(),
    "imageio": MagicMock(),
}

# Cache the module once (avoids re-import issues)
_sim_module = None


def _get_simulation_module():
    """Import simulation module with mocked deps (cached)."""
    global _sim_module
    if _sim_module is not None:
        return _sim_module
    with patch.dict(sys.modules, _MOCK_MODULES):
        if "strands_robots.simulation" in sys.modules:
            del sys.modules["strands_robots.simulation"]
        import strands_robots.simulation as sim

        _sim_module = sim
        return sim


@pytest.fixture(autouse=True)
def _clean_env():
    """Remove MUJOCO_GL and display vars between tests."""
    env_vars = ["MUJOCO_GL", "DISPLAY", "WAYLAND_DISPLAY"]
    saved = {k: os.environ.pop(k, None) for k in env_vars}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


@pytest.fixture(autouse=True)
def _reset_mujoco_globals():
    """Reset module-level globals between tests."""
    sim = _get_simulation_module()
    sim._mujoco = None
    sim._mujoco_viewer = None
    yield
    sim._mujoco = None
    sim._mujoco_viewer = None


# ===================================================================
# _is_headless() tests
# ===================================================================


class TestIsHeadless:
    """Tests for headless environment detection."""

    def test_linux_no_display(self):
        """Linux without DISPLAY or WAYLAND_DISPLAY → headless."""
        sim = _get_simulation_module()
        os.environ.pop("DISPLAY", None)
        os.environ.pop("WAYLAND_DISPLAY", None)
        with patch.object(sys, "platform", "linux"):
            assert sim._is_headless() is True

    def test_linux_with_display(self):
        """Linux with DISPLAY set → not headless."""
        sim = _get_simulation_module()
        os.environ["DISPLAY"] = ":0"
        with patch.object(sys, "platform", "linux"):
            assert sim._is_headless() is False

    def test_linux_with_wayland(self):
        """Linux with WAYLAND_DISPLAY set → not headless."""
        sim = _get_simulation_module()
        os.environ["WAYLAND_DISPLAY"] = "wayland-0"
        with patch.object(sys, "platform", "linux"):
            assert sim._is_headless() is False

    def test_macos_always_not_headless(self):
        """macOS always returns False (CGL always available)."""
        sim = _get_simulation_module()
        os.environ.pop("DISPLAY", None)
        with patch.object(sys, "platform", "darwin"):
            assert sim._is_headless() is False

    def test_windows_always_not_headless(self):
        """Windows always returns False (WGL always available)."""
        sim = _get_simulation_module()
        os.environ.pop("DISPLAY", None)
        with patch.object(sys, "platform", "win32"):
            assert sim._is_headless() is False


# ===================================================================
# _configure_gl_backend() tests
# ===================================================================


class TestConfigureGlBackend:
    """Tests for OpenGL backend auto-configuration."""

    def test_respects_user_set_mujoco_gl(self):
        """Never override a user-set MUJOCO_GL."""
        sim = _get_simulation_module()
        os.environ["MUJOCO_GL"] = "glfw"
        sim._configure_gl_backend()
        assert os.environ["MUJOCO_GL"] == "glfw"

    def test_noop_when_display_available(self):
        """Does nothing when a display server is available."""
        sim = _get_simulation_module()
        os.environ["DISPLAY"] = ":0"
        with patch.object(sys, "platform", "linux"):
            sim._configure_gl_backend()
        assert "MUJOCO_GL" not in os.environ

    def test_headless_selects_egl_when_available(self):
        """Selects EGL when headless and libEGL.so.1 is loadable."""
        sim = _get_simulation_module()
        os.environ.pop("DISPLAY", None)
        with patch.object(sys, "platform", "linux"):
            with patch("ctypes.cdll") as mock_cdll:
                mock_cdll.LoadLibrary = MagicMock(return_value=MagicMock())
                sim._configure_gl_backend()
        assert os.environ.get("MUJOCO_GL") == "egl"

    def test_headless_falls_back_to_osmesa(self):
        """Falls back to OSMesa when EGL is not available."""
        sim = _get_simulation_module()
        os.environ.pop("DISPLAY", None)

        def load_lib(name):
            if "EGL" in name:
                raise OSError("libEGL not found")
            return MagicMock()  # OSMesa succeeds

        with patch.object(sys, "platform", "linux"):
            with patch("ctypes.cdll") as mock_cdll:
                mock_cdll.LoadLibrary = load_lib
                sim._configure_gl_backend()
        assert os.environ.get("MUJOCO_GL") == "osmesa"

    def test_headless_warns_when_no_backend(self):
        """Warns when neither EGL nor OSMesa is available."""
        sim = _get_simulation_module()
        os.environ.pop("DISPLAY", None)

        def load_lib(name):
            raise OSError(f"{name} not found")

        with patch.object(sys, "platform", "linux"):
            with patch("ctypes.cdll") as mock_cdll:
                mock_cdll.LoadLibrary = load_lib
                sim._configure_gl_backend()
        assert "MUJOCO_GL" not in os.environ  # Not set — will warn

    def test_noop_on_macos(self):
        """Does nothing on macOS regardless of DISPLAY."""
        sim = _get_simulation_module()
        os.environ.pop("DISPLAY", None)
        with patch.object(sys, "platform", "darwin"):
            sim._configure_gl_backend()
        assert "MUJOCO_GL" not in os.environ

    def test_egl_preferred_over_osmesa(self):
        """EGL is always preferred over OSMesa (GPU > CPU)."""
        sim = _get_simulation_module()
        os.environ.pop("DISPLAY", None)

        call_order = []

        def load_lib(name):
            call_order.append(name)
            return MagicMock()  # Both available

        with patch.object(sys, "platform", "linux"):
            with patch("ctypes.cdll") as mock_cdll:
                mock_cdll.LoadLibrary = load_lib
                sim._configure_gl_backend()

        assert os.environ.get("MUJOCO_GL") == "egl"
        # EGL should be probed first
        assert call_order[0] == "libEGL.so.1"


# ===================================================================
# _ensure_mujoco() integration tests
# ===================================================================


class TestEnsureMujoco:
    """Tests that _ensure_mujoco calls _configure_gl_backend before import."""

    def test_calls_configure_gl_before_import(self):
        """Verify _configure_gl_backend is called before importing mujoco."""
        sim = _get_simulation_module()
        sim._mujoco = None  # Force re-initialization

        configure_called = False
        original_configure = sim._configure_gl_backend

        def track_configure():
            nonlocal configure_called
            configure_called = True
            original_configure()

        with patch.object(sim, "_configure_gl_backend", side_effect=track_configure):
            with patch.dict(sys.modules, {"mujoco": MagicMock(), "mujoco.viewer": MagicMock()}):
                sim._ensure_mujoco()

        assert configure_called

    def test_configure_only_called_once(self):
        """_configure_gl_backend only called on first import, not subsequent calls."""
        sim = _get_simulation_module()
        sim._mujoco = None

        call_count = 0
        original_configure = sim._configure_gl_backend

        def counting_configure():
            nonlocal call_count
            call_count += 1
            original_configure()

        with patch.object(sim, "_configure_gl_backend", side_effect=counting_configure):
            with patch.dict(sys.modules, {"mujoco": MagicMock(), "mujoco.viewer": MagicMock()}):
                sim._ensure_mujoco()
                sim._ensure_mujoco()  # Second call — _mujoco already set

        assert call_count == 1

    def test_raises_import_error_without_mujoco(self):
        """Raises ImportError with helpful message if mujoco not installed."""
        sim = _get_simulation_module()
        sim._mujoco = None

        # Temporarily remove mujoco from sys.modules to trigger ImportError
        with patch.dict(sys.modules, {"mujoco": None}):
            with pytest.raises(ImportError, match="pip install"):
                sim._ensure_mujoco()
