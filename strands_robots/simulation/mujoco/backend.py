"""MuJoCo lazy import and GL backend configuration."""

import contextlib
import ctypes
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Canonical "no live world" error message shared by every world-touching
# MuJoCo facade/mixin method. Defined in this low-level module so the
# Simulation facade and its mixins (physics, randomization, rendering,
# recording) can all source the single string without a circular import -
# an agent that learns the error from one action recognises it identically
# from every other.
_NO_WORLD_MSG = "No world. Call create_world (or load_scene) first."

_mujoco = None
_mujoco_viewer = None


def _is_headless() -> bool:
    """Detect if running in a headless environment (no display server).

    Returns True on Linux when no DISPLAY or WAYLAND_DISPLAY is set,
    which means GLFW-based rendering will fail.

    Windows and macOS are always False because MuJoCo uses native
    windowing backends (WGL on Windows, CGL on macOS) that support
    offscreen rendering without X11/Wayland. The EGL/OSMesa fallback
    is Linux-specific.
    """
    if sys.platform != "linux":
        return False
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return False
    return True


# MuJoCo's own ``MUJOCO_GL`` vocabulary, transcribed from ``mujoco.gl_context``:
# it folds the value with ``.lower().strip()``, reads one family as "build no GL
# context at all", accepts a platform-dependent set of backend names, and raises
# ``RuntimeError`` at import time for anything else.
#
# Transcribed rather than asked. ``mujoco.gl_context`` computes its copy of this
# at module import, so it is already stale relative to a caller's environment;
# importing it is itself what raises for exactly the values a caller needs told
# about; and mujoco may not be installed at all, which is a separate question.
# Same reasoning as ``doctor._hf_token_path``, which transcribes the Hub's
# token-path rule for a Hub that may not be installed either.
_MUJOCO_GL_DISABLE: frozenset[str] = frozenset({"disable", "disabled", "off", "false", "0"})
_MUJOCO_GL_ANY_PLATFORM: frozenset[str] = frozenset({"enable", "enabled", "on", "true", "1", "glfw", ""})
_MUJOCO_GL_PLATFORM_ONLY: dict[str, frozenset[str]] = {
    "Linux": frozenset({"glx", "egl", "osmesa"}),
    "Windows": frozenset({"wgl"}),
    "Darwin": frozenset({"cgl"}),
}
# Backends that render with no display server: MuJoCo routes Linux ``egl`` and
# ``osmesa`` to offscreen contexts, while every other backend it selects (GLFW on
# Linux and Windows, CGL on macOS) draws through the platform's window server.
_MUJOCO_GL_OFFSCREEN: frozenset[str] = frozenset({"egl", "osmesa"})


def _mujoco_gl_value() -> str:
    """The ``MUJOCO_GL`` value MuJoCo will read, folded the way MuJoCo folds it.

    MuJoCo reads ``os.environ.get("MUJOCO_GL", "").lower().strip()``, so ``EGL``,
    ``egl`` and ``" egl "`` all select the same backend. Comparing the raw string
    instead answers about a spelling rather than about a backend: ``MUJOCO_GL=EGL``
    renders through EGL, while a case-sensitive test reads it as a value nobody
    recognises.

    Returns:
        The folded value. Empty both for an unset variable and for one holding
        only whitespace, which MuJoCo also reads as "no preference".
    """
    return os.environ.get("MUJOCO_GL", "").lower().strip()


def _mujoco_gl_valid_values(system: str | None = None) -> frozenset[str]:
    """Values MuJoCo accepts for ``MUJOCO_GL`` on a platform.

    Args:
        system: Platform name as :func:`platform.system` reports it. Defaults to
            the running platform.

    Returns:
        The platform-independent names together with that platform's own backend
        names. A folded value outside this set and outside
        :data:`_MUJOCO_GL_DISABLE` makes ``import mujoco`` raise ``RuntimeError``.
    """
    name = platform.system() if system is None else system
    return _MUJOCO_GL_ANY_PLATFORM | _MUJOCO_GL_PLATFORM_ONLY.get(name, frozenset())


def _mujoco_gl_disables_rendering(value: str) -> bool:
    """Whether a folded ``MUJOCO_GL`` value builds no GL context at all.

    MuJoCo accepts this family and then defines no ``GLContext``, so rendering is
    unavailable rather than misconfigured. It is a different answer from a value
    MuJoCo refuses at import, and naming the wrong one sends a caller after the
    wrong remedy.

    Args:
        value: A value already folded by :func:`_mujoco_gl_value`.

    Returns:
        True when MuJoCo will skip GL context creation entirely.
    """
    return value in _MUJOCO_GL_DISABLE


def _mujoco_gl_offscreen_values(system: str | None = None) -> frozenset[str]:
    """Accepted values that render on a platform with no display server.

    Empty on a platform whose only backend draws through the window server, which
    is what a caller needs in order not to recommend a value MuJoCo refuses there.

    Args:
        system: Platform name as :func:`platform.system` reports it. Defaults to
            the running platform.

    Returns:
        The offscreen backends that platform accepts.
    """
    return _MUJOCO_GL_OFFSCREEN & _mujoco_gl_valid_values(system)


# glvnd EGL vendor ICD payload that points at the NVIDIA EGL library, plus the
# standard directories glvnd scans for vendor ICD JSON files. When MUJOCO_GL is
# "egl", libglvnd loads the first vendor whose ICD JSON is registered here; an
# NVIDIA host missing 10_nvidia.json silently falls through to Mesa llvmpipe.
_NVIDIA_EGL_ICD_JSON = '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}'
_GLVND_EGL_VENDOR_DIRS: tuple[str, ...] = (
    "/usr/share/glvnd/egl_vendor.d",
    "/etc/glvnd/egl_vendor.d",
)
# Directories that may hold an installed NVIDIA EGL library (x86_64 + aarch64).
_NVIDIA_EGL_LIB_DIRS: tuple[str, ...] = (
    "/usr/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib64",
)


def _nvidia_egl_library_present() -> bool:
    """Return True when an NVIDIA EGL library (libEGL_nvidia.so.*) is installed."""
    for root in _NVIDIA_EGL_LIB_DIRS:
        try:
            if any(Path(root).glob("libEGL_nvidia.so.*")):
                return True
        except OSError:
            continue
    return False


def _nvidia_egl_icd_registered() -> bool:
    """Return True when a glvnd EGL vendor ICD already references NVIDIA."""
    for vendor_dir in _GLVND_EGL_VENDOR_DIRS:
        try:
            entries = sorted(Path(vendor_dir).glob("*.json"))
        except OSError:
            continue
        for entry in entries:
            try:
                if "nvidia" in entry.read_text(errors="replace").lower():
                    return True
            except OSError:
                continue
    return False


def _ensure_nvidia_egl_vendor_icd() -> None:
    """Route MuJoCo's EGL backend to the NVIDIA GPU when the vendor ICD is missing.

    glvnd routes ``MUJOCO_GL=egl`` to whichever EGL vendor ICD JSON is registered
    under ``/usr/share/glvnd/egl_vendor.d/``. On an NVIDIA host or container where
    ``libEGL_nvidia`` is installed but the NVIDIA ICD (``10_nvidia.json``) is
    absent - common in CUDA base images that do not request the ``graphics``
    driver capability - glvnd silently falls back to Mesa ``llvmpipe`` (CPU
    software rasterization), roughly two orders of magnitude slower. That
    throttles every policy observation, rollout video, and dataset recording with
    no signal (see :func:`_warn_if_software_rendering`, which makes the symptom
    loud; this makes it go away).

    Rather than require root to write the system ICD, this stages a vendor ICD
    JSON in the user-writable strands-robots base dir and points glvnd at it via
    the ``__EGL_VENDOR_LIBRARY_FILENAMES`` env var (the documented libglvnd
    override) - the NVIDIA ICD first, then any already-registered system ICDs as
    fallback so non-NVIDIA setups are unaffected. Must run before ``import
    mujoco``; best-effort and never raises.

    No-op when: not Linux; the user already set ``__EGL_VENDOR_LIBRARY_FILENAMES``
    or ``__EGL_VENDOR_LIBRARY_DIRS`` (an explicit override is always respected); an
    NVIDIA ICD is already registered system-wide; or no NVIDIA EGL library is
    installed (not an NVIDIA host, so Mesa is the correct backend).
    """
    if sys.platform != "linux":
        return
    # Respect an explicit user/glvnd vendor override - never second-guess it.
    if os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES") or os.environ.get("__EGL_VENDOR_LIBRARY_DIRS"):
        return
    if _nvidia_egl_icd_registered():
        return
    if not _nvidia_egl_library_present():
        return

    try:
        from strands_robots.utils import get_base_dir

        icd_dir = get_base_dir() / "egl_vendor.d"
        icd_dir.mkdir(parents=True, exist_ok=True)
        nvidia_icd = icd_dir / "10_nvidia.json"
        nvidia_icd.write_text(_NVIDIA_EGL_ICD_JSON, encoding="utf-8")
    except OSError as e:
        logger.debug("Could not stage NVIDIA EGL vendor ICD, leaving glvnd default: %s", e)
        return

    # NVIDIA first, then any system vendor ICDs (Mesa etc.) as fallback.
    filenames = [str(nvidia_icd)]
    for vendor_dir in _GLVND_EGL_VENDOR_DIRS:
        try:
            filenames.extend(str(p) for p in sorted(Path(vendor_dir).glob("*.json")))
        except OSError:
            continue
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = ":".join(filenames)
    logger.info(
        "Registered NVIDIA EGL vendor ICD for GPU-accelerated MuJoCo offscreen rendering "
        "via __EGL_VENDOR_LIBRARY_FILENAMES (%s)",
        nvidia_icd,
    )


def _configure_gl_backend() -> None:  # noqa: C901
    """Auto-configure MuJoCo's OpenGL backend for headless environments.

    MuJoCo reads MUJOCO_GL at import time to select the OpenGL backend:
    - "egl"    - EGL (GPU-accelerated offscreen, requires libEGL + NVIDIA driver)
    - "osmesa" - OSMesa (CPU software rendering, slower but always works)
    - "glfw"   - GLFW (default, requires X11/Wayland display server)

    MuJoCo folds the value with ``.lower().strip()`` before reading it, so
    ``EGL`` and ``" egl "`` select EGL as well (see :func:`_mujoco_gl_value`).

    This function MUST be called before `import mujoco`. Setting MUJOCO_GL
    after import has no effect - the backend is locked at import time.

    Never overrides a user-set MUJOCO_GL value.
    """
    existing = os.environ.get("MUJOCO_GL")
    # Read the folded value, because that is the backend MuJoCo will select.
    # ``MUJOCO_GL=EGL`` renders through EGL and so has to reach the vendor-ICD
    # guarantee too; a case-sensitive test here left glvnd on its default, which
    # on an NVIDIA host missing the NVIDIA ICD is Mesa llvmpipe - the silent
    # software-rasterizer fallback ``_ensure_nvidia_egl_vendor_icd`` exists to
    # prevent. A whitespace-only value is not a preference: MuJoCo reads it as
    # unset, so auto-configuration is what respects it.
    value = _mujoco_gl_value()
    if value:
        logger.debug(f"MUJOCO_GL already set to '{existing}', respecting user config")
        if value == "egl":
            _ensure_nvidia_egl_vendor_icd()
        return

    if not _is_headless():
        return

    # Headless Linux - probe for EGL first (GPU-accelerated), then fall back to OSMesa (CPU)
    try:
        ctypes.cdll.LoadLibrary("libEGL.so.1")
        os.environ["MUJOCO_GL"] = "egl"
        _ensure_nvidia_egl_vendor_icd()
        logger.info("Headless environment detected - using MUJOCO_GL=egl (GPU-accelerated offscreen)")
        return
    except OSError:
        pass

    try:
        ctypes.cdll.LoadLibrary("libOSMesa.so")
        os.environ["MUJOCO_GL"] = "osmesa"
        logger.info("Headless environment detected - using MUJOCO_GL=osmesa (CPU software rendering)")
        return
    except OSError:
        pass

    logger.warning(
        "Headless environment detected but neither EGL nor OSMesa found. "
        "MuJoCo rendering will likely fail. Install one of:\n"
        "  GPU: apt-get install libegl1-mesa-dev  (or NVIDIA driver provides libEGL)\n"
        "  CPU: apt-get install libosmesa6-dev\n"
        "Then set: export MUJOCO_GL=egl  (or osmesa)"
    )


# Oldest MuJoCo release this backend can drive, and the single source of truth
# for it: the packaging pins, the refusal below and the docs all read this tuple.
#
# The backend builds and edits every scene through the ``MjSpec`` procedural API
# (see :mod:`strands_robots.simulation.mujoco.spec_builder`), and that API only
# becomes complete enough in 3.5.0:
#
# * ``MjSpec.delete`` first ships in 3.3.4. Without it no scene can be built at
#   all - :meth:`add_robot` fails on the first call with ``'mujoco._specs.MjSpec'
#   object has no attribute 'delete'``.
# * ``mujoco.mjtLightType`` first ships in 3.3.3, and ``MjVisual.global_`` (the
#   spelling of the reserved-word field) in 3.3.0; below those, world creation
#   raises before any robot is added.
# * URDF loading through ``MjSpec`` first works in 3.5.0. On 3.4.0 and earlier
#   ``add_robot(urdf_path=...)`` is refused with ``Could not find decoder for
#   resource '<path>.urdf'``, so no URDF robot can enter a scene.
#
# 3.5.0 is therefore the first release on which the backend's own surface works,
# not merely imports.
_MUJOCO_API_FLOOR: tuple[int, int, int] = (3, 5, 0)


def _mujoco_api_floor_error(version: str) -> str | None:
    """Refuse an installed MuJoCo older than the API floor this backend needs.

    A build below :data:`_MUJOCO_API_FLOOR` imports cleanly and reports a
    healthy ``mujoco.__version__``, then fails deep inside the first scene
    operation with a raw ``AttributeError`` naming a private MuJoCo binding -
    a message that implicates this package rather than the install. Refusing at
    the import funnel names the version, the floor and the remedy instead.

    Args:
        version: The installed ``mujoco.__version__``.

    Returns:
        The refusal text, or ``None`` when the build satisfies the floor. Also
        ``None`` when ``version`` carries no readable ``major.minor`` prefix: an
        unreadable version is not evidence of an old one, and refusing on it
        would reject a source build over its version string alone.
    """
    parts = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", version.strip())
    if parts is None:
        return None
    installed = (int(parts.group(1)), int(parts.group(2)), int(parts.group(3) or 0))
    if installed >= _MUJOCO_API_FLOOR:
        return None
    floor = ".".join(str(part) for part in _MUJOCO_API_FLOOR)
    return (
        f"mujoco {version} is installed, but the MuJoCo simulation backend requires "
        f"mujoco>={floor}. Earlier releases cannot build or edit a scene through the "
        "MjSpec API this backend is written against: URDF loading is refused with "
        '"Could not find decoder for resource" up to 3.4.0, and MjSpec.delete does '
        "not exist before 3.3.4. Upgrade with: "
        f"uv pip install 'mujoco>={floor},<4.0.0'  (or reinstall the extra: "
        "uv pip install 'strands-robots[sim-mujoco]')."
    )


def _ensure_mujoco() -> "Any":
    """Lazy import MuJoCo to avoid hard dependency.

    Auto-configures the OpenGL backend for headless environments before
    importing mujoco, since MUJOCO_GL must be set at import time.

    Uses require_optional() for consistent dependency management across
    the strands-robots package.

    Raises:
        ImportError: if mujoco is not installed, or is older than
            :data:`_MUJOCO_API_FLOOR` - which this backend cannot drive, so it
            is refused here rather than left to fail inside a scene operation.
    """
    global _mujoco, _mujoco_viewer
    if _mujoco is None:
        _configure_gl_backend()
        from strands_robots.utils import require_optional

        module = require_optional(
            "mujoco",
            pip_install="mujoco",
            extra="sim-mujoco",
            purpose="MuJoCo simulation",
        )
        # Not cached until it passes: a refused build must be re-checked (and
        # re-reported) on the next call rather than served from the global.
        if msg := _mujoco_api_floor_error(str(getattr(module, "__version__", ""))):
            raise ImportError(msg)
        _mujoco = module
    if _mujoco_viewer is None and not _is_headless():
        try:
            import mujoco.viewer as viewer

            _mujoco_viewer = viewer
        except ImportError:
            pass
    return _mujoco


_rendering_available: bool | None = None


def mj_name_to_id(model: Any, obj_type: int, name: Any) -> int:
    """Resolve a MuJoCo entity name to its id, refusing a name that is not a string.

    ``mujoco.mj_name2id`` declares its third parameter as ``const char *``, and the
    pybind11 binding maps Python ``None`` onto a NULL pointer instead of rejecting
    it. MuJoCo then dereferences that pointer while comparing names, so the call
    does not raise - it terminates the interpreter with SIGSEGV. Nothing above it
    can recover: the agent-tool envelope, the caller's ``except`` clauses and any
    open recording all die with the process.

    A value that is not a string cannot name an entity, so it resolves to "not
    found" (``-1``) and the caller's existing unknown-entity message reports it -
    naming the value and listing what the model does contain. Routing every
    lookup through here, rather than adding a type check at each public entry
    point, means a lookup added later is safe by construction.

    Args:
        model: The compiled ``MjModel`` to search.
        obj_type: A ``mujoco.mjtObj`` enum member (``mjOBJ_BODY``, ``mjOBJ_JOINT``, ...).
        name: The entity name to resolve. Anything that is not a ``str``
            resolves to ``-1`` without reaching the binding.

    Returns:
        The entity id, or ``-1`` when *name* is not a string or names nothing
        of *obj_type* in *model*.
    """
    if not isinstance(name, str):
        return -1
    import mujoco as _mj

    return int(_mj.mj_name2id(model, obj_type, name))


# One-shot guard so the software-rendering warning fires at most once per
# process. A set (mutated via .add, never reassigned) avoids a `global`
# rebind so static analysis sees it as used.
_software_render_warned: set[str] = set()

# GL_RENDERER substrings that identify a CPU software rasterizer (no GPU).
_SOFTWARE_RENDERERS: tuple[str, ...] = (
    "llvmpipe",
    "softpipe",
    "swrast",
    "kms_swrast",
    "software rasterizer",
)


def _warn_if_software_rendering(probe_stdout: str) -> None:
    """Warn once when the active GL renderer is a CPU software rasterizer.

    MuJoCo's EGL backend silently routes to Mesa ``llvmpipe`` when no GPU EGL
    vendor ICD is registered - e.g. an NVIDIA container missing
    ``/usr/share/glvnd/egl_vendor.d/10_nvidia.json``. Offscreen rendering still
    works but runs on the CPU at roughly two orders of magnitude lower
    throughput, which silently throttles every policy observation, rollout
    video, and dataset recording. The render probe reports ``GL_RENDERER`` via
    the ``__GL_RENDERER__=`` marker; surface a software rasterizer as an
    actionable warning instead of a silent performance cliff.

    Args:
        probe_stdout: Captured stdout of the render probe subprocess. When it
            lacks the ``__GL_RENDERER__=`` marker (e.g. PyOpenGL unavailable in
            the probe) this is a no-op - detection is best-effort only.
    """
    if _software_render_warned:
        return
    renderer = ""
    for line in probe_stdout.splitlines():
        if line.startswith("__GL_RENDERER__="):
            renderer = line[len("__GL_RENDERER__=") :].strip()
            break
    if not renderer:
        return
    low = renderer.lower()
    if any(token in low for token in _SOFTWARE_RENDERERS):
        _software_render_warned.add("warned")
        logger.warning(
            "MuJoCo is rendering on a CPU software rasterizer (GL_RENDERER=%r); "
            "offscreen renders will be ~100x slower than GPU and the GPU EGL "
            "backend is NOT active. On an NVIDIA host/container, register the "
            "NVIDIA EGL vendor ICD - write /usr/share/glvnd/egl_vendor.d/10_nvidia.json "
            'containing {"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}} '
            "and ensure NVIDIA_DRIVER_CAPABILITIES includes 'graphics'.",
            renderer,
        )


def _can_render() -> bool:
    """Check if MuJoCo offscreen rendering is available.

    Probes once by creating a minimal Renderer in a subprocess. Result is cached.
    Returns False on headless environments without EGL/OSMesa.

    On headless Linux, if MUJOCO_GL is not set after _configure_gl_backend()
    ran, it means neither EGL nor OSMesa is available. In that case the
    default GLFW backend would be used, which calls glfw.init() - abort()
    at the C level (SIGABRT), killing the entire process before Python can
    catch the error. We short-circuit to False to avoid the fatal probe.

    When MUJOCO_GL IS set (e.g. "egl"), the library may still be dysfunctional
    (libEGL.so.1 loadable but no GPU/driver). In that case mj.Renderer() aborts
    at the C level too. We run the probe in a subprocess so a SIGABRT in the
    child doesn't kill the host process.
    """
    global _rendering_available
    if _rendering_available is not None:
        return _rendering_available

    # Guard: on headless systems without an offscreen GL backend configured,
    # mj.Renderer() will use GLFW which triggers a C-level abort (SIGABRT).
    # Skip the probe entirely - rendering is impossible anyway.
    if _is_headless() and not os.environ.get("MUJOCO_GL"):
        _rendering_available = False
        logger.warning(
            "Headless environment without EGL/OSMesa - rendering disabled. "
            "Physics and joint observations will still work. "
            "Install libegl1-mesa-dev or libosmesa6-dev for camera rendering."
        )
        return False

    # Probe rendering in a subprocess to survive C-level aborts (SIGABRT).
    # On some CI environments, libEGL.so.1 is loadable but non-functional -
    # mj.Renderer() triggers a fatal abort that kills the entire process.
    # By running the probe in a child process, we detect the failure safely.
    # The renderer creation (and close) determines availability. The GL_RENDERER
    # capture is purely additive and fully wrapped in try/except so it can never
    # change the probe's success/failure - it only lets the parent detect a
    # silent CPU software-rasterizer fallback (see _warn_if_software_rendering).
    probe_script = (
        "import sys, mujoco\n"
        "m = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')\n"
        "r = mujoco.Renderer(m, height=1, width=1)\n"
        "try:\n"
        "    r.update_scene(mujoco.MjData(m)); r.render()\n"
        "    from OpenGL import GL\n"
        "    _g = GL.glGetString(GL.GL_RENDERER)\n"
        "    sys.stdout.write('__GL_RENDERER__=' + (_g.decode(errors='replace') if _g else '') + '\\n')\n"
        "except Exception:\n"
        "    pass\n"
        "r.close()\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe_script],
            capture_output=True,
            timeout=10,
            env=os.environ.copy(),
        )
        if result.returncode == 0:
            _rendering_available = True
            logger.info("MuJoCo rendering available (subprocess probe passed)")
            _warn_if_software_rendering(result.stdout.decode(errors="replace"))
        else:
            _rendering_available = False
            stderr = result.stderr.decode(errors="replace").strip()
            # Truncate for readability
            if len(stderr) > 200:
                stderr = stderr[:200] + "..."
            logger.warning(
                "MuJoCo rendering unavailable (subprocess probe failed, rc=%d): %s. "
                "Physics/policy will work, but render/camera observations will be skipped.",
                result.returncode,
                stderr,
            )
    except (subprocess.TimeoutExpired, OSError) as e:
        _rendering_available = False
        logger.warning(
            "MuJoCo rendering probe timed out or failed to run: %s. Rendering disabled.",
            e,
        )

    return _rendering_available  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# MuJoCo stderr-noise suppression
# ---------------------------------------------------------------------------
# When two scenes are merged via ``spec.attach()`` / ``spec.compile()`` MuJoCo
# prints a block of benign "compiler option conflict" lines straight to the C
# stderr (fd 2) -- e.g.::
#
#     WARNING: Attach conflict when attaching 'scene' to 'strands_sim', ...
#     timestep: parent has 0.002, child has 0.005, keeping parent value
#     iterations: parent has 100, child has 10, keeping parent value
#     ...
#
# These are not actionable by the user (we intentionally keep the parent's
# compiler options) and they spam the console on every add_robot / world
# build. Because they originate in C, Python's ``warnings`` / ``logging`` can't
# intercept them -- we have to filter the raw file descriptor.
#
# ``filter_mujoco_attach_noise()`` is a context manager that captures fd 2 for
# the duration of the wrapped call, drops the known-benign lines, forwards
# everything else through unchanged, and re-emits any dropped lines at DEBUG
# level so they are still recoverable with STRANDS_ROBOTS_VERBOSE_MUJOCO=1.


# Lines we consider benign attach/compile chatter. Matched case-insensitively
# against each captured stderr line.
_MUJOCO_NOISE_PATTERNS = (
    re.compile(r"Attach conflict when attaching", re.IGNORECASE),
    re.compile(
        r"^(timestep|iterations|ls_iterations|impratio|integrator|cone|"
        r"jacobian|solver|tolerance|ls_tolerance|noslip_iterations|"
        r"noslip_tolerance|ccd_iterations|ccd_tolerance|sdf_iterations|"
        r"sdf_initpoints|gravity|wind|magnetic|density|viscosity):"
        r".*(parent has|keeping parent value)",
        re.IGNORECASE,
    ),
    re.compile(r"parent has .* child has .* keeping parent value", re.IGNORECASE),
)

_noise_lock = threading.Lock()


def _is_mujoco_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(p.search(stripped) for p in _MUJOCO_NOISE_PATTERNS)


@contextlib.contextmanager
def filter_mujoco_attach_noise():
    """Suppress MuJoCo's benign attach/compile spam.

    MuJoCo emits "Attach conflict ... keeping parent value" chatter when two
    scenes are merged. Depending on the MuJoCo build this surfaces either as a
    Python ``UserWarning`` (from ``spec.to_xml()`` / ``compile()``) OR as raw
    C-level writes to stderr (fd 2). We suppress BOTH:

    * a ``warnings.catch_warnings`` filter drops the matching ``UserWarning``;
    * an fd-2 capture drops the matching raw lines and forwards the rest.

    No-op (yields immediately) when STRANDS_ROBOTS_VERBOSE_MUJOCO is truthy,
    when fd 2 can't be captured (e.g. already redirected, no real stderr in
    some test/Jupyter setups), or on any failure -- never let log hygiene
    break a working sim.
    """
    if os.getenv("STRANDS_ROBOTS_VERBOSE_MUJOCO", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        yield
        return

    # Layer 1: drop the matching Python UserWarning regardless of fd capture.
    _wctx = warnings.catch_warnings()
    _wctx.__enter__()
    warnings.filterwarnings(
        "ignore",
        message=r".*Attach conflict when attaching.*",
        category=UserWarning,
    )

    # Layer 2: capture raw fd-2 writes. Need a real, dup-able stderr fd.
    try:
        orig_fd = sys.stderr.fileno()
        saved_fd = os.dup(orig_fd)
    except (AttributeError, OSError, ValueError):
        # No real fd (captured stderr / pytest capsys / Jupyter) -> warning
        # filter still applies; just skip the fd capture.
        try:
            yield
        finally:
            _wctx.__exit__(None, None, None)
        return

    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        with _noise_lock:
            os.dup2(tmp.fileno(), orig_fd)
            try:
                yield
            finally:
                # Flush C-side then restore the real stderr.
                try:
                    sys.stderr.flush()
                except (ValueError, OSError):
                    # Best-effort flush; stderr may already be closed or
                    # detached during interpreter teardown. Nothing to recover.
                    pass
                os.dup2(saved_fd, orig_fd)
    finally:
        # Replay captured output, dropping the known-benign noise lines.
        try:
            tmp.seek(0)
            captured = tmp.read().decode(errors="replace")
        except OSError:
            # Capture file unreadable (closed/truncated); treat as no output
            # rather than masking the original yielded body with an I/O error.
            captured = ""
        finally:
            tmp.close()
            try:
                os.close(saved_fd)
            except OSError:
                # saved_fd may already be closed during teardown; cleanup is
                # best-effort, so a double-close is safe to ignore.
                pass

        if captured:
            kept, dropped = [], []
            for line in captured.splitlines(keepends=True):
                (dropped if _is_mujoco_noise(line) else kept).append(line)
            if kept:
                try:
                    sys.stderr.write("".join(kept))
                    sys.stderr.flush()
                except (ValueError, OSError):
                    # Best-effort replay of kept lines; if stderr is gone the
                    # captured noise is simply dropped. Nothing to recover.
                    pass
            if dropped:
                logger.debug(
                    "Suppressed %d benign MuJoCo attach/compile line(s) "
                    "(set STRANDS_ROBOTS_VERBOSE_MUJOCO=1 to see them):\n%s",
                    len(dropped),
                    "".join(dropped).rstrip(),
                )
        # Tear down the UserWarning filter last.
        _wctx.__exit__(None, None, None)


@contextlib.contextmanager
def capture_stderr_fd():
    """Capture C-level (fd 2) stderr writes into a list-wrapped string.

    Unlike ``contextlib.redirect_stderr`` -- which only swaps Python's
    ``sys.stderr`` object and is blind to writes coming from C extensions
    such as MuJoCo's OpenGL backend -- this redirects the underlying file
    descriptor 2, so warnings like the one-time ``ARB_clip_control`` notice
    emitted by ``renderer.render()`` are actually captured.

    Yields a single-element list; after the block exits, ``box[0]`` holds the
    captured text (possibly empty).

    No-op-ish fallback: if fd 2 can't be duped (captured stderr, pytest
    capsys, Jupyter), it yields an empty box and lets writes pass through
    rather than breaking the wrapped call.
    """
    box = [""]
    try:
        orig_fd = sys.stderr.fileno()
        saved_fd = os.dup(orig_fd)
    except (AttributeError, OSError, ValueError):
        # No real fd to capture; yield empty and pass through.
        yield box
        return

    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        with _noise_lock:
            os.dup2(tmp.fileno(), orig_fd)
            try:
                yield box
            finally:
                try:
                    sys.stderr.flush()
                except (ValueError, OSError):
                    pass  # stderr may be closed/detached (pytest capsys, Jupyter)
                os.dup2(saved_fd, orig_fd)
    finally:
        try:
            tmp.seek(0)
            box[0] = tmp.read().decode(errors="replace")
        except OSError:
            box[0] = ""
        finally:
            tmp.close()
            try:
                os.close(saved_fd)
            except OSError:
                pass  # fd may already be closed during interpreter shutdown
