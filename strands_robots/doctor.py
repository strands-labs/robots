"""``strands-robots doctor`` - diagnose common setup issues in one command.

Checks: Python version, extras availability, GPU/CUDA, serial permissions,
MuJoCo GL backend, HuggingFace auth, and a sim smoke test. Prints a colored
pass/fail table so first-time users can fix problems before they hit cryptic
errors at runtime.

Usage:
    python -m strands_robots doctor
    strands-robots doctor        # (after pip install with [scripts] or console_scripts)
"""

from __future__ import annotations

import importlib
import logging
import os
import platform
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ANSI color helpers (degrade gracefully if NO_COLOR / dumb term)
_NO_COLOR = os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb"


def _green(s: str) -> str:
    return s if _NO_COLOR else f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return s if _NO_COLOR else f"\033[31m{s}\033[0m"


def _yellow(s: str) -> str:
    return s if _NO_COLOR else f"\033[33m{s}\033[0m"


def _bold(s: str) -> str:
    return s if _NO_COLOR else f"\033[1m{s}\033[0m"


def _pass(msg: str) -> str:
    return _green(f"  PASS  {msg}")


def _fail(msg: str, fix: str = "") -> str:
    line = _red(f"  FAIL  {msg}")
    if fix:
        line += f"\n        {_yellow('Fix: ' + fix)}"
    return line


def _warn(msg: str, note: str = "") -> str:
    line = _yellow(f"  WARN  {msg}")
    if note:
        line += f"\n        {note}"
    return line


def _skip(msg: str) -> str:
    return f"  SKIP  {msg}"


def _resolve_version(import_name: str, dist_name: str) -> str:
    """Resolve a package version, preferring installed distribution metadata.

    Neither ``strands_robots`` nor ``strands`` exposes a module-level
    ``__version__`` attribute, so reading ``module.__version__`` yields a
    useless placeholder. The authoritative version lives in the installed
    distribution metadata (``pyproject.toml`` -> wheel/egg-info), which
    ``importlib.metadata.version`` reads. Fall back to a module ``__version__``
    attribute only if metadata lookup fails (e.g. running from a source tree
    that was never installed).

    Args:
        import_name: Importable module name (e.g. ``"strands_robots"``).
        dist_name: Installed distribution name (e.g. ``"strands-robots"``).

    Returns:
        A version string, or ``"unknown"`` if neither source resolves one.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist_name)
    except PackageNotFoundError:
        pass
    try:
        module = importlib.import_module(import_name)
    except ImportError:
        return "unknown"
    return str(getattr(module, "__version__", "unknown"))


def _hf_token_path() -> Path:
    """Path a cached HuggingFace login token is read from.

    ``huggingface_hub`` owns this resolution: ``HF_TOKEN_PATH`` when set, else
    ``<HF_HOME>/token``, where ``HF_HOME`` is ``$HF_HOME`` when set, else
    ``<XDG_CACHE_HOME>/huggingface``, else ``~/.cache/huggingface``. Answering
    about a hardcoded ``~/.cache/huggingface/token`` instead describes a file the
    Hub will not open on any host that relocated its cache, and it is wrong in
    both directions: a relocated token reads as "not logged in", and a stale
    token left at the default path reads as "logged in" for an environment where
    every Hub call resolves no token at all.

    Prefer the constant the Hub computed for itself, so the two cannot drift.
    ``huggingface_hub`` ships only in the extras that pull a checkpoint
    (``[wbc]``, ``[kimodo]``, ``[protomotions]``), so a base install has no Hub
    to ask and the fallback transcribes the same rule. Each variable is read by
    presence rather than truthiness, so an explicitly empty value resolves the
    way the Hub resolves it rather than being treated as unset.

    Returns:
        The token path, with ``~`` and ``$VAR`` expanded.
    """
    try:
        from huggingface_hub.constants import HF_TOKEN_PATH

        return Path(HF_TOKEN_PATH)
    except ImportError:
        # No Hub installed, so nothing here can ask it where it would look;
        # fall through to the transcription of its rule below.
        pass

    default_home = os.path.join(os.path.expanduser("~"), ".cache")
    hf_home = os.path.expandvars(
        os.path.expanduser(
            os.environ.get(
                "HF_HOME",
                os.path.join(os.environ.get("XDG_CACHE_HOME", default_home), "huggingface"),
            )
        )
    )
    return Path(os.path.expandvars(os.path.expanduser(os.environ.get("HF_TOKEN_PATH", os.path.join(hf_home, "token")))))


def check_python_version() -> str:
    """Python >= 3.12 required."""
    v = sys.version_info
    if v >= (3, 12):
        return _pass(f"Python {v.major}.{v.minor}.{v.micro}")
    return _fail(
        f"Python {v.major}.{v.minor}.{v.micro} (need >= 3.12)",
        fix="Install Python 3.12+: https://docs.astral.sh/uv/guides/install-python/",
    )


def check_strands_robots_version() -> str:
    """strands-robots importable and version."""
    try:
        importlib.import_module("strands_robots")
    except ImportError as e:
        return _fail(f"strands-robots not importable: {e}", fix='uv pip install "strands-robots[sim-mujoco]"')
    return _pass(f"strands-robots {_resolve_version('strands_robots', 'strands-robots')}")


def check_mujoco() -> str:
    """MuJoCo importable (sim-mujoco extra)."""
    try:
        import mujoco

        return _pass(f"mujoco {mujoco.__version__}")
    except ImportError:
        return _fail("mujoco not installed", fix='uv pip install "strands-robots[sim-mujoco]"')


def check_mujoco_gl() -> str:
    """MUJOCO_GL names a backend MuJoCo accepts, and one that can render here.

    Answers about the value MuJoCo will read rather than about the string that was
    typed. MuJoCo folds ``MUJOCO_GL`` with ``.lower().strip()``, reads one family
    of values as "build no GL context at all", and raises ``RuntimeError`` at
    import for anything outside the set it builds for the platform. Re-deriving
    that vocabulary loosely answered about a spelling: ``MUJOCO_GL=EGL`` renders
    through EGL and read as unrecognised, while every unrecognised value - a
    disabled GL context, or one MuJoCo refuses outright - read as "not set" and so
    passed on any machine with a display.

    The vocabulary and the display question both belong to the MuJoCo backend
    module, which sets this variable when it is unset, so both are asked there
    rather than restated here.
    """
    from strands_robots.simulation.mujoco.backend import (
        _is_headless,
        _mujoco_gl_disables_rendering,
        _mujoco_gl_offscreen_values,
        _mujoco_gl_valid_values,
        _mujoco_gl_value,
    )

    raw = os.environ.get("MUJOCO_GL", "")
    value = _mujoco_gl_value()
    # Name the value MuJoCo reads whenever folding changed it, so a verdict about
    # ``EGL`` does not read as a verdict about a variable nobody set.
    shown = f"MUJOCO_GL={raw}" if raw == value else f"MUJOCO_GL={raw!r} (MuJoCo reads it as {value!r})"
    valid = _mujoco_gl_valid_values()
    # What to recommend has to be valid here: on a platform whose only backend
    # draws through the window server there is no offscreen value to offer, and
    # naming one would send the reader after a value MuJoCo refuses.
    offscreen_here = sorted(_mujoco_gl_offscreen_values())

    if _mujoco_gl_disables_rendering(value):
        fix = (
            f"export MUJOCO_GL={offscreen_here[0]}  # or unset it for the platform default"
            if offscreen_here
            else "unset MUJOCO_GL  # the platform's own backend renders through its window server"
        )
        return _fail(f"{shown} disables MuJoCo's GL context, so nothing can render", fix=fix)

    if value not in valid:
        offered = ", ".join(sorted(v for v in valid if v))
        return _fail(
            f"{shown} is a value MuJoCo refuses at import on {platform.system()}",
            fix=f"export MUJOCO_GL=<one of: {offered}>  # or unset it for the platform default",
        )

    if value in offscreen_here:
        return _pass(shown)

    # Every remaining accepted value routes MuJoCo to a backend that draws through
    # the platform's window server.
    if value:
        note = (
            f"Set MUJOCO_GL={' or '.join(offscreen_here)} for headless"
            if offscreen_here
            else f"{platform.system()} has no offscreen MuJoCo backend, so a window server is required"
        )
        return _warn(f"{shown} (needs display)", note=note)

    if not _is_headless():
        if platform.system() == "Linux":
            return _pass("MUJOCO_GL unset (display detected, glfw will work)")
        return _pass(f"MUJOCO_GL unset ({platform.system()} renders through its native backend)")
    # ``_is_headless`` is only ever true on Linux, so the remedy below is reached
    # on the one platform where those two backends exist.
    return _fail(
        "MUJOCO_GL not set and no display detected",
        fix="export MUJOCO_GL=egl  # or osmesa; add to ~/.bashrc",
    )


def check_lerobot() -> str:
    """LeRobot importable (lerobot extra)."""
    try:
        import lerobot

        ver = getattr(lerobot, "__version__", "?")
        return _pass(f"lerobot {ver}")
    except ImportError:
        return _warn(
            "lerobot not installed (needed for real hardware + dataset recording)",
            note='uv pip install "strands-robots[lerobot]"',
        )


def check_cuda() -> str:
    """CUDA / GPU availability via torch."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return _pass(f"CUDA available: {name} (torch {torch.__version__})")
        # torch installed but no CUDA
        cuda_ver = getattr(torch.version, "cuda", None)
        if cuda_ver is None:
            return _warn(
                f"torch {torch.__version__} is CPU-only build",
                note="Policy inference will run on CPU. For GPU: install torch with CUDA "
                "(e.g. UV_TORCH_BACKEND=auto uv pip install torch)",
            )
        return _warn(
            f"torch {torch.__version__} has CUDA {cuda_ver} but torch.cuda.is_available()=False",
            note="Check CUDA drivers (nvidia-smi) and CUDA_VISIBLE_DEVICES",
        )
    except ImportError:
        return _warn("torch not installed (needed for policy inference)", note="uv pip install torch")


def check_serial_permissions() -> str:
    """Serial port permissions for real hardware."""
    if platform.system() != "Linux":
        return _skip("serial permissions (non-Linux)")

    # Check if user is in dialout group
    import grp

    username = os.environ.get("USER", "")
    try:
        dialout_members = grp.getgrnam("dialout").gr_mem
    except KeyError:
        return _skip("serial permissions (no dialout group)")

    in_dialout = username in dialout_members
    # Also check effective groups
    try:
        dialout_gid = grp.getgrnam("dialout").gr_gid
        in_effective = dialout_gid in os.getgroups()
    except (KeyError, OSError):
        in_effective = False

    if in_dialout or in_effective:
        # Check if any serial devices exist
        devs = list(Path("/dev").glob("ttyACM*")) + list(Path("/dev").glob("ttyUSB*"))
        if devs:
            # Check read/write permission on first device
            dev = devs[0]
            if os.access(dev, os.R_OK | os.W_OK):
                return _pass(f"serial: user in dialout, {dev} accessible")
            return _fail(
                f"serial: user in dialout but {dev} not accessible",
                fix=f"sudo chmod 666 {dev}  # or add udev rule",
            )
        return _pass("serial: user in dialout (no devices connected)")
    return _fail(
        f"serial: user '{username}' not in dialout group",
        fix="sudo usermod -aG dialout $USER && newgrp dialout  # then re-login",
    )


def check_hf_auth() -> str:
    """HuggingFace Hub authentication (needed for dataset push + gated models)."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return _pass("HF_TOKEN set")
    # Read the file the Hub reads (see _hf_token_path), and name it: on a host
    # that relocated its cache the default path is not the one being consulted,
    # so a verdict that does not say which file it read cannot be acted on.
    hf_token_path = _hf_token_path()
    # ``is_file`` rather than ``exists``: an explicitly empty ``HF_TOKEN_PATH``
    # resolves to ``Path(".")``, a directory that exists, and reading it raises.
    if hf_token_path.is_file() and hf_token_path.read_text().strip():
        return _pass(f"HuggingFace token found ({hf_token_path})")
    return _warn(
        "No HuggingFace token found",
        note=f"Looked in {hf_token_path}. Needed for dataset push + gated models. "
        "Run: hf auth login  # or export HF_TOKEN=hf_...",
    )


def check_sim_smoke() -> str:
    """Run Robot('so100') -> step -> get_observation as a smoke test."""
    try:
        # Suppress mesh warnings during doctor
        os.environ.setdefault("STRANDS_MESH", "false")
        from strands_robots import Robot

        sim = Robot("so100")
        sim.step()
        obs = sim.get_observation("so100")
        if obs and len(obs) > 0:
            return _pass(f"sim smoke test: Robot('so100') works ({len(obs)} obs keys)")
        return _fail("sim smoke test: observation empty")
    except Exception as e:
        return _fail(f"sim smoke test failed: {e}", fix="Check MUJOCO_GL and mujoco install")


def check_strands_agents() -> str:
    """strands-agents importable (needed for Agent(tools=[robot]))."""
    try:
        import strands  # noqa: F401
    except ImportError:
        return _fail("strands-agents not importable", fix='uv pip install "strands-agents>=1.0"')
    return _pass(f"strands-agents {_resolve_version('strands', 'strands-agents')}")


def check_mesh() -> str:
    """Zenoh mesh availability."""
    try:
        import zenoh  # noqa: F401

        return _pass("zenoh available (mesh networking)")
    except ImportError:
        return _warn("zenoh not installed (mesh disabled)", note='uv pip install "strands-robots[mesh]"')


def run_doctor() -> int:
    """Run all checks. Returns 0 if all pass, 1 if any fail."""
    print(_bold("\nstrands-robots doctor"))
    print(_bold("=" * 50))
    print()

    checks = [
        ("Python", check_python_version),
        ("Package", check_strands_robots_version),
        ("Strands SDK", check_strands_agents),
        ("MuJoCo", check_mujoco),
        ("MuJoCo GL", check_mujoco_gl),
        ("LeRobot", check_lerobot),
        ("CUDA/GPU", check_cuda),
        ("Serial", check_serial_permissions),
        ("HF Auth", check_hf_auth),
        ("Mesh", check_mesh),
        ("Sim Test", check_sim_smoke),
    ]

    has_fail = False
    for name, check_fn in checks:
        try:
            result = check_fn()
        except Exception as e:
            result = _fail(f"{name}: unexpected error: {e}")
        # Detect failures via the stable text marker, not the ANSI color code:
        # under NO_COLOR / TERM=dumb the red escape is absent, so gating on it
        # silently swallowed failures and returned exit 0 in CI. The "  FAIL  "
        # prefix is emitted by ``_fail`` in both colored and plain output.
        if "  FAIL  " in result:
            has_fail = True
        print(result)

    print()
    if has_fail:
        print(_red("Some checks failed. Fix the issues above and re-run: python -m strands_robots doctor"))
        return 1
    print(_green("All checks passed. Ready to use strands-robots."))
    return 0


def main() -> None:
    """Console-script entry point: run every check and exit with its status code."""
    sys.exit(run_doctor())


if __name__ == "__main__":
    main()
