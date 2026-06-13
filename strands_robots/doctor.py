"""``strands-robots doctor`` — diagnose common setup issues in one command.

Checks: Python version, extras availability, GPU/CUDA, serial permissions,
MuJoCo GL backend, HuggingFace auth, and a sim smoke test. Prints a colored
pass/fail table so first-time users can fix problems before they hit cryptic
errors at runtime.

Usage:
    python -m strands_robots doctor
    strands-robots doctor        # (after pip install with [scripts] or console_scripts)
"""

from __future__ import annotations

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
        import strands_robots

        ver = getattr(strands_robots, "__version__", "unknown")
        return _pass(f"strands-robots {ver}")
    except ImportError as e:
        return _fail(f"strands-robots not importable: {e}", fix='uv pip install "strands-robots[sim-mujoco]"')


def check_mujoco() -> str:
    """MuJoCo importable (sim-mujoco extra)."""
    try:
        import mujoco

        return _pass(f"mujoco {mujoco.__version__}")
    except ImportError:
        return _fail("mujoco not installed", fix='uv pip install "strands-robots[sim-mujoco]"')


def check_mujoco_gl() -> str:
    """MUJOCO_GL set for headless rendering."""
    gl = os.environ.get("MUJOCO_GL", "")
    if gl in ("egl", "osmesa"):
        return _pass(f"MUJOCO_GL={gl}")
    if gl == "glfw":
        return _warn("MUJOCO_GL=glfw (needs display)", note="Set MUJOCO_GL=egl or osmesa for headless")
    # Not set — check if a display exists
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return _pass("MUJOCO_GL unset (display detected, glfw will work)")
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
    # Check huggingface-cli login token
    hf_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if hf_token_path.exists() and hf_token_path.read_text().strip():
        return _pass("HuggingFace token found (~/.cache/huggingface/token)")
    return _warn(
        "No HuggingFace token found",
        note="Needed for dataset push + gated models. Run: huggingface-cli login  # or export HF_TOKEN=hf_...",
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
        import strands

        ver = getattr(strands, "__version__", "?")
        return _pass(f"strands-agents {ver}")
    except ImportError:
        return _fail("strands-agents not importable", fix='uv pip install "strands-agents>=1.0"')


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
        if "FAIL" in result and "\033[31m" in result:
            has_fail = True
        print(result)

    print()
    if has_fail:
        print(_red("Some checks failed. Fix the issues above and re-run: python -m strands_robots doctor"))
        return 1
    print(_green("All checks passed. Ready to use strands-robots."))
    return 0


def main() -> None:
    sys.exit(run_doctor())


if __name__ == "__main__":
    main()
