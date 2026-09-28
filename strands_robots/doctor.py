"""``strands-robots doctor`` - diagnose common setup issues in one command.

Every row of :data:`CHECKS` is one probe: read-only, sub-second, no network
and no hardware, and it returns the same verdict the runtime would reach on
the same configuration (a PASS here must never precede a refusal there). The
probe's remedy is the runtime's own env-name text where the runtime has one.
Rows cover the interpreter and package, the sim extra and its GL backend,
lerobot and the torch/torchcodec ABI, the GPU and the torch/warp builds for it,
serial permissions, the Hub token, the device-connect and mesh postures, and a
sim smoke test.

Usage:
    python -m strands_robots doctor
    strands-robots doctor        # (after pip install with [scripts] or console_scripts)
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import platform
import re
import sys
import warnings
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
    from strands_robots._mujoco_gl import (
        _MUJOCO_GL_OFFSCREEN_LIBRARIES,
        _is_headless,
        _mujoco_gl_disables_rendering,
        _mujoco_gl_loadable_offscreen_values,
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
    #
    # Valid is not the same as reachable. Each offscreen backend loads a system
    # library, so on a host missing them every offscreen value MuJoCo accepts is
    # one no export can render through, and the remedy is the library rather than
    # a different variable. Both sets are kept, to tell "this platform has no
    # offscreen backend" apart from "this host is missing their libraries".
    accepted_offscreen = sorted(_mujoco_gl_offscreen_values())
    offscreen_here = sorted(_mujoco_gl_loadable_offscreen_values())
    absent_libraries = [
        library
        for value, library in _MUJOCO_GL_OFFSCREEN_LIBRARIES
        if value in accepted_offscreen and value not in offscreen_here
    ]

    def _no_offscreen_remedy(platform_has_none: str) -> str:
        """The remedy when no offscreen backend is reachable.

        Args:
            platform_has_none: What to say when the platform accepts none of them,
                which is a property of the platform rather than of this host.

        Returns:
            The library advice when the platform accepts offscreen backends and
            none of their libraries loads here, ``platform_has_none`` otherwise.
        """
        if accepted_offscreen:
            return (
                f"install an offscreen GL library ({', '.join(absent_libraries)}) - "
                "no exported value can render without one"
            )
        return platform_has_none

    if _mujoco_gl_disables_rendering(value):
        fix = (
            f"export MUJOCO_GL={offscreen_here[0]}  # or unset it for the platform default"
            if offscreen_here
            else _no_offscreen_remedy("unset MUJOCO_GL  # the platform's own backend renders through its window server")
        )
        return _fail(f"{shown} disables MuJoCo's GL context, so nothing can render", fix=fix)

    if value not in valid:
        offered = ", ".join(sorted(v for v in valid if v))
        return _fail(
            f"{shown} is a value MuJoCo refuses at import on {platform.system()}",
            fix=f"export MUJOCO_GL=<one of: {offered}>  # or unset it for the platform default",
        )

    # Classification is the platform's question, not this host's: an offscreen
    # backend is one whether or not its library is installed here, so a value the
    # reader set keeps the verdict it earns. Only what to *recommend* narrows to
    # what can load, which is why this reads the accepted set and the remedies
    # above and below read the reachable one.
    if value in accepted_offscreen:
        return _pass(shown)

    # Every remaining accepted value routes MuJoCo to a backend that draws through
    # the platform's window server.
    if value:
        note = (
            f"Set MUJOCO_GL={' or '.join(offscreen_here)} for headless"
            if offscreen_here
            else _no_offscreen_remedy(
                f"{platform.system()} has no offscreen MuJoCo backend, so a window server is required"
            )
        )
        return _warn(f"{shown} (needs display)", note=note)

    if not _is_headless():
        if platform.system() == "Linux":
            return _pass("MUJOCO_GL unset (display detected, glfw will work)")
        return _pass(f"MUJOCO_GL unset ({platform.system()} renders through its native backend)")
    # ``_is_headless`` is only ever true on Linux, the one platform where those two
    # backends exist - so what remains to decide is which of them this host can
    # actually load.
    if not offscreen_here:
        return _fail(
            "MUJOCO_GL not set and no display detected",
            fix=_no_offscreen_remedy(
                f"{platform.system()} has no offscreen MuJoCo backend, so a window server is required"
            ),
        )
    # The command has to stay runnable, so any second choice goes in the comment.
    alternatives = " or ".join(offscreen_here[1:])
    fix = f"export MUJOCO_GL={offscreen_here[0]}  # "
    fix += f"or {alternatives}; add to ~/.bashrc" if alternatives else "add to ~/.bashrc"
    return _fail("MUJOCO_GL not set and no display detected", fix=fix)


def check_lerobot() -> str:
    """LeRobot importable (lerobot extra).

    The reachability question goes to :func:`~strands_robots.utils.lerobot_install_error`
    rather than to a bare ``import lerobot``: a *directory* named ``lerobot`` on
    the import path imports as an empty namespace package, which passed this
    check as ``lerobot ?`` - the doctor reporting an install for a host that has
    none, with the unresolvable version as the only tell.
    """
    from strands_robots.utils import lerobot_install_error

    problem = lerobot_install_error()
    if problem is not None:
        return _warn(
            f"{problem} (needed for real hardware + dataset recording)",
            note='uv pip install "strands-robots[lerobot]"',
        )

    import lerobot

    ver = getattr(lerobot, "__version__", "?")
    return _pass(f"lerobot {ver}")


#: The dynamic loader's own line inside a torchcodec load failure - what is
#: actually missing - as against the wrapper's "Likely causes:" preamble.
_LOADER_REASON = re.compile(
    r"(?:Symbol not found|Library not loaded|image not found|undefined symbol|cannot open shared object)[^\n]{0,80}"
)


def check_torchcodec_abi() -> str:
    """torchcodec loads against the installed torch and ffmpeg.

    ``import lerobot`` succeeds with a torchcodec that cannot load - the first
    ``LeRobotDataset(...)`` then prints the ~100-line loader traceback and falls
    back to pyav (or, with ``drop_videos=True``, has no fallback and raises). The
    probe imports torchcodec's native ops with stderr captured and reports the
    first line of the refusal, telling an ffmpeg-not-found apart from a
    torch/torchcodec ABI mismatch because the two have different remedies.
    Skips when either wheel is absent: the pair can be installed without lerobot,
    so the row is not gated on it.
    """
    from importlib.metadata import PackageNotFoundError, version

    pair: dict[str, str] = {}
    for dist in ("torch", "torchcodec"):
        try:
            pair[dist] = version(dist)
        except PackageNotFoundError:
            return _skip(f"{dist} not installed (torchcodec ABI check needs torch + torchcodec)")
    label = f"torchcodec {pair['torchcodec']} / torch {pair['torch']}"

    import contextlib
    import io

    try:
        with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            importlib.import_module("torchcodec._core.ops")
    except (ImportError, OSError, RuntimeError) as e:
        text = str(e)
        # The loader's own reason, not torchcodec's "Likely causes:" preamble.
        match = _LOADER_REASON.search(text)
        first = (
            match.group(0) if match else next((ln.strip() for ln in text.splitlines() if ln.strip()), type(e).__name__)
        )
        if "libav" in text:
            from strands_robots import _dyld

            ffmpeg_dir = _dyld._find_ffmpeg_lib_dir() if sys.platform == "darwin" else None
            if ffmpeg_dir:
                fix = f"export {_dyld._DYLD_VAR}={ffmpeg_dir} (ffmpeg is installed; dyld cannot see it from this shell)"
            else:
                fix = "brew install ffmpeg (macOS) / apt install ffmpeg (Linux); torchcodec supports ffmpeg 4-8"
            return _fail(f"{label}: ffmpeg shared libraries not found ({first})", fix=fix)
        return _fail(
            f"{label}: torchcodec was built for a different torch ({first})",
            fix="install the torchcodec that matches this torch - "
            "https://github.com/pytorch/torchcodec#installing-torchcodec",
        )
    return _pass(f"{label} loads")


def cuda_devices_per_driver() -> int | None:
    """How many CUDA devices the driver reports, asked without torch.

    ``libcuda.so.1`` is the driver's own library and is present wherever a CUDA
    device is - on a Jetson it ships with L4T, never through pip - so asking it
    is what lets this command tell "no GPU" from "no torch". Falls back to the
    kernel module's per-GPU directory. ``None`` means no driver was found, which
    is a real "no CUDA device", not an unanswered question.
    """
    try:
        import ctypes

        lib = ctypes.CDLL("libcuda.so.1")
        count = ctypes.c_int(0)
        if lib.cuInit(0) == 0 and lib.cuDeviceGetCount(ctypes.byref(count)) == 0:
            return count.value
    except OSError:
        # No libcuda.so.1 on the loader path. That is the expected shape of a
        # machine without a driver (and of a container the driver is not
        # mounted into), not an error to report: the /proc fallback below is
        # the second opinion, and None is the honest answer when it is empty.
        pass
    gpus = Path("/proc/driver/nvidia/gpus")
    return len(list(gpus.iterdir())) if gpus.is_dir() else None


def _torch_cuda_remedy() -> str:
    """The install command that yields a CUDA torch on THIS machine.

    PyPI's ``linux_aarch64`` torch wheel carries CUDA from 2.11 on: 2.11.0 is
    420 MB and its ``nvidia-*-cu13`` dependencies apply to any Linux, while
    2.10.0 was a 146 MB CPU build whose CUDA dependencies were marked
    ``platform_machine == "x86_64"`` (2.9.1: 104 MB) - the same fact behind this
    project's aarch64 ``torch>=2.11`` requirement. So the generic command is
    right on x86_64 and on a Jetson whose L4T ships a CUDA 13 driver (R38,
    JetPack 7). A JetPack 6 board (R36) has a 12.6 driver that cannot load those
    wheels; its torch comes from NVIDIA's Jetson index. A release this cannot
    read as ``R<major>`` gets the generic command rather than a confidently
    wrong index - and the major is compared as a number, so an R100 board is not
    sent to the JetPack 6 index the way ``"R100" < "R38"`` would send it.
    """
    tegra = Path("/etc/nv_tegra_release")
    if platform.machine() == "aarch64" and tegra.exists():
        release = tegra.read_text(encoding="utf-8", errors="replace").split(",")[0].strip("# ").split(" ")[0]
        major = int(release[1:]) if release.startswith("R") and release[1:].isdigit() else None
        if major is not None and major < 38:
            return (
                f"Jetson L4T {release} ships CUDA 12.6, which PyPI's CUDA 13 aarch64 torch cannot use: "
                "uv pip install torch --index-url https://pypi.jetson-ai-lab.io/jp6/cu126"
            )
        if major is not None:
            return f"uv pip install torch  (PyPI's aarch64 wheel carries CUDA 13; L4T {release} supports it)"
    return "UV_TORCH_BACKEND=auto uv pip install torch"


def check_cuda() -> str:
    """CUDA / GPU availability: the driver's answer first, then torch's."""
    devices = cuda_devices_per_driver()
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return _pass(f"CUDA available: {name} (torch {torch.__version__})")
        # torch installed but no CUDA
        cuda_ver = getattr(torch.version, "cuda", None)
        if cuda_ver is None:
            if devices:
                return _warn(
                    f"torch {torch.__version__} is a CPU-only build, but the driver reports {devices} CUDA device(s)",
                    note=_torch_cuda_remedy(),
                )
            return _warn(
                f"torch {torch.__version__} is CPU-only build",
                note="Policy inference will run on CPU (no CUDA device found on this machine)",
            )
        return _warn(
            f"torch {torch.__version__} has CUDA {cuda_ver} but torch.cuda.is_available()=False",
            note="Check CUDA drivers (nvidia-smi) and CUDA_VISIBLE_DEVICES",
        )
    except ImportError:
        if devices:
            return _warn(
                f"torch not installed, but the driver reports {devices} CUDA device(s) (needed for policy inference)",
                note=_torch_cuda_remedy(),
            )
        return _warn("torch not installed (needed for policy inference)", note="uv pip install torch")


def _driver_compute_arch() -> int | None:
    """Architecture the CUDA driver reports for device 0, as an ``sm_NN`` integer.

    Read through ``torch.cuda.get_device_capability``, which queries the driver
    for the device's own properties rather than reporting anything about the
    build torch was compiled for.

    Returns:
        The device architecture (``110`` for an ``sm_110`` GPU), or ``None`` when
        torch cannot see a CUDA device - including when torch is not installed,
        which is why callers pair this with :func:`cuda_devices_per_driver` to
        tell a missing device from a missing torch.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    return major * 10 + minor


def _warp_cuda_report() -> tuple[tuple[int, ...], int, tuple[int, int], tuple[int, int]] | None:
    """What the installed Warp build reports about CUDA device 0.

    Returns:
        A ``(supported_archs, chosen_arch, toolkit_version, driver_version)``
        tuple, or ``None`` when Warp is not installed - it ships in the
        ``sim-newton`` extra - or when it can see no CUDA device to report on.
    """
    try:
        warp = importlib.import_module("warp")
    except ImportError:
        return None
    if not warp.is_cuda_available() or warp.get_cuda_device_count() < 1:
        return None
    supported = tuple(int(arch) for arch in warp.get_cuda_supported_archs())
    chosen = int(warp.get_device("cuda:0").arch)
    toolkit = warp.get_cuda_toolkit_version()
    driver = warp.get_cuda_driver_version()
    return supported, chosen, (int(toolkit[0]), int(toolkit[1])), (int(driver[0]), int(driver[1]))


def _arch_entry_version(entry: str) -> int | None:
    """The architecture an entry of torch's arch list names, as an ``sm_NN`` integer.

    torch spells its entries ``sm_90`` and ``compute_80``, and appends a family
    suffix for architecture-specific code (``sm_90a``, ``sm_120f``). That spelling
    is torch's, so prefer torch's own parser and fall back to reading the
    documented shape only on an install too old to expose it.

    Args:
        entry: One element of :func:`torch.cuda.get_arch_list`.

    Returns:
        The architecture as an integer (``90`` for ``sm_90a``), or ``None`` when
        the entry does not carry one.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - the caller resolved a report first
        parse = None
    else:
        parse = getattr(torch.cuda, "_extract_arch_version", None)
    if parse is not None:
        try:
            return int(parse(entry))
        except (IndexError, ValueError):
            return None
    parts = entry.split("_", maxsplit=2)
    if len(parts) < 2:
        return None
    digits = parts[1].removesuffix("a").removesuffix("f")
    return int(digits) if digits.isdigit() else None


def _torch_cuda_report() -> tuple[tuple[int, ...], str] | None:
    """The architectures the installed torch build carries, and its version.

    Returns:
        A ``(build_archs, version)`` tuple, or ``None`` when torch is not
        installed - it ships in the extras that run a policy - or when it cannot
        report an arch list at all.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        entries = torch.cuda.get_arch_list()
    except Exception:
        return None
    archs = tuple(arch for arch in (_arch_entry_version(entry) for entry in entries) if arch is not None)
    return archs, str(torch.__version__)


def _torch_build_supports(device_arch: int, build_archs: tuple[int, ...]) -> bool:
    """Whether torch's own compatibility rule admits this device for this build.

    The rule is torch's rather than one derived here, so this check and torch
    cannot reach opposite verdicts about one install. ``_code_compatible_with_device``
    is the predicate torch's own capability check consults, and it reads a table
    of intervals: a build carrying ``sm_80`` code supports ``>=8.0,<9.0 except
    {8.7}``, so it does *not* cover a Jetson Orin. The coarser
    backward-compatible-within-a-major-version rule that torch's cubin check
    documents admits that pair, which is why the interval table is preferred and
    the coarse rule is only the fallback for an install too old to expose it.

    Args:
        device_arch: Architecture the driver reports, as an ``sm_NN`` integer.
        build_archs: Architectures the installed build carries code for.

    Returns:
        ``True`` when some entry of the build covers the device.
    """
    compatible = None
    try:
        import torch
    except ImportError:  # pragma: no cover - the caller resolved a report first
        pass
    else:
        compatible = getattr(torch.cuda, "_code_compatible_with_device", None)

    if compatible is None:
        return any(arch // 10 == device_arch // 10 for arch in build_archs)

    # The predicate warns when a build names an architecture its own table does
    # not know - a note about the table's age rather than about this device. The
    # verdict below carries that answer, so the note is suppressed rather than
    # printed beside it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return any(bool(compatible(device_arch, arch)) for arch in build_archs)


def _torch_compatible_releases(device_arch: int) -> tuple[str, ...]:
    """CUDA versions whose torch release carries code this device can run.

    Read from the table torch ships for its own remedy, so the versions offered
    are the ones torch would name rather than a guess about what PyPI serves.

    Args:
        device_arch: Architecture the driver reports, as an ``sm_NN`` integer.

    Returns:
        The CUDA versions, or an empty tuple when torch ships no such table.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - the caller resolved a report first
        return ()
    table = getattr(torch.cuda, "PYTORCH_RELEASES_CODE_CC", None)
    if not isinstance(table, dict):
        return ()
    return tuple(str(cuda) for cuda, build_ccs in table.items() if _torch_build_supports(device_arch, tuple(build_ccs)))


def check_torch_arch() -> str:
    """torch's build carries CUDA code for the GPU architecture the driver reports.

    A torch build that carries no code for the device runs no CUDA kernel at all,
    and ``torch.cuda.is_available()`` is ``True`` regardless - it answers about the
    driver, not about the build. So :func:`check_cuda` passes on such an install,
    naming the device and the version, and the only signal is a ``warnings.warn``
    torch emits on its first CUDA call: it goes to stderr while this table goes to
    stdout, it is shown once per process, and it lands beside this command's "All
    checks passed" verdict - the one place a new reader looks to find out whether
    their setup is sound.

    The verdict is torch's own (see :func:`_torch_build_supports`), so this check
    reports what torch already decided rather than deciding it again. It is
    reported as a failure rather than a warning because the consequence is total:
    where Warp substitutes a nearby architecture and keeps returning right answers
    for simple kernels, torch's own words for this build are "not compatible".
    """
    device_arch = _driver_compute_arch()
    if device_arch is None:
        if cuda_devices_per_driver():
            return _skip("torch arch: a CUDA device is present but torch cannot see it (see CUDA line)")
        return _skip("torch arch: no CUDA device to compare against")
    report = _torch_cuda_report()
    if report is None:
        return _skip("torch arch: torch not installed (needed for policy inference)")
    build_archs, version = report
    if not build_archs:
        return _skip(f"torch arch: torch {version} reports no CUDA architectures")

    offered = " ".join(f"sm_{arch}" for arch in sorted(set(build_archs)))
    if _torch_build_supports(device_arch, build_archs):
        return _pass(f"torch {version} carries code for sm_{device_arch} (build: {offered})")

    releases = _torch_compatible_releases(device_arch)
    fix = "Install a torch build for this GPU: https://pytorch.org/get-started/locally/"
    if releases:
        fix = (
            f"Install a torch release built against CUDA {', '.join(releases)} - torch's own table names "
            f"those as carrying code for sm_{device_arch}: https://pytorch.org/get-started/locally/"
        )
    return _fail(
        f"torch {version} carries code for {offered}, and torch's own compatibility check reports "
        f"this sm_{device_arch} device as not compatible with that build",
        fix=fix,
    )


def check_warp_arch() -> str:
    """Warp's build can compile for the GPU architecture the driver reports.

    Warp chooses a device's architecture out of the table its binary was built
    with, so a build older than the GPU compiles for a different architecture
    rather than refusing. On an NVIDIA Thor - the driver reports ``sm_110`` - the
    CUDA-12 build offers a table with no ``110`` in it and settles on ``sm_101``,
    and nothing says so: a simple kernel still returns the right answer, so the
    substitution surfaces only once a kernel needs something ``sm_101`` cannot
    express.

    The verdict is about the arch table Warp reports rather than about the wheel
    the environment was installed from. A wheel tag cannot answer it: PyPI's
    filenames carry no local version segment, so one release's CUDA-12 and
    CUDA-13 builds are indistinguishable by name, while the table is the value
    Warp itself reads.

    torch's arch table is asked by :func:`check_torch_arch` rather than here,
    because the two builds fail differently: Warp substitutes a nearby
    architecture and keeps computing, while torch's own compatibility check
    reports the device as not compatible with the build.
    """
    device_arch = _driver_compute_arch()
    if device_arch is None:
        if cuda_devices_per_driver():
            return _skip("Warp arch: a CUDA device is present but torch, which reads its architecture, cannot see it")
        return _skip("Warp arch: no CUDA device to compare against")
    report = _warp_cuda_report()
    if report is None:
        return _skip("Warp arch: warp not installed (sim-newton extra)")
    supported, chosen, toolkit, driver = report
    built_for = f"CUDA {toolkit[0]}.{toolkit[1]}"

    if device_arch in supported:
        return _pass(f"Warp targets sm_{device_arch} (built against {built_for})")

    # Name the arch Warp settled on rather than deriving it. The substitution is
    # not the nearest supported arch below the device's - a table offering both
    # 101 and 103 settled on 101 for an sm_110 device - so a computed value here
    # would describe a rule Warp does not follow.
    return _fail(
        f"Warp is built against {built_for}, whose arch table has no sm_{device_arch}, "
        f"so it compiles for sm_{chosen} on this sm_{device_arch} device",
        fix=(
            f"Install the Warp build for CUDA {driver[0]}, the driver's own version: take the wheel "
            f"whose version carries a '+cu{driver[0]}' tag from "
            "https://github.com/NVIDIA/warp/releases (pip install --force-reinstall --no-deps <url>). "
            "PyPI serves one build per release, so its wheel may not be the one this driver needs."
        ),
    )


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
    if hf_token_path.is_file() and hf_token_path.read_text(encoding="utf-8").strip():
        return _pass(f"HuggingFace token found ({hf_token_path})")
    return _warn(
        "No HuggingFace token found",
        note=f"Looked in {hf_token_path}. Needed for dataset push + gated models. "
        "Run: hf auth login  # or export HF_TOKEN=hf_...",
    )


def check_sim_smoke() -> str:
    """Run Robot('so100') -> step -> get_observation -> release as a smoke test.

    The release is part of what this check verifies, not housekeeping after it.
    ``get_observation`` opens a MuJoCo GL context to render the robot's cameras,
    and a context left to the finalizer is freed during interpreter teardown -
    after EGL has already been de-initialised - so MuJoCo's own
    ``Renderer.__del__`` writes an ``Exception ignored in`` traceback to stderr.
    That lands beside this command's "All checks passed" verdict, which is the
    one place a new reader looks for a signal that their setup is sound.

    ``cleanup`` is the release verb rather than ``with``: it is the one every
    object :func:`strands_robots.Robot` can return carries, because the hardware
    wrapper implements no context-manager protocol.
    """
    try:
        # Suppress mesh warnings during doctor
        os.environ.setdefault("STRANDS_MESH", "false")
        from strands_robots.robot import Robot

        sim = Robot("so100")
        try:
            sim.step()
            obs = sim.get_observation("so100")
            if obs and len(obs) > 0:
                return _pass(f"sim smoke test: Robot('so100') works ({len(obs)} obs keys)")
            return _fail("sim smoke test: observation empty")
        finally:
            # Released here rather than left to the finalizer, and deliberately
            # not swallowed: a sim that cannot be released is a real defect on
            # this machine, and the verdict this check reports covers the whole
            # lifecycle it drives.
            sim.cleanup()
    except Exception as e:
        return _fail(f"sim smoke test failed: {e}", fix="Check MUJOCO_GL and mujoco install")


def check_strands_agents() -> str:
    """strands-agents importable (needed for Agent(tools=[robot]))."""
    try:
        import strands  # noqa: F401
    except ImportError:
        return _fail(
            "strands-agents not importable",
            # The bound is the one pyproject declares. A remedy that names a
            # lower floor is satisfied by a release this package refuses, so
            # pip reports "Requirement already satisfied" against a stale
            # environment and the diagnosis this line answers survives it.
            fix='uv pip install "strands-agents>=1.57.0,<2.0.0"',
        )
    return _pass(f"strands-agents {_resolve_version('strands', 'strands-agents')}")


def check_device_connect() -> str:
    """The posture ``Robot(...).run()`` would take when it brings the device online.

    ``run()`` decides at start between four outcomes and this row names the one
    the current environment selects, through the runtime's own decision functions
    (``resolve_allow_insecure`` and ``transport_is_authenticated``) so the verdict
    cannot drift from the decision: the extra is missing (SKIP,
    with the install line), TLS is configured (PASS), no TLS and no opt-in
    (WARN - ``run()`` refuses), or the insecure opt-in (WARN - plaintext on the
    LAN; FAIL when ``DEVICE_CONNECT_RPC_ALLOW`` is also empty, because then any
    peer may call ``execute``/``stop``). Never opens a session and never reads a
    credentials file.
    """
    from strands_robots.device_connect import _authz

    try:
        impl = importlib.import_module("strands_robots.device_connect._impl")
    except ImportError as e:
        return _skip(
            f'device-connect extra not installed ({e.name or e}); uv pip install "strands-robots[device-connect]"'
        )

    backend = os.environ.get("MESSAGING_BACKEND", "zenoh")
    allow_insecure = impl.resolve_allow_insecure(None, os.environ.get(_authz._INSECURE_ENV))
    if impl.transport_is_authenticated(backend, None):
        configured = [name for name in impl._TLS_ENV if os.environ.get(name)]
        how = f"credentials via {configured[0]}" if configured else "a TLS endpoint scheme"
        return _pass(f"device-connect ({backend}): authenticated transport, {how}")
    if not allow_insecure:
        return _warn(
            f"device-connect ({backend}): run() will refuse - no TLS configured and no opt-in",
            note=f"MESSAGING_CREDENTIALS_FILE=<bundle>.creds.json, or {_authz._INSECURE_ENV}=true on an isolated network",
        )
    if not os.environ.get(_authz._RPC_ALLOW_ENV, "").strip():
        return _fail(
            f"device-connect ({backend}): run() will be online UNENCRYPTED and any peer may call execute/stop",
            fix=f"{_authz._RPC_ALLOW_ENV}=<caller-id,...> restricts callers; unset {_authz._INSECURE_ENV} to require TLS",
        )
    return _warn(
        f"device-connect ({backend}): run() will be online UNENCRYPTED on the LAN ({_authz._INSECURE_ENV} set)",
        note=f"callers restricted by {_authz._RPC_ALLOW_ENV}",
    )


def _mesh_port() -> tuple[int, str]:
    """The port a zenoh session would listen on, and why it is not the configured one.

    ``open_session`` parses ``STRANDS_MESH_PORT`` with a 1-65535 range check and
    warns once before falling back to 7447 rather than raising, so a value the
    runtime will not use must not be printed here as the hub.

    Returns:
        The port the runtime would use, and the reason a configured value was
        rejected (empty when the value was taken as written).
    """
    raw = os.environ.get("STRANDS_MESH_PORT", "7447")
    try:
        port = int(raw)
        if not (1 <= port <= 65535):
            raise ValueError(f"port {port} out of range")
    except ValueError as exc:
        return 7447, f"STRANDS_MESH_PORT={raw!r} ({exc}) - the runtime warns and falls back to 7447"
    return port, ""


def _tcp_reachable(host: str, port: int, timeout_s: float) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _listener_owner(port: int) -> str:
    """``pid/command`` of the process listening on ``127.0.0.1:port``, or ``""``."""
    import shutil
    import subprocess

    lsof = shutil.which("lsof")
    if not lsof:
        return ""
    try:
        out = subprocess.run(
            [lsof, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpc"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=1.0,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    pid = cmd = ""
    for line in out.splitlines():
        if line.startswith("p") and not pid:
            pid = line[1:]
        elif line.startswith("c") and not cmd:
            cmd = line[1:]
    return f"{pid}/{cmd}" if pid else ""


def check_mesh() -> str:
    """Whether ``Robot(..., mesh=True)`` would start, and what it would join.

    ``import zenoh`` succeeding is not a mesh: with the default mTLS mode and
    the built-in permissive ACL, ``Mesh.start`` logs an ERROR and hands back a
    session that never opens, so ``emergency_stop()`` reaches nobody. This row
    runs the same gate (``resolve_auth_mode`` + ``snapshot_acl``) without
    opening zenoh and prints the runtime's own "Pick one" text on refusal; then
    it reports the hub port the runtime would really bind (free: this process
    becomes the local router; bound: who owns it - 7447 is zenoh's default, so a
    foreign ``zenohd`` becomes the hub silently), each ``ZENOH_CONNECT``
    endpoint's reachability
    with a one-second deadline, and whether LAN multicast scouting is on. The
    mesh is opt-in (``STRANDS_MESH`` / ``mesh=True``), so a refusal is a WARN
    unless the operator has turned it on, when it is the FAIL ``run()`` would hit.
    """
    try:
        import zenoh  # noqa: F401
    except ImportError:
        return _warn("zenoh not installed (mesh disabled)", note='uv pip install "strands-robots[mesh]"')

    from strands_robots.mesh import _acl_config, _zenoh_config
    from strands_robots.mesh.core import PERMISSIVE_ACL_REFUSAL

    mesh_requested = os.environ.get("STRANDS_MESH", "").strip().lower() in ("1", "true", "yes")
    refuse = _fail if mesh_requested else _warn
    lines: list[str] = []

    # (1) the same gate Mesh.start runs, in the same order.
    try:
        auth_mode = _zenoh_config.resolve_auth_mode()
        is_permissive, _resolved = _acl_config.snapshot_acl(_zenoh_config.resolve_namespace())
    except ValueError as e:
        return refuse(f"mesh=True would not start: {e}")
    if auth_mode == "none":
        lines.append("local-dev (plaintext, localhost only)")
    elif is_permissive and not _acl_config.permissive_acl_acknowledged():
        why, pick_one = PERMISSIVE_ACL_REFUSAL.split("\n", 1)
        return refuse(
            f"mesh=True would not start: {why.split(': ', 1)[1]}", pick_one.strip().replace("\n  ", "\n        ")
        )
    else:
        missing = [
            name
            for name in ("STRANDS_MESH_TLS_CA", "STRANDS_MESH_TLS_CERT", "STRANDS_MESH_TLS_KEY")
            if not os.environ.get(name, "").strip()
        ]
        if missing:
            return refuse(
                "mesh=True would not start: STRANDS_MESH_AUTH_MODE=mtls needs certificates",
                f"set {', '.join(missing)}",
            )
        acl = os.environ.get("STRANDS_MESH_ACL_FILE", "").strip() or "built-in permissive ACL (acknowledged)"
        lines.append(f"mtls + ACL {acl}")

    # (2) hub port.
    port, port_note = _mesh_port()
    owner = _listener_owner(port) if _tcp_reachable("127.0.0.1", port, 0.2) else ""
    if owner:
        lines.append(f"hub 127.0.0.1:{port} owned by {owner} (this process would join it as a client)")
    elif _tcp_reachable("127.0.0.1", port, 0.2):
        lines.append(f"hub 127.0.0.1:{port} bound by another process")
    else:
        lines.append(f"hub 127.0.0.1:{port} free (this process would be the local router)")

    # (3) explicit endpoints, one second each.
    unreachable: list[str] = []
    for endpoint in (e.strip() for e in os.environ.get("ZENOH_CONNECT", "").split(",") if e.strip()):
        hostport = endpoint.split("/", 1)[-1]
        host, _, port_s = hostport.rpartition(":")
        if not host or not port_s.isdigit() or not _tcp_reachable(host.strip("[]"), int(port_s), 1.0):
            unreachable.append(endpoint)

    # (4) scouting.
    multicast = _zenoh_config._bool_env("STRANDS_MESH_MULTICAST", default=False)
    lines.append("LAN multicast scouting 224.0.0.224:7446 ON" if multicast else "gossip-only")

    notes = [
        note
        for note in (
            port_note,
            f"ZENOH_CONNECT unreachable: {', '.join(unreachable)}" if unreachable else "",
            "every zenoh app on the LAN sees this peer (STRANDS_MESH_MULTICAST=true)" if multicast else "",
        )
        if note
    ]
    summary = "; ".join(lines)
    return _warn(f"mesh: {summary}", note="; ".join(notes)) if notes else _pass(f"mesh: {summary}")


#: The doctor's table: one ``(label, probe)`` row per check, in print order.
#: Probes are looked up by name at run time so a test (or an operator's
#: ``python -c``) can replace one without rebuilding the table.
CHECKS: tuple[tuple[str, str], ...] = (
    ("Python", "check_python_version"),
    ("Package", "check_strands_robots_version"),
    ("Strands SDK", "check_strands_agents"),
    ("MuJoCo", "check_mujoco"),
    ("MuJoCo GL", "check_mujoco_gl"),
    ("LeRobot", "check_lerobot"),
    ("Torchcodec", "check_torchcodec_abi"),
    ("CUDA/GPU", "check_cuda"),
    ("Torch Arch", "check_torch_arch"),
    ("Warp Arch", "check_warp_arch"),
    ("Serial", "check_serial_permissions"),
    ("HF Auth", "check_hf_auth"),
    ("Device Connect", "check_device_connect"),
    ("Mesh", "check_mesh"),
    ("Sim Test", "check_sim_smoke"),
)


def run_doctor() -> int:
    """Run every row of :data:`CHECKS`. Returns 0 if all pass, 1 if any fail."""
    print(_bold("\nstrands-robots doctor"))
    print(_bold("=" * 50))
    print()

    has_fail = False
    for name, probe in CHECKS:
        try:
            result = globals()[probe]()
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


def _parser() -> argparse.ArgumentParser:
    """The ``strands-robots doctor`` argument parser: ``--list`` or nothing."""
    parser = argparse.ArgumentParser(
        prog="strands-robots doctor",
        description="Check this machine for a working strands-robots install.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the check names and exit without probing anything",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point: run every check and exit with its status code.

    ``--help`` and ``--list`` exit before any probe runs, and an argument the
    parser does not know exits 2 with the usage line instead of being ignored
    while the full check runs under it.
    """
    args = _parser().parse_args(argv)
    if args.list:
        for name, _probe in CHECKS:
            print(name)
        sys.exit(0)
    sys.exit(run_doctor())


if __name__ == "__main__":
    main()
