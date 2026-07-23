#!/usr/bin/env python3
"""LIBERO backend matrix -- same task, every installed backend, side-by-side.

The flagship demo of this repo. Runs one LIBERO task on whichever of
the per-backend driver scripts (``run_mujoco.py``, ``run_isaac.py``,
``run_isaac_fleet.py``) the host can actually execute, and prints a
single side-by-side table
with ``success_rate`` and ``wall_time`` per backend. Missing backends
are skipped gracefully -- a CPU-only laptop with just MuJoCo installed
should produce a table where every Isaac row reads ``unavailable``
and only the MuJoCo row carries a measurement.

Why a subprocess-and-parse driver, not direct ``create_simulation``?
--------------------------------------------------------------------
The per-backend driver files (``examples/libero/run_<backend>.py``)
already encode every backend-specific quirk needed to make a LIBERO
eval succeed: GR00T container lifecycle, scene pre-warm, real-vs-
procedural robot loading, container-name disambiguation across MuJoCo
and Isaac runs on the same host, etc. Reproducing all of that inline
here would either drift from those files (regressions visible only in
the matrix run) or vendor them (~200 LOC of duplicated setup). The
matrix script instead trusts the per-backend file as the single source
of truth and parses its two grep-stable output lines:

* ``benchmark_name=<task>``
* ``policy=<...>  task=<...>  success_rate=<float>  wall_time=<float>s  videos=<path>``

This contract is documented in ``examples/README.md`` under "Two
execution patterns" and survives every per-backend file's rebases.

Detection
---------
For each backend row, the matrix script:

1. Checks that the per-backend driver file exists in this checkout.
   Missing → ``unavailable (file missing)``. (E.g.
   ``run_isaac_fleet.py`` doesn't land until R23 / `#27`.)
2. Spawns ``python <file> --policy=mock --n-episodes=N --seed=S``.
   If that exits non-zero, captures the last few stderr lines and
   marks the row ``skip (<reason>)``. The most common skip reason is
   the backend's ``is_available()`` short-circuit firing on a host
   that lacks Isaac Sim / CUDA / a CDN-reachable assets root.
3. On success, parses the two grep-stable lines and records
   ``(label, success_rate, wall_time, "ok")``.
4. After every row runs, prints a fixed-width table with one row per
   backend slot.

The output is stable enough that a follow-up CI job can parse it for
regression tracking without further plumbing.

Stage 4 of `#8 <https://github.com/strands-labs/robots-sim/issues/8>`_;
filed as `R15 / #22 <https://github.com/strands-labs/robots-sim/issues/22>`_.

Usage
-----
::

    # 1) Whatever's installed -- the typical path. Mock policy by
    #    default, so no GR00T container needed; missing backends are
    #    cleanly skipped:
    python examples/libero/libero_backend_matrix.py

    # 2) Real eval against the matching `libero_<suite>/` GR00T
    #    sub-checkpoint. Requires the per-backend setup each
    #    `run_<backend>.py` documents (HF token, GPU, etc.):
    python examples/libero/libero_backend_matrix.py --policy groot

    # 3) Limit which backend rows are attempted (faster smoke runs):
    python examples/libero/libero_backend_matrix.py --backends mujoco

Install combinations
--------------------
The script never imports backend modules itself, so missing backends
never crash it -- they just produce ``unavailable`` rows. The
underlying per-backend files do import their backends, and the
combinations that produce non-skip rows are::

    # Just MuJoCo (mujoco row only -- isaac rows skip):
    pip install 'strands-robots[sim-mujoco,benchmark-libero]'

    # + Isaac Sim single-env + fleet (isaac-1, isaac-4096 rows):
    pip install 'strands-robots[sim-isaac,benchmark-libero]'

Verification status
-------------------
CLI / subprocess-and-parse / table-printing logic is verified on a
CPU-only dev box where every ``run_*`` driver is expected to short-
circuit on Isaac ``is_available()`` calls and only the
``mujoco`` row produces a measurement. Full multi-backend numbers
(the actual side-by-side table this script exists to deliver) land
once R8 (`#15`) and R23 (`#27`) finish wiring their data planes;
``examples/README.md`` and the umbrella issue `#8` track those
landings.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Per-backend driver files, in matrix-row order. Each tuple is
# (display_label, driver_filename, extra_argv). The driver_filename
# is resolved relative to this script's directory; extra_argv lets a
# row pass backend-specific flags (e.g., the fleet rows force a
# 4096-env count).
#
# Driver files that don't yet exist on disk produce an
# ``unavailable (file missing)`` row -- the matrix script doesn't
# need them all to land at once. R5 / R8 / R23 / R12 each add their
# own driver as part of the staged plan tracked in #8.
_BACKEND_ROWS: list[tuple[str, str, list[str]]] = [
    ("mujoco", "run_mujoco.py", []),
    ("isaac-1", "run_isaac.py", []),
    ("isaac-4096", "run_isaac_fleet.py", []),
]

# The grep-stable result line that every per-backend driver produces.
# Kept in sync with the spec in examples/README.md "Two execution
# patterns" + the `print(...)` calls at the tail of run_mujoco.py and
# run_isaac.py. If those drift, the parser surfaces an empty
# ``success_rate`` rather than crashing -- the row will read ``ok``
# with ``--`` cells and a stderr hint.
_RE_RESULT = re.compile(
    r"^policy=\S+\s+task=\S+\s+" r"success_rate=(?P<sr>[0-9]+\.[0-9]+)\s+" r"wall_time=(?P<wt>[0-9]+\.[0-9]+)s\b",
    re.MULTILINE,
)


@dataclass
class RowResult:
    """One row of the backend-matrix table."""

    label: str
    success_rate: float | None
    wall_time: float | None
    status: str  # "ok" | "unavailable (...)" | "skip (...)"


def _find_driver(driver_filename: str) -> Path | None:
    """Locate ``examples/libero/<driver_filename>`` if it exists."""
    here = Path(__file__).resolve().parent
    candidate = here / driver_filename
    return candidate if candidate.is_file() else None


def _parse_driver_output(stdout: str) -> tuple[float | None, float | None]:
    """Extract (success_rate, wall_time) from a driver's stdout.

    Returns ``(None, None)`` if the result line is missing -- the
    driver may have exited early before the eval started (e.g., HF
    token missing for ``--policy=groot``) and the matrix script will
    still mark the row ``ok`` but with ``--`` cells; the user can
    re-run the offending driver standalone to see the full traceback.
    """
    m = _RE_RESULT.search(stdout)
    if not m:
        return None, None
    try:
        return float(m["sr"]), float(m["wt"])
    except (TypeError, ValueError):
        return None, None


def _short_skip_reason(stderr: str, returncode: int) -> str:
    """Boil a driver's stderr down to a one-line skip reason.

    Heuristic: scan the last ~20 stderr lines for the first marker
    that looks like an availability short-circuit (``not available``,
    ``ImportError``, ``ModuleNotFoundError``, ``isaac-sim``, etc.) and
    surface that single line. Falls back to the exit code so the
    table stays informative even when the driver crashed silently.
    """
    tail_lines = [ln for ln in stderr.strip().splitlines()[-20:] if ln.strip()]
    needles = (
        "is not available",
        "not available",
        "ImportError",
        "ModuleNotFoundError",
        "isaac-sim",
        "Isaac Sim",
        "CUDA",
        "no module named",
        "is_available",
    )
    for ln in tail_lines:
        if any(n.lower() in ln.lower() for n in needles):
            return ln.strip()[:120]
    if tail_lines:
        return tail_lines[-1][:120]
    return f"exit={returncode}"


def _run_one(
    label: str,
    driver_path: Path,
    extra_argv: list[str],
    *,
    policy: str,
    task: str,
    n_episodes: int,
    seed: int,
    port: int,
    timeout: float,
) -> RowResult:
    """Spawn one driver subprocess and parse its result."""
    argv = [
        sys.executable,
        str(driver_path),
        "--policy",
        policy,
        "--task",
        task,
        "--n-episodes",
        str(n_episodes),
        "--seed",
        str(seed),
        "--port",
        str(port),
        *extra_argv,
    ]
    env = os.environ.copy()
    # Headless rendering on a no-display dev box. Drivers that
    # already export these are unaffected; drivers that don't (e.g.,
    # bare ``run_mujoco.py``) need them or they crash on the first
    # `mjr_makeContext` call.
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")

    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return RowResult(label, None, None, f"skip (timeout > {int(timeout)}s)")
    except FileNotFoundError:
        return RowResult(label, None, None, "skip (python not found)")

    if completed.returncode != 0:
        return RowResult(label, None, None, f"skip ({_short_skip_reason(completed.stderr, completed.returncode)})")

    sr, wt = _parse_driver_output(completed.stdout)
    return RowResult(label, sr, wt, "ok")


def _print_table(rows: list[RowResult], *, task: str, n_episodes: int, seed: int) -> None:
    """Print the side-by-side matrix table to stdout.

    Format is fixed-width and stable across runs so a downstream CI
    job can parse it. The marker ``=== libero_backend_matrix ===`` /
    ``=== /libero_backend_matrix ===`` brackets make the table easy to
    locate in a longer log.
    """
    print()
    print("=== libero_backend_matrix ===")
    print(f"Task: {task}  ({n_episodes} episodes, seed={seed})")
    print(f"{'backend':<12} {'success_rate':>13} {'wall_time':>11}  status")
    print("-" * 64)
    for r in rows:
        sr_s = f"{r.success_rate:.2f}" if r.success_rate is not None else "--"
        wt_s = f"{r.wall_time:.1f}s" if r.wall_time is not None else "--"
        print(f"{r.label:<12} {sr_s:>13} {wt_s:>11}  {r.status}")
    print("=== /libero_backend_matrix ===")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--task",
        default="libero-spatial-pick_up_the_red_cube",
        help=(
            "LIBERO benchmark name forwarded to each per-backend driver. "
            "The driver auto-derives the matching `libero_<suite>/` "
            "sub-checkpoint when ``--policy groot`` is set."
        ),
    )
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--policy",
        choices=["mock", "groot"],
        default="mock",
        help=(
            "Policy provider forwarded to each driver. Default: mock "
            "(no GR00T container needed, success_rate is meaningless "
            "but the wall-time column is comparable across backends)."
        ),
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="GR00T inference port forwarded to each driver (only used with --policy=groot).",
    )
    p.add_argument(
        "--backends",
        default=None,
        help=(
            "Comma-separated subset of backend labels to run (e.g. "
            "`mujoco,isaac-1`). Default: all rows. Useful for "
            "running just the rows that have data planes wired."
        ),
    )
    p.add_argument(
        "--per-backend-timeout",
        type=float,
        default=600.0,
        help=(
            "Seconds to wait for each per-backend driver subprocess "
            "before marking it ``skip (timeout)``. Default: 600s "
            "(10 minutes)."
        ),
    )
    args = p.parse_args()

    selected_labels = None
    if args.backends:
        selected_labels = {s.strip() for s in args.backends.split(",") if s.strip()}

    rows: list[RowResult] = []
    matrix_t0 = time.time()
    for label, driver_filename, extra_argv in _BACKEND_ROWS:
        if selected_labels is not None and label not in selected_labels:
            continue

        driver_path = _find_driver(driver_filename)
        if driver_path is None:
            rows.append(RowResult(label, None, None, f"unavailable (no {driver_filename})"))
            continue

        # Fleet rows force the env-count flag if/when their driver
        # files start accepting one. Until R23 lands ``run_isaac_fleet.py``
        # this branch never executes; the row produces an
        # ``unavailable (no run_isaac_fleet.py)`` result instead.
        if label.endswith("-4096"):
            row_argv = [*extra_argv, "--num-envs", "4096"]
        elif label == "isaac-1":
            row_argv = (
                [*extra_argv, "--num-envs", "1"] if "--num-envs" in _driver_args(driver_path) else list(extra_argv)
            )
        else:
            row_argv = list(extra_argv)

        print(f"[matrix] running {label} → {driver_filename} ...", flush=True)
        rows.append(
            _run_one(
                label,
                driver_path,
                row_argv,
                policy=args.policy,
                task=args.task,
                n_episodes=args.n_episodes,
                seed=args.seed,
                port=args.port,
                timeout=args.per_backend_timeout,
            )
        )

    _print_table(rows, task=args.task, n_episodes=args.n_episodes, seed=args.seed)
    print(f"[matrix] total wall_time={time.time() - matrix_t0:.1f}s", flush=True)
    return 0


def _driver_args(driver_path: Path) -> str:
    """Return the driver's source as a string for cheap flag-presence checks.

    Used to gate fleet-only flags (``--num-envs``) on driver files
    that actually accept them. Reading the source is intentionally
    coarser than ``--help``: a few of the drivers shell out to Isaac
    on import which would defeat a ``--help`` probe on non-Isaac
    hosts. Source-text inspection has no such side effects.
    """
    try:
        return driver_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


if __name__ == "__main__":
    sys.exit(main())
