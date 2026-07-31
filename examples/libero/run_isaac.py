#!/usr/bin/env python3
"""LIBERO on the Isaac Sim backend (``strands-robots[sim-isaac]``).

Companion to ``run_mujoco.py``: same CLI shape, same two grep-stable
output lines, same ``evaluate_benchmark(...)`` driver. Differs in
backend choice (``IsaacSimulation`` instead of MuJoCo's
``Simulation``), real-asset robot loading (loads the bundled Franka
Panda USD via ``add_robot(usd_path=...)`` rather than a LIBERO MJCF),
and an explicit ``add_camera(...)`` call (Isaac doesn't auto-attach
viewport cameras the way MuJoCo does -- the camera prim has to land
on the stage before ``render`` / recorder pulls from it).

Robot asset
-----------
By default this loads Isaac Sim's bundled **Franka Panda USD**,
resolved from the assets root. The asset sub-path moved under a vendor
folder in Isaac Sim 6.0
(``Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd``) from the legacy
4.x layout (``Isaac/Robots/Franka/franka.usd``); the resolver HEAD-probes
both and uses whichever exists (reachable over HTTPS from the Omniverse
CDN, no local Nucleus required). Override with ``--robot-usd PATH`` or
``--robot-urdf PATH`` to load your own asset.

This deliberately loads a **real** robot, not the procedural builder
(``add_robot(data_config="panda")``): the procedural Panda is a
kinematically approximate stick-figure (right joint count, wrong link
geometry / masses / joint origins) -- fine for lifecycle smoke tests,
useless for a LIBERO manipulation policy whose end-effector targets
depend on correct kinematics. Loading the real Franka USD constructs
a true ``omni.isaac.core.articulations.Articulation`` whose joints are
observable via ``get_observation`` and actuatable via ``send_action``
(GPU-validated: 9 DoF -- 7 arm + 2 fingers).

Dependency status (as of 2026-06)
---------------------------------
The real-asset robot load rides on the USD- / URDF-loaded
``add_robot`` paths:

* `PR #63 <https://github.com/strands-labs/robots-sim/pull/63>`_ /
  `PR #70 <https://github.com/strands-labs/robots-sim/pull/70>`_ --
  ``add_robot(usd_path=...)`` Articulation construction (the default
  path this script uses). **Merged.**
* `PR #64 <https://github.com/strands-labs/robots-sim/pull/64>`_ /
  `PR #70 <https://github.com/strands-labs/robots-sim/pull/70>`_ --
  ``add_robot(urdf_path=...)`` for the ``--robot-urdf`` override. The
  URDF importer module path differs across releases; the backend
  tries the modern ``isaacsim.asset.importer.urdf`` first and falls
  back to the legacy ``omni.importer.urdf`` for pre-4.5 builds.

The camera / video path rides on:

* `PR #61 (add_camera Phase 2) <https://github.com/strands-labs/robots-sim/pull/61>`_
  + `PR #62 (render frame-path) <https://github.com/strands-labs/robots-sim/pull/62>`_
  -- both merged. ``--policy=mock`` works fully (doesn't read images);
  ``--policy=groot`` will read from the camera handle.

Rollout video (MP4)
-------------------
At parity with ``run_mujoco.py``: this script wraps the eval in an
``IsaacSimulation.start_cameras_recording`` /
``stop_cameras_recording`` pair, producing a real
``rollouts/<date>/{rec_name}__image.mp4`` (one frame per applied
control step, captured on the eval thread via ``evaluate_benchmark``'s
``on_frame=`` hook). The ``videos=`` line below points at the file
that gets written, matching MuJoCo's ``{name}__{camera}.mp4`` filename
convention so the R15 backend-matrix glob picks up Isaac rows. See
strands-labs/robots-sim#112.

Tracks `#15 <https://github.com/strands-labs/robots-sim/issues/15>`_
(R8 — example file). Filename is ``run_isaac.py`` rather than the
issue's pre-rescope ``libero_isaac.py`` to match the post-rescope
``examples/libero/run_<backend>.py`` layout that ``run_mujoco.py``
established (see ``examples/libero_example.py:126``).

Usage
-----
::

    # 1) Smoke test, GPU + Isaac Sim required (see is_available()
    #    error path for setup hints). Loads the default Franka USD:
    python examples/libero/run_isaac.py --policy mock --n-episodes 5

    # 1b) Bring your own robot asset:
    python examples/libero/run_isaac.py --policy mock --robot-usd /path/to/robot.usd
    python examples/libero/run_isaac.py --policy mock --robot-urdf /path/to/robot.urdf

    # 2) Real LIBERO eval against `nvidia/GR00T-N1.7-LIBERO`. Same
    #    GR00T-inference-container lifecycle as `run_mujoco.py`.
    python examples/libero/run_isaac.py --policy groot --port 8000 --n-episodes 50

Requires
--------
``pip install 'strands-robots[sim-isaac,benchmark-libero]'``
plus a working Isaac Sim 6.0+ install on the host (RTX GPU, Ubuntu
22.04+, CUDA 12+). ``strands-robots`` requires Python 3.12 (the
interpreter bundled by Isaac Sim 6.0). On a non-Isaac host the script exits early with a
diagnostic from :meth:`IsaacSimulation.is_available` rather than
crashing on the first ``omni.*`` / ``isaacsim.*`` import.

Verification status (as of 2026-06)
-----------------------------------
Target runtime is the Isaac Sim 6.0 NGC docker image
(``nvcr.io/nvidia/isaac-sim:6.0``, Python 3.12, RTX/L4 GPU, headless),
matching the ``isaacsim>=6.0`` / ``requires-python>=3.12`` migration.
The end-to-end lifecycle was previously validated on the Isaac Sim 4.5
image (``nvcr.io/nvidia/isaac-sim:4.5.0``); the dual-path
``isaacsim.*`` / ``omni.isaac.*`` imports in
``strands_robots/simulation/isaac/simulation.py`` keep the same code path
working on both. The
script runs past ``IsaacSimulation.is_available`` → ``create_world``
→ ``add_robot`` (real Franka USD over the Omniverse CDN) →
``add_camera`` → physics ``step``.
On a non-Isaac host (no GPU, no Omniverse) ``is_available()`` still
short-circuits cleanly with the install-hint reason string, so this
file remains safe to import / lint on CPU-only CI runners.

LIBERO benchmark on Isaac (scene loading implemented)
-----------------------------------------------------
``evaluate_benchmark`` runs on the Isaac backend:
``LiberoAdapter.on_episode_start`` calls ``sim.load_scene(...)`` to
realize each task's scene, and ``IsaacSimulation.load_scene`` now
translates the LIBERO/BDDL-compiled MJCF into USD prims on the Isaac
stage (the substantive LIBERO-on-Isaac work -- implemented in
`#129 <https://github.com/strands-labs/robots-sim/issues/129>`_, which
superseded the fail-fast stub PR #117 shipped for the closed #116).
``--policy=mock`` / ``--policy=groot`` runs of this driver complete
end-to-end and the #112 recorder writes a real rollout MP4 (at MuJoCo
parity). The ``__main__`` guard below still forces ``os._exit(1)`` on a
genuine failure so Isaac's SimulationApp fast-shutdown can't swallow a
non-zero exit into 0. A *meaningful* (non-zero) ``success_rate`` also
needs the articulation-control fix (#123); scene loading is the
prerequisite this driver depended on. ``examples/libero/run_mujoco.py``
remains the CPU-friendly reference path, and the Isaac backend can also
be driven directly via the manual ``create_world`` -> ``add_robot`` ->
``add_object`` -> ``add_camera`` -> ``step`` -> ``render`` quickstart in
``docs/index.md``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import time

from strands_robots.simulation import create_simulation
from strands_robots.simulation.isaac import IsaacSimulation


def _date_dir(date_root: str = "rollouts") -> str:
    """Return a date-stamped subdirectory of ``date_root``, creating it.

    Mirrors :func:`run_mujoco._date_dir` so the post-eval video
    aggregation script (R15 backend matrix) finds artifacts under the
    same convention regardless of backend.
    """
    out = os.path.join(date_root, _dt.date.today().strftime("%Y_%m_%d"))
    os.makedirs(out, exist_ok=True)
    return out


def _suite_for_task(task: str) -> str:
    """Auto-derive a LIBERO suite name from a benchmark task ID.

    Identical to :func:`run_mujoco._suite_for_task` -- LIBERO task IDs
    follow the same ``libero-<suite>-<task_stem>`` convention regardless
    of which backend is going to evaluate them.

    >>> _suite_for_task("libero-spatial-pick_up_the_red_cube")
    'libero_spatial'
    >>> _suite_for_task("libero-10-LIVING_ROOM_SCENE5_...")
    'libero_10'
    """
    parts = task.split("-", 2)
    if len(parts) < 3 or parts[0] != "libero":
        raise ValueError(
            f"--task must look like 'libero-<suite>-<task_stem>', got {task!r}. "
            "See `load_libero_suite` for registered names."
        )
    return f"libero_{parts[1]}"


def _default_checkpoint_dir() -> str:
    """Default ``--checkpoint-dir`` that clears ``gr00t_inference``'s mount guard.

    ``gr00t_inference`` (strands-robots >= 0.4.0) downloads to
    ``~/.strands_robots/checkpoints/`` by default, but its ``start_container``
    step refuses to bind-mount any path under ``/home`` (a "protected host
    path" guard), so the OOTB ``--policy groot`` lifecycle aborts
    (strands-labs/robots-sim#125). Default to a non-``/home`` cache: honor an
    explicit ``STRANDS_ROBOTS_CHECKPOINT_DIR`` override, then fall back to
    ``$XDG_CACHE_HOME`` only when it lives outside ``/home``, else
    ``/tmp/strands_robots/checkpoints``.
    """
    override = os.environ.get("STRANDS_ROBOTS_CHECKPOINT_DIR")
    if override:
        return override
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg and not os.path.realpath(xdg).startswith("/home"):
        return os.path.join(xdg, "strands_robots", "checkpoints")
    return "/tmp/strands_robots/checkpoints"


def _explain_lifecycle_failure(result: dict, checkpoint_dir: str, container: str) -> str:
    """Turn a ``gr00t_inference`` lifecycle failure into an actionable message.

    Surfaces the two most common OOTB blockers with a concrete next step:
    the ``/home`` "protected host path" mount guard
    (strands-labs/robots-sim#125), and a stale ``gr00t-libero-*`` container
    that ``start_container`` won't recreate without ``force=True``.
    """
    blob = repr(result)
    hint = ""
    if "protected host path" in blob:
        hint = (
            "\n\nHINT: the checkpoint dir is under a path `gr00t_inference` refuses to "
            "bind-mount (the `/home` mount guard). Pass `--checkpoint-dir` (or set "
            f"$STRANDS_ROBOTS_CHECKPOINT_DIR) to a non-`/home` path; current value: {checkpoint_dir!r}."
        )
    elif "already in use" in blob or "Conflict" in blob or "is already" in blob:
        hint = (
            f"\n\nHINT: a stale container named {container!r} is blocking `start_container` "
            f"(it won't recreate an existing one without force=True). Remove it with "
            f"`docker rm -f {container}` and retry."
        )
    return f"gr00t_inference lifecycle=full failed: {result}{hint}"


def _configure_gr00t_image(image: str) -> None:
    """Point ``gr00t_inference`` at *image* via operator env config.

    In ``strands-robots>=0.4.0`` the GR00T docker image is no longer a
    ``gr00t_inference`` kwarg — it's operator-configured through the
    ``STRANDS_GR00T_IMAGE`` env var and validated against
    ``STRANDS_GR00T_IMAGE_ALLOW`` (defaults: ``gr00t:*`` and
    ``nvcr.io/nvidia/isaac-gr00t:*``). This sets the env var to the
    requested ``--image`` and, when the image doesn't already match the
    allowlist, appends it so resolution doesn't fail closed.
    """
    os.environ["STRANDS_GR00T_IMAGE"] = image
    allow = os.environ.get("STRANDS_GR00T_IMAGE_ALLOW", "")
    patterns = [p.strip() for p in allow.split(",") if p.strip()]
    default_allow = ("gr00t:", "nvcr.io/nvidia/isaac-gr00t:")
    already_allowed = (
        image in patterns
        or any(image.startswith(prefix) for prefix in default_allow)
        or any(p.endswith("*") and image.startswith(p[:-1]) for p in patterns)
    )
    if not already_allowed:
        patterns.append(image)
        os.environ["STRANDS_GR00T_IMAGE_ALLOW"] = ",".join(patterns)


def _build_parser() -> argparse.ArgumentParser:
    """Mirror :func:`run_mujoco.main`'s parser surface.

    Argument names / defaults / choices are kept identical to the
    MuJoCo file so a matrix-driver shell wrapper that supplies the
    same flags works against both backends (the only difference is
    which ``run_<backend>.py`` is invoked).
    """
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["mock", "groot"], default="mock")
    p.add_argument("--port", type=int, default=8000, help="GR00T inference port (only used with --policy=groot)")
    p.add_argument(
        "--task",
        default="libero-spatial-pick_up_the_red_cube",
        help="Any registered LIBERO benchmark name; suite is auto-derived.",
    )
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--robot-usd",
        default=None,
        help="Path / URL to a USD robot asset to load via add_robot(usd_path=...). "
        "Default: Isaac Sim's bundled Franka Panda resolved from the assets root "
        "(Isaac Sim 6.0: `Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd`, "
        "legacy 4.x: `Isaac/Robots/Franka/franka.usd` -- whichever exists). Mutually "
        "exclusive with --robot-urdf.",
    )
    p.add_argument(
        "--robot-urdf",
        default=None,
        help="Path to a URDF robot asset to load via add_robot(urdf_path=...). "
        "Mutually exclusive with --robot-usd. Converted to USD on import via "
        "the Isaac URDF importer.",
    )
    p.add_argument(
        "--eef-body-name",
        default=_ISAAC_FRANKA_EEF_BODY_NAME,
        help="Robot link whose pose feeds the LIBERO state.x/y/z/roll/pitch/yaw "
        "keys via IsaacSimulation.get_body_state (the MuJoCo defaults "
        "gripper0_grip_site / robot0_right_hand don't exist on the Isaac "
        "Franka USD). Default 'panda_hand' resolves on the bundled Franka; "
        "override when --robot-usd / --robot-urdf loads an asset with "
        "different link names.",
    )
    p.add_argument(
        "--auto-server",
        dest="auto_server",
        action="store_true",
        default=True,
        help="(--policy=groot only) Bring up the GR00T inference service before the eval.",
    )
    p.add_argument(
        "--no-auto-server",
        dest="auto_server",
        action="store_false",
        help="(--policy=groot only) Don't manage the inference service.",
    )
    p.add_argument(
        "--image",
        default="gr00t:latest",
        help="(--auto-server only) Docker image tag of the GR00T container. "
        "In strands-robots>=0.4.0 the image is operator-configured via the "
        "STRANDS_GR00T_IMAGE env var (validated against STRANDS_GR00T_IMAGE_ALLOW); "
        "this flag sets that env var (and extends the allowlist if needed) before "
        "calling `gr00t_inference`.",
    )
    p.add_argument(
        "--container",
        default="gr00t-libero-isaac",
        help="(--auto-server only) Docker container name to (re)use. Defaults to "
        "the Isaac-specific name so Isaac and MuJoCo eval runs don't clobber "
        "each other's containers when run side-by-side on the same host.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help="(--auto-server only) Where to cache the HF checkpoint. "
        "Default: a non-`/home` path (`$STRANDS_ROBOTS_CHECKPOINT_DIR`, an "
        "outside-`/home` `$XDG_CACHE_HOME/strands_robots/checkpoints`, or "
        "`/tmp/strands_robots/checkpoints`). This avoids `gr00t_inference`'s "
        "`start_container` mount guard, which refuses to bind-mount any path "
        "under `/home` (see strands-labs/robots-sim#125).",
    )
    return p


def _resolve_hf_token() -> str:
    """Resolve a HuggingFace token for the gated GR00T checkpoint download.

    Prefers the ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) environment
    variable -- CI / container environments typically inject the token that
    way and don't have the ``~/.cache/huggingface/token`` file that
    ``huggingface-cli login`` writes. Falls back to that file for interactive
    dev boxes. Raises if neither is present.
    """
    from pathlib import Path

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()
    hf_token_path = Path("~/.cache/huggingface/token").expanduser()
    if hf_token_path.is_file():
        return hf_token_path.read_text().strip()
    raise RuntimeError(
        "--policy groot needs an HF token (Cosmos-Reason2-2B is gated). "
        "Set the HF_TOKEN env var (preferred for CI), or run "
        "`huggingface-cli login` to write ~/.cache/huggingface/token, then retry."
    )


def _orchestrate_groot_server(args: argparse.Namespace, suite: str) -> dict | None:
    """Bring up the GR00T inference container if ``--policy=groot``.

    Same lifecycle shape as :func:`run_mujoco.main`'s setup block:
    `gr00t_inference(action='lifecycle', lifecycle='full', ...)` →
    poll GPU memory until the model loads → return a handle the
    teardown path picks up. Pulled into its own function so the Isaac
    main is shorter and the MuJoCo-vs-Isaac comparison stays focused
    on the simulation-side differences.

    Returns ``None`` if the server doesn't need to be brought up
    (i.e. ``--policy=mock`` or ``--no-auto-server``).
    """
    if args.policy != "groot" or not args.auto_server:
        return None

    from strands_robots.tools import gr00t_inference

    _configure_gr00t_image(args.image)

    # Default the checkpoint cache to a non-`/home` path so the downloaded
    # checkpoint clears `gr00t_inference`'s `start_container` mount guard
    # (strands-labs/robots-sim#125). Explicit `--checkpoint-dir` wins.
    if args.checkpoint_dir is None:
        args.checkpoint_dir = _default_checkpoint_dir()
    print(f"[setup] checkpoint dir: {args.checkpoint_dir}")

    hf_token = _resolve_hf_token()
    result = gr00t_inference(
        action="lifecycle",
        lifecycle="full",
        hf_repo="nvidia/GR00T-N1.7-LIBERO",
        hf_subfolder=suite,
        hf_local_dir=args.checkpoint_dir,
        container_name=args.container,
        hf_token=hf_token,
        checkpoint_path=f"/data/checkpoints/{suite}",
        embodiment_tag="libero_sim",
        protocol="n1.7",
        use_sim_policy_wrapper=True,
        port=args.port,
    )
    if result.get("status") != "success":
        raise RuntimeError(_explain_lifecycle_failure(result, args.checkpoint_dir, args.container))
    print(f"[setup] {result.get('message')}")

    # Wait for model load. Same heuristic as run_mujoco.py: GR00T-N1.7
    # loads to ~6.3 GB on the L4; gate at 4 GiB so the check fires once the
    # model is resident (the old 10 GiB gate never tripped — the model
    # footprint is below it — so --auto-server always timed out at 180 s).
    import subprocess
    from time import monotonic, sleep

    deadline = monotonic() + 180
    loaded_threshold_mib = 4_000
    while monotonic() < deadline:
        try:
            used = int(
                subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
                .decode()
                .strip()
                .splitlines()[0]
            )
        except Exception:
            used = 0
        if used > loaded_threshold_mib:
            print(f"[setup] GR00T model loaded (gpu_mem={used} MiB)")
            break
        sleep(5)
    else:
        raise RuntimeError(
            "GR00T model didn't reach load threshold within 180 s. Check `docker logs <container>` for stderr."
        )

    return result


def _resolve_task(suite: str, requested_task: str, adapter_kwargs: dict | None = None) -> str:
    """Register the LIBERO suite and resolve ``requested_task``.

    Identical fallback-to-first-registered semantics as the MuJoCo
    file: the spec's default placeholder
    ``libero-spatial-pick_up_the_red_cube`` isn't an actual LIBERO
    task name; if the user passes the default and it doesn't resolve,
    fall back to the first registered task with a clear note.
    Explicitly-supplied unknown tasks still error loudly.

    ``adapter_kwargs`` configures every adapter's backend-specific state
    sources (#1802) -- see :func:`_isaac_adapter_kwargs`.
    """
    from strands_robots.benchmarks.libero import load_libero_suite

    registered = load_libero_suite(suite, adapter_kwargs=adapter_kwargs)
    if not registered:
        raise RuntimeError(
            f"load_libero_suite({suite!r}) registered 0 tasks. "
            "Apply upstream fix from strands-labs/robots#147 if it isn't merged."
        )
    if requested_task in registered:
        return requested_task
    if requested_task == "libero-spatial-pick_up_the_red_cube":
        fallback = next(iter(registered))
        print(
            f"NOTE: default --task {requested_task!r} isn't in real LIBERO "
            f"(it's the spec's aspirational placeholder); falling back "
            f"to first registered task {fallback!r}."
        )
        return fallback
    raise RuntimeError(f"--task {requested_task!r} is not in the {suite} suite. Available: {sorted(registered)[:3]}…")


def _asset_exists(url: str) -> bool | None:
    """Best-effort HEAD-probe for an asset URL.

    Returns ``True`` / ``False`` when the probe is conclusive, or ``None``
    when it can't be determined (non-HTTP URL such as an ``omniverse://``
    Nucleus path, or a network error). ``None`` means "inconclusive --
    don't rule the candidate in or out".
    """
    if not url.lower().startswith(("http://", "https://")):
        return None
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            return 200 <= resp.status < 400
    except urllib.error.HTTPError as exc:
        return exc.code < 400
    except Exception:  # noqa: BLE001
        return None


# Default Franka Panda USD sub-paths relative to the Isaac assets root.
# NVIDIA relocated the asset under a vendor folder in Isaac Sim 6.0, so the
# layout differs across releases. Probe the 6.0 path first (the current
# target runtime), then fall back to the legacy 4.x path. See
# strands-labs/robots-sim#110.
_FRANKA_USD_SUBPATHS = (
    "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",  # Isaac Sim 6.0+
    "Isaac/Robots/Franka/franka.usd",  # Isaac Sim 4.x and earlier
)

# --- EEF state-source calibration for the bundled Isaac Franka USD (#1802) --
#
# The libero_panda GR00T data-config requires state.x/y/z/roll/pitch/yaw/
# gripper, which LiberoAdapter injects from RoboSuite's split source:
# position from the gripper0_grip_site (gripper tip), orientation from the
# robot0_right_hand body (wrist). Neither name exists on the Isaac Franka
# USD, so the adapter reads the wrist body via IsaacSimulation.get_body_state
# and applies a site-equivalent correction, configured here.
#
# Measured on this repo's Isaac 6.0 + MuJoCo backends (2026-07-31), both
# robots at the episode-0 canonical init config of the first libero_spatial
# task, poses compared base-relative (panda_link0 vs robot0_link0):
#
#   * Isaac's `panda_hand` frame COINCIDES with RoboSuite's
#     `robot0_right_hand` frame: relative rotation 0.03 deg, position
#     delta < 0.6 mm (both within position-controller settle error).
#     No quaternion correction is therefore applied (`eef_quat_offset`
#     omitted = identity); the residual is quantified above rather than
#     hand-waved (#168's success rates are sensitive to exactly this).
#   * The grip-site offset in the hand frame measured [0, 0, 0.097] m on
#     MuJoCo (exact to 1e-15 -- it is the authored site transform) and
#     [3.8e-5, -6.5e-6, 0.0964] via the cross-sim derivation; we use the
#     authored value, valid on Isaac because the frames coincide.
#
# Gripper: the Isaac Franka USD drives BOTH fingers positive (URDF mimic
# convention, panda_finger_joint2 in [0, 0.04]) while RoboSuite trains
# state.gripper with opposite signs ([+q, -q]); the [1, -1] signs restore
# the trained convention.
_ISAAC_FRANKA_EEF_BODY_NAME = "panda_hand"
_ISAAC_FRANKA_EEF_POS_OFFSET = [0.0, 0.0, 0.097]
_ISAAC_FRANKA_GRIPPER_JOINT = "panda_finger_joint1"
_ISAAC_FRANKA_STATE_GRIPPER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
_ISAAC_FRANKA_STATE_GRIPPER_SIGNS = [1.0, -1.0]


def _isaac_adapter_kwargs(args: argparse.Namespace) -> dict:
    """LiberoAdapter kwargs that make EEF state injection work on Isaac (#1802).

    Forwarded through ``load_libero_suite(adapter_kwargs=...)`` to every
    registered task. ``eef_state_site_name=""`` is the adapter's documented
    "no site available" sentinel -- Isaac has no MuJoCo sites, so the
    body-fallback (``get_body_state`` + the offsets above) is the only
    source and we say so explicitly rather than letting the site lookup
    fail per step.
    """
    return {
        "eef_body_name": args.eef_body_name,
        "eef_state_site_name": "",
        "eef_pos_offset": list(_ISAAC_FRANKA_EEF_POS_OFFSET),
        "gripper_joint_name": _ISAAC_FRANKA_GRIPPER_JOINT,
        "state_gripper_joint_names": list(_ISAAC_FRANKA_STATE_GRIPPER_JOINTS),
        "state_gripper_signs": list(_ISAAC_FRANKA_STATE_GRIPPER_SIGNS),
    }


def _resolve_default_franka_usd(assets_root: str) -> str:
    """Pick the Franka USD candidate that exists under ``assets_root``.

    HEAD-probes each candidate in :data:`_FRANKA_USD_SUBPATHS` order and
    returns the first that resolves. If no probe is conclusive (e.g. a
    Nucleus ``omniverse://`` root that can't be HEAD-probed over HTTP),
    falls back to the first (6.0) candidate. Raises with an actionable
    hint only when every HTTP candidate definitively 404s.
    """
    candidates = [f"{assets_root}/{sub}" for sub in _FRANKA_USD_SUBPATHS]
    saw_definitive_miss = False
    for url in candidates:
        exists = _asset_exists(url)
        if exists is True:
            return url
        if exists is False:
            saw_definitive_miss = True
    if saw_definitive_miss:
        raise RuntimeError(
            "Default Franka USD not found under the Isaac assets root "
            f"({assets_root}); tried {candidates}. The asset layout changed "
            "between Isaac Sim 4.x and 6.0 -- pass --robot-usd / --robot-urdf "
            "with an explicit asset path."
        )
    return candidates[0]


def _resolve_robot_asset(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Resolve which robot asset to load → ``(usd_path, urdf_path)``.

    Precedence:

    1. ``--robot-urdf`` if given → ``(None, urdf)``.
    2. ``--robot-usd`` if given → ``(usd, None)``.
    3. Default → the bundled Franka Panda USD resolved from Isaac Sim's
       assets root. The asset sub-path moved under a vendor folder in
       Isaac Sim 6.0 (``Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd``)
       from the legacy 4.x layout (``Isaac/Robots/Franka/franka.usd``), so
       :func:`_resolve_default_franka_usd` HEAD-probes both and returns
       whichever exists. Reachable over HTTPS from the public Omniverse CDN
       even without a local Nucleus server, so the example runs
       out-of-the-box on any Isaac Sim install with internet.

    Why a *real* asset (not the procedural builder): the procedural
    ``add_robot(data_config="panda")`` path produces a kinematically
    approximate stick-figure (correct joint count, wrong link geometry /
    masses / joint origins) -- fine for lifecycle smoke tests, useless
    for a LIBERO manipulation policy. Loading the real Franka USD gives
    the correct kinematics a GR00T / LIBERO policy expects.

    ``get_assets_root_path`` is imported lazily (only resolvable after
    ``create_world`` has booted ``SimulationApp``), so this is called
    *after* ``sim.create_world()`` in :func:`main`. Tries the modern
    ``isaacsim.storage.native`` namespace first (Isaac Sim 4.5+ supported
    path) and falls back to the legacy ``omni.isaac.nucleus`` shim --
    matches the dual-path policy in ``strands_robots/simulation/isaac/simulation.py``.
    """
    if args.robot_urdf is not None:
        return None, args.robot_urdf
    if args.robot_usd is not None:
        return args.robot_usd, None
    try:
        from isaacsim.storage.native import (  # type: ignore[import-not-found]
            get_assets_root_path,
        )
    except ImportError:
        from omni.isaac.nucleus import (  # type: ignore[import-not-found]
            get_assets_root_path,
        )

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError(
            "Could not resolve the Isaac Sim assets root for the default Franka USD. "
            "Pass --robot-usd / --robot-urdf with an explicit asset path, or configure "
            "a Nucleus server / internet access for the Omniverse CDN."
        )
    return _resolve_default_franka_usd(assets_root), None


def main() -> None:
    args = _build_parser().parse_args()
    if args.robot_usd is not None and args.robot_urdf is not None:
        raise SystemExit("--robot-usd and --robot-urdf are mutually exclusive; pass at most one.")
    suite = _suite_for_task(args.task)

    # Fail-fast on hosts without Isaac Sim. The is_available probe is
    # cheap (it only does importlib.util.find_spec on omni.isaac.kit /
    # isaacsim; zero omni.* / isaacsim.* modules land in sys.modules)
    # so we run it before touching the GR00T container or any benchmark
    # side effects -- a misconfigured host should exit with a structured
    # error before a docker pull starts.
    available, reason = IsaacSimulation.is_available()
    if not available:
        raise RuntimeError(
            f"Isaac Sim is not available on this host: {reason}. "
            "Install Isaac Sim 6.0+ via the Omniverse Launcher / Isaac Lab / NGC "
            "Docker image and ensure `isaacsim` (6.0+, Python 3.12) or the legacy "
            "`omni.isaac.kit` is importable in this Python environment."
        )

    # Bring up GR00T container (idempotent; no-op for --policy=mock).
    server_handle = _orchestrate_groot_server(args, suite)

    if args.policy == "groot":
        # Same client-side data_config + groot_version as run_mujoco.py
        # — see that file for the rationale on why both strings are
        # required (data_config tells the local client how to format
        # observations; groot_version forces 5D video serialization
        # for the N1.7 server).
        policy_kwargs = {
            "policy_provider": "groot",
            "policy_config": {
                "host": "localhost",
                "port": args.port,
                "data_config": "libero_panda",
                "groot_version": "n1.7",
            },
        }
    else:
        policy_kwargs = {"policy_provider": "mock"}

    # Construct the Isaac sim via the strands-robots factory. Now that
    # strands-robots>=0.4.1 walks the `strands_robots.backends`
    # entry-point group, `create_simulation("isaac", ...)` resolves to
    # this package's IsaacSimulation — same UX as create_simulation("mujoco").
    # headless=True avoids opening a Kit viewport (the GR00T eval doesn't
    # need an interactive GUI); it does NOT control the render pipeline.
    # render_mode="rtx_realtime" makes render() take the RTX frame path
    # instead of short-circuiting to zero-filled (blank) frames in the
    # default render_mode="headless", which is what produced all-black
    # rollout MP4s. (Allowed modes: "headless", "rtx_realtime",
    # "rtx_pathtracing" -- see config.py.
    # STRANDS_ISAAC_RTX_PATHTRACING=1 upgrades to photoreal pathtracing.)
    # The kwargs are forwarded to the IsaacConfig dataclass (a pure-Python
    # construct with no omni.* imports), so this call is cheap and runs on
    # a non-Isaac host — the actual SimulationApp boot happens inside
    # create_world().
    sim = create_simulation("isaac", headless=True, num_envs=1, render_mode="rtx_realtime")
    try:
        result = sim.create_world()
        if result.get("status") != "success":
            raise RuntimeError(f"create_world failed: {result}")

        # Load a *real* robot asset (default: Isaac's bundled Franka
        # Panda USD; override via --robot-usd / --robot-urdf). This
        # routes through add_robot's usd_path / urdf_path branch, which
        # constructs a real ``omni.isaac.core.articulations.Articulation``
        # (joints observable via get_observation, actuatable via
        # send_action) -- see PR #63 (USD) / PR #64 (URDF).
        #
        # Name "robot" is deliberately NOT a procedural alias (so the
        # usd_path / urdf_path branch is taken, not procedural lookup)
        # and matches the LIBERO/RoboSuite convention for the Franka.
        robot_usd, robot_urdf = _resolve_robot_asset(args)
        if robot_urdf is not None:
            print(f"[setup] loading robot from URDF: {robot_urdf}")
            result = sim.add_robot(name="robot", urdf_path=robot_urdf)
        else:
            print(f"[setup] loading robot from USD: {robot_usd}")
            result = sim.add_robot(name="robot", usd_path=robot_usd)
        if result.get("status") != "success":
            raise RuntimeError(f"add_robot failed: {result}")

        # Phase-2 camera: Isaac doesn't auto-attach viewport cameras
        # the way MuJoCo's mjData does. Add an explicit RTX camera at
        # the same over-the-shoulder vantage that the LIBERO ``image``
        # camera uses on MuJoCo (`agentview` ≈ [2, 0, 1.5] looking at
        # origin).
        #
        # With #61 (add_camera Phase 2) + #62 (render frame-path) merged,
        # this constructs an actual ``isaacsim.sensors.camera.Camera`` and
        # ``render(camera_name="image")`` returns real RGB frames keyed off
        # it. Those frames feed both ``--policy=groot`` (which reads images)
        # and the rollout-video recorder wired below.
        recording_camera = "image"
        result = sim.add_camera(
            name=recording_camera,
            position=[2.0, 0.0, 1.5],
            target=[0.0, 0.0, 0.5],
            fov=60.0,
        )
        if result.get("status") != "success":
            raise RuntimeError(f"add_camera failed: {result}")

        # The libero_panda data-config ALSO requires a ``wrist_image`` video
        # key. Pre-add it here -- at the same static fallback vantage
        # ``LiberoAdapter.LIBERO_CAMERAS`` declares -- rather than letting the
        # adapter install it inside ``on_episode_start`` (#1802): an RTX
        # camera added AFTER the LIBERO scene prims are on the stage never
        # accumulates a frame on Isaac 6.0 (its Replicator render product
        # returns a 0-D buffer through 100+ step/update ticks; verified on
        # this host), whereas the same camera added before scene-load warms
        # in one step. The adapter's install step skips cameras that already
        # exist, so pre-adding is the supported path.
        wrist_camera = "wrist_image"
        result = sim.add_camera(
            name=wrist_camera,
            position=[0.0, 0.0, 1.4],
            target=[0.0, 0.0, 0.85],
            fov=60.0,
            width=256,
            height=256,
        )
        if result.get("status") != "success":
            raise RuntimeError(f"add_camera failed: {result}")

        # Warm up the RTX render product before recording. A camera added
        # after the last world step has an empty render product: its first
        # `get_rgba()` returns a malformed `(0,)` buffer and `render()`
        # surfaces a structured "RTX render product likely hasn't accumulated
        # a frame yet" error (simulation.py:2135-2141). If we start recording
        # immediately, every `on_frame` call hits that error, the recorder
        # buffer stays empty for the whole rollout, and `stop_cameras_recording`
        # writes a 0-frame mp4 that imageio silently drops -- the file never
        # lands on disk despite a "success" status line.
        #
        # Step + render the world a few times until `render(camera_name="image")`
        # returns a real (non-error) frame, mirroring isaac_gs/render_demo.py's
        # `sim.step(5)`-before-render pattern. `sim.step` under
        # render_mode="rtx_realtime" calls `world.step(render=True)`, so the RTX
        # render product accumulates samples each iteration. Bounded so a host
        # that never populates fails loudly instead of recording blank frames.
        _RTX_WARMUP_MAX_ITERS = 10
        for cam_name in (recording_camera, wrist_camera):
            warmed_up = False
            for _ in range(_RTX_WARMUP_MAX_ITERS):
                sim.step(1)
                probe = sim.render(camera_name=cam_name)
                if probe.get("status") == "success":
                    warmed_up = True
                    break
            if not warmed_up:
                raise RuntimeError(
                    f"RTX render product for camera {cam_name!r} never "
                    f"accumulated a frame after {_RTX_WARMUP_MAX_ITERS} warm-up "
                    "step+render iterations; recording would produce a 0-frame "
                    f"(dropped) mp4. Last render probe: {probe}"
                )
            print(f"[setup] RTX camera {cam_name!r} warmed up; render product populated")

        # Keep the CLI-requested task distinct from the resolved one. The
        # default placeholder transparently falls back to the first registered
        # task, but the grep-stable line below must echo what the caller passed
        # (``requested_task``) so the R15 matrix driver can replay a run from
        # its recorded output -- re-running with the *resolved* value would
        # change behaviour (the fallback wouldn't fire). ``resolved_task`` is
        # what actually executes and what the filename / benchmark call use.
        requested_task = args.task
        resolved_task = _resolve_task(suite, args.task, adapter_kwargs=_isaac_adapter_kwargs(args))
        args.task = resolved_task

        # Filename convention matches run_mujoco.py so the matrix
        # driver's video discovery glob (`rollouts/*/*--task=*.mp4`)
        # picks up Isaac and MuJoCo runs uniformly.
        ts = _dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        rec_name = (
            f"{ts}--task={resolved_task}--n_eps={args.n_episodes}"
            f"--seed={args.seed}--policy={args.policy}--backend=isaac"
        )
        video_dir = _date_dir()

        # Arm the synchronous Isaac recorder. Unlike MuJoCo's daemon-thread
        # recorder, IsaacSimulation captures frames on the eval thread via
        # the `on_frame` closure threaded into `evaluate_benchmark` -- the
        # RTX renderer + Camera.get_rgba are bound to the thread that booted
        # SimulationApp, so a background recorder would deadlock. On stop,
        # the buffers flush to `{rec_name}__{camera}.mp4` under the same
        # `rollouts/<date>/` layout MuJoCo uses (see
        # strands_robots/simulation/isaac/simulation.py:start_cameras_recording).
        rec = sim.start_cameras_recording(
            cameras=[recording_camera],
            output_dir=video_dir,
            name=rec_name,
        )
        if rec.get("status") != "success":
            raise RuntimeError(f"start_cameras_recording failed: {rec}")
        base_on_frame = next(c["json"]["on_frame"] for c in rec["content"] if "json" in c)

        # The library `on_frame` closure is best-effort per the sim API: a
        # transient bad render (e.g. a `(0,)`-shaped RTX buffer when the
        # render product hasn't accumulated a sample for *this* step yet)
        # is silently skipped -- it bumps an internal error counter and
        # drops the frame rather than raising. Across a whole rollout a run
        # of such skips can leave the buffer empty, and `stop_cameras_recording`
        # then writes a 0-frame mp4 that imageio drops -- the exact failure
        # this PR set out to kill. The warm-up loop above guards the *first*
        # frame, but mid-rollout RTX hiccups are not covered by it.
        #
        # Wrap the library closure so a skipped frame is retried (step +
        # re-render) up to N times before giving up on that step. We detect
        # a skip by watching the per-camera buffer length: if it didn't grow
        # after calling the library closure, the frame was dropped, so we
        # step+render once more and retry. A small, bounded retry budget keeps
        # a steady-state RTX stall from silently producing a blank video while
        # not turning a single transient hiccup into a hard failure. The
        # fail-loud backstop is the post-rollout zero-frame assertion below.
        _ON_FRAME_MAX_RETRIES = 3
        retry_stats = {"skipped": 0, "retried": 0, "recovered": 0}

        def _buffer_len() -> int:
            st = getattr(sim, "_cams_rec_state", None)
            if not st:
                return 0
            return len(st.get("buffers", {}).get(recording_camera, []))

        def on_frame(step: int, observation: dict, action: dict) -> None:
            before = _buffer_len()
            base_on_frame(step, observation, action)
            if _buffer_len() > before:
                return
            # Frame was skipped (bad/missing render). Retry step+render.
            retry_stats["skipped"] += 1
            for _ in range(_ON_FRAME_MAX_RETRIES):
                retry_stats["retried"] += 1
                sim.step(1)
                base_on_frame(step, observation, action)
                if _buffer_len() > before:
                    retry_stats["recovered"] += 1
                    return

        video_path = os.path.join(video_dir, f"{rec_name}__{recording_camera}.mp4")

        t0 = time.time()
        try:
            result = sim.evaluate_benchmark(
                benchmark_name=resolved_task,
                # robot_name omitted for the same reason run_mujoco.py
                # omits it: the benchmark's on_episode_start may rename /
                # reload the robot; ``evaluate_benchmark`` auto-picks the
                # single robot when there's only one.
                n_episodes=args.n_episodes,
                seed=args.seed,
                on_frame=on_frame,
                # GR00T-N1.7-LIBERO was trained at 20 Hz control (#168).
                # On MuJoCo the OSC controller owns its 25-substep loop, so
                # the runner's rate never mattered there; on Isaac the
                # delta-EEF controller (#1812) relies on the runner to step
                # a full 1/20 s per action (physics_dt=1/120 -> 6 substeps)
                # so the PD drives actually track each joint target.
                control_frequency=20.0,
                **policy_kwargs,
            )
        finally:
            stop = sim.stop_cameras_recording()
            print(f"[recording] {stop['content'][0]['text']}")
        wall_time = time.time() - t0

        # Fail loud if the rollout captured zero frames. `stop_cameras_recording`
        # always returns status="success" (it's best-effort and idempotent),
        # and imageio silently drops a 0-frame mp4 -- so without this check a
        # blank rollout would ship a "success" line pointing at a file that
        # never landed on disk. Inspect the per-camera artifact frame count
        # and raise rather than emit a misleading grep-stable success line.
        stop_json = next((c["json"] for c in stop.get("content", []) if "json" in c), None)
        frames_written = 0
        frame_errors = 0
        if stop_json is not None:
            for art in stop_json.get("artifacts", []):
                if art.get("camera") == recording_camera:
                    frames_written = int(art.get("frames", 0))
                    frame_errors = int(art.get("errors", 0))
                    break
        if retry_stats["skipped"]:
            print(
                f"[recording] on_frame retries: {retry_stats['skipped']} skipped, "
                f"{retry_stats['retried']} step+render retries, "
                f"{retry_stats['recovered']} recovered "
                f"(max {_ON_FRAME_MAX_RETRIES}/frame)"
            )
        if frames_written == 0:
            raise RuntimeError(
                f"rollout recorded 0 frames for camera {recording_camera!r} "
                f"({frame_errors} per-frame render errors, "
                f"{retry_stats['skipped']} skipped frames not recovered by "
                f"{_ON_FRAME_MAX_RETRIES} step+render retries each). "
                "imageio drops a 0-frame mp4, so no video would land at "
                f"{video_path!r}; failing loud instead of shipping a blank rollout."
            )

        if result.get("status") != "success":
            raise RuntimeError(f"evaluate_benchmark failed: {result}")

        json_payload = next(c["json"] for c in result["content"] if "json" in c)
        success_rate = json_payload["success_rate"]

        # Two grep-stable lines for the R15 matrix script -- exact
        # same format as run_mujoco.py's so subprocess-and-parse
        # consumers don't have to special-case the Isaac backend.
        # ``task=`` echoes the CLI-REQUESTED task so the run is replayable
        # from this line; ``resolved_task=`` records what actually ran when
        # the aspirational-placeholder fallback rewrote it (identical to
        # ``task=`` in the common case). ``backend=isaac`` is the discriminator.
        print(f"benchmark_name={requested_task}")
        print(
            f"policy={args.policy}  task={requested_task}  "
            f"resolved_task={resolved_task}  "
            f"success_rate={success_rate:.2f}  "
            f"wall_time={wall_time:.1f}s  videos={video_path}  backend=isaac"
        )
    finally:
        sim.destroy()
        if server_handle is not None:
            from strands_robots.tools import gr00t_inference

            gr00t_inference(action="lifecycle", lifecycle="teardown", container_name=args.container)


if __name__ == "__main__":
    # Force a non-zero exit on failure even when Isaac Sim's SimulationApp
    # fast-shutdown has registered an atexit/_exit hook that would
    # otherwise swallow the interpreter's normal non-zero status into a
    # misleading exit 0. ``os._exit(1)`` bypasses atexit handlers
    # (including SimulationApp's), so a failed eval is visible to the exit
    # status / CI (scene loading itself is now implemented, #129).
    import sys
    import traceback

    try:
        main()
    except SystemExit:
        raise
    except (KeyboardInterrupt, Exception):  # noqa: BLE001 - top-level: log + force non-zero exit
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
