#!/usr/bin/env python3
"""Synthetic-dataset generation via Isaac Sim's Omniverse Replicator.

What this demonstrates
----------------------
Isaac Sim's distinguishing capability vs MuJoCo: the full Omniverse / USD
stack unlocks **NVIDIA Replicator**, a domain-randomization /
synthetic-data-generation framework that walks a scene through randomized
parameters (lighting, materials, camera poses, object placement) on each
frame and exports labelled images (RGB + semantic segmentation + depth +
2D/3D bounding boxes + ...) to disk. The output is a labelled dataset
suitable for training perception / VLA models without real-world
collection cost.

MuJoCo's renderer is rasterization-only and has no equivalent
SDG / labelling pipeline -- you can render frames out of MuJoCo, but
producing a *labelled* dataset with photoreal materials, ground-truth
segmentation masks, and per-frame randomization is an Isaac-specific
capability. This example sits in ``examples/isaac/`` (not
``examples/libero/``) because it isn't a LIBERO benchmark run; it's a
"what is Isaac *for* at a higher level than task evaluation" demo.

How it works
------------
1. Construct an :class:`IsaacSimulation` configured for path-traced RTX
   rendering (``render_mode="rtx_pathtracing"``) -- Replicator's photoreal
   output requires path tracing for material / lighting realism.
2. Build a small LIBERO-style scene: ground plane + Franka Panda + a
   handful of objects (cube, sphere, cylinder).
3. Hand the scene to ``sim.generate_synth_dataset(...)`` -- the
   Isaac-only extension method that wraps Replicator's randomizer +
   writer pipeline (``omni.replicator.core``).
4. Replicator iterates ``num_frames`` times, each frame randomizing
   the requested aspects (``randomize=[...]``) and writing out the
   requested annotation channels (``annotations=[...]``) under
   ``output_dir/`` in a basic-writer layout (``rgb_*.png`` /
   ``semantic_segmentation_*.png`` / ``depth_*.npy`` / ``*.json``
   metadata).

API status (target UX)
----------------------
``sim.generate_synth_dataset(...)`` is an **Isaac-specific extension
method** on :class:`IsaacSimulation`; it is *not* part of the
``SimEngine`` ABC because no other backend can implement it (the
USD / Omniverse / Replicator stack is Isaac-only). The method itself
lands with R7.4
(`#15 <https://github.com/strands-labs/robots-sim/issues/15>`_'s
rendering / Replicator plumbing slice). This example file demonstrates
the **target UX** today so the scope of R7.4 is concrete; on a
checkout where R7.4 hasn't merged yet, the call raises
``AttributeError`` and the script exits with a structured diagnostic
pointing at the tracking issue.

System requirements
-------------------
- **GPU**: NVIDIA RTX-class (RTX 20-series, A-series, L-series, or
  newer). Path tracing in ``rtx_pathtracing`` mode requires RTCore
  (the dedicated ray-tracing silicon); GTX cards / Quadro pre-RTX
  don't accelerate it and fall back to a slow CPU-emulated path that
  isn't usable for SDG at scale.
- **Driver / CUDA**: NVIDIA driver >= 535, CUDA 12.x.
- **Isaac Sim**: 6.0 or newer (Replicator API surface stabilised in
  the 4.x line, modern namespace on 6.0). Install via the Omniverse
  Launcher / Isaac Lab / NGC Docker image; ``isaacsim`` (or the legacy
  ``omni.isaac.kit``) and ``omni.replicator.core`` must be
  importable in the active Python environment.
- **OS**: Ubuntu 22.04+ (Replicator's RTX backend doesn't support macOS
  / Windows-WSL). Headless mode is supported (no X server needed).

This example exits early with a helpful diagnostic on hosts without
Isaac Sim (via :meth:`IsaacSimulation.is_available`), without crashing
on the first ``omni.*`` import.

Usage
-----
::

    pip install 'strands-robots[sim-isaac]'

    # Default: 200 randomized frames into ./synth_data/
    python examples/isaac/isaac_replicator_synthdata.py

    # Smaller smoke run (10 frames; meets the DoD's labelled-frame floor):
    python examples/isaac/isaac_replicator_synthdata.py --num-frames 10

    # Different output directory + only RGB + segmentation (no depth):
    python examples/isaac/isaac_replicator_synthdata.py \\
        --output-dir /data/replicator_runs/run42 \\
        --annotations rgb,semantic_segmentation

    # Pick which aspects to randomize:
    python examples/isaac/isaac_replicator_synthdata.py \\
        --randomize lighting,materials

References
----------
- Replicator core docs:
  https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html
- Tracking issue: `#16 <https://github.com/strands-labs/robots-sim/issues/16>`_
- Backend slice (R7.4): `#15 <https://github.com/strands-labs/robots-sim/issues/15>`_
- Umbrella: `#8 <https://github.com/strands-labs/robots-sim/issues/8>`_
"""

from __future__ import annotations

import argparse
import os
import sys

from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

# Default Replicator output knobs. The DoD requires "at least 10 labelled
# frames"; the script's default of 200 produces a usefully-sized smoke
# dataset on an L4 in ~3-5 minutes, but ``--num-frames 10`` is the
# minimum to satisfy the DoD.
_DEFAULT_NUM_FRAMES = 200
_DEFAULT_OUTPUT_DIR = "synth_data"
_DEFAULT_RANDOMIZE = ("lighting", "materials", "camera_pose")
_DEFAULT_ANNOTATIONS = ("rgb", "depth", "semantic_segmentation")


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Flag defaults match the issue's sketch. ``--randomize`` /
    ``--annotations`` accept comma-separated values so a single flag
    keeps the invocation copy-pastable into a docstring.
    """
    p = argparse.ArgumentParser(
        description="Generate a Replicator-randomized synthetic dataset from a small LIBERO-style scene.",
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=_DEFAULT_NUM_FRAMES,
        help=(
            "Number of randomized frames to generate. "
            f"Default {_DEFAULT_NUM_FRAMES}; pass --num-frames 10 for the "
            "DoD-floor smoke run."
        ),
    )
    p.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=(
            "Directory to write Replicator output into (created if absent). "
            f"Default {_DEFAULT_OUTPUT_DIR!r}; layout is the standard "
            "BasicWriter ``rgb_<frame>.png`` / "
            "``semantic_segmentation_<frame>.png`` / ``distance_to_camera_"
            "<frame>.npy`` / ``<frame>.json`` per-frame fanout."
        ),
    )
    p.add_argument(
        "--randomize",
        default=",".join(_DEFAULT_RANDOMIZE),
        help=(
            "Comma-separated list of aspects to randomize each frame. "
            f"Default: {','.join(_DEFAULT_RANDOMIZE)}. Recognised values: "
            "lighting, materials, camera_pose, object_pose, color."
        ),
    )
    p.add_argument(
        "--annotations",
        default=",".join(_DEFAULT_ANNOTATIONS),
        help=(
            "Comma-separated annotation channels to write per frame. "
            f"Default: {','.join(_DEFAULT_ANNOTATIONS)}. Recognised "
            "values: rgb, depth, semantic_segmentation, instance_"
            "segmentation, bounding_box_2d, bounding_box_3d, normals."
        ),
    )
    p.add_argument(
        "--robot-usd",
        default=None,
        help=(
            "Path / URL to a USD robot asset to load. Default: Isaac Sim's "
            "bundled Franka Panda resolved from the assets root. The asset "
            "sub-path moved under a vendor folder in Isaac Sim 6.0 "
            "(``Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd``) from the "
            "legacy 4.x layout (``Isaac/Robots/Franka/franka.usd``); the "
            "resolver HEAD-probes both and uses whichever exists, reachable "
            "from the Omniverse CDN even without a local Nucleus."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for Replicator's RNG so randomized frames are reproducible. Default 42.",
    )
    return p


def _split_csv(value: str) -> list[str]:
    """Parse a comma-separated CLI value into a clean list of stripped tokens."""
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _asset_exists(url: str) -> "bool | None":
    """Best-effort HEAD-probe for an asset URL.

    Returns ``True`` / ``False`` when the probe is conclusive, or ``None``
    when it can't be determined (non-HTTP URL such as an ``omniverse://``
    Nucleus path, or a network error). ``None`` means "inconclusive --
    don't rule the candidate in or out".

    Ported from :func:`examples.libero.run_isaac._asset_exists` so this
    SDG example resolves the same Isaac Sim 6.0 / legacy-4.x dual layout.
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
# target runtime), then fall back to the legacy 4.x path. Mirrors
# ``examples/libero/run_isaac.py``'s ``_FRANKA_USD_SUBPATHS``. See
# strands-labs/robots-sim#110 and #142.
_FRANKA_USD_SUBPATHS = (
    "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",  # Isaac Sim 6.0+
    "Isaac/Robots/Franka/franka.usd",  # Isaac Sim 4.x and earlier
)


def _resolve_default_robot_usd() -> str | None:
    """Resolve the default Franka Panda USD URL from the Isaac assets root.

    Mirrors :func:`examples.libero.run_isaac._resolve_robot_asset`'s default
    branch. The asset sub-path moved under a vendor folder in Isaac Sim 6.0
    (``Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd``) from the legacy
    4.x layout (``Isaac/Robots/Franka/franka.usd``); the resolver
    HEAD-probes each candidate in :data:`_FRANKA_USD_SUBPATHS` order and
    returns the first that resolves. The hard-coded legacy 4.x path 404s on
    the Isaac Sim 6.0 Omniverse CDN, crashing ``add_robot`` before any
    synthetic data is generated (strands-labs/robots-sim#142).

    The ``get_assets_root_path`` helper is imported lazily because it's only
    resolvable after ``create_world`` has booted ``SimulationApp``. Returns
    ``None`` if the assets root can't be resolved (no internet + no Nucleus);
    callers should surface a structured error in that case.

    Tries the modern ``isaacsim.storage.native`` namespace first (Isaac
    Sim 6.0 supported path) and falls back to the legacy
    ``omni.isaac.nucleus`` shim -- matches the dual-path policy in
    ``strands_robots/simulation/isaac/simulation.py``.

    Probe semantics match ``run_isaac._resolve_default_franka_usd``: if no
    HTTP probe is conclusive (e.g. a Nucleus ``omniverse://`` root that can't
    be HEAD-probed over HTTP), fall back to the first (6.0) candidate so the
    caller still has a path to try.
    """
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
        return None

    candidates = [f"{assets_root}/{sub}" for sub in _FRANKA_USD_SUBPATHS]
    for url in candidates:
        if _asset_exists(url) is True:
            return url
    # No conclusive hit (every HTTP candidate 404'd, or the root is a
    # non-HTTP Nucleus path that can't be probed). Fall back to the 6.0
    # candidate -- the current target runtime -- so add_robot still gets a
    # path to try rather than a silent no-op.
    return candidates[0]


def _build_scene(sim: IsaacSimulation, robot_usd: str | None) -> None:
    """Populate the simulation with a small LIBERO-style scene.

    A flat ground plane (added by ``create_world(ground_plane=True)``) plus
    a Franka Panda plus three object primitives: a red cube, a green
    sphere, and a blue cylinder. Object positions are tuned so they sit
    on the ground with no initial penetration (the procedural Panda's
    base is at the origin; objects clear the robot's footprint).

    The scene is intentionally minimal -- Replicator's value comes from
    the *randomizer*, not from scene complexity. A handful of distinct
    object classes is enough to exercise lighting / material / camera
    randomization end-to-end and produce visibly varied frames.
    """
    # Robot. Pre-#63 / #64 the usd_path branch may no-op (silent
    # registration), but the rest of the pipeline still runs end-to-
    # end and Replicator picks up whatever prims actually landed on
    # the stage.
    if robot_usd is None:
        robot_usd = _resolve_default_robot_usd()
    if robot_usd is not None:
        print(f"[scene] loading robot from USD: {robot_usd}")
        result = sim.add_robot(name="robot", usd_path=robot_usd)
        if result.get("status") != "success":
            raise RuntimeError(f"add_robot failed: {result}")
    else:
        print(
            "[scene] WARNING: could not resolve a default Franka Panda USD "
            "(no internet + no local Nucleus + no --robot-usd). Continuing "
            "with an empty articulation -- Replicator will still randomize "
            "the object primitives."
        )

    # Three small objects spread out so the camera sees all of them.
    # Colors picked for high contrast in a semantic_segmentation visualization.
    for name, shape, position, color in [
        ("cube", "box", [0.3, 0.0, 0.05], [1.0, 0.0, 0.0, 1.0]),
        ("ball", "sphere", [-0.3, 0.2, 0.05], [0.0, 1.0, 0.0, 1.0]),
        ("can", "cylinder", [-0.3, -0.2, 0.05], [0.0, 0.0, 1.0, 1.0]),
    ]:
        result = sim.add_object(name=name, shape=shape, position=position, color=color)
        if result.get("status") != "success":
            raise RuntimeError(f"add_object({name!r}) failed: {result}")

    # Camera at an over-the-shoulder vantage so all three objects + the
    # robot fit in frame. Same vantage as ``examples/libero/run_isaac.py``
    # so frames look familiar to anyone comparing Isaac LIBERO eval and
    # Replicator output side by side.
    result = sim.add_camera(
        name="image",
        position=[2.0, 0.0, 1.5],
        target=[0.0, 0.0, 0.5],
        fov=60.0,
    )
    if result.get("status") != "success":
        raise RuntimeError(f"add_camera failed: {result}")


def _generate_synth_dataset_target_ux(
    sim: IsaacSimulation,
    *,
    num_frames: int,
    output_dir: str,
    randomize: list[str],
    annotations: list[str],
    seed: int,
) -> dict:
    """Invoke ``sim.generate_synth_dataset(...)`` if the R7.4 method exists.

    Pre-R7.4, :class:`IsaacSimulation` doesn't expose this method yet;
    this wrapper short-circuits with a structured diagnostic explaining
    the dependency. Post-R7.4 the call passes through untouched.

    Returning a structured envelope (rather than raising) lets the caller
    record a deterministic exit message and an exit code that
    distinguishes "Replicator not wired up yet" (non-error from this
    example's point of view) from "Replicator wired up but failed"
    (a real bug to investigate).
    """
    fn = getattr(sim, "generate_synth_dataset", None)
    if fn is None:
        return {
            "status": "skipped",
            "reason": (
                "IsaacSimulation.generate_synth_dataset is not available on "
                "this checkout. Replicator wiring is gated on R7.4 -- see "
                "https://github.com/strands-labs/robots-sim/issues/15. "
                "Once R7.4 lands, rerun this script unchanged."
            ),
        }
    return fn(
        num_frames=num_frames,
        output_dir=output_dir,
        randomize=randomize,
        annotations=annotations,
        seed=seed,
    )


def main() -> int:
    args = _build_parser().parse_args()

    # Fail-fast on hosts without Isaac Sim. Cheap (importlib.util.find_spec
    # only) so we run it before any side effects (output dir creation,
    # assets-root resolution, ...).
    available, reason = IsaacSimulation.is_available()
    if not available:
        print(
            "Isaac Sim is not available on this host: "
            f"{reason}\n"
            "This example requires Isaac Sim 4.5+ on an RTX GPU. Install via "
            "the Omniverse Launcher / Isaac Lab / NGC Docker image and ensure "
            "`omni.isaac.kit` is importable in this Python environment. See "
            "https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html "
            "for setup details. Exiting cleanly (this is not a crash).",
            file=sys.stderr,
        )
        # Exit code 0: the DoD requires "doesn't crash on non-GPU systems --
        # exits with a helpful message via sim.is_available() check". A
        # graceful exit is not a failure.
        return 0

    randomize = _split_csv(args.randomize)
    annotations = _split_csv(args.annotations)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[setup] output_dir={output_dir}")
    print(f"[setup] num_frames={args.num_frames}")
    print(f"[setup] randomize={randomize}")
    print(f"[setup] annotations={annotations}")
    print(f"[setup] seed={args.seed}")

    # Path-traced rendering is required for Replicator's photoreal output.
    # rtx_realtime works for prototype runs but doesn't exercise the
    # material / lighting RT pipeline that distinguishes Replicator from
    # a vanilla rasterizer.
    sim = IsaacSimulation(
        IsaacConfig(
            headless=True,
            num_envs=1,
            render_mode="rtx_pathtracing",
            enable_rtx_sensors=True,
        )
    )

    try:
        result = sim.create_world(ground_plane=True)
        if result.get("status") != "success":
            raise RuntimeError(f"create_world failed: {result}")

        _build_scene(sim, args.robot_usd)

        result = _generate_synth_dataset_target_ux(
            sim,
            num_frames=args.num_frames,
            output_dir=output_dir,
            randomize=randomize,
            annotations=annotations,
            seed=args.seed,
        )

        if result.get("status") == "skipped":
            # Pre-R7.4 path: scene built, Replicator method missing. Print
            # the diagnostic + a grep-stable line so post-hoc analysis can
            # tell a "skipped" run from a "succeeded" one.
            print(f"[replicator] SKIPPED: {result['reason']}")
            print("replicator_status=skipped frames_written=0")
            return 0

        if result.get("status") != "success":
            raise RuntimeError(f"generate_synth_dataset failed: {result}")

        # Post-R7.4 success path. The grep-stable line mirrors
        # ``run_<backend>.py``'s output convention so a future matrix /
        # CI scraper can pick it up uniformly.
        frames_written = result.get("frames_written", args.num_frames)
        print(f"[replicator] wrote {frames_written} labelled frames to {output_dir}")
        print(f"replicator_status=success frames_written={frames_written} output_dir={output_dir}")
        return 0
    finally:
        sim.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
