#!/usr/bin/env python3
"""LIBERO evaluation with backend subcommands: ``run.py mujoco`` / ``run.py isaac``.

One-shot programmatic flow: a scripted ``sim.evaluate_benchmark(...)``
call - the "if you just want to run LIBERO from a Python script" file.
For the natural-language / ``Agent``-driven versions, see
``run_mujoco_agent.py`` / ``run_isaac_agent.py`` (sibling files).

Both subcommands share one driver: parser base, GR00T container
orchestration, task resolution, the eval-and-report loop, and the two
grep-stable output lines. Only the simulation setup differs - a thin
:func:`_setup_mujoco` shim (procedural Panda + LIBERO scene pre-warm)
vs. a :func:`_setup_isaac` shim (real Franka USD resolved over the
Omniverse CDN, explicit RTX cameras + warm-up, synchronous ``on_frame``
recorder). Backend-specific flags exist only on their own subcommand:
``--robot-usd`` / ``--robot-urdf`` / ``--eef-body-name`` are
Isaac-only. The chosen backend is imported only after parsing, so the
other backend's packages never load.

Consumed by the backend-matrix harness (``libero_backend_matrix.py``),
which subprocess-runs ``run.py <backend>`` and parses two grep-stable
lines (``benchmark_name=...`` and ``policy=... task=...
resolved_task=... success_rate=... wall_time=...s videos=...
backend=...``). ``task=`` echoes the CLI-requested task so a run is
replayable from its recorded output; ``resolved_task=`` records what
actually ran when the aspirational-placeholder default fell back
(identical to ``task=`` in the common case). The format is pinned by
``tests/test_examples_libero_drivers.py``.

Usage
-----
::

    # 1) Smoke test on MuJoCo, no GPU required:
    python examples/libero/run.py mujoco --policy mock --n-episodes 5

    # 2) Real LIBERO eval against `nvidia/GR00T-N1.7-LIBERO`. By default
    #    the script auto-orchestrates the GR00T inference service via the
    #    `gr00t_inference(action="lifecycle", lifecycle="full", ...)`
    #    tool - it builds the n1.7 container if missing, downloads the
    #    right `libero_<suite>/` sub-checkpoint, runs the container, and
    #    starts the inference server before the eval, then tears down on
    #    exit. Each step is idempotent so re-runs are cheap.
    #
    #    Pre-condition: an HF token (the HF_TOKEN env var, or the Hub's
    #    cached login from `hf auth login`) with access to
    #    `nvidia/Cosmos-Reason2-2B` (the gated VLM backbone) +
    #    Docker + an NVIDIA GPU.
    python examples/libero/run.py mujoco --policy groot --port 8000 --n-episodes 50

    # 2b) If you'd rather manage the inference service yourself
    #     (multi-eval session, custom container config, etc.), pass
    #     --no-auto-server and run the `gr00t_inference` lifecycle tool
    #     ahead of time.
    python examples/libero/run.py mujoco --policy groot --no-auto-server --port 8000

    # 3) Different LIBERO suite + task. Suite is auto-derived from --task,
    #    so the lifecycle tool downloads the matching `libero_<suite>/`
    #    sub-checkpoint:
    python examples/libero/run.py mujoco \\
        --policy groot --port 8000 \\
        --task libero-10-LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_…

    # 4) Same eval on Isaac Sim (GPU + Isaac Sim 6.0+ host required).
    #    Loads the bundled Franka Panda USD from the Omniverse CDN:
    python examples/libero/run.py isaac --policy mock --n-episodes 5

    # 4b) Bring your own robot asset (Isaac only):
    python examples/libero/run.py isaac --policy mock --robot-usd /path/to/robot.usd
    python examples/libero/run.py isaac --policy mock --robot-urdf /path/to/robot.urdf

    # 5) Real LIBERO eval on Isaac:
    python examples/libero/run.py isaac --policy groot --port 8000 --n-episodes 50

Requires
--------
``pip install 'strands-robots[sim-mujoco,benchmark-libero]'`` for the
``mujoco`` subcommand; ``pip install
'strands-robots[sim-isaac,benchmark-libero]'`` plus a working Isaac Sim
6.0+ host install (RTX GPU, Ubuntu 22.04+, CUDA 12+, Python 3.12) for
``isaac``. On a non-Isaac host the ``isaac`` subcommand exits early
with a diagnostic from ``IsaacSimulation.is_available`` rather than
crashing on the first ``omni.*`` / ``isaacsim.*`` import.

Notes on the MP4 output
-----------------------
``evaluate_benchmark`` does not expose a per-episode ``record_video``
plumb, so the whole run is wrapped in a single
``start_cameras_recording`` / ``stop_cameras_recording`` pair - each
invocation produces one MP4 per recorded camera capturing every episode
in sequence, under ``rollouts/<date>/``. Filename encodes the resolved
task, ``--policy``, ``--n-episodes``, ``--seed``, and the backend so
post-hoc analysis can tell what produced it. MuJoCo records on a
background thread; Isaac captures synchronously on the eval thread via
``evaluate_benchmark``'s ``on_frame=`` hook (the RTX renderer +
``Camera.get_rgba`` are bound to the thread that booted
``SimulationApp``, so a background recorder would deadlock there).

Isaac robot asset
-----------------
The ``isaac`` subcommand deliberately loads a **real** Franka Panda USD
(resolved from the Isaac assets root; the sub-path moved under a vendor
folder in Isaac Sim 6.0, so both layouts are HEAD-probed) rather than
the procedural builder (``add_robot(data_config="panda")``): the
procedural Panda is a kinematically approximate stick-figure (right
joint count, wrong link geometry / masses / joint origins) - fine for
lifecycle smoke tests, useless for a LIBERO manipulation policy whose
end-effector targets depend on correct kinematics. Override with
``--robot-usd PATH`` or ``--robot-urdf PATH``.

Verification status (`--policy=groot` end-to-end, MuJoCo)
---------------------------------------------------------
Measured 2026-07-22 on an L4 / Docker dev box against
``nvidia/GR00T-N1.7-LIBERO/libero_10``,
``libero-10-LIVING_ROOM_SCENE5_put_the_white_mug_…``, 5 episodes:

* ``--policy=mock``: ~14 s/ep (success_rate=0.0; mock can't satisfy goals).
* ``--policy=groot --seed=42``: ~33 s/ep, **success_rate=0.80 (4/5)** in 165.0 s.

Wall-time covers engine + scene + policy + I/O round-trip end-to-end.

Optional server-side determinism wrapper
-----------------------------------------
For users who need bit-exact run-to-run reproducibility (e.g. CI
pinning a specific success_rate), pass ``--deterministic``: the
auto-server lifecycle then mounts the packaged determinism wrapper
(``strands_robots/policies/groot/server_wrapper.py``) into the
container and runs the server through it (``cudnn.deterministic=True``
+ ``cudnn.benchmark=False`` + ``CUBLAS_WORKSPACE_CONFIG=":4096:8"`` +
a per-episode reseed of the server RNG from the client-forwarded
seed). The eval works WITHOUT it (verified at 4/5 above) - it's only
needed when you want the GPU's CUDA backend to produce identical
actions across re-runs of the same seed.

Escape hatch for hand-managed containers (``--no-auto-server``)::

    WRAP=$(python -c "import strands_robots.policies.groot.server_wrapper as m; print(m.__file__)")
    docker run … -v "$WRAP":/srv_wrap.py:ro \\
        gr00t:latest python /srv_wrap.py --model-path … --use-sim-policy-wrapper --port 8000
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Shared helpers (backend-agnostic)
# ---------------------------------------------------------------------------


def _date_dir(date_root: str = "rollouts") -> str:
    """Return a date-stamped subdirectory of ``date_root``, creating it.

    Both backends write video artifacts under the same convention so the
    backend-matrix video-discovery glob picks them up uniformly.
    """
    out = os.path.join(date_root, _dt.date.today().strftime("%Y_%m_%d"))
    os.makedirs(out, exist_ok=True)
    return out


def _suite_for_task(task: str) -> str:
    """Auto-derive a LIBERO suite name from a benchmark task ID.

    Task IDs follow the ``libero-<suite>-<task_stem>`` pattern (see
    ``strands_robots.benchmarks.libero.suite``), so the suite is the
    second hyphen-separated segment - regardless of which backend is
    going to evaluate them.

    >>> _suite_for_task("libero-spatial-pick_up_the_red_cube")
    'libero_spatial'
    >>> _suite_for_task("libero-10-LIVING_ROOM_SCENE5_x")
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

    ``gr00t_inference`` downloads to ``~/.strands_robots/checkpoints/`` by
    default, but its ``start_container`` step refuses to bind-mount any
    path under ``/home`` (a "protected host path" guard). Those two
    defaults are mutually inconsistent, so the OOTB ``--policy groot``
    lifecycle would abort at ``start_container``.

    We default to a non-``/home`` cache so the mount guard is satisfied
    without the user having to pass ``--checkpoint-dir`` manually. Honor an
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

    * the ``/home`` "protected host path" mount guard (see
      :func:`_default_checkpoint_dir`),
    * a stale ``gr00t-libero-*`` container that ``start_container`` won't
      recreate without ``force=True``.
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

    The GR00T docker image is not a ``gr00t_inference`` kwarg - it's
    operator-configured through the ``STRANDS_GR00T_IMAGE`` env var and
    validated against ``STRANDS_GR00T_IMAGE_ALLOW`` (defaults:
    ``gr00t:*`` and ``nvcr.io/nvidia/isaac-gr00t:*``). This sets the env
    var to the requested ``--image`` and, when the image doesn't already
    match the allowlist, appends it so resolution doesn't fail closed.
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


def _resolve_hf_token() -> str:
    """Resolve a HuggingFace token for the gated GR00T checkpoint download.

    Prefers the ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) environment
    variable - CI / container environments typically inject the token
    that way and have no cached login. Otherwise takes the cached login,
    asking ``huggingface_hub`` where that lives rather than assuming: it
    resolves ``HF_TOKEN_PATH``, else ``<HF_HOME>/token``, else
    ``<XDG_CACHE_HOME>/huggingface/token``, else ``~/.cache/huggingface/token``.
    Reading the last of those directly names a file the Hub will not open on a
    host that relocated its cache, so a logged-in box was refused here.

    Raises:
        RuntimeError: Neither source yields a token. The message names the
            path the Hub resolves on this host, so a relocated cache is
            actionable, and ``hf auth login`` - the login entry point the
            declared ``huggingface_hub>=1.5`` floor ships. ``huggingface-cli``
            is not published as a console script at that floor, and the later
            1.x releases that do install it exit "deprecated and no longer
            works", so it is a dead end across the whole declared range.
    """
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()
    token_path = "the Hub's cached login"
    try:
        from huggingface_hub import get_token
        from huggingface_hub.constants import HF_TOKEN_PATH
    except ImportError:
        # No Hub installed to ask where its cached login lives; the
        # checkpoint download reports that missing dependency itself.
        pass
    else:
        token_path = HF_TOKEN_PATH
        cached = get_token()
        if cached and cached.strip():
            return cached.strip()
    raise RuntimeError(
        "--policy groot needs an HF token (Cosmos-Reason2-2B is gated). "
        "Set the HF_TOKEN env var (preferred for CI), or run `hf auth login` "
        f"to write {token_path}, then retry."
    )


def _orchestrate_groot_server(args: argparse.Namespace, suite: str) -> dict | None:
    """Bring up the GR00T inference container if ``--policy=groot``.

    Backend-agnostic: `gr00t_inference(action='lifecycle',
    lifecycle='full', ...)` builds the n1.7 container if missing,
    downloads the right ``libero_<suite>/`` sub-checkpoint, runs the
    container, and starts the inference server; then this polls GPU
    memory until the model loads and returns a handle the teardown path
    picks up. Each sub-step is idempotent so re-runs are cheap.

    Returns ``None`` if the server doesn't need to be brought up
    (i.e. ``--policy=mock`` or ``--no-auto-server``).
    """
    if args.policy != "groot" or not args.auto_server:
        return None

    # Import the function from its defining module (not the lazy
    # `strands_robots.tools` re-export) so static analysis resolves
    # the callable instead of the same-named submodule.
    from strands_robots.tools.gr00t_inference import gr00t_inference

    _configure_gr00t_image(args.image)

    # Default the checkpoint cache to a non-`/home` path so the
    # downloaded checkpoint clears `gr00t_inference`'s `start_container`
    # mount guard. When the user passes `--checkpoint-dir` explicitly we
    # honor it verbatim.
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
        # The lifecycle tool mounts `hf_local_dir` (or its default cache
        # dir when `None`) -> `/data/checkpoints`, and the HF download
        # places `<suite>/...` directly under that. So the in-container
        # path is `/data/checkpoints/<suite>`, NOT
        # `/data/checkpoints/GR00T-N1.7-LIBERO/<suite>`.
        checkpoint_path=f"/data/checkpoints/{suite}",
        embodiment_tag="libero_sim",
        protocol="n1.7",
        use_sim_policy_wrapper=True,
        deterministic=args.deterministic,
        port=args.port,
    )
    if result.get("status") != "success":
        raise RuntimeError(_explain_lifecycle_failure(result, args.checkpoint_dir, args.container))
    print(f"[setup] {result.get('message')}")

    # The lifecycle tool returns success when the server's port is
    # bound, but the model itself loads asynchronously after that -
    # a too-eager `evaluate_benchmark` call can race the load and
    # hang on the first inference request. Wait until GPU memory
    # crosses a heuristic load-complete threshold before continuing;
    # remove this loop once `gr00t_inference` blocks until ready.
    import subprocess
    from time import monotonic, sleep

    deadline = monotonic() + 180
    # N1.7 loads to ~6.3 GB on the L4; gate at 4 GiB so the readiness
    # check fires once the model is resident (a 10 GiB gate never
    # tripped - the model footprint is below it - so --auto-server
    # always timed out at 180 s).
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

    The default placeholder ``libero-spatial-pick_up_the_red_cube``
    isn't an actual LIBERO task name; if the user passes the default
    and it doesn't resolve, fall back to the first registered task with
    a clear note. Explicitly-supplied unknown tasks still error loudly.

    ``adapter_kwargs`` configures every adapter's backend-specific state
    sources (#1802) - see :func:`_isaac_adapter_kwargs`.
    """
    from strands_robots.benchmarks.libero import load_libero_suite

    registered = load_libero_suite(suite, adapter_kwargs=adapter_kwargs)
    if not registered:
        raise RuntimeError(
            f"load_libero_suite({suite!r}) registered 0 tasks. Check that the [benchmark-libero] extra is installed."
        )
    if requested_task in registered:
        return requested_task
    if requested_task == "libero-spatial-pick_up_the_red_cube":
        fallback = next(iter(registered))
        print(
            f"NOTE: default --task {requested_task!r} isn't in real LIBERO "
            f"(it's an aspirational placeholder); falling back "
            f"to first registered task {fallback!r}."
        )
        return fallback
    raise RuntimeError(f"--task {requested_task!r} is not in the {suite} suite. Available: {sorted(registered)[:3]}…")


def _policy_kwargs(args: argparse.Namespace) -> dict:
    """``evaluate_benchmark`` policy kwargs for the chosen ``--policy``."""
    if args.policy != "groot":
        return {"policy_provider": "mock"}
    # Client-side `data_config="libero_panda"` - this is the registered
    # key in the GR00T data-config registry that tells the local
    # `Gr00tPolicy` how to format LIBERO observations into the
    # GR00T-N1.7 input layout. Note this is *separate from* the server's
    # `--embodiment-tag libero_sim` (an alias of `LIBERO_PANDA` per the
    # checkpoint's `embodiment_id.json`); the two sides happen to mean
    # the same thing but the strings are not interchangeable.
    # `groot_version="n1.7"` is required when the client doesn't have
    # the upstream `gr00t` package installed (auto-detection only works
    # when it does); without it the client serializes 4D video and the
    # N1.7 server rejects with "must be (B, T, H, W, C), got (B, H, W, C)".
    return {
        "policy_provider": "groot",
        "policy_config": {
            "host": "localhost",
            "port": args.port,
            "data_config": "libero_panda",
            "groot_version": "n1.7",
        },
    }


def _format_result_lines(
    *,
    policy: str,
    requested_task: str,
    resolved_task: str,
    success_rate: float,
    wall_time: float,
    video_path: str,
    backend: str,
) -> tuple[str, str]:
    """The two grep-stable result lines - the de-facto output contract.

    ``libero_backend_matrix.py`` subprocess-runs this driver and parses
    these lines with its ``_RE_RESULT`` regex; keep the exact key set and
    format specs (``success_rate={:.2f}``, ``wall_time={:.1f}s``) stable
    across rebases / refactors. ``task=`` echoes the CLI-requested task
    so a run is replayable from its recorded output; ``resolved_task=``
    records what actually ran. Pinned by
    ``tests/test_examples_libero_drivers.py``.
    """
    return (
        f"benchmark_name={requested_task}",
        (
            f"policy={policy}  task={requested_task}  "
            f"resolved_task={resolved_task}  "
            f"success_rate={success_rate:.2f}  "
            f"wall_time={wall_time:.1f}s  videos={video_path}  backend={backend}"
        ),
    )


# ---------------------------------------------------------------------------
# Per-backend eval plans
# ---------------------------------------------------------------------------


@dataclass
class _EvalPlan:
    """What the shared eval-and-report loop needs from a backend setup shim.

    ``arm_recording`` starts the run-long camera recording and returns
    any extra ``evaluate_benchmark`` kwargs the recording needs (Isaac's
    synchronous recorder returns an ``on_frame=`` closure; MuJoCo's
    background-thread recorder returns nothing). ``check_recording``
    validates the ``stop_cameras_recording`` payload after the eval
    (Isaac's zero-frame fail-loud check); ``None`` skips the check.
    """

    requested_task: str
    resolved_task: str
    recording_camera: str  # camera named in the videos= path
    arm_recording: Callable[[str, str], dict]  # (video_dir, rec_name) -> extra eval kwargs
    check_recording: Callable[[dict], None] | None = None
    eval_kwargs: dict = field(default_factory=dict)


def _setup_mujoco(args: argparse.Namespace, suite: str):
    """MuJoCo setup shim: procedural Panda + LIBERO scene pre-warm.

    Returns ``(sim, plan)``. The heavy backend import happens here -
    after parsing - so the ``isaac`` subcommand never pays for it.
    """
    from strands_robots.simulation import Simulation

    sim = Simulation(tool_name="libero_sim", mesh=False)
    try:
        sim.create_world()
        # Pre-add a Panda named ``robot`` so:
        #   1. evaluate_benchmark's pre-flight check (`No robots in sim`)
        #      passes BEFORE on_episode_start runs scene loading.
        #   2. The resolved-name `evaluate_benchmark` picks up here
        #      survives the rename that LIBERO scene MJCFs do - the
        #      scenes ship a Franka Panda named `robot` (LIBERO/RoboSuite
        #      convention), so picking the same name client-side keeps
        #      the resolved robot stable across `on_episode_start`.
        sim.add_robot("robot", data_config="panda")

        resolved_task = _resolve_task(suite, args.task)

        # Pick the cameras to record from. The LIBERO scene auto-loaded by
        # `LiberoAdapter` supplies cameras named `image` (third-person
        # agentview) and `wrist_image` (gripper view). Without LIBERO
        # loaded - e.g. on `--policy=mock` paths that hit the scene-gen
        # ImportError fallback - only the world's `default` camera exists.
        recording_camera = "image" if args.policy == "groot" else "default"
        recording_cameras = ["image", "wrist_image"] if args.policy == "groot" else ["default"]

        # Pre-warm the scene so `image` actually exists at recording-start
        # time. `start_cameras_recording` looks up the camera by name in
        # the live model and resolving fails if the scene hasn't been
        # loaded yet - but `on_episode_start` (where scene-load happens)
        # only runs *inside* `evaluate_benchmark`. We force the
        # auto-generation + load here so the camera is registered before
        # the recorder starts; subsequent per-episode reloads in the eval
        # loop reuse the cached scene_path so the camera name stays
        # stable across them.
        if args.policy == "groot":
            from strands_robots.simulation.benchmark import get_benchmark

            spec = get_benchmark(resolved_task)
            if hasattr(spec, "ensure_scene"):
                spec.ensure_scene()
            if spec.scene_path:
                sim.load_scene(spec.scene_path)
                # Prewarm BEFORE the redundant-Panda check below, so
                # prewarm's robot registration wraps the scene-supplied
                # Panda first -> list_robots() returns ['robot'] -> the
                # if-check below is False -> no redundant add_robot
                # recompile that would change model.nq away from the
                # LIBERO width init_states[0] is sized for.
                if hasattr(spec, "prewarm"):
                    spec.prewarm(sim)
                # Defensive fallback for non-LIBERO benchmarks that
                # don't expose `prewarm` and don't ship a Panda in
                # the loaded scene MJCF.
                if "robot" not in sim.list_robots():
                    sim.add_robot("robot", data_config="panda")

        def arm_recording(video_dir: str, rec_name: str) -> dict:
            # MuJoCo's recorder runs on a background thread; nothing to
            # thread into evaluate_benchmark.
            sim.start_cameras_recording(cameras=recording_cameras, output_dir=video_dir, name=rec_name)
            return {}

        plan = _EvalPlan(
            requested_task=args.task,
            resolved_task=resolved_task,
            recording_camera=recording_camera,
            arm_recording=arm_recording,
        )
        return sim, plan
    except BaseException:
        # Cleanup-and-reraise: the Simulation owns a thread pool + MuJoCo
        # world; a failed setup must not leak them.
        sim.destroy()
        raise


# Default Franka Panda USD sub-paths relative to the Isaac assets root.
# NVIDIA relocated the asset under a vendor folder in Isaac Sim 6.0, so the
# layout differs across releases. Probe the 6.0 path first (the current
# target runtime), then fall back to the legacy 4.x path.
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
       assets root (see :data:`_FRANKA_USD_SUBPATHS` +
       :func:`_resolve_default_franka_usd`). Reachable over HTTPS from
       the public Omniverse CDN even without a local Nucleus server, so
       the example runs out-of-the-box on any Isaac Sim install with
       internet.

    ``get_assets_root_path`` is imported lazily (only resolvable after
    ``create_world`` has booted ``SimulationApp``), so this is called
    *after* ``sim.create_world()`` in :func:`_setup_isaac`. Tries the
    modern ``isaacsim.storage.native`` namespace first (Isaac Sim 4.5+
    supported path) and falls back to the legacy ``omni.isaac.nucleus``
    shim -- matches the dual-path policy in
    ``strands_robots/simulation/isaac/simulation.py``.
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


def _setup_isaac(args: argparse.Namespace, suite: str):
    """Isaac setup shim: real Franka USD + explicit RTX cameras + warm-up.

    Returns ``(sim, plan)``. The heavy backend import happens here -
    after parsing - and the actual ``SimulationApp`` boot is deferred by
    the library into ``create_world()``.
    """
    from strands_robots.simulation import create_simulation

    # headless=True avoids opening a Kit viewport (the GR00T eval doesn't
    # need an interactive GUI); it does NOT control the render pipeline.
    # render_mode="rtx_realtime" makes render() take the RTX frame path
    # instead of short-circuiting to zero-filled (blank) frames in the
    # default render_mode="headless", which is what produced all-black
    # rollout MP4s. (Allowed modes: "headless", "rtx_realtime",
    # "rtx_pathtracing" -- see config.py.
    # STRANDS_ISAAC_RTX_PATHTRACING=1 upgrades to photoreal pathtracing.)
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
        # send_action).
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

        # Isaac doesn't auto-attach viewport cameras the way MuJoCo's
        # mjData does. Add an explicit RTX camera at the same
        # over-the-shoulder vantage that the LIBERO ``image`` camera uses
        # on MuJoCo (`agentview` ≈ [2, 0, 1.5] looking at origin); frames
        # from it feed both ``--policy=groot`` (which reads images) and
        # the rollout-video recorder wired below.
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
        # a frame yet" error. If we start recording immediately, every
        # `on_frame` call hits that error, the recorder buffer stays empty
        # for the whole rollout, and `stop_cameras_recording` writes a
        # 0-frame mp4 that imageio silently drops -- the file never lands on
        # disk despite a "success" status line.
        #
        # Step + render the world a few times until `render(camera_name=...)`
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

        resolved_task = _resolve_task(suite, args.task, adapter_kwargs=_isaac_adapter_kwargs(args))

        # The library `on_frame` closure is best-effort per the sim API: a
        # transient bad render (e.g. a `(0,)`-shaped RTX buffer when the
        # render product hasn't accumulated a sample for *this* step yet)
        # is silently skipped -- it bumps an internal error counter and
        # drops the frame rather than raising. Across a whole rollout a run
        # of such skips can leave the buffer empty, and `stop_cameras_recording`
        # then writes a 0-frame mp4 that imageio drops. The warm-up loop
        # above guards the *first* frame, but mid-rollout RTX hiccups are
        # not covered by it.
        #
        # Wrap the library closure so a skipped frame is retried (step +
        # re-render) up to N times before giving up on that step. We detect
        # a skip by watching the per-camera buffer length: if it didn't grow
        # after calling the library closure, the frame was dropped, so we
        # step+render once more and retry. A small, bounded retry budget keeps
        # a steady-state RTX stall from silently producing a blank video while
        # not turning a single transient hiccup into a hard failure. The
        # fail-loud backstop is the post-rollout zero-frame check below.
        _ON_FRAME_MAX_RETRIES = 3
        retry_stats = {"skipped": 0, "retried": 0, "recovered": 0}

        def _buffer_len() -> int:
            st = getattr(sim, "_cams_rec_state", None)
            if not st:
                return 0
            return len(st.get("buffers", {}).get(recording_camera, []))

        def arm_recording(video_dir: str, rec_name: str) -> dict:
            # Arm the synchronous Isaac recorder. Unlike MuJoCo's
            # daemon-thread recorder, IsaacSimulation captures frames on the
            # eval thread via the `on_frame` closure threaded into
            # `evaluate_benchmark` -- the RTX renderer + Camera.get_rgba are
            # bound to the thread that booted SimulationApp, so a background
            # recorder would deadlock.
            rec = sim.start_cameras_recording(
                cameras=[recording_camera],
                output_dir=video_dir,
                name=rec_name,
            )
            if rec.get("status") != "success":
                raise RuntimeError(f"start_cameras_recording failed: {rec}")
            base_on_frame = next(c["json"]["on_frame"] for c in rec["content"] if "json" in c)

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

            return {"on_frame": on_frame}

        def check_recording(stop: dict) -> None:
            # Fail loud if the rollout captured zero frames.
            # `stop_cameras_recording` always returns status="success" (it's
            # best-effort and idempotent), and imageio silently drops a
            # 0-frame mp4 -- so without this check a blank rollout would ship
            # a "success" line pointing at a file that never landed on disk.
            print(f"[recording] {stop['content'][0]['text']}")
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
                    "imageio drops a 0-frame mp4, so no video would land on "
                    "disk; failing loud instead of shipping a blank rollout."
                )

        plan = _EvalPlan(
            requested_task=args.task,
            resolved_task=resolved_task,
            recording_camera=recording_camera,
            arm_recording=arm_recording,
            check_recording=check_recording,
            # GR00T-N1.7-LIBERO was trained at 20 Hz control (#168).
            # On MuJoCo the OSC controller owns its 25-substep loop, so
            # the runner's rate never mattered there; on Isaac the
            # delta-EEF controller (#1812) relies on the runner to step
            # a full 1/20 s per action (physics_dt=1/120 -> 6 substeps)
            # so the PD drives actually track each joint target.
            eval_kwargs={"control_frequency": 20.0},
        )
        return sim, plan
    except BaseException:
        # Cleanup-and-reraise: a failed setup must not leak the booted
        # SimulationApp / stage.
        sim.destroy()
        raise


# ---------------------------------------------------------------------------
# Shared eval-and-report loop
# ---------------------------------------------------------------------------


def _evaluate_and_report(sim, args: argparse.Namespace, plan: _EvalPlan) -> None:
    """Record → ``evaluate_benchmark`` → validate → print the result lines."""
    # Filename convention is shared across backends so the matrix
    # driver's video discovery glob (`rollouts/*/*--task=*.mp4`) picks
    # up every backend's runs uniformly.
    ts = _dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    rec_name = (
        f"{ts}--task={plan.resolved_task}--n_eps={args.n_episodes}"
        f"--seed={args.seed}--policy={args.policy}--backend={args.backend}"
    )
    video_dir = _date_dir()
    video_path = os.path.join(video_dir, f"{rec_name}__{plan.recording_camera}.mp4")

    extra_eval_kwargs = plan.arm_recording(video_dir, rec_name)
    t0 = time.time()
    try:
        result = sim.evaluate_benchmark(
            benchmark_name=plan.resolved_task,
            # robot_name omitted on purpose - the benchmark's
            # on_episode_start may rename / reload the robot (the LIBERO
            # scene MJCFs ship a Panda named `robot`), so any name
            # pre-resolved here is gone after scene load.
            # `evaluate_benchmark` auto-picks when there's only one
            # robot, which is the LIBERO case.
            n_episodes=args.n_episodes,
            seed=args.seed,
            **plan.eval_kwargs,
            **extra_eval_kwargs,
            **_policy_kwargs(args),
        )
        wall_time = time.time() - t0
    finally:
        stop = sim.stop_cameras_recording()

    if plan.check_recording is not None:
        plan.check_recording(stop)

    if result.get("status") != "success":
        raise RuntimeError(f"evaluate_benchmark failed: {result}")

    json_payload = next(c["json"] for c in result["content"] if "json" in c)
    success_rate = json_payload["success_rate"]

    for line in _format_result_lines(
        policy=args.policy,
        requested_task=plan.requested_task,
        resolved_task=plan.resolved_task,
        success_rate=success_rate,
        wall_time=wall_time,
        video_path=video_path,
        backend=args.backend,
    ):
        print(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Backend subcommands over one shared flag base.

    Shared flags live on a parent parser so both subcommands accept the
    identical set; Isaac-only flags (``--robot-usd`` / ``--robot-urdf`` /
    ``--eef-body-name``) exist only on the ``isaac`` subcommand, so the
    ``mujoco`` subcommand rejects them at parse time instead of silently
    dropping them.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--policy", choices=["mock", "groot"], default="mock")
    common.add_argument("--port", type=int, default=8000, help="GR00T inference port (only used with --policy=groot)")
    common.add_argument(
        "--task",
        default="libero-spatial-pick_up_the_red_cube",
        help="Any registered LIBERO benchmark name; suite is auto-derived.",
    )
    common.add_argument("--n-episodes", type=int, default=10)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument(
        "--auto-server",
        dest="auto_server",
        action="store_true",
        default=True,
        help="(--policy=groot only) Bring up the GR00T inference service via "
        "`gr00t_inference(action='lifecycle', lifecycle='full', ...)` before "
        "the eval and tear it down on exit. Default: enabled.",
    )
    common.add_argument(
        "--no-auto-server",
        dest="auto_server",
        action="store_false",
        help="(--policy=groot only) Don't manage the inference service; "
        "expect one to already be listening on `--port`.",
    )
    common.add_argument(
        "--image",
        default="gr00t:latest",
        help="(--auto-server only) Docker image tag of the GR00T container. "
        "The image is operator-configured via the STRANDS_GR00T_IMAGE env var "
        "(validated against STRANDS_GR00T_IMAGE_ALLOW); this flag sets that "
        "env var (and extends the allowlist if needed) before calling "
        "`gr00t_inference`.",
    )
    common.add_argument(
        "--container",
        default=None,
        help="(--auto-server only) Docker container name to (re)use. Default: "
        "`gr00t-libero-<backend>`, so Isaac and MuJoCo eval runs don't clobber "
        "each other's containers when run side-by-side on the same host.",
    )
    common.add_argument(
        "--checkpoint-dir",
        default=None,
        help="(--auto-server only) Where to cache the HF checkpoint. "
        "Default: a non-`/home` path (`$STRANDS_ROBOTS_CHECKPOINT_DIR`, an "
        "outside-`/home` `$XDG_CACHE_HOME/strands_robots/checkpoints`, or "
        "`/tmp/strands_robots/checkpoints`). This avoids `gr00t_inference`'s "
        "`start_container` mount guard, which refuses to bind-mount any path "
        "under `/home`.",
    )
    common.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="(--auto-server only) Run the GR00T server through the packaged "
        "determinism wrapper (per-episode server-side reseed + strict cuDNN/"
        "cuBLAS flags) for bit-exact run-to-run reproducibility at a fixed "
        "--seed. See the 'Optional server-side determinism wrapper' section "
        "in the module docstring.",
    )

    parser = argparse.ArgumentParser(
        description="LIBERO evaluation - one driver, per-backend subcommands.",
    )
    sub = parser.add_subparsers(dest="backend", required=True, metavar="{mujoco,isaac}")

    sub.add_parser(
        "mujoco",
        parents=[common],
        help="Evaluate on the default MuJoCo backend (CPU-friendly with --policy mock).",
        description="LIBERO eval on the default MuJoCo backend shipped by strands-robots.",
    )

    isaac = sub.add_parser(
        "isaac",
        parents=[common],
        help="Evaluate on the Isaac Sim backend (needs an Isaac Sim 6.0+ host).",
        description="LIBERO eval on the Isaac Sim backend (create_simulation('isaac')).",
    )
    isaac.add_argument(
        "--robot-usd",
        default=None,
        help="Path / URL to a USD robot asset to load via add_robot(usd_path=...). "
        "Default: Isaac Sim's bundled Franka Panda resolved from the assets root "
        "(Isaac Sim 6.0: `Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd`, "
        "legacy 4.x: `Isaac/Robots/Franka/franka.usd` -- whichever exists). Mutually "
        "exclusive with --robot-urdf.",
    )
    isaac.add_argument(
        "--robot-urdf",
        default=None,
        help="Path to a URDF robot asset to load via add_robot(urdf_path=...). "
        "Mutually exclusive with --robot-usd. Converted to USD on import via "
        "the Isaac URDF importer.",
    )
    isaac.add_argument(
        "--eef-body-name",
        default=_ISAAC_FRANKA_EEF_BODY_NAME,
        help="Robot link whose pose feeds the LIBERO state.x/y/z/roll/pitch/yaw "
        "keys via IsaacSimulation.get_body_state (the MuJoCo defaults "
        "gripper0_grip_site / robot0_right_hand don't exist on the Isaac "
        "Franka USD). Default 'panda_hand' resolves on the bundled Franka; "
        "override when --robot-usd / --robot-urdf loads an asset with "
        "different link names.",
    )
    return parser


def _resolve_container_name(args: argparse.Namespace) -> None:
    """Resolve the ``--container`` default from the parsed subcommand, in place.

    ``--container`` parses to ``None`` so an explicit value stays
    distinguishable from the default; the default is derived here as
    ``gr00t-libero-<backend>`` so the two subcommands never resolve to
    the same container name - the flag's own help text promises MuJoCo
    and Isaac runs don't clobber each other's containers when run
    side-by-side on one host. This preserves the two hardcoded defaults
    the pre-merge ``run_mujoco.py`` / ``run_isaac.py`` shipped. The
    ``is None`` guard (not a truthiness ``or``) is deliberate: an
    explicit value must survive resolution verbatim. Pinned by
    ``tests/test_examples_libero_drivers.py``.
    """
    if args.container is None:
        args.container = f"gr00t-libero-{args.backend}"


def main(args: argparse.Namespace) -> None:
    _resolve_container_name(args)
    suite = _suite_for_task(args.task)

    if args.backend == "isaac":
        if args.robot_usd is not None and args.robot_urdf is not None:
            raise SystemExit("--robot-usd and --robot-urdf are mutually exclusive; pass at most one.")
        # Fail-fast on hosts without Isaac Sim. The is_available probe is
        # cheap (it only does importlib.util.find_spec on omni.isaac.kit /
        # isaacsim; zero omni.* / isaacsim.* modules land in sys.modules)
        # so we run it before touching the GR00T container or any benchmark
        # side effects -- a misconfigured host should exit with a structured
        # error before a docker pull starts.
        from strands_robots.simulation.isaac import IsaacSimulation

        available, reason = IsaacSimulation.is_available()
        if not available:
            raise RuntimeError(
                f"Isaac Sim is not available on this host: {reason}. "
                "Install Isaac Sim 6.0+ via the Omniverse Launcher / Isaac Lab / NGC "
                "Docker image and ensure `isaacsim` (6.0+, Python 3.12) or the legacy "
                "`omni.isaac.kit` is importable in this Python environment."
            )

    # Bring up the GR00T container (idempotent; no-op for --policy=mock
    # or --no-auto-server).
    server_handle = _orchestrate_groot_server(args, suite)
    sim = None
    try:
        if args.backend == "mujoco":
            sim, plan = _setup_mujoco(args, suite)
        else:
            sim, plan = _setup_isaac(args, suite)
        _evaluate_and_report(sim, args, plan)
    finally:
        if sim is not None:
            sim.destroy()
        # Tear down the GR00T inference container if we brought it up.
        if server_handle is not None:
            from strands_robots.tools.gr00t_inference import gr00t_inference

            gr00t_inference(action="lifecycle", lifecycle="teardown", container_name=args.container)


def _cli() -> None:
    args = _build_parser().parse_args()
    if args.backend != "isaac":
        main(args)
        return
    # Isaac-only exit-code epilogue: force a non-zero exit on failure even
    # when Isaac Sim's SimulationApp fast-shutdown has registered an
    # atexit/_exit hook that would otherwise swallow the interpreter's
    # normal non-zero status into a misleading exit 0. ``os._exit(1)``
    # bypasses atexit handlers (including SimulationApp's), so a failed
    # eval is visible to the exit status / CI. The MuJoCo subcommand has
    # no such hook, so it keeps the ordinary traceback-and-exit path.
    import sys
    import traceback

    try:
        main(args)
    except SystemExit:
        raise
    except (KeyboardInterrupt, Exception):  # noqa: BLE001 - top-level: log + force non-zero exit
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    _cli()
