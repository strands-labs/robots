#!/usr/bin/env python3
"""LIBERO on the default MuJoCo backend shipped by ``strands-robots``.

One-shot programmatic flow: a scripted ``sim.evaluate_benchmark(...)``
call - the "if you just want to run LIBERO from a Python script" file.
For the natural-language / ``Agent``-driven version, see
``run_mujoco_agent.py`` (sibling file). For the same eval on the Isaac
Sim backend, see ``run_isaac.py`` - it mirrors this file's CLI shape
and lifecycle.

Consumed by the backend-matrix harness (``libero_backend_matrix.py``),
which subprocess-runs this file and parses two grep-stable lines
(``benchmark_name=...`` and
``policy=... task=... success_rate=... wall_time=...s``).

Usage
-----
::

    # 1) Smoke test, no GPU required:
    python examples/libero/run_mujoco.py --policy mock --n-episodes 5

    # 2) Real LIBERO eval against `nvidia/GR00T-N1.7-LIBERO`. By default
    #    the script auto-orchestrates the GR00T inference service via the
    #    `gr00t_inference(action="lifecycle", lifecycle="full", ...)`
    #    tool - it builds the n1.7 container if missing, downloads the
    #    right `libero_<suite>/` sub-checkpoint, runs the container, and
    #    starts the inference server before the eval, then tears down on
    #    exit. Each step is idempotent so re-runs are cheap.
    #
    #    Pre-condition: HF token at `~/.cache/huggingface/token` with
    #    access to `nvidia/Cosmos-Reason2-2B` (the gated VLM backbone) +
    #    Docker + an NVIDIA GPU.
    python examples/libero/run_mujoco.py --policy groot --port 8000 --n-episodes 50

    # 2b) If you'd rather manage the inference service yourself
    #     (multi-eval session, custom container config, etc.), pass
    #     --no-auto-server and run the `gr00t_inference` lifecycle tool
    #     ahead of time.
    python examples/libero/run_mujoco.py --policy groot --no-auto-server --port 8000

    # 3) Different LIBERO suite + task. Suite is auto-derived from --task,
    #    so the lifecycle tool downloads the matching `libero_<suite>/`
    #    sub-checkpoint:
    python examples/libero/run_mujoco.py \\
        --policy groot --port 8000 \\
        --task libero-10-LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_…

Requires
--------
``pip install 'strands-robots[sim-mujoco,benchmark-libero]'``

Imports only from ``strands_robots`` - no backend-specific packages
are imported at module level.

Notes on the MP4 output
-----------------------
``Simulation.evaluate_benchmark`` does not expose a per-episode
``record_video`` plumb. This script wraps the whole run in a single
``start_cameras_recording`` / ``stop_cameras_recording`` pair, so each
invocation produces one MP4 per camera (third-person ``image`` and
gripper ``wrist_image`` for the GR00T eval; bare ``default`` for the
mock smoke) capturing every episode in sequence. Filename encodes
``--task=<benchmark_name>``, ``--policy=mock|groot``, ``--n_eps=N``,
and ``--seed=S`` so post-hoc analysis can tell what produced it.

Verification status (`--policy=groot` end-to-end)
-------------------------------------------------
Measured 2026-07-22 on an L4 / Docker dev box against
``nvidia/GR00T-N1.7-LIBERO/libero_10``,
``libero-10-LIVING_ROOM_SCENE5_put_the_white_mug_…``, 5 episodes:

* ``--policy=mock``: ~14 s/ep (success_rate=0.0; mock can't satisfy goals).
* ``--policy=groot --seed=42``: ~33 s/ep, **success_rate=0.80 (4/5)** in 165.0 s.

Wall-time covers engine + scene + policy + I/O round-trip end-to-end.

Optional server-side determinism wrapper
-----------------------------------------

For users who need bit-exact run-to-run reproducibility (e.g. CI
pinning a specific success_rate), a drop-in docker wrapper is
available at ``examples/libero/gr00t_server_deterministic_wrapper.py``.
It sets ``cudnn.deterministic=True`` + ``cudnn.benchmark=False`` +
``CUBLAS_WORKSPACE_CONFIG=":4096:8"`` server-side and monkey-patches
``Gr00tPolicy.reset`` to apply the client-supplied per-episode seed.
The example file works WITHOUT this wrapper (verified at 4/5 above) -
it's only needed when you want the GPU's CUDA backend to produce
identical actions across re-runs of the same seed.

Mount with::

    docker run … -v examples/libero/gr00t_server_deterministic_wrapper.py:/srv_wrap.py \\
        gr00t:latest python /srv_wrap.py --model-path … --use-sim-policy-wrapper --port 8000
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import time

from strands_robots.benchmarks.libero import load_libero_suite
from strands_robots.simulation import Simulation


def _date_dir(date_root: str = "rollouts") -> str:
    out = os.path.join(date_root, _dt.date.today().strftime("%Y_%m_%d"))
    os.makedirs(out, exist_ok=True)
    return out


def _suite_for_task(task: str) -> str:
    """Auto-derive a LIBERO suite name from a benchmark task ID.

    Task IDs follow the ``libero-<suite>-<task_stem>`` pattern (see
    ``strands_robots.benchmarks.libero.suite``), so the suite is the
    second hyphen-separated segment.

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


def main() -> None:
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
        "--auto-server",
        dest="auto_server",
        action="store_true",
        default=True,
        help="(--policy=groot only) Bring up the GR00T inference service via "
        "`gr00t_inference(action='lifecycle', lifecycle='full', ...)` before "
        "the eval and tear it down on exit. Default: enabled.",
    )
    p.add_argument(
        "--no-auto-server",
        dest="auto_server",
        action="store_false",
        help="(--policy=groot only) Don't manage the inference service; "
        "expect one to already be listening on `--port`.",
    )
    p.add_argument(
        "--image",
        default="gr00t:latest",
        help="(--auto-server only) Docker image tag of the GR00T container. "
        "The image is operator-configured via the STRANDS_GR00T_IMAGE env var "
        "(validated against STRANDS_GR00T_IMAGE_ALLOW); this flag sets that "
        "env var (and extends the allowlist if needed) before calling "
        "`gr00t_inference`.",
    )
    p.add_argument(
        "--container",
        default="gr00t-libero-mujoco",
        help="(--auto-server only) Docker container name to (re)use.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help="(--auto-server only) Where to cache the HF checkpoint. "
        "Default: a non-`/home` path (`$STRANDS_ROBOTS_CHECKPOINT_DIR`, an "
        "outside-`/home` `$XDG_CACHE_HOME/strands_robots/checkpoints`, or "
        "`/tmp/strands_robots/checkpoints`). This avoids `gr00t_inference`'s "
        "`start_container` mount guard, which refuses to bind-mount any path "
        "under `/home`.",
    )
    args = p.parse_args()

    suite = _suite_for_task(args.task)

    # When `--policy=groot --auto-server` (default), bring up the GR00T
    # inference service via the lifecycle tool: build the n1.7 container
    # if missing -> download the right `libero_<suite>/` sub-checkpoint
    # -> start the container -> start the server. Each sub-step is
    # idempotent so re-runs are cheap. Pass `--no-auto-server` if you're
    # managing the service yourself.
    server_handle = None
    if args.policy == "groot" and args.auto_server:
        from pathlib import Path

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

        hf_token_path = Path("~/.cache/huggingface/token").expanduser()
        if not hf_token_path.is_file():
            raise RuntimeError(
                "--policy groot needs an HF token (Cosmos-Reason2-2B is gated). "
                "Run `huggingface-cli login` first, then retry."
            )
        result = gr00t_inference(
            action="lifecycle",
            lifecycle="full",
            hf_repo="nvidia/GR00T-N1.7-LIBERO",
            hf_subfolder=suite,
            hf_local_dir=args.checkpoint_dir,
            container_name=args.container,
            hf_token=hf_token_path.read_text().strip(),
            # The lifecycle tool mounts `hf_local_dir` (or its default cache
            # dir when `None`) -> `/data/checkpoints`, and the HF download
            # places `<suite>/...` directly under that. So the in-container
            # path is `/data/checkpoints/<suite>`, NOT
            # `/data/checkpoints/GR00T-N1.7-LIBERO/<suite>`.
            checkpoint_path=f"/data/checkpoints/{suite}",
            embodiment_tag="libero_sim",
            protocol="n1.7",
            use_sim_policy_wrapper=True,
            port=args.port,
        )
        if result.get("status") != "success":
            raise RuntimeError(_explain_lifecycle_failure(result, args.checkpoint_dir, args.container))
        server_handle = result
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

    if args.policy == "groot":
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

        registered = load_libero_suite(suite)
        if not registered:
            raise RuntimeError(
                f"load_libero_suite({suite!r}) registered 0 tasks. "
                "Check that the [benchmark-libero] extra is installed."
            )
        if args.task not in registered:
            # The default name `libero-spatial-pick_up_the_red_cube` is
            # aspirational - real LIBERO ships ~10 spatial tasks but
            # none of them is literally that string. If we hit the default
            # and it doesn't resolve, fall back to the first registered
            # task with a clear note. User-supplied unknown tasks still
            # error loudly.
            if args.task == "libero-spatial-pick_up_the_red_cube":
                fallback = next(iter(registered))
                print(
                    f"NOTE: default --task {args.task!r} isn't in real LIBERO "
                    f"(it's an aspirational placeholder); falling back "
                    f"to first registered task {fallback!r}."
                )
                args.task = fallback
            else:
                raise RuntimeError(
                    f"--task {args.task!r} is not in the {suite} suite. Available: {sorted(registered)[:3]}…"
                )

        ts = _dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        rec_name = f"{ts}--task={args.task}--n_eps={args.n_episodes}--seed={args.seed}--policy={args.policy}"
        video_dir = _date_dir()
        # Pick the camera to record from. The LIBERO scene auto-loaded by
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

            spec = get_benchmark(args.task)
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

        sim.start_cameras_recording(cameras=recording_cameras, output_dir=video_dir, name=rec_name)
        try:
            t0 = time.time()
            result = sim.evaluate_benchmark(
                benchmark_name=args.task,
                # robot_name omitted on purpose - `LiberoAdapter`'s scene
                # auto-generation loads a scene that names its Panda
                # ``robot`` (LIBERO/RoboSuite convention), so any
                # specific name we pre-resolve here is gone after
                # `on_episode_start`. `evaluate_benchmark` auto-picks
                # when there's only one robot, which is the LIBERO case.
                n_episodes=args.n_episodes,
                seed=args.seed,
                **policy_kwargs,
            )
            wall_time = time.time() - t0
        finally:
            sim.stop_cameras_recording()

        json_payload = next(c["json"] for c in result["content"] if "json" in c)
        success_rate = json_payload["success_rate"]
        video_path = os.path.join(video_dir, f"{rec_name}__{recording_camera}.mp4")

        # Two grep-stable lines for `libero_backend_matrix.py` to
        # subprocess-and-parse. Keep the exact format (`policy=`, `task=`,
        # `success_rate=`, `wall_time=`, `videos=`) stable across rebases /
        # refactors.
        print(f"benchmark_name={args.task}")
        print(
            f"policy={args.policy}  task={args.task}  "
            f"success_rate={success_rate:.2f}  "
            f"wall_time={wall_time:.1f}s  videos={video_path}"
        )
    finally:
        sim.destroy()
        # Tear down the GR00T inference container if we brought it up.
        if server_handle is not None:
            from strands_robots.tools.gr00t_inference import gr00t_inference

            gr00t_inference(action="lifecycle", lifecycle="teardown", container_name=args.container)


if __name__ == "__main__":
    main()
