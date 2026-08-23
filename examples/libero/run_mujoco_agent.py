#!/usr/bin/env python3
"""LIBERO on MuJoCo, driven by a Strands ``Agent`` in natural language.

The agent receives a single prompt describing the eval, picks
``evaluate_benchmark`` on the registered ``Simulation`` tool, sets the
kwargs from prompt context, runs, and returns a natural-language
summary. For the scripted / deterministic version of the same eval,
see ``run.py`` (sibling file, ``mujoco`` subcommand); for the Isaac Sim equivalent,
see ``run_isaac_agent.py``.

What the script handles deterministically (NOT the agent)
---------------------------------------------------------
LLM agents are reliable at "pick the right tool from a small set, fill
its kwargs from prompt context, summarise the result". They are *not*
reliable at multi-step infrastructure orchestration where each step
has a brittle invariant (docker container names, HF-cache locations,
recorder API selection, scene pre-warm timing). So this script keeps
the latter under deterministic Python control and gives the agent the
one decision it's actually good at: invoking ``evaluate_benchmark``.

Owned by the script:

* GR00T inference container lifecycle (start, wait-for-load, teardown
  on exit) via ``gr00t_inference(action='lifecycle', ...)`` - same
  block as ``run.py``. Idempotent: reuses an already-running
  container with the matching name on ``--port``; no redundant
  checkpoint re-download if the cache is already populated.
* LIBERO scene pre-warm: ``spec.ensure_scene()`` ->
  ``sim.load_scene(...)`` -> ``spec.prewarm(sim)``. Without this the
  GR00T server rejects the first observation with ``Video key
  'video.image' must be in observation`` because the scene's cameras
  haven't been registered yet.
* MP4 recording via ``start_cameras_recording`` (daemon-thread
  recorder, same path as ``run.py``). Under Strands ``Agent``
  tool dispatch the eval runs on a worker thread distinct from the
  recorder thread, and the two race on shared ``mjData``; in practice
  this means lower frame coverage and the occasional greenish GL
  clear-colour artifact. The synchronous-mode alternative -
  ``start_cameras_recording_synchronous`` + ``evaluate_benchmark(
  on_frame=...)`` - eliminates the race by capturing one frame per
  sim step from the eval thread, but its on_frame closure can't cross
  the agent's tool-dispatch JSON boundary, so wiring it requires a
  ``@tool`` wrapper that captures the closure in Python scope.
  Earlier shapes of this file used that wrapper and produced
  bit-exact 1-frame-per-step videos at the cost of (a) a 5-6x
  wall-time multiplier from per-step render, (b) ~70 lines of
  closure-plumbing, (c) reducing the agent's tool surface from the
  full ``Simulation`` action enum to a single wrapper. We chose
  the simpler daemon-thread shape here for matrix-quality wall-time
  and so the agent demo exercises the natural-language -> action-pick
  -> kwarg-fill flow over the real ``Simulation`` surface; users who
  need guaranteed-clean video should use ``run.py mujoco``
  programmatically. The agent does *not* pick a recorder API -
  earlier shapes of this script let the agent decide and it
  consistently picked LeRobot's ``Dataset`` recorder which then
  crashed on ``[Errno 17] File exists: 'rollouts'``.

Owned by the agent: the single ``evaluate_benchmark(...)`` call with
benchmark_name + n_episodes + seed + policy_provider + policy_config
filled from natural language, picked from the registered
``Simulation`` tool's multi-action enum (the prompt names the action
explicitly to prevent verb-matching to ``run_policy`` /
``eval_policy`` which skip the LIBERO observation adapter), plus the
natural-language summary at the end.

Usage
-----
::

    # 1) Smoke test (mock policy; no GPU / Docker needed):
    python examples/libero/run_mujoco_agent.py --policy mock --n-episodes 5

    # 2) Real run against `nvidia/GR00T-N1.7-LIBERO`. Script auto-
    #    orchestrates the GR00T inference container (idempotent). Pre-
    #    condition: an HF token, from HF_TOKEN or `hf auth login` (gated
    #    Cosmos-Reason2-2B backbone) + Docker + an NVIDIA GPU.
    python examples/libero/run_mujoco_agent.py --policy groot --port 8000 --n-episodes 5

    # 2b) Reuse an already-running container instead of letting the
    #     script manage one (e.g. for multi-eval sessions):
    python examples/libero/run_mujoco_agent.py --policy groot --no-auto-server --port 8000

    # 3) Different LIBERO suite + task; suite auto-derived from --task,
    #    so the lifecycle tool downloads the matching `libero_<suite>/`
    #    sub-checkpoint:
    python examples/libero/run_mujoco_agent.py \\
        --policy groot --port 8000 \\
        --task libero-10-LIVING_ROOM_SCENE5_put_the_white_mug_…

Requires
--------
- ``pip install 'strands-robots[sim-mujoco,benchmark-libero]' strands-agents``
- A configured LLM provider for Strands. Default is Anthropic Claude
  via AWS Bedrock - see https://strandsagents.com/ for setup. Without
  one the ``Agent(...)`` call below raises an authentication /
  configuration error pointing at the SDK setup docs.
- For ``--policy=groot``: Docker + an NVIDIA GPU + ~30 GB free disk for
  the GR00T checkpoint (cached across re-runs).

Notes
-----
- Output is non-deterministic by design (LLM-generated summary). The
  backend matrix (``libero_backend_matrix.py``) does not ingest this
  file; the deterministic numbers live in ``run.py`` (sibling
  file).
- Records video to ``rollouts/YYYY_MM_DD/`` with a filename that ends
  ``--policy=mock|groot--agent`` so post-hoc analysis can tell which
  driver produced it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import time

from strands import Agent

from strands_robots.benchmarks.libero import load_libero_suite
from strands_robots.simulation import Simulation

# Import the function from its defining module (not the lazy
# `strands_robots.tools` re-export) so static analysis resolves the
# callable instead of the same-named submodule.
from strands_robots.tools.gr00t_inference import gr00t_inference


def _date_dir(date_root: str = "rollouts") -> str:
    out = os.path.join(date_root, _dt.date.today().strftime("%Y_%m_%d"))
    os.makedirs(out, exist_ok=True)
    return out


def _suite_for_task(task: str) -> str:
    """Auto-derive a LIBERO suite name from a benchmark task ID.

    Same shape as ``run.py``'s helper; see that file for the
    canonical doctest examples.
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
    path under ``/home`` (a "protected host path" guard), so the OOTB
    ``--policy groot`` lifecycle would abort. Default to a non-``/home``
    cache: honor an explicit ``STRANDS_ROBOTS_CHECKPOINT_DIR`` override,
    then fall back to ``$XDG_CACHE_HOME`` only when it lives outside
    ``/home``, else ``/tmp/strands_robots/checkpoints``.
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
    the ``/home`` "protected host path" mount guard (see
    :func:`_default_checkpoint_dir`), and a stale ``gr00t-libero-*``
    container that ``start_container`` won't recreate without
    ``force=True``.
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
    variable -- CI / container environments typically inject the token
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


def _bring_up_gr00t_server(args: argparse.Namespace, suite: str) -> dict | None:
    """Start the GR00T inference container and block until model is loaded.

    Mirrors the lifecycle block in ``run.py`` so the agent file
    has identical "real-eval" plumbing. Returns the lifecycle handle
    (or ``None`` if ``--policy=mock`` / ``--no-auto-server``).
    """
    if args.policy != "groot" or not args.auto_server:
        return None

    import subprocess
    from time import monotonic, sleep

    _configure_gr00t_image(args.image)

    # Default the checkpoint cache to a non-`/home` path so the downloaded
    # checkpoint clears `gr00t_inference`'s `start_container` mount guard.
    # Explicit `--checkpoint-dir` wins.
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

    # Same readiness wait as run.py - the lifecycle tool returns
    # success when the port is bound, but the model loads asynchronously
    # after that. Block until GPU memory crosses a heuristic threshold.
    deadline = monotonic() + 180
    # N1.7 loads to ~6.3 GB on the L4; gate at 4 GiB so the readiness check
    # fires once the model is resident (a 10 GiB gate never tripped - the
    # model footprint is below it - so --auto-server always timed out).
    loaded_threshold_mib = 4_000
    while monotonic() < deadline:
        try:
            used = int(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ]
                )
                .decode()
                .strip()
                .splitlines()[0]
            )
        except Exception:
            used = 0
        if used > loaded_threshold_mib:
            print(f"[setup] GR00T model loaded (gpu_mem={used} MiB)")
            return result
        sleep(5)
    raise RuntimeError(
        "GR00T model didn't reach load threshold within 180 s. Check `docker logs <container>` for stderr."
    )


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

    # Build the policy_config dict that the agent will pass through to
    # `evaluate_benchmark`. We construct it deterministically here so
    # the agent doesn't have to invent dict literals from the prompt.
    if args.policy == "groot":
        policy_phrase = (
            f"using the GR00T policy with `policy_provider='groot'` and "
            f"`policy_config={{'host': 'localhost', 'port': {args.port}, "
            f"'data_config': 'libero_panda', 'groot_version': 'n1.7'}}`"
        )
    else:
        policy_phrase = "using the mock policy (`policy_provider='mock'`)"

    server_handle = _bring_up_gr00t_server(args, suite)

    sim = Simulation(tool_name="libero_sim", mesh=False)
    try:
        sim.create_world()
        # Pre-add a Panda named ``robot`` so:
        #   1. evaluate_benchmark's pre-flight check (`No robots in
        #      sim`) passes BEFORE on_episode_start runs scene loading.
        #   2. The resolved-name `evaluate_benchmark` picks up here
        #      survives the rename that LIBERO scene MJCFs do - the
        #      scenes ship a Franka Panda named `robot` (LIBERO/
        #      RoboSuite convention), so picking the same name here
        #      keeps the resolved robot stable across `on_episode_start`.
        sim.add_robot("robot", data_config="panda")

        registered = load_libero_suite(suite)
        if not registered:
            raise RuntimeError(
                f"load_libero_suite({suite!r}) registered 0 tasks. "
                "Check that the [benchmark-libero] extra is installed."
            )
        if args.task not in registered:
            # The default `libero-spatial-pick_up_the_red_cube` is
            # aspirational; fall back to the first registered task.
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
        rec_name = f"{ts}--task={args.task}--n_eps={args.n_episodes}--seed={args.seed}--policy={args.policy}--agent"
        video_dir = _date_dir()
        recording_cameras = ["image", "wrist_image"] if args.policy == "groot" else ["default"]

        # Pre-warm the LIBERO scene so the cameras the GR00T server
        # expects (`image`, `wrist_image`) are registered before
        # recording / inference starts. Same block as run.py;
        # see that file for the longer rationale on ordering.
        if args.policy == "groot":
            from strands_robots.simulation.benchmark import get_benchmark

            spec = get_benchmark(args.task)
            if hasattr(spec, "ensure_scene"):
                spec.ensure_scene()
            if spec.scene_path:
                sim.load_scene(spec.scene_path)
                if hasattr(spec, "prewarm"):
                    spec.prewarm(sim)
                if "robot" not in sim.list_robots():
                    sim.add_robot("robot", data_config="panda")

        # Let the agent pick `evaluate_benchmark` from the full
        # `Simulation` action surface via natural language. Trade-off:
        # must use the daemon-thread recorder (`start_cameras_recording`,
        # NOT the synchronous variant) because the synchronous mode
        # returns Python closures that can't cross Strands' tool-dispatch
        # JSON boundary. Daemon-thread recording under the Strands worker
        # thread races with the eval thread on shared `mjData` and
        # produces reduced frame capture plus occasional greenish
        # artifacts - see the module docstring for the full trade-off.
        sim.start_cameras_recording(cameras=recording_cameras, output_dir=video_dir, name=rec_name)
        try:
            agent = Agent(tools=[sim])
            t0 = time.time()
            # Explicit instruction to use the `evaluate_benchmark`
            # action - the Simulation tool exposes dozens of sub-actions
            # and the agent will otherwise verb-match "run" -> `run_policy`
            # or "eval" -> `eval_policy`, both of which skip the LIBERO
            # observation adapter and trigger server-side rejections
            # like "State key 'state.x' must be in observation".
            result = agent(
                f"Make exactly one tool call: invoke the `libero_sim` "
                f"tool with `action='evaluate_benchmark'`, "
                f"`benchmark_name='{args.task}'`, "
                f"`n_episodes={args.n_episodes}`, `seed={args.seed}`, "
                f"`robot_name='robot'`, {policy_phrase}. Do not call "
                f"any other action - the world, robot, scene, and "
                f"video recording have already been set up. When the "
                f"call returns, parse the `success_rate` field from "
                f"the JSON payload and report it as a percentage of "
                f"the {args.n_episodes} episodes."
            )
            wall_time = time.time() - t0
            print(result)
            video_path = os.path.join(video_dir, f"{rec_name}__{recording_cameras[0]}.mp4")
            print(f"[agent-eval] policy={args.policy} task={args.task} wall_time={wall_time:.1f}s videos={video_path}")
        finally:
            sim.stop_cameras_recording()
    finally:
        try:
            sim.destroy()
        except Exception as exc:
            # Best-effort teardown: destroy() can fail if the world was
            # already torn down mid-eval. Report it, but never let cleanup
            # mask the primary error or skip the container teardown below.
            print(f"[agent-eval] warning: sim.destroy() failed during cleanup: {exc}")
        # Tear down the GR00T inference container if we brought it up.
        if server_handle is not None:
            gr00t_inference(action="lifecycle", lifecycle="teardown", container_name=args.container)


if __name__ == "__main__":
    main()
