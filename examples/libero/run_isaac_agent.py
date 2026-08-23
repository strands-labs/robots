#!/usr/bin/env python3
"""LIBERO on Isaac Sim, driven by a Strands ``Agent`` in natural language.

Companion to ``run_mujoco_agent.py``: same prompt-to-tool-pick agent
shape, same one-shot ``evaluate_benchmark`` invocation, same GR00T
container lifecycle. Differs in backend choice
(``IsaacSimulation``) and the tool-wrapping shape: where the MuJoCo
file passes the ``Simulation`` instance directly to ``Agent(tools=...)``
because ``MuJoCoSimEngine`` inherits from ``AgentTool`` and exposes
its 64-action enum, this Isaac file wraps :meth:`IsaacSimulation.evaluate_benchmark`
in a single ``@tool``-decorated function and passes that to the agent.

Status (as of 2026-06)
----------------------
Target runtime is the Isaac Sim 6.0 NGC docker image
(``nvcr.io/nvidia/isaac-sim:6.0``, Python 3.12, RTX/L4 GPU, headless),
matching the ``isaacsim>=6.0`` / ``requires-python>=3.12`` migration.
The end-to-end lifecycle was previously validated on the Isaac Sim 4.5
image (``nvcr.io/nvidia/isaac-sim:4.5.0``); the dual-path
``isaacsim.*`` / ``omni.isaac.*`` imports keep the same code path
working on both. The script
runs past ``IsaacSimulation.is_available`` → ``create_world`` →
``add_robot`` (real Franka USD over the Omniverse CDN) → ``add_camera``,
and the agent invokes ``evaluate_isaac_benchmark`` on a one-tool
surface.

LIBERO benchmark on Isaac is NOT yet runnable end-to-end
--------------------------------------------------------
The agent's ``evaluate_isaac_benchmark`` tool call routes through the
same ``IsaacSimulation.evaluate_benchmark`` path as ``run.py isaac``,
so it runs the same way: ``LiberoAdapter.on_episode_start`` calls
``sim.load_scene(...)`` and ``IsaacSimulation.load_scene`` now realizes
the LIBERO/BDDL-compiled MJCF as USD prims on the Isaac stage (the
substantive LIBERO-on-Isaac work -- implemented in
`#129 <https://github.com/strands-labs/robots-sim/issues/129>`_, which
superseded the fail-fast stub PR #117 shipped for the closed #116). The
eval completes end-to-end; the agent surfaces the ``success_rate`` in
its summary, and this script **exits non-zero** only on a genuine eval
error (the ``__main__`` guard forces ``os._exit(1)`` so Isaac's
SimulationApp fast-shutdown can't swallow a failure into exit 0). A
*meaningful* (non-zero) ``success_rate`` also needs the articulation
fix (#123). Use ``examples/libero/run_mujoco_agent.py`` for the
CPU-friendly reference path, or drive the Isaac backend directly via
the manual ``create_world`` -> ``add_robot`` -> ``add_object`` ->
``add_camera``
-> ``step`` -> ``render`` quickstart in ``docs/index.md`` (which works
end-to-end on Isaac Sim 6.0).

The earlier ``--policy=groot`` ``success_rate=0.0`` warning predates
PRs `#61 <https://github.com/strands-labs/robots-sim/pull/61>`_ /
`#70 <https://github.com/strands-labs/robots-sim/pull/70>`_ landing the
camera + Articulation Phase-2 wiring; with both merged on ``main``,
the procedural / real-asset robot load + camera frames are now wired
correctly. End-to-end ``--policy=groot`` numbers still need a host
that satisfies all of: Isaac Sim 6.0+, libero (BDDL files),
``strands-robots`` under Isaac's Python (#71), and a GR00T inference
container.

Why a single-tool wrapper instead of full enum
-----------------------------------------------
``run_mujoco_agent.py``'s preamble notes that earlier sessions
"reduced the agent's tool surface from the full 64-action ``Simulation``
enum to a single wrapper" and judged that **worse** for MuJoCo
because: (a) MuJoCo already has the AgentTool wiring, (b) the full
enum exercises the natural-language → action-pick → kwarg-fill flow
the agent demo is meant to showcase. The Isaac case is different:

- ``IsaacSimulation`` does NOT yet inherit ``AgentTool``; it extends
  the ``SimEngine`` ABC alone. Wiring it up to mirror MuJoCo's
  ``MuJoCoSimEngine(PhysicsMixin, RenderingMixin, RecordingMixin,
  RandomizationMixin, SimEngine, AgentTool)`` is its own (substantial)
  refactor on `#14 <https://github.com/strands-labs/robots-sim/issues/14>`_'s
  Phase 3 list.
- Until that refactor lands, the only way to give the agent a tool
  is the ``@tool`` decorator on a hand-written wrapper. The wrapper
  necessarily collapses the full surface -- the ``Simulation`` tool's
  64-action JSON enum has no ``IsaacSimulation`` equivalent yet.

So the agent here picks ``evaluate_isaac_benchmark`` from a 1-tool
surface, fills its kwargs from prompt context, runs, and summarises.
The agent demo still exercises the prompt → tool-call → kwarg-fill
path; it's just a degenerate "1-of-1 pick". Once the Isaac AgentTool
wrapper exists in ``strands-robots``,
this file should be migrated to the full-enum shape (single-line
diff: drop ``@tool`` wrapper + change ``Agent(tools=[...])`` arg) so
the agent regains the natural-language action-pick demo value.

What the script handles deterministically (NOT the agent)
---------------------------------------------------------
Same partition as ``run_mujoco_agent.py``. The agent owns ONE
decision (invoking ``evaluate_isaac_benchmark`` with kwargs filled
from prompt context); the script owns:

* ``is_available()`` short-circuit on a non-Isaac host -- the cheap
  ``importlib.util.find_spec("omni.isaac.kit") | find_spec("isaacsim")``
  probe runs before the GR00T container side effects so a misconfigured
  matrix run exits with a structured ``RuntimeError`` rather than wasting
  30 s on a docker pull.
* GR00T inference container lifecycle (start, wait-for-load, teardown
  on exit) via ``gr00t_inference(action='lifecycle', ...)`` -- same
  block as ``run_mujoco_agent.py``, container name defaulted to
  ``gr00t-libero-isaac`` so Isaac and MuJoCo eval runs don't collide
  on the same host.
* Procedural Panda + RTX camera setup. Isaac doesn't auto-attach
  viewport cameras the way MuJoCo's ``mjData`` does, so the camera
  prim has to land on the stage before any ``render`` / recorder
  pulls from it.
* The Isaac threading model (#1896). Isaac's kit runtime only pumps
  updates on the thread that created ``SimulationApp``, while Strands
  executes tool calls on worker threads -- driving the eval from the
  tool thread directly deadlocked forever (``SimulationContext.stop()``
  waiting on a pump the parked main thread could never run). ``main()``
  therefore runs the ``Agent(...)`` call on a side thread, parks the
  main thread in ``IsaacSimulation.run_pump_forever`` until the agent
  finishes, and the tool wrapper submits the eval body back to the main
  thread via ``run_on_main`` -- the same pump-on-main + worker-callers
  architecture as ``examples/isaac_gs/app.py``.
* Rollout-video MP4 recording, at parity with
  ``run_mujoco_agent.py``: the script arms
  ``IsaacSimulation.start_cameras_recording`` before the agent runs and
  flushes via ``stop_cameras_recording`` after. Because Isaac's RTX
  renderer is thread-bound, capture is synchronous -- the recorder's
  ``on_frame`` closure (stashed at module scope so it can cross the
  ``@tool`` boundary) is threaded into ``evaluate_benchmark`` and grabs
  one ``render()`` frame per control step on the main (pump) thread,
  where ``run_on_main`` executes the eval. Produces a
  real ``rollouts/<date>/{rec_name}__image.mp4``. See
  strands-labs/robots-sim#112.

Owned by the agent: the single ``evaluate_isaac_benchmark(...)`` call
with ``benchmark_name`` + ``n_episodes`` + ``seed`` + ``policy_provider`` +
``policy_config`` filled from natural language, plus the natural-language
summary at the end.

Usage
-----
::

    # 1) Smoke test (mock policy; no GPU / Docker needed beyond Isaac
    #    Sim itself):
    python examples/libero/run_isaac_agent.py --policy mock --n-episodes 5

    # 2) Real run against `nvidia/GR00T-N1.7-LIBERO`. Script auto-
    #    orchestrates the GR00T inference container (idempotent). Pre-
    #    condition: an HF token, from HF_TOKEN or `hf auth login` (gated
    #    Cosmos-Reason2-2B backbone) + Docker + an NVIDIA GPU + Isaac
    #    Sim 6.0+ installed.
    python examples/libero/run_isaac_agent.py --policy groot --port 8000 --n-episodes 5

Requires
--------
- ``pip install 'strands-robots[sim-isaac,benchmark-libero]' strands-agents``
- A configured LLM provider for Strands. Default is Anthropic Claude
  via AWS Bedrock -- see https://strandsagents.com/ for setup. Without
  one the ``Agent(...)`` call below raises an authentication /
  configuration error pointing at the SDK setup docs.
- Isaac Sim 6.0+ installed via Omniverse Launcher / Isaac Lab / NGC
  Docker image (Python 3.12). Pure-Python ``pip install`` doesn't suffice.
- For ``--policy=groot``: Docker + an NVIDIA GPU + ~30 GB free disk
  for the GR00T checkpoint (cached across re-runs).

Notes
-----
- Output is non-deterministic by design (LLM-generated summary); the
  R15 backend matrix consumes ``run.py isaac`` (sibling file) for
  grep-stable numbers.
- Rollout video is recorded via the synchronous Isaac recorder (one
  ``render()`` frame per control step), producing a real
  ``rollouts/<date>/{rec_name}__image.mp4`` at parity with the MuJoCo
  agent.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _dt
import os
import threading
import time
from typing import Any

from strands import Agent, tool

from strands_robots.benchmarks.libero import load_libero_suite
from strands_robots.simulation import create_simulation
from strands_robots.simulation.isaac import IsaacSimulation
from strands_robots.tools import gr00t_inference

# Module-level handle for the @tool-wrapped function below.
# The wrapper has to access ``_sim`` from outer scope because
# ``@tool``'s OpenAPI-schema generator inspects the function's
# signature and would surface ``sim: IsaacSimulation`` to the LLM as
# a JSON-castable parameter -- which it isn't. Keeping ``_sim`` as a
# module attribute is the cleanest stopgap until the Isaac AgentTool
# wrapper lands and the indirection goes away.
_sim: IsaacSimulation | None = None

# Module-level rollout-video capture hook. Set in main() from the
# recorder's start_cameras_recording result and consumed by the tool
# wrapper below. It lives at module scope (not in the tool's signature)
# for the same reason ``_sim`` does: the closure isn't JSON-castable, so
# it can't cross @tool's OpenAPI-schema boundary. Threading it through
# module scope lets the synchronous Isaac recorder capture frames on the
# main (pump) thread -- where the tool wrapper's ``run_on_main`` executes
# the eval -- even though the eval is dispatched from a Strands tool
# call. See strands-labs/robots#191 for the closure-vs-tool-boundary
# rationale on the MuJoCo side.
_on_frame: Any = None

# Module-level capture of the last ``evaluate_isaac_benchmark`` result
# envelope. The agent consumes the tool's dict and folds it into a
# natural-language summary, so ``main()`` can't see the raw
# ``{"status": ...}`` directly. Stashing it here lets ``main()``
# deterministically fail-fast (non-zero exit) when the eval errored.
# Without this, the agent would happily summarise the failure and the
# script would still exit 0.
_last_eval_result: dict[str, Any] | None = None


def _date_dir(date_root: str = "rollouts") -> str:
    out = os.path.join(date_root, _dt.date.today().strftime("%Y_%m_%d"))
    os.makedirs(out, exist_ok=True)
    return out


def _suite_for_task(task: str) -> str:
    """Auto-derive a LIBERO suite name from a benchmark task ID.

    Same shape as ``run_mujoco_agent.py``'s helper; see that file for
    the canonical doctest examples.
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
    """Start the GR00T inference container and block until model loads.

    Mirrors the lifecycle block in ``run_mujoco_agent.py`` so the Isaac
    agent file has identical "real-eval" plumbing. Returns the
    lifecycle handle (or ``None`` if ``--policy=mock`` /
    ``--no-auto-server``).
    """
    if args.policy != "groot" or not args.auto_server:
        return None

    import subprocess
    from time import monotonic, sleep

    hf_token = _resolve_hf_token()
    _configure_gr00t_image(args.image)

    # Default the checkpoint cache to a non-`/home` path so the downloaded
    # checkpoint clears `gr00t_inference`'s `start_container` mount guard
    # (strands-labs/robots-sim#125). Explicit `--checkpoint-dir` wins.
    if args.checkpoint_dir is None:
        args.checkpoint_dir = _default_checkpoint_dir()
    print(f"[setup] checkpoint dir: {args.checkpoint_dir}")

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

    deadline = monotonic() + 180
    # N1.7 loads to ~6.3 GB on the L4; gate at 4 GiB so the readiness check
    # fires once the model is resident (the old 10 GiB gate never tripped —
    # the model footprint is below it — so --auto-server always timed out).
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


@tool(
    name="evaluate_isaac_benchmark",
    description=(
        "Run a registered LIBERO benchmark on the configured Isaac Sim "
        "environment. The world, robot, and camera have already been "
        "set up by the calling script -- this tool only invokes the "
        "evaluation loop. Returns a status dict whose JSON payload "
        "carries `success_rate`, `avg_reward`, `avg_steps`, plus per-"
        "episode cumulative reward."
    ),
)
def evaluate_isaac_benchmark(
    benchmark_name: str,
    n_episodes: int = 10,
    seed: int = 42,
    policy_provider: str = "mock",
    policy_config: dict[str, Any] | None = None,
    instruction: str = "",
) -> dict[str, Any]:
    """Tool wrapper around :meth:`IsaacSimulation.evaluate_benchmark`.

    Stopgap until ``IsaacSimulation`` itself becomes a Strands
    ``AgentTool`` (Phase-3 work on #14). Forwards a fixed-shape
    subset of ``evaluate_benchmark``'s kwargs -- the ones the agent
    needs to fill from prompt context -- onto the module-scoped
    ``_sim`` instance configured in :func:`main`.

    Parameters
    ----------
    benchmark_name : str
        Registered LIBERO task name, e.g.
        ``"libero-spatial-pick_up_the_red_cube"``.
    n_episodes : int
        Number of episodes to roll out. Default 10.
    seed : int
        Master RNG seed for per-episode reproducibility. Default 42.
    policy_provider : str
        Strands policy registry key. ``"mock"`` (default) or
        ``"groot"``.
    policy_config : dict, optional
        Provider-specific kwargs. For ``policy_provider="groot"``,
        carries ``host`` / ``port`` / ``data_config`` /
        ``groot_version``.
    instruction : str
        Optional natural-language instruction forwarded to the policy.

    Returns
    -------
    dict
        Standard ``{"status", "content": [...]}`` envelope. On success,
        ``content[0]["json"]`` carries ``success_rate``, ``avg_reward``,
        ``avg_steps``, and per-episode cumulative reward.
    """
    if _sim is None:
        return {
            "status": "error",
            "content": [{"text": "evaluate_isaac_benchmark: _sim is not initialised. main() must run first."}],
        }
    # Isaac's kit runtime is main-thread affine: world.reset()/step() only
    # complete while kit updates are being pumped on the thread that created
    # SimulationApp. Strands executes tool calls on worker threads, so running
    # the eval here directly deadlocked forever (#1896: the main thread was
    # parked inside Agent.__call__ waiting on this tool's future while
    # SimulationContext.stop() waited for a pump only that thread could run).
    # main() therefore parks the main thread in run_pump_forever for the
    # agent's whole run, and this wrapper submits the eval body back to it:
    # run_on_main executes the closure inline on the pump (owning) thread and
    # re-raises any exception here, so the @tool envelope contract is intact.
    sim = _sim
    result = sim.run_on_main(
        lambda: sim.evaluate_benchmark(
            benchmark_name=benchmark_name,
            n_episodes=n_episodes,
            seed=seed,
            policy_provider=policy_provider,
            policy_config=policy_config,
            instruction=instruction,
            # Thread the module-scoped rollout-video capture hook into the
            # eval. ``_on_frame`` is None until main() arms the recorder, so
            # this is a no-op for callers that don't want video.
            on_frame=_on_frame,
        )
    )
    # Stash the raw envelope so main() can fail-fast on an eval error
    # the agent would otherwise only mention in prose.
    global _last_eval_result
    _last_eval_result = result
    return result


def _build_parser() -> argparse.ArgumentParser:
    """Mirror ``run_mujoco_agent.py``'s parser surface.

    Argument names / defaults are kept identical to the MuJoCo file
    (and to ``run.py isaac``) so a matrix-driver shell wrapper that
    supplies the same flags works against any of the three.
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
        help="USD robot asset for add_robot(usd_path=...). Default: bundled Franka "
        "Panda from the assets root. Mutually exclusive with --robot-urdf.",
    )
    p.add_argument(
        "--robot-urdf",
        default=None,
        help="URDF robot asset for add_robot(urdf_path=...). Mutually exclusive with --robot-usd.",
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
        help="(--auto-server only) Docker container name to (re)use. Defaults "
        "to the Isaac-specific name so Isaac and MuJoCo eval runs don't "
        "collide on the same host.",
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


def _asset_exists(url: str) -> bool | None:
    """Best-effort HEAD-probe for an asset URL.

    Returns ``True`` / ``False`` when the probe is conclusive, or ``None``
    when it can't be determined (non-HTTP URL such as an ``omniverse://``
    Nucleus path, or a network error).
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
# NVIDIA relocated the asset under a vendor folder in Isaac Sim 6.0; probe
# the 6.0 path first, then fall back to the legacy 4.x path. See
# strands-labs/robots-sim#110.
_FRANKA_USD_SUBPATHS = (
    "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",  # Isaac Sim 6.0+
    "Isaac/Robots/Franka/franka.usd",  # Isaac Sim 4.x and earlier
)


def _resolve_default_franka_usd(assets_root: str) -> str:
    """Pick the Franka USD candidate that exists under ``assets_root``.

    HEAD-probes each candidate in :data:`_FRANKA_USD_SUBPATHS` order and
    returns the first that resolves. Falls back to the first (6.0)
    candidate when no probe is conclusive; raises with an actionable hint
    only when every HTTP candidate definitively 404s.
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

    Same contract as ``run._resolve_robot_asset`` (sibling ``run.py``):
    ``--robot-urdf`` > ``--robot-usd`` > default Franka Panda USD from the
    assets root. The
    asset sub-path moved under a vendor folder in Isaac Sim 6.0
    (``Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd``) from the legacy
    4.x layout (``Isaac/Robots/Franka/franka.usd``); the resolver
    HEAD-probes both and uses whichever exists. Loads a *real* robot rather
    than the procedural stick-figure (see ``run.py`` for the
    rationale). ``get_assets_root_path`` is imported lazily (only resolvable
    after ``create_world``) and tries the modern ``isaacsim.storage.native``
    namespace first, falling back to the legacy ``omni.isaac.nucleus`` shim
    -- matches the dual-path policy in
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


def main() -> None:
    global _sim, _on_frame, _last_eval_result
    _last_eval_result = None

    args = _build_parser().parse_args()
    if args.robot_usd is not None and args.robot_urdf is not None:
        raise SystemExit("--robot-usd and --robot-urdf are mutually exclusive; pass at most one.")
    suite = _suite_for_task(args.task)

    # Fail-fast on hosts without Isaac Sim. Same probe as run.py isaac
    # -- runs before the GR00T container side effects so a CPU-only
    # host exits cleanly without wasting docker bandwidth.
    available, reason = IsaacSimulation.is_available()
    if not available:
        raise RuntimeError(
            f"Isaac Sim is not available on this host: {reason}. "
            "Install Isaac Sim 6.0+ via the Omniverse Launcher / Isaac Lab / NGC "
            "Docker image and ensure `isaacsim` (6.0+, Python 3.12) or the legacy "
            "`omni.isaac.kit` is importable in this Python environment."
        )

    # Build the policy-config phrase the agent will paste into its
    # ``evaluate_isaac_benchmark`` call. Constructed deterministically
    # here so the agent doesn't have to invent dict literals from the
    # prompt -- mirrors run_mujoco_agent.py's policy_phrase pattern.
    if args.policy == "groot":
        policy_phrase = (
            f"with `policy_provider='groot'` and `policy_config={{'host': 'localhost', "
            f"'port': {args.port}, 'data_config': 'libero_panda', "
            f"'groot_version': 'n1.7'}}`"
        )
    else:
        policy_phrase = "with `policy_provider='mock'`"

    server_handle = _bring_up_gr00t_server(args, suite)

    # Construct via the strands-robots factory: strands-robots>=0.4.1
    # walks the `strands_robots.backends` entry-point group, so
    # create_simulation("isaac", ...) resolves to this package's
    # IsaacSimulation. render_mode="rtx_realtime" makes render() take the
    # RTX frame path instead of returning zero-filled (blank) frames in
    # the default render_mode="headless" -- the latter produced all-black
    # rollout MP4s. headless=True only suppresses the Kit viewport, not
    # the render pipeline. (STRANDS_ISAAC_RTX_PATHTRACING=1 -> photoreal.)
    _sim = create_simulation("isaac", headless=True, num_envs=1, render_mode="rtx_realtime")
    try:
        result = _sim.create_world()
        if result.get("status") != "success":
            raise RuntimeError(f"create_world failed: {result}")

        # Load a *real* robot asset (default: bundled Franka Panda USD;
        # override via --robot-usd / --robot-urdf). Routes through
        # add_robot's usd_path / urdf_path branch (real Articulation,
        # observable joints) rather than the procedural builder, which
        # produces a kinematically-approximate stick-figure unusable for
        # LIBERO. See run.py's _resolve_robot_asset docstring.
        robot_usd, robot_urdf = _resolve_robot_asset(args)
        if robot_urdf is not None:
            print(f"[setup] loading robot from URDF: {robot_urdf}")
            result = _sim.add_robot(name="robot", urdf_path=robot_urdf)
        else:
            print(f"[setup] loading robot from USD: {robot_usd}")
            result = _sim.add_robot(name="robot", usd_path=robot_usd)
        if result.get("status") != "success":
            raise RuntimeError(f"add_robot failed: {result}")

        # Phase-2 RTX camera at the same over-the-shoulder vantage
        # LIBERO's `agentview` uses on MuJoCo. With #61 + #62 merged it's
        # a real `isaacsim.sensors.camera.Camera` whose frames feed both
        # `--policy=groot` and the rollout-video recorder armed below.
        recording_camera = "image"
        result = _sim.add_camera(name=recording_camera, position=[2.0, 0.0, 1.5], target=[0.0, 0.0, 0.5], fov=60.0)
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
        warmed_up = False
        for _ in range(_RTX_WARMUP_MAX_ITERS):
            _sim.step(1)
            probe = _sim.render(camera_name=recording_camera)
            if probe.get("status") == "success":
                warmed_up = True
                break
        if not warmed_up:
            raise RuntimeError(
                f"RTX render product for camera {recording_camera!r} never "
                f"accumulated a frame after {_RTX_WARMUP_MAX_ITERS} warm-up "
                "step+render iterations; recording would produce a 0-frame "
                f"(dropped) mp4. Last render probe: {probe}"
            )
        print(f"[setup] RTX camera {recording_camera!r} warmed up; render product populated")

        # Resolve the LIBERO task. Same default-aspirational fallback
        # as run_mujoco_agent.py / run.py. Keep the CLI-requested
        # task distinct from the resolved one so the [agent-eval] line
        # below echoes what the caller passed (replayable) while the
        # actual eval / filename use what really ran.
        requested_task = args.task
        registered = load_libero_suite(suite)
        if not registered:
            raise RuntimeError(
                f"load_libero_suite({suite!r}) registered 0 tasks. "
                "Apply upstream fix from strands-labs/robots#147 if it isn't merged."
            )
        if args.task not in registered:
            if args.task == "libero-spatial-pick_up_the_red_cube":
                fallback = next(iter(registered))
                print(
                    f"NOTE: default --task {args.task!r} isn't in real LIBERO "
                    f"(it's the spec's aspirational placeholder); falling back "
                    f"to first registered task {fallback!r}."
                )
                args.task = fallback
            else:
                raise RuntimeError(
                    f"--task {args.task!r} is not in the {suite} suite. Available: {sorted(registered)[:3]}…"
                )
        resolved_task = args.task

        # Arm the synchronous Isaac recorder BEFORE the agent runs, so the
        # `evaluate_isaac_benchmark` tool call captures a real MP4 -- at
        # parity with run_mujoco_agent.py's start/stop_cameras_recording
        # pair. The recorder returns an `on_frame` closure; we stash it at
        # module scope (`_on_frame`) because it can't cross @tool's
        # JSON-schema boundary (see #191). The tool wrapper threads it into
        # `evaluate_benchmark(on_frame=...)` so frames are captured on the
        # main (pump) thread where `run_on_main` executes the eval (Isaac's
        # RTX renderer is thread-bound). On stop the
        # buffers flush to `rollouts/<date>/{rec_name}__image.mp4`, matching
        # MuJoCo's `{name}__{camera}.mp4` convention. See
        # strands-labs/robots-sim#112.
        ts = _dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        rec_name = (
            f"{ts}--task={resolved_task}--n_eps={args.n_episodes}"
            f"--seed={args.seed}--policy={args.policy}--backend=isaac--agent"
        )
        video_dir = _date_dir()
        rec = _sim.start_cameras_recording(
            cameras=[recording_camera],
            output_dir=video_dir,
            name=rec_name,
        )
        if rec.get("status") != "success":
            raise RuntimeError(f"start_cameras_recording failed: {rec}")
        _on_frame = next(c["json"]["on_frame"] for c in rec["content"] if "json" in c)
        video_path = os.path.join(video_dir, f"{rec_name}__{recording_camera}.mp4")

        # Hand the agent a 1-tool surface (`evaluate_isaac_benchmark`)
        # and a prompt that fills the eval kwargs from --task /
        # --n-episodes / --seed / --policy. The "1-of-1 tool pick" is
        # the degenerate shape that lets us ship the agent demo today
        # against an IsaacSimulation that doesn't yet inherit
        # AgentTool. See module docstring for migration plan.
        agent = Agent(tools=[evaluate_isaac_benchmark])
        prompt = (
            f"Make exactly one tool call: invoke `evaluate_isaac_benchmark` "
            f"with `benchmark_name='{resolved_task}'`, "
            f"`n_episodes={args.n_episodes}`, `seed={args.seed}`, "
            f"{policy_phrase}. Do not call any other action -- the world, "
            f"robot, and camera have already been set up. When the call "
            f"returns, parse the `success_rate` field from the JSON "
            f"payload and report it as a percentage of the {args.n_episodes} "
            f"episodes."
        )

        # Threading model (#1896): Isaac's kit runtime only pumps updates on
        # the thread that created SimulationApp -- this main thread. Strands
        # executes the agent's tool calls on worker threads, where
        # world.reset()/step() block forever waiting for a pump that can never
        # run while the main thread is itself parked inside Agent.__call__
        # waiting on the tool future. So the agent runs on a side thread, the
        # main thread owns the kit pump (run_pump_forever), and the tool
        # wrapper above submits the eval body back here via run_on_main --
        # the same pump-on-main + worker-callers architecture as
        # examples/isaac_gs/app.py (Gradio in a daemon thread). This also
        # keeps every RTX render on the owning thread, which the thread-bound
        # recorder requires.
        #
        # The side thread is a ThreadPoolExecutor rather than a hand-rolled
        # Thread + result box (review on #1899): since we create this thread
        # ourselves, `Future.result()` re-raises the agent's exception --
        # RuntimeError, SystemExit and KeyboardInterrupt alike -- with object
        # identity preserved, whereas a manual `except BaseException` box is
        # only obliged when marshalling onto an *existing* foreign thread
        # (run_on_main). The done-callback fires on success, exception and
        # cancellation, so a prompt Agent(...) failure (e.g. no model
        # credentials, see Requires above) still sets `agent_done` and the
        # pump exits promptly rather than hanging. One caveat vs. the old
        # daemon thread: executor threads are non-daemon and joined by
        # concurrent.futures' atexit hook, so `shutdown(wait=False,
        # cancel_futures=True)` on an interrupted-pump exit defers to that
        # hook instead of abandoning the thread; on the normal path it is
        # moot -- the callback only fires once the agent call has returned.
        agent_done = threading.Event()
        t0 = time.time()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="strands-agent")
        try:
            future = pool.submit(agent, prompt)
            future.add_done_callback(lambda _f: agent_done.set())
            try:
                _sim.run_pump_forever(stop_event=agent_done)
            finally:
                stop = _sim.stop_cameras_recording()
                print(f"[recording] {stop['content'][0]['text']}")
            # Re-raises the agent's exception, identity intact.
            result = future.result()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        wall_time = time.time() - t0
        print(result)

        # Fail-fast (non-zero exit) if the agent's eval tool call errored.
        # The agent only summarises the failure in prose; main() inspects
        # the raw envelope the tool stashed so a genuine LIBERO-on-Isaac
        # eval error is visible to the exit status / CI rather than
        # swallowed into the agent's natural-language report + exit 0.
        if _last_eval_result is None:
            raise RuntimeError(
                "Agent did not invoke evaluate_isaac_benchmark (no eval result captured). "
                "Expected exactly one tool call; check the agent transcript above."
            )
        if _last_eval_result.get("status") != "success":
            err_text = (_last_eval_result.get("content") or [{}])[0].get("text", "")
            raise RuntimeError(f"evaluate_isaac_benchmark failed: {err_text}")

        # Echo the CLI-requested task (replayable) plus the resolved one.
        print(
            f"[agent-eval] policy={args.policy} task={requested_task} "
            f"resolved_task={resolved_task} wall_time={wall_time:.1f}s videos={video_path}"
        )
    finally:
        try:
            if _sim is not None:
                _sim.destroy()
        except Exception:  # noqa: BLE001
            pass  # best-effort teardown; a destroy() failure must not mask the eval result
        _sim = None
        _on_frame = None
        if server_handle is not None:
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
