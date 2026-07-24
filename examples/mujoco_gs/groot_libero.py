# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic GR00T-on-LIBERO runner for the MuJoCo-GS demo.

A real NVIDIA GR00T vision-language-action policy drives a Franka **Panda**
through a **LIBERO** task, invoked **agentically**: a Strands ``Agent`` picks
``evaluate_benchmark`` off the ``Simulation`` tool's action surface and fills
its kwargs from a natural-language instruction (the pattern from
``examples/libero/run_mujoco_agent.py``). Meanwhile a background thread renders
the scene through the :class:`HybridCompositor` into a JPEG buffer so the UI's
``/live`` MJPEG route shows the arm in near-real-time, and collects frames for
a recorded clip.

Getting a *successful* episode (the recipe verified against
``nvidia/GR00T-N1.7-LIBERO``):

* **Match the task suite to the served checkpoint.** The bundled container
  serves ``/data/checkpoints/libero_10``, so run ``libero_10`` tasks. Running a
  different suite against it is an embodiment/skill mismatch and scores ~0.
* **Don't cap ``max_steps``.** LIBERO-Long episodes need ~500 steps; capping
  them truncates before completion. We use the adapter's default.
* **Pre-warm the scene** (generate BDDL scene → ``load_scene`` → ``prewarm``)
  so the ``image``/``wrist_image`` cameras exist before inference/recording.
* **Let ``evaluate_benchmark`` auto-pick the robot** (omit ``robot_name``) and
  use its **default ``action_horizon``** — the LIBERO scene renames its Panda
  to ``robot`` on episode start.

Concurrency note: only the eval thread (sim's internal camera renders for the
policy) and the compositor's single render thread touch GL; the ``/live`` route
just re-serves ``latest_jpeg``. The LIBERO ``viz_option`` (collision-geom / site
markers hidden) is applied by the compositor, so the arm renders clean.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("mujoco_gs.groot_libero")

# Suite whose checkpoint the bundled GR00T container serves. Override if your
# server hosts a different `libero_<suite>/` sub-checkpoint.
DEFAULT_SUITE = "libero_10"


class GrootLiberoRunner:
    """Agentic driver for real-GR00T-on-LIBERO episodes used by the UI."""

    def __init__(
        self,
        suite: str = DEFAULT_SUITE,
        host: str = "127.0.0.1",
        port: int = 8000,
        data_config: str = "libero_panda",
        groot_version: str = "n1.7",
        render_width: int = 512,
        render_height: int = 384,
        model_id: str | None = None,
    ) -> None:
        self.suite = suite
        self.host = host
        self.port = port
        self.data_config = data_config
        self.groot_version = groot_version
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.model_id = model_id

        self._adapters: dict | None = None
        self._task_label_to_name: dict = {}
        # Latest composited JPEG, served by the app's /live MJPEG route.
        self.latest_jpeg: bytes | None = None

    # ----- task discovery ----- #

    def available_tasks(self) -> list[str]:
        self._ensure_suite_loaded()
        return list(self._task_label_to_name.keys())

    def _ensure_suite_loaded(self) -> None:
        if self._adapters is not None:
            return
        from strands_robots.benchmarks.libero import load_libero_suite

        logger.info("Loading LIBERO suite %r…", self.suite)
        # NOTE: no max_steps cap — LIBERO-Long needs the full default budget.
        self._adapters = load_libero_suite(self.suite)
        for full_name in self._adapters:
            label = full_name.split("-", 2)[-1].replace("_", " ")
            self._task_label_to_name[label] = full_name
        logger.info("Loaded %d LIBERO tasks.", len(self._task_label_to_name))

    def default_task_label(self) -> str | None:
        """A task likely to succeed with the default (libero_10) checkpoint."""
        labels = self.available_tasks()
        if not labels:
            return None
        # Prefer the verified white-mug task if present.
        for lab in labels:
            if "white mug" in lab and "left plate" in lab:
                return lab
        return labels[0]

    # ----- run one episode, agentically ----- #

    def run(self, task_label: str, n_episodes: int = 3, seed: int = 42, on_progress=None) -> dict:
        """Agentically run ``n_episodes`` of ``task_label`` under GR00T.

        A Strands ``Agent`` invokes a one-shot ``run_libero_eval`` tool that
        wraps ``evaluate_benchmark`` with a **synchronous** ``on_frame`` —
        each step renders the composite through ``HybridCompositor`` into the
        live JPEG buffer (so ``/live`` streams it) and collects clip frames.
        Synchronous capture (vs a concurrent render thread) is what keeps the
        MJPEG stream responsive: a background render thread contends with the
        eval and stalls the stream (the "frozen live view" bug).

        Returns ``{task, instruction, success_rate, n_episodes, video,
        agent_summary, error?}``.
        """
        self._ensure_suite_loaded()
        task = self._task_label_to_name.get(task_label)
        if task is None:
            return {"error": f"Unknown task: {task_label!r}"}
        instruction = self._adapters[task].problem.language

        import cv2

        from examples.mujoco_gs.backgrounds import PanoramaBackground
        from examples.mujoco_gs.compositor import HybridCompositor

        try:
            from strands import Agent, tool

            from strands_robots.simulation import Simulation
            from strands_robots.simulation.benchmark import get_benchmark
        except ImportError as e:  # pragma: no cover
            return {"error": f"missing deps: {e}"}

        sim = Simulation(tool_name="libero_sim", mesh=False)
        compositor = HybridCompositor(sim, background=PanoramaBackground())
        frames: list[np.ndarray] = []
        # Throttle the live render to ~every other step so per-step capture
        # doesn't dominate wall-time, while staying smooth.
        render_every = 2

        def on_frame(step: int, obs: dict, action: dict) -> None:
            # Synchronous: runs inside the eval loop (no concurrent thread).
            if step % render_every != 0:
                return
            try:
                rgb = compositor.render(camera_name="image", width=self.render_width, height=self.render_height).rgb
                frames.append(rgb)
                ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    self.latest_jpeg = buf.tobytes()
                if on_progress:
                    on_progress(len(frames))
            except Exception as e:  # pragma: no cover
                logger.debug("on_frame render skipped: %s", e)

        captured: dict = {}
        try:
            sim.create_world()
            sim.add_robot("robot", data_config="panda")

            # Pre-warm: generate + load the LIBERO scene so its cameras/Panda
            # exist before eval.
            spec = get_benchmark(task)
            if spec.scene_path is None and getattr(spec, "_auto_generate_scene", False):
                generated = spec._generate_scene_from_bddl()
                if generated:
                    spec.scene_path = generated
            if spec.scene_path:
                sim.load_scene(spec.scene_path)
                if hasattr(spec, "prewarm"):
                    spec.prewarm(sim)
                if "robot" not in sim.list_robots():
                    sim.add_robot("robot", data_config="panda")
            compositor.clear_caches()
            # Seed the live buffer with the initial (pre-eval) scene frame.
            on_frame(0, {}, {})

            # --- agentic: a one-shot tool wrapping evaluate_benchmark + the
            #     synchronous on_frame closure (the closure can't cross the
            #     agent's JSON tool boundary, so we capture it in Python scope
            #     and expose a single tool the agent picks). ---
            host, port = self.host, self.port
            data_config, groot_version = self.data_config, self.groot_version

            @tool
            def run_libero_eval(benchmark_name: str, n_episodes: int = 3, seed: int = 42) -> dict:
                """Run a LIBERO benchmark under the GR00T policy and return the
                success_rate. Use this for the LIBERO eval.

                Args:
                    benchmark_name: the registered LIBERO task id.
                    n_episodes: number of episodes to run.
                    seed: RNG seed.
                """
                res = sim.evaluate_benchmark(
                    benchmark_name=benchmark_name,
                    policy_provider="groot",
                    policy_config={
                        "host": host,
                        "port": port,
                        "data_config": data_config,
                        "groot_version": groot_version,
                    },
                    n_episodes=n_episodes,
                    seed=seed,
                    instruction=instruction,
                    on_frame=on_frame,
                )
                payload = next((c["json"] for c in res["content"] if "json" in c), {}) if isinstance(res, dict) else {}
                captured["success_rate"] = payload.get("success_rate")
                return res

            agent_kwargs = {"tools": [run_libero_eval]}
            if self.model_id:
                agent_kwargs["model"] = self.model_id
            agent = Agent(**agent_kwargs)
            prompt = (
                f"Make exactly one tool call: invoke `run_libero_eval` with "
                f"`benchmark_name='{task}'`, `n_episodes={n_episodes}`, "
                f"`seed={seed}`. When it returns, report the `success_rate` from "
                f"the JSON payload as a percentage of the {n_episodes} episodes, "
                f"plus a one-sentence description of the task: '{instruction}'."
            )
            agent_result = agent(prompt)
            agent_summary = _extract_text(agent_result)
            success_rate = captured.get("success_rate")
            if success_rate is None:
                success_rate = _extract_success_rate(agent)
        except Exception as e:  # pragma: no cover
            logger.exception("Agentic GR00T run failed.")
            return {"error": f"{type(e).__name__}: {e}", "task": task, "instruction": instruction}
        finally:
            # Always release the MuJoCo model + the compositor's render-executor
            # thread and cached renderers, even when eval raises (GR00T server
            # unreachable, BDDL gen fails, EGL context error) — otherwise stale
            # EGL contexts + MJ models leak across runs of the Gradio app.
            try:
                compositor.close()
            except Exception:  # pragma: no cover
                pass
            try:
                sim.destroy()
            except Exception:  # pragma: no cover
                pass

        video = _encode_mp4(frames) if frames else None

        return {
            "task": task,
            "instruction": instruction,
            "success_rate": success_rate,
            "n_episodes": n_episodes,
            "video": video,
            "agent_summary": agent_summary,
            "n_frames": len(frames),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _extract_text(result) -> str:
    if result is None:
        return ""
    msg = getattr(result, "message", None)
    if msg is not None:
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
            if texts:
                return "\n".join(t for t in texts if t)
    return str(result)


def _extract_success_rate(agent) -> float | None:
    """Find the evaluate_benchmark toolResult's ``success_rate`` in the
    agent's message history."""
    try:
        for msg in agent.messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or "toolResult" not in block:
                    continue
                for c in block["toolResult"].get("content", []):
                    if isinstance(c, dict) and "json" in c:
                        payload = c["json"]
                        if isinstance(payload, dict) and "success_rate" in payload:
                            return float(payload["success_rate"])
    except Exception:  # pragma: no cover
        pass
    return None


def _encode_mp4(frames: list[np.ndarray], fps: int = 20) -> str | None:
    import os
    import tempfile

    from strands_robots.rendering import encode_clip  # shared library encoder (issue #1537)

    if not frames:
        return None
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix="groot_libero_")
    os.close(fd)
    try:
        encode_clip(frames, path, fps=fps, quality=7, macro_block_size=8)
    except ImportError:  # pragma: no cover - imageio missing: video is best-effort here
        os.unlink(path)
        return None
    return path
