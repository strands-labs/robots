# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Gradio app: a real GR00T policy driving a LIBERO Panda task.

This is a **separate** demo from the SO-101 hybrid-render app (``app.py``).
Where ``app.py`` drives a small arm with a scripted/agent motion against a
3DGS/panorama backdrop, this app hands control to a **real NVIDIA GR00T
vision-language-action policy** (served over ZMQ) driving a **Franka Panda**
through a **LIBERO** manipulation task, and shows it two ways:

* a near-real-time **MJPEG live view** (a single long-lived HTTP stream that an
  ``<img>`` renders incrementally — proxy-friendly, unlike Gradio's buffered
  SSE queue), and
* a recorded **MP4 clip** of the episode.

It reuses :class:`examples.mujoco_gs.groot_libero.GrootLiberoRunner` and the
:class:`examples.mujoco_gs.compositor.HybridCompositor`.

Prerequisites (same as ``libero_groot.py``): a GR00T server reachable over
ZMQ (``--groot-port``, default 8000), plus ``libero`` + ``robosuite`` and
``strands-robots[sim-mujoco]``.

Run:
    python -m examples.mujoco_gs.app_groot_libero --groot-port 8000

Then open http://127.0.0.1:7861 (this app defaults to port 7861 so it can run
alongside the SO-101 app on 7860).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

# Make the example importable both as a module and as a script.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent.parent))

from examples.mujoco_gs.groot_libero import GrootLiberoRunner  # noqa: E402

logger = logging.getLogger("mujoco_gs.app_groot_libero")

# Live MJPEG view geometry (matches the runner's render size aspect 4:3).
LIVE_W, LIVE_H, LIVE_FPS = 512, 384, 15


def _live_img_html() -> str:
    """A visible, fixed-aspect panel whose <img> streams the GR00T live view.

    ``gr.HTML`` injects markup via innerHTML (no inline <script>), so the
    stream URL is set directly on the root-absolute ``/live`` route.
    """
    bust = int(time.time() * 1000)
    return f"""
<div style="border:2px solid #8250df; border-radius:8px; padding:6px; background:#0b0b0b;
            max-width:{LIVE_W + 16}px; margin:0 auto;">
  <div style="color:#c9b8f2; font-size:13px; margin-bottom:4px;">
    [GR00T] GR00T live view — Franka Panda / LIBERO (near real-time MJPEG)
  </div>
  <img src="/live?t={bust}"
       style="width:100%; aspect-ratio:{LIVE_W} / {LIVE_H}; height:auto; display:block;
              object-fit:contain; background:#000; border-radius:6px;"
       alt="connecting to GR00T live stream…"
       onerror="this.alt='live stream did not load (connection may be buffering) - use the clip below';" />
</div>
"""


def _mjpeg_buffer(runner: GrootLiberoRunner):
    """Stream the runner's latest JPEG buffer as MJPEG (no rendering here).

    The runner's ``on_frame`` produces frames on its own (eval + compositor)
    threads; this route only re-serves the latest buffered JPEG, so it never
    adds a GL-using thread.
    """
    import cv2
    import numpy as np

    frame_dt = 1.0 / float(LIVE_FPS)
    blank = None
    try:
        while True:
            jpg = runner.latest_jpeg
            if jpg is None:
                if blank is None:
                    img = np.zeros((LIVE_H, LIVE_W, 3), np.uint8)
                    cv2.putText(
                        img,
                        "GR00T idle - pick a task and press Run",
                        (20, LIVE_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 200, 200),
                        1,
                    )
                    ok, buf = cv2.imencode(".jpg", img)
                    blank = buf.tobytes() if ok else b""
                payload = blank
            else:
                payload = jpg
            if payload:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
            time.sleep(frame_dt)
    except GeneratorExit:
        return


def build_app(groot_host: str = "127.0.0.1", groot_port: int = 8000):
    try:
        import gradio as gr
    except ImportError as e:  # pragma: no cover
        raise ImportError("gradio is required for the UI. Run `pip install gradio`.") from e

    runner = GrootLiberoRunner(host=groot_host, port=groot_port)

    def on_run(task_label: str):
        """Agentically run a GR00T episode; stream status + final clip. The
        live view streams independently via the /live MJPEG route."""
        if not task_label:
            yield "Pick a task first.", gr.update()
            return
        result: dict = {}

        def _run():
            try:
                result["out"] = runner.run(task_label)
            except Exception as e:  # pragma: no cover
                result["out"] = {"error": f"{type(e).__name__}: {e}"}

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        secs = 0
        while th.is_alive():
            yield f"Agent is running the GR00T eval… ({secs}s) — watch the live view.", gr.update()
            time.sleep(1.0)
            secs += 1
        th.join()

        out = result.get("out", {})
        if out.get("error"):
            yield f"Warning: {out['error']}", (out.get("video") or gr.update())
            return
        sr = out.get("success_rate")
        sr_str = (
            f"{sr * 100:.0f}% ({int(round(sr * out.get('n_episodes', 0)))}/{out.get('n_episodes')})"
            if sr is not None
            else "?"
        )
        status = (
            f"**Task:** {out.get('instruction', '?')}\n\n"
            f"**Success rate:** {sr_str} | frames: {out.get('n_frames', '?')}\n\n"
            f"**Agent:** {out.get('agent_summary', '')[:300]}"
        )
        yield status, (out.get("video") or gr.update())

    title = "GR00T + LIBERO — real VLA policy on a Franka Panda"
    with gr.Blocks(title=title) as demo:
        gr.Markdown(
            f"# {title}\n"
            "A **real NVIDIA GR00T** vision-language-action policy (served over ZMQ) drives a "
            "Franka **Panda** through a **LIBERO** manipulation task. The arm streams to the "
            "live view in near real-time (MJPEG); a clip is recorded below.\n\n"
            "This is the *real-policy* companion to the SO-101 hybrid-render demo (`app.py`)."
        )
        with gr.Row():
            with gr.Column(scale=6):
                live_view = gr.HTML(_live_img_html())
            with gr.Column(scale=4):
                try:
                    tasks = runner.available_tasks()
                    default_task = runner.default_task_label()
                except Exception as e:  # pragma: no cover
                    logger.warning("Could not load LIBERO tasks: %s", e)
                    tasks, default_task = [], None
                task_dd = gr.Dropdown(
                    choices=tasks,
                    value=default_task,
                    label="LIBERO task",
                )
                run_btn = gr.Button("Run GR00T policy", variant="primary")
                refresh_btn = gr.Button("Reconnect live view")
                status = gr.Markdown("")
                clip = gr.Video(label="Episode clip (autoplays)", autoplay=True, interactive=False, height=384)

        run_btn.click(on_run, inputs=[task_dd], outputs=[status, clip])
        # Re-point the <img> (fresh cache-buster) if the stream needs a kick.
        refresh_btn.click(_live_img_html, outputs=[live_view])

    return demo, runner


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--groot-host", type=str, default="127.0.0.1", help="GR00T ZMQ host.")
    p.add_argument("--groot-port", type=int, default=8000, help="GR00T ZMQ port.")
    p.add_argument("--server-name", type=str, default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=7861, help="UI port (default 7861, alongside app.py on 7860).")
    p.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.environ.setdefault("MUJOCO_GL", "egl")

    demo, runner = build_app(groot_host=args.groot_host, groot_port=args.groot_port)
    demo.queue(default_concurrency_limit=2)
    launch_kwargs: dict = {
        "server_name": args.server_name,
        "server_port": args.server_port,
        "share": args.share,
        "prevent_thread_lock": True,
    }
    try:
        import gradio as gr

        if hasattr(gr, "themes"):
            launch_kwargs["theme"] = gr.themes.Soft()
    except Exception:  # pragma: no cover
        pass
    demo.launch(**launch_kwargs)

    # Mount the MJPEG live-stream route (serves the runner's JPEG buffer).
    from fastapi.responses import StreamingResponse

    def _live_route():
        return StreamingResponse(
            _mjpeg_buffer(runner),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
        )

    demo.app.add_api_route("/live", _live_route, methods=["GET"])
    logger.info("GR00T MJPEG live stream mounted at /live")

    try:
        demo.block_thread()
    except (KeyboardInterrupt, AttributeError):  # pragma: no cover
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
