#!/usr/bin/env python3
"""Gradio web app for the Isaac Sim + 3DGS hybrid-render demo.

Browser-accessible companion to the ``render_demo`` CLI -- the Isaac
analogue of ``examples/mujoco_gs/app.py``: a hands-free **live MJPEG
stream** of the composite plus on-demand full-res stills.

    +------------------------------------+ +---------------------------+
    |  * Live view (Isaac RTX + 3DGS)    | |  Controls                 |
    |   <streamed Franka on backdrop>    | |  [Camera v] oblique       |
    +------------------------------------+ |  [Background v] tabletop  |
    |  Composite still (full-res)        | |  [Upload .ply ]           |
    |   <rendered Franka on backdrop>    | |  [Render still] [Wave]    |
    +------------------------------------+ |  status: ...              |
                                           +---------------------------+

Live vs still: Isaac's RTX renderer isn't real-time-cheap like MuJoCo's
offscreen path, and the ``SimulationApp`` boot is heavyweight (~200 s).
So the app boots the sim once at startup; the ``/live`` MJPEG route then
re-renders continuously (a few fps) for a hands-free view, while the
buttons grab a full-res still on demand. Both go through the same
main-thread render queue.

Threading model -- the important bit. Isaac's ``SimulationApp`` must be
created on the **main thread** (it installs SIGINT handlers, and
``signal.signal`` only works on the main thread), and its RTX render
context is thread-affine. But Gradio serves callbacks on worker
threads. So we invert control flow:

* The **main thread** owns Isaac: :meth:`IsaacGsApp.boot` creates
  ``SimulationApp`` + builds the scene, then :meth:`serve_forever`
  drains a render-request queue (executing each render on the main
  thread).
* **Gradio** is launched non-blocking (``prevent_thread_lock=True``) so
  it serves in background threads; its callbacks enqueue a
  :class:`_RenderRequest` and block on its event, keeping the handler
  synchronous while the render runs on the main thread.

Run::

    python -m examples.isaac_gs.app --server-port 7862
    # open http://127.0.0.1:7862  (7860/7861 are the mujoco_gs apps)

Needs a working Isaac Sim (RTX GPU) + the Phase-2 wiring (#61 / #62 /
#63). The render handler is importable (``boot`` + ``render_once``) so
it can be exercised headlessly for validation without a browser.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent.parent))

logger = logging.getLogger("isaac_gs.app")

# Background dropdown sentinels (non-scene choices).
PANORAMA_CHOICE = "procedural panorama"
UPLOAD_CHOICE = "uploaded .ply"

# Live MJPEG stream settings (the Isaac analogue of mujoco_gs's live view).
# Isaac's RTX render isn't real-time, so the effective rate is whatever the
# render achieves (a few fps); LIVE_FPS just caps the busy-loop. The stream is
# downscaled to keep a remote `gradio.live` share tunnel smooth.
LIVE_W, LIVE_H, LIVE_FPS = 480, 360, 10


def _live_img_html(camera: str) -> str:
    """A visible, labeled panel with an ``<img>`` pulling the MJPEG live stream.

    Mirrors ``examples.mujoco_gs.app``: the browser renders
    ``multipart/x-mixed-replace`` JPEG frames as they arrive (one long-lived
    HTTP response), which is far more proxy/share-tunnel friendly than Gradio's
    buffered SSE queue. The ``<img>`` is wrapped in a fixed-aspect box so the
    panel is visible before the first frame, with an ``onerror`` fallback.
    """
    bust = int(time.time() * 1000)
    return f"""
<div style="border:2px solid #4a90d9; border-radius:8px; padding:6px; background:#0b0b0b;
            max-width:{LIVE_W + 16}px; margin:0 auto;">
  <div style="color:#9cc; font-size:13px; margin-bottom:4px;">
    <span style="color:#e33;">&#9679;</span> Live view (Isaac RTX + 3DGS, near real-time)
    &mdash; follows the camera selector / agent
  </div>
  <img src="/live?camera={camera}&t={bust}"
       style="width:100%; aspect-ratio:{LIVE_W} / {LIVE_H}; height:auto; display:block;
              object-fit:contain; background:#000; border-radius:6px;"
       alt="connecting to live stream…"
       onerror="this.alt='live stream did not load (Isaac may still be booting) — use Render below';" />
</div>
"""


def _mjpeg_frames(app: "IsaacGsApp", camera: str):
    """Yield ``multipart/x-mixed-replace`` JPEG chunks of the live composite.

    Byte framing + pacing come from the shared library generator
    (``strands_robots.rendering.mjpeg_frames``, issue #1537); this wrapper
    supplies the demo's frame source: each frame is rendered on Isaac's
    **main thread** via ``app.render_once`` (the same render queue the
    buttons use), keeping the RTX context thread-affine. We only drive
    renders once the main-thread serve loop is live -- rendering Isaac off
    the main thread (pre-serve) is unsafe -- and block-render one frame at a
    time, so the queue never backs up. The stream follows
    ``app.current_camera`` (driven by the dropdown / agent) so it never
    needs to reconnect when the view changes -- it's one persistent stream
    that is never a Gradio event output, hence never greyed out during
    processing.
    """
    from strands_robots.rendering import mjpeg_frames

    def _current_frame():
        if not getattr(app, "_serving", False):
            return None  # Isaac still booting; the generator idles
        rgb, _status = app.render_once(camera=app.current_camera)
        return np.asarray(rgb)

    return mjpeg_frames(_current_frame, fps=LIVE_FPS, quality=75, size=(LIVE_W, LIVE_H))


class _RenderRequest:
    """A job marshalled from a Gradio worker to Isaac's main thread.

    Usually a render (``camera`` + ``wave``). If ``op`` is set, the serve
    loop runs that callable on the main thread instead and stores its return
    value in ``result`` -- this is how stage mutations that MUST run on the
    main thread (e.g. ``add_object``) are marshalled, mirroring renders.
    """

    __slots__ = ("camera", "wave", "op", "done", "result", "error")

    def __init__(self, camera: str = "oblique", wave: bool = False, op=None) -> None:
        self.camera = camera
        self.wave = wave
        self.op = op
        self.done = threading.Event()
        self.result: "object | None" = None
        self.error: "Exception | None" = None


class IsaacGsApp:
    """Owns the persistent Isaac sim + compositor behind the Gradio UI.

    See the module docstring for the (important) threading model: Isaac
    on the main thread, Gradio in background threads, renders marshalled
    to the main thread via a queue.
    """

    def __init__(
        self,
        default_camera: str = "oblique",
        panorama_path: Optional[str] = None,
        gsplat_ply: Optional[str] = None,
        gsplat_scene: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        robot_usd: Optional[str] = None,
        camera_presets: Optional[dict] = None,
    ) -> None:
        self.default_camera = default_camera
        self.panorama_path = panorama_path
        self.gsplat_ply = gsplat_ply
        self.gsplat_scene = gsplat_scene
        self.width = int(width)
        self.height = int(height)
        # Robot: default None -> build_default_scene loads the bundled Franka.
        # Pass an SO-101 (or other) USD + matching camera presets to swap it.
        self.robot_usd = robot_usd
        self.camera_presets = camera_presets

        self._queue: "queue.Queue[_RenderRequest]" = queue.Queue()
        self._bg_lock = threading.Lock()
        self._pending_bg: "dict | None" = None
        self._sim = None
        self._compositor = None
        self._build = None
        self._cameras: list[str] = []
        self._stop = threading.Event()
        self._serving = False
        # The "current" camera the agent / chat drive (the live view + agent
        # renders follow it). Buttons still pass their own camera explicitly.
        self._ui_camera = default_camera
        self._spawn_count = 0
        self._rest_joints = None

    @property
    def current_camera(self) -> str:
        """The camera the agent/chat currently target."""
        cam = self._ui_camera
        if cam in self._cameras or not self._cameras:
            return cam
        return self._cameras[0]

    def set_camera(self, view: str) -> None:
        """Set the current agent/chat camera (validated against presets)."""
        self._ui_camera = view

    # --- main-thread lifecycle ------------------------------------------

    def boot(self) -> None:
        """Create SimulationApp + build the scene. **Main thread only.**"""
        from strands_robots.simulation import create_simulation

        from strands_robots.simulation.isaac import IsaacSimulation

        available, reason = IsaacSimulation.is_available()
        if not available:
            raise RuntimeError(f"Isaac Sim not available: {reason}")

        from examples.isaac_gs.compositor import IsaacHybridCompositor
        from examples.isaac_gs.scene import add_preset_cameras, build_default_scene

        logger.info("Booting IsaacSimulation on the main thread (~200 s)...")
        sim = create_simulation("isaac", headless=True, num_envs=1, render_mode="rtx_realtime")
        # Robot-aware: build_default_scene creates the "front" camera, so align
        # it with the chosen presets' "front" pose; add_preset_cameras adds the
        # rest. Defaults (no robot_usd / presets) = the bundled Franka.
        presets = self.camera_presets
        front = (presets or {}).get("front")
        bd_kwargs: dict = {"camera_width": self.width, "camera_height": self.height}
        if self.robot_usd:
            bd_kwargs["robot_usd"] = self.robot_usd
        if front:
            bd_kwargs["camera_position"] = list(front[0])
            bd_kwargs["camera_target"] = list(front[1])
            if len(front) > 2:
                bd_kwargs["camera_fov"] = float(front[2])
        self._build = build_default_scene(sim, **bd_kwargs)
        self._cameras = add_preset_cameras(sim, width=self.width, height=self.height, presets=presets)
        self._compositor = IsaacHybridCompositor(sim, background=self._make_background())
        self._sim = sim
        self._warmup_cameras()
        logger.info("Scene ready: cameras=%s robot=%s", self._cameras, self._build.robot_name)

    def _warmup_cameras(self, steps: int = 30) -> None:
        """Prime each camera's RTX render product before first real render.

        Cameras added after the initial ``sim.step`` (the preset cameras)
        haven't produced a frame yet, so ``get_rgba()`` can come back
        malformed (empty / 1-D) until the render product is triggered.
        Stepping the world + a guarded throwaway render per camera warms
        them so the first user render returns a well-formed frame.
        """
        if self._sim is None:
            return
        self._sim.step(steps)
        for cam in self._cameras:
            for _ in range(3):
                try:
                    self._sim.render(camera_name=cam)
                except Exception:  # noqa: BLE001 - warmup is best-effort
                    pass
                self._sim.step(2)

    def serve_forever(self, poll: float = 0.05) -> None:
        """Main-thread loop: drain render requests. Runs after :meth:`boot`."""
        self._serving = True
        try:
            while not self._stop.is_set():
                try:
                    req = self._queue.get(timeout=poll)
                except queue.Empty:
                    continue
                try:
                    self._apply_pending_background()
                    if req.op is not None:
                        req.result = req.op()
                    else:
                        req.result = self._render_on_main(req.camera, req.wave)
                except Exception as exc:  # noqa: BLE001 - surfaced to caller
                    req.error = exc
                finally:
                    req.done.set()
        finally:
            self._serving = False

    def stop(self) -> None:
        self._stop.set()

    def _render_on_main(self, camera: str, wave: bool) -> "tuple[np.ndarray, str]":
        camera = camera if camera in self._cameras else (self._cameras[0] if self._cameras else "front")
        if wave and self._build and self._build.robot_joint_count > 0:
            self._wave_arm()
        frame = self._compositor.render(camera_name=camera)
        fg_px = int(frame.foreground_mask.sum())
        status = f"camera={camera} foreground_px={fg_px} size={frame.rgb.shape[1]}x{frame.rgb.shape[0]}"
        return frame.rgb, status

    def _wave_arm(self) -> None:
        """Move a joint so the arm visibly "waves" while staying fully in frame.

        The robot USDs load without actuator drive gains that track position
        *targets* (``send_action``), so for a reliable visual we set joint
        positions directly (kinematic) via the articulation, oscillating ONE
        joint around the captured rest pose:

        * SO-101: the ELBOW. A base pan rotates the whole arm toward the camera
          and foreshortens it -- in the tight front view that clips the arm to a
          partial sliver (the reported bug). The elbow swings the forearm
          *within* the broadside plane, so the arm stays fully framed in every
          preset while still clearly moving.
        * Franka (bundled fallback): the base joint, as before.

        Falls back to ``send_action`` if the articulation handle isn't reachable.
        """
        import math

        name = self._build.robot_name
        # SO-101 (custom presets) waves the elbow; the bundled Franka waves base.
        wave_joint = 2 if self.camera_presets else 0
        amp = 0.22 if self.camera_presets else 0.3
        delta = amp * math.sin(time.time())
        robot = getattr(self._sim, "_robots", {}).get(name)
        art = getattr(robot, "articulation", None) if robot is not None else None
        if art is not None:
            try:
                cur = art.get_joint_positions()
                arr = np.asarray(cur.cpu().numpy() if hasattr(cur, "cpu") else cur, dtype=float).copy()
                # Capture the rest pose once and always oscillate around it (the
                # live view doesn't step physics back to home between waves, so
                # reading "current" would let the pose drift).
                if self._rest_joints is None or len(self._rest_joints) != len(arr):
                    self._rest_joints = arr.copy()
                j = wave_joint if wave_joint < len(arr) else 0
                arr[j] = float(self._rest_joints[j]) + delta
                art.set_joint_positions(arr)
                self._sim.step(2)
                return
            except Exception:  # noqa: BLE001 - fall back to the action API
                pass
        jn = self._sim.robot_joint_names(name)
        if jn:
            j = wave_joint if wave_joint < len(jn) else 0
            base = float(self._rest_joints[j]) if self._rest_joints is not None and j < len(self._rest_joints) else 0.0
            self._sim.send_action({jn[j]: base + delta}, robot_name=name)
            self._sim.step(5)

    def _apply_pending_background(self) -> None:
        with self._bg_lock:
            pending = self._pending_bg
            self._pending_bg = None
        if pending is None:
            return
        # pending is a dict of resolve_background kwargs.
        self.gsplat_ply = pending.get("gsplat_ply")
        self.gsplat_scene = pending.get("gsplat_scene")
        self.panorama_path = pending.get("panorama")
        self._prefer_gs = pending.get("prefer_gs", True)
        if self._compositor is not None:
            self._compositor.background = self._make_background()
            self._compositor._bg_cache.clear()

    def _make_background(self):
        from examples.isaac_gs.background import resolve_background

        return resolve_background(
            gsplat_ply=self.gsplat_ply,
            gsplat_scene=self.gsplat_scene,
            panorama=self.panorama_path,
            prefer_gs=getattr(self, "_prefer_gs", True),
        )

    # --- public handlers (called from Gradio worker threads) ------------

    def render_once(
        self, camera: str = "oblique", wave: bool = False, timeout: float = 600.0
    ) -> "tuple[np.ndarray, str]":
        """Enqueue a render for the main thread and block for the result.

        When no :meth:`serve_forever` loop is active (e.g. a headless
        test that called :meth:`boot` on the main thread itself), renders
        inline on the caller's thread instead.
        """
        if not self._serving:
            self._apply_pending_background()
            return self._render_on_main(camera, wave)
        req = _RenderRequest(camera, wave)
        self._queue.put(req)
        if not req.done.wait(timeout=timeout):
            raise TimeoutError(f"render timed out after {timeout}s")
        if req.error is not None:
            raise req.error
        assert req.result is not None
        return req.result

    def _run_on_main(self, op, timeout: float = 120.0):
        """Run ``op`` on Isaac's main thread (or inline if not serving).

        Stage mutations (e.g. ``add_object``) MUST run on the thread that owns
        the RTX/USD context, so we marshal them through the same queue the
        renders use rather than touching the stage from a Gradio worker.
        """
        if not self._serving:
            self._apply_pending_background()
            return op()
        req = _RenderRequest(op=op)
        self._queue.put(req)
        if not req.done.wait(timeout=timeout):
            raise TimeoutError(f"main-thread op timed out after {timeout}s")
        if req.error is not None:
            raise req.error
        return req.result

    def spawn_cube(self, color: str = "red", position: "list[float] | None" = None, size: float = 0.04) -> str:
        """Add a small **static** colored cube to the live scene.

        Marshalled onto the main thread via :meth:`_run_on_main`. The cube is
        static because the digital-twin composite has no ground plane (the 3DGS
        room is the visible floor) -- a dynamic cube would fall through it.
        ``position`` defaults to the current camera's look-at point so the cube
        lands in frame; ``color`` is a common name.
        """
        if self._sim is None:
            return "Scene isn't ready yet."
        colors = {
            "red": [0.85, 0.12, 0.12],
            "green": [0.12, 0.70, 0.20],
            "blue": [0.15, 0.35, 0.90],
            "yellow": [0.95, 0.85, 0.10],
            "orange": [0.95, 0.50, 0.10],
            "purple": [0.60, 0.20, 0.80],
            "white": [0.90, 0.90, 0.90],
            "black": [0.08, 0.08, 0.08],
        }
        cname = (color or "red").strip().lower()
        rgb = colors.get(cname)
        if rgb is None:
            cname, rgb = "red", colors["red"]
        self._spawn_count += 1
        name = f"cube_{self._spawn_count}"
        pos = [float(p) for p in position] if position is not None else self._default_object_pos()

        def _op():
            r = self._sim.add_object(
                name=name,
                shape="box",
                position=pos,
                size=[size, size, size],
                color=rgb,
                mass=0.1,
                is_static=True,
            )
            # The live render path only READS each camera's RTX render
            # product; it's refreshed when the sim steps. A freshly-created
            # USD prim also needs an update tick to flush onto the render
            # stage, so a single step renders the stage *before* the cube is
            # realized. Step a few times so the new prim is flushed AND
            # accumulated into every camera's product before the next read.
            self._sim.step(6)
            return r

        r = self._run_on_main(_op)
        if not (isinstance(r, dict) and r.get("status") == "success"):
            text = r
            if isinstance(r, dict):
                try:
                    text = r.get("content", [{}])[0].get("text", "") or r
                except Exception:  # noqa: BLE001
                    text = r
            return f"Couldn't add the cube: {text}"
        return f"Added a {cname} cube ({name}) at {[round(p, 2) for p in pos]}."

    def _default_object_pos(self) -> "list[float]":
        """A validated, in-frame spawn point on the workspace surface.

        The 3DGS "counter" the arm sits on is at z~=0.09 (not the world floor),
        so cubes need that height to rest on it rather than sink onto the
        cabinet faces below. SO-101 gets a spot front-right of the arm (clear of
        it, visible from every preset); Franka mirrors its startup cube. A small
        per-spawn x-jitter keeps repeated cubes from stacking exactly.
        """
        base = [0.20, -0.20, 0.10] if self.camera_presets else [0.40, 0.0, 0.40]
        # Cycle x over {0.15, 0.20, 0.25} (1-based spawn count) so repeated
        # cubes sit side-by-side on the counter instead of stacking.
        jitter = 0.05 * ((self._spawn_count - 1) % 3 - 1)
        return [base[0] + jitter, base[1], base[2]]

    def set_background(self, choice: str, ply_upload: Optional[str] = None) -> str:
        """Queue a background swap; applied on the main thread before the next render.

        ``choice`` is a UI dropdown value: a 3DGS preset scene name, the
        ``PANORAMA_CHOICE`` sentinel, or the ``UPLOAD_CHOICE`` sentinel
        (in which case ``ply_upload`` is the uploaded ``.ply`` path).
        """
        if ply_upload and choice == UPLOAD_CHOICE:
            kwargs = {"gsplat_ply": ply_upload}
            msg = f"background queued: uploaded 3DGS {ply_upload}"
        elif choice == PANORAMA_CHOICE:
            kwargs = {"prefer_gs": False}
            msg = "background queued: procedural panorama"
        else:
            kwargs = {"gsplat_scene": choice}
            msg = f"background queued: 3DGS scene {choice!r}"
        with self._bg_lock:
            self._pending_bg = kwargs
        return msg + " — applies on next render"

    def shutdown(self) -> None:
        self.stop()
        if self._sim is not None:
            try:
                self._sim.destroy()
            except Exception:  # noqa: BLE001
                pass


def build_ui(app: IsaacGsApp, agent: "object | None" = None, robot_label: str = "robot arm"):
    """Construct the Gradio Blocks UI bound to an :class:`IsaacGsApp`.

    If ``agent`` (a Strands ``Agent``) is provided, a chat panel drives the
    scene by natural language; otherwise the chat panel is shown disabled and
    the buttons still work. ``robot_label`` names the loaded arm (e.g.
    "SO-101 arm" / "Franka arm") so the header matches what's on screen.
    """
    import gradio as gr

    from examples.isaac_gs.scene import CAMERA_PRESETS

    # Background choices: the curated 3DGS skybox presets + procedural
    # panorama + an uploaded-.ply sentinel.
    try:
        from strands_robots.rendering import gsplat_skybox_scene_names

        gs_scenes = gsplat_skybox_scene_names()
    except Exception:  # noqa: BLE001
        gs_scenes = []
    bg_choices = gs_scenes + [PANORAMA_CHOICE, UPLOAD_CHOICE]
    # Default the dropdown to the configured default GS scene if it's a
    # known skybox preset, else the first available scene, else panorama.
    from examples.isaac_gs.background import DEFAULT_GS_SCENE

    default_bg = DEFAULT_GS_SCENE if DEFAULT_GS_SCENE in gs_scenes else (gs_scenes[0] if gs_scenes else PANORAMA_CHOICE)

    with gr.Blocks(title="Isaac Sim + 3DGS hybrid render") as demo:
        gr.Markdown(
            "# Isaac Sim + 3DGS hybrid render\n"
            f"An RTX-rendered **simulated {robot_label}** composited into a real captured "
            "**3D Gaussian Splatting** room (the digital-twin use case). A **live "
            "MJPEG view** streams the composite hands-free; the buttons grab a "
            "full-res still. Type commands to the **agent** on the right (e.g. "
            "*“switch to topdown and wave”*, *“use the bonsai background”*)."
        )
        initial_camera = app.default_camera if app.default_camera in CAMERA_PRESETS else "oblique"
        with gr.Row():
            with gr.Column(scale=3):
                # Live MJPEG view (hands-free) + a full-res still on demand.
                # Not bound to a variable: it's never a Gradio event in/output
                # (see camera_dd.change below), so the <img> never reconnects.
                gr.HTML(_live_img_html(initial_camera))
                preview = gr.Image(label="Composite still (Isaac RTX + 3DGS)", height=420)
                with gr.Row():
                    camera_dd = gr.Dropdown(
                        choices=list(CAMERA_PRESETS.keys()), value=initial_camera, label="Camera", scale=2
                    )
                    render_btn = gr.Button("Render still", variant="primary")
                    wave_btn = gr.Button("Wave + render")
                with gr.Accordion("Background", open=False):
                    bg_dd = gr.Dropdown(choices=bg_choices, value=default_bg, label="Background")
                    ply_upload = gr.File(label="3DGS .ply upload (with 'uploaded .ply')", file_types=[".ply"])
                    apply_bg_btn = gr.Button("Apply background")
                status = gr.Textbox(label="status", interactive=False)
            with gr.Column(scale=2):
                chat_label = "Agent" if agent is not None else "Agent (disabled — no LLM backend)"
                # Gradio's Chatbot message format: newer versions (>=5) use the
                # OpenAI-style {"role","content"} list and dropped the ``type``
                # kwarg; older ones need type="messages". Feature-test so the UI
                # builds on either.
                try:
                    chatbot = gr.Chatbot(label=chat_label, type="messages", height=480)
                except TypeError:
                    chatbot = gr.Chatbot(label=chat_label, height=480)
                msg_box = gr.Textbox(
                    label="Message",
                    placeholder="e.g. 'switch to topdown and wave' / 'use the bonsai background'",
                    lines=2,
                )
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")

        def on_render(cam):
            app.set_camera(cam)
            return app.render_once(camera=cam, wave=False)

        def on_wave(cam):
            app.set_camera(cam)
            return app.render_once(camera=cam, wave=True)

        def on_apply_bg(choice, ply_file):
            ply_path = ply_file.name if ply_file is not None else None
            return app.set_background(choice=choice, ply_upload=ply_path)

        def on_chat(message, history):
            history = list(history or [])
            if not message or not message.strip():
                return "", history, gr.update(), gr.update()
            history.append({"role": "user", "content": message})
            if agent is None:
                history.append({"role": "assistant", "content": "Agent chat is disabled (no LLM backend configured)."})
                return "", history, gr.update(), gr.update()
            from examples.isaac_gs.agent import extract_text

            try:
                reply = extract_text(agent(message)) or "(done)"
            except Exception as exc:  # noqa: BLE001
                reply = f"agent error: {type(exc).__name__}: {exc}"
            history.append({"role": "assistant", "content": reply})
            cam = app.current_camera
            try:
                frame, _status = app.render_once(camera=cam)
            except Exception:  # noqa: BLE001
                frame = gr.update()
            return "", history, frame, cam

        render_btn.click(on_render, inputs=[camera_dd], outputs=[preview, status])
        wave_btn.click(on_wave, inputs=[camera_dd], outputs=[preview, status])
        apply_bg_btn.click(on_apply_bg, inputs=[bg_dd, ply_upload], outputs=[status])
        # The live MJPEG stream follows app.current_camera, so the dropdown just
        # updates that. We deliberately do NOT output to the live <img>: it is
        # never recreated (no reconnect) and is never a Gradio event output, so
        # it keeps streaming and never greys out while a chat turn processes.
        camera_dd.change(lambda cam: app.set_camera(cam), inputs=[camera_dd], outputs=[])
        chat_io = dict(inputs=[msg_box, chatbot], outputs=[msg_box, chatbot, preview, camera_dd])
        send_btn.click(on_chat, **chat_io)
        msg_box.submit(on_chat, **chat_io)
        clear_btn.click(lambda: [], outputs=[chatbot])

    return demo


def main(argv: "list[str] | None" = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--panorama", default=None, help="Equirectangular panorama image for the background.")
    parser.add_argument("--gsplat-ply", default=None, help="3DGS .ply for the background (needs gsplat).")
    parser.add_argument(
        "--gsplat-scene",
        default=None,
        help="Named built-in 3DGS preset (default: the tabletop scene when gsplat is installed).",
    )
    parser.add_argument("--camera", default="oblique", help="Initial camera preset (oblique / front / topdown).")
    parser.add_argument(
        "--robot",
        default="franka",
        choices=["franka", "so101"],
        help="Which arm to load (default: bundled Franka). 'so101' needs --robot-usd (an MJCF-imported SO-101 USD).",
    )
    parser.add_argument(
        "--robot-usd",
        default=None,
        help="USD path for a non-default robot (e.g. the MJCF-imported SO-101). Required for --robot so101.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7862, help="Default 7862 (7860/7861 are mujoco_gs).")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument(
        "--model",
        default=None,
        help="Strands model id for the chat agent (e.g. a Bedrock model). Default: Strands' default.",
    )
    parser.add_argument("--no-agent", action="store_true", help="Disable the chat agent (buttons-only UI).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.environ.setdefault("MUJOCO_GL", "egl")

    # Robot selection: default Franka, or an MJCF-imported SO-101 USD with its
    # own (smaller-arm) camera presets.
    robot_usd = None
    camera_presets = None
    robot_label = "Franka arm"
    if args.robot == "so101":
        from examples.isaac_gs.scene import SO101_CAMERA_PRESETS

        robot_usd = args.robot_usd
        camera_presets = SO101_CAMERA_PRESETS
        robot_label = "SO-101 arm"
        if not robot_usd:
            logger.warning("--robot so101 needs --robot-usd; falling back to the bundled Franka.")
            camera_presets = None
            robot_label = "Franka arm"

    app = IsaacGsApp(
        default_camera=args.camera,
        panorama_path=args.panorama,
        gsplat_ply=args.gsplat_ply,
        gsplat_scene=args.gsplat_scene,
        width=args.width,
        height=args.height,
        robot_usd=robot_usd,
        camera_presets=camera_presets,
    )

    # Build the natural-language agent (optional: degrades to a buttons-only UI
    # if strands-agents / an LLM backend isn't available).
    agent = None
    if not args.no_agent:
        try:
            from examples.isaac_gs.agent import build_agent

            agent = build_agent(app, model_id=args.model, robot_label=robot_label)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent unavailable (%s); running buttons-only UI.", exc)
    demo = build_ui(app, agent=agent, robot_label=robot_label)

    # MJPEG live-stream route. Streams near-real-time JPEG frames over a single
    # long-lived HTTP response (the <img> in the UI renders them incrementally),
    # bypassing Gradio's buffered SSE queue, which a port-forward / share proxy
    # coalesces into an end-of-turn burst. Each frame is rendered on Isaac's
    # main thread via the same render queue the buttons use (see _mjpeg_frames).
    from fastapi.responses import StreamingResponse

    def _live_route(camera: str = "oblique"):
        return StreamingResponse(
            _mjpeg_frames(app, camera),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
        )

    # Mount /live *before* the server accepts requests so an early hit can't
    # 404. Gradio rebuilds `demo.app` inside launch(), discarding routes added
    # beforehand, so we wrap `App.create_app` to attach the route at app-creation
    # time (keeps `demo.launch(share=...)` intact). Falls back to a post-launch
    # mount if Gradio's internals differ.
    mounted_pre_serve = False
    try:
        import gradio.routes as _gr_routes

        _orig_create_app = _gr_routes.App.create_app

        def _create_app_with_live(*a, **k):
            fastapi_app = _orig_create_app(*a, **k)
            fastapi_app.add_api_route("/live", _live_route, methods=["GET"])
            return fastapi_app

        _gr_routes.App.create_app = staticmethod(_create_app_with_live)
        mounted_pre_serve = True
    except Exception:  # noqa: BLE001 - fall back to post-launch mount
        logger.debug("Could not hook App.create_app; mounting /live post-launch.", exc_info=True)

    # Boot Isaac on the MAIN thread (SimulationApp requires it), THEN
    # launch Gradio non-blocking so it serves in background threads while
    # the main thread runs the Isaac render loop.
    app.boot()
    demo.queue(default_concurrency_limit=1)  # Gradio events serialized; /live is separate.
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        prevent_thread_lock=True,
    )
    if not mounted_pre_serve:
        demo.app.add_api_route("/live", _live_route, methods=["GET"])
    logger.info("MJPEG live stream mounted at /live")
    print(f"[isaac_gs] UI at http://{args.server_name}:{args.server_port} — Ctrl-C to stop", flush=True)
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        # Ctrl-C: fall through to finally/shutdown
        logger.info("Interrupted (Ctrl-C); shutting down.")
    finally:
        app.shutdown()
        try:
            demo.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
