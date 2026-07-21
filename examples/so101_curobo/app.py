# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gradio app for the SO-101 cuRobo synthetic-data demo (issue #67 T9).

A small UI over :class:`SO101CuroboDemo`: a camera preview, "plan & execute"
and "generate N episodes" buttons, and (optionally) a Strands chat panel. Runs
on the MuJoCo backend today; pass ``--backend isaac`` once the Isaac Sim runtime
+ backend registration (T1) are present. Degrades gracefully: missing
cuRobo/Isaac/lerobot/LLM each disable only their feature with a clear message.
"""

from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger("so101_curobo.app")

DEFAULT_TASK = "pick up the red cube and place it in the bin"


def build_ui(demo, agent: object | None = None):
    import gradio as gr

    cameras = list(getattr(demo.scene, "cameras", []) or ["front"])
    initial_cam = demo.current_camera if demo.current_camera in cameras else (cameras[0] if cameras else "front")

    with gr.Blocks(title="SO-101 cuRobo synthetic data") as ui:
        gr.Markdown(
            "# SO-101 synthetic data with cuRobo\n"
            "Plan collision-aware pick-and-place for a **simulated SO-101**, execute it, "
            "and record **LeRobot** episodes for policy training. "
            "Runs on the MuJoCo backend with a scripted planner today; flips to **Isaac Sim + "
            "cuRobo** once those runtimes are installed."
        )
        gr.Markdown(f"**Scene:** {demo.describe()}")
        with gr.Row():
            with gr.Column(scale=3):
                preview = gr.Image(label="Camera preview", height=360)
                with gr.Row():
                    cam_dd = gr.Dropdown(choices=cameras, value=initial_cam, label="Camera", scale=2)
                    refresh_btn = gr.Button("Refresh view")
                run_btn = gr.Button("Plan & execute (record 1 episode)", variant="primary")
                with gr.Row():
                    n_eps = gr.Number(value=5, precision=0, label="Episodes", minimum=1, maximum=100, scale=1)
                    gen_btn = gr.Button("Generate N episodes", variant="primary", scale=2)
                status = gr.Textbox(label="status", interactive=False, value=demo.describe())
                video = gr.Video(label="Recorded episode (selected camera)", height=360, autoplay=True)
            with gr.Column(scale=2):
                disabled = agent is None
                chat_label = "Agent" + (" (disabled - no LLM backend)" if disabled else "")
                # gradio >=5/6 default to the OpenAI-style messages format and
                # dropped the `type` kwarg; <5 needs type="messages". Feature-test
                # so the UI builds on any supported gradio (all messages-mode, so
                # the dict-history handler below is correct either way).
                try:
                    chatbot = gr.Chatbot(label=chat_label, type="messages", height=420)
                except TypeError:
                    chatbot = gr.Chatbot(label=chat_label, height=420)
                msg = gr.Textbox(
                    label="Message", placeholder="e.g. 'generate 10 episodes' / 'plan and execute'", lines=2
                )
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")

        def on_refresh(cam):
            demo.set_camera(cam)
            return demo.render(cam)

        def on_run(cam):
            text = demo.plan_and_execute(task=DEFAULT_TASK)
            return demo.render(cam), text, demo.latest_video(cam)

        def on_generate(cam, n):
            summary = demo.record_dataset(n_episodes=int(n or 1))
            if summary.get("status") != "success":
                return demo.render(cam), f"Could not record: {summary.get('message', summary)}", None
            text = (
                f"Recorded {summary['episodes']} episodes ({summary['total_frames']} frames, "
                f"planner={summary['planner']}, success_rate={summary['success_rate']:.0%}) -> {summary['repo_id']}"
            )
            return demo.render(cam), text, demo.latest_video(cam)

        def on_show_video(cam):
            # Re-fetch the recorded video for the selected camera (no re-run).
            return demo.latest_video(cam)

        def on_chat(message, history):
            history = list(history or [])
            if not message or not message.strip():
                return "", history, None
            history.append({"role": "user", "content": message})
            if agent is None:
                history.append({"role": "assistant", "content": "Chat is disabled (no LLM backend configured)."})
                return "", history, None
            from examples.so101_curobo.agent import extract_text

            try:
                reply = extract_text(agent(message)) or "(done)"
            except Exception as exc:  # noqa: BLE001
                reply = f"agent error: {type(exc).__name__}: {exc}"
            history.append({"role": "assistant", "content": reply})
            return "", history, demo.render(demo.current_camera)

        refresh_btn.click(on_refresh, inputs=[cam_dd], outputs=[preview])
        cam_dd.change(on_refresh, inputs=[cam_dd], outputs=[preview])
        cam_dd.change(on_show_video, inputs=[cam_dd], outputs=[video])
        run_btn.click(on_run, inputs=[cam_dd], outputs=[preview, status, video])
        gen_btn.click(on_generate, inputs=[cam_dd, n_eps], outputs=[preview, status, video])
        chat_io = dict(inputs=[msg, chatbot], outputs=[msg, chatbot, preview])
        send_btn.click(on_chat, **chat_io)
        msg.submit(on_chat, **chat_io)
        clear_btn.click(lambda: [], outputs=[chatbot])

    return ui


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--backend", default="mujoco", choices=["mujoco", "isaac"])
    p.add_argument("--planner", default="auto", choices=["auto", "scripted", "curobo", "precomputed"])
    p.add_argument(
        "--curobo-urdf",
        default=None,
        help="SO-101 URDF for cuRobo. Defaults to env SO101_URDF, else the "
        "auto-downloaded strands-robots SO-101 cache URDF (no flag needed once "
        "the MuJoCo demo has populated the asset cache).",
    )
    p.add_argument(
        "--curobo-asset",
        default="",
        help="Mesh root for the SO-101 URDF. Defaults to env SO101_ASSET, else the cache mesh dir.",
    )
    p.add_argument(
        "--curobo-traj",
        default=None,
        help="Replay a cuRobo trajectory pre-planned offline (JSON from plan_curobo_offline.py; "
        "or env SO101_CUROBO_TRAJ). Used on the Isaac backend, where in-kit cuRobo can't run "
        "(warp version conflict): --planner curobo replays this instead of falling back to scripted.",
    )
    p.add_argument("--repo-id", default="local/so101_curobo_pickplace")
    p.add_argument("--root", default=None, help="On-disk dataset dir (default: HF cache).")
    p.add_argument("--no-images", action="store_true", help="Record state+action only (no GL/EGL needed).")
    p.add_argument("--episodes", type=int, default=2, help="Episodes for --smoke.")
    p.add_argument("--smoke", action="store_true", help="Headless: record episodes, print summary, exit.")
    p.add_argument("--server-name", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=7863)
    p.add_argument("--share", action="store_true")
    p.add_argument("--no-agent", action="store_true")
    p.add_argument("--model", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if not args.no_images:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from examples.so101_curobo.controller import SO101CuroboDemo

    planner_kwargs = {}
    if args.curobo_urdf:
        planner_kwargs["urdf_path"] = args.curobo_urdf
    if args.curobo_asset:
        planner_kwargs["asset_path"] = args.curobo_asset
    if args.curobo_traj:
        planner_kwargs["traj_path"] = args.curobo_traj

    demo = SO101CuroboDemo(
        backend=args.backend,
        repo_id=args.repo_id,
        root=args.root,
        prefer_planner=args.planner,
        record_images=not args.no_images,
        planner_kwargs=planner_kwargs,
    ).build()

    if args.smoke:
        summary = demo.record_dataset(n_episodes=max(1, args.episodes), randomize=False)
        print("[so101_curobo] smoke summary:", summary)
        demo.close()
        return

    agent = None
    if not args.no_agent:
        try:
            from examples.so101_curobo.agent import build_agent

            agent = build_agent(demo, model_id=args.model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent unavailable (%s); buttons-only UI.", exc)

    ui = build_ui(demo, agent=agent)
    print(f"[so101_curobo] UI at http://{args.server_name}:{args.server_port} - Ctrl-C to stop", flush=True)

    # Isaac Sim's renderer/physics may only be driven from the thread that
    # created SimulationApp (the main thread); Gradio serves callbacks on worker
    # threads where that deadlocks. So for the Isaac backend we launch Gradio in
    # a daemon thread and run the sim's pump() loop on the main thread (it
    # applies queued actions, steps, and caches camera frames the UI reads).
    sim = getattr(demo, "sim", None)
    if args.backend in ("isaac", "isaacsim", "isaac_sim") and hasattr(sim, "run_pump_forever"):
        import threading

        threading.Thread(
            target=ui.launch,
            kwargs=dict(server_name=args.server_name, server_port=args.server_port, share=args.share),
            daemon=True,
        ).start()
        try:
            sim.run_pump_forever()
        except KeyboardInterrupt:
            print("[so101_curobo] stopping...", flush=True)
        return

    ui.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
