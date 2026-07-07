#!/usr/bin/env python3
"""AI-Fabric-orchestrated swarm bed-making demo (two equal Unitree G1 peers).

This script implements the issue #2 feedback (comment 4580165360): instead of
a control robot and a worker robot, **both G1s are equal swarm peers** that
independently pursue the shared goal state ``"the bed is made"``. Every peer
can *see* every other peer over Device Connect and can **ask any peer for help**
or **offer help** to any peer that asks. **AI Fabric acts as the swarm
orchestrator** — it observes the swarm and nudges peers toward the goal by
invoking their Device Connect functions, but the peers also self-organise
(a free peer auto-offers help when it sees a ``helpRequested`` event).

What it demonstrates, all visible in the Device Connect dashboard:

1. Both peers register and appear online with callable functions.
2. The orchestrator drives the bed-making sub-tasks via callable ``@rpc``
   actions: ``pickUpBedSheet`` -> ``walkToNextCorner`` -> ``putDownBedSheet``.
3. A deliberate "a placed corner started to drift" event triggers one peer to
   ``askForHelp`` and the other to ``offerHelp`` — the classic
   "ask another robot for help when stuck" scenario.
4. The full **event history** and the dedicated **help-request history** are
   printed at the end (and are queryable live via ``getEventHistory`` /
   ``getHelpHistory`` from the dashboard).

Two transports are supported:

* ``--broker`` (default): register both peers on the real Device Connect NATS
  broker using the JWT credentials in ``.credentials/`` so they appear live in
  the dashboard (callable functions invocable from the portal UI). The AI
  Fabric orchestrator drives the peers in-process; the Device Connect events
  they emit traverse the broker so the dashboard reflects every transition.
  Requires network access to ``fabric.deviceconnect.dev``.
* ``--loopback``: run with no broker at all. An in-process bus fans every
  emitted event out to the other peer's ``@on`` handlers, so the swarm
  coordinates exactly as it would over NATS. Useful for CI, offline dev, and
  on machines without broker access.

Usage::

    # Live dashboard (real broker, default):
    .venv/bin/python examples/mujoco_bed_making/bed_making_swarm/unitree_g1_bed_making_swarm_demo.py

    # Offline, no broker:
    .venv/bin/python examples/mujoco_bed_making/bed_making_swarm/unitree_g1_bed_making_swarm_demo.py --loopback

    # Keep the peers registered after the scenario so you can poke them from
    # the dashboard (broker mode only):
    .venv/bin/python examples/mujoco_bed_making/bed_making_swarm/unitree_g1_bed_making_swarm_demo.py --hold 120
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.mujoco_bed_making.bed_making_swarm.unitree_g1_bed_making_g1_driver import (  # noqa: E402
    GOAL_STATE,
    SHEET_TO_BED,
    BedMakingG1Driver,
)

logger = logging.getLogger("swarm_demo")

DEFAULT_NATS_URL = "nats://fabric.deviceconnect.dev:4222"


# ── In-process loopback event bus ──────────────────────────────────────────


class LoopbackBus:
    """Fans every emitted event out to the OTHER peers' ``@on`` handlers.

    This reproduces Device Connect's event routing without a broker: when a
    peer emits ``helpRequested``, every other registered peer's matching
    ``@on(event_name="helpRequested")`` handler is invoked with
    ``(device_id, event_name, payload)`` — identical to the NATS path.
    """

    def __init__(self) -> None:
        # event_name -> list[(device_id, handler)]
        self._subs: Dict[str, List[Tuple[str, Any]]] = {}
        self._drivers: List[BedMakingG1Driver] = []

    def register(self, driver: BedMakingG1Driver) -> None:
        self._drivers.append(driver)
        for _, method in inspect.getmembers(driver, predicate=inspect.ismethod):
            event_name = getattr(method, "_sub_event_name", None)
            if event_name:
                self._subs.setdefault(event_name, []).append((driver.device_id, method))
        driver.set_emit_sink(self._make_sink(driver.device_id))

    def _make_sink(self, source_id: str):
        async def _sink(event_name: str, payload: dict) -> None:
            for sub_device_id, handler in self._subs.get(event_name, []):
                if sub_device_id == source_id:
                    continue  # @on handlers also self-filter, but skip early
                try:
                    await handler(source_id, event_name, payload)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("loopback handler %s failed: %s", event_name, exc)

        return _sink


# ── Orchestrator ───────────────────────────────────────────────────────────


def _banner(msg: str) -> None:
    print(f"\n=== {msg} ===")


async def run_scenario(
    peers: List[BedMakingG1Driver],
    *,
    invoke,
    step_delay: float = 0.0,
) -> dict:
    """Drive the swarm to the goal state and exercise ask/offer-help.

    ``invoke(driver, fn_name, **params)`` actually calls the function. RPCs are
    always in-process (both broker and loopback modes use
    ``make_direct_invoker``); only Device Connect events traverse the broker.
    Returns a structured result dict for assertions/reporting.
    """

    p0, p1 = peers[0], peers[1]

    async def pause() -> None:
        if step_delay:
            await asyncio.sleep(step_delay)

    _banner(f"AI Fabric orchestrator: goal = {GOAL_STATE!r}")
    print(f"swarm peers: {', '.join(p.device_id for p in peers)}")

    # The two peers split the four corners and place them, alternating so both
    # are visibly contributing to the shared goal.
    assignments = [
        (p0, "A"),
        (p1, "B"),
        (p0, "C"),
        (p1, "D"),
    ]

    for peer, corner in assignments:
        _banner(f"{peer.device_id}: place sheet corner {corner} (bed {SHEET_TO_BED[corner]})")
        await invoke(peer, "pickUpBedSheet", corner=corner)
        await pause()
        walk = await invoke(peer, "walkToNextCorner", direction="counterclockwise")
        print(f"  walked to bed corner {walk.get('target_corner')}")
        await pause()

        # After the second corner is down, simulate the realistic failure from
        # the issue: a previously placed corner starts to drift, so this peer
        # gets "stuck" and asks the swarm for help holding it. The other peer
        # responds autonomously — every peer runs with auto_offer, so seeing a
        # ``helpRequested`` it can service makes it offer help on its own (no
        # orchestrator step needed). This is the peer-to-peer "respond to
        # requests for help" behaviour from the issue feedback.
        if corner == "C":
            _banner(f"{p0.device_id} is STUCK: a placed corner is drifting — asking the swarm for help")
            await invoke(
                p0,
                "askForHelp",
                corner="A",
                reason="placed corner A is drifting while I place C",
            )
            await pause()
            offered_by = [
                h["from"] for peer_drv in peers for h in peer_drv.agent.help_history() if h["kind"] == "offer"
            ]
            if offered_by:
                _banner(f"{offered_by[0]} autonomously offered to hold corner A")

        place = await invoke(peer, "putDownBedSheet", corner=corner)
        made = place.get("bed_made")
        print(f"  placed {corner} -> bed_made={made}")
        await pause()

    # Final goal check from each peer's own perspective.
    goals = {}
    for peer in peers:
        g = await invoke(peer, "getGoalState")
        goals[peer.device_id] = g
        print(f"  {peer.device_id}: progress={g['progress']} bed_made={g['bed_made']}")

    return {"goals": goals}


def _print_history(peers: List[BedMakingG1Driver]) -> None:
    for peer in peers:
        snap = peer.agent.snapshot()
        _banner(f"event history — {peer.device_id} ({snap['event_count']} events)")
        for e in peer.agent.event_history(limit=50):
            print(f"  [{e['ts']}] #{e['seq']:>3} {e['kind']:<12} {e['summary']}")
        help_log = peer.agent.help_history(limit=50)
        _banner(f"help history — {peer.device_id} ({len(help_log)} help records)")
        for h in help_log:
            print(f"  [{h['ts']}] #{h['seq']:>3} {h['kind']:<8} {h['summary']}")


# ── Invokers (broker vs loopback) ──────────────────────────────────────────


def make_direct_invoker():
    """Invoke a function by calling the driver method directly (loopback)."""

    async def _invoke(driver: BedMakingG1Driver, fn_name: str, **params) -> dict:
        method = getattr(driver, fn_name)
        result = await method(**params)
        return result if isinstance(result, dict) else {"result": result}

    return _invoke


# ── Credentials / specs ────────────────────────────────────────────────────


def _default_specs() -> List[Tuple[str, Path]]:
    """Return [(device_id, creds_path)] for the two bed-making G1s."""

    out: List[Tuple[str, Path]] = []
    for name in (
        "beta-unitree-g1-humanoid-0.creds.json",
        "beta-unitree-g1-humanoid-1.creds.json",
    ):
        path = REPO_ROOT / ".credentials" / name
        if path.exists():
            device_id = json.loads(path.read_text()).get("device_id") or path.stem
            out.append((device_id, path))
    return out


# ── Modes ──────────────────────────────────────────────────────────────────


async def run_loopback(step_delay: float) -> int:
    bus = LoopbackBus()
    peers = [
        BedMakingG1Driver(device_id="beta-unitree-g1-humanoid-0", role="peer", auto_offer=True),
        BedMakingG1Driver(device_id="beta-unitree-g1-humanoid-1", role="peer", auto_offer=True),
    ]
    for peer in peers:
        bus.register(peer)

    result = await run_scenario(peers, invoke=make_direct_invoker(), step_delay=step_delay)
    _print_history(peers)

    # Verify the goal was reached and help was exchanged.
    all_made = all(g["bed_made"] for g in result["goals"].values())
    helped = any(peer.agent.help_history() for peer in peers)
    _banner("RESULT")
    print(f"goal reached (bed made by all peers' view): {all_made}")
    print(f"help was requested/offered between peers:   {helped}")
    return 0 if (all_made and helped) else 1


async def run_broker(nats_url: str, hold: float, step_delay: float) -> int:
    from device_connect_edge import DeviceRuntime

    specs = _default_specs()
    if len(specs) < 2:
        print(
            "Need two credential files under .credentials/ for broker mode. "
            "Use --loopback to run without a broker.",
            file=sys.stderr,
        )
        return 2

    peers: List[BedMakingG1Driver] = []
    runtimes: List[Any] = []
    for device_id, creds_path in specs:
        driver = BedMakingG1Driver(device_id=device_id, role="peer", auto_offer=True)
        runtime = DeviceRuntime(
            driver=driver,
            device_id=device_id,
            messaging_backend="nats",
            messaging_urls=[nats_url],
            credentials_file=str(creds_path),
        )
        peers.append(driver)
        runtimes.append(runtime)

    # Start both runtimes (they register + subscribe to commands on the broker).
    tasks = [asyncio.create_task(rt.run(), name=f"runtime-{p.device_id}") for rt, p in zip(runtimes, peers)]
    await asyncio.sleep(3.0)  # allow registration + command subscription
    print(f"Registered {len(peers)} peers on {nats_url}: {', '.join(p.device_id for p in peers)}")
    print(
        "Both peers are live on the Device Connect dashboard — their callable\n"
        "functions are invocable from the portal UI, and the events they emit\n"
        "below traverse the fabric broker in real time."
    )

    # The AI Fabric orchestrator drives the peers in-process. Each action
    # mutates the peer's state and emits its Device Connect events over the
    # broker, so the dashboard reflects every transition (presence, status,
    # event stream) live. We do not stand up a separate agent-tools client:
    # that needs its own client credentials and its connect() targets the
    # Zenoh backend, whereas these device peers authenticate to the NATS
    # fabric with their JWT device credentials.
    try:
        result = await run_scenario(peers, invoke=make_direct_invoker(), step_delay=step_delay)
        _print_history(peers)
        all_made = all(g["bed_made"] for g in result["goals"].values())
        _banner("RESULT")
        print(f"goal reached: {all_made}")
        if hold > 0:
            print(f"holding peers online for {hold:.0f}s — poke them from the dashboard now…")
            await asyncio.sleep(hold)
        return 0 if all_made else 1
    finally:
        for rt in runtimes:
            stop = getattr(rt, "stop", None)
            if stop:
                try:
                    res = stop()
                    if inspect.isawaitable(res):
                        await res
                except Exception:
                    # Best-effort per-runtime stop during shutdown; drain the rest.
                    pass
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def run_mujoco(args: argparse.Namespace) -> int:
    """Show the swarm physically making the bed in MuJoCo.

    Delegates to the companion visualisation ``unitree_g1_bed_making_demo.py``
    (run as a subprocess so its import-time GL/viewer setup stays isolated and
    the two peers are registered exactly once). By default it opens the
    interactive MuJoCo viewer and registers both peers on the Device Connect
    dashboard; with ``--render-video`` it renders headlessly and encodes an
    mp4 you can share.
    """

    demo = REPO_ROOT / "examples" / "mujoco_bed_making" / "bed_making_demo" / "unitree_g1_bed_making_demo.py"
    # Pin the demo's frame output dir so the ffmpeg input dir below matches
    # exactly what the subprocess wrote (the demo's own default output-dir is a
    # different path, which would leave ffmpeg with no frames to encode).
    frames_dir = REPO_ROOT / "artifacts" / "mujoco_bed_making" / "unitree_g1_bed_making"
    cmd = [sys.executable, str(demo), "--substeps", str(args.substeps)]
    env = dict(os.environ)
    if args.no_device_connect:
        cmd.append("--no-device-connect")
    if args.render_video:
        cmd += ["--no-viewer", "--render-frames", "--output-dir", str(frames_dir)]
        # GPU offscreen rendering — the interactive viewer is not used here.
        env.setdefault("MUJOCO_GL", "egl")

    print("Launching the MuJoCo bed-making visualisation:")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"MuJoCo visualisation exited with status {result.returncode}.", file=sys.stderr)
        return result.returncode

    if args.render_video:
        frames = frames_dir
        out = frames / "bed_making.mp4"
        if shutil.which("ffmpeg") is None:
            print(
                f"Frames are in {frames}. Install ffmpeg to encode an mp4, or view the PNGs directly.",
                file=sys.stderr,
            )
            return 0
        encode = [
            "ffmpeg", "-y", "-framerate", "15",
            "-pattern_type", "glob", "-i", str(frames / "frame_*.png"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p",
            str(out),
        ]
        enc = subprocess.run(encode, capture_output=True, text=True)
        if enc.returncode == 0:
            print(f"Wrote {out}")
        else:
            print(f"ffmpeg encode failed; PNG frames remain in {frames}.", file=sys.stderr)
            return enc.returncode
    return 0


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--broker",
        action="store_true",
        help="Connect both peers to the real Device Connect NATS broker (default).",
    )
    mode.add_argument(
        "--loopback",
        action="store_true",
        help="Run with no broker; coordinate via an in-process event bus.",
    )
    mode.add_argument(
        "--mujoco",
        action="store_true",
        help=(
            "Show the two G1s physically making the bed in MuJoCo (opens the "
            "interactive viewer). This launches the companion visualisation "
            "unitree_g1_bed_making_demo.py, which also registers both peers on "
            "the Device Connect dashboard while it runs."
        ),
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        help=(
            "With --mujoco: render the run headlessly to PNG frames and encode "
            "an mp4 (no interactive window) instead of opening the viewer."
        ),
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=8,
        help=(
            "With --mujoco: physics substeps per control frame passed to the "
            "visualisation. 8+ keeps the cloth-grasp solver stable (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--no-device-connect",
        action="store_true",
        help="With --mujoco: do not register the peers on the Device Connect dashboard.",
    )
    parser.add_argument(
        "--nats-url",
        default=os.environ.get("DEVICE_CONNECT_NATS_URL", DEFAULT_NATS_URL),
        help="NATS broker URL for broker mode (default: %(default)s).",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=0.0,
        help="Broker mode: keep peers online this many seconds after the scenario.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Seconds to pause between sub-tasks (nicer to watch live).",
    )
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "WARNING"))
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s %(message)s")
    if args.mujoco:
        return run_mujoco(args)
    if args.loopback:
        return asyncio.run(run_loopback(args.step_delay))
    return asyncio.run(run_broker(args.nats_url, args.hold, args.step_delay))


if __name__ == "__main__":
    raise SystemExit(main())
