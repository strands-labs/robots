"""In-process Device Connect swarm coordination for the Isaac Sim demo.

Wraps :class:`BedMakingG1Driver` from ``swarm_driver.py`` — a self-contained copy of the
equal-peer swarm driver, bundled here so the Isaac demo has **no dependency on the MuJoCo
demo** — so the Isaac Sim main loop, which is synchronous, can drive two G1 peers that
coordinate over Device Connect on a background asyncio thread. Two transports:

* ``loopback`` — no broker; an in-process bus fans events between peers. Offline.
* ``broker``   — both peers register on the real Device Connect NATS fabric using
  the JWT creds in ``.credentials/`` so they appear live in the dashboard with
  callable functions and a real event stream.

The sim loop calls :meth:`SwarmCoordinator.invoke` (blocking) and
:meth:`SwarmCoordinator.snapshot`; all driver/runtime work happens on the
coordinator's own event loop thread.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.isaac_bed_making.swarm_driver import (  # noqa: E402
    GOAL_STATE,
    SHEET_TO_BED,
    BedMakingG1Driver,
)

DEFAULT_NATS_URL = "nats://fabric.deviceconnect.dev:4222"


class _LoopbackBus:
    """Fan every emitted event out to the other peers' ``@on`` handlers."""

    def __init__(self) -> None:
        self._subs: Dict[str, List[Tuple[str, Any]]] = {}

    def register(self, driver: BedMakingG1Driver) -> None:
        for _, method in inspect.getmembers(driver, predicate=inspect.ismethod):
            event_name = getattr(method, "_sub_event_name", None)
            if event_name:
                self._subs.setdefault(event_name, []).append((driver.device_id, method))
        driver.set_emit_sink(self._make_sink(driver.device_id))

    def _make_sink(self, source_id: str):
        async def _sink(event_name: str, payload: dict) -> None:
            for sub_id, handler in self._subs.get(event_name, []):
                if sub_id == source_id:
                    continue
                try:
                    await handler(source_id, event_name, payload)
                except Exception:
                    # One subscriber's failure must not break fan-out to the others.
                    pass

        return _sink


def default_creds_specs() -> List[Tuple[str, Path]]:
    """[(device_id, creds_path)] for the two bed-making G1s, if present."""
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


class SwarmCoordinator:
    """Runs two G1 Device Connect peers on a background event loop."""

    def __init__(self, mode: str = "loopback", nats_url: str = DEFAULT_NATS_URL) -> None:
        self.mode = mode
        self.nats_url = nats_url
        self.goal_state = GOAL_STATE
        self.sheet_to_bed = SHEET_TO_BED
        self.peers: List[BedMakingG1Driver] = []
        self._runtimes: List[Any] = []
        self._tasks: List[Any] = []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="dc-swarm")
        self._ready = threading.Event()

    # ── lifecycle ─────────────────────────────────────────────────────────
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def start(self, device_ids: Optional[List[str]] = None) -> List[str]:
        """Start the loop thread and register the two peers. Returns device ids."""
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._async_start(device_ids), self._loop)
        ids = fut.result(timeout=30)
        self._ready.set()
        return ids

    async def _async_start(self, device_ids: Optional[List[str]]) -> List[str]:
        if self.mode == "broker":
            specs = default_creds_specs()
            if len(specs) < 2:
                raise RuntimeError(
                    "broker mode needs two .credentials/*.creds.json files; "
                    "use mode='loopback' to run offline."
                )
            from device_connect_edge import DeviceRuntime

            for device_id, creds in specs:
                drv = BedMakingG1Driver(device_id=device_id, role="peer", auto_offer=True)
                rt = DeviceRuntime(
                    driver=drv, device_id=device_id, messaging_backend="nats",
                    messaging_urls=[self.nats_url], credentials_file=str(creds),
                )
                self.peers.append(drv)
                self._runtimes.append(rt)
            self._tasks = [self._loop.create_task(rt.run()) for rt in self._runtimes]
            await asyncio.sleep(3.0)  # registration + command subscription
        else:
            ids = device_ids or ["beta-unitree-g1-humanoid-0", "beta-unitree-g1-humanoid-1"]
            bus = _LoopbackBus()
            for device_id in ids:
                drv = BedMakingG1Driver(device_id=device_id, role="peer", auto_offer=True)
                bus.register(drv)
                self.peers.append(drv)
        return [p.device_id for p in self.peers]

    def stop(self) -> None:
        async def _shutdown() -> None:
            for rt in self._runtimes:
                stop = getattr(rt, "stop", None)
                if stop:
                    try:
                        res = stop()
                        if inspect.isawaitable(res):
                            await res
                    except Exception:
                        # Best-effort per-driver stop; keep tearing down the rest.
                        pass
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), self._loop).result(timeout=10)
        except Exception:
            # Best-effort shutdown; stop the loop regardless of a drain error.
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    # ── synchronous bridge for the sim loop ───────────────────────────────
    def invoke(self, peer_idx: int, fn_name: str, timeout: float = 15.0, **params) -> dict:
        """Call a peer's Device Connect @rpc function and return its result."""

        async def _call() -> dict:
            method = getattr(self.peers[peer_idx], fn_name)
            result = await method(**params)
            return result if isinstance(result, dict) else {"result": result}

        return asyncio.run_coroutine_threadsafe(_call(), self._loop).result(timeout=timeout)

    def snapshot(self) -> List[dict]:
        return [p.agent.snapshot() for p in self.peers]

    def event_history(self, peer_idx: int = 0, limit: int = 50) -> List[dict]:
        return self.peers[peer_idx].agent.event_history(limit=limit)

    def help_history(self, peer_idx: int = 0, limit: int = 50) -> List[dict]:
        return self.peers[peer_idx].agent.help_history(limit=limit)
