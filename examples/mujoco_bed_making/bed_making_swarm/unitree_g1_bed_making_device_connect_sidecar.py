#!/usr/bin/env python3
"""Register the bed-making G1s with Device Connect so they appear in the dashboard.

Runs one ``DeviceRuntime`` per robot against the Device Connect NATS broker. The
two runtimes share a single ``asyncio`` event loop. Each robot uses its own JWT
credentials file and registers as an **equal swarm peer** exposing the
bed-making swarm surface defined in
``examples/mujoco_bed_making/bed_making_swarm/unitree_g1_bed_making_g1_driver.py``:

- callable actions (``@rpc``): ``askForHelp`` / ``offerHelp`` /
  ``pickUpBedSheet`` / ``walkToNextCorner`` / ``putDownBedSheet`` plus
  read-only ``getStatus`` / ``getGoalState`` / ``listPeers`` /
  ``getEventHistory`` / ``getHelpHistory`` / ``emergencyStopAll``;
- broadcast events (``@emit``): ``helpRequested`` / ``helpOffered`` /
  ``walkingTo`` / ``goalReached`` / ``cornerHeld`` / ``cornerPlaced`` / ...

Why a separate sidecar?
-----------------------
The two peers only exist on the Device Connect dashboard while *some* process
is hosting their ``DeviceRuntime``. The swarm demo
(``unitree_g1_bed_making_swarm_demo.py``) and the MuJoCo visualisation
(``unitree_g1_bed_making_demo.py``) host them only for the duration of their
run, then exit and unregister the robots. This sidecar hosts the two runtimes
**on their own**, with no simulation and no scripted scenario, so that:

- the robots stay **online and idle** in the dashboard for as long as you like,
  ready for a human or an AI Fabric orchestrator to invoke their callable
  functions interactively (e.g. click ``askForHelp`` / ``pickUpBedSheet`` from
  the portal, or call them via ``device-connect-agent-tools``);
- you can bring the peers up on a machine that **cannot run the simulation**
  (no MuJoCo assets, no GPU/GL, headless CI) — registration only needs the
  network and the credentials;
- you can validate Device Connect connectivity, credentials, and the dashboard
  presence/RPC surface **in isolation** from the simulation, which makes
  debugging broker/auth issues much easier.

This sidecar does **not** run the MuJoCo simulation and does **not** drive the
robots toward the goal. To watch the swarm reach "the bed is made" and exercise
ask-for-help/offer-help between the peers, use
``unitree_g1_bed_making_swarm_demo.py``; to run the MuJoCo visualisation, use
``unitree_g1_bed_making_demo.py``.

Usage:

    .venv/bin/python examples/mujoco_bed_making/bed_making_swarm/unitree_g1_bed_making_device_connect_sidecar.py

Override the broker URL with ``--nats-url`` or ``$DEVICE_CONNECT_NATS_URL``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import threading  # noqa: E402

from device_connect_edge import DeviceRuntime  # noqa: E402

from examples.mujoco_bed_making.bed_making_swarm.unitree_g1_bed_making_g1_driver import BedMakingG1Driver  # noqa: E402

DEFAULT_NATS_URL = "nats://fabric.deviceconnect.dev:4222"
# Both G1s are equal swarm peers (no control/worker split) — the role label
# is kept only so multiple peers have distinct, readable identifiers.
DEFAULT_CREDENTIALS = (
    (".credentials/beta-unitree-g1-humanoid-0.creds.json", "peer-0"),
    (".credentials/beta-unitree-g1-humanoid-1.creds.json", "peer-1"),
)


@dataclass(frozen=True)
class RobotSpec:
    role: str
    device_id: str
    credentials_file: Path


def default_specs(repo_root: Optional[Path] = None) -> List["RobotSpec"]:
    """Return the standard (control, worker) robot specs from repo creds."""

    root = repo_root or REPO_ROOT
    specs: List[RobotSpec] = []
    for path, role in DEFAULT_CREDENTIALS:
        creds_path = root / path
        if not creds_path.exists():
            continue
        specs.append(
            RobotSpec(
                role=role,
                device_id=_device_id_from_creds(creds_path),
                credentials_file=creds_path,
            )
        )
    return specs


def start_runtimes_in_thread(
    specs: List["RobotSpec"],
    *,
    nats_url: str = DEFAULT_NATS_URL,
    tenant: Optional[str] = None,
) -> "BackgroundRuntime":
    """Run :func:`_run_runtime` for each spec in a background daemon thread.

    The returned :class:`BackgroundRuntime` can be used to cleanly stop the
    runtimes when the caller exits. Failures inside the thread are logged but
    do not propagate.
    """

    return BackgroundRuntime.start(specs, nats_url=nats_url, tenant=tenant)


class BackgroundRuntime:
    """Wrapper around an asyncio event loop running in a daemon thread.

    Lets the MuJoCo demo register both robots with Device Connect while the
    simulation runs on the main thread. Call :meth:`stop` (or use it as a
    context manager) to cancel the runtimes and join the thread.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._tasks: List[asyncio.Task] = []
        self._stopped = threading.Event()
        # role -> DeviceRuntime mapping so the demo can reach the underlying
        # driver to emit DC events from the main thread.
        self.runtimes: Dict[str, Any] = {}

    @classmethod
    def start(
        cls,
        specs: List["RobotSpec"],
        *,
        nats_url: str,
        tenant: Optional[str],
    ) -> "BackgroundRuntime":
        bg = cls()
        bg._start(specs, nats_url=nats_url, tenant=tenant)
        return bg

    def _start(self, specs: List["RobotSpec"], *, nats_url: str, tenant: Optional[str]) -> None:
        ready = threading.Event()

        def _thread_main() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            try:
                for spec in specs:
                    runtime = build_runtime(spec, nats_url=nats_url, tenant=tenant)
                    # Key by device_id (peers are equal — there is no unique
                    # "control"/"worker" role to key on) and keep a role alias
                    # for readability/back-compat.
                    self.runtimes[spec.device_id] = runtime
                    self.runtimes.setdefault(spec.role, runtime)
                    self._tasks.append(
                        loop.create_task(
                            _run_runtime(spec, nats_url=nats_url, tenant=tenant, runtime=runtime),
                            name=f"runtime-{spec.role}",
                        )
                    )
                ready.set()
                loop.run_forever()
            except Exception:
                logging.getLogger("sidecar.background").exception("runtime thread crashed")
            finally:
                for task in self._tasks:
                    if not task.done():
                        task.cancel()
                pending = [t for t in self._tasks if not t.done()]
                if pending:
                    try:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    except Exception:
                        pass
                loop.close()
                self._stopped.set()

        thread = threading.Thread(target=_thread_main, name="device-connect-runtime", daemon=True)
        self._thread = thread
        thread.start()
        ready.wait(timeout=10.0)

    def stop(self, timeout: float = 10.0) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            return
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._stopped.wait(timeout=timeout)

    def __enter__(self) -> "BackgroundRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nats-url",
        default=os.environ.get("DEVICE_CONNECT_NATS_URL", DEFAULT_NATS_URL),
        help="NATS broker URL for Device Connect (default: %(default)s).",
    )
    parser.add_argument(
        "--credentials",
        action="append",
        metavar="ROLE=PATH",
        help=(
            "Register a robot under ROLE ('control' or 'worker') using the JWT "
            "credentials file at PATH. Can be passed multiple times. Defaults to "
            "the two repo-local credentials in `.credentials/`."
        ),
    )
    parser.add_argument(
        "--tenant",
        default=os.environ.get("DEVICE_CONNECT_TENANT"),
        help=(
            "Override the tenant from the credentials file. Usually unnecessary; the "
            "JWT in the credentials file already encodes the tenant."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Python log level (default: %(default)s).",
    )
    return parser.parse_args()


def _resolve_specs(args: argparse.Namespace) -> List[RobotSpec]:
    raw_pairs: List[tuple[str, str]] = []
    if args.credentials:
        for entry in args.credentials:
            if "=" not in entry:
                raise SystemExit(
                    f"--credentials must be ROLE=PATH; got {entry!r}"
                )
            role, path = entry.split("=", 1)
            raw_pairs.append((role.strip(), path.strip()))
    else:
        raw_pairs = [(role, str(REPO_ROOT / path)) for path, role in DEFAULT_CREDENTIALS]

    specs: List[RobotSpec] = []
    for role, path in raw_pairs:
        creds_path = Path(path).expanduser()
        if not creds_path.is_absolute():
            creds_path = REPO_ROOT / creds_path
        if not creds_path.exists():
            raise SystemExit(f"Credentials file not found: {creds_path}")
        device_id = _device_id_from_creds(creds_path)
        specs.append(RobotSpec(role=role, device_id=device_id, credentials_file=creds_path))
    return specs


def _device_id_from_creds(path: Path) -> str:
    import json

    payload = json.loads(path.read_text())
    device_id = payload.get("device_id")
    if not device_id:
        raise SystemExit(f"No device_id in credentials file {path}")
    return device_id


def build_runtime(spec: RobotSpec, *, nats_url: str, tenant: Optional[str]) -> Any:
    """Build a DeviceRuntime for ``spec`` ready to be run on an asyncio loop.

    Every robot registers as an equal swarm ``peer`` with ``auto_offer``
    enabled, so a free peer automatically offers help when it sees another
    peer broadcast a ``helpRequested`` event — the swarm self-organises
    without a central controller (AI Fabric acts as orchestrator/observer).
    ``messaging_urls`` overrides the internal URL embedded in the credentials
    file so the peer connects to the public fabric broker.
    """

    driver = BedMakingG1Driver(role="peer", device_id=spec.device_id, auto_offer=True)
    runtime_kwargs = {
        "driver": driver,
        "device_id": spec.device_id,
        "messaging_backend": "nats",
        "messaging_urls": [nats_url],
        "credentials_file": str(spec.credentials_file),
    }
    if tenant:
        runtime_kwargs["tenant"] = tenant
    return DeviceRuntime(**runtime_kwargs)


async def _run_runtime(
    spec: RobotSpec,
    *,
    nats_url: str,
    tenant: Optional[str],
    runtime: Optional[Any] = None,
) -> None:
    logger = logging.getLogger(f"sidecar.{spec.role}")
    runtime = runtime if runtime is not None else build_runtime(spec, nats_url=nats_url, tenant=tenant)
    logger.info(
        "starting runtime device_id=%s role=%s creds=%s broker=%s",
        spec.device_id,
        spec.role,
        spec.credentials_file,
        nats_url,
    )
    try:
        await runtime.run()
    except Exception as exc:
        logger.exception("runtime for %s exited with error: %s", spec.device_id, exc)
        raise


async def _main(args: argparse.Namespace) -> int:
    specs = _resolve_specs(args)
    logging.info(
        "registering %d robots against %s: %s",
        len(specs),
        args.nats_url,
        ", ".join(f"{s.role}={s.device_id}" for s in specs),
    )
    tasks = [
        asyncio.create_task(
            _run_runtime(spec, nats_url=args.nats_url, tenant=args.tenant),
            name=f"runtime-{spec.role}",
        )
        for spec in specs
    ]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    return 0


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        return asyncio.run(_main(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
