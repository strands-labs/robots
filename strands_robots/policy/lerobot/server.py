"""Serve a LeRobot deployment from this process: the same code path as ``lerobot-policy-server``."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .contract import lerobot
from .quiet import quiet_load


@contextmanager
def serve(cfg: Any, *, ready_timeout: float = 120.0) -> Iterator[str]:
    """``with serve(server_config(desc, ckpt)) as url: robot.run(Policy(url), ...)``. Yields ``zenoh://...``.

    One deployment admits one session (LeRobot's rule): serve one per robot, each with its own name.
    """
    lerobot()
    from lerobot.remote_inference import PolicyServer, SessionWorker, load_deployment
    from lerobot.transport import ZenohTransport

    with quiet_load():
        runner, identity = load_deployment(cfg)
    worker = SessionWorker(
        runner,
        deployment=cfg.deployment,
        artifact_identity=identity,
        semantics=cfg.semantics,
        action_deadline_s=cfg.execution.action_deadline_s,
        language_deadline_s=cfg.language.deadline_s,
        idle_timeout_s=cfg.execution.idle_timeout_s,
        max_input_chars=cfg.language.max_input_chars,
        max_output_chars=cfg.language.max_output_chars,
    )
    transport = ZenohTransport(cfg.zenoh)
    ready = threading.Event()
    declare = transport.declare_token

    def declare_and_signal(key: str) -> Any:  # readiness = the instance's presence token exists
        token = declare(key)
        ready.set()
        return token

    transport.declare_token = declare_and_signal  # type: ignore[method-assign]
    server = PolicyServer(worker, transport)
    thread = threading.Thread(target=server.serve, name=f"lerobot-serve-{cfg.deployment}", daemon=True)
    thread.start()
    try:
        if not ready.wait(ready_timeout):
            raise TimeoutError(f"deployment {cfg.deployment!r} did not become ready in {ready_timeout:g} s")
        endpoint = cfg.zenoh.listen_endpoints[0].removeprefix("tcp/")
        yield f"zenoh://{endpoint}/{cfg.deployment}"
    finally:
        server.stop()
        thread.join(10)
        transport.close()
        worker.close()
