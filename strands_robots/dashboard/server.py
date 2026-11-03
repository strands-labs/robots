"""uvicorn in a thread, on a loopback port, with a clean stop."""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

import uvicorn

_LOOPBACK = ("127.0.0.1", "localhost", "::1")


class Server:
    def __init__(self, app: Any, *, host: str = "127.0.0.1", port: int = 0) -> None:
        if host not in _LOOPBACK:
            raise ValueError(f"host={host!r}: the dashboard binds loopback only")
        if port == 0:
            with socket.socket() as s:
                s.bind((host, 0))
                port = s.getsockname()[1]
        self.host, self.port = host, port
        self._server = uvicorn.Server(
            uvicorn.Config(app, host=host, port=port, log_level="warning", ws="websockets-sansio")
        )
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def run(self) -> None:
        self._server.run()

    def start(self, timeout: float = 10.0) -> Server:
        self._thread = threading.Thread(target=self._server.run, name="dashboard", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError(f"dashboard did not start on {self.url}")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)

    def __enter__(self) -> Server:
        return self.start() if self._thread is None else self

    def __exit__(self, *exc: Any) -> None:
        self.stop()
