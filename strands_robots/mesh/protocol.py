"""The slice of ``Mesh`` that a remote body, the broadcast helpers and the agent tool speak to, as a Protocol.

``remote.py``, ``broadcast.py`` and ``tool.py`` each take a ``Mesh`` but need only this handful of its verbs. They
type against ``MeshView`` here -- a leaf module that imports nothing from ``mesh.py`` -- so none of them forms an
import cycle through ``mesh.py``. ``Mesh`` satisfies ``MeshView`` structurally; nothing need inherit it.
"""

from __future__ import annotations

from typing import Any, Protocol

from .device import Device
from .transports.base import Transport


class MeshView(Protocol):
    """One tenant's mesh as its helpers use it: who it is, the devices it sees, one rpc, one broadcast."""

    tenant: str
    operator: str
    peer_id: str
    tasks: dict[str, Any]
    transports: list[Transport]

    def invoke(self, device_id: str, rpc: str, *, timeout: float = ..., **args: Any) -> Any: ...

    def discover(self, selector: str = ..., *, min_count: int = ..., timeout: float = ...) -> list[Device]: ...

    def broadcast(
        self, command: str, *, selector: str = ..., deadline: float = ..., approved_by: str | None = ...
    ) -> list[str]: ...

    def connect(self, device_id: str, *, images: bool = ...) -> Any: ...

    def _present(self) -> list[Device]: ...

    def _transport_of(self, device_id: str) -> Transport: ...
