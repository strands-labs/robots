"""The dashboard as the routes and helpers see it.

A route or helper module types its ``dash`` parameter against :class:`DashboardLike`, not the concrete
:class:`~strands_robots.dashboard.dashboard.Dashboard`, so it never imports the module that imports it back.
The one ``Dashboard`` satisfies this structurally; this module imports nothing else in the package (a leaf).
"""

from __future__ import annotations

from typing import Any, Protocol


class DashboardLike(Protocol):
    """What a route or helper reads off the dashboard: the world, the hardware robots, and the page's verbs."""

    sim: Any
    real: dict[str, Any]
    hz: float
    model: Any
    mesh: Any
    voice: Any
    auth: Any
    _agent: Any

    @property
    def connections(self) -> Any: ...

    @property
    def agent(self) -> Any: ...

    def _robot(self, rid: str) -> Any: ...

    def act(self, rid: str, targets: Any, *, confirmed: bool) -> dict[str, Any]: ...

    def add_robot(self, name: Any, *, id: str | None = None, position: Any = None) -> dict[str, Any]: ...

    def estop(self, rid: str) -> dict[str, Any]: ...

    def resume(self, rid: str, *, approved_by: Any, confirmed: bool = False) -> dict[str, Any]: ...

    def world(self, call: Any, *, confirmed: bool = False) -> Any: ...

    def frame(self) -> dict[str, Any]: ...

    def scene_json(self) -> dict[str, Any]: ...

    def cameras_json(self) -> list[dict[str, Any]]: ...

    def asset_file(self, path: str) -> str: ...

    def mesh_json(self) -> dict[str, Any]: ...

    def mesh_estop(self, device_id: str) -> Any: ...
