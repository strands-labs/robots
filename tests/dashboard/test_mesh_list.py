"""With a mesh, the page lists the other peers' devices and can e-stop one (the world's own robots are marked local)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402

from .browser import page_on  # noqa: E402


@dataclass
class Found:  # the shape of strands_robots.mesh.Device.to_dict()
    id: str
    device_type: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "device_type": self.device_type, "tenant": "t", "labels": {}, "manifest": {}}


@dataclass
class StubMesh:
    """Mesh.discover / Mesh.invoke / tenant, recorded (tests/mesh covers the wire)."""

    tenant: str = "t"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def discover(self, selector: str = "device(*)") -> list[Found]:
        return [Found("so101", "arm"), Found("dog_1", "quadruped")]

    def invoke(self, device_id: str, rpc: str, **args: Any) -> Any:
        self.calls.append((device_id, rpc))
        return {"estop": True}


def test_the_page_lists_remote_mesh_devices_and_estops_one() -> None:
    sim = Simulation()
    Robot("so101", sim=sim)
    mesh = StubMesh()
    server = Dashboard(sim, mesh=mesh).start()
    try:
        with page_on(server.url) as page:
            row = page.locator('#mesh-devices [data-device="dog_1"]')
            row.wait_for(timeout=10_000)
            assert page.locator('#mesh-devices [data-device="so101"]').count() == 0  # local: listed once, above
            row.locator("button.estop").click()
            page.wait_for_function("window.__safety.notices.some(n => n.includes('over the mesh'))", timeout=5_000)
            assert mesh.calls == [("dog_1", "estop")]
            assert not page.errors, page.errors
    finally:
        server.stop()
        sim.close()
