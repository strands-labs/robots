"""GPU-gated integration: IsaacSimulation.get_body_state + predicate DSL (#1802).

Pins the two consumers the new ``get_body_state`` unblocked on Isaac against a
real Kit runtime:

* the MuJoCo-envelope-compatible body pose read (object registry + USD
  robot-link prim walk), and
* the predicate DSL (``body_above_z`` / ``distance_less_than``), whose
  ``_body_position`` primitive previously found no ``get_body_state`` on the
  Isaac backend and silently degraded every body predicate to ``False``.

Run with::

    STRANDS_GPU_TEST=1 hatch run test-integ tests_integ/simulation/test_isaac_body_state_gpu.py -m gpu -v
"""

from __future__ import annotations

import os

import pytest

# The isaac subpackage import itself is CPU-safe (heavy omni/isaacsim imports
# are lazy, deferred to create_world()); importorskip guards against a
# broken/partial install rather than the Isaac Kit runtime.
pytest.importorskip("strands_robots.simulation.isaac")

_GPU_ENABLED = os.environ.get("STRANDS_GPU_TEST", "0") == "1"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _GPU_ENABLED,
        reason="Requires an NVIDIA GPU + Isaac Sim 6.0. Set STRANDS_GPU_TEST=1 to enable.",
    ),
]


def _skip_if_isaac_unavailable() -> None:
    from strands_robots.simulation.isaac import IsaacSimulation

    available, reason = IsaacSimulation.is_available()
    if not available:
        pytest.skip(f"Isaac Sim not available: {reason}")


def _json_payload(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if "json" in c)


class TestIsaacBodyStateGPU:
    def test_body_state_and_predicates_end_to_end(self):
        """One Kit boot covers the object read, the robot-link prim read, and
        the predicate-DSL unlock (SimulationApp can only boot once per
        process, so the journeys share a session like the other GPU tests)."""
        from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation
        from strands_robots.simulation.predicates import PREDICATE_REGISTRY

        _skip_if_isaac_unavailable()
        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
        try:
            r = sim.create_world()
            assert r["status"] == "success", f"create_world: {r}"

            # --- object registry path ---------------------------------------
            r = sim.add_object(name="probe_cube", shape="box", position=[0.3, 0.1, 0.5], is_static=True)
            assert r["status"] == "success", f"add_object: {r}"
            sim.step(2)

            r = sim.get_body_state(body_name="probe_cube")
            assert r["status"] == "success", f"get_body_state(object): {r}"
            payload = _json_payload(r)
            assert payload["position"] == pytest.approx([0.3, 0.1, 0.5], abs=1e-3)
            assert len(payload["quaternion"]) == 4
            assert sum(c * c for c in payload["quaternion"]) == pytest.approx(1.0, abs=1e-6)

            # --- unknown body: structured error, not an exception -----------
            r = sim.get_body_state(body_name="no_such_body")
            assert r["status"] == "error"
            assert "no_such_body" in r["content"][0]["text"]

            # --- robot-link prim path (bundled Franka USD) -------------------
            try:
                from isaacsim.storage.native import get_assets_root_path
            except ImportError:
                from omni.isaac.nucleus import get_assets_root_path
            assets_root = get_assets_root_path()
            if not assets_root:
                pytest.skip("No Isaac assets root (Nucleus/CDN) reachable for the Franka USD")
            usd_path = f"{assets_root}/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
            r = sim.add_robot("robot", usd_path=usd_path)
            if r["status"] != "success":
                usd_path = f"{assets_root}/Isaac/Robots/Franka/franka.usd"
                r = sim.add_robot("robot", usd_path=usd_path)
            assert r["status"] == "success", f"add_robot: {r}"
            sim.step(2)

            # Namespaced and bare link forms both resolve; #1802's LIBERO
            # EEF source reads exactly this body.
            for name in ("robot/panda_hand", "panda_hand"):
                r = sim.get_body_state(body_name=name)
                assert r["status"] == "success", f"get_body_state({name!r}): {r}"
                hand = _json_payload(r)
                assert len(hand["position"]) == 3
                assert sum(c * c for c in hand["quaternion"]) == pytest.approx(1.0, abs=1e-6)

            # --- predicate DSL unlock ----------------------------------------
            above = PREDICATE_REGISTRY["body_above_z"](body="probe_cube", z=0.2)
            below = PREDICATE_REGISTRY["body_above_z"](body="probe_cube", z=5.0)
            assert above(sim) is True
            assert below(sim) is False

            near = PREDICATE_REGISTRY["distance_less_than"](body_a="probe_cube", body_b="panda_hand", threshold=10.0)
            assert near(sim) is True
        finally:
            sim.destroy()
