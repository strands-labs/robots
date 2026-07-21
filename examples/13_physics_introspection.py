#!/usr/bin/env python3
"""Query the physics quantities behind a robot: Jacobian, mass matrix, torques, energy.

Goal: Show the classic robotics-analysis primitives the simulation exposes, the
math you reach for when writing a controller or planner - not just stepping the
sim. On a spawned SO-100 arm:

  - ``list_bodies``       - the body tree (pick the end-effector).
  - ``get_jacobian``      - the end-effector Jacobian (maps joint velocities to
                            end-effector linear/angular velocity).
  - ``get_mass_matrix``   - the joint-space inertia matrix (shape, rank,
                            condition number, total mass).
  - ``inverse_dynamics``  - the joint torques for the current state (e.g. the
                            gravity-compensation torques at rest).
  - ``get_energy``        - potential / kinetic / total energy.

Runs on CPU, no GPU/checkpoint/hardware. These are read-only queries against the
current sim state, so they compose with any scene you build.

Dependencies: pip install "strands-robots[sim-mujoco]"
Expected output: one summary line per quantity. Runtime: ~3 seconds on CPU.
"""

from __future__ import annotations

import argparse

from strands_robots import Robot


def _dispatch(sim, action: str, params: dict) -> dict:
    """Dispatch a sim action and fail loud on error."""
    result = sim._dispatch_action(action, params)
    if isinstance(result, dict) and result.get("status") == "error":
        msg = "; ".join(c.get("text", "") for c in result.get("content", []))
        raise RuntimeError(f"{action} failed: {msg}")
    return result


def _json(result: dict) -> dict:
    """Pull the structured payload out of a sim action result."""
    return next((c["json"] for c in result.get("content", []) if "json" in c), {})


def _text(result: dict) -> str:
    return next((c["text"] for c in result.get("content", []) if "text" in c), "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="so100", help="registered robot to analyze")
    args = parser.parse_args()

    sim = Robot(args.robot, mesh=False)
    # Settle a few steps so gravity is reflected in the dynamics quantities.
    for _ in range(3):
        sim.step()

    # The body tree; the last body is the end-effector for the Jacobian.
    bodies = _json(_dispatch(sim, "list_bodies", {}))["bodies"]
    ee_body = bodies[-1]
    print(f"bodies ({len(bodies)}): end-effector = {ee_body}")

    # End-effector Jacobian.
    jac = _dispatch(sim, "get_jacobian", {"body_name": ee_body})
    print(f"jacobian     : {_text(jac)[:100]}")

    # Joint-space inertia (mass) matrix.
    mm = _json(_dispatch(sim, "get_mass_matrix", {}))
    print(
        f"mass_matrix  : shape={mm.get('shape')} rank={mm.get('rank')} "
        f"cond={mm.get('condition_number'):.2f} total_mass={mm.get('total_mass'):.3f} kg"
    )

    # Inverse dynamics: the joint torques holding the current state (gravity comp at rest).
    idyn = _dispatch(sim, "inverse_dynamics", {})
    print(f"inv_dynamics : {_text(idyn)[:100]}")

    # Energy budget.
    energy = _json(_dispatch(sim, "get_energy", {}))
    print(
        f"energy       : potential={energy.get('potential'):.4f} J  "
        f"kinetic={energy.get('kinetic'):.4f} J  total={energy.get('total'):.4f} J"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
