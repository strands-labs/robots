"""End-effector frame auto-discovery for VERA eef/cartesian-delta IK.

VERA's ``eef_delta`` (mimicgen) and ``cartesian_delta`` (droid) embodiments emit
end-effector pose *deltas*; driving a MuJoCo arm needs an IK target frame (the
body/site the Cartesian task tracks). The robot registry does **not** record an
ee-frame, so it is discovered from the compiled ``mujoco.MjModel`` with a
robust, namespace-aware heuristic - making eef-delta embodiments **zero-config**.

The heuristic now lives in the shared :mod:`strands_robots.simulation.ik`
module (one home for IK target-frame discovery, used by the VERA provider and
the simulation motion primitives alike); this module re-exports
:func:`discover_ee_frame` so existing imports keep working. See the shared
module for the full heuristic contract (site hints, body hints, leaf-body
fallback) and namespace scoping semantics.

``mujoco`` is imported lazily inside the shared function, so importing this
module in the light base env is free.
"""

from __future__ import annotations

from strands_robots.simulation.ik import discover_ee_frame

__all__ = ["discover_ee_frame"]
