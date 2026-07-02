"""Bed-cover cloth model for Newton (VBD FEM triangle cloth).

Units: **centimetres** (gravity −981). Newton's own cloth examples (cloth_franka)
run VBD in cm scale for numerical conditioning; the GB10 spike reproduced that —
metre-scale VBD with bed-sized cloth is softer-conditioned, cm is the proven regime.
Masses are in kg, so forces come out in kg·cm/s² = 0.01 N ("centinewtons" ×10).

The parameter recipe matches the validated #2 "MuJoCo recipe" in spirit:
featherlight sheet (~0.5 kg), coarse grid, low bend stiffness so it drapes and
crumples, real self-collision so folds stack instead of interpenetrating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

# 1 N expressed in this unit system (kg · cm / s²)
N = 100.0


@dataclass
class BedsheetParams:
    """FEM bed-cover recipe (cm / kg)."""

    dim_x: int = 40
    dim_y: int = 32
    cell: float = 2.5                 # cm — 100 x 80 cm cover at the defaults
    # Per-particle mass in the VBD working regime. NOTE: this is the mass scale Newton's
    # own cm-tuned cloth recipe (cloth_franka) is conditioned around — a literal 0.5 kg
    # sheet (4e-4/particle) destabilises the same contact constants (eye-verified: the
    # cover jittered off the box during settle). The grasp question is governed by force
    # *ratios* (pinch force / sheet weight, friction capacity / drag load), so the
    # gripper derives its force budget FROM the sheet weight at the real-world ratio
    # (Inspire ~20 N pinch vs ~4.9 N sheet → 4.1×) instead of absolute newtons.
    particle_mass: float = 0.4
    # VBD FEM membrane (cm-scale recipe from Newton's cloth_franka example)
    tri_ke: float = 1.0e4
    tri_ka: float = 1.0e4
    tri_kd: float = 1.5e-6
    edge_ke: float = 5.0              # low bend → drapes/crumples like a cover
    edge_kd: float = 1.0e-2
    particle_radius: float = 0.8      # cm — cloth contact thickness the fingers feel
    # contact materials
    cloth_self_friction: float = 0.7   # also the particle side of the
                                       # geometric-mean rigid-contact mix
    soft_contact_ke: float = 1.0e4
    soft_contact_kd: float = 1.0e-2


def sheet_weight(params: BedsheetParams | None = None) -> float:
    """Total sheet weight in sim force units (mass · 981). The gripper's force budget
    is expressed as real-world multiples of this (see :class:`PinchParams`)."""
    p = params or BedsheetParams()
    return (p.dim_x + 1) * (p.dim_y + 1) * p.particle_mass * 981.0


def add_bedsheet(builder: newton.ModelBuilder, *, pos, rot=None, params: BedsheetParams | None = None,
                 pleat_x: float | None = None, pleat_halfwidth: float = 6.0,
                 pleat_amp: float = 5.0, pleat_gather: float = 0.45) -> dict:
    """Add the bed-cover cloth grid to ``builder`` at ``pos`` (cm). Returns a dict of
    grid metadata (particle index helpers, corner vertex ids) mirroring the
    issue-#2 ``Bedsheet`` handle so behaviour code can address corners/edges.

    ``pleat_x`` (local cm along the grid x) raises a **real pleat** — the strip's
    material is laterally gathered (× ``pleat_gather``) and lifted into a ridge of
    height ``pleat_amp``, so the fold has actual slack instead of elastic stretch
    (the same trick as the #2 accordion crumple: the robots crumple the cover so
    there are raised folds to pinch — a flat sheet on a mattress offers nothing to
    enclose, which is the whole grasp problem)."""
    p = params or BedsheetParams()
    base = builder.particle_count
    builder.add_cloth_grid(
        pos=wp.vec3(*pos),
        rot=rot if rot is not None else wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=p.dim_x, dim_y=p.dim_y, cell_x=p.cell, cell_y=p.cell,
        mass=p.particle_mass,
        tri_ke=p.tri_ke, tri_ka=p.tri_ka, tri_kd=p.tri_kd,
        edge_ke=p.edge_ke, edge_kd=p.edge_kd,
        particle_radius=p.particle_radius,
    )

    if pleat_x is not None:
        # gather the strip's x-material toward the pleat line and lift the slack
        import math
        for idx in range(base, builder.particle_count):
            q = builder.particle_q[idx]
            lx = q[0] - pos[0]
            d = lx - pleat_x
            if abs(d) < pleat_halfwidth:
                w = 0.5 * (1.0 + math.cos(math.pi * d / pleat_halfwidth))   # 1 at line, 0 at edge
                nlx = pleat_x + d * (1.0 - (1.0 - pleat_gather) * w)
                nz = q[2] + pleat_amp * w
                builder.particle_q[idx] = wp.vec3(pos[0] + nlx, q[1], nz)

    nx, ny = p.dim_x, p.dim_y

    def vid(i: int, j: int) -> int:
        # add_cloth_grid lays particles out row-major over (dim_x+1, dim_y+1)
        return base + i * (ny + 1) + j

    return {
        "base": base,
        "count": (nx + 1) * (ny + 1),
        "dim": (nx, ny),
        "size": (nx * p.cell, ny * p.cell),
        "vid": vid,
        "corner_vids": {
            "SW": vid(0, 0), "SE": vid(nx, 0), "NW": vid(0, ny), "NE": vid(nx, ny),
        },
        "params": p,
    }


def finalize_cloth_materials(model: newton.Model, params: BedsheetParams | None = None) -> None:
    """Apply the cloth contact-material recipe to a finalized model."""
    p = params or BedsheetParams()
    model.soft_contact_ke = p.soft_contact_ke
    model.soft_contact_kd = p.soft_contact_kd
    model.soft_contact_mu = p.cloth_self_friction


def cloth_points(state: newton.State, sheet: dict) -> np.ndarray:
    """Live (N,3) cloth particle positions for this sheet (cm, world frame)."""
    q = state.particle_q.numpy()
    return q[sheet["base"]: sheet["base"] + sheet["count"]]
