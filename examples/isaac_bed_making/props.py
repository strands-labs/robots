"""Rounded visual props for the Isaac bed-making scene.

Isaac Lab's ``CuboidCfg`` spawns an analytic, hard-edged USD cube — fine for the
mattress, too square for pillows. ``superellipsoid_mesh`` builds a smooth
rounded-box mesh (a superellipsoid) so the pillows read as real pillows. It is a
VISUAL prim only; pair it with a hidden box collider (a ``CuboidCfg`` with a
rest/contact offset that rounds its collision corners) so the cloth still has a
cheap, stable shape to drape over. See ``.isaac/polish_tune.py``.
"""

from __future__ import annotations

import math
from typing import Tuple

from pxr import Gf, UsdGeom, Vt


def _sgnpow(v: float, e: float) -> float:
    """Signed power |v|^e — the superellipsoid roundness exponent."""
    s = -1.0 if v < 0.0 else 1.0
    return s * (abs(v) ** e)


def superellipsoid_mesh(
    stage,
    path: str,
    *,
    size: Tuple[float, float, float],
    center: Tuple[float, float, float],
    color: Tuple[float, float, float] = (0.93, 0.93, 0.96),
    roundness: float = 0.35,
    res: int = 28,
    rot_y_deg: float = 0.0,
):
    """Author a smooth rounded-box (superellipsoid) ``UsdGeom.Mesh`` of full
    ``size`` centred at ``center``. ``roundness`` in (0,1]: 1.0 = ellipsoid,
    ~0.3 = boxy-with-rounded-edges (pillow). ``rot_y_deg`` tips the box about the
    y (bed-width) axis around its centre — used to prop the pillows up against the
    headboard. Visual only (no collision)."""
    a, b, c = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    e = roundness
    th = math.radians(rot_y_deg)
    ct, st = math.cos(th), math.sin(th)
    nlat, nlon = res, res * 2
    pts = []
    for i in range(nlat + 1):
        u = -math.pi / 2.0 + math.pi * i / nlat
        cu, su = _sgnpow(math.cos(u), e), _sgnpow(math.sin(u), e)
        for j in range(nlon + 1):
            v = -math.pi + 2.0 * math.pi * j / nlon
            cv, sv = _sgnpow(math.cos(v), e), _sgnpow(math.sin(v), e)
            lx, ly, lz = a * cu * cv, b * cu * sv, c * su  # local (centred) point
            rx, rz = lx * ct + lz * st, -lx * st + lz * ct  # rotate about +y
            pts.append(Gf.Vec3f(rx + center[0], ly + center[1], rz + center[2]))
    idx, counts = [], []

    def vid(i: int, j: int) -> int:
        return i * (nlon + 1) + j

    for i in range(nlat):
        for j in range(nlon):
            idx += [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
            counts.append(4)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray(pts))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    m.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    m.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)  # already smooth
    UsdGeom.Xformable(m).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    return m
