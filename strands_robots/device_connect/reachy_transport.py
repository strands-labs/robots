"""Shared transport helpers for Reachy Mini robots.

REST API helpers and pose math used by ReachyMiniDriver.
Real-time Zenoh I/O is handled by Device Connect's DriverTransport.
"""

import json
import logging
import math
import socket
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_host(host: str) -> str:
    """Resolve hostname to IP address."""
    try:
        return socket.gethostbyname(host)
    except socket.gaierror:
        return host


# ── REST API ─────────────────────────────────────────────────────

def api(host: str, port: int, path: str, method: str = "GET", data: Optional[dict] = None) -> dict:
    """Call Reachy Mini daemon REST API."""
    import urllib.error
    import urllib.request
    url = f"http://{host}:{port}{path}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    body = json.dumps(data).encode() if data else None
    try:
        with urllib.request.urlopen(req, body, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(), "code": e.code}
    except Exception as e:
        return {"error": str(e)}


# ── Pose math ────────────────────────────────────────────────────

def rpy_to_pose(pitch_deg: float, roll_deg: float, yaw_deg: float,
                x_mm: float = 0, y_mm: float = 0, z_mm: float = 0) -> list:
    """Convert RPY (degrees) + XYZ (mm) to 4x4 pose matrix."""
    p, r, y = math.radians(pitch_deg), math.radians(roll_deg), math.radians(yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return [
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, x_mm/1000],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, y_mm/1000],
        [-sp,   cp*sr,            cp*cr,             z_mm/1000],
        [0,     0,                0,                 1],
    ]


def identity_pose() -> list:
    """Return a 4x4 identity pose matrix."""
    return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
