"""A small TF reader: latest transforms from /tf and /tf_static, composed along the frame tree.

No tf2_ros needed (it is often missing from a venv); good for "where is the hand in the base frame now".
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .bus import RosBus

TF_TYPE = "tf2_msgs/msg/TFMessage"


class TfReader:
    def __init__(self, bus: RosBus) -> None:
        self.bus = bus
        self.edges: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}  # child -> (parent, xyz, quat wxyz)
        bus.subscribe("/tf", TF_TYPE, qos="sensor", listener=self._on)
        bus.subscribe("/tf_static", TF_TYPE, qos="latched", listener=self._on)

    def _on(self, msg: Any) -> None:
        for t in getattr(msg, "transforms", ()):
            tr, r = t.transform.translation, t.transform.rotation
            self.edges[t.child_frame_id.lstrip("/")] = (
                t.header.frame_id.lstrip("/"),
                np.array([tr.x, tr.y, tr.z]),
                np.array([r.w, r.x, r.y, r.z]),
            )

    def lookup(self, target: str, source: str) -> dict[str, Any]:
        """Pose of ``source`` in ``target``: {"xyz": [...], "quat_wxyz": [...]}; KeyError if not connected."""
        target, source = target.lstrip("/"), source.lstrip("/")
        up_t, up_s = self._chain(target), self._chain(source)
        common = next((f for f, _ in up_s if f in dict(up_t)), None)
        if common is None:
            raise KeyError(f"no TF path between {target!r} and {source!r} (known frames: {sorted(self.frames())})")
        p_s, q_s = dict(up_s)[common]
        p_t, q_t = dict(up_t)[common]
        qi = _qinv(q_t)
        return {"xyz": (_rot(qi, p_s - p_t)).round(6).tolist(), "quat_wxyz": _qmul(qi, q_s).round(6).tolist()}

    def frames(self) -> set[str]:
        return set(self.edges) | {p for p, _, _ in self.edges.values()}

    def _chain(self, frame: str) -> list[tuple[str, tuple[np.ndarray, np.ndarray]]]:
        """[(ancestor, pose of ``frame`` in that ancestor)], starting with the frame itself."""
        p, q = np.zeros(3), np.array([1.0, 0, 0, 0])
        out: list[tuple[str, tuple[np.ndarray, np.ndarray]]] = [(frame, (p, q))]
        seen: set[str] = set()
        while frame in self.edges and frame not in seen:  # a cycle in a broken tree ends the walk
            seen.add(frame)
            parent, pt, qt = self.edges[frame]
            p, q = pt + _rot(qt, p), _qmul(qt, q)
            out.append((parent, (p, q)))
            frame = parent
        return out


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _qinv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]]) / float(np.dot(q, q))


def _rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _qmul(_qmul(q, np.array([0.0, *v])), _qinv(q))[1:]
