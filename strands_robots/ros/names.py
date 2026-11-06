"""Checks on what an agent or a profile names on a ROS graph, and the one install hint.

Agent input is untrusted: a topic, service or action name is ASCII graph-name characters only, a type is
``pkg/(msg|srv|action)/Name``, and a domain id is an int in 0..232 (RTPS ports overflow at 233). Each check
raises ``ValueError`` naming what was wrong, before anything touches rclpy or the process environment.
"""

from __future__ import annotations

import re
from numbers import Integral

_NAME_RE = re.compile(r"^/?[A-Za-z_~{][A-Za-z0-9_/~{}]*\Z")
_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*/(msg|srv|action)/[A-Z][A-Za-z0-9_]*\Z")
_SHORT_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*/[A-Z][A-Za-z0-9_]*\Z")

INSTALL_HINT = (
    "rclpy is not importable in this interpreter. It ships with a ROS 2 distro, not with pip: "
    "source one first (e.g. `source /opt/ros/jazzy/setup.bash`), "
    "and make a venv with --system-site-packages so rclpy and the robot's message packages are visible. "
    "pip install 'strands-robots[ros]' adds only the pure-Python helpers."
)


class RosUnavailable(ImportError):
    """rclpy (or a message package a profile needs) is not importable here."""


def canonical(name: str, *, what: str = "topic") -> str:
    """``name`` as an absolute ROS 2 graph name: leading ``/`` added, trailing ``/`` refused."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"{what} name must be a non-empty string, got {name!r}")
    full = name if name.startswith("/") else "/" + name
    if not _NAME_RE.match(full) or full.endswith("/") or "//" in full or "__" in full:
        raise ValueError(f"{what} name {name!r} is not a valid ROS 2 name (ASCII letters, digits, _ and /)")
    if any(tok[:1].isdigit() for tok in full.split("/") if tok):
        raise ValueError(f"{what} name {name!r}: a name token may not start with a digit")
    return full


def interface_type(t: str, *, kind: str = "msg") -> str:
    """``pkg/msg/Name`` (``pkg/Name`` is accepted and expanded with ``kind``); anything else is refused."""
    if not isinstance(t, str):
        raise ValueError(f"{kind} type must be a string like 'std_msgs/{kind}/String', got {t!r}")
    if _SHORT_TYPE_RE.match(t):
        pkg, name = t.split("/")
        t = f"{pkg}/{kind}/{name}"
    if not _TYPE_RE.match(t):
        raise ValueError(f"type {t!r} must look like 'pkg/{kind}/Name'")
    if t.split("/")[1] != kind:
        raise ValueError(f"type {t!r} is a {t.split('/')[1]}, this call needs a {kind}")
    return t


def domain_id(value: object) -> int:
    """A ROS_DOMAIN_ID: a plain int in 0..232. Bools and floats are refused, not coerced."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"domain_id must be an int in 0..232, got {value!r}")
    d = int(value)
    if not 0 <= d <= 232:
        raise ValueError(f"domain_id {d} is outside 0..232 (RTPS ports for 233+ pass 65535)")
    return d


def positive(value: object, what: str, *, integer: bool = False) -> float:
    """A finite positive number (or int) for timeouts, counts and rates."""
    import math

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{what} must be a positive number, got {value!r}")
    if integer and not isinstance(value, int):
        raise ValueError(f"{what} must be a positive integer, got {value!r}")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{what} must be positive and finite, got {value!r}")
    return value
