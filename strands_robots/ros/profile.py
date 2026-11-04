"""A ROS profile: everything robot-specific about talking to one robot's ROS graph, as data.

Topic names, message types, QoS, rates, the wire id of each model joint, what "armed" means, default
gains, the STOP sequence and extra motion surfaces. A renamed topic is a JSON change, never a code change.

Where a profile comes from, first hit wins: a ``.json`` path; a directory on ``STRANDS_ROBOTS_ROS_PROFILES``
(``os.pathsep``-separated); an installed package's ``strands_robots.ros_profiles`` entry point (named after
the profile, pointing at a JSON file, a directory of them, or a dict); the built-ins in ``profiles/``. Then the
user overlay ``~/.strands_robots/ros/<name>.json`` and ``overrides=`` are merged on top, key by key.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from .names import canonical, interface_type

_BUILTIN = Path(__file__).parent / "profiles"
_OVERLAY = Path("~/.strands_robots/ros")
_STREAMS = ("state", "command", "status", "odom", "imu", "shadow", "model", "gains_from", "telemetry")
ENTRY_POINTS = "strands_robots.ros_profiles"


@dataclass(frozen=True)
class Profile:
    name: str
    family: str
    data: Mapping[str, Any]

    def stream(self, key: str) -> Mapping[str, Any] | None:
        return self.data.get(key)

    def need(self, key: str) -> Mapping[str, Any]:
        s = self.data.get(key)
        if not s:
            raise KeyError(f"ROS profile {self.name!r} has no {key!r} stream")
        return s

    @property
    def joints(self) -> Mapping[str, Any]:
        return self.data.get("joints") or {}

    @property
    def cameras(self) -> Mapping[str, Mapping[str, Any]]:
        return self.data.get("cameras") or {}

    @property
    def discovery(self) -> Mapping[str, Any]:
        return self.data.get("discovery") or {}

    def wire_id(self, action_key: str) -> Any:
        """The wire id/name of a model joint; profile keys may drop the ``_joint`` suffix."""
        j = self.joints
        for k in (action_key, action_key.removesuffix("_joint"), action_key + "_joint"):
            if k in j:
                return j[k]
        raise KeyError(action_key)

    def gains(self, which: str, action_keys: tuple[str, ...]) -> list[float]:
        """Per-joint kp or kd in action_keys order from ``gains.<which>`` = {"default": x, "<joint>": y}."""
        g = (self.data.get("gains") or {}).get(which) or {}
        out = []
        for k in action_keys:
            v = g.get(k, g.get(k.removesuffix("_joint"), g.get("default")))
            if v is None:
                raise KeyError(f"ROS profile {self.name!r} gives no {which} for {k} and no default")
            out.append(float(v))
        return out

    @property
    def surfaces(self) -> tuple[str, ...]:
        return tuple(self.data.get("surfaces") or ())

    @property
    def stop(self) -> list[Mapping[str, Any]]:
        return list(self.data.get("stop") or ())


def _dirs() -> list[Path]:
    raw = os.environ.get("STRANDS_ROBOTS_ROS_PROFILES", "")
    return [Path(d).expanduser() for d in raw.split(os.pathsep) if d.strip()]


def _from_entry_point(name: str) -> dict[str, Any] | None:
    for ep in entry_points(group=ENTRY_POINTS):
        obj = ep.load()
        if isinstance(obj, Mapping):
            if ep.name == name:
                return dict(obj)
            continue
        p = Path(str(obj))
        f = p / f"{name}.json" if p.is_dir() else p
        if f.is_file() and f.stem == name:
            return json.loads(f.read_text())
    return None


def _names_from_entry_points() -> Iterator[str]:
    for ep in entry_points(group=ENTRY_POINTS):
        obj = ep.load()
        p = None if isinstance(obj, Mapping) else Path(str(obj))
        if p is not None and p.is_dir():
            yield from (f.stem for f in p.glob("*.json"))
        else:
            yield ep.name


def available() -> list[str]:
    """Every profile name this process can load (search dirs, entry points, built-ins)."""
    names = {p.stem for d in [*_dirs(), _BUILTIN] for p in d.glob("*.json")}
    return sorted(names | set(_names_from_entry_points()))


def _find(name: str) -> dict[str, Any] | None:
    for d in _dirs():
        if (d / f"{name}.json").is_file():
            return json.loads((d / f"{name}.json").read_text())
    data = _from_entry_point(name)
    if data is None and (_BUILTIN / f"{name}.json").is_file():
        data = json.loads((_BUILTIN / f"{name}.json").read_text())
    return data


def load(name_or_path: str | Path, *, overrides: Mapping[str, Any] | None = None) -> Profile:
    """A profile by name or JSON file path, plus the user overlay and ``overrides``."""
    p = Path(name_or_path)
    if p.suffix == ".json":
        data, name = json.loads(p.expanduser().read_text()), p.stem
    else:
        name = str(name_or_path)
        data = _find(name)
    home = _OVERLAY.expanduser() / f"{name}.json"
    if home.is_file():
        data = _merge(data or {}, json.loads(home.read_text()))
    if not data:
        raise KeyError(
            f"no ROS profile {name!r}; known: {available()} (pass a .json path, add a directory to "
            f"STRANDS_ROBOTS_ROS_PROFILES, or install a package with a {ENTRY_POINTS} entry point)"
        )
    if overrides:
        data = _merge(data, dict(overrides))
    return validate(Profile(name, str(data.get("family", "")), data))


def motion_surfaces() -> tuple[str, ...]:
    """Every topic a known profile commands, stops through or lists as a surface: gated for raw writes."""
    out: list[str] = []
    for name in available():
        try:
            prof = load(name)
        except (KeyError, ValueError, OSError):
            continue
        out += [*prof.surfaces, *(st["topic"] for st in prof.stop)]
        if prof.stream("command"):
            out.append(prof.need("command")["topic"])
    return tuple(dict.fromkeys(out))


def validate(prof: Profile) -> Profile:
    """Refuse a malformed profile at load, not on the first tick."""
    if not prof.family and not prof.data.get("codec"):
        raise ValueError(f"ROS profile {prof.name!r} names no family (or codec) for its messages")
    for key in _STREAMS:
        s = prof.data.get(key)
        if s is None:
            continue
        if "topic" in s:
            canonical(s["topic"])
        if "type" in s:
            interface_type(s["type"])
    for cam in prof.cameras.values():
        canonical(cam["topic"])
        interface_type(cam["type"])
    for st in prof.stop:
        canonical(st["topic"])
        interface_type(st["type"])
    if "state" not in prof.data:
        raise ValueError(f"ROS profile {prof.name!r} has no state stream")
    return prof


def _merge(base: Mapping[str, Any], top: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in top.items():
        out[k] = _merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), Mapping) else v
    return out
