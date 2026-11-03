"""What a robot *is*, independent of any world: its model file, joints, actuators, limits, keyframes.

The registry (``registry/robots.json`` + ``registry/urdf_robots.json``) maps a
name or alias to an asset shipped by ``robot_descriptions`` (MuJoCo Menagerie and
friends). ``Description.load("so101")`` resolves the file; the joint/actuator
facts are read from the model itself — via MuJoCo when it is installed, via a
plain XML walk when it is not — so a hardware-only install still knows its
``action_keys``. Facts a world or a bus needs (its driver, its LeRobot norm
modes, which action keys are the gripper) are registry data, read into the typed
``Hardware`` and ``Gripper`` records.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .introspect import introspect
from .registry import _NAME_RE, normalize_name, registry
from .resolve import _resolve_asset

_SCENE_CANDIDATES = ("scene.xml", "scene_mjx.xml")
_FILES = (".xml", ".urdf", ".mjcf")


def default_id(name: str | Path) -> str:
    """The id a body gets when none is given: the name as the caller wrote it (``"g1"``, not ``"unitree_g1"``)."""
    return Path(name).stem if isinstance(name, Path) or name.endswith(_FILES) else normalize_name(name)


@dataclass(frozen=True)
class Hardware:
    """How a robot is driven on a bus (registry ``hardware`` block)."""

    driver: str | None = None
    lerobot_type: str | None = None
    lerobot_norm: Mapping[str, str] | None = None  # action key (or "*") -> LeRobot MotorNormMode value
    safe: Mapping[str, tuple[float, float]] | None = None  # action key -> soft bound inside the hard limit
    on_disconnect: str | None = None  # what every control does when the command path lapses
    position_hold: bool | None = None  # the bus keeps its last goal under torque (None: ask the driver)


@dataclass(frozen=True)
class Gripper:
    """A robot's gripper: the labelled action keys it drives, and which end is closed/open."""

    keys: tuple[str, ...]
    closed: str
    open: str


@dataclass(frozen=True)
class Description:
    """A robot model resolved to a file plus the facts a world or a bus needs about it."""

    name: str
    model_path: Path
    format: str  # "mjcf" | "urdf"
    category: str = "unknown"
    device_type: str | None = None  # registry ``device.type``: robot_arm, humanoid, quadruped, …
    joints: tuple[str, ...] = ()  # hinge/slide joints, model order, labelled
    actuators: tuple[str, ...] = ()  # actuator names, model order
    action_keys: tuple[str, ...] = ()  # the joint each actuator drives, labelled, actuator order
    limits: Mapping[str, tuple[float, float]] = field(default_factory=dict)  # joint -> (lo, hi) in rad
    keyframes: tuple[str, ...] = ()
    floating: bool = False
    joint_labels: Mapping[str, str] = field(default_factory=dict)  # model joint name -> label
    spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hardware: Hardware = field(default_factory=Hardware)
    gripper: Gripper | None = None
    scene_path: Path | None = None
    _unlabel: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_unlabel", {v: k for k, v in self.joint_labels.items()})

    @property
    def package_dir(self) -> Path:
        return self.model_path.parent

    @property
    def dof(self) -> int:
        return len(self.action_keys)

    def label(self, model_joint: str) -> str:
        return self.joint_labels.get(model_joint, model_joint)

    def unlabel(self, label: str) -> str:
        return self._unlabel.get(label, label)

    def describe(self) -> dict[str, Any]:
        """JSON-safe summary (what an agent tool and the dashboard read)."""
        return {
            "name": self.name,
            "category": self.category,
            "format": self.format,
            "model": str(self.model_path),
            "dof": self.dof,
            "action_keys": list(self.action_keys),
            "limits": {k: [float(a), float(b)] for k, (a, b) in self.limits.items()},
            "keyframes": list(self.keyframes),
            "floating": self.floating,
        }

    @classmethod
    def load(cls, name: str | Path) -> Description:
        """Resolve a registry name, alias, ``robot_descriptions`` name or MJCF/URDF path."""
        if isinstance(name, Path) or name.endswith(_FILES):
            path = Path(name).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"robot model file not found: {path}")
            fmt = "urdf" if path.suffix == ".urdf" else "mjcf"
            entry: dict[str, Any] = {"name": path.stem, "category": "file"}
        else:
            norm = normalize_name(name)
            if not _NAME_RE.match(norm):
                raise ValueError(f"robot name {name!r} may only contain letters, digits, '_' and '.'")
            entry = registry().get(norm)  # type: ignore[assignment]
            path, fmt, entry = _resolve_asset(norm, entry)
        return cls._from_file(path, fmt, entry)

    @classmethod
    def _from_file(cls, path: Path, fmt: str, entry: dict[str, Any]) -> Description:
        labels_raw = entry.get("joint_labels") or {}
        scene = None
        if fmt == "mjcf":
            asset = entry.get("asset") or {}
            root = path.parents[len(Path(asset["model_xml"]).parts) - 1] if "model_xml" in asset else path.parent
            for cand in (path.parent / c for c in _SCENE_CANDIDATES):
                if cand.is_file():
                    scene = cand
                    break
            if scene is None and asset.get("scene_xml") and (root / asset["scene_xml"]).is_file():
                scene = root / asset["scene_xml"]
        facts = introspect(path, fmt, scene)
        labels = {j: labels_raw[j] for j in facts["joints"] if j in labels_raw}

        def lab(j: str) -> str:
            return labels.get(j, j)

        sp = entry.get("spawn_position") or (0.0, 0.0, 0.0)
        hw = entry.get("hardware") or {}
        g = entry.get("gripper")
        return cls(
            name=entry.get("name", path.stem),
            model_path=path,
            format=fmt,
            category=entry.get("category", "unknown"),
            joints=tuple(lab(j) for j in facts["joints"]),
            actuators=tuple(facts["actuators"]),
            action_keys=tuple(lab(j) for j in facts["action_joints"]),
            limits={lab(j): lim for j, lim in facts["limits"].items()},
            keyframes=tuple(facts["keyframes"]),
            floating=facts["floating"],
            joint_labels=labels,
            spawn_position=(float(sp[0]), float(sp[1]), float(sp[2])),
            device_type=(entry.get("device") or {}).get("type"),
            hardware=Hardware(
                hw.get("driver"),
                hw.get("lerobot_type"),
                hw.get("lerobot_norm"),
                {k: (float(a), float(b)) for k, (a, b) in (hw.get("safe") or {}).items()} or None,
                hw.get("on_disconnect"),
                hw.get("position_hold"),
            ),
            gripper=Gripper(tuple(g["keys"]), g["closed"], g["open"]) if g else None,
            scene_path=scene,
        )
