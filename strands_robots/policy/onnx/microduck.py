"""The microduck walking preset: Pollen's ``alpha_walking.onnx`` on the 14-joint biped.

Observation ``base_ang_vel(3) · projected_gravity(3) · joint_pos-default(14) · joint_vel(14) · last_action(14) ·
command``; decode ``default_pose + action · action_scale``. Joint names, default pose and scale come from the
graph's own metadata; the repo and file names from ``presets.json``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...types import Action, Observation
from .graph import CPU, OnnxPolicy, base_twist, projected_gravity
from .presets import preset_data


def _floats(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.split(",")], dtype=float)


class MicroduckPolicy(OnnxPolicy):
    """Biped locomotion. Command ``vx, vy`` in m/s and ``wz`` in rad/s; ``skill`` names a file in the preset."""

    def __init__(
        self,
        *,
        skill: str = "alpha_walking",
        vx: float = 0.0,
        vy: float = 0.0,
        wz: float = 0.0,
        providers: Sequence[str] = CPU,
    ) -> None:
        facts = preset_data(MicroduckPolicy)
        if skill not in facts["files"]:
            raise ValueError(f"microduck: unknown skill {skill!r}; known: {', '.join(sorted(facts['files']))}")
        name = facts["files"][skill]
        super().__init__(name, repo=facts["repo"], sha256=facts["sha256"].get(name), providers=providers)
        meta = self.metadata
        if "joint_names" not in meta or "default_joint_pos" not in meta:
            raise ValueError(f"{self.path.name} has no joint_names/default_joint_pos metadata: not a microduck export")
        self.joint_names: tuple[str, ...] = tuple(s.strip() for s in meta["joint_names"].split(","))
        self.default_pose = _floats(meta["default_joint_pos"])
        self.action_scale = float(meta.get("action_scale", "1.0"))
        n = len(self.joint_names)
        if self.default_pose.shape != (n,):
            raise ValueError(f"{self.path.name}: {self.default_pose.size} default angles for {n} joints")
        declared = self.input_width - (6 + 3 * n) if self.input_width > 0 else -1
        names = [s.strip() for s in meta.get("command_names", "twist").split(",")]
        widths: dict[str, int] = facts["command_widths"]
        width = declared if declared > 0 else sum(widths.get(c, 0) for c in names)
        self.command = np.zeros(width, dtype=float)
        self.command[:3] = (vx, vy, wz)
        self.last_action = np.zeros(n, dtype=float)

    def reset(self) -> None:
        self.last_action = np.zeros(len(self.joint_names), dtype=float)

    def observation(self, obs: Observation) -> np.ndarray:
        ang, quat = base_twist(obs, repr(self))
        q = obs.joints.positions
        if q.shape != self.default_pose.shape:
            raise ValueError(f"{self!r} drives {len(self.joint_names)} joints, the body has {q.size}")
        return np.concatenate(
            [ang, projected_gravity(quat), q - self.default_pose, obs.joints.velocities, self.last_action, self.command]
        ).astype(np.float32)

    def act(self, obs: Observation, *, instruction: str | None = None) -> Action:
        a = self.infer(self.observation(obs)).astype(float)
        self.last_action = a.copy()
        return self.default_pose + a * self.action_scale
