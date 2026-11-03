"""The holosoma preset: a 29-DoF humanoid standing and walking policy (FastSAC or PPO export).

Terms are concatenated in alphabetical order, as upstream does; a two-foot gait clock freezes at (pi, pi) while
the command is ~zero. The PD gains the checkpoint was trained with (``kp``/``kd`` metadata) are ``policy.gains``,
which ``Robot.run`` installs once before step one. Layout, scales and default pose come from ``presets.json``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np

from ...types import Action, Observation
from .graph import CPU, OnnxPolicy, base_twist, projected_gravity
from .presets import preset_data


class HolosomaPolicy(OnnxPolicy):
    """Humanoid locomotion. Command ``vx, vy`` in m/s and ``wz`` in rad/s; zero command = stand."""

    def __init__(
        self,
        *,
        algorithm: str = "fastsac",
        vx: float = 0.0,
        vy: float = 0.0,
        wz: float = 0.0,
        providers: Sequence[str] = CPU,
    ) -> None:
        f = preset_data(HolosomaPolicy)
        if algorithm not in f["files"]:
            raise ValueError(f"holosoma: algorithm must be one of {', '.join(sorted(f['files']))}, got {algorithm!r}")
        name = f["files"][algorithm]
        super().__init__(name, repo=f["repo"], sha256=f["sha256"].get(name), providers=providers)
        n = int(f["joints"])
        try:
            self.joint_names: tuple[str, ...] = tuple(json.loads(self.metadata["dof_names"]))
            kp = np.array(json.loads(self.metadata["kp"]), float)
            kd = np.array(json.loads(self.metadata["kd"]), float)
        except KeyError as exc:
            raise ValueError(f"{name} lacks the {exc} metadata the holosoma preset reads") from exc
        if len(self.joint_names) != n or kp.shape != (n,) or kd.shape != (n,):
            raise ValueError(f"{name}: expected {n} joints/gains, got {len(self.joint_names)}/{kp.shape}/{kd.shape}")
        self.gains: tuple[np.ndarray, np.ndarray] = (kp, kd)
        self.default_pose = np.array(f["default_pose"], dtype=float)
        self.scales: dict[str, float] = f["scales"]
        self.terms: tuple[tuple[str, int], ...] = tuple((t, int(w)) for t, w in f["terms"])
        self.action_scale, self.action_clip = float(f["action_scale"]), float(f["action_clip"])
        self.gait_period, self.stand_eps = float(f["gait_period"]), float(f["stand_eps"])
        self.command_lin = np.array([vx, vy], dtype=float)
        self.command_ang = float(wz)
        self.slices: dict[str, slice] = {}
        start = 0
        for term, width in self.terms:
            self.slices[term] = slice(start, start + width)
            start += width
        if self.input_width > 0 and start != self.input_width:
            raise ValueError(f"{name} takes {self.input_width} inputs, the preset layout is {start}")
        self.reset()

    @property
    def phase_dt(self) -> float:
        return 2.0 * np.pi / (self.hz * self.gait_period)

    def reset(self) -> None:
        self.last_action = np.zeros(len(self.joint_names), dtype=float)
        self.phase, self._standing = np.array([0.0, np.pi]), False

    def _step_clock(self) -> None:
        advanced = self.phase + self.phase_dt
        self.phase = np.fmod(advanced + np.pi, 2.0 * np.pi) - np.pi
        if float(np.linalg.norm(self.command_lin)) < self.stand_eps and abs(self.command_ang) < self.stand_eps:
            self.phase, self._standing = np.array([np.pi, np.pi]), True
        elif self._standing:
            self.phase, self._standing = np.array([0.0, np.pi]), False

    def observation(self, obs: Observation) -> np.ndarray:
        ang, quat = base_twist(obs, repr(self))
        q, dq = obs.joints.positions, obs.joints.velocities
        if q.shape != self.default_pose.shape:
            raise ValueError(f"{self!r} drives {self.default_pose.size} joints, the body has {q.size}")
        self._step_clock()
        blocks = {
            "actions": self.last_action,
            "base_ang_vel": ang * self.scales["base_ang_vel"],
            "command_ang_vel": np.array([self.command_ang]),
            "command_lin_vel": self.command_lin,
            "cos_phase": np.cos(self.phase),
            "dof_pos": q - self.default_pose,
            "dof_vel": dq * self.scales["dof_vel"],
            "projected_gravity": projected_gravity(quat),
            "sin_phase": np.sin(self.phase),
        }
        return np.concatenate([blocks[t] for t, _ in self.terms]).astype(np.float32)

    def act(self, obs: Observation, *, instruction: str | None = None) -> Action:
        raw = self.infer(self.observation(obs)).astype(float)
        a: np.ndarray = np.clip(raw, -self.action_clip, self.action_clip)
        self.last_action = a.copy()
        return self.default_pose + self.action_scale * a
