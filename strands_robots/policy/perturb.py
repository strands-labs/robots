"""Sim-to-real noise around any policy: noisy joint readings, noisy pixels, actions that land late."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ..types import Action, Observation
from .base import Policy, PolicyLike
from .callable import as_policy
from .requirements import Requirements


class Perturb(Policy):
    """Wrap a policy so it sees the world the way a real arm reports it.

    ``joint_noise`` adds N(0, sigma) to every joint position the policy reads (in the units it reads them),
    ``image_noise`` adds N(0, sigma * 255) to every pixel, and ``action_delay`` makes each action land that many
    control steps late; until the first one arrives the robot holds the pose it measured. The robot, the world
    and the recording see the true observation; only the wrapped policy sees the perturbed one.
    """

    def __init__(
        self,
        policy: PolicyLike | Any,
        *,
        joint_noise: float = 0.0,
        image_noise: float = 0.0,
        action_delay: int = 0,
        seed: int | None = None,
    ) -> None:
        if joint_noise < 0 or image_noise < 0 or action_delay < 0:
            raise ValueError("joint_noise, image_noise and action_delay must be >= 0")
        self.policy = as_policy(policy)
        self.units = getattr(self.policy, "units", None)
        self.joint_noise, self.image_noise, self.action_delay = (
            float(joint_noise),
            float(image_noise),
            int(action_delay),
        )
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.policy.reset()
        self._rng = np.random.default_rng(self.seed)
        self._queue: deque[Action] = deque()

    def act(self, obs: Observation, *, instruction: str | None = None) -> Action | None:
        seen = obs
        if self.joint_noise:
            seen = seen.with_joints(seen.joints.positions + self._rng.normal(0.0, self.joint_noise, len(seen.joints)))
        if self.image_noise and seen.images:
            images = {k: self._noisy(v) for k, v in seen.images.items()}
            seen = Observation(seen.t, seen.joints, seen.pose, images, seen.extras)
        action = self.policy.act(seen, instruction=instruction)
        if action is None:  # the wrapped policy holds (waiting for its server): a hold is not an action to delay
            return None
        self._queue.append(action)
        if len(self._queue) > self.action_delay:
            return self._queue.popleft()
        return np.array(obs.joints.positions)  # a hold: the measured pose, in the units the loop hands the policy

    def _noisy(self, img: np.ndarray) -> np.ndarray:
        out = img.astype(np.float32) + self._rng.normal(0.0, self.image_noise * 255.0, img.shape)
        return np.clip(out, 0, 255).astype(img.dtype)

    def requirements(self) -> Requirements:
        req = getattr(self.policy, "requirements", None)
        return req() if callable(req) else Requirements()

    def needs_observation(self, *, instruction: str | None = None) -> bool:
        needs = getattr(self.policy, "needs_observation", None)
        return bool(needs(instruction=instruction)) if callable(needs) else True

    def __getattr__(self, name: str) -> Any:  # attach/detach/hz/realtime: whatever the wrapped policy declares
        if name.startswith("_") or name == "policy":
            raise AttributeError(name)
        return getattr(self.policy, name)

    def __repr__(self) -> str:
        return (
            f"Perturb({self.policy!r}, joint_noise={self.joint_noise}, image_noise={self.image_noise}, "
            f"action_delay={self.action_delay})"
        )
