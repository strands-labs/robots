"""A LeRobot checkpoint run in this process, through the exact loader ``lerobot-policy-server`` uses."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ...types import Observation
from ...units import Units
from ...units.lerobot import norm_hint
from ..requirements import Requirements
from .contract import Contract, lerobot, server_config
from .quiet import quiet_load

if TYPE_CHECKING:
    from ...description import Description


class LeRobotPolicy:
    """``act_chunk`` = LeRobot's ``PolicyRunner.predict`` on one frame; wrap in ``ChunkedPolicy`` to step it.

    ``Policy("lerobot/smolvla_base")`` does that wrapping. The checkpoint, its saved processors and its
    validation come from ``lerobot.remote_inference.deployment.load_deployment``, so a local run and a
    served run of the same checkpoint are the same computation. A checkpoint does not record its norm
    mode, so on a real body ``norm=`` or ``semantics=`` must say it; a simulated body takes the registry's.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str | None = None,
        revision: str | None = None,
        semantics: str | None = None,
        norm: Mapping[str, str] | None = None,
        fps: float = 30.0,
        n_action_steps: int | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self._kw: dict[str, Any] = dict(
            device=device, revision=revision, semantics=semantics, norm=norm, fps=fps, n_action_steps=n_action_steps
        )
        self.hz = fps
        self.contract: Contract | None = None
        self._runner: Any = None
        self._task = ""
        self._task_version = 0

    def attach(self, robot: Any) -> None:
        if self.contract is not None and self.contract.description is robot.description:
            return
        if robot.is_real and not self._kw["semantics"] and not self._kw["norm"]:
            raise ValueError(  # a checkpoint stores no norm mode: on a real body the registry's default is a guess
                f"{self!r} does not say which LeRobot norm mode its data was recorded in, and {robot.id} is a real "
                f"body: pass {norm_hint(robot.description)} (LeRobot's SO arms default to degrees since 0.4.4, "
                "range_m100_100 before) or semantics= (see units/semantics.json)"
            )
        lerobot()
        from lerobot.remote_inference.deployment import load_deployment

        cfg = server_config(robot.description, self.checkpoint, **self._kw)
        self.contract = Contract(robot.description, tuple(cfg.features), cfg.action_feature, cfg.semantics)
        self.contract.check()
        with quiet_load():
            self._runner, self.identity = load_deployment(cfg)

    def _need(self) -> Contract:
        if self.contract is None:
            raise RuntimeError(f"{self!r} is not attached to a robot; run it with robot.run(policy, ...)")
        return self.contract

    def units_for(self, description: Description) -> Units | None:
        return self._need().units()

    def requirements(self) -> Requirements:
        return self._need().requirements() if self.contract is not None else Requirements()

    def reset(self) -> None:
        if self._runner is not None:
            self._runner.reset(full=True)

    def act_chunk(self, obs: Observation, *, instruction: str | None = None) -> np.ndarray:
        from lerobot.inference import ObservationSnapshot

        c = self._need()
        task = instruction or ""
        if task != self._task:
            self._task, self._task_version = task, self._task_version + 1
        snap = ObservationSnapshot(c.frame(obs), time.monotonic(), task, self._task_version, f"obs-{obs.t:.6f}")
        chunk = self._runner.predict(snap)
        return c.to_keys_chunk(chunk.canonical_actions.detach().cpu().numpy())

    def __repr__(self) -> str:
        return f"Policy({self.checkpoint!r})"
