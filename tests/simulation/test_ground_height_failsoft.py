"""Terrain-relative height predicates fail soft: a missing or throwing
``_ground_height_at`` hook degrades to flat ground (``0.0``), never raising.

``base_below_z`` / ``base_height`` measure a floating base's clearance above the
LOCAL ground beneath it (``base z - _ground_height_at(x, y)``), so a collapse on
a raised-terrain plateau is still detected instead of being silently missed by
an absolute-z test (#1233). The ground-height read is a best-effort convenience,
not a correctness invariant, so it is guarded three ways:

* ``SimEngine._ground_height_at`` defaults to ``0.0`` -- flat ground, and any
  backend that ships no heightfield (only the MuJoCo backend overrides it to
  sample a ``create_world(terrain=...)`` field).
* ``predicates._ground_height`` returns ``0.0`` when the sim exposes no hook at
  all, so a duck-typed engine keeps flat-ground behaviour.
* ``predicates._ground_height`` swallows a throwing hook and returns ``0.0``,
  so a predicate evaluated mid-rollout can never turn a heightfield sampling
  error into an episode-killing exception.

These tests pin that fail-soft contract. Without it, a backend that omits the
hook -- or one whose heightfield sampler raises -- would crash a routine
predicate check.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.predicates import _ground_height


class _MinimalSim(SimEngine):
    """Concrete ``SimEngine`` with only the abstract methods stubbed.

    Deliberately does NOT override ``_ground_height_at`` so the base-class
    flat-ground default is the code path under test.
    """

    def create_world(
        self,
        timestep: float | None = None,
        gravity: list[float] | None = None,
        ground_plane: bool = True,
        terrain: str | None = None,
        difficulty: float = 1.0,
    ) -> dict[str, Any]:
        return {"status": "success"}

    def destroy(self) -> dict[str, Any]:
        return {"status": "success"}

    def reset(self) -> dict[str, Any]:
        return {"status": "success"}

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        return {"status": "success"}

    def get_state(self) -> dict[str, Any]:
        return {"sim_time": 0.0, "step_count": 0}

    def add_robot(
        self,
        name: str,
        urdf_path: str | None = None,
        data_config: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        keyframe: str | int | None = None,
    ) -> dict[str, Any]:
        return {"status": "success"}

    def remove_robot(self, name: str) -> dict[str, Any]:
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return []

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return []

    def add_object(
        self,
        name: str,
        shape: str = "box",
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        size: list[float] | None = None,
        color: list[float] | None = None,
        mass: float = 0.1,
        is_static: bool | None = None,
        mesh_path: str | None = None,
        material: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {"status": "success"}

    def remove_object(self, name: str) -> dict[str, Any]:
        return {"status": "success"}

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        return {}

    def send_action(
        self,
        action: dict[str, Any] | Sequence[float],
        robot_name: str | None = None,
        n_substeps: int = 1,
    ) -> dict[str, Any]:
        return {"status": "success"}

    def render(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> dict[str, Any]:
        return {}


class _RaisingGroundSim(_MinimalSim):
    """A backend whose heightfield sampler always raises."""

    def _ground_height_at(self, x: float, y: float) -> float:
        raise RuntimeError("heightfield sampler exploded")


def test_simengine_ground_height_at_defaults_to_flat_ground() -> None:
    """The base ``SimEngine`` reports 0.0 everywhere -- a flat ground plane."""
    sim = _MinimalSim()
    assert sim._ground_height_at(0.0, 0.0) == 0.0
    assert sim._ground_height_at(12.5, -7.25) == 0.0


def test_ground_height_reads_the_backend_hook_when_present() -> None:
    """Positive control: the wrapper returns whatever the hook samples."""

    class _Hooked(_MinimalSim):
        def _ground_height_at(self, x: float, y: float) -> float:
            return 0.42

    assert _ground_height(_Hooked(), 1.0, 2.0) == 0.42


def test_ground_height_without_a_hook_is_flat_ground() -> None:
    """A duck-typed sim that exposes no hook degrades to flat ground (0.0)."""

    class _NoHook:
        """Not a SimEngine and has no ``_ground_height_at`` attribute."""

    # Deliberately a non-SimEngine object: the getattr(..., None) guard is the
    # branch under test, and every real SimEngine already carries the default.
    assert _ground_height(_NoHook(), 1.0, 2.0) == 0.0  # type: ignore[arg-type]


def test_ground_height_swallows_a_throwing_hook(caplog: Any) -> None:
    """A heightfield sampler that raises must not propagate: predicates never raise."""
    sim = _RaisingGroundSim()
    with caplog.at_level(logging.DEBUG, logger="strands_robots.simulation.predicates"):
        result = _ground_height(sim, 3.0, -1.0)
    assert result == 0.0
    assert any("_ground_height_at" in rec.getMessage() for rec in caplog.records)
