# Copyright Strands Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SimulationBackend ABC — shared contract for all physics backends.

This module defines the abstract interface that all simulation backends
(MuJoCo, Isaac Sim, Newton) must implement. It guarantees policy
interoperability: any ``Policy`` can run on any ``SimulationBackend``
with zero code changes.

Design principles:

* **Zero heavy dependencies** — imports only ``abc`` and ``typing``.
  Backends import this without triggering cross-backend dependencies.
* **Core intersection only** — defines the 10 methods shared by all
  backends. Backend-specific features (Newton's ``add_cloth``,
  Isaac's ``add_terrain``, MuJoCo's ``randomize``) stay as extensions.
* **Flexible signatures** — shared parameters are explicit, divergent
  ones use ``**kwargs`` so backends can extend without breaking the
  contract.

Return format convention::

    {"status": "success"|"error", "content": [{"text": "..."}]}

See Also:
    :class:`~strands_robots.policies.Policy` — the companion ABC for
    VLA/WFM policy providers.

Example::

    from strands_robots.simulation_backend import SimulationBackend

    class MyCustomBackend(SimulationBackend):
        def create_world(self, gravity=None, ground_plane=True, **kw):
            ...
        # ... implement all abstract methods ...

    backend = MyCustomBackend()
    backend.create_world(gravity=[0, 0, -9.81])
    backend.add_robot("panda")
    backend.run_policy("panda", "mock", "wave hello")
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class SimulationBackend(ABC):
    """Abstract base class for simulation backends.

    All backends must implement these 10 methods to guarantee that:

    1. An AI agent can create worlds, add robots, and run policies
       through a uniform interface.
    2. The ``factory.Robot()`` function can return any backend
       transparently.
    3. A ``Policy`` trained in one backend can execute in another.

    Backends are free to add extra methods (e.g. ``add_cloth``,
    ``run_diffsim``, ``add_terrain``) — those are extensions, not
    part of the core contract.
    """

    # ── World lifecycle ─────────────────────────────────────────────

    @abstractmethod
    def create_world(
        self,
        gravity: Any = None,
        ground_plane: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a new simulation world.

        Args:
            gravity: Gravity vector. Accepts ``[x, y, z]`` list, scalar
                (applied to z-axis), or ``None`` for default ``-9.81``.
            ground_plane: Whether to add a ground plane.
            **kwargs: Backend-specific options (e.g. ``timestep``,
                ``up_axis``).

        Returns:
            ``{"status": "success"|"error", "content": [...]}``.
        """
        ...

    @abstractmethod
    def destroy(self) -> Dict[str, Any]:
        """Destroy the simulation and release all resources.

        Returns:
            ``{"status": "success"|"error", "content": [...]}``.
        """
        ...

    @abstractmethod
    def reset(self, **kwargs: Any) -> Dict[str, Any]:
        """Reset the simulation to its initial state.

        Args:
            **kwargs: Backend-specific options (e.g. ``env_ids`` for
                GPU backends that support per-environment reset).

        Returns:
            ``{"status": "success"|"error", "content": [...]}``.
        """
        ...

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the simulation.

        Returns a summary of robots, objects, physics parameters,
        and simulation time.

        Returns:
            ``{"status": "success"|"error", "content": [...]}``.
        """
        ...

    # ── Robot management ────────────────────────────────────────────

    @abstractmethod
    def add_robot(
        self,
        name: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Add a robot to the simulation.

        Args:
            name: Unique robot identifier.
            **kwargs: Backend-specific options. Common keys include
                ``urdf_path``, ``usd_path``, ``data_config``,
                ``position``, ``orientation``, ``scale``.

        Returns:
            ``{"status": "success"|"error", "content": [...]}``.
        """
        ...

    # ── Simulation loop ─────────────────────────────────────────────

    @abstractmethod
    def step(self, **kwargs: Any) -> Dict[str, Any]:
        """Advance the simulation by one or more steps.

        Args:
            **kwargs: Backend-specific options. MuJoCo uses ``n_steps``;
                GPU backends accept ``actions`` (tensor).

        Returns:
            ``{"status": "success"|"error", "content": [...]}``.
        """
        ...

    # ── Observation ─────────────────────────────────────────────────

    @abstractmethod
    def get_observation(
        self,
        robot_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get observations from the simulation.

        Returns joint positions (and optionally velocities, images)
        in a format compatible with the ``Policy`` ABC's
        ``get_actions()`` input.

        Args:
            robot_name: Which robot to observe. ``None`` for default.
            **kwargs: Backend-specific options (e.g. ``camera_name``).

        Returns:
            Observation dict. Format varies by backend but always
            includes joint-name → value mappings.
        """
        ...

    # ── Rendering ───────────────────────────────────────────────────

    @abstractmethod
    def render(
        self,
        camera_name: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Render a camera view.

        Args:
            camera_name: Camera to render from (default backend camera).
            width: Image width in pixels.
            height: Image height in pixels.
            **kwargs: Backend-specific options.

        Returns:
            ``{"status": "success"|"error", "content": [...]}``.
            On success, ``content`` includes an image block.
        """
        ...

    # ── Policy execution ────────────────────────────────────────────

    @abstractmethod
    def run_policy(
        self,
        robot_name: str,
        policy_provider: str = "mock",
        instruction: str = "",
        duration: float = 10.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a strands-robots Policy on this backend.

        Uses the ``Policy`` ABC — any provider (GR00T, LeRobot,
        DreamGen, mock, etc.) works on any backend.

        Args:
            robot_name: Target robot.
            policy_provider: Provider name or model path.
            instruction: Natural language instruction for the policy.
            duration: Max execution time in seconds.
            **kwargs: Extra args forwarded to ``create_policy()``.

        Returns:
            ``{"status": "success"|"error", "content": [...]}``.
        """
        ...

    # ── Recording ───────────────────────────────────────────────────

    @abstractmethod
    def record_video(self, **kwargs: Any) -> Dict[str, Any]:
        """Record a video of the simulation.

        Args:
            **kwargs: Backend-specific options (e.g. ``duration``,
                ``camera_name``, ``fps``, ``output_path``).

        Returns:
            ``{"status": "success"|"error", "content": [...]}``.
        """
        ...
