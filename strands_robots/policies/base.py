"""Abstract base class for robot policies (VLA, motion planners, MPC, scripted).

The :class:`Policy` ABC is intentionally agnostic about *how* actions are
produced.  Built-in providers (`mock`, `groot`, `lerobot_local`) are VLA-style,
but the same interface is the right shape for:

* **Classical motion planners** — cuRobo, MoveIt2, OMPL, RRT*: take a goal
  pose and joint state, return a collision-free trajectory.
* **Model-predictive controllers** (MPC) — solve a finite-horizon optimal
  control problem each tick.
* **Scripted / pure-IK trajectories** — analytic IK followed by interpolation;
  zero learning involved.

Non-VLA implementations typically set :attr:`Policy.requires_images` to
``False`` to skip camera rendering (~10x throughput win at 500Hz) and read
their goal from the well-known ``**kwargs`` keys documented on
:meth:`Policy.get_actions` rather than parsing the natural-language
``instruction`` string.

See ``MockPolicy`` (``strands_robots/policies/mock.py``) for the canonical
non-VLA reference implementation.
"""

import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from typing import Any


class Policy(ABC):
    """Abstract base class for robot policies (VLA, motion planners, MPC, scripted).

    All policies implement async :meth:`get_actions`.  For convenience, a
    synchronous wrapper :meth:`get_actions_sync` is provided.

    The interface is general enough to cover both **VLA-style** providers
    (consume images + instruction, output joint targets) and **non-VLA**
    providers such as classical motion planners (cuRobo, MoveIt2),
    model-predictive controllers, and pure-IK / scripted trajectories.
    Non-VLA providers typically set :attr:`requires_images` to ``False``
    and read their goal from the well-known ``**kwargs`` keys documented
    on :meth:`get_actions`.
    """

    @abstractmethod
    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Get actions from policy given observation and instruction.

        Args:
            observation_dict: Robot observation (cameras + state).  VLA
                providers consume both ``observation.images.*`` and
                ``observation.state``.  Non-VLA providers typically
                consume ``observation.state`` only and set
                :attr:`requires_images` to ``False`` to skip camera
                rendering.
            instruction: Natural language instruction.  Required by the
                signature for VLA providers; non-VLA providers (motion
                planners, MPC, scripted) may ignore it and read the goal
                from ``**kwargs`` instead.
            **kwargs: Provider-specific parameters.  The following keys
                are **well-known** and SHOULD be honoured by non-VLA
                providers when present so callers don't have to JSON-encode
                goals into the ``instruction`` string:

                - ``target_pose: list[float]`` — Cartesian goal as
                  ``[x, y, z, qw, qx, qy, qz]`` (position in metres,
                  orientation as a unit quaternion in the robot base frame).
                - ``target_joints: dict[str, float]`` — joint-space goal
                  keyed by joint name; values are in radians (revolute) or
                  metres (prismatic).
                - ``world_update: dict | None`` — per-call world refresh
                  for collision-aware planners (e.g. point cloud / depth
                  image / mesh updates).  ``None`` means "reuse the world
                  configured at init time".

                Providers MUST ignore unknown ``**kwargs`` rather than
                raising, so callers can pass shared keys across providers.

        Returns:
            List of action dicts for robot execution.  Each dict maps
            robot state key (joint name) to the target value for that
            tick; consumers typically execute the list at a fixed
            control rate (e.g. 50Hz).
        """
        pass

    def get_actions_sync(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Synchronous convenience wrapper around get_actions().

        Safe to call from sync code, event loops, or notebooks.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self.get_actions(observation_dict, instruction, **kwargs),
                ).result()
        else:
            return asyncio.run(self.get_actions(observation_dict, instruction, **kwargs))

    @abstractmethod
    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Configure the policy with robot state keys."""
        pass

    def reset(self, seed: int | None = None) -> None:
        """Reset per-episode policy state.

        Default implementation is a no-op. Policies that hold per-episode
        state (e.g. diffusion sampler RNG, action chunk caches, KV-caches)
        should override to apply the reset.

        For SERVICE-mode policies (e.g. ``Gr00tPolicy(host=...)`` over
        ZMQ), the override forwards the call to the server so its
        per-episode RNG state can be re-initialised — without this,
        ``set_eval_seed`` only seeds the client-side process, leaving
        the server's diffusion sampler RNG drifting across calls and
        breaking reproducibility (#187).

        Args:
            seed: Optional master seed forwarded to the policy's
                random-number generators. When ``None``, implementations
                may apply a default seed or leave RNG state untouched.
        """
        # Default no-op. Concrete policies override to apply per-episode
        # state reset (RNG seeding, action-cache flush, server-side
        # reset endpoint call, etc.).
        return None

    @property
    def requires_images(self) -> bool:
        """Whether this policy needs camera frames in its observation.

        Default ``True`` (most VLA policies do). Subclasses that only
        consume joint state (e.g. ``MockPolicy``, classical motion planners
        such as cuRobo / MoveIt2, MPC, pure-IK controllers, scripted
        trajectories) can return ``False`` to let the simulation skip
        expensive camera rendering — a ~10x throughput win at 500Hz when
        no cameras are needed.
        """
        return True

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get provider name for identification."""
        pass
