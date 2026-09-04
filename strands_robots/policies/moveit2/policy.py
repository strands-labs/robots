"""MoveIt2Policy - service-mode :class:`Policy` client backed by ROS 2 / MoveIt2.

The policy reads a goal from the well-known ``**kwargs`` keys defined in
issue #300 (``target_pose``, ``target_joints``, ``world_update``), forwards
the request to a sidecar ROS 2 node via ZMQ + msgpack, and unpacks the
returned joint trajectory into the per-step action dicts that
:class:`~strands_robots.robot.Robot` consumes.

Construction mirrors :class:`~strands_robots.policies.groot.policy.Gr00tPolicy`'s
service mode:

.. code-block:: python

    from strands_robots.policies import create_policy

    policy = create_policy(
        "moveit2",
        host="127.0.0.1",
        port=5556,
        planning_group="arm",
    )

    actions = policy.get_actions_sync(
        observation_dict={"observation.state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        instruction="reach for the red block",
        target_pose=[0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
    )

The ROS 2 / ``moveit_py`` deps stay out of ``pyproject.toml`` - only the
client side (``pyzmq``, ``msgpack``) is installed via the ``[moveit2]`` extra.
See :mod:`strands_robots.policies.moveit2.server` for the sidecar reference
implementation, and the ``docker-compose.yml`` beside it for the recommended
deployment - a compose file is not an importable module, so it is named the way
that package's own docstring names it rather than dressed as a :mod: target.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

from strands_robots.policies._log_safety import sanitize_log_value
from strands_robots.policies.base import Policy
from strands_robots.utils import name_list_error, tcp_port_error

from .client import MoveIt2InferenceClient

logger = logging.getLogger(__name__)


# Joint-name allowlist regex - matches the same pattern used by
# ``mesh.security.validate_command`` for ``target_joints`` keys, so a
# value the mesh accepts can flow end-to-end without a second
# allowlist mismatch.
_JOINT_NAME_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]*\Z"


class MoveIt2Policy(Policy):
    """ZMQ + msgpack client for the MoveIt2 sidecar.

    The policy is intentionally thin - all motion-planning state lives in
    the sidecar. This keeps the Python process free of ROS 2 deps and lets
    a single sidecar serve multiple agent processes.

    Args:
        host: Sidecar hostname. Default ``"127.0.0.1"`` (loopback only -
            users opt into network exposure).
        port: Sidecar port, an ``int`` in ``[1, 65535]``. A value outside
            the range is refused rather than interpolated into
            ``tcp://<host>:<port>``.
        planning_group: Default MoveIt2 planning-group name. Per-call
            ``planning_group`` kwargs override this.
        timeout_ms: ZMQ socket timeout (send + recv) in milliseconds.
        api_token: Optional token included in every request. Falls back
            to the ``MOVEIT2_API_TOKEN`` environment variable if not
            provided.
        **kwargs: Forward-compatibility absorber for the smart-string
            resolution path (e.g. ``zmq://host:port`` extras the factory
            adds). Per the #300 contract, providers MUST ignore unknown
            kwargs rather than raising.

    Examples:
        Direct construction::

            from strands_robots.policies.moveit2 import MoveIt2Policy

            policy = MoveIt2Policy(host="127.0.0.1", port=5556)

        Via the registry::

            from strands_robots.policies import create_policy

            policy = create_policy("moveit2", host="127.0.0.1", port=5556)
            policy = create_policy("moveit", port=5556)  # alias
            policy = create_policy("zmq://127.0.0.1:5556", planning_group="arm")
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        planning_group: str = "arm",
        timeout_ms: int = 15000,
        api_token: str | None = None,
        **kwargs: Any,
    ) -> None:
        # ``port`` addresses the moveit_py sidecar this client dials, so a
        # value that cannot name one is refused before it reaches
        # ``tcp://<host>:<port>``. The domain is the shared one the sibling
        # transports use, so the same port cannot be refused onto a service by
        # one and accepted by the next.
        if (port_error := tcp_port_error(port, "port", type(self).__name__)) is not None:
            raise ValueError(port_error)
        self.host = host
        self.port = port
        self.planning_group = planning_group
        self._robot_state_keys: list[str] = []

        resolved_token = api_token or os.environ.get("MOVEIT2_API_TOKEN")
        self._client: MoveIt2InferenceClient = MoveIt2InferenceClient(
            host=host,
            port=port,
            timeout_ms=timeout_ms,
            api_token=resolved_token,
        )

        # Per the #300 contract: silently ignore unknown kwargs. The
        # smart-string resolver (``zmq://...``) and ``register_policy``
        # may fan extra kwargs through this constructor; the policy
        # only needs ``host`` / ``port`` / ``planning_group`` / etc.
        if kwargs:
            logger.debug(
                "MoveIt2Policy ignoring unknown constructor kwargs: %s",
                sorted(kwargs.keys()),
            )

        logger.info(
            "MoveIt2Policy ready [host=%s port=%d planning_group=%s]",
            host,
            port,
            planning_group,
        )

    # Policy interface

    @property
    def provider_name(self) -> str:
        """Registry key for this provider (``"moveit2"``)."""
        return "moveit2"

    @property
    def requires_images(self) -> bool:
        """MoveIt2 plans from joint state + collision world, never images.

        Returning ``False`` lets the simulation skip camera rendering for
        this provider - same throughput optimisation
        :class:`~strands_robots.policies.mock.MockPolicy` exposes.
        """
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Configure the joint names this policy emits actions for.

        Used to map the ``trajectory`` rows the sidecar returns
        (``[t, q0, q1, ...]``) onto per-joint action dicts. When unset,
        ``get_actions`` falls back to ``observation.state`` length and
        emits ``"joint_<i>"`` keys.

        Raises:
            ValueError: If ``robot_state_keys`` is not an ordered list of
                distinct non-blank names, per
                :func:`~strands_robots.utils.name_list_error`. A single name
                passed as a bare string is the mistake this catches: ``str`` is
                iterable per character, so it would bind one joint per letter.
        """
        if robot_state_keys and (
            error := name_list_error(robot_state_keys, "robot_state_keys", "set_robot_state_keys")
        ):
            raise ValueError(error)
        self._robot_state_keys = list(robot_state_keys)

    def reset(self, seed: int | None = None) -> None:
        """Reset per-episode state on the sidecar.

        Forwards to the server's ``reset`` endpoint so any plan caches /
        partial trajectories from the previous episode are discarded. The
        ``seed`` argument is passed through for reproducibility on
        randomised samplers (RRT-Connect, KPIECE) that the sidecar may
        expose.

        Best-effort - any failure (server doesn't expose ``reset``,
        endpoint raises, network timeout) is logged and swallowed. Eval
        correctness is preserved even when reset is a no-op (the next
        ``plan`` call re-derives state from ``joint_state``).
        """
        try:
            payload: dict[str, Any] = {}
            if seed is not None:
                payload = {"options": {"seed": int(seed)}}
            self._client.call_endpoint("reset", payload if payload else None)
            logger.debug("MoveIt2Policy.reset: forwarded to server (seed=%r)", seed)
        except Exception as e:  # noqa: BLE001 - reset is best-effort
            logger.info(
                "MoveIt2Policy.reset: server did not accept reset (seed=%r): %s; "
                "continuing without per-episode server-side reset",
                seed,
                e,
            )

    async def get_actions(
        self,
        observation_dict: dict[str, Any],
        instruction: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Plan a trajectory and unpack it into per-step action dicts.

        Reads the goal from ``**kwargs``. Exactly one of ``target_pose``
        or ``target_joints`` must be provided; if both are set,
        ``target_joints`` wins (matches the MoveIt2 ``setJointValueTarget``
        precedence). If neither is provided, raises ``ValueError`` rather
        than returning a no-op trajectory - the issue #300 contract says
        unknown kwargs are silently ignored, but a missing goal is a
        caller bug, not an unknown extension.

        Args:
            observation_dict: Robot observation. ``observation.state`` is
                forwarded as the ``joint_state`` start configuration when
                present; otherwise the sidecar uses its own latest state.
                The natural-language ``instruction`` is unused (planner
                providers consume goals through structured kwargs).
            instruction: Natural-language instruction. Ignored.
            **kwargs: Well-known goal payload from #300:

                * ``target_pose`` (``list[float]``):
                  ``[x, y, z, qw, qx, qy, qz]`` in the planning group's
                  base frame.
                * ``target_joints`` (``dict[str, float]``): joint-space
                  goal keyed by joint name (radians / metres).
                * ``world_update`` (``dict | None``): per-call world
                  refresh for collision-aware planning.
                * ``planning_group`` (``str``): override the default
                  planning group for this call.

                Unknown kwargs are silently ignored.

        Returns:
            List of action dicts; one entry per trajectory waypoint (the
            time column from the sidecar is dropped - :class:`Robot`
            consumes per-step joint targets, the runner schedules the
            timing).

        Raises:
            ValueError: If neither ``target_pose`` nor ``target_joints``
                is provided.
            RuntimeError: If the sidecar returns ``success=False`` or an
                ``error`` field.
        """
        target_pose = kwargs.get("target_pose")
        target_joints = kwargs.get("target_joints")
        world_update = kwargs.get("world_update")
        planning_group = kwargs.get("planning_group", self.planning_group)

        if target_pose is None and target_joints is None:
            raise ValueError(
                "MoveIt2Policy.get_actions requires at least one of "
                "target_pose=[x,y,z,qw,qx,qy,qz] or target_joints={joint:value}. "
                "These are the well-known kwargs from issue #300; the "
                "natural-language `instruction` is ignored by motion "
                "planners."
            )

        # Validate target_joints keys to give the same defence-in-depth
        # the mesh.security path applies. Sidecar should validate too,
        # but a clear ValueError on the client saves a network round-trip
        # and keeps malformed data out of the ROS 2 process.
        if target_joints is not None:
            self._validate_target_joints(target_joints)

        # Validate target_pose shape - 7 floats (x, y, z + quaternion).
        if target_pose is not None:
            self._validate_target_pose(target_pose)

        # Validate planning_group name - prevent shell-meta / XML traversal
        # if the sidecar interpolates this into a parameter file.
        self._validate_planning_group(planning_group)

        joint_state = self._extract_joint_state(observation_dict)

        response = self._client.plan(
            joint_state=joint_state,
            planning_group=planning_group,
            target_pose=list(target_pose) if target_pose is not None else None,
            target_joints=dict(target_joints) if target_joints is not None else None,
            world_update=world_update,
        )

        if not response.get("success", False):
            status = response.get("status", "unknown")
            raise RuntimeError(
                f"MoveIt2 planning failed: status={status!r}, "
                f"target_pose={target_pose!r}, target_joints={target_joints!r}, "
                f"planning_group={planning_group!r}"
            )

        trajectory = response.get("trajectory", [])
        return self._unpack_trajectory(trajectory)

    # Helpers

    def _extract_joint_state(self, observation_dict: dict[str, Any]) -> list[float] | None:
        """Pull ``observation.state`` out of the observation dict.

        Accepts list / tuple / numpy array; returns a plain Python list of
        floats so msgpack serialises without numpy support on the wire.
        """
        state = observation_dict.get("observation.state")
        if state is None:
            return None
        try:
            # ``tolist`` for numpy arrays; ``list(map(float, ...))`` for
            # lists / tuples; both produce JSON-shaped output.
            if hasattr(state, "tolist"):
                state = state.tolist()
            return [float(x) for x in state]
        except (TypeError, ValueError) as e:
            logger.warning(
                "MoveIt2Policy: failed to extract joint_state from observation.state=%s (%s); "
                "letting sidecar use its own state estimate",
                sanitize_log_value(repr(state)),
                sanitize_log_value(e),
            )
            return None

    def _unpack_trajectory(self, trajectory: list[list[float]]) -> list[dict[str, Any]]:
        """Convert ``[[t, q0, q1, ...], ...]`` rows into per-step action dicts.

        The leading time column is dropped - the runner schedules the
        timing. If ``set_robot_state_keys`` was called, joint names come
        from there; otherwise we emit ``"joint_<i>"`` keys derived from
        the row width.
        """
        if not trajectory:
            return []

        actions: list[dict[str, Any]] = []
        for row in trajectory:
            if not row:
                continue
            # Drop the leading time column. ``len(row) >= 2`` is enforced
            # implicitly: if the row only has the time column the per-
            # joint slice below is empty and we emit an empty dict, which
            # the runner skips.
            joint_values = list(row[1:])
            keys = self._resolve_joint_keys(len(joint_values))
            actions.append({k: float(v) for k, v in zip(keys, joint_values)})
        return actions

    def _resolve_joint_keys(self, n: int) -> list[str]:
        """Resolve the joint key names for an n-element trajectory row.

        If ``set_robot_state_keys`` was called with a matching length,
        use those names; otherwise fall back to positional ``joint_<i>``
        labels (consistent with :class:`MockPolicy`).
        """
        if self._robot_state_keys and len(self._robot_state_keys) == n:
            return list(self._robot_state_keys)
        return [f"joint_{i}" for i in range(n)]

    @staticmethod
    def _validate_target_pose(target_pose: Any) -> None:
        """Validate ``target_pose`` is a 7-element list of finite floats."""
        try:
            poses = list(target_pose)
        except TypeError as e:
            raise ValueError(f"target_pose must be a 7-element list, got {type(target_pose).__name__}") from e
        if len(poses) != 7:
            raise ValueError(f"target_pose must have exactly 7 elements [x,y,z,qw,qx,qy,qz], got {len(poses)}")
        for i, v in enumerate(poses):
            try:
                f = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"target_pose[{i}] must be a number, got {type(v).__name__}") from e
            # NaN / +inf / -inf would crash the sidecar planner with an
            # opaque ROS error. Reject up front.
            if math.isnan(f) or math.isinf(f):
                raise ValueError(f"target_pose[{i}]={f!r} must be finite")

    @staticmethod
    def _validate_target_joints(target_joints: Any) -> None:
        """Validate ``target_joints`` is a name->finite-float mapping."""
        import re

        if not isinstance(target_joints, dict):
            raise ValueError(f"target_joints must be a dict[str, float], got {type(target_joints).__name__}")
        pattern = re.compile(_JOINT_NAME_PATTERN)
        for k, v in target_joints.items():
            if not isinstance(k, str) or not pattern.match(k):
                raise ValueError(
                    f"target_joints key {k!r} must match {_JOINT_NAME_PATTERN!r} (letters, digits, underscore, hyphen)"
                )
            try:
                f = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"target_joints[{k!r}]={v!r} must be a number") from e
            if math.isnan(f) or math.isinf(f):
                raise ValueError(f"target_joints[{k!r}]={f!r} must be finite")

    @staticmethod
    def _validate_planning_group(planning_group: Any) -> None:
        """Validate ``planning_group`` is a short identifier."""
        import re

        if not isinstance(planning_group, str):
            raise ValueError(f"planning_group must be a str, got {type(planning_group).__name__}")
        # Same charset as joint names - matches the MoveIt2 group naming
        # conventions documented at https://moveit.picknik.ai/.
        if not re.match(r"^[A-Za-z][A-Za-z0-9_-]{0,63}\Z", planning_group):
            raise ValueError(
                f"planning_group {planning_group!r} must match "
                "'^[A-Za-z][A-Za-z0-9_-]{0,63}\\Z' (letters, digits, "
                "underscore, hyphen; max 64 chars)"
            )


__all__ = ["MoveIt2Policy"]
