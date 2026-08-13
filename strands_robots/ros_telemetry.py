"""Shared rclpy publisher for robot telemetry on a ROS 2 domain.

Both the simulation bridge (:class:`strands_robots.simulation.ros_bridge.SimRosBridge`)
and the hardware bridge (:class:`strands_robots.hardware_ros_bridge.HardwareRosBridge`)
advertise the same per-robot topics so a real arm and its digital twin look
identical on the ROS 2 graph:

* ``/<robot>/joint_states`` (``sensor_msgs/msg/JointState``) - joint names and
  positions.
* ``/<robot>/<camera>/image_raw`` (``sensor_msgs/msg/Image``, ``rgb8``) - one
  message per camera frame.

The ROS 2 *wire contract* - topic names, name sanitization, and inbound
``joint_command`` parsing - lives in the transport-agnostic
:class:`RosTelemetryBase`, so the rclpy and pure-RTPS
(:class:`strands_robots.hardware_rtps_bridge.HardwareRtpsBridge`) transports
are byte-identical on the graph by construction, not by two independent
codepaths happening to agree. :class:`RosTelemetryBridge` adds the rclpy
machinery on top - node ownership, per-robot publisher caching, and message
construction - so the sim and hardware bridges are thin, symmetric subclasses
that differ only in their default node name. ``rclpy`` and the ROS 2 message
packages are optional, system-provided dependencies (they are not on PyPI);
they are imported lazily through
:func:`strands_robots.utils.require_optional`, so importing this module - and
running with the bridge disabled - never requires ROS 2.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from strands_robots.utils import dds_domain_id_error, finite_number_error, require_optional

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


#: Env var an operator sets to explicitly run an INBOUND ``joint_command``
#: surface on an unsecured DDS graph (no ``dds_security_config``). Truthy values
#: mirror the mesh insecure opt-out (``STRANDS_MESH_I_KNOW_THIS_IS_INSECURE``):
#: ``1``, ``true``, ``yes`` (case-insensitive). This is a deliberate second
#: factor so a forgotten config cannot silently expose a drivable arm.
ROS2_INSECURE_ENV = "STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE"

#: Keys a ``dds_security_config`` dict must supply (non-empty). Each names a
#: credential the RTPS bridge wires into its DDS Security participant: the
#: identity CA, the participant's own certificate and private key (identity is
#: unprovable without the key), and the signed governance + permissions
#: documents. ``permissions_ca`` is optional and applied when present.
_DDS_SECURITY_REQUIRED_KEYS = (
    "identity_ca",
    "certificate",
    "private_key",
    "governance",
    "permissions",
)


class RosTelemetryBase:
    """Transport-agnostic ROS 2 wire contract shared by every telemetry bridge.

    The rclpy bridges (:class:`RosTelemetryBridge` and its
    :class:`~strands_robots.hardware_ros_bridge.HardwareRosBridge` /
    :class:`~strands_robots.simulation.ros_bridge.SimRosBridge` subclasses) and
    the pure-RTPS bridge
    (:class:`~strands_robots.hardware_rtps_bridge.HardwareRtpsBridge`) all derive
    from this base, so the ROS 2 topic names and the inbound ``joint_command``
    parsing live in exactly one place. That makes the two transports
    byte-identical on the ROS 2 graph by construction rather than relying on two
    independent codepaths staying in sync as the contract evolves.

    This base pulls in no transport library; each subclass owns its own
    rclpy / cyclonedds machinery and message construction.
    """

    @staticmethod
    def _safe(name: str) -> str:
        """Map a robot/camera name to a valid ROS 2 topic segment."""
        return "".join(c if (c.isalnum() or c == "_") else "_" for c in name).strip("_") or "robot"

    @staticmethod
    def _resolve_robot_name(robot: Any) -> str:
        """Namespace topics under the bound robot's name.

        Prefers the lerobot device ``.name`` (e.g. ``so101``), falling back to
        the strands tool name, so ``joint_command`` and ``joint_states`` share
        one namespace - a controller can echo our published joint names straight
        back to drive the arm.
        """
        inner = getattr(robot, "robot", None)
        return getattr(inner, "name", None) or getattr(robot, "tool_name_str", None) or "robot"

    @classmethod
    def joint_states_topic(cls, robot: str) -> str:
        """ROS 2 topic a robot's ``JointState`` telemetry is published on."""
        return f"/{cls._safe(robot)}/joint_states"

    @classmethod
    def image_topic(cls, robot: str, camera: str) -> str:
        """ROS 2 topic a robot camera's ``Image`` frames are published on."""
        return f"/{cls._safe(robot)}/{cls._safe(camera)}/image_raw"

    @classmethod
    def joint_command_topic(cls, robot: str) -> str:
        """ROS 2 topic inbound ``joint_command`` messages are read from."""
        return f"/{cls._safe(robot)}/joint_command"

    # -- security / safety gate (shared by both hardware bridges) ---------

    @staticmethod
    def _validate_joint_limits(
        joint_limits: dict[str, Any] | None,
    ) -> dict[str, tuple[float, float]] | None:
        """Validate and normalize a ``joint_limits`` mapping at construction.

        Returns ``{motor: (min, max)}`` with floats and ``min <= max``, or
        ``None`` when no limits are configured. Failing fast here (rather than
        per-command) means a malformed bound surfaces at bridge construction,
        not as a silent mid-run rejection of every command.

        Raises:
            ValueError: If ``joint_limits`` is not a mapping of name to a
                ``(min, max)`` pair of finite numbers with ``min <= max``.
        """
        if joint_limits is None:
            return None
        if not isinstance(joint_limits, dict):
            raise ValueError(f"joint_limits must be a dict[str, (min, max)], got {type(joint_limits).__name__}")
        normalized: dict[str, tuple[float, float]] = {}
        for name, bounds in joint_limits.items():
            try:
                low, high = bounds
            except (TypeError, ValueError):
                raise ValueError(f"joint_limits[{name!r}] must be a (min, max) numeric pair, got {bounds!r}") from None
            # Each bound goes through the shared signed-scalar domain before the
            # ordering comparison below, which cannot see a non-finite bound:
            # every comparison against ``nan`` is False, so ``(low, nan)`` passes
            # ``low > high`` and then makes the ``low <= pos <= high`` test in
            # :meth:`_command_action` False for every position - the bridge drops
            # EVERY inbound command for that joint, which is the silent mid-run
            # rejection this validator exists to surface at construction.
            # ``(inf, inf)`` and ``(-inf, -inf)`` pass the ordering check the same
            # way and admit nothing either. The domain also answers for an int
            # past the float64 range, which raised ``OverflowError`` out of the
            # ``float()`` below rather than the documented ``ValueError``. The
            # sibling declaration of this parameter,
            # :class:`~strands_robots.simulation.isaac.delta_eef.IsaacDeltaEEFController`,
            # refuses a non-finite bound for the same reason.
            for side, bound in (("min", low), ("max", high)):
                if error := finite_number_error(bound, side, f"joint_limits[{name!r}]"):
                    raise ValueError(error)
            low, high = float(low), float(high)
            if low > high:
                raise ValueError(f"joint_limits[{name!r}] has min {low} > max {high}")
            normalized[str(name)] = (low, high)
        return normalized

    @staticmethod
    def _validate_dds_security_config(config: Any) -> dict[str, str]:
        """Validate a ``dds_security_config`` dict supplies every required key.

        Returns the config unchanged on success. The required keys
        (:data:`_DDS_SECURITY_REQUIRED_KEYS`) are the credentials a DDS Security
        participant cannot authenticate or be governed without; each must be a
        non-empty string (a path, ``file:`` / ``data:`` URI per the DDS Security
        spec). Validated at construction so a half-filled config refuses the
        bridge rather than silently degrading wire security.

        Raises:
            ValueError: If ``config`` is not a dict or a required key is missing
                or empty.
        """
        if not isinstance(config, dict):
            raise ValueError(f"dds_security_config must be a dict, got {type(config).__name__}")
        missing = [k for k in _DDS_SECURITY_REQUIRED_KEYS if not str(config.get(k, "")).strip()]
        if missing:
            raise ValueError(
                f"dds_security_config is missing required keys: {missing}. "
                f"All of {list(_DDS_SECURITY_REQUIRED_KEYS)} must be supplied "
                "(identity CA, participant certificate + private key, governance, permissions)."
            )
        return config

    @staticmethod
    def _insecure_opt_out() -> bool:
        """True when the operator explicitly accepted an unsecured command surface.

        Mirrors the mesh insecure opt-out contract: ``1`` / ``true`` / ``yes``
        (case-insensitive) on :data:`ROS2_INSECURE_ENV`.
        """
        return os.getenv(ROS2_INSECURE_ENV, "").strip().lower() in ("1", "true", "yes")

    @classmethod
    def _require_secure_command_surface(
        cls,
        *,
        enable_commands: bool,
        dds_security_config: Any,
    ) -> None:
        """Refuse to expose an inbound command surface on an unsecured DDS graph.

        An inbound ``joint_command`` subscription lets any participant on the
        DDS domain drive the physical arm. When ``enable_commands`` is in
        effect, require either a ``dds_security_config`` (DDS Security: signed
        governance + permissions, authenticated identities) OR an explicit
        operator opt-out via :data:`ROS2_INSECURE_ENV` (``=1``). A telemetry-only
        bridge (``enable_commands`` False) is publish-only and is not gated.

        Raises:
            ValueError: When commands are enabled with neither a security config
                nor the explicit insecure opt-out.
        """
        if not enable_commands:
            return
        if dds_security_config:
            return
        if cls._insecure_opt_out():
            logger.warning(
                "%s: exposing an inbound joint_command surface on an UNSECURED DDS graph "
                "(%s set). Any participant on the domain can drive the arm. Provide a "
                "dds_security_config for production.",
                cls.__name__,
                ROS2_INSECURE_ENV,
            )
            return
        raise ValueError(
            "Refusing to start an inbound joint_command surface on an unsecured DDS graph. "
            "An enabled command bridge lets any DDS participant drive the physical arm. "
            "Pass a dds_security_config (identity CA, participant certificate + private key, "
            f"governance, permissions) or set {ROS2_INSECURE_ENV}=1 to explicitly accept the risk."
        )

    def _command_action(
        self,
        msg: Any,
        *,
        skip_empty: bool = False,
        joint_limits: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, float] | None:
        """Parse an inbound ``joint_command`` ``JointState`` into an action dict.

        Returns ``{motor: pos}`` ready for ``Robot.send_action``, or ``None``
        when the message must be ignored. With ``skip_empty=True`` a wholly
        empty sample (a DDS dispose / keep-alive, which is not a real actuation
        request) is dropped silently; an empty or length-mismatched message is
        otherwise rejected with a warning rather than partially applied, so a
        malformed command never drives the arm to a surprising pose. A position
        that is not a readable number rejects the message the same way - it is
        reported and returns ``None`` rather than raising, so the failure costs
        exactly the one message on either transport.

        When ``joint_limits`` is supplied (``{motor: (min, max)}``), the command
        is range-checked against the declared bounds: if ANY commanded joint
        falls outside its range the ENTIRE command is rejected (returns
        ``None``) - no partial application - so one out-of-range joint can never
        drive part of the arm to a surprising pose while the rest holds. Joints
        without a declared bound are not constrained.

        Args:
            msg: The inbound ``JointState``-like message (``name``/``position``).
            skip_empty: Drop a wholly empty sample silently (DDS keep-alive).
            joint_limits: Optional ``{motor: (min, max)}`` clamp ranges; a
                command with any joint outside its range is rejected whole.
        """
        names = list(getattr(msg, "name", []) or [])
        positions = list(getattr(msg, "position", []) or [])
        if skip_empty and not names and not positions:
            return None
        if not names or len(names) != len(positions):
            logger.warning(
                "%s: ignoring joint_command with name/position length mismatch (%d vs %d)",
                type(self).__name__,
                len(names),
                len(positions),
            )
            return None
        action: dict[str, float] = {}
        for name, pos in zip(names, positions):
            try:
                action[name] = float(pos)
            except (TypeError, ValueError):
                # A position this cannot read is a malformed command, not a
                # reason to raise: this method's contract is an action dict or
                # None. Refusing the message WHOLE matches the length-mismatch
                # and out-of-range branches - no partial application - and it
                # keeps the failure inside the parser, where it costs exactly
                # the one message. Escaping instead reached each transport's
                # loop tolerance, and the cyclonedds loop has already taken a
                # batch by then, so the samples behind this one were dropped
                # with it. Finiteness is deliberately not checked here: a nan
                # position is a readable number that ``send_action`` refuses
                # naming the joint.
                logger.warning(
                    "%s: ignoring joint_command with a non-numeric position for %r: %r",
                    type(self).__name__,
                    name,
                    pos,
                )
                return None
        if joint_limits:
            for name, pos in action.items():
                bounds = joint_limits.get(name)
                if bounds is None:
                    continue
                low, high = bounds
                if not (low <= pos <= high):
                    logger.warning(
                        "%s: rejecting joint_command - %s=%.4f outside declared range "
                        "[%.4f, %.4f] (whole command dropped, no partial application)",
                        type(self).__name__,
                        name,
                        pos,
                        low,
                        high,
                    )
                    return None
        return action

    def _drive_from_command(self, robot: Any, msg: Any, *, skip_empty: bool = False) -> None:
        """Forward an inbound ``joint_command`` to ``robot.send_action``.

        Shared by both hardware bridges: parse via :meth:`_command_action`, then
        dispatch the flat ``{motor.pos: float}`` action, surfacing (never
        raising) a ``send_action`` failure so a bad command cannot kill the
        command loop.
        """
        action = self._command_action(msg, skip_empty=skip_empty, joint_limits=getattr(self, "_joint_limits", None))
        if action is None:
            return
        try:
            result = robot.send_action(action)
        except Exception:
            logger.warning("%s: send_action raised on joint_command; arm not moved", type(self).__name__, exc_info=True)
            return
        if isinstance(result, dict) and result.get("status") == "error":
            logger.warning("%s: send_action rejected joint_command: %s", type(self).__name__, result)


class RosTelemetryBridge(RosTelemetryBase):
    """A thin rclpy publisher for per-robot joint state and camera frames.

    Subclassed by :class:`SimRosBridge` and :class:`HardwareRosBridge`, which
    set a distinguishing default ``node_name`` but share every publish path so
    a simulated robot and a real one are byte-for-byte identical on the wire.
    Topic naming and the inbound command contract come from
    :class:`RosTelemetryBase`, which the pure-RTPS bridge also derives from, so
    the rclpy and cyclonedds transports advertise the same topics.

    Args:
        domain_id: ROS 2 domain (``ROS_DOMAIN_ID``) the bridge publishes on.
            Only an ``int`` in ``[0, 232]`` names a domain: RTPS derives its
            discovery ports from it, and 233 lands past the end of the port space.
        node_name: Name of the internal rclpy node.
        qos_depth: Depth of the publishers' KEEP_LAST history.

    Raises:
        ValueError: If ``domain_id`` is outside ``[0, 232]``. Checked before the
            process-wide ``ROS_DOMAIN_ID`` write, so a refused domain leaves the
            environment as it found it.
        ImportError: When ``rclpy`` / the ROS 2 message packages are not
            importable, with an install hint (system ROS 2 or the docker image).
    """

    #: Default rclpy node name; subclasses override to identify their source.
    default_node_name = "strands_robots"

    def __init__(self, domain_id: int = 0, node_name: str | None = None, qos_depth: int = 10) -> None:
        # Refuse a domain id outside the RTPS port map BEFORE writing it. The
        # write below is process-wide and lands ahead of the rclpy import, so an
        # unusable value would otherwise outlive this call and steer every later
        # participant - a value the transport is never given the chance to reject.
        if error := dds_domain_id_error(domain_id, "domain_id", type(self).__name__):
            raise ValueError(error)

        # Pin the domain before rclpy reads it. Set it unconditionally so the
        # bridge publishes where the caller asked, not where the shell happened
        # to point.
        os.environ["ROS_DOMAIN_ID"] = str(domain_id)

        rclpy_mod: Any = require_optional(
            "rclpy", extra="ros2", purpose="the ROS 2 telemetry bridge (ros2_bridge=True)"
        )
        sensor_msgs: Any = require_optional(
            "sensor_msgs.msg", pip_install="ros-<distro>-sensor-msgs", purpose="the ROS 2 telemetry bridge"
        )
        self._rclpy = rclpy_mod
        self._JointState = sensor_msgs.JointState
        self._Image = sensor_msgs.Image

        self._owns_context = not self._rclpy.ok()
        if self._owns_context:
            self._rclpy.init()
        self._node = self._rclpy.create_node(node_name or self.default_node_name)
        self._qos_depth = qos_depth
        self._joint_pubs: dict[str, Any] = {}
        self._image_pubs: dict[str, Any] = {}

    def _now(self) -> Any:
        return self._node.get_clock().now().to_msg()

    def _joint_publisher(self, robot: str) -> Any:
        pub = self._joint_pubs.get(robot)
        if pub is None:
            pub = self._node.create_publisher(self._JointState, self.joint_states_topic(robot), self._qos_depth)
            self._joint_pubs[robot] = pub
        return pub

    def _image_publisher(self, robot: str, camera: str) -> Any:
        key = f"{robot}/{camera}"
        pub = self._image_pubs.get(key)
        if pub is None:
            pub = self._node.create_publisher(self._Image, self.image_topic(robot, camera), self._qos_depth)
            self._image_pubs[key] = pub
        return pub

    def publish_joint_states(self, robot: str, names: list[str], positions: list[float]) -> None:
        """Publish one ``JointState`` for ``robot`` on ``/<robot>/joint_states``."""
        msg = self._JointState()
        msg.header.stamp = self._now()
        msg.header.frame_id = self._safe(robot)
        msg.name = list(names)
        msg.position = [float(p) for p in positions]
        self._joint_publisher(robot).publish(msg)

    def publish_image(self, robot: str, camera: str, image: np.ndarray) -> None:
        """Publish one RGB ``Image`` on ``/<robot>/<camera>/image_raw``.

        Args:
            robot: Robot name (topic namespace).
            camera: Camera name (topic sub-namespace).
            image: ``(H, W, 3)`` uint8 RGB frame.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            return
        height, width = int(image.shape[0]), int(image.shape[1])
        msg = self._Image()
        msg.header.stamp = self._now()
        msg.header.frame_id = f"{self._safe(robot)}/{self._safe(camera)}"
        msg.height = height
        msg.width = width
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = width * 3
        msg.data = image.astype("uint8", copy=False).tobytes()
        self._image_publisher(robot, camera).publish(msg)

    def shutdown(self) -> None:
        """Destroy the node and, if this bridge initialized rclpy, shut it down."""
        node = getattr(self, "_node", None)
        if node is not None:
            try:
                node.destroy_node()
            finally:
                self._node = None
        if getattr(self, "_owns_context", False) and self._rclpy.ok():
            self._rclpy.shutdown()
            self._owns_context = False
