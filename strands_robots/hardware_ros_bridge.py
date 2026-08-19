"""Publish hardware-robot telemetry on a ROS 2 domain - and drive it from one.

When a :class:`strands_robots.hardware_robot.Robot` is constructed with
``ros2_bridge=True``, it owns a :class:`HardwareRosBridge` that makes the real
arm a first-class ROS 2 participant in **both** directions:

* **publish** (outbound) - the arm's live observation is advertised as
  ``/<robot>/joint_states`` (``sensor_msgs/msg/JointState``) and, per camera,
  ``/<robot>/<camera>/image_raw`` (``sensor_msgs/msg/Image``, ``rgb8``). This is
  driven by the control loop (and ``publish_ros_observation`` on demand) and is
  inherited unchanged from :class:`~strands_robots.ros_telemetry.RosTelemetryBridge`.
* **subscribe** (inbound) - a ``/<robot>/joint_command``
  (``sensor_msgs/msg/JointState``) listener forwards each message straight into
  ``robot.send_action(...)`` using the flat ``{motor.pos: float}`` contract the
  hardware ``Robot`` already speaks, so an external ROS 2 stack (a teleop node,
  MoveIt, a trajectory replayer) can **drive** the physical arm. A daemon thread
  spins the node so commands are serviced concurrently with publishing - true
  full duplex.

:class:`HardwareRosBridge` is the hardware half of a symmetric pair: the
simulation half is :class:`strands_robots.simulation.ros_bridge.SimRosBridge`.
Both subclass :class:`~strands_robots.ros_telemetry.RosTelemetryBridge` and emit
identical *telemetry* topics, so a real arm and its digital twin are
indistinguishable on the wire. The inbound command surface lives only here, not
in the shared base or the sim sibling: a simulation is driven by its physics
engine, but a real arm is the one thing on the graph that an external controller
can actually move - so only the hardware bridge subscribes.

When ``enable_commands=False`` (or no robot is bound, as in the pure-publisher
construction the sim-symmetry tests use) no subscription and no thread are
created - the bridge degrades cleanly to the publish-only base behavior.

``rclpy`` and the ROS 2 message packages are optional, system-provided
dependencies (they are not on PyPI); they are imported lazily by the base, so
importing this module - and running hardware with ``ros2_bridge=False`` - never
requires ROS 2.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from strands_robots.ros_telemetry import RosTelemetryBridge
from strands_robots.utils import positive_finite_number_error

if TYPE_CHECKING:
    from strands_robots.hardware_robot import Robot

logger = logging.getLogger(__name__)


class HardwareRosBridge(RosTelemetryBridge):
    """Full-duplex ROS 2 bridge for a real robot (node name ``strands_hardware``).

    Telemetry (outbound) is byte-identical to its simulation sibling
    :class:`~strands_robots.simulation.ros_bridge.SimRosBridge` - see
    :class:`~strands_robots.ros_telemetry.RosTelemetryBridge` for the publish
    API. On top of that, when constructed with a bound ``robot`` and
    ``enable_commands=True`` (the default for a hardware robot), this bridge
    also subscribes to ``/<robot>/joint_command`` and forwards inbound messages
    to ``robot.send_action`` over a background spin thread.

    Args:
        robot: The hardware ``Robot`` to drive on inbound commands. When
            ``None`` (the pure-publisher construction), no command surface is
            created and the bridge behaves exactly like the base telemetry
            bridge - this preserves the sim-symmetry contract and lets callers
            that only publish (the per-step control-loop path) opt out of the
            inbound half entirely.
        domain_id: ROS 2 domain (``ROS_DOMAIN_ID``) to publish/subscribe on.
            Only an ``int`` in ``[0, 232]`` names a domain: RTPS derives its
            discovery ports from it, and 233 lands past the end of the port space.
        node_name: Internal rclpy node name (defaults to ``strands_hardware``).
        qos_depth: Depth of the publishers'/subscription's KEEP_LAST history.
        enable_commands: When True (default) and a ``robot`` is bound, subscribe
            to ``/<robot>/joint_command`` and drive the arm. Set False for a
            read-only (telemetry-only) bridge.
        command_robot_name: Topic namespace for the inbound command topic.
            Defaults to the bound robot's name (matching the namespace this
            bridge *publishes* ``joint_states`` under), so a controller can echo
            our own joint names straight back to drive the arm.
        spin_period: Seconds between ``spin_once`` calls on the command thread.
            Only a positive finite number paces a loop. The value is handed
            straight to ``Event.wait`` on the backoff path, where ``0``,
            a negative and ``nan`` all return immediately - turning the
            thread into a busy-spin with no bound - and ``inf`` raises
            ``OverflowError`` out of it, killing the loop while the bridge
            reports a successful construction.
        joint_limits: Optional ``{"<motor>.pos": (min, max)}`` clamp ranges,
            keyed by the joint name as it arrives in ``joint_command`` - the
            same ``<motor>.pos`` names this bridge publishes in
            ``joint_states``, so a controller can echo them straight back. A
            key that names no commanded joint constrains nothing. Each bound
            must be a finite number - a non-finite one declares a range that
            admits nothing, so the bridge refuses it at construction rather than
            dropping every inbound command for that joint mid-run. When set,
            an inbound ``joint_command`` whose ANY commanded joint is outside
            its declared range is rejected whole (no partial application), so a
            single out-of-range joint can never drive part of the arm.

    Raises:
        ValueError: If ``domain_id`` is outside ``[0, 232]``, if ``spin_period``
            is not a positive finite number, or if ``joint_limits`` is not a
            ``{"<motor>.pos": (min, max)}`` mapping of finite numeric pairs with
            ``min <= max``.
    """

    default_node_name = "strands_hardware"

    def __init__(
        self,
        robot: Robot | None = None,
        *,
        domain_id: int = 0,
        node_name: str | None = None,
        qos_depth: int = 10,
        enable_commands: bool = True,
        command_robot_name: str | None = None,
        spin_period: float = 0.02,
        joint_limits: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        # Refuse a spin period that cannot pace the command thread before the
        # base constructor runs - the placement ``domain_id`` already uses - so
        # the same caller mistake reports identically on an install with the
        # [ros2] extra and one without it. It also lands ahead of the
        # process-wide ``ROS_DOMAIN_ID`` write the base performs, so a refused
        # period leaves the environment as it found it.
        if error := positive_finite_number_error(spin_period, "spin_period", type(self).__name__):
            raise ValueError(error)

        super().__init__(domain_id=domain_id, node_name=node_name, qos_depth=qos_depth)

        self._robot = robot
        # Optional {"<motor>.pos": (min, max)} clamp ranges enforced on inbound
        # commands
        # by RosTelemetryBase._command_action (validated up front, fail fast).
        self._joint_limits = self._validate_joint_limits(joint_limits)
        # Commands require a robot to drive; a pure-publisher bridge (robot
        # None) is telemetry-only and stays symmetric with the sim sibling.
        self._enable_commands = bool(enable_commands) and robot is not None
        # ``float`` after the guard rather than instead of it: the shared domain
        # accepts any real scalar - a ``np.float32`` read from a config array is
        # documented as usable - and ``Event.wait`` rejects ``np.float32``
        # outright, so the conversion is what makes an accepted value consumable.
        self._spin_period = float(spin_period)
        self._command_sub: Any = None
        self._stop = threading.Event()
        self._spin_thread: threading.Thread | None = None

        if self._enable_commands:
            name = command_robot_name or self._resolve_robot_name(robot)
            self._command_robot_name = self._safe(name)
            self._command_sub = self._node.create_subscription(
                self._JointState,
                self.joint_command_topic(name),
                self._on_command,
                self._qos_depth,
            )
            self._start_spin()

    # -- inbound: drive the arm from /<robot>/joint_command ----------------

    def _on_command(self, msg: Any) -> None:
        """Forward an inbound ``joint_command`` JointState to ``send_action``.

        Delegates to :meth:`RosTelemetryBase._drive_from_command` (shared with
        the RTPS bridge): the message's ``name``/``position`` arrays are zipped
        into the flat ``{motor.pos: float}`` action dict the hardware ``Robot``
        accepts - the exact shape this bridge publishes in ``joint_states`` - and
        a mismatched/empty message is rejected rather than partially applied, so
        a malformed command never drives the arm to a surprising pose.
        """
        self._drive_from_command(self._robot, msg)

    # -- command spin thread ----------------------------------------------

    def _spin_loop(self) -> None:
        """Service inbound command callbacks without blocking the publisher.

        Publishing is driven elsewhere (the control loop / ``publish_ros_observation``)
        and needs no spin; this loop exists purely to deliver subscription
        callbacks. ``spin_once`` with a short timeout keeps shutdown responsive.
        """
        while not self._stop.is_set():
            try:
                self._rclpy.spin_once(self._node, timeout_sec=self._spin_period)
            except Exception:
                # A transient executor hiccup must not kill the command loop;
                # the next iteration retries. Persistent failure is bounded by
                # the stop event set on shutdown.
                logger.debug("HardwareRosBridge: spin_once raised", exc_info=True)
                self._stop.wait(self._spin_period)

    def _start_spin(self) -> None:
        """Start the command spin thread. Idempotent."""
        if self._spin_thread is not None and self._spin_thread.is_alive():
            return
        self._stop.clear()
        self._spin_thread = threading.Thread(
            target=self._spin_loop,
            name=f"{self._command_robot_name}_ros_cmd",
            daemon=True,
        )
        self._spin_thread.start()
        logger.info(
            "HardwareRosBridge: driving %r from /%s/joint_command",
            self._command_robot_name,
            self._command_robot_name,
        )

    # -- lifecycle --------------------------------------------------------

    def shutdown(self) -> None:
        """Stop the command thread, then destroy the node (base). Idempotent."""
        self._stop.set()
        thread = self._spin_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._spin_thread = None
        self._command_sub = None
        super().shutdown()
