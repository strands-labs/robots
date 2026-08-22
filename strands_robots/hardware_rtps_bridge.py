"""Pure-RTPS (cyclonedds) hardware bridge - a real robot on ROS 2, no rclpy.

This is the rclpy-free sibling of
:class:`strands_robots.hardware_ros_bridge.HardwareRosBridge`. Both derive from
:class:`strands_robots.ros_telemetry.RosTelemetryBase` - the single source of
truth for the ROS 2 topic names and the inbound ``joint_command`` contract - so
they are byte-compatible on the wire by construction. ``HardwareRtpsBridge``
exposes the exact same ROS 2 topics for a physical
:class:`strands_robots.hardware_robot.Robot` - and to real ROS 2 nodes - but
speaks DDS/RTPS directly through the pip-installable ``cyclonedds`` binding
instead of a sourced ROS 2 distro:

* **publish** (outbound) - ``/<robot>/joint_states``
  (``sensor_msgs/msg/JointState``) and, per camera,
  ``/<robot>/<camera>/image_raw`` (``sensor_msgs/msg/Image``, ``rgb8``).
* **subscribe** (inbound) - ``/<robot>/joint_command``
  (``sensor_msgs/msg/JointState``) forwarded into
  ``robot.send_action({motor.pos: float})`` over a background poll thread, so an
  external ROS 2 stack can drive the physical arm. Full duplex, same contract as
  the rclpy bridge (shared via :class:`~strands_robots.ros_telemetry.RosTelemetryBase`).

Why this exists alongside ``HardwareRosBridge``: ``rclpy`` needs a *sourced ROS 2
distro* (apt / RoboStack / docker), which is heavy and version-pinned (Humble vs
Jazzy vs Rolling). ``cyclonedds`` is a single self-contained pip wheel that
speaks the RTPS wire protocol every ROS 2 distro shares, so this bridge runs on
a bare dev laptop or a minimal robot image with ``pip install
'strands-robots[ros2]'`` and nothing else. The trade-off is type coverage: RTPS
publishing needs a *local* IDL definition, so only the messages in
:mod:`strands_robots.rtps.idl` work (now ``geometry_msgs`` + the ``sensor_msgs``
``JointState``/``Image`` chain this bridge needs). The rclpy bridge keeps full
``sensor_msgs`` fidelity for anything outside the bundle.

Selection is the hardware ``Robot``'s job (``ros2_transport="rclpy"|"rtps"``);
this module only implements the RTPS path. Both bridges present an identical
``publish_joint_states`` / ``publish_image`` / ``shutdown`` surface so the
``Robot`` telemetry path is transport-agnostic.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from strands_robots.ros_telemetry import RosTelemetryBase
from strands_robots.utils import (
    boolean_flag_error,
    dds_domain_id_error,
    partial_construction_repr,
    positive_finite_number_error,
    require_optional,
)

if TYPE_CHECKING:
    import numpy as np

    from strands_robots.hardware_robot import Robot

from strands_robots.mesh.pacing import Ticker

logger = logging.getLogger(__name__)

# ROS 2 type strings this bridge publishes/subscribes. All must be present in
# the RTPS IDL bundle (strands_robots.rtps.idl.REGISTRY).
_JOINT_STATE_TYPE = "sensor_msgs/msg/JointState"
_IMAGE_TYPE = "sensor_msgs/msg/Image"


# Map the operator-facing dds_security_config keys to the DDS Security
# participant property names cyclonedds reads (OMG DDS-Security v1.1, table 8.20
# "dds.sec.*"). Values are passed through verbatim so the operator controls the
# scheme (``file:/path`` or ``data:`` per the spec). ``permissions_ca`` is
# optional; the rest are required (validated by RosTelemetryBase).
_DDS_SECURITY_PROPERTY = {
    "identity_ca": "dds.sec.auth.identity_ca",
    "certificate": "dds.sec.auth.identity_certificate",
    "private_key": "dds.sec.auth.private_key",
    "permissions_ca": "dds.sec.access.permissions_ca",
    "governance": "dds.sec.access.governance",
    "permissions": "dds.sec.access.permissions",
}

# The OMG DDS-Security builtin plugins cyclonedds ships. Set unconditionally
# alongside the credentials so an operator only supplies cert/CA/governance/
# permissions, not the plugin wiring.
_DDS_SECURITY_PLUGINS = {
    "dds.sec.auth.library.path": "dds_security_auth",
    "dds.sec.auth.library.init": "init_authentication",
    "dds.sec.auth.library.finalize": "finalize_authentication",
    "dds.sec.crypto.library.path": "dds_security_crypto",
    "dds.sec.crypto.library.init": "init_crypto",
    "dds.sec.crypto.library.finalize": "finalize_crypto",
    "dds.sec.access.library.path": "dds_security_ac",
    "dds.sec.access.library.init": "init_access_control",
    "dds.sec.access.library.finalize": "finalize_access_control",
}


class HardwareRtpsBridge(RosTelemetryBase):
    """Full-duplex hardware ROS 2 bridge over pure RTPS (cyclonedds, no rclpy).

    The rclpy-free sibling of
    :class:`~strands_robots.hardware_ros_bridge.HardwareRosBridge`. Both derive
    from :class:`~strands_robots.ros_telemetry.RosTelemetryBase`, so they share
    the topic names and the ``joint_command`` -> ``send_action`` contract and are
    wire-compatible by construction; they differ only in transport (cyclonedds
    RTPS vs rclpy) and in type coverage (bounded by the local IDL bundle).

    Args:
        robot: The hardware ``Robot`` to drive on inbound commands. When
            ``None``, no command surface is created (telemetry-only), mirroring
            the rclpy bridge's pure-publisher mode.
        domain_id: ROS 2 / DDS domain id to publish/subscribe on.
            Only an ``int`` in ``[0, 232]`` names a domain: RTPS derives its
            discovery ports from it, and 233 lands past the end of the port space.
        enable_commands: When True (default) and a ``robot`` is bound, subscribe
            to ``/<robot>/joint_command`` and drive the arm. Only a boolean names
            a posture: the value is checked, not read by truthiness, so
            ``"false"`` cannot select the surface it asks to close.
        command_robot_name: Topic namespace for the command topic; defaults to
            the bound robot's name (the namespace we publish ``joint_states``
            under).
        poll_period: Seconds between inbound command reads on the poll thread.
            Only a positive finite number paces a loop. It is the sole pacing
            of ``_poll_loop``, handed to ``Event.wait``, where ``0``, a
            negative and ``nan`` all return immediately - turning the thread
            into a busy-spin with no bound - and ``inf`` raises
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
            an inbound ``joint_command`` whose ANY commanded joint falls outside
            its declared range is rejected whole (no partial application), so a
            single out-of-range joint can never drive part of the arm.
        dds_security_config: Optional DDS Security credentials. Required keys
            (``identity_ca``, ``certificate``, ``private_key``, ``governance``,
            ``permissions``; ``permissions_ca`` optional) wire the participant's
            DDS Security plugins so the whole graph is authenticated and
            access-controlled. When ``enable_commands`` is in effect this (or
            the ``STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE=1`` opt-out) is
            REQUIRED - the bridge refuses to expose an arm-driving command
            surface on an unsecured DDS graph.

    Raises:
        ImportError: If ``cyclonedds`` (the ``[ros2]`` extra) is not installed.
        ValueError: If ``enable_commands`` is not a boolean, ``domain_id`` is
            outside ``[0, 232]`` or ``poll_period`` is not a positive finite
            number (all three checked before the ``cyclonedds`` probe, so the
            same caller mistake reports identically on an install without the
            extra), if ``joint_limits`` /
            ``dds_security_config`` is malformed, or if commands are enabled
            with neither a security config nor the explicit insecure opt-out.
    """

    def __init__(
        self,
        robot: Robot | None = None,
        *,
        domain_id: int = 0,
        enable_commands: bool = True,
        command_robot_name: str | None = None,
        poll_period: float = 0.02,
        joint_limits: dict[str, tuple[float, float]] | None = None,
        dds_security_config: dict[str, str] | None = None,
    ) -> None:
        # Refuse a domain id outside the RTPS port map first, so the same caller
        # mistake reports identically whether or not the [ros2] extra is
        # installed - and so it is answered before any DDS state is built.
        if error := dds_domain_id_error(domain_id, "domain_id", type(self).__name__):
            raise ValueError(error)

        # The poll period is answered here too, for the same two reasons: it is
        # the only pacing ``_poll_loop`` has, and a value that cannot pace a
        # loop is not made usable by having a transport to poll.
        if error := positive_finite_number_error(poll_period, "poll_period", type(self).__name__):
            raise ValueError(error)

        # ``enable_commands`` selects whether this bridge exposes an inbound,
        # arm-driving surface, so it is checked rather than read by truthiness.
        # Every non-empty string is truthy, so ``"false"`` would open the very
        # surface the caller asked to close - and, because the DDS Security gate
        # below branches on the same flag, it would also refuse a read-only
        # request with a message about "an enabled command bridge" and advise
        # the insecure opt-out that opens it. Answered alongside the two guards
        # above, so the refusal lands before any DDS state exists and reports
        # identically with and without the [ros2] extra.
        if error := boolean_flag_error(enable_commands, "enable_commands", type(self).__name__):
            raise ValueError(error)

        # cyclonedds is the only dependency - no rclpy, no sourced ROS 2 distro.
        require_optional(
            "cyclonedds",
            extra="ros2",
            purpose="the pure-RTPS hardware bridge (Robot ros2_transport='rtps')",
        )
        from cyclonedds.domain import DomainParticipant

        from strands_robots.rtps.idl import get_type
        from strands_robots.rtps.mangling import dds_topic_name

        self._get_type = get_type
        self._dds_topic_name = dds_topic_name

        self._robot = robot
        self._domain_id = domain_id

        # Whether this bridge exposes an inbound (arm-driving) command surface.
        # Resolved BEFORE the participant is built so the DDS Security gate and
        # the participant's security QoS are decided together.
        self._enable_commands = bool(enable_commands) and robot is not None

        # Validate the optional clamp ranges up front (fail fast at construction).
        self._joint_limits = self._validate_joint_limits(joint_limits)

        # DDS Security gate: an enabled inbound command surface lets any DDS
        # participant drive the physical arm, so refuse to start one on an
        # unsecured graph unless given a dds_security_config or an explicit
        # operator opt-out (STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE=1).
        if dds_security_config is not None:
            dds_security_config = self._validate_dds_security_config(dds_security_config)
        self._require_secure_command_surface(
            enable_commands=self._enable_commands,
            dds_security_config=dds_security_config,
        )
        self._dds_security_config = dds_security_config

        # Build the participant with DDS Security QoS when a config is supplied,
        # so BOTH the outbound telemetry and the inbound command surface ride a
        # secured (authenticated + access-controlled) DDS graph.
        if dds_security_config is not None:
            self._participant: Any = DomainParticipant(
                self._domain_id, qos=self._build_security_qos(dds_security_config)
            )
        else:
            self._participant = DomainParticipant(self._domain_id)

        self._robot_name = self._safe(self._resolve_robot_name(robot) if robot is not None else "robot")
        self._joint_writer: Any = None
        self._image_writers: dict[str, Any] = {}

        # Cache the resolved IDL classes once (KeyError here = the bundle is
        # missing a type, a packaging bug, surfaced at construction not mid-run).
        self._JointState = get_type(_JOINT_STATE_TYPE)
        self._Image = get_type(_IMAGE_TYPE)

        # ``float`` after the guard rather than instead of it: the shared domain
        # accepts any real scalar - a ``np.float32`` read from a config array is
        # documented as usable - and ``Event.wait`` rejects ``np.float32``
        # outright, so the conversion is what makes an accepted value consumable.
        self._poll_period = float(poll_period)
        self._command_reader: Any = None
        self._stop = threading.Event()
        self._poll_thread: threading.Thread | None = None

        if self._enable_commands:
            name = command_robot_name or self._resolve_robot_name(robot)
            self._command_robot_name = self._safe(name)
            self._command_reader = self._make_reader(self.joint_command_topic(name), self._JointState)
            self._start_poll()

    def _build_security_qos(self, config: dict[str, str]) -> Any:
        """Build a cyclonedds ``Qos`` carrying the DDS Security participant properties.

        Combines the builtin DDS-Security plugin wiring (:data:`_DDS_SECURITY_PLUGINS`)
        with the operator-supplied credentials, mapped to their ``dds.sec.*``
        property names (:data:`_DDS_SECURITY_PROPERTY`). Optional keys absent
        from ``config`` (e.g. ``permissions_ca``) are simply not set.
        """
        from cyclonedds.qos import Policy, Qos

        properties = dict(_DDS_SECURITY_PLUGINS)
        for key, prop in _DDS_SECURITY_PROPERTY.items():
            value = config.get(key)
            if value:
                properties[prop] = str(value)
        return Qos(*[Policy.Property(name, value) for name, value in properties.items()])

    # -- helpers ----------------------------------------------------------

    def _make_writer(self, ros_topic: str, idl_cls: Any) -> Any:
        from cyclonedds.pub import DataWriter
        from cyclonedds.topic import Topic

        topic = Topic(self._participant, self._dds_topic_name(ros_topic), idl_cls)
        return DataWriter(self._participant, topic)

    def _make_reader(self, ros_topic: str, idl_cls: Any) -> Any:
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic

        topic = Topic(self._participant, self._dds_topic_name(ros_topic), idl_cls)
        return DataReader(self._participant, topic)

    # -- publish (outbound) ----------------------------------------------

    def publish_joint_states(self, robot: str, names: list[str], positions: list[float]) -> None:
        """Publish one ``JointState`` for ``robot`` on ``/<robot>/joint_states``.

        Signature matches ``RosTelemetryBridge.publish_joint_states`` so the
        hardware ``Robot`` telemetry path is transport-agnostic.
        """
        if self._joint_writer is None:
            self._joint_writer = self._make_writer(self.joint_states_topic(robot), self._JointState)
        msg = self._JointState(
            header=self._header(self._safe(robot)),
            name=list(names),
            position=[float(p) for p in positions],
            velocity=[],
            effort=[],
        )
        self._joint_writer.write(msg)

    def publish_image(self, robot: str, camera: str, image: np.ndarray) -> None:
        """Publish one RGB ``Image`` on ``/<robot>/<camera>/image_raw``."""
        if image.ndim != 3 or image.shape[2] != 3:
            return
        key = f"{robot}/{camera}"
        writer = self._image_writers.get(key)
        if writer is None:
            writer = self._make_writer(self.image_topic(robot, camera), self._Image)
            self._image_writers[key] = writer
        height, width = int(image.shape[0]), int(image.shape[1])
        msg = self._Image(
            header=self._header(f"{self._safe(robot)}/{self._safe(camera)}"),
            height=height,
            width=width,
            encoding="rgb8",
            is_bigendian=0,
            step=width * 3,
            data=image.astype("uint8", copy=False).tobytes(),
        )
        writer.write(msg)

    def _header(self, frame_id: str) -> Any:
        """Build a std_msgs/Header with a wall-clock stamp (sec/nanosec)."""
        Header = self._get_type("std_msgs/msg/Header")
        Time = self._get_type("builtin_interfaces/msg/Time")
        now = time.time()
        sec = int(now)
        nanosec = int((now - sec) * 1e9)
        return Header(stamp=Time(sec=sec, nanosec=nanosec), frame_id=frame_id)

    # -- subscribe (inbound) ---------------------------------------------

    def _on_command(self, msg: Any) -> None:
        """Forward an inbound ``joint_command`` JointState to ``send_action``.

        Delegates to :meth:`RosTelemetryBase._drive_from_command` (shared with
        the rclpy bridge): zip ``name``/``position`` into a flat
        ``{motor.pos: float}`` action; reject mismatched messages rather than
        partially applying; surface (never raise) ``send_action`` errors.
        ``skip_empty=True`` because cyclonedds ``take()`` may surface a wholly
        empty sample (DDS dispose / keep-alive) that is not a real actuation
        request, so it is dropped quietly rather than warned on.
        """
        self._drive_from_command(self._robot, msg, skip_empty=True)

    def _poll_loop(self) -> None:
        """Poll the command reader and dispatch new samples to ``_on_command``.

        cyclonedds has no rclpy-style executor; we ``take()`` available samples
        each tick. ``take`` (not ``read``) so each command is delivered once.

        Paced by :class:`~strands_robots.mesh.pacing.Ticker` rather than
        ``self._stop.wait(period)``. That wait is a delay, so the time spent
        delivering a batch of commands was added to the poll period instead of
        being subtracted from it -- at the 0.02s default an inbound actuation
        request sat unread in the reader for longer than the 50Hz the period
        asks for. Nothing here reported that: the loop keeps no rate counter,
        and ``take`` returning a batch makes a late poll look like a busy one.
        """
        with Ticker(self._poll_period, self._stop) as ticker:
            while not self._stop.is_set():
                try:
                    for sample in self._command_reader.take(N=10):
                        self._on_command(sample)
                except Exception:
                    logger.debug("HardwareRtpsBridge: command poll raised", exc_info=True)
                if ticker.wait():
                    break

    def _start_poll(self) -> None:
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._stop.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name=f"{self._command_robot_name}_rtps_cmd",
            daemon=True,
        )
        self._poll_thread.start()
        logger.info(
            "HardwareRtpsBridge: driving %r from /%s/joint_command (cyclonedds, no rclpy)",
            self._command_robot_name,
            self._command_robot_name,
        )

    # -- lifecycle --------------------------------------------------------

    def shutdown(self) -> None:
        """Stop the poll thread and drop DDS entities. Idempotent."""
        self._stop.set()
        thread = self._poll_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._poll_thread = None
        # Dropping references lets cyclonedds reclaim the readers/writers and
        # the participant; there is no explicit close() in the python binding.
        self._command_reader = None
        self._joint_writer = None
        self._image_writers = {}
        self._participant = None

    def __repr__(self) -> str:
        try:
            return (
                f"HardwareRtpsBridge(robot={self._robot_name!r}, domain_id={self._domain_id}, "
                f"enable_commands={self._enable_commands})"
            )
        except AttributeError:
            return partial_construction_repr(self)
