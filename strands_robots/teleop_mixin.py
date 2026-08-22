"""TeleopMixin - attach/drive teleoperators on any robot or simulation.

Shared by :class:`strands_robots.hardware_robot.Robot` and the MuJoCo
:class:`strands_robots.simulation.Simulation`. The only contract a host
class must satisfy is a ``send_action(action: dict, robot_name: str | None
= None) -> dict`` method (both already have it) and, for mesh publishing,
the ``mesh`` / ``peer_id`` attributes (both already have them).

Design
------
* **Multi / dict storage** - ``_teleops: dict[str, AttachedTeleop]`` so a
  gamepad + leader arm can drive one follower simultaneously
  (``List[Teleoperator]`` semantics, lerobot's direction of travel).
* **Lazy** - ``attach_teleop`` never touches hardware. Devices are
  ``connect()``-ed only when ``teleoperate()`` runs.
* **map_fn** - per-device optional ``(action: dict) -> dict`` remap, the
  bridge for driving a *sim* arm from a real leader whose joint/actuator
  names differ. Identity by default.
* **Local + mesh** - ``teleoperate()`` runs the local merge+apply loop
  (lerobot ``teleop_loop`` equivalent). ``teleoperate(publish=True)`` ALSO
  publishes each device to the mesh via the host's
  ``start_teleop_publish`` (hardware Robot) so remote followers can mirror.
* **Slew-bounded** - every merged frame is held to the same per-joint speed
  bound the mesh receive path applies to an inbound frame
  (``STRANDS_TELEOP_SLEW_ABS``, default 500 units/second -- wide enough for
  degree-valued and range-0-100 devices at shipped defaults). One device can
  drive a local follower and, via ``publish=True``, remote ones from the same
  ``get_action()`` stream, so the two paths have to judge a frame identically
  or the follower next to the operator is the only unguarded one. The bound is
  a speed above what a leader arm's own servos can produce, so a physical
  leader does not trip it; an over-speed frame (an encoder glitch, a USB
  re-enumerate, a synthetic stream) is refused and counted in
  ``slew_rejected`` rather than clamped, since clamping toward the commanded
  value would silently alter an actuator command.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from strands_robots.utils import name_list_error, positive_finite_number_error

logger = logging.getLogger(__name__)

# Default local control loop frequency (Hz). Matches InputPublisher.
_TELEOP_HZ_DEFAULT = 50.0

ActionDict = dict[str, float]
MapFn = Callable[[ActionDict], ActionDict]


@dataclass
class AttachedTeleop:
    """A teleoperator registered to a host robot/sim (lazy, not yet connected).

    Attributes:
        device: The raw lerobot Teleoperator (or a built spec). Duck-types to
            ``get_action() -> dict``, ``connect()``, ``disconnect()``,
            ``is_connected``.
        name: Stable key in the host's ``_teleops`` dict.
        method: Input-method label ("arm", "gamepad", "keyboard", "phone") -
            forwarded to mesh publish so receivers know the stream shape.
        map_fn: Optional ``(action) -> action`` remap applied before the
            action is sent to the host. Identity when ``None``.
    """

    device: Any
    name: str
    method: str = "arm"
    map_fn: MapFn | None = None


class TeleopMixin:
    """Mixin adding ``attach_teleop`` / ``teleoperate`` to a robot or sim.

    The host class MUST provide ``send_action(action, robot_name=None)``.
    For ``teleoperate(publish=True)`` the host must also provide
    ``start_teleop_publish`` (hardware ``Robot`` does).
    """

    # --- host contract ----------------------------------------------------
    # The host class (hardware Robot / MuJoCo Simulation) MUST provide this.
    # Declared here (not implemented) so static analysis knows the mixin's
    # _teleop_loop may call it; at runtime Python resolves the host's concrete
    # method via MRO. A bare TeleopMixin (no host) raises NotImplementedError.
    def start_teleop_receive(
        self,
        source_peer_id: str,
        device_name: str = "leader",
        apply_fn: Any | None = None,
        robot_name: str | None = None,
    ) -> dict[str, Any]:
        """Follow a remote leader's input stream and apply it to this host.

        Host-agnostic (N6): previously this existed only on the hardware
        Robot, so a sim digital twin could not follow a real leader arm over
        the mesh - "practice on the twin before touching metal" was
        impossible, and every teleop verb answered sim peers with "robot does
        not support teleop_receive". The receiver's default apply is
        ``robot.send_action(action)``, which this mixin defines for both
        hosts; on a sim, ``robot_name`` scopes the target arm (single-robot
        worlds resolve it automatically).

        Args:
            source_peer_id: Peer id of the publishing leader.
            device_name: Input stream name to subscribe to.
            apply_fn: Optional custom ``(robot, action_dict) -> None``.
            robot_name: Sim only - which robot in the world receives the
                actions. Ignored on hardware hosts.

        Returns:
            Status dict; error when the mesh is inactive or an identifier is
            not a valid mesh identifier.
        """
        mesh = getattr(self, "mesh", None)
        if not mesh or not getattr(mesh, "alive", False):
            return {"status": "error", "content": [{"text": "Mesh not active. Cannot receive input."}]}

        from strands_robots.mesh.security import ValidationError, validate_mesh_identifier

        try:
            validate_mesh_identifier(source_peer_id, "start_teleop_receive.source_peer_id")
            validate_mesh_identifier(device_name, "start_teleop_receive.device_name")
        except ValidationError as exc:
            return {"status": "error", "content": [{"text": str(exc)}]}

        from strands_robots.mesh import InputReceiver

        if apply_fn is None and robot_name is not None:
            # Sim host with an explicit target arm: bind robot_name so the
            # receiver's send_action lands on the right robot in the world.
            def apply_fn(host: Any, action: dict[str, float], _rn: str | None = robot_name) -> None:  # noqa: ANN401
                host.send_action(action, robot_name=_rn)

        if not hasattr(self, "_input_receivers"):
            self._input_receivers: dict[str, Any] = {}

        key = f"{source_peer_id}/{device_name}"
        if key in self._input_receivers:
            self._input_receivers[key].stop()

        # Hardware hosts apply to the inner lerobot driver (self.robot);
        # sim hosts apply to self (send_action routes by robot_name).
        target = getattr(self, "robot", None) if apply_fn is None else self
        if target is None:
            target = self

        receiver = InputReceiver(
            mesh=mesh,
            robot=target,
            source_peer_id=source_peer_id,
            device_name=device_name,
            apply_fn=apply_fn,
        )
        receiver.start()
        self._input_receivers[key] = receiver

        return {
            "status": "success",
            "content": [
                {
                    "text": f"Teleop receive started: following {source_peer_id}/{device_name}"
                    + (f" -> {robot_name}" if robot_name else "")
                }
            ],
        }

    def get_teleop_status(self) -> dict[str, Any]:
        """Status of all active teleop publishers/receivers (host-agnostic)."""
        publishers = {}
        receivers = {}
        if hasattr(self, "_input_publishers"):
            for name, pub in self._input_publishers.items():
                publishers[name] = pub.stats
        if hasattr(self, "_input_receivers"):
            for key, rcv in self._input_receivers.items():
                receivers[key] = rcv.stats
        return {
            "status": "success",
            "content": [
                {
                    "text": f"Teleop status:\n"
                    f"  Publishers: {len(publishers)} active\n"
                    f"  Receivers: {len(receivers)} active\n"
                    + "".join(
                        f"  [pub] {n}: {s.get('frames', 0)} frames @ {s.get('hz_actual', 0):.1f}Hz\n"
                        for n, s in publishers.items()
                    )
                    + "".join(
                        f"  [rcv] {k}: {s.get('frames_received', 0)} frames @ {s.get('hz_actual', 0):.1f}Hz\n"
                        for k, s in receivers.items()
                    )
                },
                {"json": {"publishers": publishers, "receivers": receivers}},
            ],
        }

    def send_action(self, action: ActionDict, robot_name: str | None = None) -> dict[str, Any]:
        """Apply ``action`` to the host robot/sim. Implemented by the host."""
        raise NotImplementedError(
            "TeleopMixin requires the host class to implement send_action(action, robot_name=None)."
        )

    # --- lazy per-instance state ------------------------------------------
    # Stored via _ensure_teleop_state so the mixin needs no __init__ and both
    # hosts (which call their own super().__init__) get it for free.

    def _ensure_teleop_state(self) -> None:
        if not hasattr(self, "_teleops"):
            self._teleops: dict[str, AttachedTeleop] = {}
            self._teleop_thread: threading.Thread | None = None
            self._teleop_stop_event: threading.Event = threading.Event()
            self._teleop_running: bool = False
            self._teleop_robot_name: str | None = None
            self._teleop_frames: int = 0
            self._teleop_errors: int = 0
            # ``time.monotonic()`` reading taken when the session began.
            # Every reader subtracts it from a later reading of the same clock
            # (the deadline, the elapsed/Hz report), and none reports it as a
            # point in time, so a wall-clock step cannot move any of them.
            self._teleop_start_mono: float = 0.0
            self._teleop_slew_rejected: int = 0
            # Baseline for the per-joint slew bound: for each joint, the last
            # value actually sent and when. Merged rather than replaced, so a
            # device whose frame carries a subset of the joints cannot erase
            # the baseline of the ones it omits.
            self._teleop_slew_baseline: dict[str, tuple[float, float]] = {}

    # --- attach / detach --------------------------------------------------

    def attach_teleop(
        self,
        device_or_spec: Any,
        *,
        name: str | None = None,
        method: str | None = None,
        map_fn: MapFn | None = None,
        **kwargs: Any,
    ) -> TeleopMixin:
        """Attach a teleoperator (lazy - no hardware touched here).

        Args:
            device_or_spec: Either a built lerobot ``Teleoperator`` instance,
                or a teleoperator *type string* ("so101_leader", "gamepad",
                ...) which is built via the :func:`Teleoperator` factory using
                ``**kwargs``.
            name: Stable key for this input stream. Defaults to the device's
                lerobot ``id`` if set, else its ``name`` (type), else
                ``"leader"``. Used in ``teleoperate(names=[...])``, mesh
                topics, and ``detach_teleop``.
            method: Input-method label ("arm", "gamepad", "keyboard",
                "phone"). Auto-derived from the teleop type when omitted.
            map_fn: Optional ``(action: dict) -> dict`` applied before the
                action reaches ``send_action``. The bridge for sim teleop
                (remap leader joint names -> sim actuator names). Identity by
                default.
            **kwargs: Forwarded to the :func:`Teleoperator` factory when
                ``device_or_spec`` is a type string (e.g. ``port=``, ``id=``).
                Rejected (``TypeError``) when a built device is passed.

        Returns:
            ``self`` - chainable:
            ``robot.attach_teleop("so101_leader", port=...).attach_teleop("gamepad")``.

        Raises:
            ValueError: If the resolved ``name`` collides with an already
                attached device, or the device has no ``get_action``.
            TypeError: If ``**kwargs`` are passed alongside a built device.
        """
        self._ensure_teleop_state()

        if isinstance(device_or_spec, str):
            # Build lazily via the factory. Import here to avoid a hard import
            # cycle (teleoperator.py imports lerobot; mixin stays light).
            from strands_robots.teleoperator import Teleoperator

            device = Teleoperator(device_or_spec, **kwargs)
            derived_type = device_or_spec
        else:
            if kwargs:
                raise TypeError(
                    f"attach_teleop(**kwargs) is only valid when building from a "
                    f"type string; a pre-built device was passed with kwargs "
                    f"{sorted(kwargs)}. Build the device with those kwargs via "
                    f"Teleoperator(...) instead, or pass a type string."
                )
            device = device_or_spec
            derived_type = getattr(device, "name", None) or type(device).__name__

        if not callable(getattr(device, "get_action", None)):
            raise ValueError(
                f"Attached teleoperator {device!r} has no callable get_action(); "
                "it does not satisfy the teleoperator contract."
            )

        # Resolve a stable name: explicit > lerobot id > lerobot type > 'leader'.
        resolved = name or getattr(device, "id", None) or getattr(device, "name", None) or "leader"
        if resolved in self._teleops:
            raise ValueError(
                f"A teleoperator named {resolved!r} is already attached. Pass an "
                f"explicit name= to attach multiple devices "
                f"(attached: {sorted(self._teleops)})."
            )

        resolved_method = method or _infer_method(str(derived_type))

        self._teleops[resolved] = AttachedTeleop(
            device=device,
            name=resolved,
            method=resolved_method,
            map_fn=map_fn,
        )
        logger.info(
            "[teleop] attached %r (type=%s, method=%s, map_fn=%s)",
            resolved,
            derived_type,
            resolved_method,
            "yes" if map_fn else "no",
        )
        return self

    def detach_teleop(self, name: str | None = None) -> dict[str, Any]:
        """Detach a specific teleoperator, or all when ``name`` is None.

        Stops the local loop first if it's running and would be left with no
        devices. Disconnects each detached device if it was connected.

        Args:
            name: Which attached stream to detach. ``None`` = every attached
                device. Read by membership, so only ``None`` selects all: any
                other value names one stream, and a value naming no attached
                stream is refused rather than widened to the whole set.

        Returns:
            Status dict; error when ``name`` names no attached teleoperator.
        """
        self._ensure_teleop_state()

        # ``name`` selects which attached devices this call operates on, so it is
        # read by membership: ``None`` is the documented "detach every attached
        # device", and any other value names one. Read by truthiness, ``""`` took
        # the all-devices branch, so a detach aimed at a single stream removed the
        # whole set - and, with a session running, ended it, because the branch
        # below stops the loop once nothing is left to drive. An empty name is not
        # a spelling of "all"; it names no attached device, so it now reaches the
        # not-found refusal below.
        names = list(self._teleops) if name is None else [name]
        detached = []
        for n in names:
            att = self._teleops.pop(n, None)
            if att is None:
                continue
            # Best-effort disconnect; a device may never have been connected.
            try:
                if getattr(att.device, "is_connected", False):
                    att.device.disconnect()
            except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
                logger.warning("[teleop] disconnect of %r failed: %s", n, exc)
            detached.append(n)

        if not self._teleops and self._teleop_running:
            self.stop_teleoperate()

        if not detached:
            return {"status": "error", "content": [{"text": f"No teleop named {name!r}."}]}
        return {
            "status": "success",
            "content": [{"text": f"Detached: {detached}"}],
        }

    def list_teleops(self) -> dict[str, Any]:
        """List attached teleoperators and their connection state."""
        self._ensure_teleop_state()
        rows = []
        for n, att in self._teleops.items():
            connected = bool(getattr(att.device, "is_connected", False))
            rows.append(
                f"  {n}: type={getattr(att.device, 'name', type(att.device).__name__)}, "
                f"method={att.method}, map_fn={'yes' if att.map_fn else 'no'}, "
                f"connected={connected}"
            )
        body = "\n".join(rows) if rows else "  (none)"
        return {
            "status": "success",
            "content": [
                {"text": f"Attached teleoperators ({len(self._teleops)}):\n{body}"},
                {"json": {"teleops": list(self._teleops)}},
            ],
        }

    # --- drive ------------------------------------------------------------

    def teleoperate(
        self,
        *,
        names: list[str] | None = None,
        robot_name: str | None = None,
        hz: float = _TELEOP_HZ_DEFAULT,
        publish: bool = False,
        block: bool = False,
        duration: float | None = None,
    ) -> dict[str, Any]:
        """Drive this robot/sim from its attached teleoperator(s).

        Connects the selected teleoperators (lazy -> active) and runs a
        control loop that, each tick, polls every device's ``get_action()``,
        applies its ``map_fn``, merges the results (last-wins on key conflict,
        with a one-time warning), and applies the merged action via
        ``self.send_action(merged, robot_name=...)``.

        Args:
            names: Subset of attached device names to drive. ``None`` = every
                attached device. Read by membership, so an explicitly empty
                selection is not a spelling of "all": ``names=[]`` is refused
                rather than widened to every device. The list itself is held to
                the shared name-list domain - several distinct non-blank names,
                as a sequence - so a single name passed as a bare string, a
                repeated name, and a one-shot iterator are refused rather than
                reinterpreted.
            robot_name: Target robot for ``send_action``. ``None`` -> the
                host's default (single hardware robot, or first sim robot).
                In a multi-robot sim, name the specific robot.
            hz: Local control-loop frequency. Must be a positive finite
                number - the loop period is ``1 / hz``.
            publish: Also publish each selected device to the mesh via the
                host's ``start_teleop_publish`` so remote peers can follow.
                Requires the host to expose ``start_teleop_publish`` and a
                live mesh.
            block: Run the loop in the calling thread (blocks until
                ``duration`` elapses or KeyboardInterrupt). When ``False``
                (default) the loop runs in a managed background thread and the
                call returns immediately with a handle/status.
            duration: Stop automatically after N seconds. Must be a positive
                finite number when given; ``None`` = run until
                ``stop_teleoperate()`` (background) / Ctrl+C (block). Measured
                from the end of setup, so it is time spent teleoperating: device
                connection and the one-time resolution of the slew helpers happen
                before the clock starts and are not charged to it.

        Returns:
            Status dict. Background mode returns immediately; ``block=True``
            returns after the loop ends with frame/error stats, including
            ``slew_rejected``: frames refused for commanding a joint faster
            than the per-joint slew bound the mesh receive path also applies
            (see the module docstring). Refusals are not errors, but a session
            with any of them does not report ``success``, so a device whose
            units the bound does not expect cannot look like a clean run while
            moving nothing. An ``hz`` or ``duration`` the loop cannot honor is
            refused here rather than reported as a started session, as is a
            ``names`` that does not name a usable subset - both are refused
            before any device is connected.
        """
        self._ensure_teleop_state()

        # Validate the loop knobs BEFORE anything is connected or published:
        # both are consumed only inside the loop (``1 / hz`` for the period,
        # ``start + duration`` for the deadline), so an unusable value used to
        # be reported as a started session and only misbehave on the background
        # thread - see the module docstring's rate/duration contract.
        for value, param in ((hz, "hz"), (duration, "duration")):
            if param == "duration" and value is None:
                # Documented: run until stop_teleoperate() / Ctrl+C.
                continue
            error = positive_finite_number_error(value, param, "teleoperate")
            if error:
                return {"status": "error", "content": [{"text": error}]}

        if not self._teleops:
            return {
                "status": "error",
                "content": [{"text": "No teleoperators attached. Use attach_teleop() first."}],
            }
        if self._teleop_running:
            return {
                "status": "error",
                "content": [{"text": "Teleoperation already running. Call stop_teleoperate() first."}],
            }

        # ``names`` selects a SUBSET of the attached devices, so it is read by
        # membership - the same rule ``duration`` is read by in this call and
        # ``cameras`` is resolved by on the render path, where an empty selection
        # resolves to no camera rather than to every one. ``None`` is the
        # documented "every attached device"; an explicitly empty selection is the
        # opposite of that, not a spelling of it. Read by truthiness, ``names=[]``
        # - what a filter that matched nothing produces - connected and drove
        # every attached device, so a call that selected no leader energised all
        # of them and reported success.
        #
        # The shape goes through the shared name-list domain, because the other
        # ways this selector cannot be honored as written were reinterpreted too:
        # a single name as a bare string is iterable per character, so it was read
        # as one device per letter; a repeated name polled that device twice per
        # tick and then warned that it conflicted with itself; and a one-shot
        # iterator was consumed by the membership check below, leaving the loop
        # nothing to poll while the session still reported success.
        if names is None:
            selected = list(self._teleops)
        else:
            if error := name_list_error(names, "names", "teleoperate"):
                return {"status": "error", "content": [{"text": error}]}
            selected = list(names)
            if not selected:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                "teleoperate(names=[]) selects no teleoperator, so there is nothing to "
                                "drive. Pass names=None to drive every attached device "
                                f"(attached: {sorted(self._teleops)}), or name the subset to drive."
                            )
                        }
                    ],
                }

        unknown = [n for n in selected if n not in self._teleops]
        if unknown:
            return {
                "status": "error",
                "content": [{"text": f"Unknown teleop name(s): {unknown}. Attached: {sorted(self._teleops)}"}],
            }

        # Connect selected devices NOW (lazy -> active). Fail loudly: a teleop
        # session with a dead leader is worse than a clean error.
        connect_errors = []
        for n in selected:
            att = self._teleops[n]
            try:
                if not getattr(att.device, "is_connected", False):
                    att.device.connect()
            except Exception as exc:  # noqa: BLE001 - surface as a clean status
                connect_errors.append(f"{n}: {exc}")
        if connect_errors:
            # Roll back any we connected, so a partial failure leaves no
            # half-open hardware.
            for n in selected:
                att = self._teleops[n]
                with contextlib.suppress(Exception):
                    if getattr(att.device, "is_connected", False):
                        att.device.disconnect()
            return {
                "status": "error",
                "content": [{"text": "Failed to connect teleoperator(s):\n  " + "\n  ".join(connect_errors)}],
            }

        # Optional mesh publish: delegate to the host's existing publisher so
        # the actuation stream rides the documented Mesh.publish() chokepoint.
        publish_results = []
        if publish:
            start_pub = getattr(self, "start_teleop_publish", None)
            if not callable(start_pub):
                # Roll back connects before erroring.
                for n in selected:
                    with contextlib.suppress(Exception):
                        if getattr(self._teleops[n].device, "is_connected", False):
                            self._teleops[n].device.disconnect()
                return {
                    "status": "error",
                    "content": [
                        {"text": "publish=True requires the host to expose start_teleop_publish (hardware Robot)."}
                    ],
                }
            for n in selected:
                att = self._teleops[n]
                res = start_pub(
                    teleoperator=att.device,
                    device_name=att.name,
                    method=att.method,
                    hz=hz,
                )
                publish_results.append(res)

        self._teleop_robot_name = robot_name
        self._teleop_stop_event.clear()
        self._teleop_frames = 0
        self._teleop_errors = 0
        self._teleop_slew_rejected = 0
        self._teleop_slew_baseline = {}
        # Resolve the mesh slew module before the session clock starts. The loop
        # judges every frame with helpers from strands_robots.mesh.security, which
        # it imports lazily (the mesh package reaches strands_robots.simulation,
        # which this mixin must not depend on - see the layering note in
        # _teleop_loop). Resolving it there put ~2s of one-time import cost on a
        # cold process INSIDE the window ``duration`` bounds, because the deadline
        # is ``self._teleop_start_mono + duration`` and that stamp is taken here,
        # before the loop runs at all. A session shorter than the import ended
        # without polling the leader once and still reported success; a longer one
        # was silently shortened by the same amount. Resolving it here - with the
        # rest of setup, which already includes connecting every device - leaves
        # the loop's import a sys.modules lookup, so ``duration`` measures
        # teleoperation rather than teleoperation plus setup. Stated as a call
        # rather than an unused ``import`` so the effect is the statement.
        importlib.import_module("strands_robots.mesh.security")

        self._teleop_start_mono = time.monotonic()
        self._teleop_running = True

        loop = lambda: self._teleop_loop(selected, robot_name, hz, duration)  # noqa: E731

        if block:
            try:
                loop()
            except KeyboardInterrupt:
                logger.info("[teleop] interrupted by user")
            finally:
                self._teleop_running = False
                # Block mode finished (duration elapsed / Ctrl+C): tear down to
                # the same clean state stop_teleoperate() leaves -- stop any
                # mesh publishers and disconnect every device we connected, so
                # a subsequent teleoperate() call starts fresh.
                self._stop_publishers()
                for _att in self._teleops.values():
                    with contextlib.suppress(Exception):
                        if getattr(_att.device, "is_connected", False):
                            _att.device.disconnect()
            return self._teleop_stats(blocking=True, publish_results=publish_results)

        self._teleop_thread = threading.Thread(
            target=loop, name=f"teleop-{getattr(self, 'tool_name_str', 'robot')}", daemon=True
        )
        self._teleop_thread.start()
        pub_note = f" (+{len(publish_results)} mesh publisher(s))" if publish else ""
        return {
            "status": "success",
            "content": [
                {
                    "text": f"Teleoperation started: driving {selected} @ {hz:.0f}Hz "
                    f"-> {self.tool_name_label(robot_name)}{pub_note}.\n"
                    f"Call stop_teleoperate() to stop."
                },
                {"json": {"devices": selected, "publish": publish}},
            ],
        }

    def stop_teleoperate(self) -> dict[str, Any]:
        """Stop the local teleop loop, any mesh publishers, and disconnect devices."""
        self._ensure_teleop_state()

        if not self._teleop_running and self._teleop_thread is None:
            # Still try to stop publishers in case publish=True was used.
            self._stop_publishers()
            return {"status": "success", "content": [{"text": "No active teleoperation."}]}

        self._teleop_running = False
        self._teleop_stop_event.set()
        if self._teleop_thread is not None:
            self._teleop_thread.join(timeout=3.0)
            self._teleop_thread = None

        self._stop_publishers()

        # Disconnect devices we connected.
        for n, att in self._teleops.items():
            with contextlib.suppress(Exception):
                if getattr(att.device, "is_connected", False):
                    att.device.disconnect()

        return self._teleop_stats(blocking=False)

    def get_teleoperate_status(self) -> dict[str, Any]:
        """Status of the local teleop loop (distinct from mesh get_teleop_status)."""
        self._ensure_teleop_state()
        elapsed = time.monotonic() - self._teleop_start_mono if self._teleop_start_mono else 0
        hz = self._teleop_frames / elapsed if elapsed > 0 else 0
        return {
            "status": "success",
            "content": [
                {
                    "text": f"Local teleop: running={self._teleop_running}, "
                    f"frames={self._teleop_frames}, errors={self._teleop_errors}, "
                    f"slew_rejected={self._teleop_slew_rejected}, "
                    f"hz={hz:.1f}, devices={list(self._teleops)}"
                },
                {
                    "json": {
                        "running": self._teleop_running,
                        "frames": self._teleop_frames,
                        "errors": self._teleop_errors,
                        "slew_rejected": self._teleop_slew_rejected,
                        "hz_actual": hz,
                        "devices": list(self._teleops),
                    }
                },
            ],
        }

    # --- internals --------------------------------------------------------

    def tool_name_label(self, robot_name: str | None) -> str:
        """Human label for the actuation target (sim may name a robot)."""
        base = getattr(self, "tool_name_str", type(self).__name__)
        return f"{base}/{robot_name}" if robot_name else base

    def _teleop_loop(
        self,
        selected: list[str],
        robot_name: str | None,
        hz: float,
        duration: float | None,
    ) -> None:
        # ``hz`` and ``duration`` are validated in :meth:`teleoperate` (the only
        # caller), so the division is safe. ``duration`` is read by membership,
        # not truthiness: a falsy-but-supplied value must not read as "absent".
        period = 1.0 / float(hz)
        deadline = (self._teleop_start_mono + duration) if duration is not None else None
        warned_conflicts: set[str] = set()

        # Per-joint slew bound, the same one the mesh receive path applies, so a
        # leader frame is judged identically whether it reaches a follower over
        # the network or on this host. ``teleoperate(publish=True)`` drives both
        # from one device, so without this the same frame was bounded on every
        # remote follower and unbounded on the local one. The bound is a speed
        # above what a leader arm's own servos can produce, so only a synthetic
        # or glitched frame trips it. The local path defaults to 500 units/s
        # (``STRANDS_TELEOP_SLEW_ABS``) so it accommodates degree-valued and
        # range-0-100 devices at their shipped defaults without env-var tuning.
        #
        # Imported here rather than at module scope because the mesh package
        # pulls :mod:`strands_robots.simulation` in, and this mixin must not
        # depend on it (see
        # :func:`strands_robots.utils.positive_finite_number_error`). One import
        # per session, not per tick - and :meth:`teleoperate` resolves the module
        # before it stamps the session clock, so this reads an already-imported
        # module rather than charging its cost to the caller's ``duration``.
        from strands_robots.mesh.security import (
            input_frame_slew_violation,
            merge_slew_baseline,
        )

        # This default is sized directly against the shipped SO hardware units:
        # the joints speak degrees (90 deg/s for a calm sweep, 372 deg/s for the
        # STS3215 no-load max) and the gripper speaks 0-100. 500 units/s is above
        # the fastest servo in any shipped unit system (deg, range-0-100, rad)
        # while still catching encoder glitches and full-scale jumps that would
        # strip gears.
        #
        # It stays a constant of its own now that the mesh path is frame-unit
        # scoped too, because the two bound different things rather than the same
        # thing at two scales: the mesh default is its whole value envelope
        # traversed once per second, deliberately loose enough to admit any
        # driver unit on a stream arriving from another machine, while this loop
        # reads a leader attached to this host whose unit system is known. So the
        # mesh bound is the looser of the two, and narrowing it here would be a
        # policy decision about remote streams made in the local path.
        _LOCAL_SLEW_DEFAULT = 500.0
        _local_slew_str = os.environ.get("STRANDS_TELEOP_SLEW_ABS", "")
        if _local_slew_str:
            try:
                _local_slew = float(_local_slew_str)
                if _local_slew <= 0 or not math.isfinite(_local_slew):
                    raise ValueError
            except (ValueError, TypeError):
                logger.warning(
                    "[teleop] STRANDS_TELEOP_SLEW_ABS=%r is not a positive finite number; using default %.1f",
                    _local_slew_str,
                    _LOCAL_SLEW_DEFAULT,
                )
                _local_slew = _LOCAL_SLEW_DEFAULT
        else:
            _local_slew = _LOCAL_SLEW_DEFAULT

        # Magnitude envelope for the baseline prune, and the reason it is
        # infinite here. ``merge_slew_baseline`` drops an entry once enough time
        # has passed that it "can no longer refuse anything", computing that
        # horizon as ``(value_abs + abs(value)) / max_slew`` - a premise that
        # holds only while the pruner and the checker are parameterised alike.
        # The mesh path pairs its bound with ``validate_input_frame``'s
        # magnitude clamp, so a permissible command there can only reach
        # ``STRANDS_MESH_INPUT_VALUE_ABS``. The local path runs no such clamp
        # (``input_frame_slew_violation`` takes no envelope, and nothing else
        # bounds a leader's reach), so no finite displacement exists after which
        # an entry provably cannot refuse: the only envelope that keeps the
        # prune a no-verdict-change operation is an unbounded one, under which
        # it prunes nothing.
        #
        # That costs no unbounded growth, because the prune's own motive does
        # not apply here. It exists because a *remote* stream chooses its key
        # names; these keys are the attached devices' motor names plus whatever
        # ``map_fn`` emits - a set fixed by the session's own hardware - and the
        # baseline is reset per session (see :meth:`teleoperate`). The size
        # bound is that key set, not a time window.
        _LOCAL_VALUE_ABS = math.inf

        # Paced by mesh.pacing.Ticker. Like InputPublisher._publish_loop this loop
        # already subtracted its own body from the period with perf_counter, so the
        # arithmetic was not what was wrong with it -- the wait that arithmetic fed
        # was. The subtraction now has one owner instead of a copy per loop, and on a
        # host that inflates Event.wait (see mesh.pacing) this loop no longer pays it.
        # Imported here rather than at module scope: this module is imported by
        # hardware paths that must not pull in the mesh package on import.
        from strands_robots.mesh.pacing import Ticker

        with Ticker(period, self._teleop_stop_event) as ticker:
            while self._teleop_running and not self._teleop_stop_event.is_set():
                # ``duration`` is an elapsed-time budget, so the deadline is
                # compared on the clock it was built from. Read on ``time.time()``
                # an NTP correction or a resume from suspend moved this comparison
                # by the size of the step: forward the session ended early with the
                # leader still held, and backward it kept driving the follower past
                # the budget the caller asked for. Neither was reported.
                if deadline is not None and time.monotonic() >= deadline:
                    logger.info("[teleop] duration elapsed (%.1fs); stopping", duration)
                    break

                merged: ActionDict = {}
                try:
                    for n in selected:
                        att = self._teleops[n]
                        action = att.device.get_action()
                        action = _normalize_action(action)
                        if att.map_fn is not None:
                            action = att.map_fn(action)
                        # Merge: last-wins, warn once per conflicting key.
                        for k, v in action.items():
                            if k in merged and k not in warned_conflicts:
                                logger.warning(
                                    "[teleop] key %r set by multiple devices; last-wins "
                                    "(device %r). Use map_fn to namespace if unintended.",
                                    k,
                                    n,
                                )
                                warned_conflicts.add(k)
                            merged[k] = v

                    if merged:
                        # Refuse-and-count, matching the mesh path: clamping toward
                        # the commanded value would silently alter an actuator
                        # command. The bound is measured per joint from that joint's
                        # last sent value, so the allowance grows while a joint is
                        # still and a refused stream resumes by itself once the
                        # commanded pose is reachable safely - no resync handshake.
                        apply_mono = time.perf_counter()
                        slew_reason = input_frame_slew_violation(
                            merged, self._teleop_slew_baseline, apply_mono, period, max_slew=_local_slew
                        )
                        if slew_reason is not None:
                            self._teleop_slew_rejected += 1
                            if self._teleop_slew_rejected <= 5:
                                logger.warning("[teleop] frame refused: %s", slew_reason)
                        else:
                            result = self.send_action(merged, robot_name=robot_name)
                            if isinstance(result, dict) and result.get("status") == "error":
                                self._teleop_errors += 1
                                if self._teleop_errors <= 5:
                                    txt = result.get("content", [{}])[0].get("text", "")
                                    logger.warning("[teleop] send_action error: %s", txt)
                            # Explicitly parameterised, like the check above: a
                            # mesh default reaching either call site describes a
                            # bound this path does not enforce.
                            self._teleop_slew_baseline = merge_slew_baseline(
                                self._teleop_slew_baseline,
                                merged,
                                apply_mono,
                                max_slew=_local_slew,
                                value_abs=_LOCAL_VALUE_ABS,
                            )
                            self._teleop_frames += 1
                except Exception as exc:  # noqa: BLE001 - hot loop, count + rate-limit
                    self._teleop_errors += 1
                    if self._teleop_errors <= 5:
                        logger.warning("[teleop] loop error: %s", exc)

                if ticker.wait():
                    break

        self._teleop_running = False
        logger.info("[teleop] loop stopped (%d frames, %d errors)", self._teleop_frames, self._teleop_errors)

    def _stop_publishers(self) -> None:
        """Stop any mesh publishers we started (publish=True path)."""
        stop_pub = getattr(self, "stop_teleop", None)
        if callable(stop_pub):
            with contextlib.suppress(Exception):
                stop_pub()  # stops all publishers/receivers on the host

    def _teleop_stats(self, *, blocking: bool, publish_results: list | None = None) -> dict[str, Any]:
        elapsed = time.monotonic() - self._teleop_start_mono if self._teleop_start_mono else 0
        hz = self._teleop_frames / elapsed if elapsed > 0 else 0
        note = ""
        if publish_results:
            note = f"\nMesh publishers started: {len(publish_results)}"
        # Derive the session-end status from the counters instead of hardcoding
        # "success", so a dead teleop is not reported as healthy. Two failure
        # modes with distinct counter signatures (see _teleop_loop):
        #   soft: send_action returns {"status": "error"} -> errors += 1 AND
        #         frames += 1 (an unpowered follower gives errors == frames)
        #   hard: get_action()/send raises -> errors += 1 only, no frame that
        #         tick (a dead leader gives frames == 0)
        # A third signature is a *refused* frame: one the per-joint slew bound
        # would not let reach the follower. That is not an error - nothing
        # failed - but it is not a healthy frame either, so it is counted
        # separately and still moves the session off "success". A leader whose
        # every frame is refused (a device whose units are larger than the
        # bound expects) otherwise reports 0 frames, 0 errors and "success":
        # a silent no-op, which is the outcome this derivation exists to refuse.
        frames, errors = self._teleop_frames, self._teleop_errors
        refused = self._teleop_slew_rejected
        if errors == 0 and refused == 0:
            status = "success"  # clean run (or idle: no actions attempted)
        elif frames == 0 or errors >= frames:
            status = "error"  # every attempt failed or was refused
        else:
            status = "degraded"  # some ok, some failed or refused
        telemetry = {
            "frames": frames,
            "errors": errors,
            "slew_rejected": refused,
            "hz_actual": hz,
            "elapsed_s": elapsed,
            "status": status,
            "blocking": blocking,
            "publish_count": len(publish_results) if publish_results else 0,
        }
        return {
            "status": status,
            "content": [
                {
                    "text": f"Teleoperation {'completed' if blocking else 'stopped'}: "
                    f"{frames} frames, {errors} errors, "
                    f"{hz:.1f}Hz over {elapsed:.1f}s."
                    f"{f' {refused} frame(s) refused by the slew bound.' if refused else ''}{note}"
                },
                {"json": telemetry},
            ],
        }


def _infer_method(teleop_type: str) -> str:
    """Map a teleoperator type string to an input-method label."""
    t = teleop_type.lower()
    if "gamepad" in t:
        return "gamepad"
    if "keyboard" in t:
        return "keyboard"
    if "phone" in t:
        return "phone"
    # leaders, gloves, arms, default
    return "arm"


def _normalize_action(action: Any) -> ActionDict:
    """Convert a teleoperator action to a flat ``{str: float}`` dict.

    Mirrors ``InputPublisher._normalize_action`` so local and mesh paths agree
    on the wire shape. lerobot leaders already return ``{f'{motor}.pos': float}``.
    """
    if isinstance(action, dict):
        result: ActionDict = {}
        for k, v in action.items():
            if hasattr(v, "item"):
                result[k] = float(v.item())
            else:
                result[k] = float(v)
        return result
    if hasattr(action, "tolist"):
        # ``tolist()`` flattens an ndarray/tensor to nested Python lists, but a
        # numpy/torch *scalar* or 0-d array returns a bare Python number, not a
        # list -- enumerating that raises ``'float' object is not iterable``.
        # Treat a non-list result as a single-DOF scalar so a 1-DOF leader does
        # not crash the teleop loop.
        arr = action.tolist()
        if isinstance(arr, list):
            return {f"j{i}": float(v) for i, v in enumerate(arr)}
        return {"raw": float(arr)}
    return {"raw": float(action)}


__all__ = ["TeleopMixin", "AttachedTeleop"]
