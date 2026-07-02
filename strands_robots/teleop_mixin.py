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
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
            self._teleop_start_time: float = 0.0

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
        """
        self._ensure_teleop_state()

        names = [name] if name else list(self._teleops)
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
            "content": [{"text": f"Attached teleoperators ({len(self._teleops)}):\n{body}"}],
            "teleops": list(self._teleops),
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
            names: Subset of attached device names to drive. ``None`` = all.
            robot_name: Target robot for ``send_action``. ``None`` -> the
                host's default (single hardware robot, or first sim robot).
                In a multi-robot sim, name the specific robot.
            hz: Local control-loop frequency.
            publish: Also publish each selected device to the mesh via the
                host's ``start_teleop_publish`` so remote peers can follow.
                Requires the host to expose ``start_teleop_publish`` and a
                live mesh.
            block: Run the loop in the calling thread (blocks until
                ``duration`` elapses or KeyboardInterrupt). When ``False``
                (default) the loop runs in a managed background thread and the
                call returns immediately with a handle/status.
            duration: Stop automatically after N seconds. ``None`` = run until
                ``stop_teleoperate()`` (background) / Ctrl+C (block).

        Returns:
            Status dict. Background mode returns immediately; ``block=True``
            returns after the loop ends with frame/error stats.
        """
        self._ensure_teleop_state()

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

        selected = names or list(self._teleops)
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
        self._teleop_start_time = time.time()
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
                }
            ],
            "devices": selected,
            "publish": publish,
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
        elapsed = time.time() - self._teleop_start_time if self._teleop_start_time else 0
        hz = self._teleop_frames / elapsed if elapsed > 0 else 0
        return {
            "status": "success",
            "running": self._teleop_running,
            "frames": self._teleop_frames,
            "errors": self._teleop_errors,
            "hz_actual": hz,
            "devices": list(self._teleops),
            "content": [
                {
                    "text": f"Local teleop: running={self._teleop_running}, "
                    f"frames={self._teleop_frames}, errors={self._teleop_errors}, "
                    f"hz={hz:.1f}, devices={list(self._teleops)}"
                }
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
        period = 1.0 / hz if hz > 0 else 0.0
        deadline = (self._teleop_start_time + duration) if duration else None
        warned_conflicts: set[str] = set()

        while self._teleop_running and not self._teleop_stop_event.is_set():
            loop_start = time.perf_counter()
            if deadline is not None and time.time() >= deadline:
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
                    result = self.send_action(merged, robot_name=robot_name)
                    if isinstance(result, dict) and result.get("status") == "error":
                        self._teleop_errors += 1
                        if self._teleop_errors <= 5:
                            txt = result.get("content", [{}])[0].get("text", "")
                            logger.warning("[teleop] send_action error: %s", txt)
                    self._teleop_frames += 1
            except Exception as exc:  # noqa: BLE001 - hot loop, count + rate-limit
                self._teleop_errors += 1
                if self._teleop_errors <= 5:
                    logger.warning("[teleop] loop error: %s", exc)

            elapsed = time.perf_counter() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                self._teleop_stop_event.wait(sleep_time)

        self._teleop_running = False
        logger.info("[teleop] loop stopped (%d frames, %d errors)", self._teleop_frames, self._teleop_errors)

    def _stop_publishers(self) -> None:
        """Stop any mesh publishers we started (publish=True path)."""
        stop_pub = getattr(self, "stop_teleop", None)
        if callable(stop_pub):
            with contextlib.suppress(Exception):
                stop_pub()  # stops all publishers/receivers on the host

    def _teleop_stats(self, *, blocking: bool, publish_results: list | None = None) -> dict[str, Any]:
        elapsed = time.time() - self._teleop_start_time if self._teleop_start_time else 0
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
        frames, errors = self._teleop_frames, self._teleop_errors
        if errors == 0:
            status = "success"  # clean run (or idle: no actions attempted)
        elif frames == 0 or errors >= frames:
            status = "error"  # every attempt failed, either mode
        else:
            status = "degraded"  # some ok, some failed
        return {
            "status": status,
            "content": [
                {
                    "text": f"Teleoperation {'completed' if blocking else 'stopped'}: "
                    f"{self._teleop_frames} frames, {self._teleop_errors} errors, "
                    f"{hz:.1f}Hz over {elapsed:.1f}s.{note}"
                }
            ],
            "frames": self._teleop_frames,
            "errors": self._teleop_errors,
            "hz_actual": hz,
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
