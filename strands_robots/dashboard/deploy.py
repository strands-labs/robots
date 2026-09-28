"""Render a spawn payload or remembered profile as a deployable Python script.

The snippet recreates one rig on another machine: same robot, same port shape,
same cameras, same mesh posture, pointed at this dashboard's zenoh hub.
"""

from __future__ import annotations

import ipaddress
import re
import time
from collections.abc import Mapping
from typing import Any

__all__ = ["render_snippet", "snippet_filename"]

#: Env the dashboard's own spawner sets. Rendered as setdefault so an edge
#: box with a deliberate different posture (real ACL file, mTLS) wins.
_MESH_ENV: tuple[tuple[str, str], ...] = (
    ("STRANDS_ROBOTS_NO_DYLD_SHIM", "1"),
    ("STRANDS_MESH_LOCAL_DEV", "1"),
    ("STRANDS_MESH_MULTICAST", "true"),
    ("STRANDS_MESH", "true"),
    ("STRANDS_MESH_CAMERA_HZ", "5"),
)

#: Default zenoh port, mirroring ``strands_robots.mesh.session``.
DEFAULT_HUB_PORT = 7447

_LIVE_KEYS: frozenset[str] = frozenset({"STRANDS_MESH_CAMERA_HZ", "STRANDS_MESH_MULTICAST"})

_SECURITY_LOOSENING_KEYS: frozenset[str] = frozenset({"STRANDS_MESH_LOCAL_DEV"})


def resolve_mesh_env(env: Mapping[str, str] | None) -> list[tuple[str, str]]:
    """The env block to render, taking live values over the frozen defaults."""
    live = env or {}
    out: list[tuple[str, str]] = []
    for key, default in _MESH_ENV:
        value = str(live.get(key, "")).strip()
        if key in _SECURITY_LOOSENING_KEYS:
            if value:
                out.append((key, value))
            continue
        if key in _LIVE_KEYS and value:
            out.append((key, value))
        else:
            out.append((key, default))
    return out


#: What a hub host may look like in the rendered file: a hostname, an IPv4, or a bracketed
#: IPv6. Anything else is refused rather than rendered, because the value lands inside a
#: Python string literal in a script the operator is told to run.
_HOST_RE = re.compile(r"^[A-Za-z0-9.:\[\]-]{1,253}\Z")

#: What a caller-chosen peer id may look like - the same rule ``device_manager.validate_peer_id``
#: applies at spawn, spelled here so the snippet module stands on its own.
_PEER_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}\Z")


def hub_host_error(hub_host: Any) -> str | None:
    """Why this hub host must not be rendered, or None if it is a plain host/IP."""
    text = str(hub_host or "").strip()
    if not text:
        return None
    if not _HOST_RE.match(text):
        return "hub_host must be a hostname, an IPv4 address or a bracketed IPv6 address"
    return None


def _doc(value: Any) -> str:
    """A payload-derived value made safe inside the module docstring.

    Backslashes are doubled and a triple quote is broken so nothing a client, a USB
    descriptor or an agent put in a field can close the docstring and become code;
    control characters become spaces so a value cannot start a new line either.
    """
    text = str(value).replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return "".join(ch if ch.isprintable() else " " for ch in text)


def snippet_filename(peer_id: str) -> str:
    """A filename safe to offer as a download, derived from the peer id."""
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in peer_id)
    return f"{safe or 'robot'}.py"


def _fmt(value: Any, indent: int = 0) -> str:
    """Render a payload value as Python source (JSON types only). ``repr`` is almost right but renders
    dict one-line; camera configs are nested and read better expanded.
    """
    if isinstance(value, Mapping):
        if not value:
            return "{}"
        pad = " " * (indent + 4)
        items = ",\n".join(f"{pad}{k!r}: {_fmt(v, indent + 4)}" for k, v in value.items())
        return "{\n" + items + ",\n" + " " * indent + "}"
    return repr(value)


def hub_host_from_reached(reached_on: str | None) -> tuple[str | None, str | None]:
    """Is the host the browser reached this dashboard on usable as a ZENOH hub address?"""
    reached = (reached_on or "").strip().strip("[]").lower()
    if not reached:
        return None, None
    if reached in ("localhost", "127.0.0.1", "::1") or reached.endswith(".localhost"):
        return None, (
            "you opened this dashboard on localhost, and on the edge device that name means the edge device itself"
        )
    try:
        ip = ipaddress.ip_address(reached)
    except ValueError:
        # A NAME. Treat the LAN-shaped ones as usable; anything else is an entry point, not a host.
        if reached.endswith((".local", ".lan", ".internal", ".home", ".arpa")) or "." not in reached:
            return reached, None
        return None, (
            f"you reached this dashboard at {reached}, which is a public name - usually a tunnel or "
            "reverse proxy that forwards HTTP only. The zenoh hub is a raw TCP port and is almost "
            "certainly not published there, so a peer pointed at it would connect to nothing"
        )
    if ip.is_loopback:
        return None, "that address is loopback: on the edge device it would mean the edge device"
    if ip.is_private or ip.is_link_local:
        return reached, None
    return None, (
        f"{reached} is a public address; the zenoh hub should be reached over your own network, not "
        "the internet. Use this machine's LAN address (the Mesh tab shows it)"
    )


def _stamped_names(cameras: Any) -> dict[str, str]:
    """{camera: the roster name this index carried when it was configured}, for the cameras that have one."""
    if not isinstance(cameras, Mapping):
        return {}
    out: dict[str, str] = {}
    for cam, cfg in cameras.items():
        if isinstance(cfg, Mapping):
            was = cfg.get("device_name")
            if isinstance(was, str) and was.strip():
                out[str(cam)] = was.strip()
    return out


def render_snippet(
    payload: Mapping[str, Any],
    *,
    hub_host: str | None = None,
    hub_note: str | None = None,
    mesh_env: Mapping[str, str] | None = None,
    hub_port: int | str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Render a spawn payload/profile as a deployable Python script."""
    robot_name = str(payload.get("robot_name") or "").strip()
    if not robot_name:
        return {"error": "payload has no robot_name"}
    mode = str(payload.get("mode") or "").strip()
    if mode not in ("real", "sim"):
        return {"error": f"mode must be real or sim, got {mode!r}"}
    port = payload.get("port")
    if mode == "real" and not port:
        return {"error": "port required for mode=real"}
    peer_id = str(payload.get("peer_id") or payload.get("name") or "").strip()
    if not peer_id:
        return {"error": "payload has no peer_id"}
    if not _PEER_ID_RE.match(peer_id):
        return {"error": "peer_id must match [A-Za-z0-9._:-]{1,64}"}
    if (host_problem := hub_host_error(hub_host)) is not None:
        return {"error": host_problem}

    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now if now is not None else time.time()))
    name = payload.get("name") or peer_id

    lines: list[str] = [
        "#!/usr/bin/env python3",
        f'"""{_doc(name)} - generated by the strands-robots dashboard, {stamp}.',
    ]
    lines += [
        "",
        "Recreates this exact rig on any machine with strands-robots installed:",
        "",
        "    pip install strands-robots",
        f"    python {snippet_filename(peer_id)}",
        "",
    ]
    role = payload.get("role")
    if role in ("leader", "follower") and payload.get("role_source") == "measured":
        volts = payload.get("role_volts")
        volts_txt = f" at {volts}V" if volts is not None else ""
        lines.append(f"This arm measured as the {role.upper()}{_doc(volts_txt)} on its servo bus.")
        lines.append("")
    if payload.get("cameras"):
        lines.append("Camera indices are PER-MACHINE: index 1 here may be index 0 on the")
        lines.append("edge device. Re-check them there (lerobot-find-cameras opencv).")
        # The roster name this index carried when it was configured is the ONE thing that makes that
        # re-check possible: "index 0" is a position in a list that closes up when a device is
        # unplugged, while "USB2.0_CAM1" is a camera.
        for cam, was in sorted(_stamped_names(payload.get("cameras")).items()):
            lines.append(
                f'  {_doc(cam)}: was "{_doc(was)}" on the dashboard machine - check that camera is still that index.'
            )
        lines.append("")
    if mode == "real":
        serial = payload.get("serial_number")
        lines.append(f"The port below ({_doc(port)}) is how THIS machine names that USB device.")
        lines.append("On Linux the same arm is usually /dev/ttyACM0 - check with")
        lines.append("`lerobot-find-port`, or list /dev/serial/by-id/ where the name is stable.")
        if serial and serial != port:
            # The serial IS the identity the dashboard itself keys profiles by, so it is the one
            # string that survives the move. Saying so turns a reboot-shuffled port from a
            # mystery into a lookup.
            lines.append(f"That device's USB serial is {_doc(serial)} - unlike the port, it does not")
            lines.append("change when the device is replugged or the machine reboots.")
        lines.append("")
    lines[-1] = lines[-1] if lines[-1] else '"""'
    if lines[-1] != '"""':
        lines.append('"""')
    lines += ["", "import os", "import time", ""]

    lines.append("# The mesh posture this dashboard runs with (setdefault: your own env wins).")
    for key, val in resolve_mesh_env(mesh_env):
        lines.append(f"os.environ.setdefault({key!r}, {str(val)!r})")
    try:
        port_txt = str(int(str(hub_port))) if hub_port not in (None, "") else str(DEFAULT_HUB_PORT)
    except (TypeError, ValueError):
        port_txt = str(DEFAULT_HUB_PORT)
    if hub_host:
        lines.append("# Reach this dashboard's zenoh hub from the edge device:")
        lines.append(f"os.environ.setdefault('ZENOH_CONNECT', {f'tcp/{str(hub_host).strip()}:{port_txt}'!r})")
    else:
        if hub_note:
            # The operator asked for a deployable file and got a commented-out line; without the
            # reason, the only reading is "the dashboard forgot".
            lines.append(f"# NOTE: {hub_note}.")
        lines.append("# Deploying to ANOTHER machine? Point the peer at this dashboard's hub:")
        lines.append(f'# os.environ.setdefault("ZENOH_CONNECT", "tcp/<dashboard-host>:{port_txt}")')
    lines += ["", "from strands_robots import Robot", "", "robot = Robot("]
    lines.append(f"    {robot_name!r},")
    lines.append(f"    mode={mode!r},")
    if mode == "real":
        serial = payload.get("serial_number")
        comment = f"  # USB serial {_doc(serial)}" if serial and serial != port else ""
        lines.append(f"    port={str(port)!r},{comment}")
    if payload.get("robot_id"):
        lines.append(f"    id={str(payload['robot_id'])!r},  # lerobot calibration identity")
    cameras = payload.get("cameras")
    if cameras:
        # STRIPPED, not rendered verbatim: a spawn payload or remembered profile carries the
        # dashboard's own bookkeeping key (device_name), the child never sees it (the spawner strips
        # it before Popen for exactly this reason), and hardware_robot refuses an unknown camera
        # option BY NAME - so a snippet generated for a camera-stamped arm produced a file that died
        # at connect with "Unknown option(s) for camera 'main': ['device_name']".
        from strands_robots.dashboard import camera_liveness

        lines.append(f"    cameras={_fmt(camera_liveness.without_annotations(cameras), 4)},")
    lines.append("    mesh=True,")
    lines.append(f"    peer_id={peer_id!r},")
    lines.append(")")
    if mode == "real":
        lines += [
            "",
            "# Connect eagerly so joints + camera frames publish immediately (a camera",
            "# this machine will not open costs the camera, not the whole arm).",
            "ok, degraded, err = robot.connect_eagerly()",
            "for cam, reason in (degraded or {}).items():",
            '    print(f"camera {cam!r} unavailable, dropped: {reason}")',
            "if not ok:",
            '    print(f"eager connect failed (will retry on first task): {err}")',
        ]
    lines += [
        "",
        f"print({f'{peer_id} online - visible on the dashboard fleet'!r})",
        "while True:",
        "    time.sleep(1)",
        "",
    ]
    return {
        "snippet": "\n".join(lines),
        "filename": snippet_filename(peer_id),
        "peer_id": peer_id,
    }
