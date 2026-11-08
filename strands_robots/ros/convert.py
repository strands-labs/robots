"""Messages <-> plain data: dynamic types, dict payloads, and short previews an agent can read.

Types are resolved at runtime with ``rosidl_runtime_py`` so any installed interface works, including a
vendor's private messages (``unitree_hg``, ``unitree_go``, your own) that strands-robots never ships.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .names import RosUnavailable, interface_type

MAX_LIST = 32  # list fields longer than this are summarised in previews
_BIG = {"data"}  # byte blobs (images, point clouds, audio): shape only, never the bytes


def _utilities() -> Any:
    try:
        from rosidl_runtime_py import utilities
    except ImportError as e:  # noqa: F841
        from .names import INSTALL_HINT

        raise RosUnavailable(INSTALL_HINT) from None
    return utilities


def message_class(t: str) -> Any:
    t = interface_type(t, kind="msg")
    try:
        return _utilities().get_message(t)
    except (AttributeError, ModuleNotFoundError, ValueError) as e:
        raise RosUnavailable(f"message type {t} is not installed here ({e}); source the robot's workspace") from None


def service_class(t: str) -> Any:
    return _utilities().get_service(interface_type(t, kind="srv"))


def action_class(t: str) -> Any:
    return _utilities().get_action(interface_type(t, kind="action"))


def type_name(msg_or_cls: Any) -> str:
    """``pkg/msg/Name`` for a message instance or class."""
    cls = msg_or_cls if isinstance(msg_or_cls, type) else type(msg_or_cls)
    pkg = cls.__module__.split(".")[0]
    return f"{pkg}/msg/{cls.__name__}"


def from_dict(cls: Any, fields: Mapping[str, Any] | None) -> Any:
    """``cls()`` with ``fields`` applied the standard ROS 2 way (nested dicts, lists, unknown keys refused)."""
    msg = cls()
    if fields:
        try:
            from rosidl_runtime_py.set_message import set_message_fields
        except ImportError:
            raise RosUnavailable("rosidl_runtime_py is not importable") from None
        set_message_fields(msg, dict(fields))
    return msg


def to_data(msg: Any, *, max_list: int | None = None) -> Any:
    """A message as JSON-ready data. ``max_list`` truncates long arrays (previews); None keeps everything."""
    if msg is None or isinstance(msg, bool | int | float | str):
        return msg
    if isinstance(msg, bytes | bytearray | memoryview):
        return {"bytes": len(msg)}
    slots = getattr(type(msg), "get_fields_and_field_types", None)
    if slots is not None:
        out: dict[str, Any] = {}
        for f in slots():
            v = getattr(msg, f)
            if max_list is not None and f in _BIG and _length(v) > max_list:
                out[f] = {"len": _length(v)}
            else:
                out[f] = to_data(v, max_list=max_list)
        return out
    if hasattr(msg, "tolist"):
        msg = msg.tolist()
    if isinstance(msg, list | tuple):
        items = [to_data(v, max_list=max_list) for v in msg[: max_list or None]]
        if max_list is not None and len(msg) > max_list:
            items.append(f"... {len(msg) - max_list} more")
        return items
    return str(msg)


def _length(v: Any) -> int:
    try:
        return len(v)
    except TypeError:
        return 0


def field_types(cls: Any) -> dict[str, str]:
    """Field name -> ROS type string, for ``use_ros info`` and for profile checks against a robot's msgs."""
    get = getattr(cls, "get_fields_and_field_types", None)
    return dict(get()) if get else {}


def stamp_seconds(stamp: Any) -> float | None:
    """A ``builtin_interfaces/Time`` as float seconds (None if absent)."""
    if stamp is None:
        return None
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9
