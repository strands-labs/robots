"""What a robot is: its model and the facts a world or a bus needs about it."""

from __future__ import annotations

from .description import Description, Gripper, Hardware, default_id
from .registry import device_types, normalize_name, registry, urdf_registry

__all__ = [
    "Description",
    "Gripper",
    "Hardware",
    "default_id",
    "device_types",
    "normalize_name",
    "registry",
    "urdf_registry",
]
