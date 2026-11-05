"""The command gate for raw ROS writes: what an agent may publish, call or send as a goal.

Reads are never gated. A write aimed at a motion surface needs an exact-name allowlist entry or an
operator's yes; with neither it is refused (fail closed). A topic a strands Robot streams is refused
outright: that robot's own gate (``robot.act``) is the only way to command it. STOP is never gated; it
goes through the profile's stop sequence, not through an agent-chosen payload.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .names import canonical

AUDIT = logging.getLogger("strands_robots.audit")

# Path suffixes (matched on whole segments, so "/cmd_vel" also covers "/robot1/cmd_vel").
MOTION_SURFACES: tuple[str, ...] = (
    "/cmd_vel",
    "/cmd_vel_unstamped",
    "/manual_drive",
    "/joint_command",
    "/joint_trajectory",
    "/joint_trajectory_controller/joint_trajectory",
    "/follow_joint_trajectory",
    "/commands",
    "/emergency_stop",
    "/e_stop",
    "/motor_enable",
    "/enable_motor",
    "/disable_motor",
    "/vehicle_state",
    "/enable_state",
    "/navigate_to_pose",
    "/follow_path",
    "/initialpose",
    "/lowcmd",
    "/user_lowcmd",
    "/arm_sdk",
    "/armsdk",
    "/api/sport/request",
    "/api/loco/request",
    "/api/arm/request",
    "/api/motion_switcher/request",
    # parameters: a node's live limits and gains (velocity caps, safety toggles), set through its services
    "/set_parameters",
    "/set_parameters_atomically",
    "/rosapi/set_param",
    # ros2_control's controller manager: activating a controller or a hardware component starts motion
    "/switch_controller",
    "/load_controller",
    "/configure_controller",
    "/unload_controller",
    "/set_hardware_component_state",
)

Approver = Callable[[str, str, Mapping[str, Any]], bool]


class CommandRefused(PermissionError):
    """The gate said no; the text says why and what would allow it."""


def _segments(name: str) -> tuple[str, ...]:
    return tuple(s for s in name.split("/") if s)


def matches(name: str, surfaces: Iterable[str]) -> str | None:
    """The surface ``name`` ends with (on whole segments), or None."""
    segs = _segments(canonical(name))
    for s in surfaces:
        tail = _segments(s)
        if tail and segs[-len(tail) :] == tail:
            return s
    return None


def env_allowlist() -> frozenset[str]:
    """``STRANDS_ROS_ALLOW=/a/cmd_vel,/b/cmd_vel``: exact names only (a base-name match is fail-open)."""
    raw = os.environ.get("STRANDS_ROS_ALLOW", "")
    return frozenset(canonical(x.strip()) for x in raw.split(",") if x.strip())


@dataclass
class CommandGate:
    surfaces: tuple[str, ...] = MOTION_SURFACES
    allow: frozenset[str] = field(default_factory=env_allowlist)
    approve: Approver | None = None
    owners: Mapping[str, str] = field(default_factory=dict)
    source: str = "use_ros"

    def check(self, verb: str, target: str, preview: Mapping[str, Any] | None = None) -> str:
        """Return how the write was allowed ("free" / "allowlist" / "operator"); raise ``CommandRefused``."""
        target = canonical(target, what=verb)
        owner = self.owners.get(target)
        if owner is not None:
            self._audit(verb, target, "refused", f"owned by {owner}")
            raise CommandRefused(
                f"{target} is streamed by robot {owner!r}; command it with that robot "
                f"(robot.act / its tool), which runs its safety chain"
            )
        hit = matches(target, self.surfaces)
        if hit is None:
            return "free"
        if target in self.allow:
            self._audit(verb, target, "allowed", "allowlist")
            return "allowlist"
        if self.approve is None:
            self._audit(verb, target, "refused", "no operator")
            raise CommandRefused(
                f"{verb} on {target} can move a robot (matches {hit}) and no operator can be asked here. "
                f"Allow it by exact name with STRANDS_ROS_ALLOW={target}, or run with an operator who can approve."
            )
        if not self.approve(verb, target, dict(preview or {})):
            self._audit(verb, target, "refused", "operator said no")
            raise CommandRefused(f"the operator declined {verb} on {target}")
        self._audit(verb, target, "allowed", "operator")
        return "operator"

    def _audit(self, verb: str, target: str, outcome: str, why: str) -> None:
        AUDIT.info("%s %s %s %s (%s)", self.source, verb, target, outcome, why)
