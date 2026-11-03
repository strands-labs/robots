"""Units: a policy declares what it speaks (``policy.units``), the robot's description converts.

    conv = Units.for_robot(robot.description, "lerobot")
    q_policy = conv.from_rad(obs.joints.positions)     # what the policy should see
    q_rad    = conv.to_rad(policy_action)              # what Robot.act should receive

``Robot.run(policy, units="auto")`` does this around every ``act``. ``units="rad"`` or a scheme name says
what a policy that cannot say speaks; a policy that declares its units keeps them (a scheme that agrees gets
the policy's own map, a different one is refused).
"""

from __future__ import annotations

from . import lerobot, rad  # noqa: F401  (register the schemes)
from .lerobot import calibrated, lerobot_units
from .scheme import SCHEMES, Units, declared_units, register, resolve_units
from .semantics import semantics_for, units_for_semantics

__all__ = [
    "Units",
    "SCHEMES",
    "register",
    "resolve_units",
    "declared_units",
    "lerobot_units",
    "calibrated",
    "units_for_semantics",
    "semantics_for",
]
