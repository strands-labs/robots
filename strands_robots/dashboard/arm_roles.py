"""Leader/follower verdicts from a servo bus reading: which arm is which."""

from __future__ import annotations

from collections.abc import Mapping
from statistics import median
from typing import Any

# : Above this, the bus is on the 12V supply → follower.
FOLLOWER_MIN_V = 9.0
# : Below this the arm is NOT on its own supply.
POWERED_MIN_V = 6.5
#: What an unpowered Feetech bus reads through USB alone, for the message.
USB_RAIL_V = 5.5
#: Wider than any real supply ripple; a spread this large means the readings are
#: not describing one bus.
SPREAD_MAX_V = 1.5


def classify_role(volts: float | None) -> tuple[str, str]:
    """One reading → (role, reason). Roles: follower / leader / unpowered / unknown."""
    if volts is None:
        return "unknown", "no voltage could be read from the bus"
    if volts >= FOLLOWER_MIN_V:
        return "follower", f"{volts:.1f}V bus - the 12V supply, which is the follower arm"
    if volts >= POWERED_MIN_V:
        return "leader", f"{volts:.1f}V bus - the 7.4V supply, which is the leader arm"
    if volts >= USB_RAIL_V - 0.5:
        return (
            "unpowered",
            f"{volts:.1f}V - that is the USB logic rail, not a supply: this arm's power "
            f"is off. A leader's 7.4V pack reads 6.6-8.4V and a follower's 12V reads "
            f"10.5-12.7V, so the role cannot be read until it is powered",
        )
    return (
        "unpowered",
        f"{volts:.1f}V - this arm is not on its power supply, so its role cannot be read "
        f"(an unpowered arm reads like nothing at all, not like a leader)",
    )


def role_verdict(readings: Mapping[Any, float | None]) -> dict[str, Any]:
    """Several servos on one bus → one verdict, or an honest refusal."""
    good = {k: float(v) for k, v in readings.items() if isinstance(v, (int, float)) and v is not None}
    if not good:
        return {
            "role": "unknown",
            "reason": "no servo answered a voltage read",
            "remedy": "check the USB cable and that the arm's power supply is on, then retry",
            "volts": None,
            "readings": dict(readings),
        }
    values = sorted(good.values())
    mid = float(median(values))
    spread = values[-1] - values[0]
    if spread > SPREAD_MAX_V:
        return {
            "role": "mixed",
            "reason": (
                f"servos on this bus report different supplies ({values[0]:.1f}V to "
                f"{values[-1]:.1f}V) - they share one supply, so this is a fault, not a role"
            ),
            "remedy": "check the power wiring / daisy-chain, then retry before trusting any role",
            "volts": mid,
            "spread": round(spread, 2),
            "readings": good,
        }
    role, reason = classify_role(mid)
    verdict: dict[str, Any] = {
        "role": role,
        "reason": reason,
        "volts": round(mid, 2),
        "spread": round(spread, 2),
        "readings": good,
        "motors_answered": len(good),
    }
    if role == "unpowered":
        verdict["remedy"] = "switch on / plug in this arm's power supply, then retry"
    return verdict


def disagreement(profile_role: str | None, measured: Mapping[str, Any]) -> dict[str, Any] | None:
    """The point of the whole exercise: does the label match the hardware?"""
    measured_role = measured.get("role")
    if not profile_role or measured_role not in ("leader", "follower"):
        return None
    if profile_role == measured_role:
        return None
    return {
        "labelled": profile_role,
        "measured": measured_role,
        "message": (
            f"this arm is labelled {profile_role} but its bus reads "
            f"{measured.get('volts')}V, which is the {measured_role} arm"
        ),
        "remedy": f"relabel it {measured_role} before teleop or recording - the roles decide which arm is driven",
    }
