"""Universal Teleoperator factory - sibling of ``strands_robots.robot.Robot``.

Mirrors the :func:`strands_robots.robot.Robot` factory but for *input
devices* (leader arms, gamepads, keyboards, phones, ...). Resolves the
concrete lerobot ``Teleoperator`` via lerobot's draccus
``TeleoperatorConfig`` ChoiceRegistry, so every teleoperator lerobot ships
- past, present, and future - is available with zero hardcoded mapping.

Examples::

    from strands_robots import Teleoperator

    leader = Teleoperator("so101_leader", port="/dev/ttyACM1", id="blue")
    pad = Teleoperator("gamepad")

    # Attach to any robot/sim (see TeleopMixin.attach_teleop):
    follower = Robot("so101", mode="real", port="/dev/ttyACM0")
    follower.attach_teleop(leader).attach_teleop(pad, name="pad")
    follower.teleoperate()

The returned object is a raw lerobot ``Teleoperator`` - it duck-types to
``get_action() -> dict``, ``connect()``, ``disconnect()``,
``is_connected`` - so it drops straight into the existing
``InputPublisher`` / ``TeleopMixin`` machinery.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import pkgutil
from functools import cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lerobot.teleoperators.teleoperator import Teleoperator as LeRobotTeleoperator

logger = logging.getLogger(__name__)

# Union-of-all known lerobot TeleoperatorConfig fields. Mirrors
# hardware_robot._FORWARDABLE_KWARGS: a kwarg here is forwarded only if the
# resolved config dataclass actually declares it (cross-device polymorphism),
# and a kwarg in NEITHER this allowlist NOR the dataclass is rejected as a
# typo. Derived by scanning every @TeleoperatorConfig.register_subclass site.
_FORWARDABLE_TELEOP_KWARGS = (
    "port",  # serial leaders (so100/so101, koch, omx, openarm, ...)
    "baud_rate",
    "baudrate",
    "use_degrees",  # leader arms reporting joint angles
    "use_gripper",  # gamepad / keyboard_ee
    "use_velocity_and_torque",
    "use_present_position",
    "ip_address",  # network teleop (reachy2, unitree)
    "phone_os",  # phone teleop
    "side",  # bimanual (left/right)
    "left_arm_config",
    "right_arm_config",
    "joint_ids",
    "joint_ranges",
    "joint_directions",
    "frozen_joints",
    "gripper_open_pos",
    "motor_config",
    "can_interface",
    "can_bitrate",
    "can_data_bitrate",
    "use_can_fd",
    "position_kp",
    "position_kd",
    "linear_speed",
    "angular_speed",
    "linear_speed_ratio",
    "angular_speed_ratio",
    "min_linear_speed",
    "min_angular_speed",
    "speed_increment",
    "turn_assist_ratio",
    "manual_control",
    "with_antennas",
    "with_l_arm",
    "with_r_arm",
    "with_neck",
    "with_mobile_base",
)


@cache
def _ensure_lerobot_teleoperators_registered() -> None:
    """Import every teleoperator subpackage so TeleoperatorConfig is populated.

    Mirror of ``hardware_robot._ensure_lerobot_robots_registered`` - walks
    ``lerobot.teleoperators`` with ``pkgutil`` so we automatically pick up
    every teleoperator lerobot ships, including those whose registered type
    name doesn't match its subpackage name (e.g. ``so101_leader`` /
    ``so100_leader`` both in ``so_leader/``). Then invokes lerobot's
    third-party plugin loader so ``lerobot_teleoperator_*`` distributions
    register too.

    Idempotent via ``@cache`` - first call walks the tree, the rest are
    no-ops.
    """
    try:
        import lerobot.teleoperators as _lr_teleop
    except ImportError as exc:
        # Two failure modes, matched log levels (see hardware_robot):
        #   1. lerobot wholly absent -> debug (sim-only / CI hosts).
        #   2. lerobot present but lerobot.teleoperators broken -> warning
        #      (genuine partial-install signal).
        try:
            import lerobot  # noqa: F401  (probe-only)
        except ImportError:
            logger.debug("lerobot not installed: %s", exc)
        else:
            logger.warning(
                "lerobot is installed but lerobot.teleoperators is not importable (partial install?): %s",
                exc,
            )
        return

    for _, sub_name, is_pkg in pkgutil.iter_modules(_lr_teleop.__path__):
        if not is_pkg:
            continue
        full_name = f"{_lr_teleop.__name__}.{sub_name}"
        try:
            importlib.import_module(full_name)
        except (ImportError, OSError) as exc:
            # Device-specific runtime dep missing (hidapi for gamepad,
            # reachy2_sdk, unitree_sdk2py, ...) OR an OS-level probe failure
            # inside a driver's __init__. The teleoperator simply won't appear
            # in the choice registry -- the correct outcome; constructing it
            # later raises a clean "Unsupported teleoperator type". Narrow
            # (ImportError, OSError) per AGENTS.md > Review Learnings (#86).
            logger.debug("[teleoperator] skip %s: %s", full_name, exc)

    try:
        from lerobot.utils.import_utils import register_third_party_plugins
    except ImportError:
        logger.debug("[teleoperator] register_third_party_plugins unavailable")
    else:
        try:
            register_third_party_plugins()
        except (ImportError, AttributeError, OSError) as exc:
            logger.warning("[teleoperator] third-party plugin registration failed: %s", exc)


def _other_lerobot_kind_refusal(requested: str, *, wanted: str) -> str | None:
    """Name ``requested`` as the OTHER kind of lerobot device, or ``None``.

    lerobot keeps two ChoiceRegistries: ``RobotConfig`` for the arm that is
    driven and ``TeleoperatorConfig`` for the one a human holds. A leader and
    the follower it drives carry the same servo bus and the same USB-serial
    shape, so a request for one kind that names a device of the other kind is a
    wrong entry point, not a typo -- and answering it with the names of the kind
    it is not answers the wrong question. For ``Robot("so101_leader")`` that
    list is every follower type, which is exactly the retry that torque-enables
    the arm a human is holding; :func:`strands_robots.robot.Robot` already
    refuses an unregistered ``*_leader`` name without listing the registry for
    that reason, and this is the same refusal for a name lerobot knows.

    Args:
        requested: The device type string the caller asked to build.
        wanted: The kind the caller asked for -- ``"robot"`` or
            ``"teleoperator"``.

    Returns:
        A refusal naming the kind ``requested`` really is and the entry point
        that builds it, or ``None`` when the other registry does not know
        ``requested`` either. ``None`` means the name is genuinely unknown, so
        the caller keeps its own listing of the kind that was asked for.

    Raises:
        ValueError: If ``wanted`` is neither ``"robot"`` nor
            ``"teleoperator"``.
    """
    if wanted == "robot":
        from lerobot.teleoperators.config import TeleoperatorConfig

        _ensure_lerobot_teleoperators_registered()
        if requested not in TeleoperatorConfig.get_known_choices():
            return None
        return (
            f"Unsupported robot type: {requested!r}. That is a lerobot teleoperator "
            f"(leader) device, not a robot: build it with "
            f"``Teleoperator({requested!r}, port=...)`` and attach it to the follower "
            f"it drives. Passing a leader to ``Robot()`` would drive the arm a human "
            f"is holding as a position servo."
        )
    if wanted == "teleoperator":
        from lerobot.robots.config import RobotConfig

        # Imported here, not at module scope: this module is deliberately
        # stdlib-only so it stays import-safe, and the robot registry is only
        # needed on a refusal path.
        from strands_robots.hardware_robot import _ensure_lerobot_robots_registered

        _ensure_lerobot_robots_registered()
        if requested not in RobotConfig.get_known_choices():
            return None
        return (
            f"Unsupported teleoperator type: {requested!r}. That is a lerobot robot "
            f"(follower) device, not a teleoperator: build it with "
            f"``Robot({requested!r}, mode='real', port=...)``."
        )
    raise ValueError(f"wanted must be 'robot' or 'teleoperator', got {wanted!r}")


def _build_teleop_config(teleop_type: str, **kwargs: Any) -> Any:
    """Resolve + construct a lerobot ``TeleoperatorConfig`` for ``teleop_type``.

    Same kwarg-filtering contract as ``hardware_robot._create_minimal_config``:
      * resolve the dataclass via the draccus ChoiceRegistry (source of truth),
      * forward ``id`` when the dataclass declares it,
      * forward allowlisted kwargs only when declared (polymorphism carve-out),
      * forward dataclass-declared-but-not-allowlisted kwargs (future-proofing),
      * REJECT kwargs unknown to both -> typos surface immediately.
    """
    from lerobot.teleoperators.config import TeleoperatorConfig

    _ensure_lerobot_teleoperators_registered()

    try:
        ConfigClass = TeleoperatorConfig.get_choice_class(teleop_type)
    except KeyError:
        if other := _other_lerobot_kind_refusal(teleop_type, wanted="teleoperator"):
            raise ValueError(other) from None
        available = sorted(TeleoperatorConfig.get_known_choices().keys())
        raise ValueError(
            f"Unsupported teleoperator type: {teleop_type!r}. Known lerobot teleoperator types: {available}"
        ) from None

    try:
        valid_fields = {f.name for f in dataclasses.fields(ConfigClass)}
    except TypeError as exc:
        raise TypeError(
            f"lerobot returned a non-dataclass config class {ConfigClass!r} for "
            f"teleop_type={teleop_type!r}; strands_robots cannot filter kwargs "
            f"safely. Please file an issue against lerobot or strands_robots."
        ) from exc

    config_data: dict[str, Any] = {}

    # ``id`` namespaces lerobot's calibration files (left_arm.json, ...).
    if "id" in valid_fields and "id" in kwargs:
        config_data["id"] = kwargs["id"]

    forwardable = _FORWARDABLE_TELEOP_KWARGS
    for key in forwardable:
        if key in kwargs and key in valid_fields:
            config_data[key] = kwargs[key]
        elif key in kwargs:
            logger.debug(
                "[teleoperator] dropping cross-device kwarg %r for teleop_type=%r: "
                "not declared on %s (forwardable-allowlist polymorphism carve-out)",
                key,
                teleop_type,
                ConfigClass.__name__,
            )

    for key in kwargs:
        if key not in config_data and key != "id" and key in valid_fields:
            config_data[key] = kwargs[key]

    always_allowed = {"id"}
    recognised = set(forwardable) | always_allowed | valid_fields
    unknown = set(kwargs) - recognised
    if unknown:
        raise ValueError(
            f"Unknown kwarg(s) for teleop_type={teleop_type!r}: {sorted(unknown)}. "
            f"This teleoperator's dataclass accepts: {sorted(valid_fields)}. "
            f"The cross-device allowlist is: {sorted(set(forwardable) | always_allowed)}. "
            f"(If this is a typo, fix it.)"
        )

    try:
        return ConfigClass(**config_data)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Failed to construct {ConfigClass.__name__} for teleop_type {teleop_type!r}: {e}. Config: {config_data}"
        ) from e


def Teleoperator(  # noqa: N802 - uppercase by design (factory mimicking a constructor)
    name: str,
    *,
    id: str | None = None,  # noqa: A002 - matches lerobot's TeleoperatorConfig.id
    **kwargs: Any,
) -> LeRobotTeleoperator:
    """Create a teleoperator (input device) - returns a raw lerobot Teleoperator.

    Convenience factory, NOT a wrapper. You get the real lerobot
    ``Teleoperator`` instance back with full access to all its methods.

    Unlike :func:`Robot`, a teleoperator has no "sim" mode - it is always a
    real input device (or a mock). It is NOT connected here; call
    ``.connect()`` (or attach it to a robot and call ``robot.teleoperate()``,
    which connects lazily).

    Args:
        name: Teleoperator type ("so101_leader", "so100_leader", "gamepad",
              "keyboard", "phone", "koch_leader", ...). Any type registered
              via lerobot's ``@TeleoperatorConfig.register_subclass``.
        id: Optional instance identifier. Namespaces the calibration file
            (e.g. ``id="blue"`` -> ``blue.json``). Lets two same-type leaders
            keep separate calibration.
        **kwargs: Device-specific config forwarded to the resolved lerobot
                  ``TeleoperatorConfig`` dataclass iff it declares the field.
                  Common: ``port=`` (serial leaders), ``use_gripper=``
                  (gamepad). An unknown kwarg (typo) raises ``ValueError``.

    Returns:
        A connected-on-demand lerobot ``Teleoperator`` instance.

    Raises:
        ValueError: If ``name`` is not a registered teleoperator type, or a
                    kwarg is unknown to both the allowlist and the dataclass.
        TypeError: If lerobot returns a non-dataclass config class.

    Examples::

        leader = Teleoperator("so101_leader", port="/dev/ttyACM1", id="blue")
        pad = Teleoperator("gamepad", use_gripper=True)
        kb = Teleoperator("keyboard")
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"Invalid teleoperator name {name!r}. Pass a registered type string "
            "(e.g. 'so101_leader', 'gamepad', 'keyboard')."
        )

    from lerobot.teleoperators.utils import make_teleoperator_from_config

    build_kwargs = dict(kwargs)
    if id is not None:
        build_kwargs["id"] = id

    config = _build_teleop_config(name.strip(), **build_kwargs)
    return make_teleoperator_from_config(config)


__all__ = ["Teleoperator"]
