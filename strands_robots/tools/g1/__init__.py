"""G1 hardware layer - DDS engine + agent ``@tool`` verbs for the Unitree G1.

The Unitree G1 speaks raw Unitree IDL over CycloneDDS: ``rt/lowstate`` for IMU
and joints, ``rt/lf/bmsstate`` for battery, ``rt/utlidar/cloud_livox_mid360``
and ``rt/utlidar/lidar_state`` for the Livox Mid-360, ``rt/lowcmd`` and
``rt/armsdk`` for motion. Neither ROS 2 nor the lerobot serial bus can reach
those topics, so the driver owns its own subscriber layer.

Organizing principle (post-consolidation, refs #2928):
    Anything that is a 1:1 SDK call is reachable via ``use_unitree`` - the
    universal dispatcher (use_aws pattern) whose meta operations
    (``list_operations`` / ``describe_operation``) replace the per-method
    lookup modules this package used to carry. This package only keeps
    verbs that do REAL WORK beyond the SDK:

      * Driver-cache reads      (g1_state, g1_battery, g1_imu, g1_mainboard,
                                 g1_pressure, g1_lidar_state, g1_lidar_summary)
      * Driver-gated writes     (g1_send_action, g1_run_policy, g1_start_task,
                                 g1_stop_task, g1_task_status, and the
                                 execution verbs consolidated in g1_actions:
                                 g1_set_fsm, g1_move_velocity, g1_stop_move,
                                 g1_set_stand_height, g1_set_swing_height,
                                 g1_balance_stand, the safe posture
                                 transitions, the arm and loco gestures)
      * Gate introspection      (g1_motion_gates)
      * Reference data          (g1_joints, g1_error_codes, g1_arm_actions)
      * The escape hatch        (use_unitree)

``unitree_sdk2py`` is lazy-imported everywhere: importing this package or
any verb module pulls no SDK submodule, so a machine without the SDK -
every headless CI runner - can build the driver, list it in the registry
and run every test with a mocked bus. The SDK only loads when a verb
executes against a real robot.

Verb modules are imported lazily via ``__getattr__`` for the same reason
the top-level ``strands_robots.tools`` does it: a caller who only wants
:func:`ensure_dds` should not pay for the whole verb tree.
"""

import importlib as _importlib

from strands_robots.tools.g1._g1_common import (
    ERR_CODES,
    HANDSHAKE_FSMS,
    WALK_FSMS,
    decode_code,
    ensure_dds,
    reset_dds_state,
)

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Driver-cache reads (P0)
    "g1_get_state": (".g1_state", "g1_get_state"),
    "g1_battery": (".g1_battery", "g1_battery"),
    "g1_imu": (".g1_imu", "g1_imu"),
    "g1_mainboard": (".g1_mainboard", "g1_mainboard"),
    "g1_pressure": (".g1_pressure", "g1_pressure"),
    "g1_lidar_state": (".g1_lidar_state", "g1_lidar_state"),
    "g1_lidar_summary": (".g1_lidar_summary", "g1_lidar_summary"),
    # Driver-gated writes (P1/P2)
    "g1_send_action": (".g1_send_action", "g1_send_action"),
    "g1_run_policy": (".g1_run_policy", "g1_run_policy"),
    "g1_start_task": (".g1_start_task", "g1_start_task"),
    "g1_stop_task": (".g1_stop_task", "g1_stop_task"),
    "g1_get_task_status": (".g1_task_status", "g1_get_task_status"),
    # Driver-gated execution verbs, consolidated in one table-driven module
    "g1_arm_action": (".g1_actions", "g1_arm_action"),
    "g1_balance_stand": (".g1_actions", "g1_balance_stand"),
    "g1_move_velocity": (".g1_actions", "g1_move_velocity"),
    "g1_release_arm": (".g1_actions", "g1_release_arm"),
    "g1_safe_lie_to_stand": (".g1_actions", "g1_safe_lie_to_stand"),
    "g1_safe_squat_to_stand": (".g1_actions", "g1_safe_squat_to_stand"),
    "g1_safe_stand_to_squat": (".g1_actions", "g1_safe_stand_to_squat"),
    "g1_set_fsm": (".g1_actions", "g1_set_fsm"),
    "g1_set_stand_height": (".g1_actions", "g1_set_stand_height"),
    "g1_set_swing_height": (".g1_actions", "g1_set_swing_height"),
    "g1_shake_hand_loco": (".g1_actions", "g1_shake_hand_loco"),
    "g1_stop_move": (".g1_actions", "g1_stop_move"),
    "g1_wave_hand_loco": (".g1_actions", "g1_wave_hand_loco"),
    # Gate introspection
    "g1_list_motion_gates": (".g1_motion_gates", "g1_list_motion_gates"),
    "g1_fsm_admits": (".g1_motion_gates", "g1_fsm_admits"),
    # Reference data (pure Python, no robot needed)
    "g1_joint_reference": (".g1_joints", "g1_joint_reference"),
    "g1_joint_name": (".g1_joints", "g1_joint_name"),
    "g1_joint_index": (".g1_joints", "g1_joint_index"),
    "g1_list_error_codes": (".g1_error_codes", "g1_list_error_codes"),
    "g1_decode_error_code": (".g1_error_codes", "g1_decode_error_code"),
    "g1_list_arm_actions": (".g1_arm_actions", "g1_list_arm_actions"),
    "g1_arm_action_admits": (".g1_arm_actions", "g1_arm_action_admits"),
    # Universal SDK dispatcher (use_aws pattern)
    "use_unitree": (".use_unitree", "use_unitree"),
}

__all__ = [
    "ERR_CODES",
    "HANDSHAKE_FSMS",
    "WALK_FSMS",
    "decode_code",
    "ensure_dds",
    "reset_dds_state",
    *_LAZY_IMPORTS.keys(),
]


def __getattr__(name: str):  # noqa: N807
    if name in _LAZY_IMPORTS:
        rel_module, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(rel_module, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
