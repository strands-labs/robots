"""Built-in benchmark specs shipped with ``strands_robots``.

The predicate/reward DSL (:mod:`strands_robots.simulation.predicates`) grew a
complete floating-base locomotion vocabulary - the ``base_velocity_tracking``
exponential-kernel tracking reward, the ``base_height`` / ``base_orientation``
posture regularizers, and the ``base_beyond_x`` / ``base_beyond_y`` (forward- / lateral-progress success),
``base_tipped`` (topple) and ``base_below_z`` (height-collapse) predicates - all
reading the embodiment-agnostic floating-base surface from ``get_observation``.
Those primitives, however, wired into no runnable benchmark:
:func:`~strands_robots.simulation.benchmark.list_benchmarks` was empty until a
caller hand-authored a spec (``register_benchmark_from_file``) or loaded the
LIBERO suite. This module ships canonical velocity-tracking locomotion benchmarks composed
from those primitives - a forward-walk quadruped (``go2_walk_forward``) and two
humanoids (``g1_walk_forward``, ``t1_walk_forward``), plus a lateral strafe
quadruped task (``go2_strafe_left``) - so a floating-base robot has a runnable,
discoverable eval out of the box, and to show the same embodiment-agnostic DSL
transfers unchanged across morphologies, across two bipeds of different scale,
AND across command directions: ``go2_strafe_left`` commands a pure lateral
(``vy``) body twist and scores it with ``base_beyond_y``, and
``go2_turn_left`` commands a pure yaw-rate (``wz``) twist and scores it with
``base_yaw_beyond`` - so the three shipped Go2 tasks exercise all three axes
of the velocity-tracking command (forward ``vx``, lateral ``vy``, yaw ``wz``)
and all three progress predicates (``base_beyond_x`` / ``base_beyond_y`` /
``base_yaw_beyond``), the full omnidirectional locomotion vocabulary.

Registration is opt-in - a caller invokes :func:`register_builtin_benchmarks`
(mirroring how the LIBERO suite registers on demand via
:func:`~strands_robots.simulation.benchmark.register_benchmark`) rather than a
global import side effect - so importing ``strands_robots`` performs no registry
mutation and there is no import-order coupling.
"""

from __future__ import annotations

import copy
from typing import Any

from strands_robots.simulation.benchmark import register_benchmark
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark

# ``go2_walk_forward``: the canonical legged-locomotion velocity-tracking task -
# walk the Unitree Go2 quadruped forward at 1 m/s and stay standing.
#
#   - success: the base walks past x = 2 m (``base_beyond_x``) - a real
#     forward-progress terminal, not "did not fall" (a standing-still policy
#     never satisfies it).
#   - failure: the base topples more than ~53 deg off level (``base_tipped``,
#     tol=0.7) OR its height collapses below 0.18 m (``base_below_z``, the Go2
#     stands at ~0.32 m). Either fall mode terminates the episode early.
#   - dense_reward: the legged_gym-style shaping stack - an exponential-kernel
#     reward for tracking the commanded body-frame twist (vx=1.0, no lateral /
#     yaw command), a squared base-height regularizer toward the 0.32 m nominal
#     stance, and a flat-orientation regularizer. Composable, bounded, and
#     dense so an RL/BC policy gets a gradient every step.
#
# All predicates read ``base_pos`` / ``base_quat`` / ``base_lin_vel`` /
# ``base_ang_vel`` from ``get_observation`` - no per-embodiment base body name -
# so the same spec form transfers to any floating-base robot by swapping
# ``default_robot`` / ``supported_robots`` and the height thresholds.
_GO2_WALK_FORWARD: dict[str, Any] = {
    "name": "go2_walk_forward",
    "instruction": "Walk forward at 1 m/s.",
    "default_robot": "unitree_go2",
    "supported_robots": ["unitree_go2"],
    "max_steps": 1000,
    "success": {"all": [{"predicate": "base_beyond_x", "x": 2.0}]},
    "failure": {
        "any": [
            {"predicate": "base_tipped", "tol": 0.7},
            {"predicate": "base_below_z", "z": 0.18},
        ]
    },
    "dense_reward": [
        {
            "predicate": "base_velocity_tracking",
            "vx": 1.0,
            "vy": 0.0,
            "wz": 0.0,
            "lin_weight": 1.0,
            "ang_weight": 0.5,
            "tracking_sigma": 0.25,
        },
        {"predicate": "base_height", "target": 0.32, "weight": 0.5},
        {"predicate": "base_orientation", "weight": 0.5},
    ],
}

# ``g1_walk_forward``: the humanoid (bipedal) counterpart of ``go2_walk_forward``
# for the Unitree G1 - walk the G1 forward at 1 m/s and stay upright. It shows
# the same floating-base DSL transfers unchanged to a biped; only the thresholds
# and the regularizer stack differ, both grounded in the G1's real model:
#
#   - The G1 stands at base height ~0.79 m (measured; ``add_object``-free spawn),
#     vs the Go2's ~0.32 m - so ``base_height`` targets 0.78 m and the
#     height-collapse failure fires below 0.4 m (a fallen humanoid drops the
#     pelvis well under half its standing height, with wide margin above 0 so a
#     standing spawn never trips it).
#   - ``base_tipped`` (tol=0.7, ~53 deg off level) terminates a topple, as for
#     the quadruped: a walking biped tilts far less, so >53 deg is an
#     unambiguous fall for either morphology.
#   - The dense stack adds the ``base_lin_vel_z`` (anti-bounce) and
#     ``base_ang_vel_xy`` (anti-wobble) regularizers on top of the Go2 stack:
#     bipedal walking is far more sensitive to vertical bounce and roll/pitch
#     wobble of the base than a statically-stable quadruped, so those
#     regularizers (added for exactly this in the DSL) belong in a humanoid task.
_G1_WALK_FORWARD: dict[str, Any] = {
    "name": "g1_walk_forward",
    "instruction": "Walk forward at 1 m/s.",
    "default_robot": "unitree_g1",
    "supported_robots": ["unitree_g1"],
    "max_steps": 1000,
    "success": {"all": [{"predicate": "base_beyond_x", "x": 2.0}]},
    "failure": {
        "any": [
            {"predicate": "base_tipped", "tol": 0.7},
            {"predicate": "base_below_z", "z": 0.4},
        ]
    },
    "dense_reward": [
        {
            "predicate": "base_velocity_tracking",
            "vx": 1.0,
            "vy": 0.0,
            "wz": 0.0,
            "lin_weight": 1.0,
            "ang_weight": 0.5,
            "tracking_sigma": 0.25,
        },
        {"predicate": "base_height", "target": 0.78, "weight": 0.5},
        {"predicate": "base_orientation", "weight": 0.5},
        {"predicate": "base_lin_vel_z", "weight": 0.1},
        {"predicate": "base_ang_vel_xy", "weight": 0.1},
    ],
}

# ``t1_walk_forward``: a second humanoid velocity-tracking task, for the Booster
# T1 (a 23-actuator biped, distinct from the G1). Shipping two bipeds of
# different scale - not just G1 and Go2 - is the point: it proves the shared
# floating-base DSL and the humanoid reward stack are not tuned to a single
# robot's geometry but scale with the embodiment's own measured stance. Only the
# height thresholds differ from ``g1_walk_forward``; the reward-term set,
# velocity command, and topple threshold are identical (both are bipeds walking
# forward at 1 m/s):
#
#   - The T1 stands at base height ~0.665 m (measured from its shipped home
#     keyframe), between the Go2's ~0.32 m and the G1's ~0.79 m. So
#     ``base_height`` targets 0.66 m and the height-collapse failure fires below
#     0.35 m - roughly half the standing height, well under the stance so a
#     standing spawn never trips it, and well above 0. The G1's 0.4 m collapse
#     line is close but tuned to the taller G1; grounding each biped in its own
#     measured stance is exactly why a per-robot spec (not one shared humanoid
#     spec) is the right shape here.
#   - ``base_tipped`` (tol=0.7, ~53 deg off level) and the full anti-bounce /
#     anti-wobble dense stack carry over unchanged from the G1: both are bipeds,
#     so the vertical-bounce and roll/pitch-wobble sensitivities that motivated
#     those regularizers apply identically.
_T1_WALK_FORWARD: dict[str, Any] = {
    "name": "t1_walk_forward",
    "instruction": "Walk forward at 1 m/s.",
    "default_robot": "booster_t1",
    "supported_robots": ["booster_t1"],
    "max_steps": 1000,
    "success": {"all": [{"predicate": "base_beyond_x", "x": 2.0}]},
    "failure": {
        "any": [
            {"predicate": "base_tipped", "tol": 0.7},
            {"predicate": "base_below_z", "z": 0.35},
        ]
    },
    "dense_reward": [
        {
            "predicate": "base_velocity_tracking",
            "vx": 1.0,
            "vy": 0.0,
            "wz": 0.0,
            "lin_weight": 1.0,
            "ang_weight": 0.5,
            "tracking_sigma": 0.25,
        },
        {"predicate": "base_height", "target": 0.66, "weight": 0.5},
        {"predicate": "base_orientation", "weight": 0.5},
        {"predicate": "base_lin_vel_z", "weight": 0.1},
        {"predicate": "base_ang_vel_xy", "weight": 0.1},
    ],
}


# ``go2_strafe_left``: the LATERAL velocity-tracking counterpart of
# ``go2_walk_forward`` - strafe the Unitree Go2 sideways to its left at 0.5 m/s
# and stay standing. It is the first shipped benchmark to command a non-zero
# lateral (``vy``) twist and to score a non-forward position goal, proving the
# velocity-tracking vocabulary is omnidirectional, not forward-only:
#
#   - success: the base strafes past y = 1 m (``base_beyond_y``) - world +y is
#     the robot's left for the identity spawn orientation, so this reads
#     "strafed ~1 m left". A standing-still or purely-forward policy never
#     satisfies it (it scores lateral progress, not "did not fall").
#   - failure: the same fall modes as the forward task - topple past ~53 deg
#     (``base_tipped``, tol=0.7) OR height collapse below 0.18 m
#     (``base_below_z``, the Go2 stands at ~0.32 m).
#   - dense_reward: the same legged_gym-style stack, but the tracked twist is a
#     PURE lateral command (vx=0.0, vy=0.5, wz=0.0) - the ``vy`` term of
#     ``base_velocity_tracking`` that the pure-``vx`` walk-forward tasks leave at
#     zero - plus the 0.32 m base-height and flat-orientation regularizers.
_GO2_STRAFE_LEFT: dict[str, Any] = {
    "name": "go2_strafe_left",
    "instruction": "Strafe left at 0.5 m/s.",
    "default_robot": "unitree_go2",
    "supported_robots": ["unitree_go2"],
    "max_steps": 1000,
    "success": {"all": [{"predicate": "base_beyond_y", "y": 1.0}]},
    "failure": {
        "any": [
            {"predicate": "base_tipped", "tol": 0.7},
            {"predicate": "base_below_z", "z": 0.18},
        ]
    },
    "dense_reward": [
        {
            "predicate": "base_velocity_tracking",
            "vx": 0.0,
            "vy": 0.5,
            "wz": 0.0,
            "lin_weight": 1.0,
            "ang_weight": 0.5,
            "tracking_sigma": 0.25,
        },
        {"predicate": "base_height", "target": 0.32, "weight": 0.5},
        {"predicate": "base_orientation", "weight": 0.5},
    ],
}


# ``go2_turn_left``: the YAW (turn-in-place) velocity-tracking counterpart of
# ``go2_walk_forward`` / ``go2_strafe_left`` - turn the Unitree Go2 in place to
# its left at 0.5 rad/s and stay standing. It is the first shipped benchmark to
# command a non-zero yaw-rate (``wz``) twist and to score a HEADING goal,
# completing the omnidirectional velocity-tracking vocabulary (the walk-forward
# tasks command ``vx``, ``go2_strafe_left`` commands ``vy``, and this commands
# ``wz`` - all three axes of ``base_velocity_tracking``):
#
#   - success: the base turns past a 1.0 rad (~57 deg) heading to the left
#     (``base_yaw_beyond``) - world yaw is measured from the identity spawn, so
#     positive is a left (counter-clockwise) turn. A standing-still or
#     translating-but-not-turning policy never satisfies it (it scores heading
#     progress, not "did not fall" and not linear displacement).
#   - failure: the same fall modes as the other Go2 tasks - topple past ~53 deg
#     (``base_tipped``, tol=0.7) OR height collapse below 0.18 m
#     (``base_below_z``, the Go2 stands at ~0.32 m). Pairing the fall predicate
#     with the yaw success is deliberate: a toppled base has an ill-defined yaw,
#     so ``base_tipped`` vetoes a "turned then fell" rollout.
#   - dense_reward: the same legged_gym-style stack, but the tracked twist is a
#     PURE yaw command (vx=0.0, vy=0.0, wz=0.5) - the ``wz`` term of
#     ``base_velocity_tracking`` (with vx=vy=0 also rewarding zero drift, i.e.
#     turning IN PLACE) that the walk-forward and strafe tasks leave at zero -
#     plus the 0.32 m base-height and flat-orientation regularizers.
_GO2_TURN_LEFT: dict[str, Any] = {
    "name": "go2_turn_left",
    "instruction": "Turn left in place at 0.5 rad/s.",
    "default_robot": "unitree_go2",
    "supported_robots": ["unitree_go2"],
    "max_steps": 1000,
    "success": {"all": [{"predicate": "base_yaw_beyond", "yaw": 1.0}]},
    "failure": {
        "any": [
            {"predicate": "base_tipped", "tol": 0.7},
            {"predicate": "base_below_z", "z": 0.18},
        ]
    },
    "dense_reward": [
        {
            "predicate": "base_velocity_tracking",
            "vx": 0.0,
            "vy": 0.0,
            "wz": 0.5,
            "lin_weight": 1.0,
            "ang_weight": 0.5,
            "tracking_sigma": 0.25,
        },
        {"predicate": "base_height", "target": 0.32, "weight": 0.5},
        {"predicate": "base_orientation", "weight": 0.5},
    ],
}


# Registry of shipped built-in specs, keyed by their canonical registry name.
# Extend this dict to ship more; every entry reuses the same embodiment-agnostic
# floating-base DSL, differing only in the robot, thresholds, and reward stack.
_BUILTIN_SPECS: dict[str, dict[str, Any]] = {
    "go2_walk_forward": _GO2_WALK_FORWARD,
    "g1_walk_forward": _G1_WALK_FORWARD,
    "t1_walk_forward": _T1_WALK_FORWARD,
    "go2_strafe_left": _GO2_STRAFE_LEFT,
    "go2_turn_left": _GO2_TURN_LEFT,
}


def builtin_benchmark_specs() -> dict[str, dict[str, Any]]:
    """Return deep copies of the shipped benchmark spec dicts, keyed by name.

    Deep-copied so a caller inspecting or forking a spec cannot mutate the
    module-level canonical definitions.
    """
    return {name: copy.deepcopy(spec) for name, spec in _BUILTIN_SPECS.items()}


def register_builtin_benchmarks() -> list[str]:
    """Compile and register every shipped built-in benchmark into the registry.

    After this call the specs are discoverable via
    :func:`~strands_robots.simulation.benchmark.list_benchmarks` and runnable via
    :meth:`~strands_robots.simulation.base.SimEngine.evaluate_benchmark`.
    Idempotent by overwrite: re-registering a name replaces the prior instance
    (matching :func:`~strands_robots.simulation.benchmark.register_benchmark`).

    Returns:
        The sorted list of registered benchmark names, so a caller can
        immediately look them up / evaluate them.
    """
    names: list[str] = []
    for name, spec in _BUILTIN_SPECS.items():
        # from_dict overwrites spec["name"] with the registry key contract, so a
        # deep copy keeps the canonical dict pristine for builtin_benchmark_specs.
        bench = DeclarativeBenchmark.from_dict(copy.deepcopy(spec))
        register_benchmark(name, bench)
        names.append(name)
    return sorted(names)


__all__ = ["builtin_benchmark_specs", "register_builtin_benchmarks"]
