### Fixed: Isaac's `move_to` solves IK on the description the robot was built from

The IK solve runs on a compiled MuJoCo model, and it used to resolve one from
exactly one place: the robot's `data_config`, through the registry. The file the
robot was actually built from - the URDF `add_robot` imported, or the MJCF an
imported USD was converted from, both of which MuJoCo compiles - was discarded at
add time. Two costs, one loud and one silent:

* **loud** - a robot added via a bare `urdf_path` was refused `move_to` outright
  (`"robot 'arm' has no data_config"`), and a robot added by registry name
  (`add_robot("so100")`) was refused too, because the name path stored
  `data_config=None` unless the caller spelled it a second time;
* **silent** - a `data_config` naming a registry model that differs from the
  loaded asset while sharing every joint name solved on the wrong kinematics, and
  the convergence check runs by FK on the IK model itself - so the solve
  **confirmed** a pose the stage end-effector does not hold. "reached", with the
  arm somewhere else.

`_RobotState` now records `description_path` - the URDF, or the pre-conversion
MJCF; `None` for a plain USD, which MuJoCo cannot compile - and `_load_ik_mjcf`
prefers it over the registry lookup, so the IK model is the simulating file by
construction. The registry `data_config` path survives as the fallback for
description-less robots (every one of the seven resolution refusals pinned by
`test_move_to_ik_model_resolution.py` lives on it, unchanged), and when both
sources exist and resolve to different files the description wins with a WARNING
naming both. A description that fails to compile is a structured error, not a
silent registry fallback - falling back would knowingly reintroduce the
divergent-model solve this removes.

Verified on `nvcr.io/nvidia/isaac-sim:6.0.1` (A10G) with a 3-DOF arm added via
bare `urdf_path`, never registered anywhere: `move_to` reports
`reached [0.18, 0.0, 0.42] within 0.03 m (error 0.0296 m)` and the end-effector's
world pose read **off the USD stage** - independent of anything the IK stack
computed - measures the same 0.0296 m. The IK model's FK and the stage agree to
four decimals because they are the same kinematics.

This keeps mink/MuJoCo as the portable IK layer (testable without a GPU,
identical across backends) rather than switching to an Isaac-native planner;
a cuRobo/Lula-backed planner remains a possible opt-in upgrade, as
`examples/so101_curobo` already demonstrates.
