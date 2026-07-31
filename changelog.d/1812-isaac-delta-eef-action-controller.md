### Added: Isaac-side GR00T action actuation - delta-EEF differential-IK controller

With the observation pipeline fixed (#1811), `examples/libero/run_isaac.py
--policy groot` ran end-to-end but scored `success_rate 0.00` against the
MuJoCo baseline's 1.00: `LiberoAdapter._install_action_controller` builds
robosuite's `OSC_POSE` controller against the compiled MuJoCo model, so on
Isaac no controller was ever installed, GR00T's task-space delta-EEF keys
(`{x, y, z, roll, pitch, yaw, gripper}`) resolved to no joint, and every
action landed in `send_action`'s `unresolved_keys` - a 946-second video of a
perfectly still Franka (#1812).

`strands_robots.simulation.isaac.delta_eef.IsaacDeltaEEFController` now
converts each task-space delta into joint position targets via a
damped-least-squares differential-IK solve on the end-effector's world-frame
spatial Jacobian, preserving the trained action semantics (#168): inputs
clipped to `[-1, 1]` then scaled by robosuite `OSC_POSE`'s `output_max`
(0.05 m / 0.5 rad per 20 Hz control step), RLDS gripper convention
(`0 = close`, `1 = open`, `0.5 = hold`). The engine side gained the seams the
controller needs: `IsaacSimulation.install_action_controller` /
`uninstall_action_controller` (dict actions route through
`compute_joint_targets` before name resolution; conversion failures are error
envelopes, never a silent fall-through), `get_jacobian` (MuJoCo-envelope
signature parity; fixed-base articulation links only, loud rejection of
`site_name` / `geom_name` and unverified layouts), and a `physics_timestep`
override so `PolicyRunner` steps a full control period per action
(`physics_dt=1/120` at 20 Hz -> 6 substeps).

`LiberoAdapter._install_action_controller` tries the Isaac path when the
MuJoCo OSC path reports its dependencies unavailable: a duck-typed probe of
the engine's public seams installs the delta-EEF controller bound to the
LIBERO Franka layout (`panda_joint1..7`, `panda_finger_joint1/2`,
`panda_hand`), with a Jacobian probe at install time so broken kinematics
surface at episode start under the existing strict/non-strict policy. Setup
breakage on a genuine Isaac engine stays loud; engines with neither path keep
the pre-existing warn-and-degrade behaviour, and the warning now names both
unavailable paths. `run_isaac.py` passes `control_frequency=20.0` (the rate
GR00T-N1.7-LIBERO was trained at) instead of inheriting the 50 Hz default.

Validation status (4x L4, Isaac Sim 6.0): the controller is GPU-verified at
the engine level - scripted saturated deltas displace `panda_hand` in the
commanded world direction and the gripper channel drives both fingers
(`tests_integ/simulation/test_isaac_delta_eef_gpu.py`). The full
`--policy groot` eval installs the controller (the per-episode no-op warning
is gone) but still measures `success_rate 0.00`, because a distinct,
pre-existing `IsaacSimulation.load_scene` defect leaves the Isaac timeline
stopped for the rest of the episode - `world.step()` then renders without
integrating physics, so every action (this controller's or anyone else's)
lands on frozen physics - and restarting the timeline exposes the scene's
placeholder-posed objects interpenetrating the robot base and exploding the
articulation. That defect predates this change and is tracked as its own
issue rather than folded in here.
