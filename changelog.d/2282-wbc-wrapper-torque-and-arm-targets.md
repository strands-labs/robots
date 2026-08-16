### Fixed

- **WBC through a policy wrapper**: `sim.run_policy` now installs the MuJoCo torque
  shim for a `WBCPolicy` reached through a wrapper - a `CompositePolicy` driving the
  legs from WBC and the arms from a manipulation policy, or a `PersistentPolicy`
  holding it warm. The hook resolved the shim with `isinstance(policy, WBCPolicy)`,
  so a wrapped policy silently drove the stock uniform-gain position servos that
  override SONIC's tuned per-joint PD. `WBCTorqueController` also runs its light arm
  PD toward whatever target the action dict names for each arm joint, holding the
  nominal pose only for joints left unnamed; it previously pinned all 14 arm joints
  to nominal, so the upper body of a `CompositePolicy` was dropped without a warning
  while the robot walked correctly.
