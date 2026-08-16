### Added

- `ProtoMotionsPolicy` (`protomotions`, shorthands `gtp` / `gtp_g1` /
  `protomotions_g1`): a ProtoMotions Generalist Tracking Policy provider that
  tracks a reference motion clip on the Unitree G1 and emits balanced PD joint
  targets. Runs an ONNX export in-process, so no GPU is required. It is the
  tracking half of the two-stage pipeline whose kinematic half is
  `KimodoPolicy` or `MotionBricksPolicy`: `qpos_to_motion_data` bridges a
  generated `qpos` clip into the tracker's reference cache via MuJoCo forward
  kinematics. The policy declares `required_bodies = ("torso_link",)` so the
  runtime supplies the anchor link's world orientation, which is not derivable
  from the observation's floating-base signals - `base_quat` is the pelvis, and
  on a G1 sweeping its waist the two frames diverge by up to 42 degrees.
  `ProtoMotionsSession` injects a stub tracker for tests, so the observation to
  action mapping is exercised without onnxruntime, weights or CUDA. New
  `[protomotions]` extra; see `docs/policies/protomotions.md`.
