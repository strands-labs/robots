"""Reference: resist the draw-induced torso twist by HOLDING waist_yaw via the arm_sdk overlay.

CLEAN-ROOM reference, written from THIS repo's own overlay and hardware facts:
  - `g1_robot_io.G1RobotIO.publish_targets(targets, gains, weight)` already drives joints in Regular
    mode over `rt/arm_sdk`; the SAME mechanism that moves the 10 arm joints can command `waist_yaw`.
  - `waist_yaw` is motor index 12 on BOTH the 23-DOF and 29-DOF G1 (see the waist-stall signal work).
  - The deploy contract already carries `waist_yaw` gains `[kp=200, kd=5]` (currently unused on the
    arm-overlay path, where the vendor balancer holds the waist).
  - The `waist_yaw` actuator's URDF effort limit is 88 N-m.

WHY: with the firm grasp, dragging an anchored sheet dumps load into a TORSO TWIST -- measured at ~5 deg
(loose) up to 23.9 deg (the hero pinch) -- that the vendor balancer tolerates but does not fully resist.
Holding `waist_yaw` stiff at its draw-start angle should absorb that twist.

STATUS: NOT wired into the live pull. This is the starting point for next session's calibration. It is
MEASURE-ONLY/reference until validated.

SAFETY: this drives a balancing biped's waist. Validate incrementally -- start with a LOW kp, watch for
any fight with the vendor balancer (oscillation), and only then raise stiffness. Keep the feed-forward
torque clamp well under the 88 N-m actuator limit (e.g. <= 50 N-m).
"""

WAIST_YAW_JOINT = "waist_yaw_joint"     # motor index 12 on 23- and 29-DOF G1
WAIST_YAW_EFFORT_NM = 88.0              # URDF effort limit -- keep any torque clamp well under this


class WaistYawHold:
    """Inject a `waist_yaw` position-hold into the arm-overlay (targets, gains).

    Calibration sketch (pseudocode, next session):
        hold = WaistYawHold(kp=120.0, kd=4.0, tau_clamp_nm=40.0)
        hold.capture(io.read_state())               # at DRAW START: latch the angle to hold
        ...
        targets, gains = build_arm_overlay(...)      # the existing 10-arm-joint command
        hold.apply(targets, gains)                    # add the waist_yaw hold
        io.publish_targets(targets, gains, weight=1.0)
    """

    def __init__(self, kp: float = 120.0, kd: float = 4.0, tau_clamp_nm: float = 40.0):
        if tau_clamp_nm > WAIST_YAW_EFFORT_NM:
            raise ValueError(f"tau_clamp_nm {tau_clamp_nm} exceeds the waist_yaw actuator limit "
                             f"{WAIST_YAW_EFFORT_NM} N-m")
        self.kp, self.kd, self.tau_clamp_nm = kp, kd, tau_clamp_nm
        self._hold_angle = None

    def capture(self, state) -> float:
        """Latch the waist_yaw angle to hold. Call ONCE, at draw start."""
        self._hold_angle = float(state.q[WAIST_YAW_JOINT])
        return self._hold_angle

    def apply(self, targets: dict, gains: dict):
        """Add the waist_yaw hold to the overlay command. No-op until capture() is called."""
        if self._hold_angle is None:
            return targets, gains
        targets[WAIST_YAW_JOINT] = self._hold_angle
        gains[WAIST_YAW_JOINT] = [self.kp, self.kd]   # firmware applies kp*(q_des-q)+kd*(-dq), torque-clamped
        return targets, gains
