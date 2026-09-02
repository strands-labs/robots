### Added: `URDriver` - Universal Robots UR5e and UR10e over RTDE

lerobot registers no robot type for a UR arm, so `Robot("ur5e", mode="real")`
could only answer `Unsupported robot type: 'ur5e'`. `URDriver` closes that
through the controller's own Real-Time Data Exchange interface, reached with
`Robot("ur5e", mode="real", driver="strands", port="192.168.1.10")` and the
`ur_rtde` SDK: `state()` reads joint positions and velocities, the TCP pose and
the TCP wrench in one round trip; `send_action()` maps a joint-name-keyed action
onto `servoJ`; `run_policy()` streams a rollout at a fixed cadence on its own
thread and reports why it stopped. One driver serves both arms, and the sim
assets declare their joints in the same order RTDE uses, so an action dict
recorded in simulation streams to the controller with no remap.

Two gates stand in front of every write, because a UR controller does not
reject a bad command the way a servo bus does - it accepts the register write
and performs nothing. The **mode** gate refuses a robot mode other than
`RUNNING`, or a safety mode outside `NORMAL`/`REDUCED`, in the controller's own
vocabulary; it is asked before the control interface is opened at all, and
re-asked per write, so a protective stop landing mid-rollout ends the rollout
with that reason. The **speed** gate refuses a joint asked to move further than
its datasheet ceiling allows in one control period, naming the joint and both
figures; the ceilings are per model (every UR5e joint reaches 180 deg/s where
the UR10e's three proximal joints are held to 120 deg/s), which is why one
driver serving two arms still carries a table, and an unrecognised model is
held to the slower arm's limits rather than the faster. `servoJ`'s boolean
return decides the verdict, so a setpoint the controller declines is reported as
an error rather than as a command that was sent.

The speed gate measures each step from the **last commanded setpoint**, not from
the measured pose. `servoJ` setpoints form a trajectory the controller
interpolates toward, so the arm is always behind the setpoint it was last given;
measuring from the measured pose charges the commanded step for that lag, and a
policy streaming increments the arm can easily follow starts being refused the
moment it falls a period behind. Measured in simulation at 50 Hz, that spelling
refused 288 of 420 setpoints from a trajectory whose per-step increments were a
quarter of the ceiling. The end of a stream drops the anchor and the next
setpoint is sized from the measured pose again - whichever party ended it: a
halt verb, a rollout leaving its loop for any reason, or a controller-initiated
protective stop, which is the one after which the arm has most likely been
jogged, because clearing that stop from the pendant is when an operator moves
it. A setpoint this driver's own value gates refuse does not end the stream and
leaves the anchor standing, so a single refusal costs no tracking margin.
