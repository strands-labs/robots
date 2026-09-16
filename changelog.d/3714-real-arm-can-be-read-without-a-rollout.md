### Added: a real arm can be read by an agent without running a policy on it

`Robot("so101", mode="real")` handed to an `Agent` offered four actions -
`execute`, `start`, `status`, `stop` - so the only way an agent could learn
where a physical arm was, or whether torque was on, was to run a policy on it.
Asked to "read the joint positions and torque state, do NOT move any joint", an
agent's one call was `execute` with `policy_provider="mock"` and a ten-second
duration: a rollout requested in order to look. The operator gate caught it
(the gate doing its job; the tool failing at its own), and after the operator
declined, the agent reported that the robot was probably not connected.

The real-hardware tool gains four observe actions, none gated because none
writes a servo register: `get_state` (alias `get_robot_state`) reads every
joint's position in degrees and raw encoder ticks, its torque state and supply
voltage; `list_cameras` names the configured cameras and whether each is open;
`render` saves one PNG from a camera into the render sandbox, refusing outside
paths the way the simulation's `render` does, and returns the frame as an
`image` block so the model sees it (handed only the path, an agent described
"the arm resting on the desk" for a shot of the ceiling). `get_state` opens the motor bus
alone on first use - not the robot's `connect()`, whose `configure()` writes
operating mode and gains to every servo - and reads under the device's bus lock
beside a rollout or the mesh probes. An arm with no calibration file is still
readable: degrees are then the encoder estimate (`2048 ticks = 0°`) and the
text says so, naming `lerobot-calibrate`. A reading the arm did not give is
reported as unread, never as its reassuring opposite: a calibration flag whose
own read fails - on a lerobot bus that read sweeps every servo for homing
offsets - is `null` with the reason rather than `false`, so the degrees stay the
arm's own and the operator is not sent to `lerobot-calibrate` for a serial
fault; a bus with no notion of calibration is calibrated, by lerobot's contract
for the property, which is the reading the connect gate takes; and a
`Torque_Enable` register no motor answered is `null`, not `off`, because "torque
OFF on all joints (the arm can be moved by hand)" is a claim about a live arm. The tool description leads with the
observe actions, names the port, and says plainly that `set_joint_positions`
and `move_to` do not exist on a real robot. Measured on an SO-101:
`get_state` answered in 43 ms with six joints, torque off, 5.6 V.
