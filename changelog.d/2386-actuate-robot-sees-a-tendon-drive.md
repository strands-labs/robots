### Fixed

`actuate_robot` no longer adds position servos on top of joints an existing
tendon actuator already drives. The double-actuate guard resolved each
actuator's transmission through `actuator_joint_id`, which reports no joint for
a tendon, so a gripper whose fingers are coupled by one fixed tendon read as
undriven: the call reported `success`, added a servo per finger, and the two
drives then fought - `robotiq_2f85` lost 98% of its finger travel on the same
command it had answered before. The guard now resolves a tendon to the joints
it wraps, matching the rule the action-application path already used, and the
refusal names the joints it found and the actuators driving them.
