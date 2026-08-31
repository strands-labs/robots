### Fixed: the reported base pose is the robot's own free joint, not a prop's

A floating-base robot surfaces its 6-DoF base through four additive keys on
`get_observation` (`base_pos`, `base_quat`, `base_lin_vel`, `base_ang_vel`) and
through `get_robot_state`'s structured `base` entry. Both read one free joint,
and both picked it the same wrong way: the loop over `robot.joint_names` recorded
every free joint it met so it could skip that joint's degenerate scalar, and the
last write won. A scene that ships a free-jointed task object under the robot's
own namespace - a kick ball, a Menagerie grasping cube - puts a second named free
joint in that list after the base, so the prop won and the robot reported the
prop's pose as its own.

Nothing refused it. Every value stays finite and plausibly shaped, so the rollout
reports success and only the numbers belong to another body. On the microduck's
shipped `scene_ball.xml`, a duck standing with its trunk at 0.120 m reported
`base_pos = [0.3, 0.0, 0.035]` - the ball - on both surfaces; `base_pos` is
documented for "fall/height tracking", so it read as permanently fallen, and
`base_ang_vel`, documented as matching "the IMU-gyro frame WBC/locomotion
controllers consume", carried the prop's spin.

`_robot_base_free_joint` already owned that question and already answered it
correctly - its docstring states the guarantee in as many words, that a sibling
task object "including one shipped inside the robot's own MJCF under its
namespace, which is how every Menagerie grasping scene is authored ... is on no
seed's ancestor chain". The guarantee held; it was skipped, because it was
consulted only when the loop had found nothing at all. Both sites now let it
choose and keep the loop's find only when it declines to name one, so an unnamed
floating base - the case the resolver was written for, such as LeKiwi's - still
resolves, and a fixed-base arm still reports no base keys.
