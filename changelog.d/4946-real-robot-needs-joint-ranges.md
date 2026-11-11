### Fixed: a real robot refuses a model with a driven joint that has no range

The safety chain reads a joint without a range as unconstrained, so on hardware such a joint (`limited="false"`, or
no range on the joint or its default class) took any goal on every rail. Now `Robot(..., port=/driver=)` raises
`ValueError` naming those joints before the driver opens the bus, and `JointLimitValidator` refuses a target for a
real robot's joint without a range instead of skipping it. A joint that turns without end can drive a real robot
when its registry entry declares a `hardware.safe` bound for it; the chain then checks that bound. Every registry
robot with a driver has a range on each joint, with and without the mujoco extra. Simulated robots are unchanged.
