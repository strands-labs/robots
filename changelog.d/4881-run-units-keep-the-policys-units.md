### Fixed: run(units=) keeps the units a policy declares, a model cannot set them, and a replay's norm= names the gripper

`run(units="lerobot")` read a degrees replay, a checkpoint's `norm=` and a LeRobot leader in the registry's range
modes (a replay 85.5° off on the fake bus's wrist_roll), and `units="rad"` read the recorded degrees as radians,
driving every joint to its stop. A policy that declares its units now keeps them: a scheme that agrees uses the
policy's own map, in its norm modes, and any other is refused before anything is written, naming both; a scheme still
says what a policy that declares nothing speaks. `robot.tool` and `sim.tool` no longer offer or accept `units` (a
model's `"rad"` drove a registered LeRobot-units policy 0.98 rad further under the same approval), and the `RunPlan`
an operator approves names the units. The real-body refusals spell out the whole `norm=` they ask for, and a replay
refuses a `"*"` that would give a joint the registry names another mode: `norm={"*": "degrees"}` wrote an SO
gripper's 50 % as 50°, though LeRobot records it in `range_0_100`.
