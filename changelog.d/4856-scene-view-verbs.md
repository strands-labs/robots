### Added: `Simulation.light()`, a camera re-aimed by name, and `reach` in `describe()`

`sim.light(name, position=, direction=, color=, directional=)` adds a spot or a sun; naming it again recolors or
moves it and `remove(name)` takes it out (Isaac refuses lights by name). `sim.camera(name, ...)` on a camera that
exists now re-aims it instead of raising `already exists`; on MuJoCo it writes the compiled camera, no rebuild.
`describe()["robots"][<id>]["reach"]` (and `robot.describe()["reach"]`) give the farthest link, where it is, and
how it moves per +1 rad of each joint, so a caller can tell which way an arm faces and which sign raises a joint.
