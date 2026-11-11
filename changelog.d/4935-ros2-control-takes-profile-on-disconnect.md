### Fixed: a ros2_control arm can be commanded over the mesh

`ros.connect("ros2_control", ...)` reads the model from `/robot_description`, which has no registry `hardware` block,
so the robot had no `on_disconnect`. A real robot without a proven-safe disconnect behaviour cannot carry a
watchdog, and the mesh serves such a robot observe-only, so every move was refused. `ros.connect` now applies the
profile's declared `on_disconnect` (`"hold"` for `ros2_control`) when the model declares none; a registry entry's own
`hardware.on_disconnect` still wins.
