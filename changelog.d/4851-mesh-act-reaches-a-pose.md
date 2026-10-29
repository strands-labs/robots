### Fixed: a remote `act` takes `seconds=` and reaches the pose, as the manifest says

`mesh.invoke(id, "act", action=..., seconds=1.0)` now ramps the device to the pose (`Robot.act`'s reach) before it
answers; before, the device refused `seconds` although its manifest advertised `act(action, steps, seconds)`, so a
remote caller could only move one control step at a time. `use_mesh` offers `seconds` on `act`, so a model asked to
move a joint to 0.4 rad gets there instead of 0.015 rad along. `seconds` is refused outside (0, 10]; the caller waits
that much longer for the reply, and the device's watchdog does not count a reach in progress as a quiet invoker.
