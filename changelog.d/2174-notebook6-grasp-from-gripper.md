### Fixed: notebook 6 claims `grasp` from a declared gripper, not from a category

`examples/notebooks/06_fleet_orchestration.ipynb` derived every capability from
the registry's `category`, including `grasp`. A category cannot answer that one:
23 registry robots have category `arm` and 3 of them declare a `gripper` block,
so the notebook advertised `grasp` for 59 of 72 robots against 3 that declare the
mechanism. A `ur5e` or an `fr3` was therefore allocatable pick-and-place work on
the strength of its category alone, which is the one allocation error an
allocator cannot detect afterwards - the robot accepts the task, the plan reads
complete, and the failure happens in the world.

`grasp` now comes from the `gripper` block, the field that names the gripper's
actuators and which end of their range is closed. That is the same order of
preference the library already applies: `MuJoCoMotionPrimitives.
_registry_gripper_metadata` prefers this block over its name heuristic rather
than the reverse, and for the same reason (#1658).

An unknown robot name now raises the notebook's own `ValueError`. `get_robot`
returns `None` for a name it does not know, so `get_robot(name).get("category")`
died with `AttributeError: 'NoneType' object has no attribute 'get'` before
reaching the message written directly below it - the message naming the canonical
form and telling the reader what to add was unreachable for exactly the input it
was written for.

`aerial` and `expressive` were absent from the map, so a drone and `reachy_mini`
were refused as unmapped gaps. `aerial` now maps to `navigate` and `inspect`;
`expressive` maps to the empty set, explicitly, because advertising nothing is a
real answer and every sub-task offered to such a robot should be left
`UNASSIGNED`.

`tests/test_notebook_capability_model.py` pins all four properties against the
live registry, executing the notebook's own cell rather than a copy so the pin
cannot pass after the notebook regresses. It also asserts the gripper split is
non-empty on both sides, without which a category-derived `grasp` would satisfy
the check vacuously.
