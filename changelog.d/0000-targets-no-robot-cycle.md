### Internal: the action-to-target coercion no longer imports back into the Robot class

`robot/targets.py`, the one coercion every write path runs an `Action` through, typed its single public
function against `Robot` under `TYPE_CHECKING`, closing an import cycle with `robot/robot.py`. It now types
against a small `Targetable` protocol declared in the module itself (the five attributes the coercion reads),
so it imports nothing from `robot.py`. Behaviour is unchanged; a layout test keeps the module a leaf.
