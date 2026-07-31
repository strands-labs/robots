### Fixed: `send_action` refuses a boolean actuator command instead of writing it as 1.0/0.0

`bool` is an `int` subclass and `numpy.bool_` coerces identically, so
`send_action`'s scalar-coercion check admitted both and they reached the actuator
as a silent `1.0` / `0.0` under `status="success"`. That is not one command: 1.0
is a 1-radian target on a joint-position drive, a full-travel command on a
normalized or tendon drive, and an out-of-range value that is silently clamped
where `ctrlrange` excludes 1 - so the same `True` commands a different pose on
every actuator. Measured on the Panda's `[0, 255]` tendon gripper,
`send_action({"gripper": True})` wrote `ctrl` 255 and swung the fingers from
closed to fully open (0.0800 m), byte-identical to an explicit full-travel
`1.0` - the opposite of what a binary "grasp" flag asks for, and a boolean is the
conventional binary-gripper action rather than a typo.

The teleop wire validator already refused a boolean so it could not "masquerade
as a 1.0/0.0 command", and `InputReceiver` applies validated frames through
`send_action` - so the remote surface was held to a stricter domain than the
local call it delegates to. Both accepted action shapes (a mapping value and a
vector entry) now refuse a python or numpy boolean, naming the offending key or
vector index and the actuator's own units as the remedy. The guard lives in the
shared `SimEngine` coercion, so every backend inherits it. Numeric spellings are
unchanged, including the documented numeric-string form and the single-element
`[0.05]` unwrap.

A 0-d numpy array (`np.array(0.5)`, `np.mean(...)`) additionally raised a bare
`TypeError: len() of unsized object` past `send_action`'s structured-error
contract - it declares `__len__` but raises on `len()`, so the single-element
unwrap could not probe it and the boolean gate could not see the 0-d boolean a
policy comparison produces. It is now left for the value checks to read.
