### Added: a native Modbus TCP driver for the Robotiq 2F-85 gripper

`robotiq_2f85` had no driver of either kind. lerobot registers no robot type for
a gripper -- it is an end effector, not an arm -- and no native driver was
registered against it, so `Robot("robotiq_2f85", mode="real")` could not reach
the hardware by any route.

`RobotiqDriver` speaks Modbus TCP directly: three registers out, three back, one
socket, and no new dependency. The command path is wired end to end rather than
deferred behind a "not wired yet" envelope, because a six-byte register map and
a TCP socket is the whole protocol.

Two behaviours the register map alone does not give a caller. Activation is part
of connecting: a 2F-85 powers up unactivated and *ignores every position
command* until it has run its open-close calibration stroke, reporting no error
and simply not moving, so `connect_eagerly()` performs the sequence and waits
for `gSTA` rather than returning on an open socket. And a grasp is distinguished
from an empty close: `gOBJ` says whether the fingers stopped at the commanded
position or because something is between them, and reading only "stopped"
reports every empty close as a successful pick.

Position counts run backwards to the aperture a caller measures -- `rPR=0` is
fully open -- which is the protocol's one sign trap, so the inversion is written
in exactly one place and pinned in both directions.

Both registry entries now declare `hardware.driver="strands"`, because lerobot
cannot build them and the native driver is the only thing that can, so
`mode="real"` resolves here without naming a driver.

The tests run against a scripted gripper that answers real MBAP frames over a
real socket and enforces the hardware's own preconditions: a position lands only
when the gripper is activated and the frame set `rGTO`. A driver that built a
perfect frame but skipped activation returns identical envelopes and moves
nothing, so the assertions read what reached the wire instead.
