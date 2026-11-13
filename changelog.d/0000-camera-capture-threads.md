### Changed: each camera is read on its own capture thread and opens at connect

A hardware camera is read on a capture thread of its own, and a step takes each camera's latest frame instead of
waiting on the control thread for that camera's next one. With two 30 fps cameras in a 30 Hz loop a step no
longer waits up to a frame period per camera. `Robot(..., cameras=...)` opens all cameras at connect, side by side,
and refuses one that cannot open or sends no frame in 10 s; before, they opened one after another inside the first
`observe` or `run`. A camera whose latest frame is older than 1 s is left out, as a failed read was.
`Cameras.stamps` and `Cameras.ages()` give when each returned frame was captured.
