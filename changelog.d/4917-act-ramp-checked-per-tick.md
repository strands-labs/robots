### Fixed: a timed move from an agent, the dashboard or the mesh checks every point it writes

`confirm_act(..., seconds=)` used to run the safety chain once, on the target, and then ramp toward it unchecked. A
joint-space line between two poses inside a `WorkspaceBoundsValidator` box can leave the box. An interlock that
reads a door or a light curtain was also asked only at the start of a move of up to 10 s. Now the operator is still
asked once, for the target, and every ramp point passes the interlocks and validators again. A refused point is
never written: the move stops and the robot holds the last pose it reached.
