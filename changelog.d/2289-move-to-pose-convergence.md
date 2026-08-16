### Fixed

`move_to` now converges and reports the orientation a pose request asks for, instead of
measuring a 6-DOF target with a position-only metric. Previously the servo loop broke the
moment `position_error <= tol`, so a tolerance documented in METERS silently bounded the
ORIENTATION - the same call answered `reached=True` with the wrist anywhere from 0.76 to
20.45 degrees off the request depending only on `tol`, and the rotational error appeared
nowhere in the result. Acceptance and convergence now require the position within `tol`
metres AND the orientation within a new `orientation_tol` radians (default 0.1, clearing
the measured compliant-servo floor of the shipped arms), and the result carries
`orientation_error_rad` / `orientation_tol_rad` / `ik_orientation_residual_rad` whenever an
orientation was requested. An unreachable pose now names the out-of-reach component and
reports the same point solved position-only, rather than blaming a point that is reachable
and recommending two remedies that do not help. `orientation_tol` passed without an
`orientation` is refused rather than silently dropped. Position-only calls are unchanged.
