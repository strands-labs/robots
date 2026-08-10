### Quality: pin the motion primitives' numeric-field domains

`move_to` / `set_gripper` / `rotate_wrist` each document a numeric domain for
their tolerance and their control-tick budget and each promises "Never raises",
but only half of each continuous guard was exercised. The guards read
`not _is_finite_real(x) or float(x) <= 0.0`, and the suite drove only values the
comparison rejects (`tol=0.0`, `tol=-1`), so `_is_finite_real` never once
returned `False` and `_validate_step_budget` never once took its type branch:
`nan`, `inf`, `bool`, `str` and `list` on any of the six numeric fields were
entirely unverified. Adds a behavioural suite covering the shared predicates
both ways, all six fields, the guard placement (a refused call leaves `qpos`,
`ctrl` and the sim clock bit-identical) and the two spellings that are *not*
domain errors - an omitted `target_yaw` and a NumPy integer budget.
