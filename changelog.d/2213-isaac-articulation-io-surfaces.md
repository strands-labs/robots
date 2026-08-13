### Quality: drive the Isaac articulation layer's fallback limit source and every read/write failure report

`IsaacMotionPrimitivesMixin` documents two sources for a DOF's joint limits -
the authoritative `dof_properties` structured array and the view-shaped
`get_dof_limits()` fallback - and documents that a read or a write it cannot
complete is reported, never answered with substituted zeros. The existing
motion-primitive suite's fake articulation always supplies `dof_properties` and
always answers a read, so only the first source was exercised and every
read/write failure report was unreached: 4 of the layer's 24 documented
behaviours were driven.

Adds `tests/simulation/isaac/test_articulation_read_write_surfaces.py`, which
drives the fallback source in each shape Isaac reports it in (plain, view-shaped
and torch-tensor), every "no usable bounds" outcome, both position-read
surfaces, the write failure, and the six reports `set_gripper` and
`rotate_wrist` answer with when a read or a write fails. Tests only - no
production behaviour changes.
