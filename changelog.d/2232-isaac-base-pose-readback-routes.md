### Quality: every route the Isaac base-pose readback documents is now driven

`IsaacMotionPrimitivesMixin._articulation_base_pose` reads the articulation's
own world pose -- the frame `move_to` maps every world-frame target through --
and documents seven ways that read can fail, each answering `None` for a pose
callers "must answer loudly, never by substituting an origin base (a wrong base
makes every world-frame target silently wrong)".

One of the seven was driven. `test_motion_primitives.py`'s `_FakeArticulation`
carries no `get_world_pose` at all, and `test_articulation_read_write_surfaces.py`
-- which enumerates the articulation surfaces documented to answer loudly and
drives every failure arm of the other three -- did not list it, so the readback
fell between its two owners: `test_move_to_ik.py` pinned the single route where
the pose answers `None`, leaving the other six unreached along with the
documented torch-tensor surface and the quaternion normalization.

Every route is now driven on the plain-data surface and through `move_to`, whose
robot deliberately carries no `data_config` -- the next thing `move_to` needs
after the base pose -- so "`data_config` is not mentioned" is what separates a
refusal from a substitution. A route count pinned against the readback's own arms
fails until a route added later is driven here too. No production behaviour
changes.
