### Fixed: `add_camera(parent_body=...)` on the Isaac backend reports the unsupported mount instead of raising `TypeError`

Mounting a camera on a body (a wrist view that rides with the arm) is what
`docs/policies/camera-naming.md` prescribes, backend-agnostically, for a VLA that
declares an `observation.images.wrist_image` feature. Two of the three simulation
backends implement it. The Isaac backend does not, and did not declare the
parameter either, so the gap was answered by Python:
`TypeError: IsaacSimulation.add_camera() got an unexpected keyword argument
'parent_body'` - naming the parameter but not the capability at stake, not the
two backends that do mount, and not the world-fixed alternative Isaac does
support, and arriving as an exception out of a method whose contract is the
`{"status", "content"}` envelope.

`IsaacSimulation.add_camera` now declares `parent_body` and refuses it with a
structured error naming the reason and both mounting backends - the shape
`IsaacSimulation.add_robot` already uses for `mjcf_path`, in the same position
(before the lock and before validating the parameters the call would not use).
Omitting it is unchanged: a world-fixed camera, which this backend supports.
Implementing the mount needs an Isaac Sim runtime to verify against and is
tracked separately; this makes the gap visible and actionable in the meantime.
