### Fixed: a LIBERO end-effector orientation that encodes no rotation is dropped, not read as the identity

`LiberoAdapter` refused a config-supplied `eef_quat_offset` that was not a unit
quaternion but accepted any four numbers as the orientation read back from the
backend, and `_quat_wxyz_rotate_vec` normalised with `norm or 1.0` - so an
all-zero quaternion, the spelling of an orientation that was never written,
rotated a vector exactly as the identity does. The #1802 wrist offset was then
added along a world axis whatever the body was doing and returned as a corrected
position, and `[0, 0, 0, 0]` went out as the reported orientation; a NaN
component propagated through both. `_extract_pose` now drops an orientation that
encodes no rotation, exactly as it already dropped a wrong-length one, so the
caller takes its existing "reported no orientation" path; `_quat_wxyz_rotate_vec`
refuses one instead of substituting a norm, matching the settled
`policies/wbc/control.quat_rotate_inverse` sibling. A large-magnitude quaternion
is still accepted and normalised - it encodes an ordinary rotation.
