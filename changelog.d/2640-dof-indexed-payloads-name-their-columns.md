### Fixed: a DOF-indexed simulation payload names the joint each column belongs to

`get_jacobian` and `get_mass_matrix` report arrays indexed by MuJoCo's DOF ordering
and previously stated only the width. That width does not determine the ordering: a
free or ball joint owns several consecutive DOFs (36 of the 62 registry sim robots
have `nv != njnt`, up to a gap of 20), and `nv` spans the whole compiled model, so in
a two-robot scene one robot's Jacobian columns are an interior slice. A caller pairing
`robot_joint_names(...)` with the leading columns silently reads another robot's
Jacobian - in a Jacobian IK step that means the arm does not move at all while the
call reports success.

Both payloads now carry `dof_joint_names`, a list of length `nv` naming the joint that
owns each entry in DOF order, with `None` where a joint is unnamed so the index
alignment is preserved. The numbers are unchanged, and `inverse_dynamics` - which
already returned its forces keyed by joint name - is untouched.
