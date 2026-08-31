### Fixed: a short-form docstring cross-reference names a member, not a call

The `g1_imu` and `g1_lidar_summary` verbs cited the driver's cache accessor as
`:meth:`G1Driver._snapshot("_imu")`` -- a role whose target is the literal
`_snapshot("_imu")`, which no class defines, so
`tests/test_docstring_xref_roles_resolve.py::test_short_form_xref_roles_resolve`
refused both. `G1Driver._snapshot` is the driver's only cache accessor and is
deliberately internal, so there is no public member to cite instead; both are
now spelled as the literal ``driver._snapshot("_imu")`` their own module
docstrings and the `g1_lidar_state` sibling already use.
