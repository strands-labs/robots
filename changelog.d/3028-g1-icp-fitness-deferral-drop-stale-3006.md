### Fixed: the ICP fitness envelope docstring no longer defers to a change this repository already landed

The module docstring on `strands_robots/tools/g1/g1_slam_icp_fitness_envelope.py`
described its voxel-size twin as "open for review as strands-labs/robots#3006"
and named that sibling as a literal rather than a cross-reference, on the
premise that a Sphinx role promises an importable dotted path and this tree
did not yet carry the module. PR #3006 has since landed and
`strands_robots.tools.g1.g1_slam_relocalize_envelope` is now importable, so
the deferral is stale in the same way `tests/test_deferral_strings_do_not_cite_a_landed_change.py`
already grades against: a reader following `#3006` finds a merged change
and cannot tell whether the sibling module is missing or the note is stale.

The two references now use the Sphinx `:mod:` role that the sibling
`g1_lidar_max_points_envelope` reference in the same paragraph already
used, folding the twin into the same read shape. No behavioural code
changes; only the docstring's forward-deferral references become
backward Sphinx cross-refs.

Refs #358.
