### Fixed: whole-tree guards locate the tree they grade instead of the working directory

Four guards walked the repository through a path relative to the process working
directory (`Path("tests")`, `Path("docs")`), so they graded whichever tree the run
started in. Two then reported that tree clean - including the guard that refuses a
test calling `pose_tool(action="emergency_stop")` with no explicit `port`, whose
hazard is de-energizing an arm plugged into the machine running the suite. Each
root is now derived from `Path(__file__)` or from a package symbol, the two sweeps
refuse a vacuous scan, and a new inventory refuses the next relative locator.
