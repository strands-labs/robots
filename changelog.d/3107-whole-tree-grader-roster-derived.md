### Fixed: the whole-tree grader preflight derives its roster from the tree

`hatch run whole-tree-check` runs the graders whose input is the *rest* of the
repository, the class a `-k` keyword or a `tests/drivers/` path does not
collect. Its roster was a hand-written tuple of seven, so a grader added later
was absent from it by default - and the absence read in the reassuring
direction, because the preflight passed by never collecting the grader that
would have failed. A branch cited a green preflight while the required check
went red on `tests/test_mesh_pacing_ticker.py`, which walks the installed
package and was never named (#3105).

The roster is now read off the tree: a grader that walks the repository root or
one of its top-level Python areas is collected on arrival, so adding one needs
no second edit to register it. Membership is decided by the walk's resolved
*value* rather than its AST shape, which is what separates a repository walk
from a subject test globbing its own fixture directory - the reason an earlier
shape-based scan was rejected. All three spellings of the package root in use
here resolve, including `inspect.getfile(strands_robots)` and a root bound
under an alias.

The preflight now collects 62 graders instead of 7 (60 derived plus 2 that walk
no resolvable path and stay named explicitly, each with its reason). A run over
this tree goes from 298 to 1763 test outcomes and from 35s to 3m43s.
