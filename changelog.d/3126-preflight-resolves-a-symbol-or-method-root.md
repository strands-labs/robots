### Fixed: the whole-tree preflight resolves a grader root named by a symbol or a method

`scripts/check_whole_tree_graders.py` derives its roster by resolving each
test's directory walk to a concrete path. Two ways this repository names that
path did not resolve, so ten graders whose input is the whole package were
never collected - and the absence is silent in the reassuring direction, since
a preflight run passes by never collecting the grader that would have failed.

Both name the root through one level of indirection the resolver did not read.
A root derived from an *imported symbol* now resolves, because `inspect.getfile`
answers the same fact for a class or function as for a module:

```python
from strands_robots.simulation.base import SimEngine

root = pathlib.Path(inspect.getfile(SimEngine)).parents[1]
```

A no-argument helper declared as a *member of the test class* that uses it now
resolves too - it is the same helper already read at module scope, one scope in.

A symbol resolves only through the module that defines it, checked against that
module's own top-level names. `strands_robots.simulation` re-exports
`Simulation` from `simulation/mujoco/simulation.py`, so resolving the import to
the re-exporting file would answer with the package root where the truth is the
`simulation` subpackage - a wrong answer rather than a rescue.

The derived roster goes from 87 graders to 97, dropping none, and a preflight
run collects 1,156 test outcomes it previously skipped.
