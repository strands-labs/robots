### Fixed: the whole-tree preflight collects a grader whose walk root is a parameter

`scripts/check_whole_tree_graders.py` derives its roster from the tree: a grader
whose population is the rest of the repository walks the repository root or one
of its top-level Python areas, so the walk's receiver is resolved to a concrete
path and the module is selected on where it lands. The resolver read module-level
bindings and, since #3113, an area held in a loop variable. It did not read a
root that arrives as a **function parameter**.

That is the spelling a grader reaches for as soon as it plants source for its own
predicate: the sweep and the planted controls share one implementation, so the
walk moves into a helper and the receiver is bound per call rather than at module
scope.

```python
def _scan(root: Path) -> list[str]: ...   # root.rglob("*.py")
_SURFACES = _scan(_PACKAGE_ROOT)          # the real sweep
... _scan(tmp_path)                       # a planted control
```

Five graders were unrostered on it, and the absence is silent in the reassuring
direction - a preflight run passes without having collected them:

| grader | population | outcomes |
| --- | --- | --- |
| `tests/test_args_docstring_completeness.py` | every documented surface in the package | 545 |
| `tests/test_attributes_docstring_completeness.py` | every dataclass in the package | 40 |
| `tests/test_raises_docstring_completeness.py` | every raising surface in the package | 18 |
| `tests/test_mujoco_render_assertions_are_gl_gated.py` | all of `tests/` | 13 |
| `tests/simulation/test_no_import_cycle.py` | the package import graph | 6 |

The parameter is resolvable from the module's *own* calls, which is what the fix
reads: every argument a call in the same module passes for that parameter, plus
any default the helper declares. The asymmetry that makes such a module a grader
is visible there - the planted call passes a `tmp_path` and resolves to nothing,
the real call passes the package root. Roster 70 -> 75 graders, 1910 -> 2532
outcomes, with nothing dropped and the eight `tests/simulation` backend sweeps
still excluded (a path-scoped run over the mirroring test directory collects
them, so they are not in the class this script rescues).
