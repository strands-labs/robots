### Fixed: the whole-tree preflight collects a grader whose walk root is bound inside the function

`scripts/check_whole_tree_graders.py` derives its roster from the tree: a grader
whose population is the rest of the repository walks the repository root or one
of its top-level Python areas, so the walk's receiver is resolved to a concrete
path and the module is selected on where it lands. The resolver read module-level
bindings, an area held in a loop variable (#3113) and a root arriving as a
function parameter (#3118). It did not read a root **assigned on a line of the
function that walks it** - the plainest spelling of all, and the one a grader
reaches for when it keeps its sweep beside the assertion it feeds:

```python
def test_no_publish_loop_still_paces_on_an_inflated_wait() -> None:
    root = Path(__file__).resolve().parent.parent / "strands_robots"
    for path in sorted(root.rglob("*.py")):
```

Eight graders were unrostered on it, and the absence is silent in the reassuring
direction - a preflight run passes without having collected them:

| grader | population | outcomes |
| --- | --- | --- |
| `tests/tools/test_gr00t_numeric_option_guards.py` | every port-taking tool in the package | 148 |
| `tests/test_repr_survives_partial_construction.py` | every class in the package | 49 |
| `tests/mesh/test_gateway_mesh_kill_switch.py` | every module in the package | 36 |
| `tests/tools/test_rosbridge_transport_port_limit.py` | every transport tool in the package | 24 |
| `tests/drivers/test_booster_refuses_a_vendor_enum_member_it_cannot_read.py` | every module in the package | 22 |
| `tests/test_episode_frame_range_is_one_row_read.py` | every module in the package | 20 |
| `tests/test_mesh_state_loop_rate.py` | every pacing loop in the package | 17 |
| `tests/mesh/test_acl_cache_docstring_matches_the_cache.py` | every module in the package | 13 |

`_enclosing_assignments` reads the root from the assignments the functions
enclosing the walk own, outermost first so a name rebound closer to the walk wins,
as Python resolves it. Scoping is the whole of the fix's safety, and the tree
carries a live measure of it: `tests/policies/curobo/test_action_horizon_domain.py`
binds the repository root in one test method - to *read* a file, not to walk one -
and walks a subpackage in another. Reading every assignment in the module would
let the first lend its root to the second and select the file; reading only the
assignments the enclosing functions own leaves it out, which is correct, because a
path-scoped run over `tests/policies/` collects it already.

Roster 76 -> 84 graders, 2542 -> 2871 outcomes - the increase is exactly the 329
cells those eight files hold - with nothing dropped and the eight
`tests/simulation` backend sweeps still excluded.
