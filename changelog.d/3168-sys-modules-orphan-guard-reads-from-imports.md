### Fixed: the `sys.modules` orphan guard reads `from package import module` bindings

`tests/test_sys_modules_removal_leaves_no_orphan.py` grades a rule its own
docstring states: a test may not leave `sys.modules` missing a module a sibling
patches, because removing an entry does not undo an import, it *orphans* every
reference already bound to that module -- the sibling's double is installed on
an object nothing will look at again.

Its binding collector walked only `ast.Import`. A module bound by
`from a import b` was invisible, which is the spelling this tree uses for
nearly every submodule, because a submodule is what a test wants to patch
attributes on. The protected set read 43 modules where the rule describes 95,
and the 52 it could not see included two unrestored removals, each with a
symptom that appears only in a composed run:

* `tests/tools/g1/test_motion_switcher_decoder.py` dropped
  `strands_robots.tools.g1._motion_switcher`, orphaning the reference
  `tests/drivers/test_motion_switcher_open_is_under_the_shared_dds_lock.py`
  patches `_load_motion_switcher_client` on. The Go2 driver's lazy import then
  resolved the real loader, the open returned `None`, and the cells that grade
  whether an RPC client's DDS endpoints are constructed under the shared
  `_DDS_INIT_LOCK` reported "the open returned None instead of the recorder, so
  this cell graded nothing". That lock exists because concurrent endpoint
  construction segfaults the CycloneDDS bindings, which no `except` boundary
  can turn into an error envelope.
* `tests/simulation/test_policy_runner.py` dropped
  `strands_robots.simulation.policy_runner`, and four cells in
  `tests/simulation/test_recording_frame_loss_is_not_tolerated.py` then failed
  with `KeyError` on that entry.

Both files pass in isolation, so neither removal is visible except in a run
that composes the pair.

The collector now reads `ast.ImportFrom` as well, and both removals go through
`monkeypatch.delitem`, the restoring idiom the guard's docstring already
recommends. Recording a `from` import that names something other than a module
costs nothing: the rule only intersects protected names with the literal keys a
test removes from `sys.modules`, and a class is not registered there under that
spelling, so the collector stays a static read with no import side effects.

`test_policy_runner_import_does_not_pull_in_mujoco` also never performed the
import it measures. It cleared every `mujoco` entry and then read `sys.modules`,
so it asserted only that the entries it had just deleted were gone -- true
whatever the runner's top level does. A module-level `import mujoco` planted in
`policy_runner.py` passed it. The cell imports the module now and fails on that
plant.

The docstring described the `policy_runner` removal as a purge nothing patches,
and a cell asserted the runner was unprotected. Both are corrected: two sibling
modules patch it through a `from` import.
