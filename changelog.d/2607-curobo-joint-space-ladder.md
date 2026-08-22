### Fixed

- **policies/curobo**: the documented joint-space dispatch ladder now numbers a rung
  per planner entry point, in the order `_plan_and_cache` probes them. It numbered two
  rungs for a ladder of three and folded the other probes into the first rung's
  parenthetical, so `plan_single_js` read as both rung one's second probe and rung
  two's subject, and `plan_single` - the entry point the code actually ends on - was
  numbered nowhere. A reader sizing a stub planner from the list was told a planner
  exposing only `plan_single` has no joint-space path, when it is the rung that plans
  the goal. The Cartesian ladder in the same docstring already numbered exactly the two
  entry points it probes, and the joint-space branch's own comment already stated the
  fallback condition correctly, so no behaviour changes: the probe order and every
  dispatch outcome are identical. Both ladders are now graded against the AST rather
  than against other prose - a rung per entry point in probe order, one entry point per
  rung - and the last resort is driven through the real dispatch for the first time.
