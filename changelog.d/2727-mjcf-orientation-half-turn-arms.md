### Tests

- **simulation/isaac**: MJCF's five orientation spellings are graded against MuJoCo's
  compiler, but every declaration those suites use names a rotation of less than a third
  of a turn and is well formed, so two families of arms in the reader were reached by no
  fixture. Converting a rotation matrix to a quaternion from its trace alone degenerates
  at a half turn - the term it divides by reaches zero - so the reader branches on
  whichever diagonal term is largest instead, which is the case its own docstring says
  the branching exists for, and three of those four arms ran in no test. The arms a
  declaration with the wrong number of components or a zero-length axis takes were unrun
  for a different reason: only a value that is not a number at all was graded, and that
  takes the earlier `float()` failure rather than any of them. Measured over
  `tests/simulation/isaac` beforehand, 1431 passing: flipping the sign of the real part
  in any one of the three arms, reading a wrong matrix entry into a component, dropping
  the branching in favour of the trace form alone, dropping the antipodal-`zaxis` special
  case, and dropping any of the seven guards that keep a wrong-arity or zero-length axis
  from being divided by, each left all 1431 passing. The new file detects all fifteen, in
  fifteen distinct failing sets, with an unmutated control clean, and it says which
  fixture carries which weight: at an exact half turn the real part is zero so a sign
  flip there is invisible, which is what the ten-degrees-short rotations are for, while
  the skew rotations put all four components at distinct non-zero magnitudes so a wrong
  entry shows up. Widening a tie-break from `>` to `>=` is detected by nothing, which is
  correct - at a tie either arm yields the same rotation - and is recorded rather than
  papered over. Both readers of the format are driven, `load_mjcf_scene_objects` and
  `load_mjcf`, and every expectation for a model MuJoCo compiles is read off `body_quat`
  rather than restated by hand, with the refusal asserted as the premise for the models
  it rejects. `strands_robots/simulation/isaac/loaders.py` goes from 87% to 90% covered,
  89 uncovered lines to 73.
