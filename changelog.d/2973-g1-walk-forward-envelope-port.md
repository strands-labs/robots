### Feature

- Added `strands_robots.tools.g1.g1_list_walk_forward_envelope` and
  `strands_robots.tools.g1.g1_walk_forward_admits`, two read-only ``@tool``
  verbs that snapshot the ``(distance, speed)`` clamp pair the neon bundle
  (``cagataycali/neon-the-g1/tools/g1_locomotion.py::g1_walk_forward``)
  places on the walk-forward surface before its ``LocoClient.SetVelocity``
  dispatch. The SDK itself places no clamps on either argument: a caller
  that passes ``distance=100.0`` or ``speed=5.0`` reaches the controller
  unchanged, so the envelope lives here rather than on the SDK side. The
  neon wrapper narrows the two arguments to
  ``distance = max(-1.0, min(1.0, float(distance)))`` and
  ``speed = max(0.05, min(0.5, abs(float(speed))))``, and the two verbs
  let a caller decide the ``rc=7404`` gate-refused refusal decidably
  before a future driver-side wrapper fires. Both clamps are inclusive
  at their boundaries (a saturated command is a command) and a
  strictly-negative ``distance`` is admitted as backwards-walk (the neon
  wrapper picks the sign of ``vx`` from the sign of ``distance``). The
  admits verb reports a ``composed_duration = abs(distance) / speed``
  the caller can pass on to
  ``strands_robots.tools.g1.g1_locomotion_duration_admits`` (port #2972);
  at the walk-forward clamps that composed duration can reach ``20.0``
  seconds, above the duration envelope's ``10.0`` s upper clamp, so a
  caller composing the two admission layers sees the overhang without
  computing the arithmetic itself. Non-finite input (``math.inf``,
  ``math.nan``) on either axis refuses with ``comparison="non-finite"``
  so a caller distinguishes a bounds violation from a shape one. Both
  verbs surface :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` so
  a caller comparing an intended write against every condition
  (distance envelope + speed envelope + FSM gate) has the FSM set on
  hand. No ``unitree_sdk2py`` submodule loads on import - the envelope
  table is a module-level snapshot, matching the SDK-load hygiene the
  rest of this package carries. Refs strands-labs/robots#358.
