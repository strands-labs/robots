### Feature

- Added `strands_robots.tools.g1.g1_list_velocity_envelope` and
  `strands_robots.tools.g1.g1_velocity_admits`, two read-only ``@tool``
  verbs that snapshot the ``LocoClient.SetVelocity`` walkable range the
  neon bundle (``cagataycali/neon-the-g1/tools/g1_locomotion.py``)
  observed against the real robot on a gantry. The SDK itself places no
  clamps on ``(vx, vy, vyaw, duration)``: any finite argument reaches
  the controller unchanged, so the envelope lives here rather than on
  the SDK side. The two verbs let a caller decide the ``rc=7404``
  gate-refused refusal decidably before a future driver-side
  ``SetVelocity`` wrapper fires; ``g1_velocity_admits`` names every
  violated dimension in one call rather than one per dispatch attempt.
  The three abs-max clamps admit their boundary (a saturated command is
  still a command) while ``duration_min_seconds`` is exclusive, so a
  ``duration`` of exactly zero - a distance/speed computation that
  rounded down, say - refuses rather than reading as an admitted walk
  the controller would silently drop. Both verbs surface
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS`
  so a caller comparing an intended write against both conditions
  (envelope + gate) has the FSM set on hand. No ``unitree_sdk2py``
  submodule loads on import - the envelope table is a module-level
  snapshot, matching the SDK-load hygiene the rest of this package
  carries. Refs strands-labs/robots#358.
