### Feature

- Added `strands_robots.tools.g1.g1_list_stand_height_envelope` and
  `strands_robots.tools.g1.g1_stand_height_admits`, two read-only ``@tool``
  verbs that snapshot the ``LocoClient.SetStandHeight`` / ``LocoClient.HighStand``
  walkable range the neon bundle
  (``cagataycali/neon-the-g1/tools/g1_posture.py``) observed against the
  real robot on a gantry. The SDK itself places no clamps on the
  ``SetStandHeight`` argument: any finite value reaches the controller
  unchanged, so the envelope lives here rather than on the SDK side.
  The two verbs let a caller decide the ``rc=7404`` gate-refused
  refusal decidably before a future driver-side ``SetStandHeight``
  wrapper fires; ``g1_stand_height_admits`` also names which SDK
  dispatch path (``"set_stand_height"`` or ``"high_stand"``) a query
  would take, matching the neon bundle's own sentinel convention that
  routes a strictly-negative ``height`` argument to
  ``LocoClient.HighStand()`` (which itself dispatches
  ``SetStandHeight`` with the SDK's ``UINT32_MAX = 2**32 - 1``
  sentinel). Both the low and high bounds of the ``SetStandHeight``
  range admit their boundary (a saturated posture is still a posture
  command). ``-0.0`` routes as LOW-stand (not HighStand) to match the
  neon bundle's ``if height < 0`` conditional, which reads negative
  zero as non-negative per IEEE-754. Non-finite input (``math.inf``,
  ``math.nan``) refuses with ``comparison="non-finite"`` so a caller
  distinguishes a bounds violation from a shape one. Both verbs
  surface :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` so a
  caller comparing an intended write against both conditions (envelope
  + gate) has the FSM set on hand. No ``unitree_sdk2py`` submodule
  loads on import - the envelope table is a module-level snapshot,
  matching the SDK-load hygiene the rest of this package carries.
  Refs strands-labs/robots#358.
