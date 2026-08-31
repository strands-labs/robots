### Feature

- Added `strands_robots.tools.g1.g1_list_locomotion_duration_envelope`
  and `strands_robots.tools.g1.g1_locomotion_duration_admits`, two
  read-only ``@tool`` verbs that snapshot the duration clamp the neon
  bundle
  (``cagataycali/neon-the-g1/tools/g1_locomotion.py::g1_move_velocity``)
  fronts on the SDK's ``LocoClient.SetVelocity`` bounded-duration
  path. The Unitree SDK places no clamps on the ``duration_s``
  argument: an unfronted call with ``duration=3600.0`` reaches the
  controller unchanged and walks the robot for an hour, and
  ``duration=0.0`` reaches it as a no-op the SDK does not refuse. The
  neon wrapper narrows every dispatch to
  ``max(0.0, min(10.0, float(duration)))`` and then refuses
  ``duration <= 0`` outright with the message ``"duration<=0
  (non-continuous), refusing"`` before the SDK is touched, so this
  envelope quotes ``(0.0, 10.0]`` s as the admitted range - the lower
  bound **strict** (the neon wrapper refuses the boundary value
  itself before dispatch) and the upper bound **inclusive** (a
  saturated 10-second command is a command the neon wrapper does
  dispatch). Both verbs surface the two SDK method names the neon
  bundle routes between - ``SetVelocity`` for the bounded branch and
  ``Move`` (with the SDK's own ``continous_move=True`` spelling) for
  the unbounded ``continuous=True`` cousin - so a caller who wants a
  longer walk sees the alternative dispatch path on the envelope
  rather than having the neon wrapper drop the excess seconds
  silently. ``g1_locomotion_duration_admits`` names
  ``route="set_velocity"`` on an admitted value and ``None`` on a
  refused one so a caller distinguishes the dispatch decision from
  the admission decision. Non-finite input (``math.inf``,
  ``math.nan``) refuses with ``comparison="non-finite"`` so a caller
  distinguishes a bounds violation from a shape one. Both verbs
  surface :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` so a
  caller comparing an intended write against both conditions
  (envelope + gate) has the FSM set on hand. No ``unitree_sdk2py``
  submodule loads on import - the envelope table is a module-level
  snapshot, matching the SDK-load hygiene the rest of this package
  carries. Refs strands-labs/robots#358.
