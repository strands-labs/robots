### Feature

- Added `strands_robots.tools.g1.g1_list_turn_envelope` and
  `strands_robots.tools.g1.g1_turn_admits`, two read-only ``@tool``
  verbs that snapshot the ``(angle_rad, yaw_rate)`` clamp pair the
  neon bundle (``cagataycali/neon-the-g1/tools/g1_locomotion.py::g1_turn``)
  narrows to before it dispatches ``LocoClient.SetVelocity`` for a
  yaw-only turn-in-place. The SDK itself places *no* clamps on
  either argument, so the envelope lives in the tool layer. The
  neon wrapper narrows ``angle_rad`` to
  ``max(-2*pi, min(2*pi, float(angle_rad)))`` and ``yaw_rate`` to
  ``max(0.1, min(0.6, abs(float(yaw_rate))))`` before composing
  ``duration = abs(angle_rad) / yaw_rate`` and picking the sign of
  ``vyaw`` from the sign of ``angle_rad`` (``+yaw_rate`` on
  non-negative angle, ``-yaw_rate`` on strictly-negative). Both
  verbs surface the composed ``duration`` at the arguments
  as-passed so a caller can chain the admission against
  ``g1_locomotion_duration_admits`` (port #2972) without composing
  the arithmetic itself; the returned envelope names the composed-
  duration overhang (``~62.83`` s at ``angle_rad=2*pi, yaw_rate=0.1``,
  above the duration envelope's ``10.0`` s upper clamp) so a caller
  comparing the two admission layers sees where they disagree.
  ``g1_turn_admits`` refuses on ``rc=7404`` with a per-axis
  ``dimension in {"angle_rad", "yaw_rate"}`` refusal descriptor;
  non-finite input (``math.inf``, ``math.nan``) refuses with
  ``comparison="non-finite"`` so a caller distinguishes a bounds
  violation from a shape one. Both verbs surface
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` so a caller
  comparing an intended write against both conditions (envelope +
  gate) has the FSM set on hand. No ``unitree_sdk2py`` submodule
  loads on import - the envelope table is a module-level snapshot,
  matching the SDK-load hygiene the rest of this package carries.
  Refs strands-labs/robots#358.
