### Feature

- Added `strands_robots.tools.g1.g1_list_swing_height_envelope` and
  `strands_robots.tools.g1.g1_swing_height_admits`, two read-only ``@tool``
  verbs that snapshot the swing-height clamp the neon bundle
  (``cagataycali/neon-the-g1/tools/_g1_common.py::set_swing_height`` and
  ``cagataycali/neon-the-g1/tools/g1_posture.py::g1_set_swing_height``)
  fronts on the SDK's raw ``LocoClient._Call(7103, ...)`` path. The
  Unitree SDK does not expose a public ``SetSwingHeight`` method: the
  setter is reachable only through the ``7103`` RPC, and the SDK
  itself places no clamps on the argument. The neon wrapper narrows
  every dispatch to ``max(0.0, min(0.2, float(height)))`` before the
  ``_Call``, so this envelope quotes ``[0.0, 0.2]`` m as the hard
  clamp (both bounds inclusive - a saturated command is a command)
  and quotes the neon-bundle-documented ``[0.05, 0.15]`` m
  "Typical safe range" as a softer recommendation. Both verbs surface
  ``inside_recommended`` on admitted values so a caller can decide
  the softer refusal itself without collapsing it into the hard-clamp
  admits decision; folding recommendation into admits would drop the
  neon bundle's own admitted ``[0.0, 0.05)`` and ``(0.15, 0.2]``
  sub-intervals. The two verbs let a caller decide the ``rc=7404``
  gate-refused refusal decidably before a future driver-side
  swing-height wrapper fires; ``g1_swing_height_admits`` also names
  which SDK dispatch path (``"call_7103"``) a query would take,
  quoting the same API id the neon bundle's
  ``_g1_common.set_swing_height`` invokes. Non-finite input
  (``math.inf``, ``math.nan``) refuses with
  ``comparison="non-finite"`` so a caller distinguishes a bounds
  violation from a shape one. Both verbs surface
  :data:`~strands_robots.tools.g1._g1_common.WALK_FSMS` so a caller
  comparing an intended write against both conditions (envelope +
  gate) has the FSM set on hand. No ``unitree_sdk2py`` submodule
  loads on import - the envelope table is a module-level snapshot,
  matching the SDK-load hygiene the rest of this package carries.
  Refs strands-labs/robots#358.
