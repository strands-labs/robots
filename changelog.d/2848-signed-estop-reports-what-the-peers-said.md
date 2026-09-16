### Fixed: the dashboard's signed e-stop reports what the peers said, not only that the rail latched

`Mesh.emergency_stop` engages the issuer's lockout unconditionally, before it
broadcasts anything, so `lockout_engaged` is true whatever the fleet does.
`MeshBridge.signed_estop` returned that and nothing else, and the sheet rendered
it as `peers refuse all commands until resumed` - a claim about the room made
from a fact about this process:

```python
{k: v for k, v in signed_estop().items() if k != "responses"}
# every peer acknowledged        -> {"signed": True, "issuer": "dash-safety", "lockout_engaged": True}
# NOBODY answered at all         -> {"signed": True, "issuer": "dash-safety", "lockout_engaged": True}
# two answered, NEITHER stopped  -> {"signed": True, "issuer": "dash-safety", "lockout_engaged": True}
```

One payload for three fleets, on the one screen an operator reads to decide
whether to reach for the hardware cutoff.

`lockout_engaged` keeps its value, because it is true and because the resume
control is gated on it - reporting `False` on an unacknowledged stop would hide
the only way to clear a lockout that really is engaged. The peer half is now
carried alongside it, from the accounting `emergency_stop` already computes for
its `strands/safety/estop` envelope: `responses_received` (replies received, not
stops confirmed, keeping the meaning #1680 gave it) and `peers_not_stopped`
(responders that affirmatively reported they did not stop). The grading stays in
`mesh.core`, since a second copy of that rule on the safety path is how the sim
dispatch branch once answered `ok=True` over a refusal it had in hand.

The sheet's sentence derives from those two fields rather than from the latch -
a lockout refuses the NEXT command and does not halt motion already underway,
which is the distinction that decides whether an operator walks to the power
supply. That half lives in the dashboard frontend and lands with the server
slice of the #2848 decomposition (#2977).
