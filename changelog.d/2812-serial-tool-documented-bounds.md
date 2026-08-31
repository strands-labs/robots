### Fixed: the serial_tool bound the model is told is the bound the tool enforces

`serial_tool` is an agent tool, so each register field's `Args:` entry is lifted
into the tool's input schema and that schema is the whole of what the model
driving the bus knows about the field's domain. Two of the three bounds were
restated in prose beside a bound decided in code, and nothing compared the copies.

One had already drifted. `Goal_Velocity` is sign-magnitude, so a magnitude
reaching bit 15 is read by the servo as full speed in the *opposite* direction --
a different command rather than a faster one -- and the enforced ceiling was
tightened to 32767 for exactly that reason. The docstring kept saying `[0, 65535]`:

```
field     documented   enforced
velocity  [0, 65535]   [0, 32767]
```

So the tool advertised 32768 values it refuses, half the domain it offered, and a
model that took the schema at its word got a well-worded refusal for a call the
schema said was in range. The entry now states the ceiling that is applied and
the reason it is not the two-byte maximum.

The other copy is a drift that had not happened yet. The position full scale was
spelled twice -- once as the `position` ceiling, once as the divisor the reported
angle turns a count into degrees with -- while the ceiling's own reason string
claimed the two are "the same full scale". That claim was asserted and never
enforced, so a correction to either would have been invisible to the other. Both
now read one name.

`tests/tools/test_serial_tool_documented_bounds.py` grades both halves, deriving
the fields it checks from the module so a field added without a documented bound
is graded on arrival: the interval in the generated schema must equal the interval
the validator applies, the documented ceiling must be accepted where one past it
is refused, and the full scale must appear as a single integer literal in the
module. On the previous arrangement those cells report `velocity: the tool schema
tells the model (0, 65535) while the validator enforces (0, 32767)` and `the
position full scale 4095 is spelled as 2 integer literals, on lines [85, 410]`.

Still open, and deliberately untouched: which full scale is correct. Issue #2812
reports that the family the registry declares spans two resolutions (`sts3215` at
4096 counts, `scs0009` at 1024, the latter every motor of `hope_jr_hand`), so on an
SCS-series bus the ceiling is four times the addressable maximum and the reported
angle is four times low. Choosing between naming the model, reporting counts only,
and narrowing the claim to the STS series changes the tool's public surface and
wants hardware this suite does not have. Single-sourcing the number is the half
that issue asks for regardless of which way it is settled, and it makes that
correction one edit instead of two that can disagree.
