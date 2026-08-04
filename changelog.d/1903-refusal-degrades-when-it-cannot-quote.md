### Fixed: a refusal that cannot quote what it refuses degrades instead of raising

`utils.name_list_error` built its refusal text by reading the caller's value, and
those reads were unguarded. The mapping branch offered the mapping's own keys as
its remedy:

```python
f"pass the names as a list: {_refusal_container_repr(list(value))}."
```

The render was safe; building its argument ran the caller's `__iter__`, so a
mapping whose keys cannot be read escaped the guard whose whole purpose is to
answer an unusable input with text:

```
name_list_error(HostileKeyMapping(), "cameras", "render_all")  ->  RuntimeError: no keys for you
```

This is the same defect as #1873, #1874, #1875, #1878, #1888, #1889 and #1897, on
the one path where the read was never load-bearing. Everywhere else the verdict
depended on the read, so a read that failed *became* the verdict. Here the verdict
is settled before the read happens - a mapping is refused whatever its keys say -
so only the advice was unmeasurable, and what a refusal says when it cannot
compute its own remedy had no precedent in the file to follow.

It now keeps the verdict and degrades the clause that wanted the quotation, which
is the only available answer that states both what was measured and what was not:

```
render_all: cameras must be a list of names, not a mapping, got <HostileKeyMapping object ...>.
  A mapping is iterable over its keys, so its values would be discarded - pass the names as
  a list; its own keys could not be read to quote them here (RuntimeError: no keys for you).
```

The read failure is rendered by `_describe_failed_read`, which is the stem
`_read_name_list` already used, so a refusal degraded here reads like every other
read failure in the module rather than introducing a second vocabulary for one
branch. Dropping the verdict instead would have stopped naming the mistake the
caller actually made, and dropping the remedy silently would have omitted advice
on one path without saying why.

**#1903 described one read. There were five, across two branches.** The string
branch was the weaker of the two and none of its four escapes had been filed
anywhere: it produced the quoted characters with a comprehension over the
caller's value, counted them with a second `len` the first read was not obliged
to agree with, interpolated them with a bare f-string rather than the elementwise
renderer - so an element that cannot print re-raised #1875 inside the clause
quoting it - and called the overridable `bytes.decode` before any of that.
A `str` subclass or a `bytes` subclass reaches all four:

```
name_list_error(HostileCharacterStr("wrist"), ...)  ->  RuntimeError: no characters for you
name_list_error(HostileLengthStr("wrist"), ...)     ->  RuntimeError: no length for you
name_list_error(UndecodableBytes(b"wrist"), ...)    ->  RuntimeError: no decoding for you
```

The characters and the count beside them now come from one read, which is #1897's
property applied to a message rather than to a verdict: they can no longer
describe different values.

Every accepted and refused message is unchanged, asserted as exact text rather
than as substrings.

The escapes were invisible to `TestNoGuardRendersACallerValueDirectly`, which
reports a caller value *rendered* without a shared renderer and is satisfied
completely by `_refusal_container_repr(list(value))`. Its read-side companion,
`TestNoGuardReadsACallerValueOutsideATry`, is added beside it: it scans `utils.py`
for a public guard that runs the caller's own code outside a `try`, follows the
value through assignments so a local holding it is covered too, and stops at the
helpers that read inside a `try` and hand back an ordinary value. That scan is
what found the four unfiled reads above, and three more it reports are recorded
as #1906 with a pin that measures them - each `coerce_*` guard validates its value
with a guard that reads it and then reads it again, so `coerce_rgba` raises on the
wrong-length branch that exists to return text.
