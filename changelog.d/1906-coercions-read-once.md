### Fixed: the vector coercions compute their floats from the read their guard already made

`coerce_pose_vector`, `coerce_rgba` and `coerce_size_vector` each validated the
caller's value with a guard that reads it, and then read the value *again*,
unguarded, to build the floats:

```python
if (err := finite_vector_error(method, param_name, color)) is not None:
    return None, err
floats = [float(component) for component in color]   # second read
```

The two reads were independent, so nothing obliged them to agree, and nothing
guarded the second. A value that answers one read and refuses the next escaped
all three:

```
coerce_rgba("add_object", "color", FailsOnItsSecondRead(0.1, 0.2, 0.3))  ->  RuntimeError: no second read for you
coerce_size_vector("add_object", "size", FailsOnItsSecondRead(0.1, 0.2, 0.3))  ->  RuntimeError
coerce_pose_vector("add_object", "position", FailsOnItsSecondRead(0.1, 0.2, 0.3), 3)  ->  RuntimeError
```

Two of those reads build the return value, so "a refusal must not raise" does not
reach them. `coerce_rgba` is the one that breaks a stated contract: its second
read is quoted by the wrong-length refusal (`got {length}: {floats}`), and it
happens *before* the length is compared, so a two-component colour raised on
exactly the path whose purpose is to answer an unusable colour with text.

The element read now lives in `_read_finite_vector`, which performs the read
`finite_vector_error` always performed - element by element inside a `try`, since
#1889 - and returns the floats it built alongside the verdict. `_read_pose_vector`
is the same split for the fixed-length wrapper, keeping the length probe ahead of
any element read so a wrong-length vector is still refused without producing one.
`finite_vector_error` and `pose_vector_error` keep their signatures and their exact
messages, because 24 call sites across four modules want a verdict and nothing
else: the read moved, not the guard.

So each coercion now keeps a list this module built, and `coerce_rgba`'s refusal
quotes the components its own domain checks examined:

```
add_object: 'color' must have exactly 3 or 4 component(s) (RGB, or RGBA with alpha),
  got 2: [0.1, 0.2]. Pass every component - a partial 'color' cannot be applied
  without inventing the missing values.
```

This is #1897's property (a guard reads the caller's value once) on the three
guards that fix could not reach, and the last of the reads
`TestNoGuardReadsACallerValueOutsideATry` reported when #1903 added it -
`KNOWN_MESSAGE_READS` is now empty.

Three sibling pins moved with the code rather than being deleted, since each still
holds for a reason worth stating:

- `TestTheCoercionsSecondReadIsAStatedBoundary` measured the escape while it was a
  boundary; it becomes `TestTheCoercionsReadTheirValueOnce`, asserting the same
  probe in the accepting direction and asserting the read *count*, which is the
  property rather than its symptom.
- `TestTheRemainingConversionsRunOnlyOnTheAcceptedPath` named the three coercions
  as conversions that are safe because something upstream had already converted
  successfully. They no longer convert at all, so the module-wide scan is empty -
  and since an empty scan is also what a scanner matching nothing produces, the
  replacement asserts that the floats still come from a guarded read and that the
  scan still reports a planted conversion.
- Both "does this guard render through the shared renderer" scans now follow a
  guard into the read helper it delegates its verdict to; a body-only reading
  would call a delegating guard one that names nothing it refused.
