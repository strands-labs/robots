### Fixed: a length a validator cannot read is reported, whatever refused it

Every validator that accepts a vector first asks "how many components is this?",
and `utils.sequence_length` is the single owner of that question so it cannot be
answered two ways. It reported only a `TypeError` `__len__` as "no length", and
documented that as "the narrowest superset". It is not a superset. `len()` does
not return whatever `__len__` returned - it converts it to an index, and that
conversion has refusals of its own:

| `__len__` returns | `len()` raises | reached before |
|---|---|---|
| `3.5` | `TypeError` | yes |
| `-1` | `ValueError: __len__() should return >= 0` | no |
| `sys.maxsize + 1` | `OverflowError: cannot fit 'int' into an index-sized integer` | no |
| (raises itself) | that exception | no |

The two middle rows are the ones that matter: **the value raises nothing at
all.** Its `__len__` is ordinary Python returning an ordinary `int`, and CPython
refuses to convert it - so a length that is computed rather than stored, such as
`self._end - self._start` on a proxy whose window is inverted, escaped the probe
without anything being hostile. The last row is the argument that closed the
rendering escape, applied to `__len__`: it is a method like any other and owes
its caller nothing.

Every reader of the shared probe inherited the escape, and it landed past the
exact envelopes the probe exists to keep - the MuJoCo agent-tool router's
structured error for a rejected parameter, and `get_world_point`'s pixel checks,
which are documented as being there "to keep the never-raises envelope":

```
add_object(position=<length CPython cannot convert>)   # OverflowError, past dispatch
get_world_point(pixels=<__len__ that refuses>)         # the value's own exception
```

Such a value now reports as carrying no readable component count, which is what
the probe already answered for a 0-d array and a plain scalar alike. No caller
changed and no message changed: all of these answer the caller's question
identically, so one branch covers them and `None` needed no new vocabulary. Only
`len(value)` runs inside the clause, so widening it masks no logic of this
library's own, and `Exception` rather than `BaseException` keeps
`KeyboardInterrupt` and `SystemExit` propagating.

`pose_vector_error` had to be fixed separately, because it was not reading the
shared probe at all - it carried its own `try: len(vec) / except TypeError`, so
the rule had two implementations and widening the owner left the duplicate
untouched. It now reads through the owner, with both of its verdicts and their
wording unchanged. That the module has exactly one function asking `len()` of a
caller value is now scanned rather than assumed, keyed on the parameter annotated
`Any` - the same key the rendering scan uses for "the caller's value" - so a
second owner cannot be added beside the first again.

`coerce_size_vector` was never affected: its iteration guard runs first and
refuses a value with no iteration before any length is read.

One escape of this shape was left open here and tracked separately, then closed
in this same release cycle by #1898. `finite_vector_error` guarded `iter(vec)` and
walked the iterator in an unguarded loop, so an exception raised while producing
an element still escaped - `iter()` cannot fail for a generator, and for a value
with `__getitem__` and no `__iter__` CPython builds the iterator without calling
it. Its refusal indeed could not reuse the existing "could not be iterated" text,
since a vector that failed at element 4 of 7 had four components read and found
fine, so it names the element it stopped at instead. The laziness this paragraph
expected it to trade away was kept: the read is still one component at a time,
because a materialising `list(vec)` raises before any element has been examined
and so could not report how far the read got.
