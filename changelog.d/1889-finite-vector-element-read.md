### Fixed: a vector whose read fails part-way is refused, naming the element it stopped at

`utils.finite_vector_error` probed `iter(vec)` inside a `try` and then walked the
iterator with a plain `for`, which cannot guard the call it makes. So an exception
raised while *producing* an element escaped exactly as one from `__iter__` did
before #1878 - the same defect one level in, on a guard whose entire purpose is to
answer an unusable input with a message rather than a traceback.

Three routes reached it and none needs a hostile type:

| value | why `iter()` cannot see it |
|---|---|
| `__getitem__` and no `__iter__` | CPython synthesises the iterator *without* calling `__getitem__`, so the raise lands on the first `next()` |
| a generator that fails after a yield | `iter()` of a generator cannot fail at all |
| a container mutated during its own read | the `RuntimeError` is the stdlib's; the value raises nothing |

The first is the 0-d array's own shape, which `simulation/base.py` already notes
this library receives ("declares `__len__` and `__getitem__`"), and the second is
a generator expression over a partially readable buffer - ordinary caller code.

The read is now guarded per element, and the refusal **names the element it
stopped at**:

```
raycast: 'origin[1]' could not be read: RuntimeError: stream truncated
  (got <generator object ...>). Pass a list or tuple of numbers.
```

That wording is not the `__iter__` verdict reworded, deliberately. A read that
stopped at element 4 had four components read and found finite, which is a
different measurement from a value whose iteration never began, and #1878's own
lesson is that a refusal must not state what was never measured. `could not be
iterated` is unchanged and still answers the value whose `__iter__` refuses.

The escape reached **four** guards, not the two the issue predicted. A legacy
sequence carries a readable `__len__`, so it clears the length check first and
arrives at the iteration through `pose_vector_error` / `coerce_pose_vector` and
`coerce_rgba` as well as `coerce_size_vector` - which is why the fix is in the
shared guard rather than at a call site, and why the sibling test's rationale
("neither ever reached an iteration") is corrected to say what it actually
measures: that *that probe* carries no length and is no `Sequence`.

The guard stays lazy, and that is now measured rather than asserted in prose - a
component is read only until the first unusable one. A materialising `list(vec)`
would have covered both halves in one clause and is declined for a stronger
reason than the memory it holds: it raises before any element has been examined,
so its verdict could not say how far the read got.

One escape of this shape is left open and is filed rather than absorbed here
(#1897): `name_list_error` tests `isinstance(value, Sequence)` and then walks the
value, so a registered `Sequence` whose item access raises still escapes it. That
guard refuses against a different domain - a name, not a number - and reads the
value more than once, so its wording is a separate decision.
