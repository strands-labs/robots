### Fixed: the vector coercions count the components they read

`coerce_rgba` and `coerce_size_vector` took their component *count* from
`sequence_length(value)` - the value's `__len__` - and their components from the
element read. Those are two independent reads of one value, and nothing obliges a
`Sequence` to answer them the same way:

```python
class OverstatesItsLength(Sequence):
    def __len__(self): return 4          # reports four components
    def __getitem__(self, i):
        if i >= 3: raise IndexError(i)   # the read yields three
        return [0.1, 0.2, 0.3][i]
```

| call | before | after |
|---|---|---|
| `coerce_rgba("add_object", "color", OverstatesItsLength())` | `([0.1, 0.2, 0.3], None)` | `([0.1, 0.2, 0.3, 1.0], None)` |
| `coerce_pose_vector("add_object", "position", OverstatesItsLength(), 4)` | `([0.1, 0.2, 0.3], None)` | refused |
| `coerce_rgba("add_object", "color", UnderstatesItsLength())` | refused as `got 2: [0.1, 0.2, 0.3]` | `([0.1, 0.2, 0.3, 1.0], None)` |

Row 1 breaks a documented promise. `coerce_rgba` returns "exactly 4 finite
floats" so that the `color[:3]` reads the shape builders do are well-defined by
construction; the alpha completion was gated on the *reported* count, so a
`__len__` of 4 skipped it while the read had supplied three - the return value
was short precisely because the count was believed over the components. Row 2 is
the same defect on a buffer: a 3-component list where a wxyz quaternion is
promised reaches `data.qpos`, which is the bare `ValueError` inside the numpy
assignment `pose_vector_error` exists to prevent, now reached *through* the guard
instead of around it. Row 3 is a refusal that contradicted itself in one
sentence - it named a count of 2 and then quoted three components.

The count now comes from the read that produced the components:

- `coerce_rgba` counts `len(floats)`, so "exactly 4" holds for any accepted value
  rather than only for one whose two reads agree.
- `coerce_size_vector`'s empty-vector refusal asks whether the read produced a
  component, not whether `__len__` reported zero. A value whose length reports
  three extents and whose read yields none has no extent to write; one reporting
  zero whose read yields components has one. The per-shape count remains #1858's
  decision and is untouched here.
- `_read_pose_vector` keeps its length gate ahead of the element read - refusing a
  wrong *reported* length without producing a component is worth keeping - and
  re-checks the accepted count against what the read yielded, so no accepted pose
  vector can have the wrong component count whichever read disagrees:

  ```
  add_object: 'position' must be a 4-element vector, got 3: [0.1, 0.2, 0.3].
    Its length reported 4, so the components it produced are not the vector its
    length promised.
  ```

`sequence_length` stays where it answers its own question - *is this a sized
sequence at all* - which is what refuses a generator on both coercions and is
guarded (#1888) so it cannot raise.

This is the count read beside the component read #1906 fixed, and completes the
property that fix stated for the components alone: one read makes the verdict,
the count, and the text it quotes.
