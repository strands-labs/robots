### Fixed: a name list is read once, and a read that cannot finish is refused

`utils.name_list_error` tested `isinstance(value, Sequence)` and then read the
value on every branch that needed it, guarding none of them - the per-entry walk,
the duplicate walk, and the duplicate refusal's own `list(value)`. An acceptable
value was therefore read **twice** and a repeated one **three times**, and a
`Sequence` whose item access raises escaped from any of them:

```
name_list_error(HostileSeq(), "cameras", "render_all")  ->  RuntimeError: seq item exploded
```

A registered `Sequence` is not an exotic shape - `collections.abc.Sequence` is
what a proxy or a lazily-backed name list subclasses in order to be accepted here
at all - so the type check passing said nothing about the read succeeding. This is
the same defect as #1873, #1874, #1875, #1878, #1888 and #1889, on the one guard
those fixes could not reach: it refuses their probes on the `Sequence` check
before arriving at a walk.

The value is now read once, entry by entry, and a read that cannot finish is a
verdict:

```
render_all: cameras[2] could not be read: RuntimeError: backing store unavailable
  (got <FailsPartWay object ...>). Pass a list or tuple of names.
```

Both stems are the ones `finite_vector_error` already uses, with this guard's own
unquoted `{param}[{i}]` index and a remedy worded for names. A read that never
began is reported without an index, because there is no entry to name - the index
is what says two names were read and found usable.

Reading once also corrects a verdict that involved no exception at all. The reads
were independent, so nothing obliged them to agree, and a two-entry `Sequence`
answering `["top", "wrist"]` and then `["top", "top"]` cleared the per-entry
checks against its first read and was refused as a repeat against its second, in
a message rendering a third - a verdict, a check and a quotation describing three
different lists. It is now accepted, on the one read that was examined.

The read stays entry by entry rather than `list(value)` for the reason #1889
declined to materialise, reached from the opposite direction: this guard collects
the whole value either way, since a repeat cannot be ruled out without reading
every entry, so laziness buys it no earlier verdict - only the index.

Every existing verdict is unchanged, and the read count is pinned, being the one
property no refusal text would reveal.

One route in the same guard is measured and left open as #1903: the mapping
refusal quotes the mapping's own keys as its remedy, and that read can raise too.
It is a different question - the verdict there is never in doubt, since a mapping
is refused whatever its keys say, so only the *advice* is unmeasurable - and what
a refusal should say when it cannot compute its own remedy has no precedent here
to follow.
