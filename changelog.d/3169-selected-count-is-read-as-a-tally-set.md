### Fixed: a deselecting run is no longer reported as a run that aborted

`scripts/report_truncated_test_run.py` read the collection line's tallies as a
fixed sequence -- `deselected` immediately before `selected` -- and pytest does
not write them that way. `TerminalReporter.report_collect` appends up to four,
each only when its own count is nonzero, so `selected` carries a different
prefix in every combination. On a line pytest 9.1.1 really emits,
`collected 5 items / 2 deselected / 1 skipped / 3 selected`, neither optional
group matched and the parser fell back to `selected = collected` -- the
*pre*-deselection number.

That inverted the verdict on a successful run. Measured on a nested pytest over
five tests with two deselected by `-k` and one module skipped during collection:
the report read `truncated`, one item never ran, and raised the "The test run
was truncated" warning annotation, where the truth is `complete` and nothing was
skipped by an abort. On a genuinely truncated run the same fallback overstates
`items that never ran` by exactly the deselected count, which is the direction
that costs a reader a re-run. Neither half is hypothetical: `pyproject.toml`
documents `pytest -m 'not slow'` and `-m 'not integration'` in the markers' own
help text, and 496 test modules open with a module-level `pytest.importorskip`,
so any environment missing one optional extra reports a collect-time skip beside
the deselection. The required check's own command deselects nothing, so no CI
run has misreported.

The tallies are now read as a set keyed on the label, so `selected` is found
whatever precedes it, and a tally a future pytest inserts cannot reintroduce the
silent fallback. Where the line states no `selected` at all the count is derived
as `collected - deselected` rather than as `collected`, which is the
post-deselection number `tests/session_truncation.py` reports from inside the
session -- so the two surfaces that state a run's extent can no longer state two
different ones for one run (#3169).

Pinned by `tests/test_truncated_test_run_is_reported.py`: the three collection
lines a live pytest wrote, all 24 orderings of the four tallies, an oracle cell
that re-derives the tally set from pytest's own source so the parametrisation
cannot go stale unnoticed, and one nested pytest run that reproduces the false
verdict end to end. 20 of the added cells fail against the previous parser.
