### Fixed: a cancelled required check is not reported as a failing one owed by the author

`scripts/check_merge_blockers.py` read a `cancelled` conclusion on the required
context as `required-check-failing`, whose documented owner is *the author*, and
the `--all-open` sweep then printed `::warning title=Blocked on something no
reviewer can clear` and exited `1`. A cancellation is a statement about the
scheduler, not about the diff, so there was nothing for the author to fix and the
one signal a scheduled pass acts on pointed at work nobody owed.

Two defects composed, both in the direction that reads as diligence rather than
as a mistake. The classification fell through to the generic "not success,
neutral or skipped" branch. And in `resolve_check_conclusions` a context
appearing twice kept its worst answer with `(None, "success")` exempted -- where
`None` means *still running*, so a **superseded** run's terminal `cancelled`
overwrote the live run that had replaced it, turning `pending` into a terminal
verdict. That half also ran the other way: read as `[cancelled, failure]` the
resolved answer was `cancelled`, so a genuinely failing required check was
recorded as a cancellation, and only the shared outcome name hid it.

Measured on #3014, head `ecb07a41`, which carried the required context twice --
`cancelled` at 13:15:35Z and a run still in progress at 13:45:32Z, with all ten
sibling checks `success`. It read `required-check-failing, missing-approval` owed
by *the author*; its sibling #3015, the same change shape with no cancelled
predecessor, correctly read `required-check-pending`. Both now read alike.

A cancellation is now ranked below every verdict rather than above them: it
neither displaces a recorded answer nor survives one, so the pair resolves to the
live run in either page order -- which matters because #1914 measured two reads
of one unchanged sha disagreeing ten minutes apart, and the order this page
arrives in is the API's business. It gets its own `required-check-cancelled`
outcome owed by **a maintainer**, by re-running the run. It stays a blocker,
because the head genuinely carries no verdict for a required context, but it is
not a *finding*: a finding is defined here as what an author-side pass can clear
alone. Its detail warns off the push, which re-triggers the check but costs the
approval twice under `dismiss_stale_reviews_on_push` plus
`require_last_push_approval`.

`timed_out` deliberately stays with the author. That job ran and failed to finish
inside its own deadline, which *is* a statement about the tree, and folding it in
would send a real failure to a maintainer as a re-run.

#1800 established the roll-up reading and #1914 measured this pair on a pull
request head, but #1914 closed explicitly not claiming how such a pair is read,
and this check was written afterwards. #1915 removed the producer that was
avoidable; the one that remains is deliberate, because reopening a pull request
is the documented remedy both for a head carrying no check suite (#1987) and for
a stale `headRefOid` (#2508), and a reopen necessarily cancels the run in flight.
So the state is reachable by following AGENTS.md and can only be read correctly.
