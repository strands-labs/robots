### Docs: the one CodeQL alert class no documented disposition clears, and the question that settles it

`AGENTS.md` > Review Learnings (PR #92) > CI Security Baseline offers three
tools for clearing an alert - fix it, dismiss it with a reason when the
construct is *deliberate and test-only*, or filter the rule when *every*
instance in the tree is an obliged idiom - and warns that rewriting the flagged
code to satisfy the query is the tempting fourth option and the one that costs.
`py/catch-base-exception` falls through all three, and the gap is not abstract:
#1899 sat on two `github-advanced-security` threads whose own text is "please
resolve this thread if you agree - or say the word and I'll narrow it", each
carrying a paragraph of defence for an idiom the tree already ships.

The rule's entire alert surface here is a single construct. The query accepts a
handler that re-raises *lexically*, and six of the tree's seven
`except BaseException` handlers do - four production cleanup-and-reraise blocks
ending in a bare `raise`, two test helpers ending in
`raise AssertionError(...) from exc`. None has ever been flagged. The seventh,
`IsaacSimulation.run_on_main`'s cross-thread exception box, stores the exception
for another thread to re-raise, and it is the only one CodeQL reports.

Narrowing that box to `Exception` is not the safe default it resembles, and the
worst case is silent. What the *caller* thread observes: a `RuntimeError`
arrives either way; a `KeyboardInterrupt` arrives as `None` plus
unhandled-exception noise on stderr; a `SystemExit` arrives as `None` with no
traceback at all, because `threading` discards it in a non-main thread. So
narrowing does not relocate the exception, it deletes it, and the caller
re-raises nothing - the no-silent-defaults rule, reached from an exception
clause.

What the three dispositions were missing is a direction. The box is obliged when
marshalling *onto an existing foreign thread* - `run_on_main` handing a job to
the thread that owns the Kit pump - because `concurrent.futures` cannot target
an already-running foreign thread; there, dismiss with a reason and resolve the
thread pointing at it. It is avoidable when marshalling *off a new thread you
create*, because `concurrent.futures` is exactly that pattern and its
`except BaseException` belongs to CPython (`_WorkItem.run`) rather than to this
tree. `Future.result()` re-raises `RuntimeError`, `SystemExit` and
`KeyboardInterrupt` with object identity preserved, so delegating is strictly
better than the box rather than merely quieter, and it removes the handler, the
alert and the blocking review thread together.

The filter stays out of reach deliberately: its test is that every instance is
obliged, and the second case is a standing counter-example, so this rule id must
keep failing the two-id set `tests/test_codeql_query_filters.py` pins.

The class is worth naming because both answers are live at once and nothing else
recorded why they differ. Alert #691 - `run_on_main`'s box at
`strands_robots/simulation/isaac/simulation.py:5125` - has been open on
`refs/heads/main` since 2026-07-07 at note severity, gating nothing, carrying
only a `# noqa: BLE001` that CodeQL does not read. Alerts #853 and #854 are the
same idiom raised on a branch, one of them in that same file, and each opened a
review thread, so under `required_review_thread_resolution` they gate the merge.
Identical construct, opposite consequence, separated only by having arrived on a
branch.

`AGENTS.md` now records the census, the narrowing measurement and the
two-direction rule beside the three dispositions it completes, and
`tests/test_codeql_query_filters.py` pins it in the module that exists because
#1810 corrected `codeql.yml` and left the identical false sentence unpinned in
`AGENTS.md`, where every contributor reads it first.

Documentation and test only; no production code changes.
