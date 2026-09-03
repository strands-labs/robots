### Changed: a truncated test session states how much of the suite never ran

CI runs the suite under `-x`, so a red required check stops at the first failure
with the rest of the suite unexecuted. The counts line pytest then prints is
shaped exactly like a complete run's -- `1 failed, 34268 passed, 278 skipped`
reads as a total whether 34547 tests ran or 46583 did. pytest names the abort
(`!!! stopping after 1 failures !!!`) but does not size it, so how much of the
suite never started was recoverable only from the trailing progress token of the
last verbose line, one token at the end of a multi-megabyte log.

`tests/session_truncation.py` registers a terminal-summary section that states
the size:

```
============= session truncated: 1926 of 4878 collected tests ran ==============
2952 collected tests never started, so the counts below are a floor, not a total.
```

It reads the item count collection settled on (after deselection) and counts the
tests that were entered, so it is independent of why a session stopped -- a
`--maxfail` budget, an interrupt or a test that takes the process down all
truncate the report the same way. It is silent when the session ran everything it
collected, so the section appears only where it changes what the counts below it
mean, and it changes no flag, bound or workflow file.
