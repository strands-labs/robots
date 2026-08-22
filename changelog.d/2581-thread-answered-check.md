### Added: a command that says which review threads are still owed an author reply

`scripts/check_thread_is_answered.py` reports, per review thread on a pull request,
whether anything is owed: `settled` (a reviewer resolved it), `answered` (the last
non-bot comment is the author's, so the next move is the reviewer's), or
`awaiting-the-author` - the only outcome that is work, and the only one that exits
`1`. `--all-open` sweeps every open non-draft pull request, answering "which of
these actually need me" in one read.

One rule decides it: whoever spoke last owes the next move, unless a reviewer
resolved the thread. A thread is append-only, so a reviewer's question stays in it
verbatim after it is answered and its presence is not evidence that anything is
outstanding - #2511 collected four author replies to one question in 27 minutes and
#2577 collected two, each announcing a commit the previous reply had already named.

`isOutdated` is reported but not consulted, because it describes the diff rather
than the conversation and is unreliable in both directions: it stays `false` on an
answered thread whose fix added lines elsewhere (#2480 after `e83cf51`), and a
thread goes on accepting comments after it flips to `true` (#2577 took two more),
so reading it as terminal files a reviewer's new demand on a moved line as settled.
The AGENTS.md rule that called it terminal is corrected to match.

Whether the pull request's head has moved past the commit a thread was written
against is reported beside each thread but decides nothing - an author reply that
explains rather than changes is a complete answer. It is reported because
`originalCommit` belongs to the thread and not to the comment (identical on all
four comments of #2511, across two pushes), so `headRefOid` is the only commit a
thread can be keyed to when telling "already fixed at `<oid>`" from "not yet
fixed".

Reports and does not gate, and is wired to no workflow: what an author should do
next is not something a branch can turn green.
