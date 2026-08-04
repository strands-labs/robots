### Docs: a review thread you already answered is not an open review comment

`AGENTS.md` > PR Workflow > step 5 said "address all review comments". That is
complete for a human, who remembers replying, and incomplete for an agent that
rebuilds its context from the GitHub payload each scheduled run: the reviewer's
question is still sitting there verbatim, and the agent's own prior reply is
indistinguishable from context it has not yet acted on. Nothing in the payload
says "answered", so the instruction is satisfied by replying again - and again.

On #1899 one review thread took **12 consecutive replies from the pull request's
own author**, between 21:46 and 01:30, every one of them announcing the same
commit (`35ee25d2`). The thread was resolved at 21:52 by the second of them and
marked outdated; the branch's last push was 22:37. So from the third reply on,
each described work that was already complete and already announced twice, and
they were still arriving hourly after the pull request was green and waiting on
nothing but a reviewer.

The loop is self-feeding rather than self-limiting, which is why prose alone had
not stopped it: replying makes the thread the most recently active thing on the
pull request, so it is the first thing the next cycle reads, and its salience
rises with every restatement.

Step 5 now gates on authorship and thread state instead of on whether a question
is present. If a thread's last non-bot comment is your own, no reply is owed -
push the code if there is code to push, because the push is the message. If the
thread is `isResolved` or `isOutdated`, resolution is terminal and reopening it
to restate a landed fix reads as noise rather than diligence. A reply is owed
only when the last comment is someone else's and the existing replies do not
answer it, and then exactly once.

The authorship check is the cheap half - no semantic comparison, just the author
of the last comment - and on its own it would have prevented ten of the twelve.
Both `isResolved` and the comment authors were already in the context payload on
#1899; they were fetched and never read.

Pinned by `tests/test_review_thread_reply_gate.py`, which asserts adjacency
rather than vocabulary: a future edit tightening the step back to a bare "address
all review comments" is exactly the regression, looks like a simplification, and
nothing else in the tree would notice. Same shape as #1919 - the policy was not
wrong, it was silent on a case that recurs every scheduled cycle.
