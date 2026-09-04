### Fixed: the last-push rule names the "Update branch" button, not just a `git push`

`require_last_push_approval` keys on who the head commit is attributed to.
Every spelling this repository's guidance named was a `git push`, so the
**"Update branch" button** on the pull request page went unnamed -- and it is
the same act: one click, no local checkout, no token the operator handles, and
the same new last pusher.

Measured on #2907, a branch authored by `logesh4v`. Its head `1c66c8f3` is a
base refresh the button produced, and it carries a third commit-metadata shape
the guidance did not list:

| head | git author / committer | metadata reads as |
|---|---|---|
| #1722 `d938686` | both `strands-robots`, distinct from the approver | rule satisfied |
| #1035 `8d6a4c42` | both the maintainer outright | rule unsatisfied |
| #2907 `1c66c8f3` | author the maintainer, committer **`web-flow`** | nobody pushed this at all |

The third is the most reassuring of the three and so the least likely to prompt
a check, because a commit whose committer is a GitHub service account reads as
GitHub having merged rather than a person having pushed. The clicker survives
only in the git *author* field and in `triggering_actor` -- both `cagataycali`
across all twelve workflow runs on that head. The sole approval therefore
stopped counting, twice, four days apart.

The classifier was right every time; what was silent was the remedy printed
beside the finding. It read "pushing a fix onto a contributor's branch consumes
the approval of whoever owns the token it is pushed with" -- neither what the
operator did nor how they did it, so the warning read as being about somebody
else. `check_last_push_approval.py` and `check_pr_head_is_current.py` now name
both spellings, and the second already quoted `Head branch is out of date`, the
button's own label, while telling the reader not to *push*. `AGENTS.md` gains
the button beside `git push`, the `web-flow` row in the commit-metadata table
that existed to stop this inference, and the standing consequence: a base
refresh on a contributor's branch is theirs to make, and the button is not an
exception because it is one click.

Neither script changes an executable statement -- the guidance lives in
existing string tuples. The new pin drives the real renderers and derives the
population of that guidance from the tree, keyed on *stating the consequence*
rather than on the constant's name, so a further check that copies it joins the
requirement with no edit. `check_checkout_is_pr_head.py` defines the same
constant and its remedy is about fetching the right commit, so it is exempt and
is asserted to be.
