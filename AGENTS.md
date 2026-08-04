# AGENTS.md - strands-labs/robots

## Overview

`strands-robots` is a robot control library for [Strands Agents](https://strandsagents.com). It provides policy inference, teleoperation, calibration, and simulation tools for physical robots.

## Project Dashboard

**Board**: https://github.com/orgs/strands-labs/projects/2
**Project ID**: `PVT_kwDOD151Fs4BSRJP`

> **RULE**: ALWAYS use the project board to track work. When creating follow-up items,
> create GitHub issues and add them to this board with Status + Priority set.
> Never track work only in local markdown - the board is the source of truth.

## Repository Structure

```
strands_robots/
├── policies/              # Policy providers (pluggable via registry)
│   ├── base.py            # Abstract Policy base class
│   ├── factory.py         # create_policy() factory + registry
│   ├── mock.py            # MockPolicy for testing
│   ├── groot/             # NVIDIA GR00T N1.5/N1.6/N1.7 inference
│   │   ├── policy.py      # Gr00tPolicy (ZMQ + HTTP modes)
│   │   ├── client.py      # Gr00tInferenceClient
│   │   ├── data_config.py # Gr00tDataConfig + ModalityConfig
│   │   └── data_configs.json  # 25 robot embodiment configs
│   └── lerobot_local/     # HuggingFace LeRobot direct inference
│       ├── policy.py      # LerobotLocalPolicy (RTC support)
│       ├── processor.py   # ProcessorBridge (pre/post pipelines)
│       └── resolution.py  # Policy class resolution (v0.4/v0.5)
├── registry/              # JSON registry for policy discovery
├── tools/                 # Strands @tool functions
│   ├── gr00t_inference.py # GR00T inference tool
│   ├── lerobot_calibrate.py
│   ├── lerobot_camera.py
│   ├── lerobot_teleoperate.py
│   ├── pose_tool.py
│   └── serial_tool.py
├── robot.py               # Core Robot class
└── utils.py               # Shared utilities (require_optional, etc.)

tests/                     # Unit tests (run with: hatch run test)
tests_integ/               # Integration tests (run with: hatch run test-integ)
```

## Development

```bash
# Install with all optional deps
pip install -e ".[all,dev]"

# Run tests
hatch run test              # unit tests
hatch run test-integ        # integration tests (needs GPU + model weights)

# Lint & format
hatch run lint              # ruff check, ruff format --check, mypy
hatch run format            # ruff check --fix, ruff format
```

> **Note**: Hatch uses `uv` as installer (`installer = "uv"` in pyproject.toml) for faster
> environment creation. No manual uv install needed - hatch handles it.

## Key Conventions

1. **Python 3.12+** - `requires-python = ">=3.12"` (LeRobot >=0.5.0 requires 3.12)
2. **Dependency bounds** - `>=1.0` deps: cap major. `<1.0` deps: cap minor. E.g. `lerobot>=0.5.0,<0.6.0`
3. **`__init__.py` must be thin** - exports only, no logic
4. **Imports at file top** - unless lazy-loading heavy deps with documented reason
5. **Raise on fatal errors** - never warn-and-continue if the system will behave unexpectedly
6. **No silent defaults on error** - returning zero-valued actions on failure is forbidden
7. **Use `require_optional()`** - from `strands_robots/utils.py` for all optional deps
8. **Integration tests required** - each policy needs `tests_integ/` tests with real inference
9. **Test behavior, not implementation** - assert on outputs, not internal state
10. **No dead code** - if it's not called and not part of base class, delete it

## PR Workflow

1. Create feature branch from `main`
2. Make changes, run `hatch run format && hatch run lint && hatch run test`
3. Record the change as a news fragment: `changelog.d/<pr-number>-<slug>.md`
   (see [`changelog.d/README.md`](changelog.d/README.md)). **Never append to
   `## [Unreleased]` in `CHANGELOG.md` directly** - every branch inserts at the
   same anchor, so two PRs open at once conflict on ordering alone, and because
   stale approvals are dismissed on push each resolution costs a re-approval
   round that reviews no changed behaviour. A fragment is its own file, so there
   is nothing to conflict on. `CHANGELOG.md` is assembled from the accumulated
   fragments when a tag is cut (`python scripts/assemble_changelog.py --apply`).
   This is enforced by `.github/workflows/changelog-fragment.yml`: the rule was
   documented in two places and enforced by nothing until #1784, and a pull
   request reached `APPROVED` / `SUCCESS` / `CLEAN` having appended to the log.
4. All tests must pass, lint must be clean
5. Open PR from your fork, address all review comments

   **Check whether you are already the thread's last author before replying.**
   "Address all review comments" is a per-*concern* obligation, not a
   per-*cycle* one, and an agent that rebuilds its context each run cannot tell
   those apart from the thread alone. On #1899 a single thread collected **12
   consecutive author replies** between 21:46 and 01:30, every one of them
   announcing the same commit (`35ee25d2`):

   | reply | posted (UTC) | thread state when posted |
   |---|---|---|
   | 1st | 21:46 | open, question unanswered |
   | 2nd | 21:52 | resolved *by this reply* |
   | 3rd - 12th | 22:28 - 01:30 | `isResolved: true`, `isOutdated: true` |

   The branch's last push was 22:37, so every reply from the 3rd on described
   work that was already complete and already announced twice - and they were
   still arriving hourly after the PR was green and waiting on nothing but a
   reviewer.

   The loop is self-feeding, which is why it does not decay on its own:
   replying makes the thread the most recently active thing on the PR, so it is
   the first thing the next cycle reads, and an agent's own prior reply is
   indistinguishable from context it has not yet acted on. The reviewer's
   question is still sitting there verbatim in the serialised thread. Nothing
   in the payload says "answered".

   So gate on authorship and state, not on whether a question is present:

   - Thread's last non-bot comment is **yours** -> do not reply. You have
     already said it. If there is code to push, push it; the push is the
     message.
   - Thread is **`isResolved` or `isOutdated`** -> do not reply. Resolution is
     terminal. Reopening it to restate a landed fix reads as noise, not
     diligence.
   - Last comment is **someone else's** and your existing replies do not answer
     it -> reply once, then resolve.

   The authorship check is the cheap one, and it would have prevented ten of
   those twelve comments on its own: no semantic comparison, just the author of
   the last comment. Both `isResolved` and the comment authors are already in
   the context payload - they were fetched and not read.

   What makes this worth writing down is that the previous rule was *satisfied*
   by all twelve. "Address all review comments", and "reply when a thread asks
   a direct question", are both still true of a thread you have already
   answered, because answering does not remove the question. The cost lands on
   the next reader: the signal that the thread was settled at 21:52 is buried
   under ten paragraphs restating it, and a reviewer must scroll all of them to
   learn nothing changed. Same shape as #1919 - the policy was not wrong, it
   was silent on a case that recurs every scheduled cycle.
6. Track follow-up items as issues on the [project board](https://github.com/orgs/strands-labs/projects/2)

   **Read the board with `PAT_TOKEN`, not the Actions `GITHUB_TOKEN`.** An
   installation token that cannot see an organization project does not fail the
   query - `issue.projectItems` comes back as an empty list, which is
   indistinguishable from an issue nobody has tracked. The same one query, run
   twice on #1762, #1768 and #1770:

   | token | `issue.projectItems` |
   |---|---|
   | `GITHUB_TOKEN` | `[]` for all three |
   | `PAT_TOKEN` | one item each; #1768 and #1770 already at `Done` |

   Two things follow. First, adding an issue to the board is almost never the
   missing step: all three items were created by `github-project-automation`
   within three seconds of the issue itself, and the automation also moves them
   to `Done` on close, so a merged fix needs no manual flip. What it cannot
   infer is a status like `In review` while a PR is open - that is the part
   worth setting by hand. Second, acting on the empty read looks harmless and
   is not: `addProjectV2ItemById` is idempotent and returns the *existing*
   item's id, so no duplicate ever appears and the false-empty read leaves no
   trace - until a `Status` written on the strength of it silently overwrites a
   value that was never read. Read a field before you set it, and treat an
   empty project read as unknown rather than as absent.
7. Squash merge into `main`
8. **Verify a PR's state by reading it back - before and after you change it.**
   Neither direction can be inferred:
   - *Before.* Query `timelineItems(itemTypes: [CLOSED_EVENT, REOPENED_EVENT])`
     before closing or reopening. A lone `CLOSED_EVENT` is safe to re-apply; an
     alternating run means something is undoing you, and a further flip only
     lengthens it. #1667 - the retired `pr/consolidated-local-work` staging PR -
     was closed and reopened **ten times in under fourteen hours**, each reopen 2-22
     minutes after the preceding close, because independent contributors read
     the same PR and reached opposite conclusions. Its extraction plan and
     terminal state live in #1723; do not flip #1667 again.
   - *After.* A `mergePullRequest` mutation can report `Pull Request is not
     mergeable` on a merge that in fact landed - observed on #1756, where the
     mutation returned that error and the squash was already on `main`. Confirm
     with `state`/`merged`, or `git log origin/main`, before concluding a merge
     failed and redoing the work.
   - *And on `main` afterwards.* A rollup of `FAILURE` on a merge commit is not
     evidence that the squash broke anything: a **cancelled** check aggregates
     into `FAILURE`. `pr-and-push.yml` keys its concurrency group on
     `github.event.pull_request.number || github.ref`, and on a push there is no
     PR number, so every push to `refs/heads/main` shares one group under
     `cancel-in-progress: true` - each merge kills the run of the merge before
     it. Read each context's own `conclusion` before you believe the rollup.
     Four PRs merged in the 22 minutes from 03:03:44 to 03:25:25 left three
     consecutive commits - #1788, #1794, #1796 - each reporting rollup
     `FAILURE` whose only non-`SUCCESS` context was
     `call-test-lint / Test and Lint` = `CANCELLED`, killed at 1m07s, 15m00s
     and 5m38s into their runs. Nothing had failed. See #1800.

     The same timings carry a cost that is not a misread: that suite had not
     finished in 15m00s, so merging faster than it runs leaves **only the tip
     verified** and no intermediate commit attributable. A batch is still
     defensible - each of those four was individually green, passed
     `Detect an untested overlap with the base branch`, and touched a file set
     disjoint from the others - but price it knowingly: a red tip then costs a
     manual bisect, and an intermediate commit's green is not available to lean
     on. Do not read the intermediate `FAILURE`s as the culprit; they are the
     batching, not a defect.
   - *And when `main` itself is red, a re-run cannot clear the PRs it blocked.*
     `pr-and-push.yml` checks out the PR **head commit**, not
     `refs/pull/N/merge` - the job log reads `HEAD is now at <branch head>` - so
     a branch's green is a statement about the branch's own tree, which is the
     reason `Detect an untested overlap with the base branch` has to exist as a
     separate check. It follows that a fix landing on `main` is invisible to the
     branch until the branch absorbs it, and
     `POST /actions/runs/{id}/rerun-failed-jobs` re-uses the head SHA the run
     recorded, so re-running changes nothing. Measured on #1824, which fixed the
     three `test_deferred_physics_and_warmup.py` failures open as #1823: #1827
     and #1829, both re-run after #1824 was on `main`, reported the same single
     failure (`1 failed, 6287 passed` and `1 failed, 6277 passed`). Merge `main`
     into the branch and push; there is no cheaper route.

     **That push is free when the merge is conflict-free.** The intuition that
     it costs the approval is wrong, and the difference is measurable before you
     push. Two pushes onto #1821, both merges of `main` into an approved branch:

     | push | `git show --cc` | PR diff vs merge base | approval |
     |---|---|---|---|
     | `b365d60`, resolved a conflict | 82 lines | changed | dismissed 45s later |
     | `79cbdad`, clean base merge | 0 lines | identical, `4 files, +158/-7` | survived, still `APPROVED` |

     Dismissal keys on the **PR's own diff**, not on the head SHA changing: a
     merge that only brings the base forward leaves the diff a reviewer read
     byte-identical, and nothing is dismissed. A conflict resolution is new,
     unreviewed text - a combined diff is exactly the text belonging to neither
     parent - and that is what costs a round. So refreshing an approved branch
     over a fixed `main` is cheap, and the expensive case is the one worth
     avoiding structurally, which is what step 3's fragment rule already does for
     the file every branch used to conflict on.

     Two consequences remain. Merge the fix for a red `main` ahead of the queue
     rather than alongside it: while it is red nothing on top of it can merge at
     all, whatever its own state. And do not push the refresh onto a
     **contributor's** branch - that is the `require_last_push_approval` identity
     problem below, which is independent of dismissal and does not care that the
     merge was clean. Ask the contributor to absorb `main` so they stay the last
     pusher; #1827 was left alone for that reason.
   - *And that the mutation named the object you meant.* A mutation
     names its subject by node ID and by nothing else - `createIssue` takes a
     `repositoryId`, not an owner and a name - so a well-formed ID that is wrong
     does not fail. It succeeds against whatever object it *does* name. Filing
     an issue for this repository with a `repositoryId` carried over from an
     earlier response rather than queried - `R_kgDOD1WOFw` for `R_kgDORUMiZg` -
     created issue #1 in an unrelated third-party repository and returned
     success. The only clue was the `url` in the response, and there is no undo:
     `deleteIssue` needs admin on the *target*, so the stray issue could only be
     closed as `NOT_PLANNED` with an apology. See #1916.

     **A node ID is not opaque, which is what makes this checkable before the
     write.** It is `<TypePrefix>_<urlsafe-base64(msgpack array)>`, where a
     repository is `[0, databaseId]` and anything a repository owns is
     `[0, repository databaseId, own databaseId]` - so the type and the target
     repository are both readable with no network call:

     | node ID | decodes to | target |
     |---|---|---|
     | `R_kgDORUMiZg` | `[0, 1162027622]` | this repository |
     | `R_kgDOD1WOFw` | `[0, 257265175]` | the stray one |
     | `PR_kwDOD1WOF87DdSjQ` | `[0, 257265175, 3279235280]` | **the same stray one** |

     That third row is the finding. All three guessed IDs in that run carried
     one wrong repository, so a single stale value contaminated every mutation,
     and the two that failed did so only because their own databaseId happened
     not to exist there - `Could not resolve to a node`. Failing closed was luck
     about the guess, not a property of the API, and the guess that got lucky the
     other way is the one that wrote.

     So resolve every ID from a query in the same run whose owner and name are
     written out literally; check the prefix against the parameter, since a
     `PR_...` handed to a `repositoryId` is wrong by type alone; and read the
     `url` in the response back before treating the write as done.
     `tests/test_graphql_node_id_targeting.py` decodes this repository's own node
     IDs against the `databaseId`s the API publishes beside them, so the claim
     that the check is available offline fails loudly rather than quietly if the
     envelope ever changes.
   And before merging, `reviewDecision: APPROVED` alone is not the gate: poll
   the **required** contexts' own conclusions and `mergeStateStatus == CLEAN`
   together, since `reviewDecision` flips before the checks finish.

   Read the required set rather than the rollup, because they are not the same
   question and the rollup is the stricter one. `statusCheckRollup.state ==
   SUCCESS` is *not* a merge requirement: the `default` ruleset lists exactly one
   required check,

   ```
   GET /repos/{owner}/{repo}/rulesets/{id}  ->  required_status_checks
                                                = ["call-test-lint / Test and Lint"]
   ```

   so every other context - `CodeQL`, `dependency-review`, `Detect Breaking
   Changes` - is advisory, and any one of them non-`SUCCESS` drags the rollup to
   `FAILURE` or `NEUTRAL` while the PR remains perfectly mergeable. #1879, #1880
   and #1881 were each merged at rollup `FAILURE`/`NEUTRAL` with
   `mergeStateStatus` `CLEAN`. `mergeStateStatus` is the field that already
   accounts for the required set, which is why it is the one to trust:
   `BLOCKED` while the required check runs, then `UNSTABLE` - mergeable, with an
   advisory context red - or `CLEAN`.

   That trust has a reader attached to it, which the sentence above does not say.
   `mergeStateStatus` answers *can the viewer merge this pull request*, so it is
   scoped to the token that asks - and on a pull request editing
   `.github/workflows/**` the Actions `GITHUB_TOKEN` can never read anything but
   `BLOCKED`, because an installation token is refused writes to workflow files
   and therefore genuinely cannot perform that merge. **Read the gate with
   `PAT_TOKEN`.** A control pair, both approved with no unresolved thread and
   `call-test-lint` `SUCCESS`, read minutes apart:

   | PR | edits `.github/workflows/**` | `GITHUB_TOKEN` | `PAT_TOKEN` | truth |
   |---|---|---|---|---|
   | #1915 | yes (`pr-and-push.yml`) | `BLOCKED` | `CLEAN` | merged clean, `f4dfde6` |
   | #1902 | no | `CLEAN` | `CLEAN` | merged clean, `6cf0470` |

   The mechanism isolates to one variable - same token, same scratch branch, same
   instant, via `PUT /repos/{owner}/{repo}/contents/{path}`:

   ```
   zz_probe.txt                    GITHUB_TOKEN -> created
   .github/workflows/zz_probe.yml  GITHUB_TOKEN -> Resource not accessible by integration
   .github/workflows/zz_probe.yml  PAT_TOKEN    -> created
   ```

   So a `BLOCKED` read that way is neither a bug nor staleness; it is the honest
   answer to the question the field asks. Staleness is separately ruled out:
   `mergeable_state` on #1899 and #1035 reads `unknown` first and the settled
   value second **for both tokens identically**, so the lazy first read is
   per-PR, not per-viewer. `mergeable` agrees across tokens throughout - a text
   conflict is viewer-independent, and only mergeability-*by-you* is not.

   What makes this expensive is that the wrong answer is indistinguishable from a
   right one. On a genuinely blocked PR both tokens read `blocked`, so their
   agreement proves nothing, and the Actions token's answer on a
   workflow-touching PR is always blocked-or-unknown. No reading of the field
   separates the two cases. The agent then polls the gate exactly as documented,
   correctly declines to merge, and reports the PR as waiting on a reviewer -
   which is the presentation #1905 records for a different cause, and which had
   stood in eight consecutive scheduled scan summaries as "reviewer bandwidth is
   the sole constraint". It bites CI and process pull requests specifically,
   because those are the ones carrying workflow edits. See #1917.

   This is worth the words because the failure mode is silent and expensive in the
   opposite direction from the usual one. Treating an advisory red as a merge
   blocker does not look like a mistake; it looks like diligence, and it costs a
   round of real changes to a PR that was ready. #1879 spent a round removing a
   `__float__` from a test fixture to clear a `CodeQL` finding that never gated
   anything. Worse, the finding was not even attributable to that PR: alert #846
   (`py/non-iterable-in-for-loop`) had been open on `main` since the day before and
   was reported as "new in code changed by this pull request" only because the
   branch added lines above it and CodeQL's baseline matching is positional. When
   an advisory finding appears, check its `created_at` and whether it is open on
   `refs/heads/main` before assuming the branch introduced it:

   ```
   GET /repos/{owner}/{repo}/code-scanning/alerts?ref=refs/heads/main&state=open
   ```

   That endpoint needs `PAT_TOKEN` - the Actions `GITHUB_TOKEN` gets `403 Resource
   not accessible by integration` - and the annotation on the failing check run
   (`GET /check-runs/{id}/annotations`) reports the line as it falls in the
   *branch*, so the number will not match `main`. A genuine pre-existing alert is
   still worth fixing, but as its own tracked change on `main` rather than as an
   unplanned round on whatever PR happened to shift its line: #846 was closed that
   way by #1881, which was also the fix for #1878.

   Those three together are still not sufficient. They are all evaluated against
   the base the branch was tested on, so none of them can see a **semantic**
   conflict with a PR that landed on `main` after those checks ran. #1766 and
   #1763 both edited `_recompile_preserving_state` for unrelated reasons: the
   text merged with no conflict, #1763 stayed `MERGEABLE`/`CLEAN` with
   `SUCCESS` checks after #1766 landed, and the squash still broke `main`,
   because #1763 carried a *premise* test asserting the very defect #1766 had
   just fixed. Neither PR's CI ever compiled the two together.

   So when a second approved PR touches a file - especially a function - that a
   just-merged PR also touched, do not merge on the green alone. Merge `main`
   into the branch (or check out the merge locally) and run the affected tests
   before issuing the mutation. A `CLEAN` status is a statement about text, not
   about meaning. This is cheap: the check that would have caught the above was
   one `pytest` invocation on two files.

   Read that run as a **delta, not an absolute**. The environment you verify in
   is almost never the one CI uses, and a partial one fails tests for reasons
   that have nothing to do with the merge. Composing #1786 and #1804 - both
   approved, both `CLEAN`, both editing `simulation/predicates.py`, neither ever
   compiled with the other - the affected suite reported **376 failed, 34
   errors** on the composition, which on its own reads as a broken merge and a
   reason to stop. The same command on the unmerged base reported the same
   **376 failed, 34 errors**, and 4185 passed against the composition's 4229:
   every failure pre-existed (a hosted runner has no GPU and only software
   OSMesa, so the rendering tests fail there whatever the diff), and the whole
   effect of the merge was **+44 passes** - exactly the 24 + 20 tests the two
   branches add. The reading matters in both directions: an absolute count can
   invent a regression and cost a good merge, and it can equally hide a real one
   inside the noise. Run the same command on the base *before* you read the
   number, and compare the two.

   Then confirm the tree you verified is the tree that landed -
   `git diff --name-only <local-composition> origin/main -- strands_robots/ tests/`
   should be empty. Squash rewrites the commits, so nothing but that equivalence
   ties your local run to `main`; and on a batch, where only the tip's
   `call-test-lint` survives the concurrency group above, it is the sole evidence
   the intermediate commits were ever compiled together.

   Fixing forward beats reverting here - the two production changes were both
   correct, and only an assertion and its justification were stale. Prefer a
   narrow follow-up that re-pins the invalidated premise over reverting a
   reviewed change. If a premise test is invalidated by a fix landing, replace it
   rather than deleting it: the conclusion it supported usually still holds for a
   different reason, and that reason is what the next reader needs.

   The converse happens too: every signal above satisfied, and the PR still
   refuses to merge with no field naming the reason. #1722 carried one current
   `APPROVED` review that post-dated its head commit, all four review threads
   resolved, `call-test-lint` `SUCCESS` - and `reviewDecision`
   `REVIEW_REQUIRED`. The `default` branch ruleset sets
   `require_last_push_approval: true`, so the most recent push must be approved
   by **someone other than whoever pushed it**, and the agent had pushed that
   head commit with `PAT_TOKEN`. GitHub attributes a push to the token's
   *owner*, which was the same account that then approved. No number of further
   approvals from that account can clear it.

   What makes this worth writing down is that the commit metadata asserts the
   opposite. `d938686`'s author *and* committer are `strands-robots`, an
   identity distinct from the approver, so reading the commit list says the rule
   is satisfied. The pusher is in none of the fields you would check:
   `reviewDecision` is `REVIEW_REQUIRED` and `mergeStateStatus` is `BLOCKED`,
   which is also exactly what a PR with no approval at all looks like. The one
   place it is legible:

   ```
   GET /repos/{owner}/{repo}/actions/runs?head_sha=<head>  ->  triggering_actor
   ```

   #1035 is the control - same author, same fork, same `strands_robots/mesh/`
   files, one approval from the same account post-dating its head commit,
   threads clear, checks green, and no CODEOWNERS file in the tree to make
   `require_code_owner_review` bite. It differed in exactly one input, and read
   `APPROVED` - until a later push moved that one input and took it into the
   blocked row as well:

   | PR | commit author | `triggering_actor` | approver | `reviewDecision` |
   |---|---|---|---|---|
   | #1035 at `2be59dad` | the contributor | the contributor | the maintainer | `APPROVED` |
   | #1035 at `8d6a4c42` | the maintainer | the maintainer | the maintainer | `REVIEW_REQUIRED` |
   | #1722 | `strands-robots` | the maintainer | the maintainer | `REVIEW_REQUIRED` |

   So **pushing a fix to a contributor's branch consumes the approval of
   whoever owns the token you push with**, turning a PR one maintainer could
   merge into one that needs a second. It compounds with
   `dismiss_stale_reviews_on_push`, which drops the existing approval in the
   same motion that disqualifies that account from re-supplying it. Prefer
   leaving the change for the contributor to push, so they stay the last
   pusher; when the agent must push, that PR now requires a second approver,
   and saying so is the difference between a one-line request and a branch that
   never merges.

   **#1035 later crossed into the second row, which makes it the whole rule
   observed twice on one pull request.** CI triage on it correctly diagnosed a
   stale merge base and prescribed `git merge upstream/main`; the refresh was then
   *executed with the maintainer's token* rather than requested from the
   contributor, so `8d6a4c42` - `Merge branch 'main' into
   feat/ackermann-ros-robot`, authored and committed by the maintainer - became
   the head. The prescription was right; the hand that applied it was not.

   That case also pulls apart the two costs a push can carry, which the
   conflict-free result above makes easy to read as one. The merge was clean:
   `git show --cc` is **0 lines**, and the PR's own diff is unchanged at
   `7 files, +900/-26`. So by the #1821 table it cost no dismissal, and the
   pre-existing review is indeed **still `APPROVED`**, not `DISMISSED`.
   `reviewDecision` is nonetheless `REVIEW_REQUIRED`, and a second approval from
   that same account - on that exact head, every check `SUCCESS` - does not move
   it. Two rules, keyed on two different things:

   - `dismiss_stale_reviews_on_push` keys on the **PR's own diff**: a clean base
     merge is free.
   - `require_last_push_approval` keys on the **pusher's identity**: a clean base
     merge is not free, and re-approving from that identity cannot help.

   So "refreshing an approved branch over a fixed `main` is cheap" is scoped to
   dismissal alone. On your own branch it is cheap outright; on a contributor's
   branch it converts a pull request one maintainer could merge into one that
   needs a second, with nothing in the PR's own fields saying so. #1035 is in that
   state now, and needs an approver who is not the account that pushed
   `8d6a4c42`.

   Do not try to settle this from the commit metadata, which misleads in both
   directions. #1722's author and committer are `strands-robots`, an identity
   distinct from the approver, which reads as the rule being satisfied when it is
   not; #1035's head names the maintainer outright. Same `REVIEW_REQUIRED`,
   opposite metadata. Only `triggering_actor` is load-bearing.

   All of the above was documented here and enforced by nothing, which is the
   same shape as the changelog rule in step 3 before #1784. It is now surfaced by
   `.github/workflows/last-push-approval.yml`, which names the pusher and the
   approvers on every review event and fails when they are the same single
   account. The point of automating it is not that the check is clever - it is
   that the state it reports is *invisible*: `REVIEW_REQUIRED` / `BLOCKED` is
   byte for byte what an unreviewed pull request looks like, so the two are
   indistinguishable in every field a sweep reads and they need opposite actions.
   Verified against six pull requests, three outcomes, no false positive:

   | pull request | pushed by | approved by | outcome |
   |---|---|---|---|
   | #1722, #1035 | the maintainer | the maintainer | `pusher-only-approval` |
   | #1894, #1920 | either | a second account | `satisfied` |
   | #1899, #1901 | the author | nobody yet | `awaiting-first-review` |

   `awaiting-first-review` is a pass on purpose. It is the ordinary state of an
   open pull request and is already visible, so making it red would put a red X
   on every branch in the repository and the finding would stop meaning
   anything. Unlike the overlap check in step 8 this one is **not** self-clearing
   - its remedy is a second human, and no work the author does turns it green -
   so it reports and is deliberately absent from the required set. A gate a
   branch cannot clear by doing anything is a report, whatever it is wired to.

   The general rule behind all three: **a decision recorded only in a PR or
   issue comment is not durable** - the next contributor will not read the same
   comment. If a decision must survive, it belongs in this file.


## Registry conventions (strands_robots/registry/robots.json)

- **Flat asset paths** (e.g. `"model_xml": "scene.xml"`) are the common case.
- **Nested asset paths** (e.g. `"model_xml": "xmls/asimov.xml"`) are allowed when
  the upstream source repo uses a subdir layout. Example: `asimov_v0` maps to
  `asimovinc/asimov-v0` which has `sim-model/xmls/asimov.xml` +
  `sim-model/assets/`. The `_safe_join` helper in `strands_robots/utils.py`
  guards against traversal (`..`).
- **Auto-download strategy** - every robot with an `asset` block must declare
  exactly one of:
    1. `asset.robot_descriptions_module` (preferred)
    2. `asset.source` with `type: "github"`
    3. `asset.auto_download: false` (explicit opt-out)
  Enforced by `tests/test_registry_integrity.py`.


## Review Learnings (PR #85 - MuJoCo Backend)

Corrections from code review that apply to all future contributions:

### Thread Safety
- **Lock ALL model/data mutations** - MuJoCo `model`/`data` are not thread-safe. Any method that writes `qpos`, `qvel`, `ctrl`, `qfrc_applied`, `body_mass`, `geom_friction`, or calls `mj_step`/`mj_forward`/`mj_resetData` MUST hold `self._lock`.
- **Guard scene mutations during policy** - Use `_require_no_running_policy()` before any action that recompiles or replaces the model/data objects.
- **Document the concurrency contract** - If a method is safe to call concurrently, say so. If not, say so.

### Error Handling Contracts
- **Return error dicts, never raise** - All `AgentTool` action handlers must return `{"status": "error", "content": [...]}` on failure. Never raise exceptions that bypass the structured response.
- **Clean up on failure** - If you register state (e.g., add to `self._world.objects`) before an operation that can fail, pop/undo it in the except path.
- **Fail-fast with `strict=True`** - Silent frame dropping or catch-all `except Exception` with logging is forbidden unless gated behind a `strict=False` parameter.

### API Consistency
- **Don't export private functions** - `_`-prefixed names must never appear in `__all__`.
- **Match docstrings to semantics** - If the docstring says "single-shot" but the code is "latched", one of them must change. Always verify by reading the underlying library docs.
- **Forward all advertised kwargs** - If `tool_spec.json` exposes a parameter, the dispatch chain must forward it all the way through. Silent drops are bugs.
- **Centralize import checks at init** - Prefer checking optional deps once in `__init__` over scattered `_ensure_X()` guards. Consumers catch issues at init time.

### Data Integrity
- **Per-name state copy, not flat index** - When recompiling MuJoCo models (inject/eject), copy qpos/qvel per-joint by name. Flat-index slicing breaks when body-tree order shifts.
- **Sanitize user inputs into XML** - Validate names against `^[a-zA-Z0-9_-]+$` before interpolating into MJCF. LLM-provided strings are untrusted.
- **Match schema and data keys** - If a feature is declared with sanitized names (e.g., `__`), the data producer must emit the same sanitized keys.

### Testing
- **Test import paths must match production** - If `src/` imports `from lerobot.datasets.X`, tests must use the same path. Mismatched paths cause silent skips via `except ImportError`.
- **Round-trip tests for recording** - Any recording feature needs: start -> write -> stop -> reopen -> assert non-empty. Schema-only tests miss silent data loss.
- **Pin regression tests for reviewed fixes** - Every review fix gets a test that fails on pre-fix code. Otherwise the next refactor silently reintroduces the bug.
- **No host paths in test files** - Never commit `/Users/<name>/` or `/home/<name>/` paths. CI test `test_no_host_paths.py` enforces this.

### Performance
- **Don't create executors in hot loops** - Reuse a single `ThreadPoolExecutor` instance instead of creating one per call at 50Hz.
- **Cache expensive JSON parsing** - If a `@property` re-parses a JSON file on every access, cache the result at module load or first access.


## Review Learnings (PR #86 - Robot() factory)

Corrections from code review that apply to all future contributions:

### Resource Cleanup on Partial Failure
- **Always destroy on failure** - If `create_world()` succeeds but `add_robot()` fails, you MUST call `sim.destroy()` before raising. The `Simulation` object owns a `ThreadPoolExecutor`, MuJoCo world, and temp directory - leaking these is silent damage.
- **Pattern**: every `_dispatch_action(...)` call that could mutate persistent state needs `if result["status"] == "error": sim.destroy(); raise RuntimeError(...)`.
- **Don't discard return values** - If a step returns `{"status": ...}`, check it. The compiler won't catch a silently-ignored failure.

### Exception Clauses Must Be Narrow
- **`except Exception` is forbidden** for non-recovery code paths. Use the smallest superset of expected exception types.
- **`except (ImportError, Exception)` is a bug** - `Exception` is a superclass of `ImportError`, so the tuple collapses to `except Exception`. Lint/review will catch this; don't write it.
- **USB / hardware probing** - use `except (ImportError, OSError)`. `PermissionError` is an `OSError`, `FileNotFoundError` is an `OSError`, etc.

### Module-Level Side Effects
- **If you must run code at import time, comment WHY it can't be lazy.** `MUJOCO_GL` is the canonical example: MuJoCo locks the GL backend at first `import mujoco`, so the env var must be set before any downstream import chain triggers it.
- **Cheap-guard optional imports** - `if importlib.util.find_spec("mujoco") is not None:` before doing `from strands_robots.simulation.mujoco.backend import _configure_gl_backend`. Users without the `[sim-mujoco]` extra shouldn't pay an import-attempt cost on every `import strands_robots`.

### Public API Hygiene
- **Never recommend a `_method` in user-facing docstrings or error messages.** If `Robot()`'s docstring says "use `sim._dispatch_action(...)` to add a camera", you've just locked in a private dependency. Promote it (rename `_dispatch_action` → `dispatch_action`) or add public shorthands (`Simulation.add_camera()` / `.create_world()` / `.add_robot()`) before merging.
- **Type factory returns precisely** - never return `Any` from a factory. Use `@typing.overload` keyed on `Literal` mode args so IDEs resolve `Simulation` vs `HardwareRobot` at the call site. `# noqa: N802` is acceptable on factory functions named like classes (`Robot`), with a comment.
- **Reject silently-dropped kwargs** - if `Robot("so100", cameras={...})` is called in `mode="sim"` and the sim branch ignores `cameras`, raise `ValueError` instead of producing a sim with no cameras. Silent drops are bugs masquerading as features.
- **Don't conflate identity with schema** - `data_config` (e.g. `so100_dualcam`) is a separate concept from robot name (`so100`). Defaulting `data_config=robot_name` silently locks out multi-cam configs. Use an explicit `data_config: str | None = None` kwarg that defaults to canonical name only when omitted.

### Env Vars
- **Warn on unrecognized values** - `STRANDS_ROBOT_MODE=foo` (typo) must `logger.warning(...)`, not silently fall through. Silent typo'd env vars surprise users hours later.
- **Document every env var in README.md** - if you introduce a new `STRANDS_*` variable, add it to the Configuration section in the same PR. The list is the single source of truth for users.
- **Currently tracked**: `STRANDS_ROBOT_MODE`, `STRANDS_TRUST_REMOTE_CODE`, `MUJOCO_GL`.

### Safety Defaults
- **Sim-by-default** - any factory that can return either real hardware or a simulator must default to the simulator. Real hardware affects the physical world; users must opt in explicitly with `mode="real"` or `STRANDS_ROBOT_MODE=real`.
- **Reject invalid modes loudly** - `Robot("so100", mode="virtual")` must raise `ValueError`, not coerce to "sim".
- **Document parameter scope** - if `backend=` only applies to `mode="sim"`, say so in the docstring AND log a debug message when it's passed in `mode="real"` so it doesn't appear silently ignored.

### Naming & Module Organization
- **`robot.py` is for the `Robot()` factory**, the user-facing entry point. Hardware-specific code lives in `hardware_robot.py`. Don't have two files both named "robot something" with different responsibilities.
- **Reference module names, not filenames, in docstrings** - `strands_robots.hardware_robot` not `robot.py`. Filenames change; module paths are the public contract.

### Unicode & String Hygiene
- **No emojis in user-facing strings** - this is a project rule. Tool result dicts (`{"content": [{"text": ...}]}`), log messages, error messages: plain ASCII only. Agents read these strings programmatically; emojis just add tokenizer noise.
- **Hunt orphan combining marks after any emoji sweep** - `⏱️` is `U+23F1` + `U+FE0F` (variation selector). Stripping `U+23F1` leaves a stray invisible `U+FE0F` in the output. Sweep with:
  ```bash
  grep -nP '[^\x00-\x7F]' path/to/file.py
  ```
  or a Python check: `unicodedata.category(ch).startswith("So") or ord(ch) == 0xFE0F`.

### Testing Patterns
- **Use `monkeypatch.setenv`, never `os.environ[...] = ...`** - direct mutation leaks if the test raises before `finally`, and `del os.environ[...]` can `KeyError` under parallel runs. The pytest fixture handles teardown atomically.
- **Happy-path tests, not just error-paths** - if you have `test_factory_raises_on_bad_xml`, you also need `test_factory_returns_working_sim` gated behind `pytest.importorskip("mujoco")`. Steps physics, asserts state, destroys cleanly.
- **Pin every reviewed fix with a regression test** - every behavioral fix in this PR (warning on bad env var, rejecting `cameras=` in sim, default `mode="sim"`, etc.) has a dedicated test. "Trust me, the diff fixes it" is not a review-pass condition.
- **`importlib.reload` for module-state tests** - if a test modifies module-level state (env vars read at import time), reload the module inside the test and restore in teardown.

### Reviewing & Iteration
- **Resolve threads as you fix them** - leaving 14 unresolved threads on a PR with all fixes pushed makes re-review painful. Mark threads resolved when the commit lands; reviewers can re-open if not satisfied.
- **Reference commits in resolution comments** - "Fixed in `376376b`" + the suggested code block is dramatically faster to re-review than "fixed".
- **Force-push invalidates approvals** - after a rebase, prior `APPROVED` reviews drop to `DISMISSED` automatically. Mention it in the PR comment so reviewers know to re-approve, not re-review the whole diff.

## Review Learnings (PR #92 - CI Security Baseline)

Corrections from code review that apply to all future contributions:

### LLM Input Safety
- **Validate before subprocess interpolation** - every parameter on an agent-callable
  tool (`@tool` decorated function, `AgentTool.stream` dispatch handler) that flows
  into `subprocess.run`, `subprocess.Popen`, MJCF / XML interpolation, or filesystem
  path construction MUST be validated up front via regex allowlist, enum match, or
  range check. Argv-style subprocess does not exempt you - defense-in-depth.
- **Centralise validation in one function** - pattern: a `validate_inputs(...)` helper
  at the top of the tool module that takes every user-supplied param as a keyword arg
  and raises `ValueError` with a clear message on any rejection. Single entry-point
  is independently testable. PR #90's `gr00t_inference.validate_inputs()` is the
  canonical example.
- **Allowlist enumerable values** - `data_config`, `embodiment_tag`, dtype strings,
  container names: all match `^[a-z][a-z0-9_]+$` or an explicit `{"fp16", "fp8", ...}`
  set. Never accept arbitrary strings into enumerable surfaces.
- **Reject shell metacharacters in paths** - `;`, `|`, `$`, backticks, `>`, `<`,
  `\n`, `\r`, `\x00`. Also reject `..` path traversal components. Apply even when
  using argv-style subprocess.
- **Bind to `127.0.0.1` by default**, not `0.0.0.0`. Users explicitly opt into
  network exposure.

### CI Security Baseline
- **A CodeQL alert gates the merge; the CodeQL *check* does not.** These are two
  objects and only one of them is advisory. The `CodeQL` context is not in the
  required set (above), so it can sit at `NEUTRAL` indefinitely without blocking
  anything - but the alert also opens a `github-advanced-security` **review
  thread**, and the `default` ruleset sets
  `required_review_thread_resolution: true`, so the merge waits on that thread
  whatever the alert's severity. This file used to assert the opposite - that a
  finding does not block a pull request - which is the half that is false and the
  sentence #1810 was filed about; `.github/workflows/codeql.yml` carries the
  corrected wording and `tests/test_codeql_query_filters.py` pins it for both
  files. #1890 measured both halves at once: required check `SUCCESS`, `CodeQL`
  `NEUTRAL`, `APPROVED` - and it sat for 53 minutes on one unresolved
  note-severity thread, then merged 8 seconds after that thread was resolved.
- **Clearing an alert has three tools and they are not interchangeable.**
  - *Fix it.* The default for anything under `strands_robots/`, and the only
    option for a real finding.
  - *Dismiss it with a reason* when the flagged construct is deliberate and
    test-only: the Security tab, or `PATCH /code-scanning/alerts/{n}` with
    `dismissed_reason` and `dismissed_comment`. That comment is capped at **280
    characters** and the endpoint needs `PAT_TOKEN`, so the argument goes in the
    review thread and the dismissal points at it. This is the usual answer for a
    hostile fixture, and it is a repeat: alert 590 (`_HostileRobot.__getattr__`)
    and alert 852 (`GetItemOnly.__getitem__`, #1890) are the same rule,
    `py/unexpected-raise-in-special-method`, dismissed for the same reason. Then
    resolve the thread with a reply carrying the reasoning - dismissing alone
    leaves the gate closed.
  - *Filter the rule* in `.github/codeql/codeql-config.yml` only when **every**
    instance in the tree is an idiom the codebase is obliged to use. The set is
    two, `tests/test_codeql_query_filters.py` pins it, and appending a rule id is
    otherwise the cheapest way to clear any alert - which is how a filter file
    ends up quietly opting out of the whole suite.

  Rewriting the flagged code to satisfy the query is the tempting fourth option
  and the one that costs: #1879 spent a round removing a `__float__` from a test
  fixture for a finding that gated nothing. It can also destroy the measurement
  the code exists for. On #1890 the query asked for a `LookupError`; the one it
  names first, `IndexError`, is what CPython's `seqiter` *clears* to terminate
  legacy-protocol iteration, so taking the suggestion would have left the fixture
  raising nothing and the test asserting nothing, still green.
- **One alert class clears under none of the three, and the question that settles
  it is which thread you marshal onto.** `py/catch-base-exception` never fires on
  cleanup-and-reraise: the query accepts a handler that re-raises *lexically*, and
  six of the tree's seven `except BaseException` handlers do, so they have never
  been flagged.

  | handler | ends in | flagged |
  |---|---|---|
  | `robot.py:368` | `sim.destroy()`, bare `raise` | no |
  | `policies/persistent.py:193` | `handoff.abandon()`, bare `raise` | no |
  | `simulation/safe_output.py:185` | `os.unlink(tmp)`, bare `raise` | no |
  | `hardware_robot.py:1865` | `self._release_task()`, bare `raise` | no |
  | `tests/policies/lerobot_local/test_list_policy_types.py:70` | `raise AssertionError(...) from exc` | no |
  | `tests/policies/lerobot_local/test_vla_jepa.py:164` | `raise AssertionError(...) from exc` | no |
  | `simulation/isaac/simulation.py:5125` | `box["exc"] = exc`, no lexical raise | **yes** |

  The rule's entire alert surface here is therefore one construct: a
  **cross-thread exception-marshal box**, which parks the exception for *another*
  thread to re-raise, where the query cannot follow the control flow.

  Narrowing it to `Exception` is not the safe default it looks like, and the worst
  case is silent. What the *caller* thread observes:

  | raised on the worker | `except BaseException` | `except Exception` |
  |---|---|---|
  | `RuntimeError` | `RuntimeError` | `RuntimeError` |
  | `SystemExit` | `SystemExit` | **`None`, no traceback at all** |
  | `KeyboardInterrupt` | `KeyboardInterrupt` | `None`, plus unhandled-exception noise |

  Both escapes reach `threading.excepthook`, whose default ignores `SystemExit`
  specifically - so that one writes nothing at all to stderr, where an escaping
  `RuntimeError` prints a full traceback. Narrowing therefore does not relocate
  the exception, it deletes it silently, and the caller re-raises nothing. That is
  the no-silent-defaults rule, reached from an exception clause.

  So decide by direction, because the box is obliged in one and avoidable in the
  other:

  - *Marshalling onto an existing foreign thread* - `IsaacSimulation.run_on_main`
    handing a job to the thread that owns the Kit pump. `concurrent.futures`
    cannot target an already-running foreign thread, so the hand-rolled box is
    the only implementation there is. Obliged: dismiss with a reason and resolve
    the thread pointing at it, per the bullet above.
  - *Marshalling off a new thread you create* - running an agent off-main, or a
    test helper that calls into a worker. `concurrent.futures` **is** that
    pattern, and the `except BaseException` then belongs to CPython
    (`concurrent/futures/thread.py`, `_WorkItem.run`) rather than to this tree.
    `Future.result()` re-raises `RuntimeError`, `SystemExit` and
    `KeyboardInterrupt` with object *identity* preserved (`got is exc` for all
    three), so delegating is strictly better than the box rather than merely
    quieter. Not obliged: delegate, and the handler, the alert and the blocking
    review thread go at once.

  Do not reach for the filter. Its test is that *every* instance is an obliged
  idiom, and the second bullet is a standing counter-example, so this rule id
  must keep failing the two-id set `tests/test_codeql_query_filters.py` pins.

  What makes the class worth naming is that both answers are live right now and
  nothing else records why they differ. Alert #691 - `run_on_main`'s box at
  `simulation/isaac/simulation.py:5125` - has been open on `refs/heads/main` since
  2026-07-07 at note severity, gating nothing, carrying only a
  `# noqa: BLE001` that CodeQL does not read. Alerts #853 and #854 are the same
  idiom raised on a branch, one of them in that same file, and each opened a
  review thread, so under `required_review_thread_resolution` they gate the
  merge. Identical construct, opposite consequence, separated only by having
  arrived on a branch - and two rounds were spent arguing the idiom rather than
  applying the second bullet. See #1919.
- **Dependency Review hard-fails on high/critical CVEs in new deps.** If a PR
  needs a dep with a known critical CVE, the conversation is "do we need this
  dep" not "let's bypass the check."
- **The LLM-input-safety workflow is a hint, not a gate.** Inline annotations
  on `subprocess + f-string` and `name-into-XML` patterns flag code that needs
  validation review. Confirm validation is present, then ignore the annotation
  in review.

### Action Pinning
- **All `uses:` references in workflows pin to a full 40-character commit SHA**,
  with the version tag preserved as a trailing comment: `uses: actions/checkout@<sha>  # v4.2.2`.
- **Dependabot keeps these fresh** via the `github-actions` ecosystem entry.
  Do not manually bump tags; merge the Dependabot PR.
- **Especially `pypa/gh-action-pypi-publish`** - it uses a moving `release/v1`
  branch, which is exactly the supply-chain pattern that the `tj-actions/changed-files`
  incident exploited. This pin is non-negotiable.

### Operational Runbooks for Security Pins
- **A static security pin must ship with a rotation runbook, not just a recompute
  command.** A docstring one-liner that recomputes a pin is necessary but not
  sufficient; on-call at 3 AM needs a documented grace-period strategy. For the
  Amazon Root CA1 pin (`provision._AMAZON_ROOT_CA1_PINS`) the runbook lives in
  README.md > "CA Pin Rotation Runbook": dual-pin tuple during the overlap, ship
  the new pin first, drop the old pin in a follow-up release after fleet uptake,
  and use `STRANDS_MESH_CA_PINS` only as an emergency out-of-band override.
- **Make the accepted-pin set a collection, never a scalar.** `_resolve_ca_pins()`
  returns a `frozenset` so the dual-pin grace period is expressible. Any future
  pinned fingerprint (other roots, signing keys) should follow the same
  multi-value shape so rotation never requires a flag-day deploy.

## Review Learnings (PR-6 - mesh core safety hardening)

Corrections from the mesh safety/audit hardening review trail (#221/#225). They
apply to all future work on `strands_robots/mesh/{core,audit,security}.py`.

### Safety-handler discipline
- **Hoist env-var reads out of the hot path and the lock.** Safety handlers
  (`_on_safety_estop` / `_on_safety_resume`) run per-envelope; resolve lazy env
  vars (`_resume_forward_skew_s`, `_resume_freshness_window_s`) into locals at
  handler entry, before taking the cache lock, so the lock holds for the minimum
  window and a hot path never re-parses the environment per call.
- **Lockout-engagement is decoupled from the per-issuer cache cap.** A bounded
  replay cache that is full (flood, or a tiny operator override) must still let a
  legitimate peer ENGAGE a lockout - the cap bounds memory, not safety. Pin both
  directions: `*_per_issuer_cap_exceeded_still_engages_lockout` and
  `*_low_cache_max_does_not_deny_safety`.
- **Domain-tag trust-boundary cache keys.** A TLS-bound `wire_zid` and an
  app-level `issuer_id` that happen to share a string must not collide into one
  replay-cache slot. Prefix keys with a trust-domain tag (`("wire", …)` vs
  `("body", …)`) so the two namespaces can never alias.

### Audit poison-record symmetry
- **Every degraded audit path writes a poison record, never a silent drop.** The
  poison `sig` discriminators (`PSK_DEGRADED`, `SIGN_FAILED`, `SEQ_LOCK_DEGRADED`,
  `NEXT_SEQ_DEGRADED`) let a `verify_audit_integrity` walker attribute a stream
  gap to a specific failure class. When you add a new `_next_seq`/sign/persist
  failure branch, add the matching poison `sig` instead of returning early.

### Replay-cache eviction
- **TTL purge runs unconditionally, not only when the cache is full.** On a
  low-traffic mesh the cache may never reach `max_size`, but stale entries still
  accumulate. `_evict_replay_cache` always runs the O(n) TTL pass first, then the
  over-budget trim.

## Review Learnings (PR-7 - robot_mesh HITL patterns)

From the `robot_mesh` human-in-the-loop review trail (#227). Apply to the
`robot_mesh` tool and any agent-facing tool that gates on an operator interrupt.

### Operator responses are not an LLM channel
- **Never echo the operator's literal interrupt response back to the LLM.**
  Record the full response in the LOCAL audit row for forensics, but return a
  flat, fixed sentinel to the model. Echoing the operator's typed reply turns the
  human into a prompt-injection content side-channel (the agent could phrase the
  approval reason so the operator's answer leaks data into the context).

### Audit completeness
- **Audit read-only/observation actions too, not just actuation.** `peers`,
  `status`, `inbox`, and `unsubscribe` each leave a `_audit_tool_action(...)` row
  so the audit log is a complete record of agent mesh access - operators get the
  "agent read N frames from sub X at time T" trail that raw telemetry access
  otherwise lacks.

### Rate-limit safety semantics
- **A declined HITL approval must NOT consume a rate-limit slot.** The slot is
  recorded only after approval is granted (or atomically via
  `_rate_limit_check_and_record` on the post-approval path). Otherwise nuisance
  prompts an operator declines would lock the agent out of issuing a genuine
  `emergency_stop` - the inverse of the intended safety property.
