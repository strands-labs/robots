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
11. **A value-domain guard becomes shared when it has a second caller** - the guards
    in `strands_robots/utils.py` (`positive_finite_number_error` and friends) exist so
    the refusal for a rate, a count or a name is identical everywhere rather than
    merely equivalent in verdict, and each one has between 5 and 123 call sites. Do
    not add one for a single field. Keep the rule local to the config that needs it,
    state the domain in that field's docstring, and lift it into `utils.py` when a
    second caller appears - two copies are the evidence that a shared name is the
    right one, and one caller is evidence of nothing. #2008 asked this for a path
    field and the answer was local: the only other `if not self.<path field>` in the
    tree (`training/_inproc.py`) is a branch that skips logging, not a validation.

12. **A security floor on a transitive package is a constraint, not an override** -
    a package that arrives only through another dependency has no version declared
    anywhere, so the resolver's choice is what stands between the dependency graph
    and a HIGH advisory. State the floor in `[tool.uv] constraint-dependencies`,
    at the first version clearing the advisory rather than the version currently
    resolved, and name the GHSA id in a comment beside it. Use a constraint and
    not an override: measured on this manifest, `gymnasium>=1.1.1` as a constraint
    fails `uv lock` and names the `[vera-sim]` extra's contradicting
    `gymnasium==0.29.1`, while the same floor as an override resolves silently and
    discards that requirement - so an override hides exactly the signal a security
    floor exists to raise. `[project]` is the wrong home while the package stays
    transitive; move the bound there if it ever becomes direct. Pinned by
    tests/test_dependency_audit.py.

13. **Every parameter an agent tool exposes needs its own `Args:` entry** - a
    `@tool` function's input schema is derived from its docstring by
    `docstring_parser`, and the decorator substitutes the placeholder
    `"Parameter <name>"` for any parameter it cannot find there. The model
    driving the tool reads that schema and nothing else, so a placeholder makes
    the parameter undiscoverable however carefully the source explains it.
    Three spellings produce one, and the last two read as documentation in the
    source, which is what makes the loss silent: the entry is absent; the entry
    sits under a section header other than `Args:`, which the parser discards
    entirely (prose reaches the tool description only when it appears *before*
    `Args:`); or one entry names several parameters at once (`a / b: ...`),
    which is read as a single parameter literally named `"a / b"` and therefore
    describes neither. Pinned by
    tests/tools/test_agent_tool_parameter_descriptions.py.
14. **`__repr__` must not raise** - it is what a traceback, a debugger and a
    failing assertion render, so it must not be the thing that hides a failure.
    A class that validates its own arguments raises before it assigns the
    attributes its `__repr__` reads, and the raising frame keeps that half-built
    instance alive: rendering it reports `[AttributeError ... raised in repr()]`
    naming an attribute that has nothing to do with the refusal under
    investigation. Wrap the body in `try` / `except AttributeError` and return
    `strands_robots.utils.partial_construction_repr(self)`, which reports the
    lifecycle fact and deliberately names no attribute so nobody is sent
    chasing one. That helper owns the wording, so the phrase a reader learns to
    recognise cannot diverge between layers. Pinned by
    tests/test_repr_survives_partial_construction.py.

15. **A recording test names its own dataset root** - `DatasetRecorder.create`
    and every backend's `start_recording` resolve a `repo_id` with no `root` to
    `$HF_LEROBOT_HOME/{repo_id}`, i.e. `~/.cache/huggingface/lerobot/{repo_id}`
    by default, and `_prepare_create_target` *inspects* that directory before
    any injected fake dataset class is reached. So a unit test that writes
    nothing to the shared cache still reads it, and its verdict depends on what
    the developer's cache already holds. Measured across 39 such call sites: one
    unrelated dataset planted at `local/probe` turned 133 passed into 22 failed,
    every failure a `FileExistsError` naming a path in `$HOME` rather than the
    test's own resolution - which is what makes it hard to attribute. Pass
    `root=str(tmp_path / "dataset")`, including at the sites refused before the
    root is resolved: requiring it of those too keeps the rule one line with no
    exemptions, where the alternative has to model which guard fires first. Note
    a `repo_id` is as often positional as keyword (`create("user/data", ...)`) and
    the two forms are the same exposure. Rebinding the dataset home suite-wide
    would close the class in one line and would also break the one test that
    legitimately asserts the documented default, which is why the rule lives at
    the call site. `tests_integ/` records real datasets and is out of scope.
    Pinned by tests/test_recording_root_is_not_the_shared_cache.py.

16. **An example attests the records it shows, not the whole audit log** -
    `verify_audit_integrity()` with no argument re-reads the entire log, and an
    example's log is the developer's real `~/.strands_robots/mesh_audit.jsonl`,
    because examples deliberately do not redirect `STRANDS_MESH_AUDIT_DIR`. So
    an example that scopes its read to the run (`read_audit_log(since=...)`)
    and then attests everything prints one document describing two record sets.
    Measured on `e4fe2f9` with 4000 records of prior history in the log,
    `examples/fleet/04_emergency_evacuation.py` rendered
    `Audit integrity: ok=False (signed=5/4005)` above a five-row timeline. The
    `ok` value is the worse half: history written before a PSK was configured is
    unsigned, and an unsigned record is a forgery by definition once a PSK is
    set at verification time, so a completely successful run reports tamper
    evidence. Scoped to the records shown the same run reports
    `ok=True (signed=5/5)`. Pass the records (`verify_audit_integrity(records)`),
    or have the report pair them itself so the caller cannot get it wrong.
    `tests/` is exempt mechanically rather than by trust - a test redirects the
    audit dir to `tmp_path`, so there the whole log *is* the record set it
    means. Pinned by tests/test_examples_attest_only_what_they_report.py.

17. **A transport delegates to the raw backend path, never to the router that
    resolves it** - `strands_robots.mesh.session` exposes every Zenoh operation
    twice: a public, backend-aware entry point (`get_session`, `put`,
    `release_session`, `session_alive`) that resolves whatever
    `STRANDS_MESH_BACKEND` selects, and a private `_*_directly` helper that
    always takes the raw Zenoh path. A `MeshTransport` implementation must use
    the second kind for *every* delegation, because under
    `STRANDS_MESH_BACKEND=bridge` the router resolves the `BridgeTransport` that
    owns that very transport - so a backend-aware call routes straight back into
    the caller. Both re-entries are silent, which is what makes the rule worth
    stating rather than leaving to review: a re-entrant `put` raises
    `RecursionError`, which is a `RuntimeError` subclass and so is absorbed by
    the narrow `except (RuntimeError, ConnectionError, OSError)` that idempotent
    transport paths are required to use, and a re-entrant `close` blocks on the
    factory's non-reentrant lock from the thread already holding it. Neither
    reports anything a caller can act on. Every fixture that injects a *fake*
    leg into a composite transport hides this by construction, so the raw path
    has to be pinned structurally. Pinned by
    tests/mesh/test_zenoh_transport_bypasses_backend_routing.py.

## PR Workflow

1. Create the feature branch **on your fork**. Branch creation in the base
   repository is refused for every account, `ADMIN` included: the `default`
   ruleset's conditions are `ref_name.include: ["~ALL"]` rather than the default
   branch alone, and its rules include `creation` with `bypass_actors: []`, so
   `git push <base> HEAD:refs/heads/<new>` comes back as a `repository rule
   violation` that does not name the rule.

   That message is indistinguishable from the two failure modes this file does
   describe - a token missing a permission, and the `.github/workflows/**` write
   refusal that makes an installation token read `BLOCKED` (step 8) - and both of
   those are answered by retrying with a wider token, which is why that is the
   natural next move and why it cannot work here. A ruleset bypass is granted per
   ruleset, so no role implies one, and there is no classic branch protection to
   be exempt from (`GET /repos/{owner}/{repo}/branches/main/protection` -> 404).
   Read the rule back rather than widening the token:

   ```
   GET /repos/{owner}/{repo}/rulesets/{id}
     conditions.ref_name.include  = ["~ALL"]        # not just the default branch
     rules[].type                 contains "creation"
     bypass_actors                = []              # so no account clears it
   ```

   Push the branch to your fork and open the pull request cross-repo:
   `createPullRequest` takes the base repository as `repositoryId` and the fork
   as a separate `headRepositoryId`. Step 5 already says "from your fork"; step 1
   is where that stops being a preference.

   Before you start, check that no open pull request already claims the issue:

   ```
   python3 scripts/check_duplicate_claim.py --repo strands-labs/robots --issue <N>
   ```

   Name the repository rather than leaving it to be inferred. `$GITHUB_REPOSITORY`
   is where the command is *running*, which for a scheduled agent need not be a
   checkout of this one, and an intake check that reads a different repository's
   open pull requests reports `unique-claim` and exits `0` -- a wrong answer shaped
   exactly like the right one, and one an issue number alone gives the script no way
   to detect afterwards. Intake mode refuses an inferred repository for that reason;
   the `--pr` mode keeps the default, because a workflow reviewing a pull request
   runs where that pull request lives.

   A duplicate claim is invisible to every other check here, because they all read
   one pull request at a time and this is a property of the *set* of open ones. It
   is an intake failure rather than a drift: measured over the last 100 pull
   requests, three pairs claimed one issue each and **all three opened inside one
   ~35-minute window**. Three of the six were abandoned and every one of those had
   already been **approved**, so what a duplicate spends is a review approval on a
   change that could never ship - and review is the scarcest resource here (#1905).
   Two of the three pairs also carried a real `git merge-tree` content conflict, so
   they could not both have landed.

   One query, reading the same `closingIssuesReferences` field the closing-keyword
   gate reads. Asking it here prevents the authoring rather than capping it, which
   is why it belongs at step 1 and not at review. If a competing implementation is
   wanted on purpose, exactly one should claim the close and the other should
   cross-reference (`per #N`, `towards #N`) instead.
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
   **Ask the intake question again before the first push.** Step 1's
   duplicate-claim read is a claim about minute 0, and authoring a tested change
   takes longer than the window in which a collision becomes observable. Every
   pair #2017 measured opened inside one ~35-minute span, and the cycle that
   shipped #2030 re-read its own gate ~40 minutes after intake to find the issue
   already claimed, approved and merged.

   Two reads, because neither the command nor the branch answers this alone:

   - *Unpushed work claiming an issue.* Re-run step 1's command, and read the
     issue's own `state` and `stateReason`. The command reads
     `repository.pullRequests(states: OPEN)`, so it can see a rival only while
     that rival is open: #2030 opened 07:24:45 and merged 07:43:54 closing
     #2029, and the same command run at 07:56 reported `unique-claim` with exit
     `0` over four compared pull requests -- while #2029 was `CLOSED` /
     `COMPLETED` with #2030 recorded as its closer. That is a 19-minute
     observability window inside a ~40-minute authoring one, and the answer
     outside it is the reassuring one. The issue's state is the signal that
     stays true; the command is the one that names the rival, which is why both
     are worth asking.

   - *A review-round push on an existing pull request.* Read
     `pullRequest { state mergedAt }`. Comparing the branch against the sha you
     recorded at the start catches a sibling push, but a squash merge writes a
     new commit onto the base and never moves the head ref, so the comparison
     cannot observe it. On #2015 that cost a round: merged 23:13:13 with
     `headRefOid ea5e3ff8`, `mergeCommit 1026088`, and a round pushed a minute
     earlier left the fork branch at `e7ab4d5b` -- which is not an ancestor of
     `main`. The comparison passed, the push succeeded, and the content was
     orphaned on the fork; the recovery was a second pull request (#2018).

   The same read carries `reviewThreads`, so ask for them there rather than
   separately -- an unresolved thread is also only unresolved as of the read. On
   #2028 a thread arrived at 06:18:56, 16 minutes after the commit pushed at
   06:02:32, so a run that read threads when it pushed could not have seen it.

   Guidance rather than a check, by necessity: the collision is between an
   unpushed local tree and a remote pull request, which no workflow can see. The
   `scripts/` gate deliberately says nothing about whether an issue is closed --
   refusing a pull request for that would accuse correct work whose issue
   someone else closed first -- and that is the same mode split as its `--repo`
   default. What is decisive for unpushed work is not what a review check should
   refuse.
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
   **A closing keyword in a PR *title* links nothing.** GitHub parses closing
   keywords from the body and from commit messages, never from the title. A title
   ending `... (closes #1891)` therefore leaves that issue open on merge, and
   nothing on either side says the claim was dropped: a bare cross-reference
   renders identically to the start of a closing link, and the field that would
   contradict the title is one nobody opens. Measured over the last 100 pull
   requests here - 29 titles carry a keyword before an issue number, 27 also
   linked the issue, and two did not:

   | pull request | title claims | links | what it cost |
   |---|---|---|---|
   | #1894 | `closes #1891` | none | #1891 was still open two days after the merge |
   | #1923 | `closes #1912` | none | #1912 had to be closed by hand |

   So put the keyword in the **body** - a line reading `Closes #N` - and leave the
   title free to describe the change. This is now surfaced by
   `.github/workflows/closing-reference.yml`, the same documented-and-enforced-by-
   nothing shape as the changelog rule in step 3 before #1784.

   It deliberately does **not** scan the body for the keyword, because that
   implementation passes the incident it was written for: #1894's body *does* say
   `closes #1891` - inside a code span, which GitHub does not link - so a text
   scan and GitHub disagree on exactly the pull request that matters. The gate
   compares the title against `closingIssuesReferences`, which is the link set
   itself. Two consequences worth knowing when you write a description: a keyword
   in a code span or a fenced block links nothing, and one keyword governs one
   number, so `fixes #12 and #13` closes only #12.

   The gate is self-clearing - editing the description creates the link and the
   check re-runs on `edited` - and it says nothing about a pull request that
   claims nothing anywhere. Whether every change must trace to an issue is the
   separate question #1961 raises, where the board-coverage half of this lives.
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

     **Count the `nodes`, never `totalCount`.** An `itemTypes` argument narrows
     `nodes` and nothing else: on a filtered connection `totalCount` is the count
     of the *whole* timeline - commits, reviews, comments, project status changes
     - so it answers a question nobody asked, and it answers it in the direction
     that looks like caution. It invents a flip history where there is none. Four
     pull requests, one query each:

     | pull request | state | filtered `totalCount` | matching `nodes` |
     |---|---|---|---|
     | #2144 | open, **never closed once** | **2** | **0** |
     | #2143 | merged | 13 | 1, and it is the squash |
     | #1987 | merged, one deliberate flip | 25 | 3 |
     | #1667 | the flip war above | 119 | **45** |

     Asking for a type that cannot be there settles the mechanism, so no re-read
     helps: `itemTypes: [CONVERT_TO_DRAFT_EVENT]` on #2143 returns `totalCount:
     13` beside an **empty** `nodes`. The number is not stale, it is unrelated -
     and it is not even stable. #1667 has been closed and retired since
     2026-07-30, and two reads twenty minutes apart returned `119` then `120` with
     its 45 close/reopen events unchanged: the new item is a `CrossReferencedEvent`
     from #2146, the issue reporting this. Writing *about* a pull request raises
     its apparent flip count, so a cached count drifts upward and the drift reads
     as somebody flipping it again.

     What the misread costs is the flip, which is not optional - the close/reopen
     below is the *only* remedy for a head commit that spawned no check suite, and
     re-running and re-pushing are both unavailable there. On such a pull request
     `totalCount` reports a two-digit alternating run where the truth is zero, and
     declining then looks exactly like following this rule, leaving it `BLOCKED`
     forever and reported as reviewer bandwidth - the presentation #1905 and #1917
     each record for their own cause. On a genuine flip war both readings say "do
     not flip", so their agreement is never evidence the count is sound.

     Read the tail, not the head. `nodes` is ordered oldest-first while what makes
     a flip unsafe is what happened *last*, and on #1667 `first: 3` and `last: 3`
     are disjoint windows five days apart - so use `last: N`. A merge also writes a
     `ClosedEvent`, which is why #2143's single node is its own squash rather than
     someone undoing you: a distinction the node list draws and a count cannot,
     since it counts neither closes nor reopens. Pinned by
     tests/test_timeline_filter_count_is_unfiltered.py.
   - *After.* A `mergePullRequest` mutation can report `Pull Request is not
     mergeable` on a merge that in fact landed - observed on #1756, where the
     mutation returned that error and the squash was already on `main`. Confirm
     with `state`/`merged`, or `git log origin/main`, before concluding a merge
     failed and redoing the work.

     Read that error as uninformative rather than rare. #2249 and #2250 were
     squashed thirty seconds apart, each by a single `mergePullRequest` call
     carrying `expectedHeadOid` - so a stale oid is ruled out - against a pull
     request reading `CLEAN` / `MERGEABLE` / `APPROVED` with every required
     context `SUCCESS`. Both calls returned `Pull Request is not mergeable` beside
     `mergePullRequest: null`, and both squashes were already on `main`:
     `926beb9` at 19:24:50 and `07a759d` at 19:25:20. Three for three with
     #1756's `4bf139c`, and the payload carries no field that separates a refusal
     from a success, so only the read-back can tell you which one you got.

     The read-back also names the likely cause, in the field the error is worded
     about: after the merge #2249 reports `mergeStateStatus` and `mergeable` as
     `UNKNOWN`, consistent with the mutation re-reading a pull request it has just
     closed. That makes the retry the expensive reflex rather than the safe one -
     a second call against #2249 after it had merged returned the identical error
     beside the identical `null`, so retrying manufactures a second confirmation
     of a failure that never happened. Pinned by
     tests/test_merge_mutation_error_is_not_a_verdict.py.
   - *And on `main` afterwards.* A rollup of `FAILURE` on a merge commit is not
     evidence that the squash broke anything: a **cancelled** check aggregates
     into `FAILURE` and the rollup carries no reason, so read each context's own
     `conclusion` before you believe it.

     The producer this used to name is gone. `pr-and-push.yml` keyed its
     concurrency group on `github.event.pull_request.number || github.ref`, and a
     push carries no PR number, so every push to `refs/heads/main` shared one
     group under `cancel-in-progress: true` and each merge killed the run of the
     merge before it. Four PRs merged in the 22 minutes from 03:03:44 to 03:25:25
     left three consecutive commits - #1788, #1794, #1796 - each reporting rollup
     `FAILURE` whose only non-`SUCCESS` context was
     `call-test-lint / Test and Lint` = `CANCELLED`, killed at 1m07s, 15m00s and
     5m38s into their runs. Nothing had failed. See #1800.

     Counting the whole branch is what turned that from a wrong colour into a
     wrong answer: of the last 25 commits on `main`, 24 had a settled rollup, 11
     of those read `FAILURE`, and **9 of the 11 had no failing check at all**
     (#2304). A commit on `main` is immutable and already merged, so unlike a
     branch it gets no next push to clear the answer - and a burst of merges
     destroys the evidence for *which* commit in it broke `main` in the same act
     as creating the fault, which is the read #2303 had to make. The group now
     keys on `github.sha`, so each pushed commit gets its own group and runs its
     own suite to completion. Pinned by tests/test_push_concurrency_group.py.

     Two things survive that fix. A `main` commit can still carry a cancelled
     context from `docs.yml`, which keys on the ref and holds a `build` (a
     per-commit verdict, where cancelling has the same defect) in the same group
     as a `deploy` (a shared resource, where superseding is wanted) - three of
     #2304's eleven are that, and splitting the two is #2305. And a batch of N
     merges now runs N suites rather than 1, which is what buys each commit its
     own answer; what it no longer costs is a manual bisect after a red tip,
     because an intermediate commit's own green is available to lean on. A batch
     is still defensible on the same grounds as before - each PR individually
     green, passing `Detect an untested overlap with the base branch`, and
     touching a file set disjoint from the others.
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

     **A node ID is not opaque, which makes one direction of this checkable
     before the write.** It is `<TypePrefix>_<urlsafe-base64(msgpack array)>`,
     where a repository is `[0, databaseId]` and anything a repository owns is
     `[0, repository databaseId, own databaseId]`, so a decode costs no network
     call:

     | node ID | decodes to | resolves to |
     |---|---|---|
     | `R_kgDORUMiZg` | `[0, 1162027622]` | this repository |
     | `R_kgDOD1WOFw` | `[0, 257265175]` | the #1916 stray |
     | `PR_kwDOD1WOF87DdSjQ` | `[0, 257265175, 3279235280]` | **the same stray** |
     | `PR_kwDORUMiZs7Kw3fA` | `[0, 1162027622, 3401807808]` | **`uutils/coreutils#11342`** |

     The third row is why #1916's three guessed IDs all failed the same way: one
     stale value contaminated every mutation, and the two that failed did so only
     because their own databaseId happened not to exist there - `Could not
     resolve to a node`. Failing closed was luck about the guess, not a property
     of the API, and the guess that got lucky the other way is the one that wrote.

     **The fourth row is why a decode is a reject and never a pass.** That ID
     carries *this* repository's databaseId in its middle field and resolves to a
     merged pull request in `uutils/coreutils`, whose own repository databaseId is
     `11847500`. GitHub routes on the third field - the object's own id - and
     neither validates nor uses the middle one, so a `mergePullRequest` against it
     was aimed at a stranger's pull request and was stopped by permissions rather
     than by the check. It also shares 14 of its 19 characters with the correct ID
     for #2006 (`PR_kwDORUMiZs78G3VE`), so eyeballing it against a known-good ID
     for the same repository is the same unsound test done less precisely. See
     #2007.

     So the decode has exactly one sound use: a middle field naming another
     repository is proof of a wrong ID, and a middle field naming this one proves
     nothing at all. **The rule is the one with no decode in it** - resolve every
     ID from a query in the same run whose owner and name are written out
     literally, preferring the response of the query that named the object by
     `owner`/`name`/`number`; check the prefix against the parameter, since a
     `PR_...` handed to a `repositoryId` is wrong by type alone; and read the
     `url` in the response back before treating the write as done. Only a
     mutation takes a bare ID - a query names its subject by
     `owner`/`name`/`number`, so it cannot address the wrong repository at all.
     That is why no read has ever been implicated, and why the rule costs one
     round trip and binds only where something changes.

     **Weight it by reversibility rather than by correctness**, because the two
     directions are not symmetric. A refused `mergePullRequest` leaves nothing
     behind; a `createIssue` against a wrong ID succeeds and cannot be undone by
     the account that made it, since `deleteIssue` needs admin on the target and a
     stray write by definition lands where you have none. It has now happened
     twice: the second was `Ali111q/todo#1` at 16:23 UTC on 2026-08-07, twenty
     minutes after #2007 was filed, from a `repositoryId` whose repository field
     reads `1060491130` and not this repository's `1162027622` - the one
     direction a decode does catch. So `createIssue`,
     `addComment` and `updateIssue` earn the read-back more than the merge that
     prompted the rule, not less. If one has already landed, the remedy is not
     deletion: retitle it to mark it opened in error, replace the body with an
     explanation, close it, and do not open a replacement in the same repository -
     which is what `Ali111q/todo#1` now records, its own body noting that deletion
     was refused.
     `tests/test_graphql_node_id_targeting.py` decodes both shapes against the
     `databaseId`s the API publishes beside them, so neither the envelope changing
     nor the reject-only limit softening back into a pass goes unnoticed.
   - *And that the repository still accepts writes at all.* Archiving is
     invisible in every field a sweep already reads. `strands-labs/robots-sim`
     was archived at 01:33 UTC on 2026-08-06, between one scheduled cycle and
     the next, and a scan taken minutes afterwards returned its open pull
     request and its four open issues completely normally, next to
     `viewerPermission: ADMIN`. Nothing in that payload distinguishes it from a
     live repository. The first and only signal was the mutation:

     ```
     createIssue  ->  Repository was archived so is read-only
     ```

     `viewerPermission` is the field that misleads, and it keeps reporting
     `ADMIN` afterwards because the permission is genuine - the repository is
     what changed, not the grant. So it is not a stale or buggy read, and no
     amount of re-reading it helps; it answers a different question than the one
     being asked.

     What it costs is the whole run rather than a retry, because the refusal
     arrives at the *end*: an archived repository accepts no branch, no issue and
     no pull request, so a clone, a branch, a three-file fix, a regression pin
     verified to fail on pre-fix code, and a clean `black`/`isort`/`flake8` run
     were all completed before anything reported a problem, and none of it could
     land. A fork's branch still pushes, which makes it worse rather than better
     - the pull request it would open is the step that is refused. Ask for the
     one field on a query already being made, before the work and not after it:

     ```
     repository(owner: ..., name: ...) { isArchived viewerPermission }
     ```

     Two consequences beyond the read. An archived repository is terminal, so
     treat `strands-labs/robots-sim` as closed for good: epic robots-sim#167
     completed, and any remaining cross-repo item naming it - the
     `robots-sim MIGRATION.md` half of #1274 - can now only be satisfied or
     closed on this side. And a defect found in a repository that is already
     archived is not automatically worth fixing anywhere: the deprecation notice
     there names an undeclared upstream extra, which is the exact hazard
     `tests/test_dependency_audit.py` guards here, but robots-sim never cut a
     release carrying it - its latest tag predates the notice - so it reached no
     installer and the correct action was to drop the fix rather than relocate
     it. Check what actually shipped before deciding an archived finding needs a
     home.
   - *And that the field naming the review decision is present at all.*
     `reviewDecision` has a third reading beyond `APPROVED` and
     `REVIEW_REQUIRED`: **`null`** - and it does not mean what an absent value
     suggests. #1974 sat at `mergeStateStatus` `BLOCKED` carrying a current
     `APPROVED` review that post-dated its head commit, the required check
     `SUCCESS`, `require_last_push_approval` satisfied, and `reviewDecision`
     `null`. Resolving its one unresolved thread moved both fields at once:

     | field | one unresolved thread | after `resolveReviewThread` |
     |---|---|---|
     | `mergeStateStatus` | `BLOCKED` | `CLEAN` |
     | `reviewDecision` | `null` | `APPROVED` |

     The gate behind it is already in this file: the `default` ruleset sets
     `required_review_thread_resolution: true`, and #1890 measured a merge
     landing 8 seconds after its last thread was resolved. What #1974 adds is
     the *signature*, and it is the one value that misreads in the reassuring
     direction - `REVIEW_REQUIRED` at least says a review is owed, whereas
     `null` reads as "no review requirement applies here" rather than "one
     resolve from merging". It is not a recompute lag: that approval was more
     than twenty minutes old when the field was read as `null`.

     **`null` is at least two states, and the resolve clears only one of them.**
     #2328 presented #1974's signature exactly - `MERGEABLE`, `BLOCKED`, `null`,
     one unresolved `github-advanced-security` thread, the required check
     `SUCCESS` - so the paragraph above prescribed resolving that thread, which
     was the right action. It settled the decision the other way:

     | pull request | approving review present | after `resolveReviewThread` |
     |---|---|---|
     | #1974 | one `APPROVED`, post-dating the head | `APPROVED` / `CLEAN` - merges |
     | #2328 | none - every review `COMMENTED` | `REVIEW_REQUIRED` / still `BLOCKED` |

     `null` is also what a pull request carrying **no approving review at all**
     reads, because a `COMMENTED` review contributes no approval. So the resolve
     was necessary on both and sufficient on one, and "this one needs no review
     at all" is true of #1974 and false of #2328 while the field is identical on
     the two.

     Read the review set beside the threads, and read it **before** you resolve
     anything. #2328's decision moved from `null` to `REVIEW_REQUIRED` on the
     resolve, so the one value that says which case you are in is gone by the
     time you re-read it:

     ```graphql
     reviews(last: 20) { nodes { state author { login } submittedAt } }
     reviewThreads(first: 50) { nodes { id isResolved isOutdated } }
     ```

     A `null` carrying no `APPROVED` node is `REVIEW_REQUIRED` wearing a
     different value: the remedy is a first approving review, and no amount of
     resolving supplies one. That leaves `REVIEW_REQUIRED` itself carrying two
     remedies - a first approval, or a second account when the only approval
     came from the pusher - which is the split
     `scripts/check_last_push_approval.py --all-open` reports and which no
     single field distinguishes either.

     `isOutdated: true` on an unresolved thread is the common form, and it is a
     prompt rather than reassurance: the diff moved on, so the request has
     usually already been satisfied by a later commit and only the resolve is
     outstanding. #1974's was addressed by the commit before its head, and the
     approving review said so, and it still held the merge.

     **`resolveReviewThread` needs `PAT_TOKEN`.** Under the Actions
     `GITHUB_TOKEN` it returns `Resource not accessible by integration` - the
     same refusal shape as the board reads in step 6. Resolving is often the
     entire remaining distance to a merge, so a sweep holding only an
     installation token cannot finish the job it has correctly diagnosed.

     Resolve rather than push. A push clears nothing here and costs the approval
     twice: `dismiss_stale_reviews_on_push` drops it, and
     `require_last_push_approval` then disqualifies the pushing account from
     re-supplying it, turning a one-approval merge into one that needs a second
     reviewer.
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

   A third cause of that same presentation needs no workflow edit, and no second
   token to see. The `mergeStateStatus` values above - `BLOCKED` while the required
   check runs, then `UNSTABLE` or `CLEAN` - assume the required check runs. A head
   commit created through the API under the Actions `GITHUB_TOKEN`
   (`createCommitOnBranch`, or `PUT /repos/{owner}/{repo}/contents/{path}`) spawns
   **no check suite at all**, because GitHub suppresses workflow triggers for events
   it attributes to that token so that a workflow cannot re-trigger itself. Nothing
   ever reports `call-test-lint / Test and Lint`, so the required set is never
   satisfied and `BLOCKED` is terminal rather than transient.

   It is legible in one field, and only as an absence:

   ```
   commits(last: 1) { nodes { commit { checkSuites { totalCount } } } }   ->  0
   ```

   Zero suites, not a red one. On #1987 every commit pushed from a clone carried
   10-13 suites and the one written through the API carried none, same branch, same
   day:

   | head | committer | `checkSuites.totalCount` |
   |---|---|---|
   | `b10f4dce` | `./c²` (clone) | 13 |
   | `a3f3e3f6` | `cagataycali` (API) | 0 |

   At `a3f3e3f6` that PR satisfied every gate this file tells you to read -
   `APPROVED` by an account other than the pusher, `MERGEABLE`, no unresolved
   thread - while `statusCheckRollup` read `null` and `mergeStateStatus` read
   `BLOCKED`. That payload is indistinguishable from a required check still
   queued, so it was reported as waiting on CI for two consecutive scheduled
   cycles.

   **Reopen it; do not re-push it.** `pr-and-push.yml` takes the default
   `pull_request` types, so `reopened` recomputes the *unchanged* head sha: no
   commit, therefore no push, therefore neither `dismiss_stale_reviews_on_push` nor
   a new last pusher, and the approval survives. Re-pushing the same tree with
   `PAT_TOKEN` triggers too, but pays a re-approval round that reviews no changed
   behaviour - and re-running is not on the table, since `totalCount` is `0` so
   there is no suite to re-run and the workflow has no `workflow_dispatch`.

   **The flip needs `PAT_TOKEN` as well.** A close/reopen attributed to the Actions
   token is suppressed for the same reason the commit was, so the remedy applied
   with the wrong token is a silent no-op that looks like the diagnosis was wrong.
   Read `timelineItems(itemTypes: [CLOSED_EVENT, REOPENED_EVENT])` first, as this
   step already requires: #1987 had none and cleared on a single flip - same head,
   still `APPROVED`, nine suites queued including the required one. #1988 has the
   full account.

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

   **One field says whether a composition exists at all, and the overlap read
   does not.** File overlap says two pull requests touched the same file; it does
   not say either one landed outside the other's ancestry, which is the thing
   that makes a pair of changes never compiled together:

   ```
   GET /repos/{owner}/{repo}/compare/main...{head_owner}:{head_repo}:{head_branch}
     ->  .behind_by
   ```

   **Qualify the head with its owner and repository.** Step 1 mandates that the
   branch live on a fork, and an unqualified fork ref does not resolve in the base
   repository, so the form a reader reaches for `404`s on every pull request here -
   same head, same instant:

   | ref | result |
   |---|---|
   | `main...feat/ackermann-ros-robot` | `404 Not Found` |
   | `main...Vivek0712:robots:feat/ackermann-ros-robot` | `diverged  ahead_by=11  behind_by=116` |

   Resolve the three parts from `pullRequest { headRepository { nameWithOwner } }`
   and `headRefName` rather than assuming them. The qualified form is also correct
   for a branch in the base repository - `main...strands-labs:robots:main` ->
   `identical` - so there is one form to remember rather than a choice to make.

   `behind_by: 0` means the head already contains every commit on `main`, so the
   tree CI tested **is** the merge result. The two cases separate exactly, each
   compared against `main` as it stood when that pull request merged:

   | branch | `compare/<main then>...<head>` | composition |
   |---|---|---|
   | #1763, which broke `main` | `diverged  ahead_by=2  behind_by=1` | owed |
   | #2012, which raised the same alarm | `ahead  ahead_by=3  behind_by=0` | none exists |

   #2012 edited `strands_robots/policies/vera/provider.py`, which #1992 had
   touched earlier the same day, so it met the trigger condition verbatim - but
   #1992 sat 13 commits back in the branch's own ancestry
   (`compare/<#1992 squash>...<head>` -> `ahead  behind_by=0`) rather than
   landing beside it. Following the rule as written would have spent a clone and
   two suite runs to rediscover that, and it would not have looked like a
   mistake; it would have looked like diligence.

   Read the field in the same direction as the decode rule below. **A
   `behind_by` of `0` proves nothing needs composing, and a `behind_by` above
   zero does not prove a conflict exists** - it is the precondition that makes
   the overlap heuristic worth spending a run on, since a semantic conflict need
   not share a file at all. Two properties make it safe to lean on. The counts
   are totals rather than page counts, so distance does not weaken them:
   `compare/v0.4.1...5757c1a2` reports `ahead_by=877` beside a `commits` array
   truncated to 250, and the reverse direction reports `behind_by=877` beside an
   **empty** one, so deriving the answer from `commits` is itself a false safe.
   And a head that cannot be compared is not a `0`: run the composition. That
   case is narrower than a `404`, which has two causes wanting opposite actions -
   an unqualified fork ref is a query to re-issue, while only a head sha that is
   genuinely gone (a force-push, a deleted fork) is uncomparable - and the status
   code does not separate them. Qualify first, then read a `404` on the
   *qualified* form as the uncomparable one.

   **Both readings above are per-branch, and neither can see the open set.**
   `M..base` is empty by construction whenever `M` is the base the branch was
   evaluated against, which is every run, so two pull requests that are both
   still open are invisible to each other: the intersection contains neither, the
   overlap check reports clean on both, and the first tree in which the two are
   compiled together is `main`. That is the #1763/#1766 topology arriving from the
   open set rather than from a merged base. It also does not clear itself over
   time - stale *approvals* are dismissed on push, a stale *pass* has no
   equivalent, and a pull request idle in review never re-runs - so the exposure
   runs until that branch's next push rather than until the sibling merges.
   `--all-open` is the caller for both, exactly as for the sweep in step 12:

   ```
   python3 scripts/check_merge_base_overlap.py --github-repo <owner/name> --all-open
   ```

   Run it when reporting repository health. It reads the open set from the API and
   computes the same intersection twice per pull request - once against each
   sibling's `M..head`, once against what has landed on the base since its own
   `M` - so the two modes cannot disagree about what counts as an overlap or as
   prose. Measured on the queue the day it was added: 10 open non-draft pull
   requests, 45 pairs, one pair sharing a behaviour-bearing path (#1035 + #1722 on
   `strands_robots/mesh/__init__.py`) and one stale base 62 commits deep (#1722 on
   `strands_robots/mesh/ros_bridge.py`), neither of which any per-branch signal
   was reporting - both read `mergeStateStatus: CLEAN`.

   Three properties of that sweep are worth knowing before leaning on it. A
   truncated path set is named as unevaluated rather than intersected: a capped
   list is indistinguishable from a complete one in the payload, and this check's
   failure mode is a *missed* overlap, so quietly intersecting a truncated set is
   how one goes missing. The two sides differ in how far away that is - the head
   side is read from the paginated `pulls/{n}/files` endpoint and stops at 3000
   entries, while the base side has no paginated equivalent and keeps the compare
   endpoint's 300 - and the head side is the input to the pairwise mode, so it is
   the one that must not drop a large diff.
   And the two path sets skip apart: the base-side set is the one that grows
   without bound, so it is the one that hits its cap - #1035 was 265 commits
   behind - and dropping the whole pull request for it would discard the pairwise
   finding this mode exists to make.

   Both sides collect `previous_filename` alongside `filename`, so a rename
   intersects a sibling still editing the old name. Without it the two share no
   path, while git - which does detect the rename - applies the sibling's edit to
   the new name and merges with no conflict marker, which is the exact silence
   this sweep exists to break. That is what keeps the two modes agreeing: the
   single-branch one reaches the same set with `--no-renames` (#2246).

   A file carrying a `strict=True` xfail is the highest-value overlap candidate
   there is, because its whole purpose is to fail when a sibling change lands: it
   breaks a composition that git merges without a single conflict marker. #2233
   pinned a defect that way and #2235 fixed it; composed, the tree was red with
   nothing to resolve, which is why `mergeStateStatus: CLEAN` is not merely
   unhelpful here but actively reassuring.

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
   ties your local run to `main`.

   **On a batch that equivalence is still the check to run, but no longer because
   the intermediate commits go untested - they do not.** Since the push
   concurrency group keys on `github.sha` (above), every commit in a batch runs
   its own suite to completion, and a commit on `main` carries merges 1..N of the
   batch, so its own green *is* a verdict on the partial composition it is.
   Measured on the six merges that took `main` from `239f24ab` to `0d811084` in
   about 53 seconds - #2320, #2327, #2329, #2333, #2334, #2335 - all six report
   `call-test-lint / Test and Lint` `success`, where before #2304 only the tip's
   would have survived. What the equivalence buys instead is the whole batch in
   one read: the tree-sha comparison below is scoped to `behind_by == 0`, and in
   a batch only the *first* pull request can satisfy that - every later one is
   behind by the merges ahead of it, so its squash tree differs from its head
   tree for entirely correct reasons and the comparison does not apply at all.
   Diffing the composition against the final tip is the one form that answers for
   all N at once, and dropping the path scoping costs nothing and catches more
   (that batch's unscoped diff was empty too).

   **Comparing the two commits' tree shas is the same claim without the clone,
   and a stronger one.** That `git diff` needs the local composition to still
   exist, and it is path-scoped, so a change under `changelog.d/`,
   `pyproject.toml` or a workflow is invisible to it:

   ```
   GET /repos/{owner}/{repo}/commits/{sha}  ->  .commit.tree.sha

   0fcdd015cb3f  tree=e174201b7ccf...   # #2012 head, call-test-lint SUCCESS
   763305edf1d4  tree=e174201b7ccf...   # its squash on `main`
   ```

   Equal trees say the bytes CI went green on are the bytes on `main` - the whole
   tree rather than two prefixes - and each commit in a batch can be checked that
   way without a suite, which is what the batching case above otherwise has no
   evidence for.

   **That equivalence is scoped to `behind_by == 0`, and it is the field above
   that says so.** When the branch is behind, the squash tree necessarily
   incorporates the intervening commits, so the trees differ for a perfectly
   correct merge. The control and the counterexample:

   | pull request | `behind_by` | head tree | squash tree |
   |---|---|---|---|
   | #2012 | `0` | `e174201b7ccf` | `e174201b7ccf` |
   | #2024 | `1` | `4af91f210d09` | `8b3e7e8a3434` |

   #2024's `main` went green on all four checks afterwards, so the inequality was
   not drift. Read unequal trees as a question rather than a verdict: at
   `behind_by: 0` they are the evidence this paragraph claims, and above zero the
   check does not apply at all, leaving the path-scoped `git diff` against the
   local composition as the only form that does. An unqualified "equal trees or
   else" invites reading a correct merge as a broken one, which is the same
   expensive-in-the-diligent-direction shape as the advisory-`CodeQL` case above.

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
   approvers on every review event and reports when they are the same single
   account. It **reports** rather than fails: a finding leaves the job green and
   lands in the step summary and in a `Needs an approver who did not push the
   head` annotation, because a red X drags `statusCheckRollup.state` to
   `FAILURE`, where it cannot be told apart from the branch's own tests failing -
   measured on #1722 at head `3a32a14`, whose rollup read `FAILURE` with every
   required context `SUCCESS` and this check as the only non-`SUCCESS` context,
   and misread as a broken diff four times. Cite that head, because the citation
   no longer reproduces from the pull request: on `741f4057` this job reads
   `SUCCESS` under its present name, and #1722's rollup is still `FAILURE` for an
   unrelated producer - a cancelled duplicate of the closing-reference check
   (#2216). The decision rests on the `3a32a14` measurement; re-deriving it from
   #1722 today finds this job green and the reasoning apparently unfounded. Red on that job now means the check itself could not
   compute an answer. The check row is named `Report the last-push-approval
   state` for the same reason: green must not assert the absence of a finding. The point of automating it is not that the check is clever - it is
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

   **Automating it is not the same as covering the population, and the gap is
   silent in the same direction as the bug.** That workflow fires on
   `pull_request` and `pull_request_review`, so it can only evaluate a pull
   request that has had one of those *since it landed* - and the pull requests
   this is written for are the ones that have not. #1035's head was pushed
   2026-08-01 and approved 51 minutes later, both before the workflow existed on
   2026-08-04, so `Report the last-push-approval state` (then named `Detect an
   approval the last pusher cannot supply`) is absent from the 11 check runs on
   that head, while every other check is present. So
   the check read `SUCCESS` on pull requests that did not have the condition and
   said nothing at all about the two that did.

   The verdict was never the problem. Run directly, the same script answers
   immediately, and did before this was noticed:

   ```
   python3 scripts/check_last_push_approval.py --repo strands-labs/robots --pr 1035
     -> Outcome: pusher-only-approval, pushed by cagataycali, exit 1
   ```

   #1905 attributes the silence to the workflow's base-branch guard instead.
   That is worth correcting rather than leaving, because it points at a fix that
   would change nothing: the guard checks out the **base**, `main` carries the
   script, so the guard passes and the script would run. What was missing was a
   caller, which is now `--all-open`:

   ```
   python3 scripts/check_last_push_approval.py --repo <owner/name> --all-open
   ```

   Run that when reporting repository health. A sweep that reads only
   `reviewDecision` and `mergeStateStatus` cannot tell
   `awaiting-first-review` from `pusher-only-approval` - both are
   `REVIEW_REQUIRED` / `BLOCKED` - and reporting the second as the first is
   exactly how "reviewer bandwidth is the sole constraint" stood for eight
   consecutive scans over two permanently unmergeable pull requests. Exit 1
   means at least one open pull request needs an approver who is not its pusher,
   and no amount of reviewer time supplies one.

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

### Actuators: a joint pose goes only where `ctrl` IS a joint pose
- **`data.ctrl` is not a pose channel, it is whatever the actuator's force law reads.**
  A `<position kp>` reads it as the joint target, a `<velocity kv>` as a rate, a `<motor>`
  as a torque, an `<intvelocity>` integrates it as a rate. Writing a joint coordinate into
  any of the latter commands a different physical quantity that happens to be numerically
  equal to an angle, and nothing raises: the joint simply moves somewhere the caller did
  not ask for.
- **Anything that writes a pose asks
  `strands_robots.simulation.mujoco.scene_ops.joint_drive_map` first.** Resolving the
  transmission is not enough - a `<velocity>` actuator's transmission IS the joint, and
  every stock tendon gripper measured clears the bias-type and position-gain terms a naive
  servo check would look at. The classification is per actuator, not per robot: `openarm`
  ships 2 position servos beside 16 motors, and 19 of the loadable registry robots have at
  least one joint-transmission actuator that is not a servo.
- **A drive that cannot take the pose is left uncommanded and named, not written to.** The
  motion primitives hold the joints they can hold and report the rest, because writing a
  live joint angle into a rate drive is what *moves* the joint the call meant to hold - and
  a wheel angle accumulates, so that "hold" grows with every turn the wheel has already
  made. Where the drive being targeted is the one that cannot take a pose, the primitive
  refuses instead: a servo loop on a rate drive meets its convergence test when the joint
  sweeps *past* the number while still accelerating, so it reports the set-point as reached
  and the joint keeps going after the call returns.
- Pinned by `tests/simulation/mujoco/test_pose_write_reports_whether_the_servos_hold_it.py`
  for `set_joint_positions(hold=True)` and by
  `tests/simulation/mujoco/test_primitives_write_a_pose_only_into_a_pose_drive.py` for
  `move_to` / `rotate_wrist` / `set_gripper`, which also pins the boundary: a usable
  `ctrlrange` is authoritative and already in the drive's own units, so only the
  *substitution* of a driven joint's limits needs the drive to be a servo.

### MuJoCo enums are matched by value, never by operand order

`mjModel` / `mjData` expose their type fields as numpy integer arrays, while the
matching vocabulary is a pybind11 enum. Whether the two compare equal depends on
which side the enum is on.

- **Compare `int()` to `int()`.** `int(model.geom_type[i]) in (int(mjGEOM_PLANE),
  int(mjGEOM_HFIELD))`, `int(model.actuator_trntype[a]) != int(mjTRN_TENDON)`.
- **`x in (enum, ...)` is the trap, not a hand-written reversed `==`.** CPython
  compares `element == needle`, so membership puts the ENUM on the left. That is
  `True` on mujoco 3.9.0 / 3.10.0 / 3.11.0 and `False` on 3.12.0 - all inside the
  declared `mujoco>=3.5.0,<4.0.0` range. A `set` of enum members degrades the
  same way: the hashes still collide and the confirming equality is the failing
  direction.
- **The failure is silent.** The membership test simply answers `False` for every
  element. Measured: a heightfield ground geom stopped being recognised as
  ground, so a "lowest robot geom" scan returned the ground's own `z=0.0`; and an
  example's home-pose helper matched 0 of 3 joints, writing no `qpos` and no
  `ctrl` while logging that the pose had been set.
- **The rule is uniform, with no exemption for a spec-side value.** `MjsGeom.type`
  really is an enum, so `in` works there - but a reader cannot tell an `MjsGeom`
  attribute from an `MjModel` array element at the call site, so both are written
  by value.
- Pinned by `tests/test_mujoco_enum_comparisons_are_value_based.py`, which grades
  `strands_robots`, `tests`, `tests_integ` and `examples`.

### Clocks: a duration is measured, a stamp is recorded
- **A duration belongs on `time.monotonic()`.** Anything that decides *how long to keep
  waiting* - a timeout, a deadline, a TTL, a retry window, a rate window, a frame pacer -
  must be measured on a clock that only moves forward at one second per second.
  `time.time()` is the current opinion about the date, and an NTP correction, a `date -s`,
  or a VM resume moves it by an arbitrary amount mid-wait: forward, the wait ends early
  with the work still in flight; backward, it runs past the caller's budget by the size of
  the step. Neither is reported, because nothing raised.
- **An absolute stamp stays on `time.time()`.** A record's `timestamp`, a session
  `start_time` persisted to disk, the `t` field of a wire envelope whose freshness another
  machine judges - those name a point in time that something off this process correlates,
  and seconds of local process uptime is meaningless to that reader.
- **The two can share a function and must not share a variable.** `serial_tool`'s monitor
  bounds its window on `time.monotonic()` and stamps each returned record with
  `time.time()`; the mesh keeps `_last_estop_mono` beside `_last_estop_ts` for the same
  reason. If one value is used both to decide and to report, split it rather than picking
  a compromise clock.
- Pinned by `tests/tools/test_tool_wait_budgets_survive_a_clock_step.py` (a scan over the
  agent-callable tools, no exemption list) and, for the safety subsystem, by
  `tests/mesh/test_replay_cache_monotonic.py` and
  `tests/mesh/test_corroboration_clock_domain.py`. The frame pacer named above is pinned
  by `tests/simulation/test_rollout_durations_survive_a_clock_step.py`, which asserts the
  *achieved* frame interval across a clock step rather than the value the pacer computed,
  and the RTC inference-delay estimate by
  `tests/policies/lerobot_local/test_rtc_latency_survives_a_clock_step.py` - a latency that
  feeds a decision is a duration, however much it also reads as telemetry. The two
  rendering pacers - the MJPEG stream generator and the multi-camera recorder thread - are
  pinned on the same achieved-interval basis by
  `tests/simulation/test_rendering_pacers_survive_a_clock_step.py`, along with the duration
  each recording reports. The Isaac backend's own two - the idle gate that decides when
  `run_pump_forever` refreshes the live preview, and the duration its camera recording
  reports - are pinned by
  `tests/simulation/isaac/test_isaac_durations_survive_a_clock_step.py`, which asserts the
  achieved refresh timeline against the unstepped one rather than a tolerance. A duration
  base also carries its clock in its name (`started_mono`, `last_idle_render_mono`), so a
  later reader cannot mistake it for a stamp and subtract `time.time()` from it.

### Posture flags are checked, never read by truthiness
- **A flag that selects a posture is checked; a knob that scales a quantity is
  validated.** Both live in the same signatures and both are caller input, but they fail
  differently. `boolean_flag_error` is the domain for the first kind - a confirmation gate,
  a security opt-out, a preview mode, a search region - and the numeric domains
  (`positive_whole_number_error`, `positive_finite_number_error`) for the second. The two
  are inverses: the numeric ones reject `bool` because it is an `int` subclass that would
  pass as a silent `1`, and this one requires the boolean they turn away.
- **Truthiness inverts exactly the spellings an operator reaches for.** Every non-empty
  string is truthy, so `"false"`, `"no"`, `"off"` and `"0"` select the *other* posture from
  the one they read as, and `None`, `0`, `""` and `[]` take a branch without ever being a
  declared spelling of it. Nothing raises and nothing logs, so the wrong posture is
  indistinguishable from the right one - `actuate_robot`'s `disable_self_collision="no"`
  disabled every collision in the scene, `derive_key_light`'s
  `upper_hemisphere="false"` searched the hemisphere the value asks to skip, and
  `ros2_commands="false"` opened an inbound arm-driving `joint_command` subscription for a
  caller who had asked for a read-only telemetry bridge.
- **Do not parse a vocabulary as a fallback.** A flag arrives already typed, unlike an
  environment variable whose only shape is a string, so the honest answer is to check it.
  Parsing only moves which spellings invert: `"on"`, `"enabled"` and `"y"` are absent from
  every such vocabulary here and would each resolve to the restrictive posture while
  reading as an opt-in.
- **A refusal that branches on the same flag inherits the inversion.** If an error message
  chooses its wording or its remedy from the flag, a truthy non-boolean makes it describe
  the branch the caller did not ask for - `derive_key_light` reported a region black
  "above the horizon" to a caller who had asked for the full sphere, and advised passing
  the value they believed they had passed. The RTPS bridge's DDS Security gate is the same
  shape at higher stakes: it branches on `enable_commands`, so a truthy non-boolean made it
  refuse a *read-only* request as "an enabled command bridge" and advise the insecure
  opt-out - the one remedy that turns that refusal into a silent open of the surface the
  caller asked to close. Checking the flag makes such a branch reachable only where its
  advice is actionable.
- **A facade that checks a flag does not check it for the method it forwards to.**
  Where one contract is reachable through a convenience surface and a documented
  direct API, both read the same caller input, so a guard on one leaves the other
  disagreeing about which values are usable - which is what the shared *numeric*
  domains in the same signatures exist to prevent. Every backend's
  `start_recording` checked the `overwrite` it forwards to
  `DatasetRecorder.create`, and `create` read it by truthiness and deleted the
  caller's dataset. Guard the flag where it is *read*, on the same domain, and
  ahead of the side effect it selects - for a confirmation gate in front of a
  delete, that means before the target is touched, so a refusal cannot arrive
  after the deletion it was refusing. Pinned by
  `tests/test_dataset_recorder_posture_flag_domain.py`, which also records why
  the neighbouring surfaces are out of scope.
- **A flag whose misread only shows up in a rendered frame is checked at construction.**
  Where the branch a flag selects is applied later - a fitted transform, a compositing
  decision - the misread has no error to surface at, so it reads as a scene that looks
  slightly wrong rather than as a bad argument. `GsplatBackground` already raises for a
  nonexistent scene path for exactly that reason (its first `render` sits inside an app's
  catch-all that demotes the photoreal backdrop to a procedural fallback), and its four
  alignment flags - `auto_backdrop`, `skybox`, `metric`, `own_floor` - were read by
  truthiness beside it. `metric` is the sharpest: it also decides whether `radius` is read
  at all, so `metric="no"` kept a capture's raw scale and stood a real 500k-splat room up
  at a 4.45 m radius for a caller who asked for 2.5 m. Check such a flag where the caller
  supplied it, not where the branch is taken. Pinned by
  `tests/rendering/test_gsplat_background_posture_flag_domain.py`, which measures the
  branch each of the four selects.
- Pinned by `tests/simulation/mujoco/test_actuate_robot_posture_flag_domain.py`,
  `tests/simulation/test_recording_posture_flag_domain.py`,
  `tests/tools/test_lerobot_teleoperate_flag_domain.py`,
  `tests/mesh/test_iot_provisioning_flag_domain.py`,
  `tests/rendering/test_key_light_posture_flag_domain.py` and
  `tests/test_ros2_command_surface_flag_domain.py`, each of which parametrizes over
  `boolean_flag_error` itself rather than a copied spelling list, so a spelling added to
  the shared domain is covered without an edit.

### A subset selector is read by membership, never by truthiness
- **A parameter that names a SUBSET of a collection the call already owns is not a value
  with a default; it is a selection.** `None` means "all of it" on these surfaces, so
  reading the parameter by truthiness makes every other falsy value take that same branch
  - and for a selector, that is not a wider default but the *opposite* answer. An empty
  selection asks for nothing and was served everything: `teleoperate(names=[])`, which is
  what a filter that matched nothing produces, connected and drove every attached leader,
  and `detach_teleop("")` removed the whole attached set. Both reported success.
- **Read it `is None`, and refuse the empty selection rather than widening it.** A scalar
  default (a path, a device string, an empty `fields` dict) can be read by truthiness
  because empty and absent genuinely coincide there - the value is derived either way. A
  subset selector is the case where they diverge, so it needs the membership read plus its
  own verdict on emptiness, which the shared name-list domain deliberately leaves to the
  caller ("a surface where an absent value IS an error keeps that verdict its own").
- **The other unhonorable spellings belong to `name_list_error`.** A single name as a bare
  string is iterable per character, a repeated name makes the call do its unit of work
  twice, and a one-shot iterator is consumed by whichever check reads it first, leaving the
  real consumer nothing. Route the shape through the shared domain and keep only the
  emptiness verdict local; refuse before the call touches hardware, a filesystem or a
  thread, so a refused selection has no partial effect to undo.
- **A selection widened to everything can also reach past the call.** `detach_teleop` stops
  the local loop once nothing is left to drive, so a detach widened from one stream to all
  of them ended a running session as a side effect of the misread.
- Pinned by `tests/test_teleop_device_selection_domain.py`, whose controls assert that the
  documented spellings (`names=None`, a real subset, `detach_teleop(None)`) are unchanged,
  and by the render path's `cameras` resolution, which has read the same kind of selector
  `is None` all along - an empty camera selection there resolves to no camera rather than
  to every one.
### A resolution knob is validated before the work it sizes
- **Check the knob that sizes an expensive result before producing the result.** A
  resolution is caller input like any other, and the numeric domains
  (`positive_whole_number_error` for a pixel or frame count) are what it is checked
  against. Checking it late is not merely a worse message: `render_environment_map`
  paid six full background renders - GPU-bound for a `GsplatBackground` - before
  returning a `(H, 0, 3)` map for `equi_w=0`, and `bake_environment_map` probed and
  wrote its cache file before anything looked at the size it was baking.
- **A zero-sized grid is a resolution mistake, and the consumer will misdiagnose it
  as a property of the scene.** An empty map is not distinguishable from a dark one
  by the code that reads it, so `derive_key_light` blamed the background - "the map
  is black above the horizon" - and advised a search flag. Following that advice
  fails identically, because a map with no texels has no hemisphere to search, so
  the only remedy on offer was a dead end and the knob the caller actually got
  wrong was named nowhere. Refuse at the entry point and the accurate diagnosis is
  the first one the caller sees.
- **Check the domain, not the quality.** A resolution the module cannot use is a
  refusal; a resolution that merely buys a poor result is the caller's call, and
  the two are worth keeping apart. `face_size` is the example: the equirect
  reprojection scales by `face_size - 1`, so 1, 2 and 3 each resolve almost
  nothing within a cube face, and the detail a map carries grows smoothly from
  there with no boundary to pin a floor to. `0` cannot produce a map at all, so
  that is refused; `1` can, so it is accepted.
- **Normalize after checking.** The numeric domains deliberately accept an integral
  float and a NumPy integer, so the value has to be put through `int()` before it
  indexes anything - `HybridCompositor` does this for its own `default_width` /
  `default_height`. Where the value is formatted into a cache key that is not
  cosmetic: `2048` and `2048.0` spelled two different environment-map filenames for
  one set of pixels, so a bake already on disk was missed and paid for again.
- Pinned by `tests/rendering/test_environment_map_resolution_domain.py`, which
  parametrizes over `positive_whole_number_error` itself rather than a copied value
  list, so a value added to the shared domain is covered without an edit.

### One writer per log file
- **A file two writers share needs one file object, not one path.** Two file objects over
  one path each track their own write offset, so a buffered writer flushes at its offset
  and overwrites in place whatever the other appended there. The training backends' run
  log tees stdout/stderr *and* root-logger records into a single file, so the logger
  handler is pointed at the stream the tee already holds
  (`logging.StreamHandler(stream)`) rather than opening the path a second time
  (`logging.FileHandler(path)`).
- **The failure is silent and shaped like data, not like an error.** Records vanish, and a
  record straddling a flush boundary survives as a fragment that still reads like one - so
  the loss surfaces as a wrong *value* somewhere downstream. The log a training backend
  parses for its "RUNNING != learning" verdict reported a healthy 1200-step run as having
  produced no metrics at all, with nothing raised and nothing logged.
- Pinned by `tests/training/test_inproc.py::TestCaptureToFileIsTheOnlyWriter`, which
  asserts the log holds exactly the lines that were written - so a dropped record and a
  surviving fragment both fail - and hands records to the installed handler directly, so
  the assertion does not depend on ambient logger levels or on pytest's capture plugin.

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
- **A kill switch is honoured by every path that can start the thing it kills** - and by exactly one predicate. `STRANDS_MESH=false` is documented in README as overriding even an explicit `mesh=True`, but it was resolved by an inline `os.getenv` inside `init_mesh`, and `robot_mesh._gateway_mesh` builds its robot-less coordinator `Mesh` without going through `init_mesh`. So an operator who asked for no mesh still got a real Zenoh session, a `gateway-*` peer advertised to the fleet, and the nine threads `Mesh.start` spawns - cached until `atexit`, because the switch had no reach there. The second-order cost is test isolation: `tests/conftest.py` sets the same variable to keep the suite off Zenoh, so the escape put nine publishing threads inside unrelated tests and one of them failed on a `/health` payload it never provoked. When a flag means "do not start X", give it one predicate and call it at every construction site of X; a second inline spelling is how the first site gets forgotten. Pinned by `tests/mesh/test_gateway_mesh_kill_switch.py`.
- **Currently tracked**: `STRANDS_ROBOT_MODE`, `STRANDS_TRUST_REMOTE_CODE`, `MUJOCO_GL`.

### Safety Defaults
- **Sim-by-default** - any factory that can return either real hardware or a simulator must default to the simulator. Real hardware affects the physical world; users must opt in explicitly with `mode="real"` or `STRANDS_ROBOT_MODE=real`.
- **Reject invalid modes loudly** - `Robot("so100", mode="virtual")` must raise `ValueError`, not coerce to "sim".
- **Document parameter scope** - if `backend=` only applies to `mode="sim"`, say so in the docstring AND log a debug message when it's passed in `mode="real"` so it doesn't appear silently ignored.

### Naming & Module Organization
- **`robot.py` is for the `Robot()` factory**, the user-facing entry point. Hardware-specific code lives in `hardware_robot.py`. Don't have two files both named "robot something" with different responsibilities.
- **Reference module names, not filenames, in docstrings** - `strands_robots.hardware_robot` not `robot.py`. Filenames change; module paths are the public contract.
- **Keep a cross-reference target on one line** - a `:class:`/`:func:`/`:meth:` path is only a dotted path while it is contiguous. Wrapping `:class:`~strands_robots.policies.protomotions.motion_utils.MotionPlayer`` over a line break leaves a token carrying a newline and the next line's indentation, which imports nowhere. Break the prose before the role and give the path its own line.
- **A cross-reference in a test docstring is graded too** - the roles in `tests/` and `tests_integ/` are checked against the real API alongside the package's, because a test module's docstring is where a maintainer working on that subsystem starts reading. Name the seam a fixture actually patches (`strands_robots.mesh.core.get_session`), not a local import alias dressed up as a module path.

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
- **This covers every replay-cache lock and every lazy resolver, not just the
  two the rule was written for.** `Mesh` keeps three replay caches - estop,
  resume and inbound-command dedup (`_exec_cmd`) - and the eviction bound they
  share is a third resolver, `_resume_replay_cache_max`. A resolver is one
  `os.getenv` plus a validating parse, and on an unusable operator value it logs
  too, so the cost is not constant: at the cache sizes these actually sit at, the
  parse is the majority of the critical section rather than a rounding error.
  Resolve into a local before the `with`, then read the local inside it.
  Pinned by `tests/mesh/test_safety_tunables_cached_at_handler_entry.py`, which
  derives the lock set from the class so a fourth cache is graded on arrival.
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
  `NEXT_SEQ_DEGRADED`, `SERIALISE_FAILED`) let a `verify_audit_integrity` walker
  attribute a stream gap to a specific failure class. When you add a new
  `_next_seq`/sign/serialise/persist failure branch, add the matching poison
  `sig` instead of returning early.
- **`_next_seq` runs before serialisation, so an early `return` is a deletion.**
  The sequence number is consumed and persisted before the record is encoded, so
  a branch that gives up after that point leaves the signature this module's
  header documents as "records were deleted". A payload the JSON encoder cannot
  represent keeps the envelope (`ts` / `event` / `peer_id` / `seq`) and swaps the
  payload for a bounded diagnostic; the only remaining drop is an envelope that
  is itself unrepresentable, where there is nothing left to poison with.
- Pinned by `tests/mesh/test_audit_serialise_safety.py`, whose
  `TestEveryDegradedPathWritesARecord` drives every degraded branch from one
  table so a path added later is graded rather than quietly becoming the next
  silent drop.

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
- **The row belongs to the action, not to the backend that served it.** `robot_mesh`
  renders each action onto an agent-side Device Connect connection when one has
  devices and onto the built-in mesh otherwise, and Device Connect is the one tried
  FIRST - so auditing only the mesh rendering left the audited implementations as the
  fallback. Widen an audit contract across every backend that answers the action, and
  record the magnitude that was read (`devices=N`, `local=N remote=M`) rather than a
  bare marker. `peers` is the read worth recording: it returns every device id plus
  every function name the fleet exposes, which is the callable surface a later `rpc`
  would use.
- Pinned by `tests/mesh/test_robot_mesh_readonly_audit_parity.py`, which discovers the
  actions Device Connect answers by calling the dispatcher rather than restating a
  list, so an action added to that backend is graded without editing the test.

### Rate-limit safety semantics
- **A declined HITL approval must NOT consume a rate-limit slot.** The slot is
  recorded only after approval is granted (or atomically via
  `_rate_limit_check_and_record` on the post-approval path). Otherwise nuisance
  prompts an operator declines would lock the agent out of issuing a genuine
  `emergency_stop` - the inverse of the intended safety property.
