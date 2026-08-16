"""Contract pins for which pull_request events may start a workflow.

``pr-and-push.yml`` carries the repository's one required check
(``call-test-lint / Test and Lint``) and cancels its own in-flight run for the
same pull request (``cancel-in-progress: true``). Those two facts together make
the trigger list load-bearing: an event that cannot change the head sha, but is
subscribed to anyway, discards a run of the required check and starts an
identical one over the same commit.

It was subscribed to three such events -- ``ready_for_review``,
``review_requested`` and ``review_request_removed``. None of them can change
``github.event.pull_request.head.sha``, which is the ref the reusable workflow is
handed, so the second run can only recompute the first run's verdict.

Measured on #1899, #1901 and #1902, where two reviewers were requested seconds
after the pull request opened (#1914 records the full measurement). Each head sha ended up carrying **two** completed
``call-test-lint / Test and Lint`` check runs -- one CANCELLED, one SUCCESS::

    #1902  0734d8a7
      07:03:05  pull_request (opened)           CANCELLED  (ran 26s)
      07:03:19  pull_request (review_requested) CANCELLED  (no job started)
      07:03:19  pull_request (review_requested) SUCCESS    (ran 27m16s)

That a CANCELLED check aggregates into a ``FAILURE`` roll-up is already known
and already written down: #1800 measured it on three consecutive ``main``
commits and ``AGENTS.md`` > PR Workflow tells a reader to consult each context's
own ``conclusion`` before believing the roll-up. What that entry describes is a
*different producer* of the symptom -- pushes to ``main`` share one concurrency
group because a push carries no pull request number, so each merge kills the
previous merge's run. That producer read as wanted when this module was written -- a
superseded ``main`` run is of no interest -- and was left alone here.

It is gone now, for a reason that only shows up once the commits are counted: #2304
measured 24 settled rollups on ``main``, of which 11 read ``FAILURE`` and **9 had no
failing check at all**, and a commit on ``main`` is immutable and already merged, so
unlike a branch it gets no next push to clear the wrong answer. Worse, a burst of
merges destroys the evidence for which commit in it broke ``main`` in the same act
as creating the fault. The push side is now keyed on ``github.sha``, which leaves
the pull-request operand this module is about untouched; it is pinned by rendering
the group in ``tests/test_push_concurrency_group.py``.

This one is on pull request head shas, and it discards work for no possible gain,
which is why it was removable here first: cancelling on ``main`` at least superseded
something, while an event that cannot change the head sha supersedes nothing.
Measured twice, ten minutes apart, no new run in between::

    19:35Z   #1899 FAILURE   #1901 FAILURE   #1902 FAILURE
    19:45Z   #1899 FAILURE   #1901 SUCCESS   #1902 SUCCESS

The FAILURE column is #1800's documented aggregation. The second read is the part
that reading-discipline cannot cover: the aggregate is not even stable, so two
sweeps of one unchanged sha disagree and neither is reproducible. Removing the
producer is what makes the question moot for a pull request.

The cost is also unbounded rather than the ~20s these three happened to lose: a
reviewer requested at minute 25 of a 27-minute run discards all 25 minutes,
because when the trigger arrives has nothing to do with how far the run has got.

The remedy is to subscribe only to the events that can change the head sha,
which is exactly GitHub's default ``types`` for ``pull_request`` -- so the fix
is the *absence* of an override, and this module pins that no workflow
reintroduces one. Every other ``pull_request`` workflow in the repository
(``changelog-fragment``, ``merge-base-overlap``, ``dependency-review``,
``codeql``, ``breaking-change-check``, ``agent-api-check``,
``llm-input-safety``) already takes that default; ``pr-and-push.yml`` was the
only one that did not, and it is the only one holding a required check.

One workflow is exempt, and the shape of the exemption matters more than the
entry. The rule above is a statement about *what a workflow reads*: a run whose
input is the tree learns nothing from an event that cannot change the tree.
``closing-reference.yml`` does not read the tree -- its inputs are the pull
request's title and its ``closingIssuesReferences``, and ``edited`` is the only
event that changes either. It is also the event that check's own remedy produces:
moving a closing keyword out of the title and into the body changes no sha, so
without ``edited`` the report would ask for a fix it could never observe.

The measured harm cannot reach the *required check* from it either, and that is a
premise rather than an assurance: cancellation is per concurrency group,
``pr-and-push.yml`` keys its group on its own ``github.workflow`` name and does
not subscribe to ``edited``, so an edit starts no run in that group. Both halves
are read back by ``test_an_exempt_workflow_cannot_cancel_the_required_check``, so
an exemption cannot outlive the reasoning that admitted it.

That premise is true and it was the wrong group to check. The run an exempt event
cancels first is the exempt workflow's **own**, which shares both a workflow name
and a pull request number with the run already in flight. Being exempt is exactly
what makes this reachable -- an exempt workflow is the only kind that can be
started twice on one head sha -- so unlike the ``main`` case above, whose cancelled
context sat on an immutable commit nobody was reviewing, it lands on the *live*
head. #2216 measured it on #1722 and #2205: each head
carried ``Refuse a closing keyword that only appears in the title`` as ``SUCCESS``
and ``CANCELLED`` together, with every other context ``SUCCESS``, and #2205's
roll-up read ``SUCCESS``, then ``FAILURE``, then ``SUCCESS`` across three reads of
one unchanged sha -- the same instability this module records for #1901 and #1902,
reintroduced by the exemption it granted. Whether it happens is a race and the
inter-arrival gap does not predict it: #2204's two runs are 40 s apart and both
completed, #2205's are 71 s apart and one was cancelled.

So an exemption needs a second premise, read back by
``test_an_exempt_workflow_cannot_cancel_its_own_run``: an exempt workflow must not
cancel in progress at all. It is the same conclusion this module already reaches
for the required check -- remove the producer rather than read around it --
applied to the workflow the exemption is granted to.

``test_no_workflow_gates_on_draft_status`` is the premise behind dropping
``ready_for_review`` specifically: nothing in the fleet skips a draft pull
request, so a draft's CI has already run and marking it ready re-runs the same
sha. If a draft skip is ever added, that pin fails and says so, because
``ready_for_review`` would then be the event that starts the first real run.

These are text assertions rather than parsed YAML for the reason the sibling
CI-config pins state (``tests/test_codeql_query_filters.py``,
``tests/test_merge_base_overlap.py``): ``pyyaml`` is an optional dependency
here, and a pin that skips when a dep is missing is not a pin.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"
_REQUIRED_CHECK_WORKFLOW = _WORKFLOW_DIR / "pr-and-push.yml"

#: The only ``pull_request`` activity types that can change
#: ``github.event.pull_request.head.sha``, and therefore the only ones that can
#: give a run something new to check. This is also GitHub's default ``types``
#: for ``pull_request``, which is why omitting the key entirely is the preferred
#: spelling: there is then no second copy of this list to drift from it.
_SHA_CHANGING_TYPES = frozenset({"opened", "synchronize", "reopened"})

#: Workflows whose input is not the tree, mapped to the sha-invariant activity
#: types they consequently need. An entry here is not a waiver of the rule above
#: but an application of it -- the rule asks whether an event can change what the
#: workflow reads, and for these the answer is yes. Safety rests on the required
#: check being unreachable by the event, which
#: ``test_an_exempt_workflow_cannot_cancel_the_required_check`` reads back.
_INPUT_IS_NOT_THE_TREE = {
    # Reads the title and the link set (scripts/check_closing_reference.py), and
    # ``edited`` is both the only event that changes them and the event its own
    # remedy produces.
    "closing-reference.yml": frozenset({"edited"}),
}

#: Matches a top-level trigger key inside an ``on:`` mapping, e.g.
#: ``  pull_request:`` or ``  push:``.
_TRIGGER_RE = re.compile(r"^(?P<indent> +)(?P<name>[a-z_]+):\s*$")

#: Matches an inline activity-type list, e.g. ``types: [opened, synchronize]``.
_INLINE_TYPES_RE = re.compile(r"^ +types:\s*\[(?P<items>[^\]]*)\]\s*$")

#: Matches the opening of a block activity-type list, e.g. ``types:`` followed
#: by ``  - published`` lines.
_BLOCK_TYPES_RE = re.compile(r"^(?P<indent> +)types:\s*$")

#: Matches one entry of a block list, e.g. ``  - published``.
_BLOCK_ITEM_RE = re.compile(r"^ +- (?P<item>[a-z_]+)\s*$")


def _workflow_paths() -> list[Path]:
    return sorted(_WORKFLOW_DIR.glob("*.yml"))


def _pull_request_types(text: str) -> list[str] | None:
    """Return the activity types a workflow's ``pull_request`` trigger lists.

    ``None`` means the workflow has no ``pull_request`` trigger, or has one that
    lists no ``types`` and so takes GitHub's default. An empty list means the
    key is present and lists nothing, which is a different statement and is
    reported as such rather than folded into the default.

    Only the ``types`` belonging to ``pull_request`` are returned: a workflow may
    also carry ``push`` or ``release`` triggers with their own list (as
    ``pypi-publish-on-release.yml`` does), and those say nothing about pull
    requests.
    """
    current_trigger: str | None = None
    lines = text.splitlines()
    for index, line in enumerate(lines):
        trigger = _TRIGGER_RE.match(line)
        if trigger is not None and trigger.group("name") not in {"types"}:
            current_trigger = trigger.group("name")
            continue
        if current_trigger != "pull_request":
            continue
        inline = _INLINE_TYPES_RE.match(line)
        if inline is not None:
            return [item.strip() for item in inline.group("items").split(",") if item.strip()]
        if _BLOCK_TYPES_RE.match(line) is not None:
            items = []
            for follower in lines[index + 1 :]:
                item = _BLOCK_ITEM_RE.match(follower)
                if item is None:
                    break
                items.append(item.group("item"))
            return items
    return None


def _workflow_name(text: str) -> str:
    """Return a workflow's ``name:``, which is the ``github.workflow`` its concurrency group keys on."""
    match = re.search(r"^name:\s*(?P<name>.+?)\s*$", text, re.MULTILINE)
    return match.group("name") if match else ""


def test_the_scanner_finds_the_pull_request_workflows() -> None:
    """Guard the pins below against a regex that quietly matches nothing.

    Every assertion in this module is a statement about the workflows the
    scanner found. If it found none, they would all hold for the wrong reason,
    so the count is asserted before the properties are.
    """
    paths = _workflow_paths()
    assert paths, f"no workflows under {_WORKFLOW_DIR}"
    with_pull_request = [p.name for p in paths if "pull_request:" in p.read_text(encoding="utf-8")]
    assert _REQUIRED_CHECK_WORKFLOW.name in with_pull_request
    # Seven siblings plus the required-check workflow itself at the time of
    # writing. Asserted as a floor, not an equality: adding a pull_request
    # workflow is routine and must not fail this pin, while a scanner that stops
    # seeing them is exactly what this test is for.
    assert len(with_pull_request) >= 8, with_pull_request


def test_no_pull_request_trigger_subscribes_to_a_sha_invariant_type() -> None:
    """No workflow may be started by a pull_request event that changes no code.

    The required check is cancelled and restarted by such an event
    (``test_the_required_check_discards_its_in_flight_run``), and the restarted
    run reads the same ``head.sha``, so the only thing subscribing to one can
    produce is a second verdict on one commit -- and, measured on #1899, #1901
    and #1902, a CANCELLED check run that makes the roll-up state unstable.
    """
    offenders: dict[str, list[str]] = {}
    for path in _workflow_paths():
        types = _pull_request_types(path.read_text(encoding="utf-8"))
        if types is None:
            continue
        allowed = _SHA_CHANGING_TYPES | _INPUT_IS_NOT_THE_TREE.get(path.name, frozenset())
        extra = sorted(set(types) - allowed)
        if extra:
            offenders[path.name] = extra
    assert not offenders, (
        "these workflows are started by pull_request events that cannot change "
        f"the head sha: {offenders}. Such an event can only recompute a verdict "
        "already being computed, and it cancels the in-flight required check to "
        "do it. Omit 'types' to take GitHub's default "
        f"({sorted(_SHA_CHANGING_TYPES)})."
    )


def test_the_required_check_workflow_takes_the_default_types() -> None:
    """The required check states its trigger list by not restating it.

    The default is precisely :data:`_SHA_CHANGING_TYPES`, so an explicit list
    spelling out those three would pass the pin above while adding a second copy
    of a list that already exists -- and it is a copy in the one file where
    getting it wrong cancels a 27-minute run.
    """
    types = _pull_request_types(_REQUIRED_CHECK_WORKFLOW.read_text(encoding="utf-8"))
    assert types is None, (
        f"{_REQUIRED_CHECK_WORKFLOW.name} lists pull_request types explicitly ({types}); "
        "omit the key so there is one definition of which events can change a head sha"
    )


def test_an_exempt_workflow_cannot_cancel_the_required_check() -> None:
    """The premise every entry in :data:`_INPUT_IS_NOT_THE_TREE` rests on.

    An exemption is safe only while the exempt event cannot reach the required
    check, and that holds for two independent reasons which are both read back
    here: the required check does not subscribe to the event, and cancellation is
    scoped to a concurrency group keyed on the workflow's own name. If either
    stops being true, the exempt workflow starts discarding the required check's
    progress and this fails with the entry that did it.
    """
    required = _REQUIRED_CHECK_WORKFLOW.read_text(encoding="utf-8")
    required_types = set(_pull_request_types(required) or _SHA_CHANGING_TYPES)

    assert _INPUT_IS_NOT_THE_TREE, "the exemption table is empty; this pin has nothing to check"
    for name, exempt_types in _INPUT_IS_NOT_THE_TREE.items():
        path = _WORKFLOW_DIR / name
        assert path.exists(), f"{name} is exempt but does not exist"
        text = path.read_text(encoding="utf-8")

        assert not exempt_types & required_types, (
            f"{name} is exempt for {sorted(exempt_types)}, but the required check now subscribes to "
            f"{sorted(exempt_types & required_types)} too, so those events do discard its run"
        )
        assert _workflow_name(text) != _workflow_name(required), (
            f"{name} now shares the required check's workflow name, so it shares its concurrency group"
        )


def test_the_required_check_discards_its_in_flight_run() -> None:
    """Premise: this is why a redundant trigger is destructive rather than idle.

    Pins today's behaviour, not a wish. Were ``cancel-in-progress`` ever turned
    off, a redundant trigger would merely waste a runner instead of throwing
    away the required check's progress, and the pin above would be a
    cost-control rule rather than a correctness one. Either way the reasoning
    above needs to be re-read, which is what this failing would say.

    Only the pull-request operand is asserted. The fallback operand is the push
    side, which #2304 changed from ``github.ref`` to ``github.sha``, and pinning
    the whole expression here would put a second copy of that decision in a file
    that does not reason about it -- so the push half is asserted by rendering the
    group in ``tests/test_push_concurrency_group.py`` instead.
    """
    text = _REQUIRED_CHECK_WORKFLOW.read_text(encoding="utf-8")
    assert "cancel-in-progress: true" in text
    assert "group: ${{ github.workflow }}-${{ github.event.pull_request.number ||" in text, (
        "the concurrency group no longer keys on the pull request number, so two runs over one "
        "pull request need not share a group and the cost this module is about disappears; "
        "re-derive the pins above"
    )


def test_no_workflow_gates_on_draft_status() -> None:
    """Premise for dropping ``ready_for_review``: no draft's CI is skipped.

    Nothing in the fleet reads ``pull_request.draft``, so a draft pull request
    runs the same checks as any other and marking it ready re-runs a sha that
    has already been checked. If a draft skip is added, ``ready_for_review``
    becomes the event that starts the first real run for that sha and belongs in
    the trigger list of whatever workflow does the skipping -- so this pin
    failing is a prompt to re-derive the list, not to delete the pin.
    """
    gating = [p.name for p in _workflow_paths() if "draft" in p.read_text(encoding="utf-8")]
    assert not gating, (
        f"these workflows mention 'draft': {gating}. If one now skips draft pull requests, "
        "re-derive whether ready_for_review must start it"
    )


def test_an_exempt_workflow_cannot_cancel_its_own_run() -> None:
    """The second premise every entry in :data:`_INPUT_IS_NOT_THE_TREE` rests on.

    ``test_an_exempt_workflow_cannot_cancel_the_required_check`` establishes that
    an exempt event cannot reach ``pr-and-push.yml``'s concurrency group. That is
    necessary and not sufficient: the run an exempt event cancels first is the
    exempt workflow's own, which shares a workflow name and a pull request number
    with the run already in flight, so a group keyed on either holds both.

    Being exempt is what makes it reachable -- an exempt workflow is the only kind
    that can be started twice on one head sha -- and the cancelled context then
    sits on the live head rather than on a superseded one, where the roll-up reads
    it. Measured on #1722 and #2205 (#2216), each carrying its check as
    ``SUCCESS`` and ``CANCELLED`` at once with every other context ``SUCCESS``.

    Asserted of the table rather than of the single file in it, so a future
    exemption inherits the requirement instead of rediscovering it the same way.
    """
    assert _INPUT_IS_NOT_THE_TREE, "the exemption table is empty; this pin has nothing to check"

    offenders: dict[str, list[str]] = {}
    for name in _INPUT_IS_NOT_THE_TREE:
        path = _WORKFLOW_DIR / name
        assert path.exists(), f"{name} is exempt but does not exist"
        cancelling = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#") and line.strip().startswith("cancel-in-progress:")
        ]
        if cancelling:
            offenders[name] = cancelling

    assert not offenders, (
        f"these workflows are exempted for a sha-invariant activity type and still cancel in "
        f"progress: {offenders}. An exempt workflow can be started twice on one head sha, so its "
        "own in-flight run is what it cancels, and the cancelled check run is permanent, sits on "
        "the live head and drags statusCheckRollup.state to FAILURE from behind a same-named "
        "SUCCESS. Drop the concurrency block: a superseded head is the only case cancelling helps "
        "with, and a sha-invariant trigger does not produce one."
    )
