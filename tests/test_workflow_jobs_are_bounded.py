"""Every workflow job that runs on a runner declares a ``timeout-minutes``.

A GitHub Actions job with no ``timeout-minutes`` inherits a **6-hour** default.
Nothing in this repository set one: at ``79b1582`` all 17 jobs across 14
workflow files were unbounded, including ``Test and Lint`` -- the one required
check every merge waits on.

The cost was measured, not projected.  On #2236 (``APPROVED``, ``MERGEABLE``,
every other check green) ``call-test-lint`` read ``IN_PROGRESS`` for 103
minutes.  It was not slow or queued: pytest had already printed its verdict 18
minutes in and the process then refused to exit.  From the attempt-1 log of run
``31676048797``::

    07:04:58  step starts
    07:23:19  ==== 1 failed, 22095 passed, 257 skipped in 1093.27s (0:18:13) ====
              <-- 81 minutes 57 seconds of absolutely nothing -->
    08:44:16  ##[error]The operation was canceled.        # a human, not a timeout

The two log lines are adjacent.  Everything a reviewer or a merge queue needed
was on disk at 07:23:19, and the only thing that ended the run was somebody
cancelling it; left alone the job would have held the runner until 13:01Z.  See
#2239 for the full account, including why the hang is *after* the last test
(pytest-timeout bounds tests, and a ``ThreadPoolExecutor`` worker parked by a
timed-out future blocks interpreter exit through ``_python_exit``, which
``shutdown(wait=False)`` does not prevent).

**None of the existing bounds could have caught it.**  ``--timeout=120`` in
``addopts`` bounds tests, and this hang is outside any test.  The roll-up
reports ``PENDING``, which is indistinguishable from a healthy long run -- so a
red verdict was rendered as "still running", the one direction that misleads a
reviewer into waiting.  A job-level bound is the only mechanism that sits
outside the interpreter, which is why it fixes the class rather than the one
test that exposed it.

The bounds themselves are sized from observed durations over the last 300
workflow runs, not guessed::

    workflow / job                    n    p50 min   max min   bound
    Test and Lint / test-lint        20      25.2      26.5      45
    CodeQL / analyze                 49       3.8       4.1      20
    Agent API Check                   5       0.9       1.1      10
    every other gate job          6..52      0.1       0.6      10

The suite's band is tight (19-27 min, and zero completed runs over the sampled
300 exceeded 45), so 45 is ~1.7x the observed ceiling: it costs a healthy run
nothing while converting a six-hour silent stall into a 45-minute honest
failure.

Two exemptions, both structural rather than discretionary, and each pinned
below so it cannot quietly widen into a hole:

* **A job that calls a reusable workflow** (``uses:`` at job level) accepts no
  ``timeout-minutes`` key -- the bound belongs to the job inside the called
  workflow.  Both callers here delegate to ``test-lint.yml``, whose own job is
  bounded, so the exemption costs no coverage.  That delegation is asserted,
  not assumed.
* **A job gated on a deployment ``environment:``** can legitimately sit waiting
  on a human approval, so a wall-clock bound there would encode a guess about
  reviewer latency instead of a fact about code.  Both such jobs are the
  ``deploy`` steps of ``docs.yml`` and ``pypi-publish-on-release.yml``.

Parsing is deliberately line-based rather than via ``yaml``: ``tests/`` is
type-checked under ``ignore_missing_imports = false`` and ``types-PyYAML`` is
not a dev dependency, so importing it would either fail ``mypy`` or require a
dependency change and a ``uv.lock`` relock to satisfy the lockfile-parity gate
-- a disproportionate diff for reading two indent levels.  The existing
workflow contract pins (``tests/test_codeql_query_filters.py``,
``tests/test_lockfile_parity_gate.py``) read their YAML the same way.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"

#: A bound has to be meaningfully tighter than the 6-hour default to be a bound
#: at all. Nothing here is near it -- the widest is the 45-minute suite -- so a
#: value above this ceiling means someone silenced this guard rather than sized
#: a job.
_CEILING_MINUTES = 60

#: The suite bound must clear the measured 19-27 minute band with room to spare,
#: or it converts a healthy run into a red one. The lower edge is ~1.1x the
#: observed maximum of 26.5.
_SUITE_FLOOR_MINUTES = 30

_JOBS_KEY = re.compile(r"^jobs:\s*$")
_TOP_LEVEL_KEY = re.compile(r"^\S")
_JOB_HEADER = re.compile(r"^ {2}([A-Za-z0-9_-]+):\s*$")
_JOB_KEY = re.compile(r"^ {4}([A-Za-z0-9_-]+):(.*)$")


class Job:
    """One ``jobs.<job_id>`` block, reduced to the keys this contract reads."""

    def __init__(self, workflow: str, job_id: str, keys: dict[str, str]) -> None:
        self.workflow = workflow
        self.job_id = job_id
        self.keys = keys

    @property
    def ref(self) -> str:
        return f"{self.workflow}:{self.job_id}"

    @property
    def calls_a_reusable_workflow(self) -> bool:
        """``uses:`` at *job* level, which is the reusable-workflow call form."""
        return "uses" in self.keys

    @property
    def is_environment_gated(self) -> bool:
        return "environment" in self.keys

    @property
    def timeout_minutes(self) -> int | None:
        """The declared bound, or ``None`` when absent or not a literal int.

        An expression (``${{ ... }}``) reads as absent on purpose: this guard can
        only vouch for a value it can compare against the measured band.
        """
        raw = self.keys.get("timeout-minutes")
        if raw is None:
            return None
        try:
            return int(raw.split("#", 1)[0].strip())
        except ValueError:
            return None

    def __repr__(self) -> str:  # pragma: no cover - test ids only
        return self.ref


class _ParsedWorkflow(NamedTuple):
    path: Path
    jobs: list[Job]


def _parse(path: Path) -> _ParsedWorkflow:
    """Collect ``jobs.<job_id>`` and its 4-space-indented scalar keys."""
    jobs: list[Job] = []
    lines = path.read_text().splitlines()
    in_jobs = False
    current: Job | None = None

    for line in lines:
        if _JOBS_KEY.match(line):
            in_jobs = True
            current = None
            continue
        if not in_jobs:
            continue
        if _TOP_LEVEL_KEY.match(line):
            # A sibling of `jobs:` ends the section.
            in_jobs = False
            current = None
            continue

        header = _JOB_HEADER.match(line)
        if header:
            current = Job(path.name, header.group(1), {})
            jobs.append(current)
            continue

        key = _JOB_KEY.match(line)
        if key and current is not None:
            # First occurrence wins; nested mappings (`environment:` with a
            # `name:` under it) are recorded by key presence, which is all this
            # contract asks of them.
            current.keys.setdefault(key.group(1), key.group(2))

    return _ParsedWorkflow(path, jobs)


_PARSED = [_parse(p) for p in sorted(_WORKFLOWS.glob("*.yml"))]
_ALL_JOBS = [job for parsed in _PARSED for job in parsed.jobs]
_RUNNER_JOBS = [j for j in _ALL_JOBS if not j.calls_a_reusable_workflow and not j.is_environment_gated]


class TestTheParserActuallySeesTheTree:
    """A structural guard that parses nothing passes vacuously. Pin that it does not."""

    def test_workflows_are_found(self) -> None:
        assert _WORKFLOWS.is_dir(), f"{_WORKFLOWS} is missing"
        assert _PARSED, "no workflow files were parsed"

    @pytest.mark.parametrize("parsed", _PARSED, ids=lambda p: p.path.name)
    def test_every_workflow_yields_at_least_one_job(self, parsed: _ParsedWorkflow) -> None:
        """Every workflow declares jobs, so an empty list means the parser broke."""
        assert parsed.jobs, f"{parsed.path.name} parsed to zero jobs"

    def test_the_bounded_set_is_not_empty(self) -> None:
        assert _RUNNER_JOBS, "every job was classified as exempt, so nothing below is checked"


class TestEveryRunnerJobIsBounded:
    """The contract: a job that executes code on a runner cannot be unbounded."""

    @pytest.mark.parametrize("job", _RUNNER_JOBS, ids=lambda j: j.ref)
    def test_a_timeout_is_declared(self, job: Job) -> None:
        assert job.timeout_minutes is not None, (
            f"{job.ref} declares no literal timeout-minutes, so it inherits the 6-hour "
            f"default and a hung step holds the runner silently (#2239)"
        )

    @pytest.mark.parametrize("job", _RUNNER_JOBS, ids=lambda j: j.ref)
    def test_the_bound_is_tighter_than_the_default(self, job: Job) -> None:
        """A nominal bound near the 6-hour default is not a bound."""
        minutes = job.timeout_minutes
        assert minutes is not None and 0 < minutes <= _CEILING_MINUTES, (
            f"{job.ref} declares timeout-minutes: {minutes}, outside 1..{_CEILING_MINUTES}; "
            f"the widest job in this tree is the 45-minute suite"
        )


class TestTheExemptionsAreStructuralNotDiscretionary:
    """Each exemption is a property of the job, and neither leaves the suite unbounded."""

    @pytest.mark.parametrize("job", [j for j in _ALL_JOBS if j.is_environment_gated], ids=lambda j: j.ref)
    def test_an_unbounded_job_is_gated_on_a_human_decision(self, job: Job) -> None:
        """Only a deployment gate earns the exemption -- its wait is not code running."""
        assert job.is_environment_gated
        assert not job.calls_a_reusable_workflow, (
            f"{job.ref} is both a reusable-workflow call and environment-gated, which the "
            f"exemption reasoning below does not cover"
        )

    @pytest.mark.parametrize("job", [j for j in _ALL_JOBS if j.calls_a_reusable_workflow], ids=lambda j: j.ref)
    def test_a_caller_declares_no_bound_of_its_own(self, job: Job) -> None:
        """``timeout-minutes`` is not a valid key on a reusable-workflow call."""
        assert job.timeout_minutes is None, (
            f"{job.ref} calls a reusable workflow and also declares timeout-minutes, which "
            f"GitHub does not accept there; bound the job inside the called workflow instead"
        )

    @pytest.mark.parametrize("job", [j for j in _ALL_JOBS if j.calls_a_reusable_workflow], ids=lambda j: j.ref)
    def test_a_caller_delegates_to_a_workflow_whose_jobs_are_bounded(self, job: Job) -> None:
        """The exemption costs no coverage only if the callee is bounded. Assert it."""
        target = job.keys["uses"].split("#", 1)[0].strip()
        if not target.startswith("./"):
            pytest.skip(f"{job.ref} calls {target}, which is outside this repository")

        called = _REPO_ROOT / target[2:]
        assert called.is_file(), f"{job.ref} calls {target}, which does not exist"

        callee_jobs = _parse(called).jobs
        assert callee_jobs, f"{target} parsed to zero jobs"
        for callee in callee_jobs:
            assert callee.timeout_minutes is not None, (
                f"{job.ref} cannot carry a bound itself, and {target}:{callee.job_id} declares "
                f"none either, so the required check is unbounded end to end (#2239)"
            )


class TestTheSuiteBoundClearsTheMeasuredBand:
    """The one required check is the job the incident actually happened on."""

    def test_the_suite_job_is_bounded_above_its_observed_ceiling(self) -> None:
        suite = [j for j in _ALL_JOBS if j.workflow == "test-lint.yml" and j.job_id == "test-lint"]
        assert len(suite) == 1, "test-lint.yml:test-lint is the required check and must exist"

        minutes = suite[0].timeout_minutes
        assert minutes is not None, "the required check is unbounded (#2239)"
        assert _SUITE_FLOOR_MINUTES <= minutes <= _CEILING_MINUTES, (
            f"the suite bound is {minutes} min; it must clear the measured 19-27 min band "
            f"(max 26.5 over 20 runs) without approaching the 6-hour default"
        )
