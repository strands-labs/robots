"""The device-connect source pin is measured in the test env, not announced.

``test-lint.yml`` redirects the two device-connect distributions to a git
source when a pull request touches ``strands_robots/device_connect/``. It does so
by exporting ``UV_OVERRIDE`` before the hatch test env is created, and it prints
``Pinned device-connect packages to <repo>@<ref>``.

Until #3222 that line was the only trace the pin left. Hatch creates its env
silently, so the job log carried no ``Resolved``/``Installed`` evidence for the
interpreter the suite runs in - while the *outer* ``pip install -e ".[all,dev]"``
into the runner interpreter, which reads no ``UV_OVERRIDE`` and installs nothing
the suite imports, printed ``Downloading device_connect_edge-0.2.5-...whl`` in
full view. The issue read that as the pin never being consulted. Measured, the
pin is consulted: a hatch ``installer = "uv"`` env created under ``UV_OVERRIDE``
carrying ``six @ git+https://github.com/benjaminp/six.git@main`` records::

    {"url": "https://github.com/benjaminp/six.git",
     "vcs_info": {"vcs": "git", "commit_id": "c8e39406...", "requested_revision": "main"}}

in ``six``'s ``direct_url.json``, while the same distribution in the runner
interpreter carries no such file. So the defect was not the pin but the absence
of any measurement of it - a claim CI could not support in either direction.

``scripts/check_device_connect_source_pin.py`` is that measurement, and the
workflow runs it between env creation and the suite. This module pins two
things: the script's verdict over every origin an installer can leave behind,
and the workflow's invocation of it - gated on the same condition as the pin,
placed after the step that creates the env and before the suite, and run
*through* ``hatch run``. The last is the one worth grading rather than trusting:
a bare ``python`` measures the runner interpreter, where the published wheel
lives, so it would fail on every pull request the pin applies to and report the
wrong cause. That is the reading #3222 made from the log, arriving as a check.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "check_device_connect_source_pin.py"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "test-lint.yml"

_REPO = "arm/device-connect"
_REF = "main"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_device_connect_source_pin", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load()


def _install_dist_info(site: Path, name: str, version: str, direct_url: dict[str, Any] | None) -> None:
    """Leave the metadata an installer leaves, and nothing else.

    ``importlib.metadata`` reads a ``*.dist-info`` directory off ``sys.path``, so
    a fabricated one exercises the same code path a real install does - including
    ``read_text`` returning ``None`` for a file the installer did not write, which
    is what a registry wheel looks like.
    """
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    if direct_url is not None:
        (info / "direct_url.json").write_text(json.dumps(direct_url))


def _git_record(url: str, ref: str) -> dict[str, Any]:
    return {
        "url": url,
        "vcs_info": {"vcs": "git", "commit_id": "c8e394065cd541a16c040515dc0afb85cf22a7c3", "requested_revision": ref},
    }


class TestTheVerdictReadsTheInstallersRecord:
    """Each origin an installer can leave is named, and only the pin passes."""

    @pytest.fixture
    def site(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        site = tmp_path / "site"
        site.mkdir()
        monkeypatch.syspath_prepend(str(site))
        return site

    def test_a_git_install_of_the_named_repo_and_ref_is_the_pin(self, site: Path) -> None:
        _install_dist_info(site, "probe-edge", "0.2.5", _git_record(f"https://github.com/{_REPO}.git", _REF))
        origin = mod.inspect("probe-edge", repo=_REPO, ref=_REF)
        assert origin.outcome == mod.PINNED
        assert origin.version == "0.2.5"
        assert "c8e394065cd5" in origin.detail, "the commit actually loaded is part of the evidence"

    def test_the_url_without_a_git_suffix_names_the_same_repository(self, site: Path) -> None:
        _install_dist_info(site, "probe-bare", "0.2.5", _git_record(f"https://github.com/{_REPO}", _REF))
        assert mod.inspect("probe-bare", repo=_REPO, ref=_REF).outcome == mod.PINNED

    def test_a_wheel_from_an_index_is_a_registry_wheel(self, site: Path) -> None:
        _install_dist_info(site, "probe-wheel", "0.2.5", None)
        origin = mod.inspect("probe-wheel", repo=_REPO, ref=_REF)
        assert origin.outcome == mod.REGISTRY
        assert "direct_url.json" in origin.detail

    @pytest.mark.parametrize(
        ("record", "why"),
        [
            (_git_record(f"https://github.com/{_REPO}.git", "v0.2.5"), "the revision is not the one announced"),
            (
                _git_record("https://github.com/someone-else/device-connect.git", _REF),
                "the repository is not the one announced",
            ),
            (
                {"url": "file:///tmp/device-connect", "dir_info": {"editable": True}},
                "a local path is not the source pin",
            ),
        ],
    )
    def test_a_different_source_is_named_rather_than_accepted(
        self, site: Path, record: dict[str, Any], why: str
    ) -> None:
        _install_dist_info(site, "probe-other", "0.2.5", record)
        origin = mod.inspect("probe-other", repo=_REPO, ref=_REF)
        assert origin.outcome == mod.OTHER, why
        assert record["url"] in origin.detail, "the report says what was found, not only that it was wrong"

    def test_an_absent_distribution_is_reported_as_absent(self, site: Path) -> None:
        origin = mod.inspect("probe-absent", repo=_REPO, ref=_REF)
        assert origin.outcome == mod.MISSING
        assert origin.version == ""

    def test_unparseable_metadata_is_not_read_as_the_pin(self) -> None:
        origin = mod.classify("probe", repo=_REPO, ref=_REF, version="0.2.5", direct_url_text="{not json")
        assert origin.outcome == mod.OTHER


class TestTheExitCodeIsTheVerdict:
    """Any distribution that is not the pin fails the step, and the report says which."""

    def _run(self, site: Path, *names: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(_SCRIPT),
                "--repo",
                _REPO,
                "--ref",
                _REF,
                *sum((["--distribution", n] for n in names), []),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(site), "PATH": "/usr/bin:/bin"},
        )

    def test_every_distribution_pinned_exits_zero(self, tmp_path: Path) -> None:
        _install_dist_info(tmp_path, "edge", "0.2.5", _git_record(f"https://github.com/{_REPO}.git", _REF))
        _install_dist_info(tmp_path, "tools", "0.1.2", _git_record(f"https://github.com/{_REPO}.git", _REF))
        result = self._run(tmp_path, "edge", "tools")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "::error" not in result.stdout

    def test_one_registry_wheel_fails_and_is_named(self, tmp_path: Path) -> None:
        _install_dist_info(tmp_path, "edge", "0.2.5", _git_record(f"https://github.com/{_REPO}.git", _REF))
        _install_dist_info(tmp_path, "tools", "0.1.2", None)
        result = self._run(tmp_path, "edge", "tools")
        assert result.returncode == 1
        assert "::error" in result.stdout
        assert "tools is registry-wheel" in result.stdout
        assert "edge is " not in result.stdout, "the distribution that IS the pin is not reported as a failure"

    def test_the_default_distributions_are_the_two_the_override_redirects(self) -> None:
        assert mod.DEFAULT_DISTRIBUTIONS == ("device-connect-edge", "device-connect-agent-tools")


# --- the workflow ---------------------------------------------------------------

_STEP_START = re.compile(r"^      - name: (?P<name>.+?)\s*$", re.MULTILINE)


def _steps(text: str) -> list[tuple[str, str]]:
    """Return ``(name, body)`` per step of the single job, in order.

    A step begins at a six-space ``- name:`` and runs to the next one. The job
    has one ``steps:`` list, so this is exact rather than approximate for this
    file; the assertion in :func:`test_the_workflow_has_the_steps_this_module_reads`
    is what says so.
    """
    starts = list(_STEP_START.finditer(text))
    steps = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        steps.append((match.group("name"), text[match.end() : end]))
    return steps


def _step(steps: list[tuple[str, str]], predicate: Any) -> tuple[int, str, str]:
    hits = [(i, n, b) for i, (n, b) in enumerate(steps) if predicate(n, b)]
    assert len(hits) == 1, [n for _, n, _ in hits]
    return hits[0]


def _field(body: str, key: str) -> str:
    match = re.search(rf"^        {key}: (?P<value>.+?)\s*$", body, re.MULTILINE)
    assert match, f"step has no `{key}:`"
    return match.group("value")


@pytest.fixture(scope="module")
def steps() -> list[tuple[str, str]]:
    return _steps(_WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pin(steps: list[tuple[str, str]]) -> tuple[int, str, str]:
    return _step(steps, lambda n, b: "dc-source-override.txt" in b and "UV_OVERRIDE=" in b)


@pytest.fixture(scope="module")
def verify(steps: list[tuple[str, str]]) -> tuple[int, str, str]:
    return _step(steps, lambda n, b: _SCRIPT.name in b)


def test_the_workflow_has_the_steps_this_module_reads(steps: list[tuple[str, str]]) -> None:
    names = [n for n, _ in steps]
    assert "Install dependencies" in names
    assert "Run lint" in names
    assert "Run tests" in names


class TestThePinIsVerifiedWhereTheSuiteRuns:
    """The verify step grades the claim the pin step made, in the env the suite uses."""

    def test_the_verify_step_is_gated_on_the_same_condition_as_the_pin(
        self, pin: tuple[int, str, str], verify: tuple[int, str, str]
    ) -> None:
        assert _field(verify[2], "if") == _field(pin[2], "if")

    def test_the_verify_step_runs_after_the_env_exists_and_before_the_suite(
        self, steps: list[tuple[str, str]], verify: tuple[int, str, str]
    ) -> None:
        lint = _step(steps, lambda n, b: re.search(r"^\s+run: hatch run lint\s*$", b, re.MULTILINE) is not None)
        tests = _step(steps, lambda n, b: "hatch run test " in b)
        assert lint[0] < verify[0] < tests[0], (
            "the hatch env is created by the first `hatch run`, which is `Run lint`; "
            "before it there is nothing to inspect, and after `Run tests` the verdict arrives "
            "27 minutes late"
        )

    def test_the_verify_step_runs_through_hatch(self, verify: tuple[int, str, str]) -> None:
        run = verify[2]
        assert re.search(rf"hatch run python scripts/{re.escape(_SCRIPT.name)}", run), run
        assert not re.search(rf"^\s*python3? scripts/{re.escape(_SCRIPT.name)}", run, re.MULTILINE), (
            "a bare interpreter measures the runner install, which holds the published wheel"
        )

    def test_the_verify_step_reads_the_repo_and_ref_the_pin_step_exported(
        self, pin: tuple[int, str, str], verify: tuple[int, str, str]
    ) -> None:
        for name in ("DEVICE_CONNECT_REPO", "DEVICE_CONNECT_SOURCE_REF"):
            assert re.search(rf'echo "{name}=\$\{{{name}\}}" >> "\$GITHUB_ENV"', pin[2]), (
                f"the pin step must hand {name} to later steps; a second `vars.` read would be a second copy of the claim"
            )
        assert '--repo "${DEVICE_CONNECT_REPO}"' in verify[2]
        assert '--ref "${DEVICE_CONNECT_SOURCE_REF}"' in verify[2]

    def test_the_script_grades_every_distribution_the_override_redirects(self, pin: tuple[int, str, str]) -> None:
        redirected = re.findall(r"^\s+(?P<name>[a-z0-9-]+) @ git\+https://", pin[2], re.MULTILINE)
        assert redirected, "the override heredoc names no distribution"
        assert tuple(redirected) == mod.DEFAULT_DISTRIBUTIONS, (
            "a distribution added to the override without being added to the script's defaults "
            "would be pinned and never verified"
        )
