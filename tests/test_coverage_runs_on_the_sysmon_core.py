"""The suite measures coverage with PEP 669, not with a C trace function.

``[tool.pytest.ini_options].addopts`` carries ``--cov=strands_robots``, so every
run of this suite is an instrumented one and the cost of *measuring* is part of
the suite's wall clock. Measured over 17,102 tests
(``tests/{drivers,mesh,policies,training,rendering}``), same selection, same
reports, one machine, only the core varying:

===================== ============ ==================
core                  wall clock   cost of measuring
===================== ============ ==================
none (no ``--cov``)      321.19 s   -
``ctrace`` (default)     567.36 s   +76.6%
``sysmon``               347.47 s    +8.2%
===================== ============ ==================

The default core was therefore spending more time than the tests it measured.
All three runs agreed on the verdict (4 failed, 17102 passed, 36 skipped) and
the two instrumented runs agreed on the measurement: the same 303 files, the
same 51,904 statements, the same 605 exclusions, 56% either way.

What is pinned here is the *selection*, behaviourally rather than as a string in
a file: coverage.py installs itself either as a ``sys.monitoring`` tool or as a
trace function, and which one it chose is readable from the standard library.
The control below derives its config from this repository's own by deleting the
one setting, so a config that stops asking for the core is reported as the
trace-function run it would become rather than passing on the literal's absence.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

#: Start coverage under the given config and report how it installed itself.
#: Run out of process: this suite is itself under coverage, and a second
#: in-process ``Coverage`` would contend for the same monitoring tool id.
_PROBE = """
import json, sys, coverage
cov = coverage.Coverage(config_file=sys.argv[1])
cov.start()
installed = {
    "monitoring_tool": sys.monitoring.get_tool(sys.monitoring.COVERAGE_ID),
    "has_trace_function": sys.gettrace() is not None,
}
cov.stop()
print(json.dumps(installed))
"""


def _coverage_config() -> dict[str, object]:
    """This repository's ``[tool.coverage.run]`` table."""
    tool = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["tool"]
    return dict(tool.get("coverage", {}).get("run", {}))


def _how_coverage_installed_itself(config_file: Path) -> dict[str, object]:
    """Whether coverage.py became a monitoring tool or a trace function."""
    # The environment variable outranks the config file, so drop it: the
    # question is what the *file* selects.
    env = {k: v for k, v in os.environ.items() if k != "COVERAGE_CORE"}
    done = subprocess.run(
        [sys.executable, "-c", _PROBE, str(config_file)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return dict(json.loads(done.stdout))


def _config_without_the_core(tmp_path: Path) -> Path:
    """This repository's config with the one setting under test removed."""
    lines = _PYPROJECT.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if line.strip() != 'core = "sysmon"']
    # Compare counts, not the joined text: a trailing-newline difference would
    # make a text comparison true even when nothing was removed.
    assert len(kept) < len(lines), (
        'the config no longer declares core = "sysmon", so the control removed '
        "nothing and would pass against the very default it exists to catch"
    )
    # coverage.py reads a ``[tool.coverage.*]`` table only from a file named
    # pyproject.toml, so the control has to keep the name.
    target = tmp_path / "pyproject.toml"
    target.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return target


class TestTheCoreIsSelected:
    """The declared core is the one coverage.py installs."""

    def test_the_repo_config_installs_coverage_as_a_monitoring_tool(self) -> None:
        installed = _how_coverage_installed_itself(_PYPROJECT)
        assert installed["monitoring_tool"] == "coverage.py", (
            f"coverage did not register as a sys.monitoring tool: {installed}; "
            'expected [tool.coverage.run] core = "sysmon" to select the PEP 669 core'
        )
        assert installed["has_trace_function"] is False, (
            f"a trace function is still installed: {installed}; the PEP 669 core "
            "replaces it, so both being present means the core was not selected"
        )

    def test_dropping_the_setting_falls_back_to_a_trace_function(self, tmp_path: Path) -> None:
        """The control: without the setting this is the slow run measured above."""
        installed = _how_coverage_installed_itself(_config_without_the_core(tmp_path))
        assert installed["monitoring_tool"] is None, (
            f"a config with no core still registered a monitoring tool: {installed}; "
            "then the setting is not what selects the core and this pin proves nothing"
        )
        assert installed["has_trace_function"] is True, (
            f"a config with no core installed no trace function either: {installed}"
        )


class TestThePremisesTheSettingRestsOn:
    """The two facts that make the measurement above the suite's own cost."""

    def test_coverage_is_on_the_suites_own_hot_path(self) -> None:
        """``--cov`` is in ``addopts``, so every run pays the core's cost."""
        addopts = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]["addopts"]
        assert "--cov=strands_robots" in addopts, (
            f"addopts no longer collects coverage ({addopts!r}); the core is then a "
            "setting for ad-hoc runs rather than for this suite's wall clock"
        )

    @pytest.mark.parametrize("setting", ["branch", "concurrency"])
    def test_the_equivalence_was_measured_without_it(self, setting: str) -> None:
        """The measured agreement covers statements collected single-threadedly.

        Both settings change what the core has to track, and neither was part of
        the run that established the two cores report the same numbers. Turning
        either on is a reason to re-measure, not a reason to keep this pin.
        """
        assert setting not in _coverage_config(), (
            f"[tool.coverage.run] now declares {setting!r}, which the sysmon/ctrace "
            "equivalence was not measured with; re-measure before widening this pin"
        )
