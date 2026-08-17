"""Preflight for the packages lerobot's ``train()`` requires at call time.

:meth:`strands_robots.training.lerobot.LerobotTrainer.validate` is documented as
a pure preflight whose empty list "means the spec is launchable", and
:class:`~strands_robots.training.base.Trainer` says an implementation SHOULD
check that "any backend-required input is present". It already reports an absent
lerobot -- ``tests/training/test_lerobot.py`` pins that as surfacing the problem
"rather than deferring the failure to the training launch".

Locating lerobot does not establish that lerobot can train. Its
``scripts.lerobot_train.train`` calls ``require_package(...)`` for packages that
are NOT module-scope imports of that module, so a spec lookup answers nothing
about them, and the first such call -- ``require_package("accelerate",
extra="training")`` -- is the opening statement of ``train()``. No
``strands-robots[lerobot]`` install supplies ``accelerate``: that extra is
exactly ``lerobot[feetech,dataset]``.

So the deferral the lerobot check exists to prevent still happened one dependency
out, and not merely as a late message: ``train()`` fails closed on a validate()
problem specifically so nothing is touched before a knowably-bad run, and with
the problem unreported it instead ran ``prepare()`` and the fresh-start hygiene
that removes a checkpoint-less ``output_dir`` -- deleting a directory for a run
that could never have started.

Absence is simulated by making the package unlocatable to
:func:`importlib.util.find_spec`, which is how an uninstalled one presents to
both this preflight and lerobot's own ``is_package_available``. The alternative
-- asserting only where the package genuinely happens to be missing -- would
leave the contract unverified in CI, where ``[all]`` supplies ``accelerate``
transitively (via ``[kimodo]``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from typing import Any

import pytest

from strands_robots.training import TrainSpec
from strands_robots.training.lerobot import LerobotTrainer

# The lerobot extra each package's remedy must name, per the require_package()
# call sites in lerobot.scripts.lerobot_train. Held as a literal so this file is
# a second opinion on the mapping rather than a mirror of the module's table.
_PACKAGE_EXTRA = {"accelerate": "training", "peft": "peft"}


@pytest.fixture(autouse=True)
def _restore_lerobot_availability_cache() -> Iterator[None]:
    """Leave lerobot's process-global availability cache as it was found.

    ``lerobot.utils.import_utils.require_package`` memoizes each answer in a
    module-level dict. A test here that hides a package can reach that memo (on a
    tree whose preflight does not stop first), which would strand a ``False`` for
    the rest of the session and make a later test see an installed package as
    absent. Snapshot and restore it so hiding a package cannot outlive the test.
    """
    try:
        from lerobot.utils import import_utils
    except ImportError:
        yield
        return
    saved = dict(import_utils._require_package_cache)
    yield
    import_utils._require_package_cache.clear()
    import_utils._require_package_cache.update(saved)


@pytest.fixture
def spec(tmp_path) -> TrainSpec:
    """A spec that validates clean on this machine: only a dependency can spoil it."""
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 10}))
    return TrainSpec(
        dataset_root=str(root),
        base_model="",
        output_dir=str(tmp_path / "out"),
        steps=200,
        global_batch_size=8,
        save_freq=100,
        extra={"policy_type": "act"},
    )


def _hide(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Make ``names`` unlocatable, exactly as an uninstalled package presents."""
    hidden = set(names)
    real = importlib.util.find_spec

    def lookup(name: str, package: str | None = None) -> Any:
        if name.split(".")[0] in hidden:
            return None
        return real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", lookup)


def _problems_naming(problems: list[str], package: str) -> list[str]:
    return [p for p in problems if f"'{package}'" in p]


class TestAnAbsentCallTimeDependencyIsReported:
    """The gate: a package lerobot's train() will demand is a preflight problem."""

    def test_absent_accelerate_is_reported(self, spec, monkeypatch) -> None:
        """accelerate is required by every lerobot run, so its absence is never launchable."""
        _hide(monkeypatch, "accelerate")
        problems = LerobotTrainer().validate(spec)
        assert _problems_naming(problems, "accelerate"), (
            "validate() reported no problem for an absent 'accelerate', so it called this "
            f"spec launchable; lerobot's train() requires it as its first statement. Got: {problems}"
        )

    def test_the_reported_remedy_names_the_lerobot_extra_that_supplies_it(self, spec, monkeypatch) -> None:
        """A remedy has to be runnable: it must name the extra that ships the package."""
        _hide(monkeypatch, "accelerate")
        (problem,) = _problems_naming(LerobotTrainer().validate(spec), "accelerate")
        assert f"pip install 'lerobot[{_PACKAGE_EXTRA['accelerate']}]'" in problem, (
            f"the remedy must name lerobot's own 'training' extra, which declares accelerate: {problem}"
        )

    def test_lora_reports_absent_peft(self, spec, monkeypatch) -> None:
        """method='lora' sets cfg.peft, which is exactly what lerobot gates peft on."""
        spec.method = "lora"
        spec.lora_r, spec.lora_alpha = 16, 32
        _hide(monkeypatch, "peft")
        (problem,) = _problems_naming(LerobotTrainer().validate(spec), "peft")
        assert f"pip install 'lerobot[{_PACKAGE_EXTRA['peft']}]'" in problem, problem


class TestNothingIsMutatedForAnUnlaunchableRun:
    """train() fails closed on a validate() problem, so the output_dir survives."""

    def test_an_existing_output_dir_survives(self, spec, monkeypatch, tmp_path) -> None:
        """The fresh-start hygiene must not delete a directory for a run that cannot start."""
        out = tmp_path / "out"
        out.mkdir()
        keepsake = out / "operator_notes.txt"
        keepsake.write_text("notes that predate the run\n")
        _hide(monkeypatch, "accelerate")

        result = LerobotTrainer().train(spec)

        assert result.status == "error"
        assert keepsake.is_file(), (
            "train() removed the output_dir for a spec whose dependency was absent at "
            "preflight time; validate() must report it so train() fails closed instead."
        )
        assert "accelerate" in (result.message or ""), result.message


class TestTheGateDoesNotOverreach:
    """Controls: these hold both before and after the preflight was added."""

    def test_a_clean_spec_is_still_clean(self, spec) -> None:
        """Every call-time package is installed here, so nothing is reported."""
        assert LerobotTrainer().validate(spec) == []

    def test_full_method_does_not_require_peft(self, spec, monkeypatch) -> None:
        """peft is conditional: a non-LoRA run never sets cfg.peft, so it must not be demanded."""
        _hide(monkeypatch, "peft")
        assert _problems_naming(LerobotTrainer().validate(spec), "peft") == []

    def test_an_absent_lerobot_reports_one_root_cause(self, spec, monkeypatch) -> None:
        """With neither installed the caller gets lerobot, not a second problem behind it."""
        _hide(monkeypatch, "lerobot", "accelerate")
        problems = LerobotTrainer().validate(spec)
        assert any("lerobot is not installed" in p for p in problems), problems
        assert _problems_naming(problems, "accelerate") == [], (
            "an install with no lerobot at all should name lerobot, not also the packages "
            f"its train() would have gone on to require: {problems}"
        )

    def test_the_probe_does_not_import_what_it_locates(self, monkeypatch) -> None:
        """validate() is read-only, so locating a package must not pay its import cost."""
        from strands_robots.training.lerobot import _module_available

        # A stdlib module the suite has no reason to have imported; dropped from
        # sys.modules first so the observation cannot be satisfied by a cache hit.
        monkeypatch.delitem(sys.modules, "colorsys", raising=False)
        assert _module_available("colorsys") is True
        assert "colorsys" not in sys.modules, "the availability probe imported the module it located"

    def test_a_probe_that_raises_answers_absent(self, monkeypatch) -> None:
        """A broken lookup on a report-problems path is 'cannot confirm', not a raise."""
        from strands_robots.training.lerobot import _module_available

        def boom(name: str, package: str | None = None) -> Any:
            raise ValueError("no __spec__")

        monkeypatch.setattr(importlib.util, "find_spec", boom)
        assert _module_available("accelerate") is False
