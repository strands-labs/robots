"""start_recording must say WHICH lerobot dependency is missing, not just that one is.

``has_lerobot_dataset()`` collapses every way importing
``lerobot.datasets.lerobot_dataset`` can fail into a single ``False``, and all
three backends' ``start_recording`` turned that ``False`` into one message:
"requires the lerobot extra: pip install 'strands-robots[lerobot]'". For a
partially-provisioned environment -- lerobot itself installed, but one of the
packages its dataset stack needs (``datasets``, ``pandas``, ``pyarrow``,
``av``) absent -- that instruction names a package the caller already has, so
they verify lerobot is installed and conclude the library is wrong about its own
dependency.

``strands_robots.policies.lerobot_local.molmoact2._factory_import_error``
already draws exactly this distinction for the policy-factory import, and its
docstring states the principle: "telling the caller to reinstall lerobot is a
dead end that sends a partially-provisioned env chasing the wrong remedy". These
pin the same distinction for the recording path.

The four causes and their four different remedies:

* lerobot absent                       -> install the extra;
* lerobot present, a dataset dep absent -> ``pip install 'lerobot[dataset]'``
  (plain ``pip install lerobot`` does not pull those in);
* lerobot present, module moved/renamed -> install an in-range lerobot;
* import failed with nothing missing    -> no install fixes it.
"""

from __future__ import annotations

import importlib.metadata
import re

import pytest

from strands_robots import dataset_recorder as dr

#: A representative dependency from each layer of lerobot's dataset stack.
_DATASET_DEPS = ("datasets", "pandas", "pyarrow", "av", "torchcodec")


@pytest.fixture
def lerobot_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the "lerobot itself is installed" half of the diagnosis.

    The three lerobot-present branches must be reachable whether or not lerobot
    is installed in the environment running the tests, so the contract holds
    everywhere rather than only where a real lerobot happens to be importable.
    """
    monkeypatch.setattr(dr, "_lerobot_installed", lambda: True)
    monkeypatch.setattr(dr, "lerobot_version", lambda: "0.6.0")


class TestMissingDatasetDependencyIsNamed:
    """The regression: a missing dataset dep must not read as "install lerobot"."""

    @pytest.mark.parametrize("dep", _DATASET_DEPS)
    def test_the_missing_package_is_named(self, dep: str, lerobot_present: None) -> None:
        exc = ModuleNotFoundError(f"No module named {dep!r}", name=dep)

        text = dr._describe_lerobot_import_failure(exc)

        assert dep in text, f"diagnosis does not name the missing package: {text!r}"

    @pytest.mark.parametrize("dep", _DATASET_DEPS)
    def test_the_remedy_installs_the_dataset_extra(self, dep: str, lerobot_present: None) -> None:
        exc = ModuleNotFoundError(f"No module named {dep!r}", name=dep)

        text = dr._describe_lerobot_import_failure(exc)

        # One command covers the whole stack: a caller who installed lerobot
        # without the extra is missing all of it, not just the package that
        # happened to be imported first.
        assert "pip install 'lerobot[dataset]'" in text

    @pytest.mark.parametrize("dep", _DATASET_DEPS)
    def test_it_does_not_say_lerobot_is_missing(self, dep: str, lerobot_present: None) -> None:
        exc = ModuleNotFoundError(f"No module named {dep!r}", name=dep)

        text = dr._describe_lerobot_import_failure(exc)

        # lerobot IS installed here, so any claim that it is not, or any
        # instruction to reinstall it on its own, is the misdiagnosis.
        assert "lerobot is not installed" not in text
        assert "strands-robots[lerobot]" not in text
        assert "0.6.0 is installed" in text, "the installed version anchors the diagnosis"


class TestEachCauseGetsItsOwnRemedy:
    """Four causes, four instructions - none may collapse into another."""

    def test_absent_lerobot_points_at_the_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dr, "_lerobot_installed", lambda: False)

        text = dr._describe_lerobot_import_failure(ModuleNotFoundError("No module named 'lerobot'", name="lerobot"))

        assert "lerobot is not installed" in text
        assert "strands-robots[lerobot]" in text

    def test_moved_module_points_at_the_supported_range(self, lerobot_present: None) -> None:
        # An in-place lerobot that renamed the module reports a name INSIDE
        # lerobot -- as does `cannot import name X from lerobot.datasets...`,
        # whose `name` is the module, not the symbol.
        text = dr._describe_lerobot_import_failure(
            ImportError(
                "cannot import name 'LeRobotDataset'",
                name="lerobot.datasets.lerobot_dataset",
            )
        )

        assert "lerobot.datasets.lerobot_dataset" in text
        assert "lerobot >= 0.6.0" in text
        # Not the transitive-dep remedy: the dataset extra cannot supply a
        # module that lerobot itself no longer exports.
        assert "lerobot[dataset]" not in text

    def test_unnamed_import_error_is_not_called_a_binary_conflict(self, lerobot_present: None) -> None:
        # An ImportError with no `name` still reports a failed import, so it must
        # not be told that nothing is missing and no install will help.
        text = dr._describe_lerobot_import_failure(ImportError("something went wrong"))

        assert "no module is missing" not in text
        assert "lerobot >= 0.6.0" in text

    def test_non_import_error_says_no_install_fixes_it(self, lerobot_present: None) -> None:
        # The numpy/pandas ABI mismatch this module's own comment describes:
        # every package is present, so an install instruction is a dead end.
        text = dr._describe_lerobot_import_failure(
            ValueError("numpy.dtype size changed, may indicate binary incompatibility")
        )

        assert "no module is missing" in text
        assert "reconcile the conflicting packages" in text
        assert "pip install" not in text


class TestPredicateAgreesWithReason:
    """``has_lerobot_dataset`` must be exactly "there is no reason"."""

    @pytest.mark.parametrize(
        "reason", [None, "lerobot is not installed (...)", "lerobot 0.6.0 is installed, but 'av' ..."]
    )
    def test_predicate_is_the_negation_of_a_reason(self, reason: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dr, "lerobot_dataset_import_error", lambda: reason)

        assert dr.has_lerobot_dataset() is (reason is None)


class TestSuccessfulProbeIsStillCached:
    """The caching contract the reason-bearing probe inherited."""

    def test_a_cached_success_short_circuits_without_importing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dr, "_HAS_LEROBOT_DATASET", [True])
        # A cached success must not re-probe, so a describer that would raise is
        # never reached.
        monkeypatch.setattr(
            dr, "_describe_lerobot_import_failure", lambda exc: pytest.fail("re-probed a cached success")
        )

        assert dr.lerobot_dataset_import_error() is None

    def test_a_failure_is_not_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A failed probe is deliberately re-attempted: caching False would
        # disable recording for the rest of the process once anything transiently
        # failed to import.
        monkeypatch.setattr(dr, "_HAS_LEROBOT_DATASET", [])
        calls: list[int] = []

        def _describe(exc: BaseException) -> str:
            calls.append(1)
            return "nope"

        monkeypatch.setattr(dr, "_describe_lerobot_import_failure", _describe)
        monkeypatch.setitem(dr.sys.modules, "lerobot.datasets.lerobot_dataset", None)

        assert dr.lerobot_dataset_import_error() == "nope"
        assert dr.lerobot_dataset_import_error() == "nope"
        assert len(calls) == 2, "a failed probe must be re-attempted, not cached"


class TestRemedyActuallySuppliesTheNamedPackages:
    """Every package the message names must really ship in ``lerobot[dataset]``.

    The message tells a caller that one command supplies the whole stack. If
    lerobot moves a package out of that extra, the instruction silently stops
    being true - and a caller following it is back to the dead end this
    diagnosis exists to remove.
    """

    def test_named_packages_ship_in_the_extra_the_message_recommends(self) -> None:
        pytest.importorskip("lerobot", reason="the extra's contents are read from lerobot's metadata")
        requirements = importlib.metadata.requires("lerobot") or []

        def resolve(extra: str, seen: frozenset[str] = frozenset()) -> set[str]:
            """Transitive closure of a lerobot extra (they reference each other)."""
            if extra in seen:
                return set()
            seen = seen | {extra}
            found: set[str] = set()
            for raw in requirements:
                marker = re.search(r"""extra\s*==\s*['"]([^'"]+)['"]""", raw)
                if not marker or marker.group(1) != extra:
                    continue
                spec = raw.split(";")[0].strip()
                nested = re.match(r"lerobot\[([^\]]+)\]", spec)
                if nested:
                    for sub in nested.group(1).split(","):
                        found |= resolve(sub.strip(), seen)
                else:
                    found.add(re.split(r"[<>=!\[ ]", spec, maxsplit=1)[0].lower())
            return found

        shipped = resolve("dataset")
        assert shipped, "could not read lerobot[dataset] from lerobot's metadata"

        named = [pkg.strip() for pkg in dr._LEROBOT_DATASET_PACKAGES.split(",")]
        assert named, "the hint names no packages"
        assert not set(named) - shipped, (
            f"the diagnosis names {sorted(set(named) - shipped)}, which lerobot[dataset] "
            f"does not install (it ships {sorted(shipped)})"
        )
