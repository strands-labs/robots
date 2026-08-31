"""The torch verdict is about the code the build carries, not about the driver.

``torch.cuda.is_available()`` answers about the driver, so it is ``True`` on a
host whose torch build carries no code for the GPU that driver reports. Every
CUDA kernel on such an install is refused by torch itself - its own words are
"not compatible with the current PyTorch installation" - and the only signal is a
``warnings.warn`` on the first CUDA call: it goes to stderr while the doctor's
table goes to stdout, and it is shown once per process, so anything that touched
CUDA earlier consumes it. ``check_cuda`` therefore passed, naming the device and
the version, beside a final "All checks passed".

The architectures below are written out rather than imported, so these cases
grade the shipped behaviour against a second opinion. ``BUILD_WITHOUT_THOR`` is
the pair an Ampere/Hopper wheel carries; ``ORIN_ARCH`` on ``AMPERE_ONLY`` is the
pair the coarse rule and the interval table disagree about, which is what makes
reading torch's table rather than re-deriving a rule load-bearing.
"""

from __future__ import annotations

import sys
import warnings
from types import SimpleNamespace
from typing import Any

import pytest

# The architecture an NVIDIA Thor's driver reports, and a build that predates it.
THOR_ARCH = 110
BUILD_WITHOUT_THOR = (80, 90)
BUILD_WITH_THOR = (80, 90, 100, THOR_ARCH, 120)

# A Jetson Orin, and a build carrying Ampere code. torch's interval table excludes
# 8.7 from what ``sm_80`` code supports; the coarse major-version rule admits it.
ORIN_ARCH = 87
AMPERE_ONLY = (80,)

# CUDA versions and the architectures their torch release carries, in the shape
# torch ships for its own remedy.
RELEASES = {"12.8": {80, 90, 100, 120}, "13.0": {80, 90, 100, THOR_ARCH, 120}}
THOR_RELEASE = "13.0"

TORCH_VERSION = "2.11.0+cu130"
DEVICE_NAME = "NVIDIA Thor"


def _interval_rule(device_cc: int, code_cc: int) -> bool:
    """torch's interval rule for the architectures these cases use.

    Read off torch's own ``DEVICE_REQUIREMENT``: code carrying ``sm_80`` supports
    ``>=8.0,<9.0 except {8.7}``, and every other entry used here supports its own
    major version. ``TestAgainstTheInstalledTorch`` grades this against the real
    predicate, so the two cannot drift.
    """
    if code_cc == 80:
        return 80 <= device_cc < 90 and device_cc != ORIN_ARCH
    return code_cc // 10 == device_cc // 10


def _extract(arch_string: str) -> int:
    """torch's own spelling of an arch-list entry, as its parser reads it."""
    base = arch_string.split("_", maxsplit=2)[1]
    return int(base.removesuffix("a").removesuffix("f"))


def _install_torch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    capability: tuple[int, int] | None,
    arch_list: list[str] | None = None,
    interval_table: bool = True,
    releases: dict[str, set[int]] | None = RELEASES,
    raising_arch_list: bool = False,
) -> None:
    """Make torch report ``capability`` for device 0 and ``arch_list`` for its build.

    ``interval_table=False`` stands in for an install too old to expose the
    predicate torch's capability check consults, which is the one degradation the
    supported floor (``torch>=2.0.0``) still admits.
    """

    def _arch_list() -> list[str]:
        if raising_arch_list:
            raise RuntimeError("this build reports no arch list")
        return list(arch_list or [])

    cuda: Any = SimpleNamespace(
        is_available=lambda: capability is not None,
        get_device_capability=lambda _index: capability,
        get_device_name=lambda _index: DEVICE_NAME,
        get_arch_list=_arch_list,
        _extract_arch_version=_extract,
    )
    if interval_table:
        cuda._code_compatible_with_device = _interval_rule
    if releases is not None:
        cuda.PYTORCH_RELEASES_CODE_CC = releases
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, __version__=TORCH_VERSION))


def _thor_on_an_older_build(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> str:
    """The verdict for a Thor whose torch build predates it."""
    from strands_robots.doctor import check_torch_arch

    _install_torch(
        monkeypatch,
        capability=(11, 0),
        arch_list=[f"sm_{arch}" for arch in BUILD_WITHOUT_THOR],
        **kwargs,
    )
    return check_torch_arch()


class TestABuildWithoutTheDevicesCodeIsRefused:
    """A build that carries no code the device can run is reported, not tolerated."""

    def test_a_build_that_carries_no_code_for_the_device_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _thor_on_an_older_build(monkeypatch)
        assert "FAIL" in result
        assert f"sm_{THOR_ARCH}" in result

    def test_the_refusal_names_the_architectures_the_build_carries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The reader has to see which build they have, not only which device."""
        result = _thor_on_an_older_build(monkeypatch)
        for arch in BUILD_WITHOUT_THOR:
            assert f"sm_{arch}" in result

    def test_the_refusal_attributes_the_verdict_to_torchs_own_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The verdict is torch's, so the wording says so rather than claiming a symptom."""
        assert "not compatible" in _thor_on_an_older_build(monkeypatch)

    def test_the_remedy_names_a_release_from_torchs_own_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Which CUDA release covers this device is torch's answer, not a guess."""
        result = _thor_on_an_older_build(monkeypatch)
        assert THOR_RELEASE in result
        assert "pytorch.org/get-started/locally" in result

    def test_the_remedy_omits_a_release_that_does_not_cover_the_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Naming every release would send the reader after one that changes nothing."""
        assert "12.8" not in _thor_on_an_older_build(monkeypatch)

    def test_a_build_with_no_covering_release_still_offers_the_install_page(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Jetson wheels are not on PyPI, so torch's table names no release for an Orin."""
        from strands_robots.doctor import check_torch_arch

        _install_torch(monkeypatch, capability=(8, 7), arch_list=[f"sm_{a}" for a in AMPERE_ONLY])
        result = check_torch_arch()
        assert "FAIL" in result
        assert "pytorch.org/get-started/locally" in result


class TestThePairTheCoarseRuleWouldAdmit:
    """The interval table is preferred because a coarser rule disagrees with it."""

    def test_an_orin_on_an_ampere_only_build_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_torch_arch

        _install_torch(monkeypatch, capability=(8, 7), arch_list=[f"sm_{a}" for a in AMPERE_ONLY])
        result = check_torch_arch()
        assert "FAIL" in result
        assert f"sm_{ORIN_ARCH}" in result

    def test_the_coarse_major_version_rule_would_admit_that_pair(self) -> None:
        """Premise: the two rules really do disagree, so reading torch's is load-bearing."""
        coarse = any(arch // 10 == ORIN_ARCH // 10 for arch in AMPERE_ONLY)
        assert coarse is True
        assert _interval_rule(ORIN_ARCH, AMPERE_ONLY[0]) is False

    def test_the_fallback_rule_is_the_coarse_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An install without the table gets the rule torch's cubin check documents.

        That rule admits the Orin pair, so this passes where the case above fails -
        the difference between the two verdicts is exactly what the table buys.
        """
        from strands_robots.doctor import check_torch_arch

        _install_torch(
            monkeypatch,
            capability=(8, 7),
            arch_list=[f"sm_{a}" for a in AMPERE_ONLY],
            interval_table=False,
        )
        assert "PASS" in check_torch_arch()

    def test_the_fallback_still_refuses_a_different_major_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The fallback is coarser, not absent."""
        assert "FAIL" in _thor_on_an_older_build(monkeypatch, interval_table=False)


class TestABuildThatCoversTheDeviceIsAccepted:
    """The verdict turns on the build's own architectures, so a matching build passes."""

    def test_a_build_carrying_the_devices_arch_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_torch_arch

        _install_torch(monkeypatch, capability=(11, 0), arch_list=[f"sm_{a}" for a in BUILD_WITH_THOR])
        result = check_torch_arch()
        assert "PASS" in result
        assert f"sm_{THOR_ARCH}" in result

    def test_an_older_gpu_on_that_build_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The refusal is about this device, not about the build's age."""
        from strands_robots.doctor import check_torch_arch

        _install_torch(monkeypatch, capability=(9, 0), arch_list=[f"sm_{a}" for a in BUILD_WITHOUT_THOR])
        assert "PASS" in check_torch_arch()

    def test_a_ptx_entry_counts_as_an_architecture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A build can ship ``compute_NN`` instead of ``sm_NN``, and torch reads both."""
        from strands_robots.doctor import check_torch_arch

        _install_torch(monkeypatch, capability=(11, 0), arch_list=[f"compute_{THOR_ARCH}"])
        assert "PASS" in check_torch_arch()

    def test_an_architecture_specific_suffix_is_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """torch appends ``a``/``f`` for architecture-specific code."""
        from strands_robots.doctor import check_torch_arch

        _install_torch(monkeypatch, capability=(9, 0), arch_list=["sm_90a"])
        assert "PASS" in check_torch_arch()


class TestTheCheckDeclinesWhenThereIsNothingToCompare:
    """No device, or no torch, is not a failure - there is no disagreement to report."""

    def test_no_cuda_device_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_torch_arch

        _install_torch(monkeypatch, capability=None, arch_list=[f"sm_{a}" for a in BUILD_WITHOUT_THOR])
        assert "SKIP" in check_torch_arch()

    def test_torch_absent_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """torch ships in the extras that run a policy, so a base install has nothing to ask."""
        from strands_robots.doctor import check_torch_arch

        monkeypatch.setitem(sys.modules, "torch", None)
        assert "SKIP" in check_torch_arch()

    def test_a_build_reporting_no_architectures_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A CPU-only build carries no CUDA code, which ``check_cuda`` already reports."""
        from strands_robots.doctor import check_torch_arch

        _install_torch(monkeypatch, capability=(11, 0), arch_list=[])
        assert "SKIP" in check_torch_arch()

    def test_an_arch_list_that_raises_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_torch_arch

        _install_torch(monkeypatch, capability=(11, 0), raising_arch_list=True)
        assert "SKIP" in check_torch_arch()

    def test_a_build_without_torchs_remedy_table_still_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The table is the remedy's source, not the verdict's."""
        assert "FAIL" in _thor_on_an_older_build(monkeypatch, releases=None)


class TestTheDoctorRunsTheCheckAndFailsOnIt:
    """A verdict nothing runs is decoration, so the wiring is graded too."""

    def test_a_refusal_from_the_check_fails_the_doctor(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from strands_robots import doctor

        sentinel = "  FAIL  torch-arch-sentinel"
        monkeypatch.setattr(doctor, "check_torch_arch", lambda: sentinel)
        exit_code = doctor.run_doctor()
        out = capsys.readouterr().out
        assert sentinel in out
        assert "All checks passed" not in out
        assert exit_code == 1


class TestWhatIsDeliberatelyLeftAlone:
    """The division of labour between the two CUDA checks, recorded either way."""

    def test_check_cuda_still_answers_about_the_driver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """It reports availability and the device name, which the driver decides.

        Its verdict is unchanged for the build the new check refuses: that is the
        division of labour rather than an oversight, and moving it would drop the
        one line that names the device on a healthy host.
        """
        from strands_robots.doctor import check_cuda

        _install_torch(monkeypatch, capability=(11, 0), arch_list=[f"sm_{a}" for a in BUILD_WITHOUT_THOR])
        result = check_cuda()
        assert "PASS" in result
        assert DEVICE_NAME in result

    def test_the_warp_verdict_reads_only_warps_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Warp's arch check is unaffected by what torch's build carries."""
        from strands_robots.doctor import check_warp_arch

        _install_torch(monkeypatch, capability=(11, 0), arch_list=[f"sm_{a}" for a in BUILD_WITHOUT_THOR])
        warp: Any = SimpleNamespace(
            is_cuda_available=lambda: True,
            get_cuda_device_count=lambda: 1,
            get_cuda_supported_archs=lambda: [80, 90, 100, THOR_ARCH],
            get_device=lambda _alias: SimpleNamespace(arch=THOR_ARCH),
            get_cuda_toolkit_version=lambda: (13, 0),
            get_cuda_driver_version=lambda: (13, 0),
        )
        monkeypatch.setitem(sys.modules, "warp", warp)
        assert "PASS" in check_warp_arch()


class TestAgainstTheInstalledTorch:
    """The stand-ins above agree with a real torch about what it reports."""

    def test_the_installed_build_carries_code_for_this_devices_arch(self) -> None:
        pytest.importorskip("torch", reason="torch ships in the extras that run a policy")
        from strands_robots.doctor import _driver_compute_arch, _torch_build_supports, _torch_cuda_report

        device_arch = _driver_compute_arch()
        report = _torch_cuda_report()
        if device_arch is None or report is None:
            pytest.skip("no CUDA device for torch to report on")
        build_archs, version = report
        assert _torch_build_supports(device_arch, build_archs), (
            f"the installed torch {version} carries code for {sorted(set(build_archs))} "
            f"and this device reports sm_{device_arch}"
        )

    def test_a_real_torch_answers_every_call_the_stand_ins_make(self) -> None:
        """The stand-ins are only faithful while these remain torch's surface."""
        torch = pytest.importorskip("torch", reason="torch ships in the extras that run a policy")
        for name in ("is_available", "get_device_capability", "get_device_name", "get_arch_list"):
            assert callable(getattr(torch.cuda, name)), name
        assert isinstance(torch.__version__, str)

    def test_the_interval_rule_agrees_with_torchs_own_predicate(self) -> None:
        """The written-out rule is graded against the predicate it stands in for."""
        torch = pytest.importorskip("torch", reason="torch ships in the extras that run a policy")
        predicate = getattr(torch.cuda, "_code_compatible_with_device", None)
        if predicate is None:
            pytest.skip("this torch predates the interval table")
        pairs = ((ORIN_ARCH, 80), (THOR_ARCH, 80), (THOR_ARCH, 90), (THOR_ARCH, THOR_ARCH), (90, 90))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for device_cc, code_cc in pairs:
                assert bool(predicate(device_cc, code_cc)) is _interval_rule(device_cc, code_cc), (
                    device_cc,
                    code_cc,
                )

    def test_the_arch_parser_agrees_with_torchs_own(self) -> None:
        torch = pytest.importorskip("torch", reason="torch ships in the extras that run a policy")
        parse = getattr(torch.cuda, "_extract_arch_version", None)
        if parse is None:
            pytest.skip("this torch does not expose its arch parser")
        for entry in ("sm_80", "sm_90a", "sm_120f", "compute_80"):
            assert int(parse(entry)) == _extract(entry) == _extract_via_doctor(entry), entry

    def test_torch_calls_such_a_build_not_compatible_in_its_own_words(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The refusal's wording is quoted from torch, so torch is asked to say it."""
        torch = pytest.importorskip("torch", reason="torch ships in the extras that run a policy")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device for torch to report on")
        major, _minor = torch.cuda.get_device_capability(0)
        other_major = 8 if major != 8 else 9
        monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: [f"sm_{other_major}0"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda._check_cubins()
        assert any("not compatible" in str(entry.message) for entry in caught), [str(entry.message) for entry in caught]


def _extract_via_doctor(entry: str) -> int | None:
    """The shipped parser, read through the module under test."""
    from strands_robots.doctor import _arch_entry_version

    return _arch_entry_version(entry)
