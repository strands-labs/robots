"""The numba/coverage clash diagnostic must name a remedy that works.

``numba.misc.coverage_support`` defines ``class NumbaTracer(coverage.types.Tracer)``
with no version guard - its only guard is ``except ImportError`` around
``import coverage`` - and ``coverage`` provides ``coverage.types.Tracer`` only
from 7.6.1 onward. Measured by evaluating that class body against real coverage
releases:

===============  ============================================================
``coverage``     result of ``class NumbaTracer(coverage.types.Tracer)``
===============  ============================================================
6.5.0            ``AttributeError: module 'coverage' has no attribute 'types'``
7.3.4            ``AttributeError: module 'coverage.types' has no attribute 'Tracer'``
7.6.0            ``AttributeError: module 'coverage.types' has no attribute 'Tracer'``
7.6.1            defines cleanly
===============  ============================================================

So the clash is a coverage-too-OLD condition. Two contracts follow, and this
module pins both:

* The remedy must send the caller UP to ``coverage>=7.6.1`` (or remove coverage
  entirely, which numba's ImportError guard tolerates), and must never advise
  pinning coverage down - that does not fix the clash.
* The coverage-6.x failure shape must be recognised as the same clash. It is
  the state a caller lands in after pinning coverage down, and if the detector
  misses it the strict :class:`_ControllerInstallError` degrades into
  :class:`_ControllerDependencyMissing`, i.e. the silent "GR00T actions will
  no-op" path that #522 exists to prevent.

Every raise site that classifies the clash must carry the remedy: the adapter's
``_action_controller_remediation`` and both lazy-import guards in
``_LiberoOSCController.from_sim`` (mujoco and robosuite).
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.benchmarks.libero.adapter import (  # noqa: E402
    _COVERAGE_TRACER_MIN_VERSION,
    LiberoAdapter,
    _ControllerDependencyMissing,
    _ControllerInstallError,
    _is_numba_coverage_clash,
    _numba_coverage_clash_remedy,
)

from .test_libero_osc_controller_from_sim_construction import (  # noqa: E402
    _force_mujoco_import_error,
    _from_sim,
    _install_fake_robosuite,
    _make_panda_sim,
    _StubSim,
)

#: The clash as it surfaces on coverage 7.0-7.6.0 (``coverage/types.py`` exists
#: but names the protocol ``TTracer`` or ``TracerCore``).
_TRACER_MISSING = "module 'coverage.types' has no attribute 'Tracer'"

#: The clash as it surfaces on coverage 6.x and older, which ship no
#: ``coverage/types.py`` at all - the state that pinning coverage down produces.
_TYPES_MODULE_MISSING = "module 'coverage' has no attribute 'types'"

#: The one command that resolves the clash, as the remedy must spell it.
_INSTALL_COMMAND = f"pip install 'coverage>={_COVERAGE_TRACER_MIN_VERSION}'"


def _version_tuple(text: str) -> tuple[int, ...]:
    """Parse the leading numeric components of a version string."""
    parts: list[int] = []
    for chunk in text.split("."):
        if not chunk.isdigit():
            break
        parts.append(int(chunk))
    return tuple(parts)


class TestRemedyNamesAFixThatWorks:
    """The remedy is the only place the library tells a user how to escape the
    clash, so it must name the measured fix and nothing that fails."""

    def test_remedy_names_the_coverage_floor_to_install(self) -> None:
        """The copy-pasteable install command is present, floor included."""
        assert _INSTALL_COMMAND in _numba_coverage_clash_remedy()

    def test_remedy_offers_removing_coverage_as_the_alternative(self) -> None:
        """numba guards ``import coverage`` with ``except ImportError``, so an
        absent coverage genuinely works - keep offering it."""
        assert "pip uninstall coverage" in _numba_coverage_clash_remedy()

    @pytest.mark.parametrize(
        "bad_advice",
        [
            "coverage<7",
            "pin coverage<",
            "coverage>=7 removed",
            "upgrade numba",
        ],
    )
    def test_remedy_never_advises_a_downgrade_or_a_numba_bump(self, bad_advice: str) -> None:
        """Pinning coverage down leaves the clash in place (measured on 6.5.0),
        and numba still subclasses ``coverage.types.Tracer`` unconditionally, so
        neither is a fix."""
        assert bad_advice not in _numba_coverage_clash_remedy()

    def test_the_named_floor_provides_the_symbol_numba_needs(self) -> None:
        """Pin the floor against the installed coverage.

        If a future coverage renames ``Tracer`` again, the floor this message
        recommends stops being a fix - and this test says so instead of the
        message quietly becoming wrong.
        """
        coverage = pytest.importorskip("coverage")
        installed = _version_tuple(coverage.__version__)
        floor = _version_tuple(_COVERAGE_TRACER_MIN_VERSION)
        if installed < floor:
            pytest.skip(f"installed coverage {coverage.__version__} predates the {_COVERAGE_TRACER_MIN_VERSION} floor")
        assert hasattr(coverage.types, "Tracer"), (
            f"coverage {coverage.__version__} is at or above the "
            f"{_COVERAGE_TRACER_MIN_VERSION} floor the clash remedy recommends, "
            "but no longer provides coverage.types.Tracer - the remedy needs a new floor"
        )


class TestEveryClashRaiseSiteCarriesTheRemedy:
    """Three code paths classify the clash. All three are what a caller sees,
    so all three must hand over the fix rather than only the symptom."""

    def test_install_action_controller_remediation(self) -> None:
        """The adapter-level hint (``_install_action_controller``'s fatal path)."""
        hint = LiberoAdapter._action_controller_remediation(AttributeError(_TRACER_MISSING))
        assert _INSTALL_COMMAND in hint

    def test_robosuite_import_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``from_sim``'s robosuite guard - the message users actually hit,
        since robosuite's OSC path is what pulls numba in."""
        _install_fake_robosuite(monkeypatch, clash=True)
        sim, _model, _data = _make_panda_sim()

        with pytest.raises(_ControllerInstallError) as exc:
            _from_sim(sim)
        assert _INSTALL_COMMAND in str(exc.value)

    def test_mujoco_import_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``from_sim``'s mujoco guard, which runs before robosuite."""
        _force_mujoco_import_error(monkeypatch, ImportError(_TRACER_MISSING))

        with pytest.raises(_ControllerInstallError) as exc:
            _from_sim(_StubSim(None))
        assert _INSTALL_COMMAND in str(exc.value)


class TestTheStateAPinnedDownCoverageProduces:
    """coverage 6.x ships no ``coverage/types.py``, so the same numba class body
    fails one attribute earlier. That shape is the same clash with the same fix
    and must be classified the same way."""

    def test_detector_recognises_the_absent_types_module(self) -> None:
        assert _is_numba_coverage_clash(AttributeError(_TYPES_MODULE_MISSING)) is True

    def test_detector_recognises_it_through_a_wrapping_import_error(self) -> None:
        """The signature survives the ImportError CPython re-raises around it."""
        cause = AttributeError(_TYPES_MODULE_MISSING)
        wrapper = ImportError("cannot import name 'controller_factory'")
        wrapper.__cause__ = cause
        assert _is_numba_coverage_clash(wrapper) is True

    def test_from_sim_surfaces_it_strictly_with_the_remedy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Not the degrade-gracefully subclass: an unrecognised clash would
        report "GR00T actions will no-op" and score the eval at 0."""
        _force_mujoco_import_error(monkeypatch, ImportError(_TYPES_MODULE_MISSING))

        with pytest.raises(_ControllerInstallError) as exc:
            _from_sim(_StubSim(None))
        assert not isinstance(exc.value, _ControllerDependencyMissing)
        assert _INSTALL_COMMAND in str(exc.value)

    def test_robosuite_guard_surfaces_it_strictly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same contract on the robosuite guard, which is the path that pulls
        numba in and therefore the one a real eval trips over."""
        _install_fake_robosuite(monkeypatch, clash=True, clash_message=_TYPES_MODULE_MISSING)
        sim, _model, _data = _make_panda_sim()

        with pytest.raises(_ControllerInstallError) as exc:
            _from_sim(sim)
        assert not isinstance(exc.value, _ControllerDependencyMissing)
        assert _INSTALL_COMMAND in str(exc.value)

    def test_adapter_remediation_covers_that_shape_too(self) -> None:
        hint = LiberoAdapter._action_controller_remediation(AttributeError(_TYPES_MODULE_MISSING))
        assert _INSTALL_COMMAND in hint


class TestUnrelatedFailuresStillDegradeGracefully:
    """Widening the detector must not swallow ordinary missing-dependency
    failures, which are environmental rather than fixable setup bugs."""

    @pytest.mark.parametrize(
        "text",
        [
            "libGL.so.1: cannot open shared object file",
            "No module named 'robosuite'",
            "coverage report generation failed",
            "module 'numpy' has no attribute 'float'",
            "module 'mujoco' has no attribute 'MjSpec'",
        ],
    )
    def test_not_classified_as_the_clash(self, text: str) -> None:
        assert _is_numba_coverage_clash(ImportError(text)) is False

    def test_generic_hint_for_an_unrelated_install_failure(self) -> None:
        hint = LiberoAdapter._action_controller_remediation(RuntimeError("unrelated failure"))
        assert _INSTALL_COMMAND not in hint
        assert "robosuite" in hint
