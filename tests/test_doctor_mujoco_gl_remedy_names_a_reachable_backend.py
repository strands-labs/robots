"""A MUJOCO_GL remedy names a backend this host can reach, not merely accept.

``check_mujoco_gl`` already reasons about not recommending a value MuJoCo would
refuse: its own docstring says "what to recommend has to be valid here", and a
sibling case pins that a Darwin remedy never offers a Linux backend. That
validates the advice against the *platform's vocabulary*.

Reachability is a second axis. Each offscreen backend loads a system library -
``egl`` needs ``libEGL.so.1``, ``osmesa`` needs ``libOSMesa.so`` - and a host can
accept a value whose library is not installed. Recommending it then sends the
reader after an export that changes nothing, and on a host with neither library
the real remedy is the library rather than a different variable. The backend
module already probes exactly this, in ``_configure_gl_backend``, and its own
warning names the packages; the verdict did not read that probe, so on a host
missing OSMesa it still offered ``osmesa``, and on a host missing both it offered
an export where nothing exportable can render.

The library table below is transcribed rather than imported, so these cases grade
the shipped behaviour against a second opinion; one drift case pins the two
against each other, and another pins the configurator's own probe against it so
the two readers of that table cannot disagree about what a host can reach.
"""

from __future__ import annotations

import inspect
import platform
import re
import sys

import pytest

# The offscreen backends and the library each one loads, transcribed independently.
_OFFSCREEN_LIBRARIES = (("egl", "libEGL.so.1"), ("osmesa", "libOSMesa.so"))


def _plain(verdict: str) -> str:
    """The verdict with ANSI colour removed."""
    return re.sub(r"\033\[[0-9;]*m", "", verdict)


def _marker(verdict: str) -> str:
    """The status a caller can recover from a verdict line."""
    for name in ("FAIL", "WARN", "PASS", "SKIP"):
        if f"  {name}  " in verdict:
            return name
    raise AssertionError(f"verdict carries no status marker: {verdict!r}")


def _remedy(verdict: str) -> str:
    """The remedy lines, without the headline that names the value under complaint."""
    return "\n".join(_plain(verdict).split("\n")[1:])


def _offscreen_values_recommended(verdict: str) -> set[str]:
    """Offscreen backends the remedy tells the reader to set.

    Matched case-sensitively and on a word boundary, so the library sonames a
    remedy may name - ``libEGL.so.1``, ``libOSMesa.so`` - do not read as advice to
    export a value. :meth:`TestTheProbeSetsAnswerDifferentQuestions.
    test_a_library_soname_does_not_read_as_a_recommended_value` grades that.
    """
    remedy = _remedy(verdict)
    return {value for value, _library in _OFFSCREEN_LIBRARIES if re.search(rf"\b{value}\b", remedy)}


# Every setting that reaches a remedy naming an offscreen backend, with the
# verdict it carries. ``None`` means the variable is unset.
_REMEDY_SITES = [
    pytest.param("off", "FAIL", id="a-disabled-gl-context"),
    pytest.param("glfw", "WARN", id="a-display-backend-while-headless"),
    pytest.param(None, "FAIL", id="unset-while-headless"),
]


@pytest.fixture
def linux_host(monkeypatch: pytest.MonkeyPatch):
    """A headless Linux host whose installed GL libraries the caller chooses.

    Returns a callable taking the set of loadable backend names, so a case can
    model a host with both libraries, one, or neither without touching the
    machine running the suite.
    """
    import strands_robots.simulation.mujoco.backend as backend

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    def stage(loadable: set[str], mujoco_gl: str | None = None):
        present = {library for value, library in _OFFSCREEN_LIBRARIES if value in loadable}
        monkeypatch.setattr(backend, "_library_loads", lambda name: name in present)
        if mujoco_gl is None:
            monkeypatch.delenv("MUJOCO_GL", raising=False)
        else:
            monkeypatch.setenv("MUJOCO_GL", mujoco_gl)
        from strands_robots.doctor import check_mujoco_gl

        return check_mujoco_gl()

    return stage


class TestARemedyNamesOnlyAReachableBackend:
    """A backend whose library is absent is advice the reader cannot follow."""

    @pytest.mark.parametrize("value,expected", _REMEDY_SITES)
    @pytest.mark.parametrize("loadable", [{"egl"}, {"osmesa"}], ids=["only-egl", "only-osmesa"])
    def test_only_the_loadable_backend_is_recommended(
        self, linux_host, value: str | None, expected: str, loadable: set[str]
    ) -> None:
        verdict = linux_host(loadable, value)
        assert _marker(verdict) == expected, f"the verdict itself is unchanged: {_plain(verdict)!r}"
        assert _offscreen_values_recommended(verdict) == loadable, (
            f"a host that can load only {sorted(loadable)} must be offered only that; remedy was {_remedy(verdict)!r}"
        )


class TestNoOffscreenBackendIsReachable:
    """With neither library installed, the remedy is the library, not a variable."""

    @pytest.mark.parametrize("value,expected", _REMEDY_SITES)
    def test_the_remedy_names_the_absent_libraries(self, linux_host, value: str | None, expected: str) -> None:
        verdict = linux_host(set(), value)
        remedy = _remedy(verdict)
        for _value, library in _OFFSCREEN_LIBRARIES:
            assert library in remedy, f"the reader needs the name of what is missing; remedy was {remedy!r}"

    @pytest.mark.parametrize("value,expected", _REMEDY_SITES)
    def test_the_remedy_recommends_no_export(self, linux_host, value: str | None, expected: str) -> None:
        verdict = linux_host(set(), value)
        assert _offscreen_values_recommended(verdict) == set(), (
            "no exported value can render when neither library loads, so naming one is advice "
            f"that cannot be followed: {_remedy(verdict)!r}"
        )


class TestWhatIsUnchanged:
    """Over-reach controls: these hold before and after, and must keep holding.

    The remedy is the only thing that moved. A host missing a library still earns
    the verdict its setting earns, and the recommended command was already
    runnable - a second choice has always belonged in the comment rather than in
    the exported value, because ``export MUJOCO_GL=egl or osmesa`` is not a
    command.
    """

    @pytest.mark.parametrize("value", ["egl", "osmesa"])
    @pytest.mark.parametrize("loadable", [set(), {"egl"}, {"osmesa"}], ids=["neither", "only-egl", "only-osmesa"])
    def test_an_offscreen_backend_passes_whether_or_not_its_library_is_here(
        self, linux_host, value: str, loadable: set[str]
    ) -> None:
        """Classification is the platform's question, not this host's.

        Whether ``egl`` *is* an offscreen backend is a property of MuJoCo, so a
        reader who set one keeps the verdict it earns even where its library is
        absent. Narrowing this to what can load reclassified such a value as a
        display backend and warned that it needs a display - on a host that has
        OSMesa and no EGL, ``MUJOCO_GL=egl`` read as ``WARN (needs display)``.
        """
        assert _marker(linux_host(loadable, value)) == "PASS", (
            f"{value} is an offscreen backend on Linux however this host is provisioned: "
            f"{_plain(linux_host(loadable, value))!r}"
        )

    @pytest.mark.parametrize("value,expected", _REMEDY_SITES)
    def test_the_verdict_is_still_the_one_the_setting_earns(self, linux_host, value: str | None, expected: str) -> None:
        assert _marker(linux_host(set(), value)) == expected, "an absent library is not a different diagnosis"

    @pytest.mark.parametrize("value,expected", _REMEDY_SITES)
    @pytest.mark.parametrize("loadable", [{"egl"}, {"osmesa"}, {"egl", "osmesa"}], ids=["egl", "osmesa", "both"])
    def test_the_recommended_command_stays_runnable(
        self, linux_host, value: str | None, expected: str, loadable: set[str]
    ) -> None:
        verdict = linux_host(loadable, value)
        for match in re.finditer(r"export MUJOCO_GL=(\S+)", _remedy(verdict)):
            assert match.group(1) in {v for v, _ in _OFFSCREEN_LIBRARIES}, (
                f"an exported value has to be one backend name, not a list: {_remedy(verdict)!r}"
            )


class TestTheProbeSetsAnswerDifferentQuestions:
    """Premises: what each set is for, and that the extractor above is sound."""

    def test_the_platform_set_ignores_the_libraries(self, linux_host) -> None:
        import strands_robots.simulation.mujoco.backend as backend

        linux_host(set())
        assert sorted(backend._mujoco_gl_offscreen_values()) == ["egl", "osmesa"], (
            "the platform set answers what MuJoCo accepts on Linux, whatever is installed"
        )

    @pytest.mark.parametrize("loadable", [set(), {"egl"}, {"osmesa"}, {"egl", "osmesa"}])
    def test_the_reachable_set_is_the_installed_subset(self, linux_host, loadable: set[str]) -> None:
        import strands_robots.simulation.mujoco.backend as backend

        linux_host(loadable)
        reachable = backend._mujoco_gl_loadable_offscreen_values()
        assert reachable == loadable
        assert reachable <= backend._mujoco_gl_offscreen_values(), "reachable is a subset of accepted"

    def test_a_library_soname_does_not_read_as_a_recommended_value(self) -> None:
        """Non-vacuity for :func:`_offscreen_values_recommended`."""
        named = "  FAIL  x\n        install an offscreen GL library (libEGL.so.1, libOSMesa.so)"
        assert _offscreen_values_recommended(named) == set(), "a soname is not an export instruction"
        exported = "  FAIL  x\n        export MUJOCO_GL=egl  # or osmesa"
        assert _offscreen_values_recommended(exported) == {"egl", "osmesa"}, "an export instruction is one"


class TestTheProbeItselfAnswersAboutTheLoader:
    """The probe, not the stub the cases above install.

    Every case that models a host installs a stand-in for :func:`_library_loads`,
    so nothing there grades the probe. These do, against the loader directly.
    """

    def test_a_soname_no_distribution_supplies_is_absent(self) -> None:
        import strands_robots.simulation.mujoco.backend as backend

        assert not backend._library_loads("libstrands_robots_no_such_library.so.99"), (
            "a probe that reports every soname present would recommend a backend on any host"
        )

    @pytest.mark.parametrize("library", [library for _value, library in _OFFSCREEN_LIBRARIES])
    def test_the_probe_agrees_with_the_loader(self, library: str) -> None:
        import ctypes

        import strands_robots.simulation.mujoco.backend as backend

        try:
            ctypes.cdll.LoadLibrary(library)
            present = True
        except OSError:
            present = False
        assert backend._library_loads(library) is present, (
            f"the probe and the loader disagree about {library} on this host"
        )


class TestTheProbeHasOneOwner:
    """The configurator and the verdict must agree about what a host can reach."""

    def test_the_table_matches_the_module(self) -> None:
        import strands_robots.simulation.mujoco.backend as backend

        assert backend._MUJOCO_GL_OFFSCREEN_LIBRARIES == _OFFSCREEN_LIBRARIES, (
            "the transcribed second opinion above has drifted from the module"
        )

    def test_the_configurator_probes_the_libraries_the_table_names(self) -> None:
        """``_configure_gl_backend`` picks a backend from the same libraries.

        It keeps its own ordered first-match probe, because it also stages the
        NVIDIA vendor ICD when it selects EGL. What must not drift is *which*
        libraries decide: a configurator probing one soname while the verdict
        recommends from another would disagree about the same host.
        """
        import strands_robots.simulation.mujoco.backend as backend

        source = inspect.getsource(backend._configure_gl_backend)
        probed = set(re.findall(r"LoadLibrary\(\"([^\"]+)\"\)", source))
        assert probed == {library for _value, library in _OFFSCREEN_LIBRARIES}, (
            f"the configurator probes {sorted(probed)}, the recommendation table names "
            f"{sorted(library for _v, library in _OFFSCREEN_LIBRARIES)}"
        )
