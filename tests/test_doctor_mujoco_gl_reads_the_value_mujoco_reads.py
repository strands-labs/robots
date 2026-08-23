"""The MUJOCO_GL verdict is about the value MuJoCo reads, not the string typed.

MuJoCo folds ``MUJOCO_GL`` with ``.lower().strip()``, reads
``disable/disabled/off/false/0`` as "build no GL context at all", accepts a
platform-dependent set of backend names, and raises ``RuntimeError`` at import
for anything else (``mujoco.gl_context``). Re-deriving that vocabulary loosely
answered about a spelling: ``MUJOCO_GL=EGL`` renders through EGL and read as
unrecognised, and every unrecognised value - a disabled GL context, or one MuJoCo
refuses outright - read as "not set", which on a machine with a display passed.

The same fold decides whether the NVIDIA EGL vendor ICD is staged, so
``MUJOCO_GL=EGL`` selected EGL while skipping the guarantee that keeps glvnd off
Mesa ``llvmpipe`` - the silent software-rasterizer fallback that
``_ensure_nvidia_egl_vendor_icd`` exists to prevent.

The vocabulary below is written out here rather than imported, so these cases
grade the shipped behaviour against a second opinion; one drift test pins the two
against each other.
"""

from __future__ import annotations

import platform
import re
import sys

import pytest

# MuJoCo's own vocabulary, transcribed independently of the package.
_DISABLE = ("disable", "disabled", "off", "false", "0")
_ANY_PLATFORM = ("", "1", "enable", "enabled", "glfw", "on", "true")
_LINUX_ONLY = ("egl", "glx", "osmesa")
_DARWIN_ONLY = ("cgl",)
_WINDOWS_ONLY = ("wgl",)


def _marker(verdict: str) -> str:
    """The status a caller can recover from a verdict line."""
    for name in ("FAIL", "WARN", "PASS", "SKIP"):
        if f"  {name}  " in verdict:
            return name
    raise AssertionError(f"verdict carries no status marker: {verdict!r}")


def _plain(verdict: str) -> str:
    """The verdict with ANSI colour removed."""
    return re.sub(r"\033\[[0-9;]*m", "", verdict)


def _headline(verdict: str) -> str:
    """The verdict's own claim, without the remedy lines below it.

    A remedy legitimately says "or unset it", so a claim about what the variable
    holds has to be read from the line that makes it.
    """
    return _plain(verdict).split("\n")[0]


def _recommended_values(verdict: str) -> set[str]:
    """Every MUJOCO_GL value the verdict's remedy tells the reader to set.

    Only the remedy lines are read: the headline names the value the caller
    already has, which is exactly the value under complaint.
    """
    lines = _plain(verdict).split("\n")[1:]
    found: set[str] = set()
    for line in lines:
        for match in re.finditer(r"MUJOCO_GL=(.+?)(?:\s+#|\s+for headless|$)", line):
            body = match.group(1).strip()
            body = re.sub(r"^<one of:\s*", "", body).rstrip(">")
            for token in re.split(r",|\bor\b", body):
                token = token.strip()
                if token:
                    found.add(token)
    return found


@pytest.fixture
def linux_headless(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """A headless Linux host: the platform where every backend name is available."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    return monkeypatch


class TestTheVerdictIsAboutTheValueMujocoReads:
    """A spelling MuJoCo folds to a working backend is not a broken setup."""

    @pytest.mark.parametrize("spelling", ["EGL", " egl ", "EgL", "\tosmesa\n", "OSMesa"])
    def test_a_folded_offscreen_backend_passes(self, linux_headless, spelling: str) -> None:
        from strands_robots.doctor import check_mujoco_gl

        linux_headless.setenv("MUJOCO_GL", spelling)
        verdict = check_mujoco_gl()
        assert _marker(verdict) == "PASS", (
            f"MUJOCO_GL={spelling!r} folds to {spelling.lower().strip()!r}, which renders headlessly, "
            f"yet the verdict is {_marker(verdict)}: {_plain(verdict)!r}"
        )

    @pytest.mark.parametrize("spelling", ["GLFW", " glfw ", "GLX", "ON", "TRUE"])
    def test_a_folded_display_backend_warns_rather_than_reading_as_unset(self, linux_headless, spelling: str) -> None:
        from strands_robots.doctor import check_mujoco_gl

        linux_headless.setenv("MUJOCO_GL", spelling)
        verdict = check_mujoco_gl()
        assert _marker(verdict) == "WARN", (
            f"MUJOCO_GL={spelling!r} selects a backend that needs a display; the verdict is "
            f"{_marker(verdict)}: {_plain(verdict)!r}"
        )
        assert "not set" not in _headline(verdict), (
            f"MUJOCO_GL={spelling!r} is set, so the verdict must not report it as unset: {_headline(verdict)!r}"
        )

    def test_a_folded_verdict_names_the_value_mujoco_reads(self, linux_headless) -> None:
        from strands_robots.doctor import check_mujoco_gl

        linux_headless.setenv("MUJOCO_GL", "EGL")
        text = _plain(check_mujoco_gl())
        assert "EGL" in text and "egl" in text, (
            f"a verdict about a folded value has to name both what was set and what MuJoCo reads: {text!r}"
        )


class TestADisabledGLContextIsNotAWorkingConfiguration:
    """The disable family builds no GL context, so nothing can render."""

    @pytest.mark.parametrize("value", _DISABLE)
    @pytest.mark.parametrize("display", [None, ":0"])
    def test_a_disabled_gl_context_fails(self, linux_headless, value: str, display: str | None) -> None:
        from strands_robots.doctor import check_mujoco_gl

        linux_headless.setenv("MUJOCO_GL", value)
        if display is not None:
            linux_headless.setenv("DISPLAY", display)
        verdict = check_mujoco_gl()
        assert _marker(verdict) == "FAIL", (
            f"MUJOCO_GL={value!r} makes MuJoCo build no GL context at all (DISPLAY={display!r}), "
            f"so the verdict cannot be {_marker(verdict)}: {_plain(verdict)!r}"
        )
        headline = _headline(verdict)
        assert value in headline, f"the verdict has to name the value that disabled it: {headline!r}"
        # MuJoCo accepts this family and simply builds no context, so reporting it
        # as a value MuJoCo refuses at import would name the wrong cause.
        assert "refuses" not in headline, (
            f"MuJoCo accepts {value!r} and builds no GL context from it, so the verdict must not "
            f"report it as a value MuJoCo refuses: {headline!r}"
        )
        assert "render" in headline, f"the verdict has to name the consequence - that nothing can render: {headline!r}"

    def test_a_disabled_gl_context_is_not_reported_as_unset(self, linux_headless) -> None:
        from strands_robots.doctor import check_mujoco_gl

        linux_headless.setenv("MUJOCO_GL", "off")
        linux_headless.setenv("DISPLAY", ":0")
        headline = _headline(check_mujoco_gl())
        assert "unset" not in headline, f"MUJOCO_GL=off is set, and to a value that disables rendering: {headline!r}"


class TestAValueMujocoRefusesIsReportedAsRefused:
    """MuJoCo raises at import for an unrecognised value, so say so."""

    @pytest.mark.parametrize("value", ["egl1", "glfw2", "yes", "egl osmesa", "EGL;osmesa"])
    @pytest.mark.parametrize("display", [None, ":0"])
    def test_an_unrecognised_value_fails_and_names_the_refusal(
        self, linux_headless, value: str, display: str | None
    ) -> None:
        from strands_robots.doctor import check_mujoco_gl

        linux_headless.setenv("MUJOCO_GL", value)
        if display is not None:
            linux_headless.setenv("DISPLAY", display)
        verdict = check_mujoco_gl()
        assert _marker(verdict) == "FAIL", (
            f"MUJOCO_GL={value!r} makes ``import mujoco`` raise RuntimeError (DISPLAY={display!r}), "
            f"so the verdict cannot be {_marker(verdict)}: {_plain(verdict)!r}"
        )
        assert "refuses" in _plain(verdict), (
            f"the verdict has to say MuJoCo refuses the value rather than that none was set: {_plain(verdict)!r}"
        )

    def test_the_refusal_offers_the_values_that_platform_accepts(self, linux_headless) -> None:
        from strands_robots.doctor import check_mujoco_gl

        linux_headless.setenv("MUJOCO_GL", "egl1")
        offered = _recommended_values(check_mujoco_gl())
        assert {"egl", "osmesa", "glfw", "glx"} <= offered, (
            f"a Linux refusal has to offer the Linux backend names; offered {sorted(offered)}"
        )


class TestThePlatformDecidesWhichBackendsAreValid:
    """MuJoCo builds its accepted set per platform, so the verdict has to too."""

    @pytest.fixture
    def macos(self, monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        return monkeypatch

    @pytest.mark.parametrize("value", ["egl", "osmesa", "glx"])
    def test_a_linux_only_backend_is_refused_on_macos(self, macos, value: str) -> None:
        from strands_robots.doctor import check_mujoco_gl

        macos.setenv("MUJOCO_GL", value)
        verdict = check_mujoco_gl()
        assert _marker(verdict) == "FAIL", (
            f"MUJOCO_GL={value!r} is not in MuJoCo's accepted set on Darwin, so ``import mujoco`` raises; "
            f"the verdict is {_marker(verdict)}: {_plain(verdict)!r}"
        )

    def test_the_macos_refusal_offers_cgl_and_not_a_linux_backend(self, macos) -> None:
        from strands_robots.doctor import check_mujoco_gl

        macos.setenv("MUJOCO_GL", "egl")
        offered = _recommended_values(check_mujoco_gl())
        assert "cgl" in offered, f"a Darwin refusal has to offer cgl; offered {sorted(offered)}"
        assert not offered & {"egl", "osmesa", "glx"}, (
            f"a Darwin remedy must not name a backend MuJoCo refuses there; offered {sorted(offered)}"
        )

    def test_the_macos_native_backend_is_accepted(self, macos) -> None:
        from strands_robots.doctor import check_mujoco_gl

        macos.setenv("MUJOCO_GL", _DARWIN_ONLY[0])  # cgl, MuJoCo's macOS backend
        assert _marker(check_mujoco_gl()) != "FAIL", "cgl is MuJoCo's macOS backend"

    def test_an_unset_variable_is_not_a_failure_on_macos(self, macos) -> None:
        from strands_robots.doctor import check_mujoco_gl

        macos.delenv("MUJOCO_GL", raising=False)
        verdict = check_mujoco_gl()
        assert _marker(verdict) == "PASS", (
            "MuJoCo routes macOS to CGL whatever MUJOCO_GL holds, so an unset variable is the "
            f"working default: {_plain(verdict)!r}"
        )

    def test_the_windows_native_backend_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.doctor import check_mujoco_gl

        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("MUJOCO_GL", "wgl")
        assert _marker(check_mujoco_gl()) != "FAIL", "wgl is MuJoCo's Windows backend"


class TestARemedyNeverNamesAValueMujocoRefuses:
    """Every value a verdict tells the reader to set has to be accepted there."""

    @pytest.mark.parametrize("system", ["Linux", "Darwin", "Windows"])
    @pytest.mark.parametrize("value", ["egl1", "off", "glfw", "cgl", "wgl", "egl", ""])
    def test_a_remedy_only_offers_values_that_platform_accepts(
        self, monkeypatch: pytest.MonkeyPatch, system: str, value: str
    ) -> None:
        from strands_robots.doctor import check_mujoco_gl

        accepted = set(_ANY_PLATFORM) | set(
            {"Linux": _LINUX_ONLY, "Darwin": _DARWIN_ONLY, "Windows": _WINDOWS_ONLY}[system]
        )
        monkeypatch.setattr(platform, "system", lambda: system)
        monkeypatch.setattr(sys, "platform", {"Linux": "linux", "Darwin": "darwin", "Windows": "win32"}[system])
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        if value:
            monkeypatch.setenv("MUJOCO_GL", value)
        else:
            monkeypatch.delenv("MUJOCO_GL", raising=False)

        offered = _recommended_values(check_mujoco_gl())
        refused = offered - accepted
        assert not refused, (
            f"on {system}, MUJOCO_GL={value!r} produced a remedy naming {sorted(refused)}, which MuJoCo "
            f"refuses there - following it makes ``import mujoco`` raise"
        )


class TestNothingElseChanges:
    """Every value the previous vocabulary recognised keeps its verdict."""

    def test_egl_still_passes_with_the_same_line(self, linux_headless) -> None:
        from strands_robots.doctor import check_mujoco_gl

        linux_headless.setenv("MUJOCO_GL", "egl")
        assert _plain(check_mujoco_gl()) == "  PASS  MUJOCO_GL=egl"

    def test_glfw_still_warns_with_the_same_lines(self, linux_headless) -> None:
        from strands_robots.doctor import check_mujoco_gl

        linux_headless.setenv("MUJOCO_GL", "glfw")
        assert _plain(check_mujoco_gl()) == (
            "  WARN  MUJOCO_GL=glfw (needs display)\n        Set MUJOCO_GL=egl or osmesa for headless"
        )

    def test_unset_and_headless_still_fails_with_the_same_lines(self, linux_headless) -> None:
        from strands_robots.doctor import check_mujoco_gl

        linux_headless.delenv("MUJOCO_GL", raising=False)
        assert _plain(check_mujoco_gl()) == (
            "  FAIL  MUJOCO_GL not set and no display detected\n"
            "        Fix: export MUJOCO_GL=egl  # or osmesa; add to ~/.bashrc"
        )

    def test_unset_with_a_display_still_passes_with_the_same_line(self, linux_headless) -> None:
        from strands_robots.doctor import check_mujoco_gl

        linux_headless.delenv("MUJOCO_GL", raising=False)
        linux_headless.setenv("DISPLAY", ":0")
        assert _plain(check_mujoco_gl()) == "  PASS  MUJOCO_GL unset (display detected, glfw will work)"


class TestTheEGLVendorICDGuaranteeFollowsTheFoldedValue:
    """Whichever spelling selects EGL reaches the vendor-ICD guarantee."""

    @pytest.fixture
    def icd_spy(self, monkeypatch: pytest.MonkeyPatch) -> list[bool]:
        import strands_robots.simulation.mujoco.backend as backend

        staged: list[bool] = []
        monkeypatch.setattr(backend, "_ensure_nvidia_egl_vendor_icd", lambda: staged.append(True))
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        return staged

    @pytest.mark.parametrize("spelling", ["egl", "EGL", " egl ", "EgL"])
    def test_every_spelling_of_egl_stages_the_vendor_icd(
        self, monkeypatch: pytest.MonkeyPatch, icd_spy: list[bool], spelling: str
    ) -> None:
        import strands_robots.simulation.mujoco.backend as backend

        monkeypatch.setenv("MUJOCO_GL", spelling)
        backend._configure_gl_backend()
        assert icd_spy == [True], (
            f"MUJOCO_GL={spelling!r} selects MuJoCo's EGL backend, so glvnd has to be pointed at the "
            "NVIDIA vendor ICD; without it an NVIDIA host missing that ICD renders on Mesa llvmpipe"
        )

    @pytest.mark.parametrize("spelling", ["osmesa", "OSMESA", "glfw", "GLFW", "glx"])
    def test_a_non_egl_backend_does_not_stage_the_vendor_icd(
        self, monkeypatch: pytest.MonkeyPatch, icd_spy: list[bool], spelling: str
    ) -> None:
        import strands_robots.simulation.mujoco.backend as backend

        monkeypatch.setenv("MUJOCO_GL", spelling)
        backend._configure_gl_backend()
        assert icd_spy == [], f"MUJOCO_GL={spelling!r} does not select EGL, so no vendor ICD is needed"

    @pytest.mark.parametrize("spelling", ["egl", "EGL", " egl ", "glfw", "off"])
    def test_a_user_value_is_never_overwritten(
        self, monkeypatch: pytest.MonkeyPatch, icd_spy: list[bool], spelling: str
    ) -> None:
        import strands_robots.simulation.mujoco.backend as backend

        monkeypatch.setenv("MUJOCO_GL", spelling)
        backend._configure_gl_backend()
        assert backend.os.environ["MUJOCO_GL"] == spelling, "a value the user set is theirs"

    def test_a_whitespace_only_value_is_not_a_preference(
        self, monkeypatch: pytest.MonkeyPatch, icd_spy: list[bool]
    ) -> None:
        import strands_robots.simulation.mujoco.backend as backend

        monkeypatch.setenv("MUJOCO_GL", "   ")
        monkeypatch.setattr(backend.ctypes.cdll, "LoadLibrary", lambda _name: None)
        backend._configure_gl_backend()
        assert backend.os.environ["MUJOCO_GL"] == "egl", (
            "MuJoCo reads a whitespace-only value as no preference, so a headless host still gets "
            "an offscreen backend configured rather than being left on GLFW"
        )


class TestTheTranscribedVocabularyMatchesTheModule:
    """The vocabulary written out above and the module's own must not drift."""

    def test_the_disable_family_agrees(self) -> None:
        from strands_robots.simulation.mujoco.backend import _MUJOCO_GL_DISABLE

        assert _MUJOCO_GL_DISABLE == frozenset(_DISABLE)

    @pytest.mark.parametrize(
        ("system", "extra"),
        [("Linux", _LINUX_ONLY), ("Darwin", _DARWIN_ONLY), ("Windows", _WINDOWS_ONLY), ("FreeBSD", ())],
    )
    def test_the_accepted_set_agrees(self, system: str, extra: tuple[str, ...]) -> None:
        from strands_robots.simulation.mujoco.backend import _mujoco_gl_valid_values

        assert _mujoco_gl_valid_values(system) == frozenset(_ANY_PLATFORM) | frozenset(extra)

    @pytest.mark.parametrize("raw", ["EGL", " egl ", "\tOSMesa\n", "", "   ", "off"])
    def test_the_fold_agrees(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        from strands_robots.simulation.mujoco.backend import _mujoco_gl_value

        monkeypatch.setenv("MUJOCO_GL", raw)
        assert _mujoco_gl_value() == raw.lower().strip()
