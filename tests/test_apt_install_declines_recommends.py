"""Contract pin: a CI ``apt-get install`` declines Recommends.

``test-lint.yml``'s ``Install system dependencies (OpenGL for MuJoCo)`` step is
bounded at 12 minutes, sized when 31 successful runs put it at p50 33s and max
455s (7m35s). That band no longer holds. On 2026-08-19 two unrelated pull
requests were reaped by that bound 40 minutes apart, with the identical message
and nothing wrong in either diff::

    #2488  job 96100298786  14:27:49Z  The action 'Install system dependencies
                                       (OpenGL for MuJoCo)' has timed out after
                                       12 minutes.
    #2489  job 96107703838  15:07:23Z  (same)

Neither branch touches CI, MuJoCo or video. What they share is the download the
step performs, and the ``Get:`` timestamps in #2489's log show where the 12
minutes went -- not to a uniform slowdown, but to multi-minute stalls that land
after the largest files::

    14:56:45  Get:109  mesa-common-dev      3133 kB
    15:00:54  Get:110  mesa-va-drivers         6784 B   <- 4m09s gap
    15:01:25  Get:115  libosmesa6           3555 kB
    15:06:09  Get:116  libosmesa6-dev          9220 B   <- 4m44s gap
    15:06:11  Get:117  pocketsphinx-en-us  27400 kB
    15:07:23           step reaped, 72s into that fetch

So the step is bandwidth-bound on a mirror that stalls, and what decides whether
it survives is how many bytes it asks for. Measured with ``apt-get install
--print-uris`` on ``ubuntu-24.04``, summing the sizes apt itself reports:

===================================  =========  ==========
request                              packages   download
===================================  =========  ==========
``libosmesa6-dev ffmpeg``                  116   98.1 MiB
``libosmesa6-dev ffmpeg`` (no recs)        104   68.2 MiB
``libosmesa6-dev``                          17    8.33 MiB
``libosmesa6-dev`` (no recs)                17    8.33 MiB
===================================  =========  ==========

Declining Recommends removes 29.9 MiB, 30.5% of the transfer, and the 12
packages it drops are all reachable only as Recommends::

    i965-va-driver  intel-media-va-driver  mesa-va-drivers  mesa-vdpau-drivers
    va-driver-all   vdpau-driver-all       libigdgmm12          (GPU video accel)
    libaacs0        libbdplus0                                  (Blu-ray DRM)
    libdecor-0-plugin-1-gtk  librsvg2-common                     (GTK/SVG assets)
    pocketsphinx-en-us                                          (26.2 MiB of ASR
                                                                 acoustic models)

``pocketsphinx-en-us`` alone is 27,433,580 bytes -- the single largest file in the
set, and the one the step was fetching when it died. It arrives because
``ffmpeg`` -> ``libavfilter9`` -> ``libpocketsphinx3`` is a ``Depends`` chain and
``libpocketsphinx3`` then ``Recommends: pocketsphinx-en-us``. It backs
``libavfilter``'s speech-recognition filter, which nothing here calls.

None of the 12 is load-bearing on a hosted runner. The GPU entries are VA-API and
VDPAU *hardware* acceleration backends, and this job has no GPU -- the whole point
of ``libosmesa6-dev`` is that MuJoCo renders through OSMesa in software. The
software encoders the repository actually names (``h264``, ``hevc``,
``libsvtav1`` in ``strands_robots/video.py``'s ``SUPPORTED_CODECS``) live in
``libavcodec60``, a ``Depends``, and are untouched. ``ffmpeg`` itself and every
``libav*`` shared library it carries also stay, which is what keeps ``torchcodec``
working -- it discovers and links the system FFmpeg libraries rather than the
``ffmpeg`` binary (see ``tests/test_package_lazy_imports.py``).

Why the payload rather than the bound
-------------------------------------
Raising the step's 12 would be the other available remedy, and it is the wrong
one here: #2457 measures the job's legal spend as already ``29.1 + 12 + 4.4 =
45.5`` minutes against a 45-minute job bound, so every minute added to this step
has to be added to the job bound too. Cutting the transfer moves the step's
*observed* cost down instead, which spends none of that budget. This module
therefore does not assert the bound at all -- ``tests/test_workflow_jobs_are_bounded.py``
owns it -- and the flag lands with ``timeout-minutes: 12`` unchanged.

Why the flag is applied where it currently changes nothing
----------------------------------------------------------
``agent-api-check.yml`` installs ``libosmesa6-dev`` only, and the table above
shows that request is byte-identical with and without the flag (17 packages,
8.33 MiB either way), so the flag is a measured no-op there today. It is applied
anyway, and the guard below takes no exemption, because the no-op is a property
of *that argument list* rather than of the workflow: the day a package with heavy
Recommends is added to it -- ``ffmpeg`` being the obvious candidate, since the
step already exists to serve the same MuJoCo rendering path -- the 29.9 MiB
arrives silently and the reap it produces names a step, not a dependency class.
An invariant with no exceptions is also the only kind a reader can apply without
first re-deriving this measurement.

The guard is repo-wide over ``.github/workflows/*.yml`` for the same reason, so a
third call site added later is covered by construction rather than by remembering
to extend a list.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"

# ``install`` only. ``apt-get update`` takes no package list, so Recommends
# cannot apply to it, and the retry loops around it would otherwise be flagged.
_APT_INSTALL = re.compile(r"\bapt-get\s+install\b")
_DECLINES_RECOMMENDS = re.compile(r"--no-install-recommends\b")

# The two call sites this module was written against. Named so that a rename or a
# deletion is a visible failure here rather than a silent loss of coverage.
_KNOWN_CALL_SITES = {"test-lint.yml", "agent-api-check.yml"}


def _is_an_install(line: str) -> bool:
    """Whether ``line`` runs ``apt-get install``, ignoring YAML/shell comments.

    A commented-out call installs nothing, so counting one would let a prose
    mention of the command fail this guard.
    """
    return not line.lstrip().startswith("#") and bool(_APT_INSTALL.search(line))


def _declines_recommends(line: str) -> bool:
    return bool(_DECLINES_RECOMMENDS.search(line))


class _AptInstall(NamedTuple):
    """One ``apt-get install`` line, located for a failure message."""

    workflow: str
    lineno: int
    line: str

    @property
    def ref(self) -> str:
        return f"{self.workflow}:{self.lineno}"

    def __repr__(self) -> str:  # pragma: no cover - test ids only
        return self.ref


def _apt_installs() -> list[_AptInstall]:
    """Every ``apt-get install`` across the workflow definitions.

    Parsed line by line rather than with ``yaml``: ``tests/`` is type-checked
    under ``ignore_missing_imports = false`` and ``types-PyYAML`` is not a dev
    dependency, so importing it would either fail ``mypy`` or force a
    dependency change and a ``uv.lock`` relock. The command lives inside a
    ``run: |`` block scalar, which is opaque to a YAML parser anyway -- it would
    hand back the same string to run this regex over.
    """
    found: list[_AptInstall] = []
    for path in sorted(_WORKFLOWS.glob("*.yml")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if _is_an_install(line):
                found.append(_AptInstall(path.name, lineno, line.strip()))
    return found


class TestEveryAptInstallDeclinesRecommends:
    """The invariant, over whatever call sites exist now."""

    @pytest.mark.parametrize("install", _apt_installs(), ids=lambda i: i.ref)
    def test_the_install_declines_recommends(self, install: _AptInstall) -> None:
        assert _declines_recommends(install.line), (
            f"{install.ref} installs without --no-install-recommends, so apt "
            f"also fetches the Recommends closure (29.9 MiB / 12 packages for "
            f"`libosmesa6-dev ffmpeg`, 26.2 MiB of it pocketsphinx-en-us "
            f"speech models): {install.line}"
        )


class TestTheGuardIsNotVacuous:
    """Three ways this module could pass while checking nothing."""

    def test_call_sites_are_found(self) -> None:
        """A glob that matches nothing parametrizes to zero passing cases."""
        assert _apt_installs(), (
            f"no `apt-get install` found under {_WORKFLOWS} - either the "
            f"workflows moved or the pattern stopped matching"
        )

    def test_both_known_call_sites_are_still_covered(self) -> None:
        found = {install.workflow for install in _apt_installs()}
        assert _KNOWN_CALL_SITES <= found, (
            f"expected an apt install in {sorted(_KNOWN_CALL_SITES)}, found {sorted(found)}"
        )

    def test_an_install_missing_the_flag_is_caught(self) -> None:
        """The predicate refuses the exact form this change replaced."""
        assert _is_an_install("          sudo apt-get install -y libosmesa6-dev ffmpeg")
        assert not _declines_recommends("          sudo apt-get install -y libosmesa6-dev ffmpeg")

    def test_a_commented_out_install_is_not_counted(self) -> None:
        """Prose about the command is not a call site."""
        assert not _is_an_install("          # sudo apt-get install -y ffmpeg")
        assert not _is_an_install("        # see `apt-get install` above")

    def test_apt_get_update_is_not_treated_as_a_call_site(self) -> None:
        """``update`` takes no package list, so Recommends cannot apply."""
        assert not _is_an_install("            if sudo timeout 180 apt-get update; then")


class TestTheRationaleIsRecorded:
    """The flag is one token, and a reader who cannot see why will drop it.

    It reads as redundant next to a retry loop that already survives a bad
    mirror, so what stops it being tidied away is the measurement sitting beside
    it -- naming the package that dominates the transfer, and the class of
    dependency it arrives as.
    """

    @staticmethod
    def _step_comment() -> str:
        return (_WORKFLOWS / "test-lint.yml").read_text()

    def test_the_comment_names_the_dominant_package(self) -> None:
        assert "pocketsphinx" in self._step_comment(), (
            "test-lint.yml should name pocketsphinx-en-us beside the flag: it is "
            "the largest file in the set and the one being fetched when the step "
            "was reaped, which is what makes the flag load-bearing rather than tidy"
        )

    def test_the_comment_names_the_dependency_class(self) -> None:
        """Recommends is the whole mechanism: a Depends could not be declined."""
        assert "Recommends" in self._step_comment()
