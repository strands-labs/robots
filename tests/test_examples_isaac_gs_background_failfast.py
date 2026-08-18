# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The isaac_gs demo's photoreal (3DGS) background path fails loud (issue #2321).

``resolve_background`` used to demote *any* GS failure -- gsplat missing, the
CUDA rasterizer disabled (the default ``pip install gsplat`` in the Isaac
container), a scene download/load error -- to the procedural gradient panorama
with only a logged warning. The user asked for the photoreal path and silently
got the fallback; the repo's own gallery shipped the beige gradient. These
tests pin the corrected contract:

* a GS background that cannot initialize **raises**, both for an explicit
  ``--gsplat-ply`` and for the default ``prefer_gs=True`` resolution:
  capability and preset failures as ``RuntimeError`` carrying the
  pre-built-wheel install hint, and a caller-supplied ``gsplat_ply`` that
  fails to construct as-is (e.g. ``FileNotFoundError`` for a typo'd path),
  because the install hint is the wrong diagnosis there;
* ``allow_fallback=True`` (the CLIs' ``--allow-fallback``) restores the old
  demotion, still logged -- for every failure mode on both GS paths,
  construction included;
* the explicit panorama paths are untouched.

No gsplat/CUDA needed: the rasterizer probe is stubbed at the library seam the
resolver calls through.
"""

from __future__ import annotations

import logging

import pytest

from examples.isaac_gs.background import DEFAULT_GS_SCENE, resolve_background
from strands_robots import rendering
from strands_robots.rendering import PanoramaBackground


@pytest.fixture
def rasterizer_unavailable(monkeypatch):
    """Stub the capability probe to the failure every default install hits."""
    monkeypatch.setattr(
        rendering,
        "gsplat_rasterizer_available",
        lambda: (False, "gsplat CUDA rasterizer unavailable (stubbed)"),
    )


@pytest.fixture
def rasterizer_ok_but_construction_fails(monkeypatch):
    """A working rasterizer whose ``GsplatBackground`` construction raises.

    This is what a typo'd ``--gsplat-ply`` looks like on a machine with a
    working CUDA rasterizer: the probe passes, and the constructor guard
    (exists + readable, issue #2321) raises. Stubbed at the library seam
    ``resolve_background`` imports through, so the construction branch is
    exercised without a GPU.
    """
    monkeypatch.setattr(rendering, "gsplat_rasterizer_available", lambda: (True, "ok"))

    def _boom(*_args, **_kwargs):
        raise FileNotFoundError("Gaussian Splat not found: capture.spz")

    monkeypatch.setattr(rendering, "GsplatBackground", _boom)


class TestExplicitPlyRequest:
    def test_unavailable_rasterizer_raises_with_install_hint(self, rasterizer_unavailable) -> None:
        with pytest.raises(RuntimeError, match="gsplat") as excinfo:
            resolve_background(gsplat_ply="capture.ply")
        message = str(excinfo.value)
        # The remedy is named: the pre-built wheel (README's install line) and
        # the explicit opt-in for the demotion.
        assert "pre-built" in message
        assert "--allow-fallback" in message

    def test_allow_fallback_demotes_to_panorama_with_a_warning(self, rasterizer_unavailable, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="isaac_gs.background"):
            bg = resolve_background(gsplat_ply="capture.ply", allow_fallback=True)
        assert isinstance(bg, PanoramaBackground)
        assert any("falling back to procedural panorama" in r.getMessage() for r in caplog.records)


class TestAllowFallbackSurvivesConstruction:
    """The flag covers a *construction* failure, not only the probe.

    The constructor validation (issue #2321) moved a missing/unreadable
    ``ply_path`` from a first-``render()`` failure to a construction failure,
    which used to sit outside the ``try`` on the explicit-``gsplat_ply``
    path -- so ``--gsplat-ply <typo> --allow-fallback`` crashed rather than
    demoting, the one mode whose documented purpose is to prefer rendering
    *something* over failing.
    """

    def test_construction_failure_demotes_when_fallback_allowed(
        self, rasterizer_ok_but_construction_fails, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="isaac_gs.background"):
            bg = resolve_background(gsplat_ply="capture.spz", allow_fallback=True)
        assert isinstance(bg, PanoramaBackground)
        assert any("falling back to procedural panorama" in r.getMessage() for r in caplog.records)

    def test_construction_failure_still_raises_by_default(self, rasterizer_ok_but_construction_fails) -> None:
        # A bare re-raise, not a RuntimeError wrap: the install hint diagnoses
        # a missing CUDA wheel, which is the wrong remedy for a typo'd path.
        with pytest.raises(FileNotFoundError, match="capture.spz"):
            resolve_background(gsplat_ply="capture.spz")


class TestDefaultGsResolution:
    def test_default_resolution_raises_when_gs_cannot_initialize(self, rasterizer_unavailable) -> None:
        # The default (no flags) resolution *is* a request for the photoreal
        # path -- it must not silently produce the gradient.
        with pytest.raises(RuntimeError, match="failed to initialize") as excinfo:
            resolve_background()
        message = str(excinfo.value)
        assert DEFAULT_GS_SCENE in message
        assert "pre-built" in message
        assert "--allow-fallback" in message

    def test_named_scene_failure_names_the_scene(self, rasterizer_unavailable) -> None:
        with pytest.raises(RuntimeError, match="bonsai"):
            resolve_background(gsplat_scene="bonsai (indoor tabletop)")

    def test_download_or_load_failure_raises_with_the_cause_chained(self, monkeypatch) -> None:
        # A failure *after* the probe (download / scene load) is the same
        # class of silent demotion; the original exception stays chained so
        # the report attributes the real cause.
        monkeypatch.setattr(rendering, "gsplat_rasterizer_available", lambda: (True, "ok"))

        def _boom(name):
            raise OSError("download failed (stubbed)")

        monkeypatch.setattr(rendering, "download_gsplat_scene", _boom)
        with pytest.raises(RuntimeError, match="download failed") as excinfo:
            resolve_background()
        assert isinstance(excinfo.value.__cause__, OSError)

    def test_allow_fallback_demotes_to_panorama(self, rasterizer_unavailable, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="isaac_gs.background"):
            bg = resolve_background(allow_fallback=True)
        assert isinstance(bg, PanoramaBackground)
        assert any("falling back to procedural panorama" in r.getMessage() for r in caplog.records)


class TestExplicitPanoramaPathsAreUntouched:
    def test_prefer_gs_false_returns_panorama_without_probing(self, monkeypatch) -> None:
        def _fail():  # pragma: no cover - must not be called
            raise AssertionError("the panorama path must not probe the rasterizer")

        monkeypatch.setattr(rendering, "gsplat_rasterizer_available", _fail)
        assert isinstance(resolve_background(prefer_gs=False), PanoramaBackground)

    def test_panorama_image_request_returns_panorama(self, tmp_path, monkeypatch) -> None:
        def _fail():  # pragma: no cover - must not be called
            raise AssertionError("the panorama path must not probe the rasterizer")

        monkeypatch.setattr(rendering, "gsplat_rasterizer_available", _fail)
        bg = resolve_background(panorama=str(tmp_path / "pano.jpg"))
        assert isinstance(bg, PanoramaBackground)


class TestCliExposesTheOptIn:
    def test_render_demo_parser_accepts_allow_fallback_and_defaults_off(self) -> None:
        from examples.isaac_gs.render_demo import _build_parser

        parser = _build_parser()
        assert parser.parse_args([]).allow_fallback is False
        assert parser.parse_args(["--allow-fallback"]).allow_fallback is True
