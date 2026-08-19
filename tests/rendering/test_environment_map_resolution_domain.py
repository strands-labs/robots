# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The environment-map resolution knobs are validated before anything is rendered.

``face_size``, ``equi_w`` and ``equi_h`` size the pixel grid of every
environment map this module bakes. ``derive_key_light``, in the same module,
checks both of its own knobs -- ``brightest_fraction`` four ways and
``upper_hemisphere`` on the shared flag domain -- and validates the map's shape;
the three functions that take the resolutions checked none of them. What follows
is pinned here.

A zero-sized grid produced a map rather than a refusal. ``equi_w=0`` returned a
``(H, 0, 3)`` array of no pixels, reporting success, and only after paying six
full background renders -- GPU-bound for a ``GsplatBackground``. The refusal the
caller eventually saw came from ``derive_key_light`` and blamed the scene: "the
map is black above the horizon -- pass ``upper_hemisphere=False`` to search the
full sphere". Following that advice fails the same way, because the map has no
texels at all rather than a dark upper half, so the only remedy offered was a
dead end and the resolution the caller had asked for was named nowhere.

Only the domain is pinned here, not the quality a resolution buys. A very coarse
cube face is still a usable one, and there is no sharp boundary to hold it to:
the reprojection scales by ``face_size - 1``, so 1, 2 and 3 each resolve almost
nothing within a face (measured on a background carrying a gradient inside every
face: 4, 4 and 5 distinct colours in the whole map, growing smoothly from 4
upward). Where "too coarse" begins is a judgement about acceptable quality rather
than a value the module cannot use, so ``face_size=1`` is still accepted and the
tests below assert only that a positive whole number reaches the bake.

That shared domain accepts an integral float and a NumPy integer ("a ``30.0``
computed from a config float passes"), which these three then could not index a
grid with -- ``equi_w=32.0`` raised a bare ``TypeError`` from ``np.zeros`` and
``face_size=8.0`` an ``IndexError``. They are normalized with ``int()`` after the
check, the way ``HybridCompositor`` normalizes its own ``default_width`` /
``default_height``. In the cache path that normalization is load-bearing rather
than cosmetic: the name IS the cache key, so ``2048`` and ``2048.0`` spelled two
files for one set of pixels and a bake already on disk was paid for twice.

The domain is :func:`~strands_robots.utils.positive_whole_number_error`, the
shared one for a knob counting pixels, so what this refuses cannot diverge from
the recorders' ``width``/``height`` or ``encode_clip``'s rate. It is also the
authority these tests parametrize over, so a value added there is covered here
without an edit.
"""

import re
from typing import Any

import numpy as np
import pytest

from strands_robots.rendering import (
    bake_environment_map,
    derive_key_light,
    environment_map_cache_path,
    render_environment_map,
)
from strands_robots.utils import positive_whole_number_error

# A resolution every entry point can honor, small enough to keep the fake
# background cheap.
GOOD = {"face_size": 8, "equi_w": 32, "equi_h": 16}

# Values the shared domain refuses. Parametrized over rather than hand-listed so
# the two cannot drift; each is asserted to be out of domain as a premise.
OUT_OF_DOMAIN = [0, -8, 2.5, True, False, float("nan"), float("inf"), None, "16"]

RESOLUTION_PARAMS = ["face_size", "equi_w", "equi_h"]


def _sizes(**overrides: Any) -> dict[str, Any]:
    """The usable resolution with ``overrides`` applied, as loose kwargs.

    Typed ``Any`` deliberately: half of these cases pass a value the signatures
    reject, which is the point of the test.
    """
    return {**GOOD, **overrides}


class WindowBackground:
    """Warm window on the +X wall of a dimly bounced room.

    Direction-aware, like a real background: the face direction is the camera
    rotation's ``-Z`` (the bake builds ``T_world_cam`` columns
    ``[right, up, -fwd]``), and only the ``+X`` face carries the window. A
    background that painted every face alike would hide what a coarse cube face
    costs, since there would be nothing inside a face to lose.

    Records how many renders it was asked for, so a refusal that arrives before
    the bake can be told from one that arrives after it.
    """

    name = "window"

    def __init__(self) -> None:
        self.renders = 0

    def render(self, cam):
        self.renders += 1
        w, h = int(cam.width), int(cam.height)
        fwd = -np.asarray(cam.T_world_cam, dtype=float)[:3, 2]
        rgb = np.zeros((h, w, 3), np.uint8)
        rgb[:, :] = (46, 54, 72)
        if fwd[0] > 0.9:
            top = int(h * 0.30)
            rgb[top : max(top + 1, int(h * 0.55)), max(0, w // 2 - w // 6) : w // 2 + max(1, w // 6)] = (
                252,
                206,
                128,
            )
        return rgb, np.full((h, w), 1e3, np.float32)


class TestAZeroSizedGridIsRefusedRatherThanBaked:
    """The headline: a grid with no pixels must not be reported as a map."""

    @pytest.mark.parametrize("param", ["equi_w", "equi_h"])
    def test_a_zero_sized_equirect_is_refused_naming_the_knob(self, param: str) -> None:
        bg = WindowBackground()
        try:
            env = render_environment_map(bg, **_sizes(**{param: 0}))
        except ValueError as exc:
            assert param in str(exc), f"the refusal must name {param}, got {exc}"
            assert bg.renders == 0, f"refused after {bg.renders} background renders; check before rendering"
            return
        # Pre-fix: a map was returned. Report what the caller is left holding,
        # including the remedy the only refusal they ever see advertises.
        message = ""
        try:
            derive_key_light(env)
        except ValueError as exc:
            message = str(exc)
        remedy = re.search(r"pass (upper_hemisphere=\w+)", message)
        followed = "no remedy was offered"
        if remedy:
            try:
                derive_key_light(env, upper_hemisphere=False)
                followed = "the remedy worked"
            except ValueError as exc:
                followed = f"following it fails again: {exc}"
        raise AssertionError(
            f"{param}=0 returned a map of shape {env.shape} ({env.size} pixels) after "
            f"{bg.renders} background renders instead of naming {param}. The caller's only "
            f"refusal blames the scene: {message!r} -- and {followed}."
        )

    def test_the_scene_diagnosis_is_not_reachable_from_an_empty_grid(self) -> None:
        """A resolution mistake must not be reported as a dark scene."""
        with pytest.raises(ValueError) as excinfo:
            render_environment_map(WindowBackground(), **_sizes(equi_w=0))
        text = str(excinfo.value)
        assert "upper_hemisphere" not in text, f"a resolution refusal must not advise a search flag: {text}"
        assert "black" not in text, f"a resolution refusal must not blame the scene: {text}"


class TestEveryEntryPointSharesTheDomain:
    """The render, the bake and the cache path agree on what they can honor."""

    @pytest.mark.parametrize("bad", OUT_OF_DOMAIN)
    @pytest.mark.parametrize("param", RESOLUTION_PARAMS)
    def test_render_refuses_what_the_shared_domain_refuses(self, param: str, bad: object) -> None:
        assert positive_whole_number_error(bad, param, "ctx") is not None, "premise: out of the shared domain"
        bg = WindowBackground()
        with pytest.raises(ValueError, match=re.escape(param)):
            render_environment_map(bg, **_sizes(**{param: bad}))
        assert bg.renders == 0

    @pytest.mark.parametrize("bad", OUT_OF_DOMAIN)
    @pytest.mark.parametrize("param", RESOLUTION_PARAMS)
    def test_bake_refuses_it_too_and_writes_nothing(self, param: str, bad: object, tmp_path) -> None:
        out = tmp_path / f"env_{param}.png"
        with pytest.raises(ValueError, match=re.escape(param)):
            bake_environment_map(WindowBackground(), out, **_sizes(**{param: bad}))
        assert not out.exists()

    @pytest.mark.parametrize("bad", OUT_OF_DOMAIN)
    @pytest.mark.parametrize("param", RESOLUTION_PARAMS)
    def test_the_cache_path_names_no_file_for_a_refused_resolution(self, param: str, bad: object, tmp_path) -> None:
        with pytest.raises(ValueError, match=re.escape(param)):
            environment_map_cache_path(tmp_path / "scene.ply", (0.0, 0.0, 0.4), **_sizes(**{param: bad}))


class TestTheAcceptedDomainIsUnchanged:
    """Controls: what worked before still works, and no floor was invented."""

    def test_a_usable_resolution_still_bakes(self, tmp_path) -> None:
        bg = WindowBackground()
        env = render_environment_map(bg, **_sizes())
        assert env.shape == (GOOD["equi_h"], GOOD["equi_w"], 3)
        assert bg.renders == 6
        out = tmp_path / "env.png"
        baked = bake_environment_map(WindowBackground(), out, **_sizes())
        assert baked == out
        assert out.stat().st_size > 0

    @pytest.mark.parametrize("param", RESOLUTION_PARAMS)
    def test_an_integral_float_and_a_numpy_int_are_still_accepted(self, param: str) -> None:
        """The shared domain accepts these, so the entry points must too."""
        for value in (float(GOOD[param]), np.int64(GOOD[param])):
            assert positive_whole_number_error(value, param, "ctx") is None, "premise: in the shared domain"
            env = render_environment_map(WindowBackground(), **_sizes(**{param: value}))
            assert env.size > 0

    def test_an_integral_float_names_the_same_cache_entry_as_the_int(self, tmp_path) -> None:
        """The name is a cache key, so one resolution must spell one file."""
        scene = tmp_path / "scene.ply"
        as_int = environment_map_cache_path(scene, (0.0, 0.0, 0.4), **_sizes())
        as_float = environment_map_cache_path(
            scene, (0.0, 0.0, 0.4), **_sizes(**{k: float(v) for k, v in GOOD.items()})
        )
        assert as_int == as_float, (
            f"the same resolution spelled two cache entries ({as_int.name} vs {as_float.name}), so a bake "
            "already on disk is missed and paid for again"
        )

    @pytest.mark.parametrize("param", ["equi_w", "equi_h"])
    def test_a_single_texel_equirect_axis_is_still_accepted(self, param: str) -> None:
        """A one-texel axis is small but usable, so it stays accepted."""
        env = render_environment_map(WindowBackground(), **_sizes(**{param: 1}))
        assert env.shape[0 if param == "equi_h" else 1] == 1
