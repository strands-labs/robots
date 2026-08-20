# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GsplatBackground``'s alignment flags are checked, not read by truthiness.

``auto_backdrop``, ``skybox``, ``metric`` and ``own_floor`` each select a
branch: whether a fitted ``world_from_gs`` stands the capture up, whether the
scene keeps its own scale or is scaled to ``radius``, and whether the
compositor hides MuJoCo's grid ground. None of the four had a check, and all
four were read by truthiness -- ``bool(metric)``, ``bool(own_floor)`` and, for
the other two, the truthiness of ``skybox and transform is None``.

Two outcomes follow, and both are pinned here.

Every spelling of *off* a caller reaches for is a truthy string, so
``skybox="false"``, ``metric="no"`` and ``own_floor="off"`` each selected the
branch the value asks to skip. ``metric`` is the sharpest of the four because
it also decides whether ``radius`` is read at all: the fit keeps the capture's
raw scale when it is set, so a truthy string dropped the requested size and
stood the scene up at whatever size it happened to be captured at.
``skybox="false"`` additionally flipped the ``bg_fill`` default from black to
the skybox grey, so an unobserved zenith read as a ceiling in a scene the
caller had asked to leave unaligned.

The undeclared falsy values are the other half. ``0``, ``""``, ``[]`` and
``None`` took the default branch without being a spelling of it, and because
``skybox``/``auto_backdrop`` compose with ``transform is None`` the attribute
then held that raw value rather than a bool -- ``bg._skybox`` was ``None``.

The domain is :func:`~strands_robots.utils.boolean_flag_error`, the shared one
for a flag that selects a posture rather than scaling a quantity, and it is the
authority these tests parametrize over, so a spelling added there is covered
here without an edit. ``HybridCompositor`` in this package already checks its
own ``blend_in_linear`` against the same rule; these four were the remaining
posture flags in ``strands_robots.rendering`` that did not.

Dependency-free: the constructor defers every ``gsplat``/``torch`` import to
the first render, and the scale consequence is measured through the pure-numpy
skybox fit, so nothing here needs the ``sim-gs`` extra.
"""

from typing import Any

import numpy as np
import pytest

from strands_robots.rendering import GsplatBackground
from strands_robots.rendering.backgrounds import _fit_skybox_transform
from strands_robots.utils import boolean_flag_error

#: The constructor's posture flags, in declaration order.
POSTURE_FLAGS: list[str] = ["auto_backdrop", "skybox", "metric", "own_floor"]

#: Values that read as "off" to a human but are truthy to Python, so truthiness
#: selects the *enabled* branch -- the opposite of what they say.
TRUTHY_OFF_SPELLINGS: list[str] = ["false", "False", "no", "off", "0"]

#: Falsy values that are not a declared spelling of the default branch.
UNDECLARED_FALSY: list[object] = [0, "", [], None]


@pytest.fixture
def scene_ply(tmp_path):
    """An existing placeholder scene file: construction validates the path's
    existence and readability but defers the decode to the first render."""
    p = tmp_path / "scene.ply"
    p.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    return p


def _bg(ply_path: Any, **overrides: Any) -> GsplatBackground:
    """Construct a background with arbitrary keyword values.

    Deliberately loose: half of these cases pass values the signature rejects,
    which is exactly what is under test, so the builder does not re-assert the
    declared types.
    """
    return GsplatBackground(ply_path=ply_path, **overrides)


def _stored(bg: GsplatBackground) -> dict[str, object]:
    """The branch each flag actually selected, read off the constructed object."""
    return {
        "auto_backdrop": bg._auto_backdrop,
        "skybox": bg._skybox,
        "metric": bg._metric,
        "own_floor": bg.own_floor,
    }


def _room_points() -> np.ndarray:
    """A captured room: ~6 m x ~5.4 m footprint, ~1.7 m of height."""
    rng = np.random.default_rng(0)
    return np.column_stack(
        [
            rng.uniform(-3.0, 3.0, 4000),
            rng.uniform(-2.7, 2.7, 4000),
            rng.uniform(0.0, 1.66, 4000),
        ]
    )


def _fitted_scale(metric: Any) -> float:
    """The uniform scale the skybox fit applies for a given ``metric`` value."""
    transform = _fit_skybox_transform(_room_points(), radius=2.5, metric=metric)
    return float(np.linalg.norm(transform[:3, 0]))


class TestEveryPostureFlagIsCheckedOnTheSharedDomain:
    """The four flags share one domain, so their refusals cannot diverge."""

    @pytest.mark.parametrize("flag", POSTURE_FLAGS)
    @pytest.mark.parametrize("value", TRUTHY_OFF_SPELLINGS)
    def test_a_truthy_off_spelling_is_refused_rather_than_read_as_on(self, scene_ply, flag, value) -> None:
        # Premise: the shared domain is what refuses these, so this test tracks
        # whatever spellings it refuses rather than a copied list.
        assert boolean_flag_error(value, flag, "GsplatBackground"), f"premise: {value!r} is on the shared domain"

        with pytest.raises(ValueError, match=rf"{flag}"):
            _bg(scene_ply, **{flag: value})

    @pytest.mark.parametrize("flag", POSTURE_FLAGS)
    @pytest.mark.parametrize("value", UNDECLARED_FALSY)
    def test_an_undeclared_falsy_value_is_refused_rather_than_defaulted(self, scene_ply, flag, value) -> None:
        assert boolean_flag_error(value, flag, "GsplatBackground"), f"premise: {value!r} is on the shared domain"

        with pytest.raises(ValueError, match=rf"{flag}"):
            _bg(scene_ply, **{flag: value})

    @pytest.mark.parametrize("flag", POSTURE_FLAGS)
    def test_the_refusal_names_the_flag_and_the_value(self, scene_ply, flag) -> None:
        with pytest.raises(ValueError) as excinfo:
            _bg(scene_ply, **{flag: "false"})

        message = str(excinfo.value)
        assert flag in message
        assert "false" in message


class TestATruthyOffSpellingSelectedTheBranchItAsksToSkip:
    """What the misread cost, measured on the branch each flag decides."""

    def test_metric_off_stood_the_room_up_at_the_size_it_was_captured_at(self, scene_ply) -> None:
        # End to end: the value the constructor stores is what the skybox fit
        # reads, so a truthy "off" is measurable as a room size in metres.
        requested_radius = 2.5
        try:
            bg = _bg(scene_ply, skybox=True, radius=requested_radius, metric="no")
        except ValueError as exc:
            assert "metric" in str(exc)
            return

        points = _room_points()
        transform = _fit_skybox_transform(points, radius=requested_radius, metric=bg._metric)
        placed = (points @ transform[:3, :3].T) + transform[:3, 3]
        got = float(np.percentile(np.linalg.norm(placed[:, :2], axis=1), 95))
        raise AssertionError(
            f"GsplatBackground(metric='no') stored _metric={bg._metric!r}, so the fit kept the "
            f"capture's own scale and placed the room at a {got:.2f} m radius instead of the "
            f"requested {requested_radius:.2f} m."
        )

    @pytest.mark.parametrize("value", TRUTHY_OFF_SPELLINGS)
    def test_metric_off_kept_the_raw_capture_scale_and_dropped_radius(self, value) -> None:
        # ``metric`` decides whether ``radius`` is read at all: set, the fit
        # keeps the capture's own scale. A truthy "off" therefore stood the
        # scene up at whatever size it was captured at, ignoring the requested
        # one -- and the two answers are a whole scene size apart.
        requested = _fitted_scale(False)
        raw_capture = _fitted_scale(True)
        assert requested != pytest.approx(raw_capture), "premise: the two branches disagree on this room"

        assert _fitted_scale(value) == pytest.approx(raw_capture)
        assert _fitted_scale(value) != pytest.approx(requested)

    def test_skybox_off_selected_the_skybox_void_fill(self, scene_ply) -> None:
        # The ``bg_fill`` default is chosen from the skybox branch, so a truthy
        # "off" also painted unobserved regions the skybox grey.
        unaligned = GsplatBackground(ply_path=scene_ply)._bg_fill.tolist()
        skybox = GsplatBackground(ply_path=scene_ply, skybox=True)._bg_fill.tolist()
        assert unaligned != skybox, "premise: the fill default differs by branch"

        with pytest.raises(ValueError):
            _bg(scene_ply, skybox="false")


class TestTheDocumentedSpellingsAreUnchanged:
    """Controls: every branch a caller can legitimately ask for still works."""

    @pytest.mark.parametrize("flag", POSTURE_FLAGS)
    @pytest.mark.parametrize("value", [True, False])
    def test_a_real_bool_selects_the_branch_it_names(self, scene_ply, flag, value) -> None:
        bg = _bg(scene_ply, **{flag: value})

        assert _stored(bg)[flag] is value

    def test_defaults_leave_every_flag_off(self, scene_ply) -> None:
        bg = GsplatBackground(ply_path=scene_ply)

        assert _stored(bg) == {
            "auto_backdrop": False,
            "skybox": False,
            "metric": False,
            "own_floor": False,
        }

    def test_an_explicit_transform_still_gates_the_fitted_modes_off(self, scene_ply) -> None:
        # ``skybox``/``auto_backdrop`` compose with ``transform is None``: an
        # explicit alignment wins over a fitted one. Checking the flag must not
        # change that composition.
        bg = _bg(scene_ply, transform=np.eye(4), skybox=True, auto_backdrop=True)

        assert bg._skybox is False
        assert bg._auto_backdrop is False
        assert bg._explicit_transform is True

    def test_the_shipped_skybox_preset_is_accepted(self, scene_ply) -> None:
        # The documented route splats an authored alignment dict into the
        # constructor, and the curated ``tabletop`` preset carries two of these
        # flags -- so the guard has to accept it unchanged.
        from strands_robots.rendering import gsplat_skybox_align_for

        align = gsplat_skybox_align_for("tabletop")
        assert {"metric", "own_floor"} <= set(align), "premise: the preset carries posture flags"

        bg = _bg(scene_ply, skybox=True, **align)

        assert bg._metric is True
        assert bg.own_floor is True

    def test_the_path_refusals_still_come_first(self, tmp_path, scene_ply) -> None:
        # The path is validated where the caller supplied it; adding the flag
        # guard must not displace that verdict.
        with pytest.raises(FileNotFoundError):
            _bg(tmp_path / "missing.ply", skybox="false")

    def test_a_numeric_knob_beside_them_is_untouched(self, scene_ply) -> None:
        # ``radius`` scales a quantity rather than selecting a posture, so it
        # keeps its own (float-coercing) treatment -- the two kinds are
        # deliberately inverse.
        bg = _bg(scene_ply, skybox=True, radius=4)

        assert bg._radius == pytest.approx(4.0)
