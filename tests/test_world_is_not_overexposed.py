"""The default world a VLA sees: a textured ground and a mean pixel with headroom, not white on white.

A camera trained on an overexposed white arm on a white plane learns a white blob; colour and edge cues that
transfer to a real table are absent. The default ``Simulation()`` ground is a checker texture under toned-down
lighting; ``ground=`` chooses the look, and a bad value is refused up front.
"""

from __future__ import annotations

import pytest

from strands_robots import Robot, Simulation


def _render_default(ground: bool | str) -> tuple[float, int]:
    sim = Simulation(ground=ground)
    try:
        Robot("so101", sim=sim)
        sim.camera("cam", position=[0.6, -0.6, 0.5], target=[0.2, 0, 0.05], size=(320, 240))
        img = sim.render("cam")
        return float(img.mean()), int(sim.backend.model.ntex)
    finally:
        sim.close()


def test_the_default_world_is_textured_and_not_overexposed() -> None:
    mean, ntex = _render_default(True)
    assert ntex >= 1, "the default ground authors a texture, so a camera sees edges not a white blob"
    assert mean < 180.0, f"the default world is not blown out white on white (mean {mean:.1f}/255)"


def test_plain_ground_has_no_texture() -> None:
    _, ntex = _render_default("plain")
    assert ntex == 0, "a plain ground is a flat grey plane with no texture"


def test_no_ground_adds_no_plane() -> None:
    sim = Simulation(ground=False)
    try:
        Robot("so101", sim=sim)
        names = [sim.backend.model.geom(i).name for i in range(sim.backend.model.ngeom)]
        assert "ground" not in names, "ground=False builds no ground plane"
    finally:
        sim.close()


def test_a_ground_that_is_not_a_known_look_is_refused() -> None:
    with pytest.raises(ValueError, match="ground="):
        Simulation(ground="marble")
