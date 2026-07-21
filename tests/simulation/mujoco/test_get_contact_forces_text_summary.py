"""Regression tests for the ``get_contact_forces`` human-readable text summary.

``MuJoCoSimulation.get_contact_forces`` returns both a machine-readable JSON
block and a human-readable ``text`` block that an agent reads directly. Two
edge cases of that text rendering were previously unpinned:

1. No active contacts. When ``data.ncon`` is zero the tool returns a single
   ``text`` block reading ``"No active contacts."`` and no JSON block, rather
   than an empty ``"0 contacts:"`` header. An agent keys off this exact phrase
   to decide the scene is settled / floating.
2. Long contact lists are truncated. The summary lists at most 15 per-contact
   detail lines followed by ``"  ... and <N-15> more"`` so a scene with dozens
   of simultaneous contacts does not flood the agent's context, while the JSON
   block still carries every contact and the header still states the true total.

These pin the text contract so it cannot silently regress into an empty-list
header, an untruncated wall of lines, or a dropped total count.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# A single free box floating two metres above the floor: ``mj_forward`` (run
# inside ``get_contact_forces``) reports zero contacts, exercising the empty
# branch with a valid, compiled world (distinct from the no-world error path).
_FLOATING_SCENE = """
<mujoco model="floating_box">
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.01" pos="0 0 0"/>
    <body name="floater" pos="0 0 2.0">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _many_penetrating_boxes_scene(count: int = 8) -> str:
    """A grid of free boxes penetrating the floor plane.

    Each box-plane overlap generates several corner contacts, so a handful of
    boxes reliably produces well over 15 simultaneous contacts and forces the
    truncation branch. Boxes are spaced apart so they contact only the floor,
    not each other, keeping the count a simple function of box geometry.
    """
    bodies = "".join(
        f'<body name="b{i}" pos="{(i % 5) * 0.6} {(i // 5) * 0.6} 0.02">'
        f'<freejoint/><geom type="box" size="0.1 0.1 0.1"/></body>'
        for i in range(count)
    )
    return (
        '<mujoco model="many_boxes"><worldbody>'
        '<light pos="0 0 3" dir="0 0 -1"/>'
        '<geom name="floor" type="plane" size="10 10 0.01" pos="0 0 0"/>'
        f"{bodies}</worldbody></mujoco>"
    )


def _load(sim: Simulation, xml: str, tmp_path: Path, name: str) -> None:
    scene = tmp_path / name
    scene.write_text(xml)
    assert sim.load_scene(str(scene))["status"] == "success"


def test_get_contact_forces_reports_no_active_contacts_when_empty(tmp_path: Path) -> None:
    """A floating body yields the exact ``"No active contacts."`` phrase, no JSON block."""
    sim = Simulation()
    try:
        _load(sim, _FLOATING_SCENE, tmp_path, "floating.xml")
        result = sim.get_contact_forces()

        assert result["status"] == "success"
        # Single human-readable block, no JSON payload for the empty case.
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == "No active contacts."
        assert all("json" not in block for block in result["content"])
    finally:
        sim.cleanup()


def test_get_contact_forces_truncates_summary_beyond_fifteen(tmp_path: Path) -> None:
    """>15 contacts: header states the true total, 15 detail lines, then a truncation notice."""
    sim = Simulation()
    try:
        _load(sim, _many_penetrating_boxes_scene(), tmp_path, "many.xml")
        result = sim.get_contact_forces()

        assert result["status"] == "success"
        json_block = next(b["json"] for b in result["content"] if "json" in b)
        total = len(json_block["contacts"])
        # The scene is designed to overflow the 15-line cap.
        assert total > 15, f"expected >15 contacts to force truncation, got {total}"

        text = next(b["text"] for b in result["content"] if "text" in b)
        lines = text.splitlines()

        # Header carries the true total, not the truncated line count.
        assert lines[0] == f"{total} contacts:"
        # Exactly 15 per-contact detail lines are rendered.
        detail_lines = [ln for ln in lines if "normal=" in ln and "dist=" in ln]
        assert len(detail_lines) == 15
        # The final line accounts for the omitted remainder.
        assert lines[-1] == f"  ... and {total - 15} more"
    finally:
        sim.cleanup()
