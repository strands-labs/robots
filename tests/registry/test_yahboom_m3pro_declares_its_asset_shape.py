"""The ``yahboom_m3pro`` entry describes the robot it points at.

The Yahboom ROSMASTER M3 Pro is a mecanum chassis carrying the DOFBOT-Pro arm:
five bus-servo joints plus a gripper. Its MJCF is generated from the vendor
URDF and hosted at ``dimwael/yahboom_m3pro_description``; the model has ten
joints (three kinematic base joints, five arm hinges, two gripper cranks) and
nine actuators (three base velocities, five arm servos, one gripper servo).

``joints`` follows the ``njnt`` convention the catalog documents ("Joint counts
include any free joints / gripper actuators"), which for this model is 10:
``base_x``, ``base_y``, ``base_yaw``, ``arm1..arm5``, ``rlink1``, ``llink1``.
The description carries the *hardware* figure - a 6-DOF arm - the way ``op3``
and ``unitree_h1`` do. The registry has no same-model sibling for this robot, so
``tests/registry/test_asset_family_joint_counts.py`` says nothing about it; this
file states the convention the number was written against.

Everything here is graded from ``robots.json`` alone, so it holds on any
install: no MuJoCo, no downloaded assets, no network. The compiled-model claims
(joint names, actuator count, camera intrinsics, gripper travel) live in
``tests_integ/simulation/test_yahboom_m3pro_sim.py``, which downloads the asset.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ROBOTS_JSON = REPO_ROOT / "strands_robots" / "registry" / "robots.json"

#: Every joint the compiled model declares, in model order.
MODEL_JOINTS: tuple[str, ...] = (
    "base_x",
    "base_y",
    "base_yaw",
    "arm1",
    "arm2",
    "arm3",
    "arm4",
    "arm5",
    "rlink1",
    "llink1",
)


def _entry() -> dict:
    return json.loads(ROBOTS_JSON.read_text(encoding="utf-8"))["robots"]["yahboom_m3pro"]


class TestTheEntryIsWellFormed:
    """Graded from ``robots.json`` alone, so it holds with no MuJoCo installed."""

    def test_the_registry_declares_a_mobile_manipulator(self) -> None:
        entry = _entry()
        assert entry["category"] == "mobile_manip"
        assert "M3 Pro" in entry["description"]
        assert "6-DOF" in entry["description"], "the hardware DOF a reader sizes an arm action from"

    def test_the_declared_count_is_the_model_njnt(self) -> None:
        assert _entry()["joints"] == len(MODEL_JOINTS)

    def test_the_asset_declares_a_github_download_source(self) -> None:
        """A github source, so the asset is fetchable without a naming guess."""
        asset = _entry()["asset"]
        assert asset["dir"] == "yahboom_m3pro"
        assert asset["model_xml"] == "m3pro/m3pro.xml"
        assert asset["scene_xml"] == "scene.xml"
        source = asset["source"]
        assert source["type"] == "github"
        assert source["repo"] == "dimwael/yahboom_m3pro_description"
        assert source["subdir"] == "mjcf"

    def test_the_gripper_block_names_the_servo_and_its_closed_end(self) -> None:
        """``move_to`` and ``set_gripper`` read this rather than guessing from names."""
        gripper = _entry()["gripper"]
        assert gripper["actuators"] == ["gripper"]
        assert gripper["closed"] == "low" and gripper["open"] == "high"

    def test_no_hardware_block_until_a_driver_exists(self) -> None:
        """Declaring ``lerobot_type`` for a robot lerobot cannot build would make
        ``mode="real"`` fail late, at the bus, instead of at the factory."""
        assert "hardware" not in _entry()

    def test_no_alias_repeats_the_canonical_name(self) -> None:
        """A self-alias makes every registry read raise, not just this one."""
        assert "yahboom_m3pro" not in _entry()["aliases"]

    def test_the_declared_aliases_resolve_to_the_robot(self) -> None:
        from strands_robots.registry import resolve_name

        aliases = _entry()["aliases"]
        assert {"m3pro", "rosmaster_m3_pro"} <= set(aliases)
        for alias in aliases:
            assert resolve_name(alias) == "yahboom_m3pro"
        assert resolve_name("YAHBOOM-M3PRO") == "yahboom_m3pro"
