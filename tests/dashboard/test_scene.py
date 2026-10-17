"""The browser's copy of the world: one MJCF built from THE world's Scene, its assets, robot cards."""

from __future__ import annotations

from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("fastapi")

from strands_robots import Robot, Simulation  # noqa: E402
from strands_robots.dashboard import Dashboard  # noqa: E402


@pytest.fixture
def world():
    sim = Simulation()
    Robot("so101", sim=sim)
    Robot("unitree_g1", sim=sim, position=[1.0, 0.0, 0.0])
    yield sim
    sim.close()


def test_scene_json_lists_both_robots_and_compiles_in_mujoco(world, tmp_path: Path) -> None:
    dash = Dashboard(world)
    doc = dash.scene_json()
    assert [r["id"] for r in doc["robots"]] == ["so101", "unitree_g1"]
    assert doc["revision"] == world.revision
    for a in doc["assets"]:
        dest = tmp_path / a["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(dash.asset_file(a["path"][len("assets/") :])).read_bytes())
    (tmp_path / "scene.xml").write_text(doc["xml"])
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
    assert model.nmesh == doc["stats"]["nmesh"] > 0
    for card in doc["robots"]:
        for joint in card["joints"]:
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) >= 0, joint


def test_frame_carries_joint_values_and_root_poses(world) -> None:
    dash = Dashboard(world)
    roots = {r["id"]: r["root"] for r in dash.scene_json()["robots"]}
    frame = dash.frame()
    assert set(frame["bodies"]) == {"so101", "unitree_g1"}
    for rid, body in frame["bodies"].items():
        assert body["root"] == roots[rid]
        assert len(body["pose"]) == 7
        assert body["q"]


def test_the_dashboard_reads_the_world_it_was_given_not_a_copy(world) -> None:
    dash = Dashboard(world)
    assert dash.sim is world
    before = dash.scene_json()["revision"]
    Robot("so100", sim=world, position=[0, 1, 0])
    after = dash.scene_json()
    assert after["revision"] == world.revision != before
    assert "so100" in [r["id"] for r in after["robots"]]


def test_unknown_asset_is_refused(world) -> None:
    with pytest.raises(KeyError):
        Dashboard(world).asset_file("../../etc/passwd")
