import json

import pytest

from strands_robots import Description
from strands_robots.description import normalize_name, registry, urdf_registry
from strands_robots.description.introspect.mujoco import introspect_mujoco as _introspect_mujoco
from strands_robots.description.introspect.xml import introspect_xml as _introspect_xml

# The gate: every robot in the registry either loads a Description or refuses by name.
_CANONICAL = sorted({e["name"] for e in registry().values()})
_NO_MODEL = {
    n
    for n in _CANONICAL
    if not registry()[n].get("asset", {}).get("robot_descriptions_module")
    and not (registry()[n].get("asset", {}).get("source") or {}).get("type")
    and n not in urdf_registry()
}
_NO_ACTUATORS = {"asimov_v0"}  # its MJCF ships no <actuator> block; the PD gains live in the RL environment


@pytest.mark.network
@pytest.mark.parametrize("name", _CANONICAL)
def test_every_registry_robot_loads_a_description(name):
    if name in _NO_MODEL:
        with pytest.raises(FileNotFoundError, match=f"robot {name!r} is in the registry but ships no"):
            Description.load(name)
        return
    d = Description.load(name)
    assert d.name == name and d.model_path.is_file()
    assert len(d.action_keys) == len(d.actuators) == d.dof
    assert (d.dof == 0) == (name in _NO_ACTUATORS)
    assert set(d.limits) <= set(d.joints)
    assert all(lo <= hi for lo, hi in d.limits.values())
    json.dumps(d.describe())


def test_the_registry_has_76_robots_of_which_12_ship_no_model():
    assert len(_CANONICAL) == 76 and len(_NO_MODEL) == 12


def test_registry_has_verbatim_entries_and_aliases():
    r = registry()
    assert r["so101"]["asset"]["robot_descriptions_module"] == "so_arm101_mj_description"
    assert r["g1"] is r["unitree_g1"]
    assert normalize_name(" So-101 ") == "so_101"


@pytest.mark.parametrize("name", ["so101", "SO101", "so101_follower"])
def test_so101_resolves_with_labels(name):
    d = Description.load(name)
    assert d.name == "so101" and d.format == "mjcf" and d.model_path.is_file()
    assert d.action_keys == ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    assert d.limits["gripper"][1] > d.limits["gripper"][0]
    assert not d.floating


def test_g1_is_floating_with_stand_keyframe():
    d = Description.load("g1")
    assert d.name == "unitree_g1" and d.dof == 29 and d.floating and "stand" in d.keyframes


def test_xml_walk_agrees_with_mujoco():
    for name in ("so101", "g1"):
        d = Description.load(name)
        a, b = _introspect_mujoco(d.model_path), _introspect_xml(d.model_path, "mjcf")
        assert a["action_joints"] == b["action_joints"]
        assert a["keyframes"] == b["keyframes"] and a["floating"] == b["floating"]


def test_unknown_robot_names_the_problem():
    with pytest.raises(KeyError, match="unknown robot 'nosuchbot'"):
        Description.load("nosuchbot")
    with pytest.raises(FileNotFoundError):
        Description.load("/nope/robot.xml")


def test_describe_is_json_safe():
    json.dumps(Description.load("so101").describe())


# github-sourced assets (registry asset.source.type == "github")


def test_github_asset_is_fetched_once_into_the_cache(tmp_path, monkeypatch):
    """A registry robot whose asset lives in a GitHub subdir is downloaded into the cache and then reused."""
    from strands_robots.description import assets

    calls: list[tuple[str, str]] = []

    def fake_fetch(repo: str, subdir: str, dest, *, timeout: float = 120.0):
        calls.append((repo, subdir))
        dest.mkdir(parents=True)
        (dest / "robot_allcollisions.xml").write_text(
            '<mujoco><worldbody><body name="b"><joint name="j" range="-1 1"/><geom size="0.1"/></body>'
            '</worldbody><actuator><position name="a" joint="j"/></actuator></mujoco>'
        )
        return 1234

    monkeypatch.setenv("STRANDS_ROBOTS_CACHE", str(tmp_path))
    monkeypatch.setattr(assets, "_fetch_github_subdir", fake_fetch)
    entry = {
        "asset": {
            "dir": "microduck",
            "model_xml": "robot_allcollisions.xml",
            "source": {"type": "github", "repo": "pollen-robotics/microduck_rl", "subdir": "src/x"},
        }
    }
    p1 = assets.resolve(entry)
    p2 = assets.resolve(entry)
    assert p1 == p2 == tmp_path / "microduck" / "robot_allcollisions.xml"
    assert calls == [("pollen-robotics/microduck_rl", "src/x")]  # second call hit the cache
    d = Description.load(p1)
    assert d.action_keys == ("j",)


def test_github_asset_rejects_bad_repo_and_escaping_subdir(tmp_path, monkeypatch):
    from strands_robots.description import assets

    monkeypatch.setenv("STRANDS_ROBOTS_CACHE", str(tmp_path))
    with pytest.raises(ValueError, match="repo"):
        assets.resolve(
            {"asset": {"dir": "x", "model_xml": "a.xml", "source": {"type": "github", "repo": "https://evil"}}}
        )
    with pytest.raises(ValueError, match="subdir"):
        assets.resolve(
            {
                "asset": {
                    "dir": "x",
                    "model_xml": "a.xml",
                    "source": {"type": "github", "repo": "a/b", "subdir": "../etc"},
                }
            }
        )
    with pytest.raises(ValueError, match="source type"):
        assets.resolve({"asset": {"dir": "x", "model_xml": "a.xml", "source": {"type": "ftp"}}})


def test_lfs_pointer_is_recognised():
    from strands_robots.description.assets import _lfs_pointer

    oid = "a" * 64
    text = f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize 42\n".encode()
    assert _lfs_pointer(text) == (oid, 42)
    assert _lfs_pointer(b"solid mesh\n...") is None


@pytest.mark.network
def test_microduck_downloads_for_real_and_loads(tmp_path, monkeypatch):
    """Real network: the one measured download (size + time logged by the test)."""
    import time

    from strands_robots.description import assets

    monkeypatch.setenv("STRANDS_ROBOTS_CACHE", str(tmp_path))
    t0 = time.perf_counter()
    d = Description.load("microduck")
    dt = time.perf_counter() - t0
    size = sum(p.stat().st_size for p in (tmp_path / "microduck").rglob("*") if p.is_file())
    print(f"[measured] microduck github fetch: {size / 1e6:.1f} MB in {dt:.1f} s -> {d.model_path}")
    assert d.model_path.is_file() and d.dof == 14 and d.floating
    assert not any(assets._lfs_pointer(p.read_bytes()[:256]) for p in d.package_dir.rglob("*.stl"))


@pytest.mark.network
def test_microduck_stands_in_the_default_world():
    """Robot("microduck") attaches to Simulation.default() like any registry robot (uses the real cache)."""
    from strands_robots import Robot

    duck = Robot("microduck")
    obs = duck.observe(images=False)
    assert len(obs.joints.positions) == 14 and duck.sim is not None and duck.sim.describe()["robots"]
    duck.act(obs.joints.positions)
    assert duck.sim.time > 0
