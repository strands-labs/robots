"""ONNX policies: the generic runner, the named presets, ``from_spec`` and the task gates on sim bodies."""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from strands_robots import JointState, Observation, Policy, Pose, Robot, Simulation
from strands_robots.policy import from_spec
from strands_robots.policy.onnx import (
    HolosomaPolicy,
    MicroduckPolicy,
    OnnxPolicy,
    projected_gravity,
    quat_rotate_inverse,
    read_metadata,
)
from strands_robots.policy.onnx.graph import resolve_weights

PKG = Path(__file__).resolve().parents[1] / "strands_robots"
needs_onnx_extra = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("onnxruntime", "huggingface_hub")), reason="needs the [onnx] extra"
)


def test_quat_rotate_inverse_and_projected_gravity():
    up = np.array([1.0, 0, 0, 0])
    assert np.allclose(projected_gravity(up), [0, 0, -1])
    q = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])  # 90 deg about +y: world -z is body +x
    assert np.allclose(projected_gravity(q), [1, 0, 0], atol=1e-9)
    assert np.allclose(quat_rotate_inverse(2 * up, [1, 2, 3]), [1, 2, 3])
    with pytest.raises(ValueError, match="norm"):
        quat_rotate_inverse(np.zeros(4), [0, 0, -1])


@pytest.fixture(scope="module")
def tiny_onnx(tmp_path_factory):
    """y = 2x for a (1, 3) input, with one metadata key."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    node = helper.make_node("Mul", ["obs", "two"], ["actions"])
    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 3])],
        initializer=[two],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    meta = model.metadata_props.add()
    meta.key, meta.value = "joint_names", "a,b,c"
    path = tmp_path_factory.mktemp("onnx") / "tiny.onnx"
    onnx.save(model, str(path))
    return path


def test_onnx_policy_runs_a_graph_and_writes_the_sidecar(tiny_onnx):
    pytest.importorskip("onnxruntime")
    pol = OnnxPolicy(tiny_onnx)
    assert pol.infer(np.array([1.0, 2.0, 3.0])).tolist() == [2.0, 4.0, 6.0]
    assert pol.input_name == "obs" and pol.input_width == 3 and pol.units == "rad"
    assert read_metadata(tiny_onnx) == {"joint_names": "a,b,c"}
    side = tiny_onnx.with_name("tiny.onnx.json")
    assert pol.sidecar == side and json.loads(side.read_text())["metadata"]["joint_names"] == "a,b,c"
    with pytest.raises(ValueError, match="takes 3 inputs"):
        pol.infer(np.zeros(4))
    with pytest.raises(TypeError, match="providers"):
        OnnxPolicy(tiny_onnx, providers="cpu")  # type: ignore[arg-type]


def test_a_users_json_next_to_the_graph_is_never_touched(tiny_onnx):
    pytest.importorskip("onnxruntime")
    own = tiny_onnx.with_suffix(".json")
    own.write_text('{"mine": true}')
    OnnxPolicy(tiny_onnx)
    assert own.read_text() == '{"mine": true}'


def test_a_preset_name_never_resolves_to_a_file_in_cwd(tmp_path, monkeypatch):
    (tmp_path / "w.onnx").write_bytes(b"not the published weights")
    monkeypatch.chdir(tmp_path)
    fetched = tmp_path / "hub" / "w.onnx"
    fetched.parent.mkdir()
    fetched.write_bytes(b"published")
    calls = []
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hub, "hf_hub_download", lambda repo, name: calls.append((repo, name)) or str(fetched))
    good = hashlib.sha256(b"published").hexdigest()
    assert resolve_weights("w.onnx", repo="o/r", sha256=good) == fetched and calls == [("o/r", "w.onnx")]
    with pytest.raises(OSError, match="refusing to run unknown weights"):
        resolve_weights("w.onnx", repo="o/r", sha256="0" * 64)
    with pytest.raises(OSError, match="refusing to run unknown weights"):
        resolve_weights(tmp_path / "w.onnx", sha256=good)


def test_a_query_string_is_the_same_as_keywords(monkeypatch):
    seen: list[dict] = []
    for cls in (MicroduckPolicy, HolosomaPolicy):  # record the keywords; wraps keeps the real signature
        monkeypatch.setattr(cls, "__init__", functools.wraps(cls.__init__)(lambda self, **kw: seen.append(kw)))
    assert isinstance(Policy("microduck?vx=0.3&wz=-0.1"), MicroduckPolicy)
    assert isinstance(from_spec("microduck?vx=0.3&wz=-0.1"), MicroduckPolicy)
    assert isinstance(Policy("microduck", vx=0.3, wz=-0.1), MicroduckPolicy)
    assert seen[0] == seen[1] == seen[2] == {"vx": 0.3, "wz": -0.1}
    assert isinstance(Policy("holosoma?algorithm=ppo"), HolosomaPolicy) and seen[3] == {"algorithm": "ppo"}
    assert isinstance(Policy("holosoma"), HolosomaPolicy) and seen[4] == {}


def test_a_preset_name_is_independent_of_a_directory_of_the_same_name_in_cwd(tmp_path, monkeypatch):
    seen: list[dict] = []
    monkeypatch.setattr(
        HolosomaPolicy, "__init__", functools.wraps(HolosomaPolicy.__init__)(lambda self, **kw: seen.append(kw))
    )
    (tmp_path / "holosoma").mkdir()  # the upstream repo's name, checked out next to the user's script
    monkeypatch.chdir(tmp_path)
    assert isinstance(Policy("holosoma"), HolosomaPolicy) and seen == [{}]
    assert isinstance(Policy("holosoma?algorithm=ppo"), HolosomaPolicy) and seen[1] == {"algorithm": "ppo"}
    with pytest.raises(ValueError, match="neither a checkpoint nor a dataset"):
        Policy("./holosoma")  # a path form stays a path: this directory is neither
    (tmp_path / "nope").mkdir()
    with pytest.raises(ValueError, match="neither a checkpoint nor a dataset"):
        Policy("nope")  # not a known preset: judged as the directory it is


@pytest.mark.parametrize(
    ("spec", "kw", "err", "names"),
    [
        ("nope", {}, ValueError, "known presets: holosoma, microduck"),
        ("microduck?vz=1", {}, ValueError, "unknown parameter 'vz'; known: skill, vx, vy, wz"),
        ("holosoma?speed=1", {}, ValueError, "known: algorithm, vx, vy, wz"),
        ("microduck?vx=fast", {}, ValueError, "expected a number"),
        ("microduck?vx", {}, ValueError, "vx"),
        ("microduck", {"bogus": 1}, TypeError, "unexpected keyword 'bogus'; known: skill, vx, vy, wz"),
    ],
)
def test_unknown_preset_or_key_is_refused_naming_the_known_ones(spec, kw, err, names):
    with pytest.raises(err, match=names.replace("?", r"\?")):
        Policy(spec, **kw)  # refused before any download or graph load


def test_from_spec_takes_only_a_string():
    with pytest.raises(TypeError, match="spec string"):
        from_spec(lambda o: o)  # type: ignore[arg-type]


def test_onnxruntime_is_imported_only_inside_policy_onnx_and_only_at_use_time():
    offenders = []
    for path in PKG.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            hit = any(m.split(".")[0] == "onnxruntime" for m in mods)
            rel = path.relative_to(PKG).as_posix()
            if hit and (not rel.startswith("policy/onnx/") or node.col_offset == 0):
                offenders.append(f"{rel}:{node.lineno}")
    assert offenders == []
    code = "import sys, strands_robots.policy.onnx; print('onnxruntime' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


def _obs(names, q, angvel=(0, 0, 0)):
    n = len(names)
    js = JointState(tuple(names), np.asarray(q, float), np.zeros(n), np.zeros(n))
    return Observation(0.0, js, Pose([0, 0, 0.5], [1, 0, 0, 0], angular_velocity=angvel))


@pytest.mark.network
@needs_onnx_extra
def test_microduck_preset_layout_and_decode():
    pol = from_spec("microduck?vx=0.3")
    assert len(pol.joint_names) == 14 and pol.input_width == 61 and pol.command.shape == (13,)
    vec = pol.observation(_obs(pol.joint_names, pol.default_pose, angvel=(0.1, 0, 0)))
    assert vec.shape == (61,) and np.allclose(vec[:6], [0.1, 0, 0, 0, 0, -1]) and np.allclose(vec[6:20], 0)
    assert vec[48] == pytest.approx(0.3)
    assert np.isfinite(pol.act(_obs(pol.joint_names, pol.default_pose))).all()
    with pytest.raises(ValueError, match="angular_velocity"):
        pol.act(Observation(0.0, JointState(pol.joint_names, np.zeros(14), np.zeros(14), np.zeros(14))))


@pytest.mark.network
@needs_onnx_extra
def test_holosoma_preset_layout_gains_and_gait_clock():
    pol = Policy("holosoma")
    assert len(pol.joint_names) == 29 and pol.input_width == 100
    kp, kd = pol.gains
    assert kp.shape == kd.shape == (29,)
    s, vec = pol.slices, pol.observation(_obs(pol.joint_names, pol.default_pose))
    assert np.allclose(vec[s["dof_pos"]], 0) and np.allclose(vec[s["projected_gravity"]], [0, 0, -1])
    assert np.allclose(vec[s["cos_phase"]], -1)  # zero command: the clock stands at (pi, pi)
    walking = Policy("holosoma?vx=0.5")
    assert np.allclose(walking.observation(_obs(pol.joint_names, pol.default_pose))[s["command_lin_vel"]], [0.5, 0])
    assert not np.allclose(walking.phase, [np.pi, np.pi])


def _heights(robot, policy, seconds):
    zs: list[float] = []
    robot.run(policy, seconds=seconds, stop_when=lambda o: zs.append(float(o.pose.position[2])) or False)
    return zs


@pytest.mark.network
@needs_onnx_extra
def test_gate_holosoma_keeps_the_pelvis_up_for_ten_seconds():
    sim = Simulation("mujoco")
    g1 = Robot("g1", sim=sim, keyframe="stand")
    zs = np.array(_heights(g1, Policy("holosoma"), 10))
    low, steady = int(zs.argmin()), zs[-100:]
    print(
        f"[measured] g1 holosoma 10 s: pelvis min {zs.min():.3f} m at t={low / 50:.2f} s, final {zs[-1]:.3f} m, "
        f"last 2 s {steady.min():.3f}-{steady.max():.3f} m, {len(zs)} steps"
    )
    # The keyframe stands straight-legged; the policy's stance bends the knees (0.67 rad), so the pelvis dips once
    # in the first second and settles. Gate: the spec's floor, the dip is that transient, and the stance holds.
    assert len(zs) == 500 and zs.min() >= 0.74 and low < 50
    assert np.ptp(steady) < 0.01 and abs(zs[-1] - 0.764) < 0.03
    sim.close()


@pytest.mark.network
@needs_onnx_extra
def test_gate_holosoma_walks_through_a_scene_change():
    sim = Simulation("mujoco")
    g1 = Robot("g1", sim=sim, keyframe="stand")
    x0, zs = float(g1.pose.position[0]), []

    def watch(o):
        zs.append(float(o.pose.position[2]))
        if len(zs) == 100:  # t = 2 s: the first sim.render() (a dashboard, a notebook) recompiles the world
            sim.render()
        return False

    g1.run(Policy("holosoma?vx=0.5"), seconds=10, stop_when=watch)
    dx = float(g1.pose.position[0]) - x0
    print(f"[measured] g1 holosoma vx=0.5 10 s, a render at 2 s: pelvis min {min(zs):.3f} m, dx {dx:.2f} m")
    assert len(zs) == 500 and min(zs) > 0.7 and dx > 3.0  # dev: under the MJCF's gains it falls at 7.3 s, 1.3 m
    sim.close()


@pytest.mark.network
@needs_onnx_extra
def test_gate_microduck_walks_a_metre_in_ten_seconds():
    duck = Robot("microduck")
    x0 = float(duck.pose.position[0])
    zs = _heights(duck, from_spec("microduck?vx=0.3"), 10)
    dx = float(duck.pose.position[0]) - x0
    print(f"[measured] microduck vx=0.3 10 s: dx {dx:.2f} m, final height {zs[-1]:.3f} m, min {min(zs):.3f} m")
    assert dx >= 1.0 and min(zs) > 0.10


@pytest.mark.network
@needs_onnx_extra
@pytest.mark.parametrize(
    "skill", sorted(json.loads((PKG / "policy/onnx/presets.json").read_text())["microduck"]["files"])
)
def test_every_microduck_skill_loads_and_acts(skill):
    pol = from_spec(f"microduck?skill={skill}")
    assert np.isfinite(pol.act(_obs(pol.joint_names, pol.default_pose))).all()
