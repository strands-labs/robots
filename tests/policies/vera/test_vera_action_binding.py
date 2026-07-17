"""Contract tests for VeraPolicy action-binding (no torch / no real VERA / no sim)."""

import asyncio

import numpy as np

from strands_robots.policies.vera.provider import VeraPolicy


class FakeClient:
    def __init__(self, meta):
        self._meta = meta

    def get_server_metadata(self):
        return self._meta

    def reset(self, info):
        pass

    def configure(self, p):
        return {}

    def close(self):
        pass

    def infer(self, req):
        return self._infer_out

    _infer_out = None


def make_policy(meta, chunk):
    c = FakeClient(meta)
    c._infer_out = {"action": np.asarray(chunk, dtype=np.float32)}
    p = VeraPolicy(client=c, auto_launch_server=False)
    # simpler: disable runner
    p._runner = None
    return p


def run(p, obs, instr="do it"):
    return asyncio.run(p.get_actions(obs, instr))


# --- 1) joint_position (allegro-like): columns bind to robot joints ---
def test_joint_position_binds_to_joints():
    meta = {
        "action_space": "joint_position",
        "context_frames": 1,
        "gripper_dim_index": -1,
        "gripper_is_raw": False,
        "view_keys": ["image"],
    }
    joints = [f"j{i}" for i in range(6)]
    chunk = [[0.1 * i for i in range(6)]]  # 1 step, 6 dims
    p = make_policy(meta, chunk)
    p.set_robot_state_keys(joints)
    obs = {"image": np.zeros((8, 8, 3), np.uint8), **{j: 0.0 for j in joints}}
    out = run(p, obs)
    assert out, "no actions"
    d = out[0]
    assert set(d.keys()) == set(joints), f"keys not bound to joints: {d.keys()}"
    assert "action_0" not in d, "raw action_i leaked!"
    print("joint_position -> joint keys:", list(d.keys()))


# --- 2) gripper binarization preserved ---
def test_gripper_binarized():
    meta = {
        "action_space": "joint_position",
        "context_frames": 1,
        "gripper_dim_index": 2,
        "gripper_is_raw": True,
        "view_keys": ["image"],
    }
    joints = ["a", "b", "gripper"]
    chunk = [[0.5, 0.6, 0.9]]  # gripper 0.9 > 0.5 -> 1.0
    p = make_policy(meta, chunk)
    p.set_robot_state_keys(joints)
    obs = {"image": np.zeros((8, 8, 3), np.uint8), "a": 0.0, "b": 0.0, "gripper": 0.0}
    out = run(p, obs)
    assert out[0]["gripper"] == 1.0, f"gripper not binarized: {out[0]}"
    print("gripper binarized:", out[0])


# --- 3) eef_delta with NO ik target -> warns + falls back (no crash) ---
def test_eef_delta_without_ik_falls_back():
    meta = {
        "action_space": "eef_delta",
        "context_frames": 1,
        "gripper_dim_index": 6,
        "gripper_is_raw": True,
        "view_keys": ["image"],
    }
    chunk = [[0.01, 0.0, 0.0, 0, 0, 0, 1.0]]  # 3 trans + 3 rot + gripper
    p = make_policy(meta, chunk)
    p.set_robot_state_keys([f"j{i}" for i in range(7)])
    obs = {"image": np.zeros((8, 8, 3), np.uint8), **{f"j{i}": 0.0 for i in range(7)}}
    out = run(p, obs)
    # falls back to raw mapping (warned); should not crash, returns dicts
    assert out, "no actions on fallback"
    print("eef_delta w/o IK falls back (warned), keys:", list(out[0].keys()))


# --- 4) action_mapping override still wins ---
def test_action_mapping_override():
    meta = {
        "action_space": "joint_position",
        "context_frames": 1,
        "gripper_dim_index": -1,
        "gripper_is_raw": False,
        "view_keys": ["image"],
    }
    chunk = [[1.0, 2.0]]
    c = FakeClient(meta)
    c._infer_out = {"action": np.asarray(chunk, np.float32)}
    p = VeraPolicy(client=c, auto_launch_server=False, action_mapping={"action_0": "x", "action_1": "y"})
    p._runner = None
    p.set_robot_state_keys(["wrong1", "wrong2"])  # mapping should win over these
    obs = {"image": np.zeros((8, 8, 3), np.uint8), "wrong1": 0.0, "wrong2": 0.0}
    out = run(p, obs)
    assert set(out[0].keys()) == {"x", "y"}, f"action_mapping ignored: {out[0]}"
    print("action_mapping override wins:", list(out[0].keys()))


# --- 5) server returns a flat 1-D single-timestep action (H omitted) ---
def test_flat_1d_action_from_server_is_promoted():
    # A VERA/RemotePolicy server may return a single-timestep action as a flat
    # 1-D vector (D,) rather than a 2-D chunk (1, D). The provider must promote
    # it to a one-row chunk so the per-step mapping still yields exactly one
    # actuator dict bound to the robot joints - never iterate over scalars.
    meta = {
        "action_space": "joint_position",
        "context_frames": 1,
        "gripper_dim_index": -1,
        "gripper_is_raw": False,
        "view_keys": ["image"],
    }
    joints = [f"j{i}" for i in range(6)]
    flat = [0.1 * i for i in range(6)]  # 1-D, no leading H axis
    p = make_policy(meta, flat)
    p.set_robot_state_keys(joints)
    obs = {"image": np.zeros((8, 8, 3), np.uint8), **{j: 0.0 for j in joints}}
    out = run(p, obs)
    assert len(out) == 1, f"1-D action should map to exactly one step, got {len(out)}"
    d = out[0]
    assert set(d.keys()) == set(joints), f"flat action not bound to joints: {d.keys()}"
    assert d["j5"] == np.float32(0.5), f"column values garbled: {d}"
    print("flat 1-D action promoted + bound:", list(d.keys()))


# --- 6) standalone vector mapping reads the gripper contract from meta ---
def test_vector_to_action_dict_reads_gripper_contract_from_meta():
    # _vector_to_action_dict is the single-vector mapping helper. When called
    # without explicit gripper args it must fall back to the server's
    # gripper_dim_index / gripper_is_raw from meta and binarize a raw gripper
    # float (>0.5 -> close 1.0), so a standalone caller gets the same contract
    # as the chunk path.
    meta = {"gripper_dim_index": 2, "gripper_is_raw": True}
    joints = ["a", "b", "gripper"]
    c = FakeClient(meta)
    p = VeraPolicy(client=c, auto_launch_server=False)
    p._runner = None
    p.set_robot_state_keys(joints)
    out = p._vector_to_action_dict(np.array([0.4, 0.6, 0.9], np.float32), meta)
    assert set(out.keys()) == set(joints), f"not bound to joints: {out}"
    assert out["gripper"] == 1.0, f"raw gripper not binarized from meta: {out}"
    assert out["a"] == np.float32(0.4)
    print("standalone vector honors meta gripper contract:", out)


def test_vector_to_action_dict_meta_gripper_disabled_passthrough():
    # gripper_is_raw False (already-normalized gripper): the value passes
    # through unchanged - no >0.5 binarization applied.
    meta = {"gripper_dim_index": 2, "gripper_is_raw": False}
    joints = ["a", "b", "gripper"]
    c = FakeClient(meta)
    p = VeraPolicy(client=c, auto_launch_server=False)
    p._runner = None
    p.set_robot_state_keys(joints)
    out = p._vector_to_action_dict(np.array([0.4, 0.6, 0.9], np.float32), meta)
    assert out["gripper"] == np.float32(0.9), f"non-raw gripper altered: {out}"
    print("non-raw gripper passthrough:", out)
