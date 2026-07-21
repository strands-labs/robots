"""Unit tests for the VERA policy provider (offline — no server, no GPU, no vera pkg).

Covers: config defaults + env overrides, factory registration/resolution, the
wire client's msgpack roundtrip + error sentinel, the server runner's list-arg
command construction, and the full ``get_actions`` roundtrip against a fake
in-memory client (context-window accumulation, action-chunk queueing, action
dict mapping with gripper binarization).
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.policies.factory import create_policy, list_providers
from strands_robots.policies.vera import VeraConfig, VeraPolicy, VeraServerRunner
from strands_robots.policies.vera import _msgpack_numpy as mnp


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class TestConfig:
    def test_pusht_default_ports(self):
        c = VeraConfig(embodiment="pusht")
        assert c.server_port == 8820
        assert c.vis_port == 8821
        assert c.server_uri == "ws://127.0.0.1:8820"

    def test_mimicgen_default_ports(self):
        c = VeraConfig(embodiment="mimicgen")
        assert c.server_port == 8800
        assert c.vis_port == 8801

    def test_explicit_port_wins(self):
        c = VeraConfig(embodiment="pusht", server_port=9999)
        assert c.server_port == 9999

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("VERA_CKPT_ROOT", "/tmp/ckpts")
        monkeypatch.setenv("VERA_SAMPLE_STEPS", "12")
        monkeypatch.setenv("VERA_MOTION_PLAN_SCALE", "1.5")
        c = VeraConfig(embodiment="pusht")
        assert str(c.ckpt_root) == "/tmp/ckpts"
        assert c.sample_steps == 12
        assert c.motion_plan_scale == 1.5

    def test_server_env_overlay(self):
        c = VeraConfig(embodiment="pusht", ckpt_root="/data/ckpts", tracker_backend="cotracker")
        env = c.server_env()
        assert env["VERA_CKPT_ROOT"] == "/data/ckpts"
        assert env["VERA_TRACKER_BACKEND"] == "cotracker"

    def test_server_env_overlay_forwards_wan_ckpt_and_dynamics_run(self):
        """server_env() must forward every checkpoint/tracker knob the server
        subprocess reads, including the frozen WAN base and the IDM run id."""
        c = VeraConfig(
            embodiment="mimicgen",
            wan_ckpt_root="/data/wan",
            dynamics_run_id="idm-run-42",
        )
        env = c.server_env()
        assert env["VERA_WAN_CKPT_ROOT"] == "/data/wan"
        assert env["VERA_DYNAMICS_RUN_ID"] == "idm-run-42"

    def test_malformed_numeric_env_degrades_to_defaults(self, monkeypatch):
        """A non-numeric env override must be ignored, not crash construction.

        Deploy/CI environments frequently carry typo'd or empty numeric knobs
        (e.g. VERA_SERVER_PORT=""). The int/float env parsers swallow the
        ValueError and return None so config falls back to the per-embodiment
        default instead of raising on import of the policy.
        """
        monkeypatch.setenv("VERA_SERVER_PORT", "not-a-port")
        monkeypatch.setenv("VERA_VIS_PORT", "xyz")
        monkeypatch.setenv("VERA_RENDER_WIDTH", "wide")
        monkeypatch.setenv("VERA_SAMPLE_STEPS", "ten")
        monkeypatch.setenv("VERA_N_ACTION_STEPS", "")
        monkeypatch.setenv("VERA_MOTION_PLAN_SCALE", "fast")

        c = VeraConfig(embodiment="pusht")

        # Non-numeric int knobs fall back to the pusht per-embodiment defaults.
        assert c.server_port == 8820
        assert c.vis_port == 8821
        assert c.render_width == 252  # pusht per-embodiment default
        # Non-numeric optional knobs stay unset (planner yaml decides).
        assert c.sample_steps is None
        assert c.n_action_steps is None
        assert c.motion_plan_scale is None


# --------------------------------------------------------------------------- #
# Factory registration
# --------------------------------------------------------------------------- #
class TestFactory:
    def test_vera_registered(self):
        assert "vera" in list_providers()

    def test_create_policy_resolves(self):
        p = create_policy("vera", embodiment="pusht", auto_launch_server=False)
        assert isinstance(p, VeraPolicy)
        assert p.provider_name == "vera"
        assert p.requires_images is True


# --------------------------------------------------------------------------- #
# Vendored msgpack+numpy codec (wire compatibility)
# --------------------------------------------------------------------------- #
class TestMsgpackNumpy:
    def test_ndarray_roundtrip(self):
        a = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        out = mnp.unpackb(mnp.packb({"x": a}))
        assert np.array_equal(out["x"], a)
        assert out["x"].dtype == np.uint8

    def test_float_action_roundtrip(self):
        a = np.random.randn(10, 8).astype(np.float32)
        out = mnp.unpackb(mnp.packb({"action": a}))
        assert np.allclose(out["action"], a)

    def test_npscalar_encoded_as_tagged_map_and_roundtrips(self):
        # np.generic scalars pack as a tagged __npscalar__ map (not a bare
        # value) so the wire form is self-describing and dtype survives.
        v = np.float32(1.5)
        enc = mnp._encode(v)
        assert enc[b"__npscalar__"] is True
        assert enc[b"dtype"] == v.dtype.str
        out = mnp.unpackb(mnp.packb({"s": v}))
        assert out["s"] == pytest.approx(1.5)

    def test_npscalar_decode_reconstructs_dtype(self):
        # Decode side reconstructs the exact numpy dtype from the tag.
        dec = mnp._decode({b"__npscalar__": True, b"data": 7, b"dtype": "<i8"})
        assert dec == 7
        assert np.dtype(getattr(dec, "dtype", "<i8")) == np.dtype("<i8")

    def test_encode_passthrough_for_native_types(self):
        # Non-numpy, JSON-native values are handed back untouched so msgpack
        # serializes them directly.
        assert mnp._encode(42) == 42
        assert mnp._encode("hi") == "hi"
        assert mnp._encode([1, 2]) == [1, 2]

    def test_non_wire_safe_ndarray_dtype_rejected(self):
        # object/void dtype arrays cannot go over the wire and must raise
        # rather than silently emitting an undecodable blob.
        obj_arr = np.array([{"not": "wire-safe"}], dtype=object)
        with pytest.raises(ValueError, match="cannot serialize ndarray"):
            mnp.packb({"x": obj_arr})
        void_arr = np.zeros(2, dtype=np.dtype([("f", np.int32)]))
        with pytest.raises(ValueError, match="cannot serialize ndarray"):
            mnp.packb({"x": void_arr})


# --------------------------------------------------------------------------- #
# Server runner — list-arg command construction (no shell strings, PR #621)
# --------------------------------------------------------------------------- #
class TestServerRunner:
    def test_command_is_list_args(self):
        cfg = VeraConfig(embodiment="pusht", ckpt_root="/data/ckpts")
        cmd = VeraServerRunner(cfg)._build_command()
        assert isinstance(cmd, list)
        assert all(isinstance(tok, str) for tok in cmd)
        assert "vera.server.start_vera_server" in cmd
        assert cmd[cmd.index("--embodiment") + 1] == "pusht"
        assert cmd[cmd.index("--port") + 1] == "8820"
        assert cmd[cmd.index("--vis-port") + 1] == "8821"

    def test_no_teacache_flag(self):
        cfg = VeraConfig(embodiment="pusht", teacache=False)
        cmd = VeraServerRunner(cfg)._build_command()
        assert "--no-teacache" in cmd
        assert "--teacache-thresh" not in cmd

    def test_optional_flags_included(self):
        cfg = VeraConfig(
            embodiment="mimicgen",
            algo_config="/x/algo.yaml",
            dynamics_run_id="run42",
            text_prompt="stack the blocks",
            sample_steps=10,
        )
        cmd = VeraServerRunner(cfg)._build_command()
        assert cmd[cmd.index("--algo-config") + 1] == "/x/algo.yaml"
        assert cmd[cmd.index("--dynamics-run-id") + 1] == "run42"
        assert cmd[cmd.index("--text") + 1] == "stack the blocks"
        assert cmd[cmd.index("--sample-steps") + 1] == "10"


# --------------------------------------------------------------------------- #
# Fake client + full get_actions roundtrip
# --------------------------------------------------------------------------- #
class FakeClient:
    """In-memory stand-in for VeraWebsocketClient with a scriptable infer."""

    def __init__(self, metadata, action_chunk):
        self._meta = metadata
        self._chunk = np.asarray(action_chunk, dtype=np.float32)
        self.infer_calls = []
        self.reset_calls = []
        self.configure_calls = []

    def get_server_metadata(self):
        return dict(self._meta)

    def infer(self, observation):
        self.infer_calls.append(observation)
        return {"action": self._chunk}

    def reset(self, reset_info=None):
        self.reset_calls.append(reset_info)

    def configure(self, params):
        self.configure_calls.append(params)
        return {"applied": params}

    def close(self):
        pass


def _obs(h=64, w=64):
    return {"image": np.zeros((h, w, 3), dtype=np.uint8), "agent_x": 0.1, "agent_y": 0.2}


class TestGetActionsRoundtrip:
    def _policy(self, meta, chunk):
        client = FakeClient(meta, chunk)
        p = VeraPolicy(
            embodiment="pusht",
            auto_launch_server=False,
            client=client,
            server_runner=None,
        )
        return p, client

    def test_single_view_chunk_queueing(self):
        meta = {
            "view_keys": ["image"],
            "context_frames": 3,
            "needs_prompt": False,
            "action_dim": 2,
            "gripper_dim_index": -1,
        }
        # 4-step chunk, 2D pos actions (no gripper)
        chunk = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        p, client = self._policy(meta, chunk)

        # First call -> one infer, returns first action; queue holds the rest.
        out = p.get_actions_sync(_obs(), "push the T")
        assert len(client.infer_calls) == 1
        assert out == [{"action_0": pytest.approx(0.1), "action_1": pytest.approx(0.2)}]

        # Next 3 calls drain the queue WITHOUT another infer.
        p.get_actions_sync(_obs(), "")
        p.get_actions_sync(_obs(), "")
        last = p.get_actions_sync(_obs(), "")
        assert len(client.infer_calls) == 1  # still only one infer
        assert last == [{"action_0": pytest.approx(0.7), "action_1": pytest.approx(0.8)}]

        # 5th call -> queue empty -> a second infer.
        p.get_actions_sync(_obs(), "")
        assert len(client.infer_calls) == 2

    def test_context_window_respects_context_frames(self):
        meta = {"view_keys": ["image"], "context_frames": 2, "needs_prompt": False, "action_dim": 2}
        p, client = self._policy(meta, [[0.0, 0.0]])
        for _ in range(5):
            p.get_actions_sync(_obs(), "")
        # window capped at context_frames
        assert p._window.maxlen == 2
        assert len(p._window) == 2
        # context_rgb sent has T == 2 on later infers
        ctx = client.infer_calls[-1]["context_rgb"]
        assert ctx.shape[0] == 2 and ctx.shape[-1] == 3

    def test_gripper_binarization(self):
        meta = {
            "view_keys": ["image"],
            "context_frames": 1,
            "needs_prompt": False,
            "action_dim": 3,
            "gripper_dim_index": 2,
            "gripper_is_raw": True,
        }
        # gripper raw 0.9 -> close (1.0); next step raw 0.1 -> open (0.0)
        chunk = [[0.5, -0.5, 0.9], [0.1, 0.2, 0.1]]
        p, client = self._policy(meta, chunk)
        a0 = p.get_actions_sync(_obs(), "")[0]
        a1 = p.get_actions_sync(_obs(), "")[0]
        assert a0["action_2"] == 1.0
        assert a1["action_2"] == 0.0

    def test_action_mapping_renames_columns(self):
        meta = {"view_keys": ["image"], "context_frames": 1, "needs_prompt": False, "action_dim": 2}
        client = FakeClient(meta, [[1.0, 2.0]])
        p = VeraPolicy(
            embodiment="pusht",
            auto_launch_server=False,
            client=client,
            action_mapping={"action_0": "vx", "action_1": "vy"},
        )
        out = p.get_actions_sync(_obs(), "")[0]
        assert out == {"vx": pytest.approx(1.0), "vy": pytest.approx(2.0)}

    def test_prompt_sent_only_when_needed(self):
        meta = {"view_keys": ["image"], "context_frames": 1, "needs_prompt": True, "action_dim": 2}
        p, client = self._policy(meta, [[0.0, 0.0]])
        p.get_actions_sync(_obs(), "stack the red block")
        assert client.infer_calls[0]["prompt"] == "stack the red block"

    def test_reset_forwards_to_client(self):
        meta = {"view_keys": ["image"], "context_frames": 1, "needs_prompt": False, "action_dim": 2}
        p, client = self._policy(meta, [[0.0, 0.0]])
        p.get_actions_sync(_obs(), "")
        p.reset(seed=42)
        assert client.reset_calls and client.reset_calls[-1]["seed"] == 42
        assert len(p._queue) == 0 and len(p._window) == 0

    def test_missing_camera_raises(self):
        meta = {"view_keys": ["image"], "context_frames": 1, "needs_prompt": False, "action_dim": 2}
        p, _ = self._policy(meta, [[0.0, 0.0]])
        with pytest.raises(ValueError, match="requires at least one camera"):
            p.get_actions_sync({"agent_x": 0.1}, "")

    def test_motion_plan_scale_configured_on_start(self):
        meta = {"view_keys": ["image"], "context_frames": 1, "needs_prompt": False, "action_dim": 2}
        client = FakeClient(meta, [[0.0, 0.0]])
        p = VeraPolicy(embodiment="pusht", auto_launch_server=False, client=client, motion_plan_scale=1.25)
        p.get_actions_sync(_obs(), "")
        assert client.configure_calls and client.configure_calls[0]["motion_plan_scale"] == 1.25


# --------------------------------------------------------------------------- #
# Docker server runner — list-arg `docker run` construction (no shell strings)
# --------------------------------------------------------------------------- #
class TestDockerServerRunner:
    def test_server_mode_selects_docker_runner(self):
        from strands_robots.policies.vera.server_runner import (
            DockerServerRunner,
            VeraServerRunner,
            make_server_runner,
        )

        sub = make_server_runner(VeraConfig(embodiment="pusht", server_mode="subprocess"))
        dock = make_server_runner(VeraConfig(embodiment="pusht", server_mode="docker"))
        assert isinstance(sub, VeraServerRunner)
        assert isinstance(dock, DockerServerRunner)

    def test_unknown_mode_raises(self):
        from strands_robots.policies.vera.server_runner import make_server_runner

        with pytest.raises(ValueError, match="unknown server_mode"):
            make_server_runner(VeraConfig(embodiment="pusht", server_mode="k8s"))

    def test_docker_run_command_is_list_args(self, monkeypatch):
        from strands_robots.policies.vera import server_runner as sr

        # Stub `which("docker")` so the command builds without docker installed.
        monkeypatch.setattr(sr, "_port_open", lambda *a, **k: False)
        runner = sr.DockerServerRunner(
            VeraConfig(
                embodiment="pusht",
                server_mode="docker",
                ckpt_root="/data/vera-ckpts",
                docker_image="strands-vera-server:latest",
                docker_gpus="all",
            )
        )
        monkeypatch.setattr(runner, "_docker", lambda: "/usr/bin/docker")
        cmd = runner._build_run_command()
        assert isinstance(cmd, list) and all(isinstance(t, str) for t in cmd)
        assert cmd[:2] == ["/usr/bin/docker", "run"]
        assert "--gpus" in cmd and cmd[cmd.index("--gpus") + 1] == "all"
        assert "--ipc=host" in cmd
        assert "-v" in cmd and "/data/vera-ckpts:/ckpts:ro" in cmd
        assert "-p" in cmd and "8820:8820" in cmd
        assert "8821:8821" in cmd  # vis port published
        assert cmd[-1] == "strands-vera-server:latest"
        # embodiment + port wired via env to the container entrypoint
        assert "VERA_EMBODIMENT=pusht" in cmd
        assert "VERA_PORT=8820" in cmd

    def test_docker_container_name_default(self):
        cfg = VeraConfig(embodiment="mimicgen", server_mode="docker")
        assert cfg.docker_container_name == "vera-server-mimicgen"

    def test_docker_env_overrides(self, monkeypatch):
        monkeypatch.setenv("VERA_SERVER_MODE", "docker")
        monkeypatch.setenv("VERA_DOCKER_IMAGE", "myreg/vera:dev")
        monkeypatch.setenv("VERA_DOCKER_GPUS", "device=0")
        cfg = VeraConfig(embodiment="pusht")
        assert cfg.server_mode == "docker"
        assert cfg.docker_image == "myreg/vera:dev"
        assert cfg.docker_gpus == "device=0"
