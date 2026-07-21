"""Unit coverage for VERA provider frame coercion and action-column binding.

These exercise the embodiment-agnostic plumbing of :mod:`strands_robots.policies.vera.provider`
that does not need a running VERA server or GPU:

* ``_to_uint8_frame`` -- camera-frame dtype/shape coercion (float scaling, batch
  squeeze, integer clamping, shape rejection).
* ``_resize_frame`` -- the no-op fast path and the PIL-less numpy fallback.
* ``VeraPolicy._action_column_names`` (via the public ``get_actions`` chunk
  binding) -- joint-name binding, trailing-gripper extras, and the warn-once
  unbound fallback that emits ``action_i`` keys.
* ``VeraPolicy.close`` -- fail-soft client close plus managed-runner stop.
* ``_resolve_view_keys`` / ``_extract_frame`` -- camera-view precedence
  (explicit ``image_keys`` > server ``view_keys`` > discovered frames) and the
  width-concatenation of the selected views into one context frame.
* ``_ensure_started`` -- start-the-runner-once handshake and the live
  ``motion_plan_scale`` configure call.
* ``VeraPolicy.reset`` -- seed forwarding and best-effort server-error swallow.
* ``VeraPolicy.get_actions`` -- graceful degradation: an empty ``[0, D]`` server
  chunk yields no action (empty list) instead of a crash or a bogus zero action.
* ``_ensure_started`` -- the live ``configure(motion_plan_scale)`` call is
  best-effort: a server that rejects it is swallowed+logged and inference proceeds.
* ``VeraPolicy`` construction -- ``auto_launch_server`` builds a managed runner
  from the policy config (when none is injected) and starts it once on first use.

All assertions are on observable outputs (returned arrays/dicts, the recorded
wire payload sent to the fake client, emitted log records), not internal state.
"""

from __future__ import annotations

import asyncio
import logging
import sys

import numpy as np
import pytest

from strands_robots.policies.vera.config import VeraConfig
from strands_robots.policies.vera.provider import (
    VeraPolicy,
    _resize_frame,
    _to_uint8_frame,
)


class TestToUint8Frame:
    """``_to_uint8_frame`` coerces arbitrary camera values to (H, W, 3) uint8."""

    def test_float_frame_scaled_to_0_255(self):
        frame = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)  # (1, 1, 3)
        out = _to_uint8_frame(frame)
        assert out.dtype == np.uint8
        assert out.shape == (1, 1, 3)
        assert list(out[0, 0]) == [0, 127, 255]

    def test_float_frame_clipped_before_scaling(self):
        # Values outside [0, 1] clamp rather than wrap around.
        frame = np.array([[[-1.0, 2.0, 0.25]]], dtype=np.float64)
        out = _to_uint8_frame(frame)
        assert list(out[0, 0]) == [0, 255, 63]

    def test_batch_dim_is_squeezed(self):
        frame = np.zeros((1, 4, 5, 3), dtype=np.uint8)
        out = _to_uint8_frame(frame)
        assert out.shape == (4, 5, 3)

    def test_integer_frame_clamped_to_uint8(self):
        frame = np.array([[[300, -5, 128]]], dtype=np.int16)
        out = _to_uint8_frame(frame)
        assert out.dtype == np.uint8
        assert list(out[0, 0]) == [255, 0, 128]

    def test_uint8_frame_returned_contiguous(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)[::1]
        out = _to_uint8_frame(frame)
        assert out.dtype == np.uint8
        assert out.flags["C_CONTIGUOUS"]

    def test_bad_shape_raises_valueerror(self):
        with pytest.raises(ValueError, match=r"must be \(H, W, 3\)"):
            _to_uint8_frame(np.zeros((4, 5), dtype=np.uint8))


class TestResizeFrame:
    """``_resize_frame`` squares each view to the planner's per-view width."""

    def test_already_square_is_identity(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        out = _resize_frame(frame, 8)
        assert out is frame

    def test_numpy_fallback_when_pil_unavailable(self, monkeypatch):
        # Force `from PIL import Image` to raise so the numpy nearest-neighbour
        # branch runs; it must still produce a square (width, width, 3) frame.
        monkeypatch.setitem(sys.modules, "PIL", None)
        frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        out = _resize_frame(frame, 5)
        assert out.shape == (5, 5, 3)
        assert out.dtype == np.uint8
        assert out.flags["C_CONTIGUOUS"]


class _FakeClient:
    """Scriptable VeraWebsocketClient stand-in (no socket)."""

    def __init__(self, metadata, action_chunk, *, raise_on_close=False):
        self._meta = metadata
        self._chunk = np.asarray(action_chunk, dtype=np.float32)
        self._raise_on_close = raise_on_close
        self.closed = False
        # Recorded wire traffic (observable outputs for assertions).
        self.infer_requests: list[dict] = []
        self.reset_calls: list[dict | None] = []
        self.configure_calls: list[dict] = []

    def get_server_metadata(self):
        return dict(self._meta)

    def infer(self, observation):
        self.infer_requests.append(observation)
        return {"action": self._chunk}

    def reset(self, reset_info=None):
        self.reset_calls.append(reset_info)

    def configure(self, params):
        self.configure_calls.append(params)
        return {"applied": params}

    def close(self):
        if self._raise_on_close:
            raise RuntimeError("socket already gone")
        self.closed = True


class _FakeRunner:
    def __init__(self):
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1


def _img_obs(h=32, w=32):
    return {"image": np.zeros((h, w, 3), dtype=np.uint8)}


def _policy(meta, chunk, *, client=None, runner=None):
    client = client or _FakeClient(meta, chunk)
    return (
        VeraPolicy(
            embodiment="pusht",
            auto_launch_server=False,
            client=client,
            server_runner=runner,
        ),
        client,
    )


class TestActionColumnBinding:
    """The chunk->actuator-name mapping selected by ``_action_column_names``."""

    def test_unbound_emits_action_i_keys_and_warns_once(self, caplog):
        meta = {"action_space": "pos", "context_frames": 1, "gripper_dim_index": -1}
        policy, _ = _policy(meta, [[0.1, 0.2, 0.3]])  # no robot_state_keys, no mapping
        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.vera.provider"):
            first = asyncio.run(policy.get_actions(_img_obs(), ""))
            # Drain the queued chunk, then force a second infer + bind.
            asyncio.run(policy.get_actions(_img_obs(), ""))
        assert list(first[0].keys()) == ["action_0", "action_1", "action_2"]
        unbound_warnings = [r for r in caplog.records if "UNRESOLVED" in r.getMessage()]
        assert len(unbound_warnings) == 1  # warn-once latch, not per-call spam

    def test_joints_bind_directly_and_truncate_to_action_dim(self):
        meta = {"action_space": "joint_position", "context_frames": 1, "gripper_dim_index": -1}
        policy, _ = _policy(meta, [[0.1, 0.2, 0.3]])
        policy.set_robot_state_keys(["shoulder", "elbow", "wrist", "extra"])
        out = asyncio.run(policy.get_actions(_img_obs(), ""))
        assert list(out[0].keys()) == ["shoulder", "elbow", "wrist"]

    def test_trailing_gripper_column_kept_as_action_extra(self):
        meta = {"action_space": "joint_position", "context_frames": 1, "gripper_dim_index": -1}
        policy, _ = _policy(meta, [[0.1, 0.2, 0.3]])
        policy.set_robot_state_keys(["shoulder", "elbow"])  # fewer joints than columns
        out = asyncio.run(policy.get_actions(_img_obs(), ""))
        assert list(out[0].keys()) == ["shoulder", "elbow", "action_2"]


class TestClose:
    """``close`` is fail-soft on the client and always stops a managed runner."""

    def test_close_swallows_client_error_and_stops_runner(self):
        runner = _FakeRunner()
        client = _FakeClient({"action_space": "pos"}, [[0.0]], raise_on_close=True)
        policy, _ = _policy({}, [[0.0]], client=client, runner=runner)
        policy.close()  # must not raise despite client.close() error
        assert runner.stop_calls == 1

    def test_close_without_runner_is_noop_safe(self):
        client = _FakeClient({}, [[0.0]])
        policy, _ = _policy({}, [[0.0]], client=client, runner=None)
        policy.close()
        assert client.closed is True


def _cfg(**kw):
    """A VeraConfig with a tiny per-view render width and no auto-launch.

    render_width=8 keeps the width-concatenated context tensor small; the
    provider squares each view to this width before stacking, so assertions on
    ``context_rgb`` shape stay cheap and exact.
    """
    kw.setdefault("embodiment", "mimicgen")
    kw.setdefault("render_width", 8)
    kw.setdefault("auto_launch_server", False)
    return VeraConfig(**kw)


def _cam(h=4, w=4):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestViewResolution:
    """``_resolve_view_keys`` picks camera keys by precedence: explicit >
    server-advertised ``view_keys`` > frames discovered in the observation, and
    ``_extract_frame`` width-concatenates the selected views into one frame."""

    def test_explicit_image_keys_win_over_extra_cameras(self):
        meta = {"action_space": "pos", "context_frames": 1}
        client = _FakeClient(meta, [[0.0, 0.0]])
        policy = VeraPolicy(image_keys=["front"], client=client, config=_cfg())
        obs = {"front": _cam(), "wrist": _cam(), "joint0": 0.1}
        asyncio.run(policy.get_actions(obs, ""))
        req = client.infer_requests[-1]
        # Only the explicitly requested view is sent; the extra camera is dropped.
        assert req["view_keys"] == ["front"]
        assert req["view_widths"] == [8]
        assert req["context_rgb"].shape[1:] == (8, 8, 3)  # single squared view

    def test_server_view_keys_select_and_order_matching_cameras(self):
        # Server advertises its own view order; the provider honours it even when
        # the observation dict enumerates the cameras in a different order.
        meta = {"action_space": "pos", "context_frames": 1, "view_keys": ["cam_right", "cam_left"]}
        client = _FakeClient(meta, [[0.0]])
        policy = VeraPolicy(client=client, config=_cfg())
        obs = {"cam_left": _cam(), "cam_right": _cam(), "gripper": 0.0}
        asyncio.run(policy.get_actions(obs, ""))
        req = client.infer_requests[-1]
        assert req["view_keys"] == ["cam_right", "cam_left"]  # server order, not obs order
        assert req["view_widths"] == [8, 8]
        assert req["context_rgb"].shape[1:] == (8, 16, 3)  # two views width-concatenated

    def test_discovered_frames_used_when_server_advertises_no_views(self):
        # No explicit keys and no server view_keys: fall back to every (H, W, 3)
        # frame found in the observation, in dict order.
        meta = {"action_space": "pos", "context_frames": 1}
        client = _FakeClient(meta, [[0.0]])
        policy = VeraPolicy(client=client, config=_cfg())
        obs = {"top": _cam(), "side": _cam(), "elbow": 0.2}
        asyncio.run(policy.get_actions(obs, ""))
        req = client.infer_requests[-1]
        assert req["view_keys"] == ["top", "side"]
        assert req["context_rgb"].shape[1:] == (8, 16, 3)

    def test_no_camera_frame_raises_actionable_valueerror(self):
        meta = {"action_space": "pos", "context_frames": 1}
        client = _FakeClient(meta, [[0.0]])
        policy = VeraPolicy(client=client, config=_cfg())
        with pytest.raises(ValueError, match="at least one camera frame"):
            asyncio.run(policy.get_actions({"joint0": 0.1}, ""))


class TestServerHandshake:
    """``_ensure_started`` starts the managed runner exactly once and applies
    live-tunable knobs (``motion_plan_scale``) via a single ``configure`` call."""

    def test_motion_plan_scale_applied_once_and_runner_started_once(self):
        meta = {"action_space": "pos", "context_frames": 1}
        client = _FakeClient(meta, [[0.0]])
        runner = _FakeRunner()
        policy = VeraPolicy(client=client, server_runner=runner, config=_cfg(motion_plan_scale=1.5))
        asyncio.run(policy.get_actions(_img_obs(), ""))
        asyncio.run(policy.get_actions(_img_obs(), ""))  # second call must not re-handshake
        assert runner.start_calls == 1
        assert client.configure_calls == [{"motion_plan_scale": 1.5}]

    def test_no_configure_call_when_motion_plan_scale_unset(self):
        meta = {"action_space": "pos", "context_frames": 1}
        client = _FakeClient(meta, [[0.0]])
        policy = VeraPolicy(client=client, config=_cfg())  # motion_plan_scale defaults to None
        asyncio.run(policy.get_actions(_img_obs(), ""))
        assert client.configure_calls == []


class TestReset:
    """``reset`` forwards the seed to the server and is fail-soft on errors."""

    def test_reset_forwards_seed_and_reason(self):
        client = _FakeClient({"action_space": "pos"}, [[0.0]])
        policy = VeraPolicy(client=client, config=_cfg())
        policy.reset(seed=7)
        assert client.reset_calls[-1]["seed"] == 7
        assert client.reset_calls[-1]["reason"] == "eval_episode"

    def test_reset_omits_seed_when_none(self):
        client = _FakeClient({"action_space": "pos"}, [[0.0]])
        policy = VeraPolicy(client=client, config=_cfg())
        policy.reset()
        assert "seed" not in client.reset_calls[-1]

    def test_reset_is_best_effort_when_server_errors(self, caplog):
        class _BoomClient(_FakeClient):
            def reset(self, reset_info=None):
                raise RuntimeError("server down")

        client = _BoomClient({"action_space": "pos"}, [[0.0]])
        policy = VeraPolicy(client=client, config=_cfg())
        with caplog.at_level(logging.INFO, logger="strands_robots.policies.vera.provider"):
            policy.reset()  # must not raise despite the server error
        assert any("best-effort failed" in r.getMessage() for r in caplog.records)


class TestEmptyChunkDegradesGracefully:
    """An empty inference result yields no action, never a crash.

    When the server returns a zero-row ``[0, D]`` chunk (nothing to execute this
    tick), ``get_actions`` must return an empty list so the rollout loop simply
    skips the step, rather than raising or emitting a bogus zero action.
    """

    def test_empty_server_chunk_returns_no_actions(self):
        client = _FakeClient(
            {"action_space": "pos", "context_frames": 1},
            np.zeros((0, 3), dtype=np.float32),
        )
        policy = VeraPolicy(client=client, config=_cfg())
        out = asyncio.run(policy.get_actions(_img_obs(), ""))
        assert out == []


class TestLiveConfigureIsBestEffort:
    """``_ensure_started`` applies ``motion_plan_scale`` opportunistically.

    A server that rejects the live ``configure`` call must not abort the
    handshake: the error is swallowed and logged, and inference still proceeds.
    """

    def test_configure_error_is_swallowed_and_logged(self, caplog):
        class _ConfigureBoomClient(_FakeClient):
            def configure(self, params):
                raise RuntimeError("configure rejected by server")

        client = _ConfigureBoomClient({"action_space": "pos", "context_frames": 1}, [[0.0]])
        policy = VeraPolicy(client=client, config=_cfg(motion_plan_scale=2.0))
        with caplog.at_level(logging.INFO, logger="strands_robots.policies.vera.provider"):
            out = asyncio.run(policy.get_actions(_img_obs(), ""))  # must not raise
        assert out  # inference still produced an action despite the configure failure
        assert any("configure(motion_plan_scale) skipped" in r.getMessage() for r in caplog.records)


class TestAutoLaunchServerRunner:
    """When ``auto_launch_server`` is set and no runner is injected, the policy
    builds a managed runner from its config and starts it on first use."""

    def test_managed_runner_created_from_config_and_started_once(self, monkeypatch):
        runner = _FakeRunner()
        created_with: list = []

        def _fake_make(cfg):
            created_with.append(cfg)
            return runner

        monkeypatch.setattr(
            "strands_robots.policies.vera.provider.make_server_runner",
            _fake_make,
        )
        client = _FakeClient({"action_space": "pos", "context_frames": 1}, [[0.0]])
        cfg = _cfg(auto_launch_server=True)
        policy = VeraPolicy(client=client, config=cfg)
        assert created_with == [cfg]  # runner built from this policy's config
        asyncio.run(policy.get_actions(_img_obs(), ""))
        assert runner.start_calls == 1
