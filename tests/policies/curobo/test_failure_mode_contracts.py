"""Failure-mode and fallback contracts for :class:`CuroboPolicy`.

The smoke tests in ``test_policy.py`` cover the happy planning path against a
stubbed ``MotionGen``; ``test_native_curobo_api.py`` covers the native-tensor
construction path with a fake ``curobo`` package. This module pins the
*degradation* and *fallback* behaviours the policy promises when things go
wrong or when it runs against a minimal / non-cuRobo planner:

* ``_safe_warmup`` is best-effort - a missing hook is a no-op and a raising
  hook is swallowed with a warning (the first plan just pays the JIT cost).
* ``_next_chunk`` on an empty cache yields ``[]`` rather than indexing off the
  end of an empty trajectory.
* A planner exception is re-raised as a ``RuntimeError`` carrying the goal
  context, not an opaque cuRobo trace.
* An implausibly long interpolated plan is refused up-front (the
  ``_MAX_TRAJECTORY_WAYPOINTS`` guard) instead of being cached.
* ``_build_start_state(None)`` defers to the planner's own retract config.
* ``_resolve_tool_frames`` raises a clear ``RuntimeError`` when the planner has
  no ``kinematics`` or an empty ``tool_frames`` list.
* ``_planner_tensor_kwargs`` falls back to a sane device/dtype when the
  planner's ``device_cfg`` leaves them unset.
* ``_extract_trajectory`` accepts a plain list-of-lists ``position`` (no
  ``.cpu()``), collapsing to a ``[T, ndof]`` trajectory.

All contracts run against lightweight stubs - no GPU, no cuRobo install.
"""

from __future__ import annotations

import asyncio
import logging
import types

import pytest
import torch

import strands_robots.policies.curobo.policy as curobo_policy
from strands_robots.policies.curobo import CuroboPolicy

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubResult:
    """Stand-in for a cuRobo plan result exposing the ``trajectory`` fallback."""

    def __init__(self, ndof: int = 6, horizon: int = 10) -> None:
        self.success = True
        self.status = "ok"
        self.trajectory = [[(t + 1) / 100.0 * (i + 1) for i in range(ndof)] for t in range(horizon)]


class _StubPlanner:
    """Minimal joint-space planner: only exposes ``plan_single_js``."""

    def __init__(self, ndof: int = 6, horizon: int = 10) -> None:
        self.ndof = ndof
        self.horizon = horizon
        self.kinematics = types.SimpleNamespace(tool_frames=["tool0"])

    def plan_single_js(self, start_state: object, goal: object) -> _StubResult:
        return _StubResult(self.ndof, self.horizon)


def _reach_joints(policy: CuroboPolicy, joints: dict[str, float]) -> list[dict[str, float]]:
    return asyncio.run(policy.get_actions({"observation.state": [0.0] * 6}, "", target_joints=joints))


# ---------------------------------------------------------------------------
# Best-effort warmup
# ---------------------------------------------------------------------------


class TestSafeWarmup:
    def test_missing_warmup_hook_is_noop(self) -> None:
        """A planner without a ``warmup`` method must not error - warmup is optional."""
        p = CuroboPolicy(motion_gen=types.SimpleNamespace())
        p._safe_warmup()  # must not raise

    def test_raising_warmup_is_swallowed_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A failing ``warmup`` is best-effort: swallowed, logged, and non-fatal."""

        class _Boom:
            def warmup(self) -> None:
                raise RuntimeError("jit compile crashed")

        p = CuroboPolicy(motion_gen=_Boom())
        with caplog.at_level(logging.WARNING, logger=curobo_policy.__name__):
            p._safe_warmup()  # must not propagate
        assert any("warmup" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Cache streaming
# ---------------------------------------------------------------------------


class TestNextChunk:
    def test_empty_cache_yields_empty_list(self) -> None:
        """With nothing cached, ``_next_chunk`` returns ``[]`` (no index error)."""
        p = CuroboPolicy(motion_gen=_StubPlanner())
        assert p._next_chunk() == []


# ---------------------------------------------------------------------------
# Planning error surfacing
# ---------------------------------------------------------------------------


class TestPlanningErrors:
    def test_planner_exception_reraised_with_goal_context(self) -> None:
        """A raw planner exception surfaces as a RuntimeError naming the goal."""

        class _Raiser(_StubPlanner):
            def plan_single_js(self, start_state: object, goal: object) -> _StubResult:
                raise ValueError("solver diverged")

        p = CuroboPolicy(motion_gen=_Raiser())
        with pytest.raises(RuntimeError, match="planning failed.*target_joints"):
            _reach_joints(p, {"joint_0": 0.5})

    def test_overlong_trajectory_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A trajectory beyond the waypoint guard is rejected, not cached."""
        monkeypatch.setattr(curobo_policy, "_MAX_TRAJECTORY_WAYPOINTS", 3)
        p = CuroboPolicy(motion_gen=_StubPlanner(horizon=10))
        with pytest.raises(RuntimeError, match="exceeds 3"):
            _reach_joints(p, {"joint_0": 0.5})
        # The bad plan must not have been cached.
        assert p._cached_trajectory == []

    def test_joint_goal_falls_back_to_plan_single(self) -> None:
        """A legacy planner exposing only ``plan_single`` still handles joint goals."""

        class _LegacyOnlyPlanSingle:
            def __init__(self) -> None:
                self.kinematics = types.SimpleNamespace(tool_frames=["tool0"])
                self.plan_single_calls: list[object] = []

            def plan_single(self, start_state: object, goal: object) -> _StubResult:
                self.plan_single_calls.append(goal)
                return _StubResult(ndof=6, horizon=4)

        planner = _LegacyOnlyPlanSingle()
        p = CuroboPolicy(motion_gen=planner, action_horizon=4)
        actions = _reach_joints(p, {"joint_0": 0.5})
        assert len(actions) == 4
        assert planner.plan_single_calls  # the joint-space fallback was taken


# ---------------------------------------------------------------------------
# Start-state / tool-frame / tensor fallbacks
# ---------------------------------------------------------------------------


class TestFallbacks:
    def test_build_start_state_none_defers_to_planner(self) -> None:
        """No start joint state -> ``None`` so the planner uses its retract config."""
        p = CuroboPolicy(motion_gen=_StubPlanner())
        assert p._build_start_state(None) is None

    def test_resolve_tool_frames_requires_kinematics(self) -> None:
        p = CuroboPolicy(motion_gen=types.SimpleNamespace())
        with pytest.raises(RuntimeError, match="no .kinematics"):
            p._resolve_tool_frames()

    def test_resolve_tool_frames_requires_nonempty_frames(self) -> None:
        planner = types.SimpleNamespace(kinematics=types.SimpleNamespace(tool_frames=[]))
        p = CuroboPolicy(motion_gen=planner)
        with pytest.raises(RuntimeError, match="tool_frames is empty"):
            p._resolve_tool_frames()

    def test_tensor_kwargs_fall_back_when_device_cfg_unset(self) -> None:
        """An unset device/dtype on the planner's device_cfg falls back sanely."""

        class _DeviceCfg:
            def __init__(self, device: object = None, dtype: object = None) -> None:
                self.device = device
                self.dtype = dtype

        planner = types.SimpleNamespace(device_cfg=_DeviceCfg())
        p = CuroboPolicy(motion_gen=planner)
        device, dtype = p._planner_tensor_kwargs(_DeviceCfg, torch)
        assert device is not None
        assert dtype is torch.float32

    def test_extract_trajectory_accepts_plain_list_position(self) -> None:
        """A list-of-lists ``position`` (no ``.cpu()``) collapses to [T, ndof]."""
        plan = types.SimpleNamespace(position=[[0.1, 0.2], [0.3, 0.4]])
        result = types.SimpleNamespace(get_interpolated_plan=lambda: plan)
        assert CuroboPolicy._extract_trajectory(result) == [[0.1, 0.2], [0.3, 0.4]]
