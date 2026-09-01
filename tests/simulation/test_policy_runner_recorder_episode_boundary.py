"""Regression tests for issue #708 — silent episode collapse.

Background
==========
``PolicyRunner.evaluate`` and ``PolicyRunner._evaluate_with_spec`` run a
``for ep in range(n_episodes):`` loop. Before this fix, neither loop called
``recorder.save_episode()`` between iterations. The recorder kept one open
LeRobot episode buffer, ``add_frame`` (invoked from the sim step hook)
appended to it, and ``stop_recording`` flushed the single buffer at the end.

Net effect: a 20-episode × 60-step rollout produced one parquet episode of
1140 frames, but the eval status text reported ``Episodes: 20`` because the
runner's own bookkeeping counted iterations. A run was marked status=OK and
shipped to HuggingFace with ``total_episodes=1`` — silent data-integrity
loss across 47/47 historical molmoact-e2e runs.

The fix wires ``self._finalize_recorder_episode()`` at the end of each
``for ep ...`` body. The recorder is opaque-injected via
``world._backend_state["dataset_recorder"]`` so we don't need a real
``LeRobotDataset`` — a stub that counts ``save_episode`` calls is enough to
pin the contract.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.policy_runner import PolicyRunner


class _StubRecorder:
    """Minimal stand-in for ``DatasetRecorder``.

    Mimics the surface ``_finalize_recorder_episode`` touches:
    * ``episode_frame_count``  — incremented by the test "step hook" so
      :meth:`PolicyRunner._finalize_recorder_episode` doesn't skip the call
      due to an empty buffer.
    * ``save_episode``         — bumps ``episode_count``, resets
      ``episode_frame_count``, and records the call for assertion.

    We deliberately do NOT subclass the real recorder — keeping the contract
    explicit and minimal documents what the fix actually depends on.
    """

    def __init__(self) -> None:
        self.episode_count = 0
        self.episode_frame_count = 0
        self.frame_count = 0
        self.save_calls: list[dict[str, int]] = []

    def add_frame(self, *_a, **_kw) -> None:  # not used here but mimics surface
        self.episode_frame_count += 1
        self.frame_count += 1

    def save_episode(self) -> dict[str, Any]:
        ep_frames = self.episode_frame_count
        self.episode_count += 1
        self.episode_frame_count = 0
        self.save_calls.append({"ep": self.episode_count, "frames": ep_frames})
        return {
            "status": "success",
            "episode": self.episode_count,
            "episode_frames": ep_frames,
            "total_frames": self.frame_count,
        }


@pytest.fixture
def sim_with_robot():
    s = Simulation(tool_name="pr708", mesh=False)
    s.create_world()
    s.add_robot(name="alice", data_config="so100")
    yield s
    s.cleanup()


def _attach_recorder(sim: Simulation, frames_per_episode: int = 5) -> _StubRecorder:
    """Attach a stub recorder + simulate per-step ``add_frame`` calls.

    The real recorder gets fed by ``Simulation`` send_action callbacks. For
    this test we don't need to wire that — we manually pump ``add_frame``
    inside an ``on_frame`` hook so :meth:`_finalize_recorder_episode` sees
    a non-empty buffer when it fires.
    """
    rec = _StubRecorder()
    assert sim._world is not None
    sim._world._backend_state["dataset_recorder"] = rec
    return rec


class TestRecorderEpisodeBoundary:
    """#708: evaluate() must call save_episode between rollouts."""

    def test_evaluate_calls_save_episode_per_iteration(self, sim_with_robot):
        rec = _attach_recorder(sim_with_robot)

        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        runner = PolicyRunner(sim_with_robot)

        # Pump add_frame per policy step so the recorder buffer is non-empty
        # when finalize is called. This test wraps ``policy.get_actions``
        # (rather than ``on_frame``) to keep the recorder feed independent of
        # the hook under test elsewhere.
        def fake_step_recorder() -> None:
            rec.add_frame()

        # Wrap policy.get_actions so each policy step bumps the recorder.
        orig_get = policy.get_actions

        def wrapped(*a, **kw):
            fake_step_recorder()
            return orig_get(*a, **kw)

        policy.get_actions = wrapped  # type: ignore[method-assign]

        result = runner.evaluate(
            "alice",
            policy,
            n_episodes=3,
            max_steps=4,
            control_frequency=50,
        )

        assert result["status"] == "success"

        # The fix: save_episode must fire exactly once per evaluate iteration.
        assert rec.episode_count == 3, (
            f"Expected 3 episode boundaries, got {rec.episode_count}. Save calls: {rec.save_calls}"
        )
        assert len(rec.save_calls) == 3
        # Each boundary must have flushed some frames (not the empty-buffer
        # short-circuit path).
        for call in rec.save_calls:
            assert call["frames"] >= 1, f"empty-buffer save: {call}"

    def test_finalize_helper_no_op_without_recorder(self, sim_with_robot):
        """No recorder attached → helper must silently no-op (eval-only path)."""
        # No recorder injected.
        assert sim_with_robot._world._backend_state.get("dataset_recorder") is None

        runner = PolicyRunner(sim_with_robot)
        # Should not raise.
        runner._finalize_recorder_episode()

    def test_finalize_helper_skips_empty_buffer(self, sim_with_robot):
        """Empty recorder buffer → helper must NOT call save_episode.

        LeRobot raises on save_episode with zero frames. The helper guards
        with ``episode_frame_count <= 0`` so degenerate rollouts that wrote
        nothing don't trip an error.
        """
        rec = _attach_recorder(sim_with_robot)
        # episode_frame_count starts at 0 — never wrote a frame this ep.
        assert rec.episode_frame_count == 0

        runner = PolicyRunner(sim_with_robot)
        runner._finalize_recorder_episode()

        assert rec.episode_count == 0, "save_episode fired on empty buffer"
        assert rec.save_calls == []

    def test_finalize_helper_calls_save_when_buffer_nonempty(self, sim_with_robot):
        rec = _attach_recorder(sim_with_robot)
        rec.add_frame()
        rec.add_frame()
        rec.add_frame()
        assert rec.episode_frame_count == 3

        runner = PolicyRunner(sim_with_robot)
        runner._finalize_recorder_episode()

        assert rec.episode_count == 1
        assert rec.save_calls == [{"ep": 1, "frames": 3}]
        # After save, buffer is reset.
        assert rec.episode_frame_count == 0

    def test_finalize_helper_reports_a_save_exception_instead_of_raising(self, sim_with_robot):
        """A failed flush reaches the caller as a reason, not as an exception.

        The helper still must not raise - the eval loop reports outcomes in an
        envelope - but it no longer swallows the failure either. Continuing was
        not the milder option: the recorder closes itself on a failed flush and
        every later ``add_frame`` then writes nothing and counts no drop, so the
        evaluation would report a ``success_rate`` over episodes whose data does
        not exist. See
        ``tests/simulation/test_recording_episode_loss_is_not_tolerated.py``,
        which pins what the loops do with this reason.
        """
        rec = _attach_recorder(sim_with_robot)
        rec.episode_frame_count = 5  # non-empty buffer

        def boom() -> dict:
            raise RuntimeError("simulated lerobot crash")

        rec.save_episode = boom  # type: ignore[method-assign]

        runner = PolicyRunner(sim_with_robot)
        reason = runner._finalize_recorder_episode()  # must not raise

        assert reason is not None
        assert "simulated lerobot crash" in reason
