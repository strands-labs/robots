"""Unit tests for the ``run_policy`` Strands ``@tool`` wrapper.

Pins the contract that closes the
`#708 <https://github.com/strands-labs/robots/issues/708>`_ fabrication
vector:

* The tool MUST iterate exactly ``n_episodes`` times - it owns the loop
  and never delegates iteration counting to an LLM narrating "20/20".
* When ``dataset_root`` is supplied, the tool MUST drive a complete
  ``start_recording`` -> per-episode boundary -> ``stop_recording`` cycle.
* The returned payload MUST carry parquet-truth (``total_episodes`` /
  ``total_frames`` read from ``meta/info.json`` on disk) so a verifier
  can catch silent collapse before status=OK propagates.
* Invalid arguments (``simulation=None``, ``n_episodes<1``,
  non-int ``n_steps``) MUST fail loudly with a structured error.

Mirrors the style of ``tests/simulation/test_policy_runner_behaviour.py``
but stays at the unit-test layer - no MuJoCo, no LeRobot - so this file
is fast and self-contained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from strands_robots.tools.run_policy import run_policy as run_policy_tool


# The ``@tool`` decorator wraps the function so tests need to reach the
# original callable. ``strands.tools.decorator.tool`` exposes it as
# ``.original_function`` (and falls back to ``__wrapped__`` on older
# versions).
def _unwrap(t: Any) -> Any:
    # strands.tools.decorator.tool wraps the function in a
    # DecoratedFunctionTool instance whose underlying callable is on
    # _tool_func. Older snapshots used original_function /
    # __wrapped__ - keep both paths so this test file is resilient to
    # SDK churn.
    for attr in ("_tool_func", "original_function", "__wrapped__", "func"):
        target = getattr(t, attr, None)
        if callable(target):
            return target
    return t


run_policy = _unwrap(run_policy_tool)


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


def _ok_rollout(text: str = "ok") -> dict[str, Any]:
    return {"status": "success", "content": [{"text": text}]}


class _FakeSim:
    """Minimal Simulation stand-in.

    Records every method call so tests can assert call counts and arg
    pass-through. Mirrors the public surface ``run_policy`` actually
    touches: ``run_policy``, ``start_recording``, ``stop_recording``.
    """

    def __init__(self) -> None:
        self.run_policy_calls: list[dict[str, Any]] = []
        self.start_recording_calls: list[dict[str, Any]] = []
        self.stop_recording_calls: list[dict[str, Any]] = []

    def run_policy(self, **kwargs: Any) -> dict[str, Any]:
        self.run_policy_calls.append(kwargs)
        return _ok_rollout(f"rollout {len(self.run_policy_calls)}")

    def start_recording(self, **kwargs: Any) -> dict[str, Any]:
        self.start_recording_calls.append(kwargs)
        return _ok_rollout("recording started")

    def stop_recording(self, **kwargs: Any) -> dict[str, Any]:
        self.stop_recording_calls.append(kwargs)
        return _ok_rollout("recording stopped")


class _CameraslessSim:
    """Simulation stand-in whose ``start_recording`` has NO ``cameras``
    parameter and NO ``**kwargs`` catch-all - mirroring the Newton backend
    (``strands_robots/simulation/newton/recording.py``). Forwarding an
    unexpected ``cameras=`` kwarg here raises ``TypeError``, which is exactly
    the crash this fake exists to pin against.
    """

    def __init__(self) -> None:
        self.run_policy_calls: list[dict[str, Any]] = []
        self.start_recording_calls: int = 0
        self.stop_recording_calls: int = 0

    def run_policy(self, **kwargs: Any) -> dict[str, Any]:
        self.run_policy_calls.append(kwargs)
        return _ok_rollout(f"rollout {len(self.run_policy_calls)}")

    def start_recording(
        self,
        repo_id: str = "local/sim_recording",
        task: str = "",
        fps: int = 30,
        root: str | None = None,
        push_to_hub: bool = False,
        vcodec: str = "libsvtav1",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        self.start_recording_calls += 1
        return _ok_rollout("recording started")

    def stop_recording(self, **kwargs: Any) -> dict[str, Any]:
        self.stop_recording_calls += 1
        return _ok_rollout("recording stopped")


class _NoRecordingSim:
    """Simulation stand-in WITHOUT recording surface - exercises the
    ``hasattr`` guard in run_policy."""

    def __init__(self) -> None:
        self.run_policy_calls: list[dict[str, Any]] = []

    def run_policy(self, **kwargs: Any) -> dict[str, Any]:
        self.run_policy_calls.append(kwargs)
        return _ok_rollout("rollout")


def _write_info_json(root: Path, *, total_episodes: int, total_frames: int) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(
        json.dumps({"total_episodes": total_episodes, "total_frames": total_frames, "fps": 30})
    )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


class TestValidation:
    def test_rejects_none_simulation(self) -> None:
        result = run_policy(None, n_episodes=1, n_steps=10)
        assert result["status"] == "error"
        assert "simulation" in result["content"][0]["text"].lower()

    def test_rejects_simulation_without_run_policy(self) -> None:
        bad = object()  # no .run_policy
        result = run_policy(bad, n_episodes=1, n_steps=10)
        assert result["status"] == "error"
        assert "run_policy" in result["content"][0]["text"]

    def test_rejects_zero_episodes(self) -> None:
        sim = _FakeSim()
        result = run_policy(sim, n_episodes=0, n_steps=10)
        assert result["status"] == "error"
        assert "positive int" in result["content"][0]["text"]
        assert sim.run_policy_calls == []

    def test_rejects_negative_episodes(self) -> None:
        sim = _FakeSim()
        result = run_policy(sim, n_episodes=-3, n_steps=10)
        assert result["status"] == "error"
        assert sim.run_policy_calls == []

    def test_rejects_bool_episodes(self) -> None:
        # bool is an int subclass in Python; the loop guard MUST reject it
        # so True/False can't sneak through as "1"/"0".
        sim = _FakeSim()
        result = run_policy(sim, n_episodes=True, n_steps=10)  # type: ignore[arg-type]
        assert result["status"] == "error"

    def test_rejects_bad_n_steps(self) -> None:
        sim = _FakeSim()
        result = run_policy(sim, n_episodes=1, n_steps=0)
        assert result["status"] == "error"
        assert sim.run_policy_calls == []


# --------------------------------------------------------------------------
# Loop semantics (the core #708 contract)
# --------------------------------------------------------------------------


class TestEpisodeLoop:
    def test_invokes_run_policy_exactly_n_times_without_recording(self) -> None:
        """No dataset_root -> smoke-test mode, no recording, but N rollouts."""
        sim = _FakeSim()
        result = run_policy(
            sim,
            policy_provider="mock",
            n_episodes=5,
            n_steps=12,
            fast_mode=True,
        )
        assert result["status"] == "success"
        assert len(sim.run_policy_calls) == 5
        assert sim.start_recording_calls == []
        assert sim.stop_recording_calls == []

    def test_invokes_run_policy_exactly_n_times_with_recording(self, tmp_path: Path) -> None:
        sim = _FakeSim()
        dataset_root = tmp_path / "ds"
        # Pre-write info.json so the parquet-truth read finds N=3 / 36 frames.
        _write_info_json(dataset_root, total_episodes=3, total_frames=36)
        result = run_policy(
            sim,
            policy_provider="mock",
            n_episodes=3,
            n_steps=12,
            dataset_root=str(dataset_root),
        )
        assert result["status"] == "success"
        assert len(sim.run_policy_calls) == 3
        assert len(sim.start_recording_calls) == 1
        assert len(sim.stop_recording_calls) == 1

    def test_forwards_policy_args_to_run_policy(self) -> None:
        sim = _FakeSim()
        run_policy(
            sim,
            robot_name="alice",
            policy_provider="lerobot_local",
            policy_config={"pretrained_name_or_path": "lerobot/act"},
            instruction="pick the cube",
            n_episodes=2,
            n_steps=20,
            control_frequency=25.0,
            action_horizon=4,
            policy_kwargs={"target_pose": [0, 0, 0]},
        )
        for call in sim.run_policy_calls:
            assert call["robot_name"] == "alice"
            assert call["policy_provider"] == "lerobot_local"
            assert call["policy_config"] == {"pretrained_name_or_path": "lerobot/act"}
            assert call["instruction"] == "pick the cube"
            assert call["control_frequency"] == 25.0
            assert call["action_horizon"] == 4
            assert call["n_steps"] == 20
            assert call["max_steps"] == 20
            assert call["policy_kwargs"] == {"target_pose": [0, 0, 0]}

    def test_seed_increments_per_episode(self) -> None:
        sim = _FakeSim()
        run_policy(sim, n_episodes=3, n_steps=5, seed=42)
        seeds = [c["seed"] for c in sim.run_policy_calls]
        assert seeds == [42, 43, 44]

    def test_seed_none_passes_none_per_episode(self) -> None:
        sim = _FakeSim()
        run_policy(sim, n_episodes=2, n_steps=5, seed=None)
        seeds = [c["seed"] for c in sim.run_policy_calls]
        assert seeds == [None, None]

    def test_continues_after_episode_error_and_records_failure(self) -> None:
        """One bad episode must NOT abort the loop - partial success is reported."""
        sim = _FakeSim()
        call_count = {"n": 0}

        def flaky(**kwargs: Any) -> dict[str, Any]:
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("policy server flaked")
            return _ok_rollout(f"ep {call_count['n']}")

        sim.run_policy = flaky  # type: ignore[method-assign]

        result = run_policy(sim, n_episodes=3, n_steps=5)
        # Mixed outcomes -> status=error so a verifier doesn't false-OK.
        assert result["status"] == "error"
        payload = result["content"][1]["json"]
        assert payload["n_episodes_requested"] == 3
        assert payload["n_episodes_ok"] == 2
        ep_statuses = [e["status"] for e in payload["episodes"]]
        assert ep_statuses == ["success", "error", "success"]


# --------------------------------------------------------------------------
# Parquet-truth gate
# --------------------------------------------------------------------------


class TestParquetTruth:
    def test_returns_total_episodes_from_info_json(self, tmp_path: Path) -> None:
        sim = _FakeSim()
        ds = tmp_path / "ds"
        _write_info_json(ds, total_episodes=4, total_frames=240)
        result = run_policy(sim, n_episodes=4, n_steps=60, dataset_root=str(ds))
        payload = result["content"][1]["json"]
        assert payload["n_episodes_actual"] == 4
        assert payload["n_frames_actual"] == 240
        assert payload["warnings"] == []
        assert result["status"] == "success"

    def test_flags_fabrication_when_truth_disagrees(self, tmp_path: Path) -> None:
        """The smoking-gun case: loop says 20, parquet says 1."""
        sim = _FakeSim()
        ds = tmp_path / "ds"
        # Simulate the historical fabrication: 20 requested, 1 actually saved.
        _write_info_json(ds, total_episodes=1, total_frames=1140)
        result = run_policy(sim, n_episodes=20, n_steps=60, dataset_root=str(ds))
        assert result["status"] == "error", "Mismatch between requested and parquet-actual MUST flip status to error"
        payload = result["content"][1]["json"]
        assert payload["n_episodes_requested"] == 20
        assert payload["n_episodes_actual"] == 1
        assert any("FABRICATION GUARD" in w for w in payload["warnings"])

    def test_flags_missing_info_json(self, tmp_path: Path) -> None:
        sim = _FakeSim()
        ds = tmp_path / "ds_empty"  # no meta/info.json
        result = run_policy(sim, n_episodes=2, n_steps=5, dataset_root=str(ds))
        payload = result["content"][1]["json"]
        assert payload["n_episodes_actual"] == -1
        assert any("info.json missing" in w for w in payload["warnings"])
        assert result["status"] == "error"

    def test_no_truth_check_when_dataset_root_is_none(self) -> None:
        sim = _FakeSim()
        result = run_policy(sim, n_episodes=2, n_steps=5, dataset_root=None)
        payload = result["content"][1]["json"]
        assert payload["dataset_root"] is None
        assert payload["warnings"] == []
        assert result["status"] == "success"


# --------------------------------------------------------------------------
# Recording lifecycle
# --------------------------------------------------------------------------


class TestRecordingLifecycle:
    def test_stop_recording_runs_in_finally_on_episode_exception(self, tmp_path: Path) -> None:
        """Even if every episode errors, stop_recording MUST close the dataset."""
        sim = _FakeSim()
        ds = tmp_path / "ds"
        _write_info_json(ds, total_episodes=0, total_frames=0)

        def boom(**kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("policy not loaded")

        sim.run_policy = boom  # type: ignore[method-assign]

        result = run_policy(sim, n_episodes=2, n_steps=5, dataset_root=str(ds))
        assert result["status"] == "error"
        # The dataset MUST have been closed - even on full failure.
        assert len(sim.stop_recording_calls) == 1

    def test_surfaces_start_recording_failure(self, tmp_path: Path) -> None:
        sim = _FakeSim()

        def bad_start(**kwargs: Any) -> dict[str, Any]:
            return {"status": "error", "content": [{"text": "lerobot extra missing"}]}

        sim.start_recording = bad_start  # type: ignore[method-assign]

        result = run_policy(
            sim,
            n_episodes=1,
            n_steps=5,
            dataset_root=str(tmp_path / "ds"),
        )
        assert result["status"] == "error"
        assert "lerobot extra missing" in result["content"][0]["text"]
        # No rollouts and no stop_recording because start_recording failed.
        assert sim.run_policy_calls == []
        assert sim.stop_recording_calls == []

    def test_recording_disabled_when_simulation_lacks_start_recording(self, tmp_path: Path) -> None:
        sim = _NoRecordingSim()
        result = run_policy(
            sim,
            n_episodes=1,
            n_steps=5,
            dataset_root=str(tmp_path / "ds"),
        )
        assert result["status"] == "error"
        assert "start_recording" in result["content"][0]["text"]


# --------------------------------------------------------------------------
# Decorator surface
# --------------------------------------------------------------------------


class TestDecoratorSurface:
    def test_tool_is_lazily_importable(self) -> None:
        """``strands_robots.tools.run_policy`` must materialize via __getattr__."""
        import strands_robots.tools as tools_pkg

        # Force fresh resolution
        vars(tools_pkg).pop("run_policy", None)
        value = getattr(tools_pkg, "run_policy")
        assert value is not None
        # Tool object should expose the standard Strands @tool surface.
        # DecoratedFunctionTool exposes tool_name + a callable underlying func.
        assert hasattr(value, "tool_name")
        assert hasattr(value, "_tool_func") or callable(value)


# --------------------------------------------------------------------------
# Per-episode finalize resilience
# --------------------------------------------------------------------------


class TestFinalizeResilience:
    """The per-episode parquet boundary must never abort an in-flight rollout.

    When recording is active, ``run_policy`` delegates the per-episode
    ``save_episode`` boundary to ``PolicyRunner._finalize_recorder_episode``.
    That delegation is best-effort by contract: a missing ``PolicyRunner``
    import, a construction failure, or a save error must be logged and
    swallowed so a transient recorder fault never tears down an N-episode
    rollout mid-way. These tests pin that the loop still completes all N
    episodes and the dataset is still closed in each failure mode.
    """

    def test_finalize_swallows_policyrunner_import_error(self, tmp_path: Path, monkeypatch: Any) -> None:
        import sys

        sim = _FakeSim()
        ds = tmp_path / "ds"
        _write_info_json(ds, total_episodes=2, total_frames=24)
        # Force `from ...policy_runner import PolicyRunner` to raise ImportError:
        # a None entry in sys.modules makes the `from ... import` statement fail.
        monkeypatch.setitem(sys.modules, "strands_robots.simulation.policy_runner", None)

        result = run_policy(sim, n_episodes=2, n_steps=5, dataset_root=str(ds))

        # Import failure is non-fatal: both rollouts ran and the dataset closed.
        assert len(sim.run_policy_calls) == 2
        assert len(sim.stop_recording_calls) == 1
        assert result["status"] == "success"

    def test_finalize_swallows_policyrunner_construction_error(self, tmp_path: Path, monkeypatch: Any) -> None:
        import strands_robots.simulation.policy_runner as pr_mod

        sim = _FakeSim()
        ds = tmp_path / "ds"
        _write_info_json(ds, total_episodes=1, total_frames=12)

        class _BoomRunner:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("cannot bind PolicyRunner to this engine")

        monkeypatch.setattr(pr_mod, "PolicyRunner", _BoomRunner)

        result = run_policy(sim, n_episodes=1, n_steps=5, dataset_root=str(ds))

        assert len(sim.run_policy_calls) == 1
        assert len(sim.stop_recording_calls) == 1
        assert result["status"] == "success"

    def test_finalize_swallows_save_episode_error(self, tmp_path: Path, monkeypatch: Any) -> None:
        import strands_robots.simulation.policy_runner as pr_mod

        sim = _FakeSim()
        ds = tmp_path / "ds"
        _write_info_json(ds, total_episodes=1, total_frames=12)

        class _FlakyRunner:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def _finalize_recorder_episode(self) -> None:
                raise RuntimeError("save_episode failed: disk full")

        monkeypatch.setattr(pr_mod, "PolicyRunner", _FlakyRunner)

        result = run_policy(sim, n_episodes=1, n_steps=5, dataset_root=str(ds))

        # A per-episode save error is swallowed; the rollout still completes.
        assert len(sim.run_policy_calls) == 1
        assert len(sim.stop_recording_calls) == 1
        assert result["status"] == "success"


# --------------------------------------------------------------------------
# stop_recording fault tolerance + corrupt parquet-truth metadata
# --------------------------------------------------------------------------


class TestStopRecordingFaultTolerance:
    def test_non_success_stop_recording_is_logged_not_raised(self, tmp_path: Path) -> None:
        """A failing ``stop_recording`` in the finally block must not crash.

        The tool logs the non-success result and proceeds to read
        parquet-truth from ``meta/info.json`` (which is sync-flushed
        independently of the recorder's stop path).
        """
        sim = _FakeSim()
        ds = tmp_path / "ds"
        _write_info_json(ds, total_episodes=1, total_frames=12)

        def bad_stop(**kwargs: Any) -> dict[str, Any]:
            return {"status": "error", "content": [{"text": "flush failed"}]}

        sim.stop_recording = bad_stop  # type: ignore[method-assign]

        result = run_policy(sim, n_episodes=1, n_steps=5, dataset_root=str(ds))

        # Parquet-truth still resolved from info.json despite the stop failure.
        payload = result["content"][1]["json"]
        assert payload["n_episodes_actual"] == 1
        assert payload["n_frames_actual"] == 12
        assert result["status"] == "success"


class TestCorruptParquetTruth:
    def test_corrupt_info_json_is_surfaced_as_warning_not_traceback(self, tmp_path: Path) -> None:
        """A malformed ``meta/info.json`` MUST degrade to a structured warning.

        ``_read_parquet_truth`` swallows ``json.JSONDecodeError`` / ``OSError``
        and returns ``info_present=False`` with the error repr, so the verifier
        sees "cannot verify episode count" rather than a stack trace.
        """
        sim = _FakeSim()
        ds = tmp_path / "ds"
        meta = ds / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text("{ this is not valid json ")

        result = run_policy(sim, n_episodes=1, n_steps=5, dataset_root=str(ds))

        payload = result["content"][1]["json"]
        assert payload["n_episodes_actual"] == -1
        assert any("info.json missing" in w for w in payload["warnings"])
        # Cannot confirm the count -> status must not be a false OK.
        assert result["status"] == "error"


class TestDatasetCameraScoping:
    """Pins that ``run_policy`` can scope which cameras land in the dataset.

    The MuJoCo backend's ``start_recording`` accepts a ``cameras`` subset so a
    policy-specific dataset only carries the views the policy declares (e.g.
    SmolVLA's ``camera1/2/3``) and excludes the implicit ``default`` free
    camera every scene ships. Before this contract, ``run_policy`` - the
    recording entry point that owns the ``start_recording`` -> ``stop_recording``
    cycle - had no way to forward that subset, so every recorded dataset was
    forced to include the stray ``default`` view in ``observation.images.*``.
    """

    def test_forwards_dataset_cameras_to_start_recording(self, tmp_path: Path) -> None:
        sim = _FakeSim()
        ds = tmp_path / "ds"
        _write_info_json(ds, total_episodes=1, total_frames=12)
        result = run_policy(
            sim,
            policy_provider="mock",
            n_episodes=1,
            n_steps=12,
            dataset_root=str(ds),
            dataset_cameras=["camera1", "camera2", "camera3"],
        )
        assert result["status"] == "success"
        assert len(sim.start_recording_calls) == 1
        assert sim.start_recording_calls[0]["cameras"] == ["camera1", "camera2", "camera3"]

    def test_dataset_cameras_omitted_forwards_no_cameras_kwarg(self, tmp_path: Path) -> None:
        """Default (no scoping) forwards NO ``cameras`` kwarg at all so the
        backend keeps its record-every-camera behaviour - the opt-in subset
        never touches the default path. Forwarding ``cameras=None`` would crash
        any backend (e.g. Newton) whose ``start_recording`` has no ``cameras``
        parameter, so the kwarg must be entirely absent when unset."""
        sim = _FakeSim()
        ds = tmp_path / "ds"
        _write_info_json(ds, total_episodes=1, total_frames=6)
        run_policy(sim, n_episodes=1, n_steps=6, dataset_root=str(ds))
        assert len(sim.start_recording_calls) == 1
        assert "cameras" not in sim.start_recording_calls[0]

    def test_omitted_dataset_cameras_does_not_crash_camerasless_backend(self, tmp_path: Path) -> None:
        """Regression: a backend whose ``start_recording`` has no ``cameras``
        parameter and no ``**kwargs`` catch-all (mirroring the Newton backend,
        ``simulation/newton/recording.py``) must record cleanly when
        ``dataset_cameras`` is omitted. The pre-fix code forwarded
        ``cameras=None`` unconditionally - before the ``try:`` block - so the
        resulting ``TypeError`` propagated uncaught out of this backend-agnostic
        tool, violating the 'return error dicts, never raise' handler contract.
        """
        sim = _CameraslessSim()
        ds = tmp_path / "ds"
        _write_info_json(ds, total_episodes=1, total_frames=6)
        result = run_policy(sim, n_episodes=1, n_steps=6, dataset_root=str(ds))
        assert result["status"] == "success"
        assert sim.start_recording_calls == 1


class TestVideoPassthrough:
    """The tool MUST forward a ``video`` config to ``Simulation.run_policy``.

    The facade (``Simulation.run_policy``) records a rollout MP4 when given a
    ``video={...}`` dict, but the multi-episode tool wrapper historically had no
    ``video`` parameter at all - so an agent driving rollouts through the tool
    could record a LeRobotDataset yet had no way to capture the rollout video
    that downstream review (and visual defect-checking) needs. These tests pin
    the passthrough, the per-episode path derivation, and the validation.
    """

    def test_forwards_video_verbatim_single_episode(self) -> None:
        sim = _FakeSim()
        video = {"path": "/tmp/rollout.mp4", "fps": 30, "camera": "camera1"}
        run_policy(sim, n_episodes=1, n_steps=4, video=video)
        assert len(sim.run_policy_calls) == 1
        # Single episode uses the path verbatim (the common artifact case).
        assert sim.run_policy_calls[0]["video"] == video

    def test_default_video_is_none(self) -> None:
        sim = _FakeSim()
        run_policy(sim, n_episodes=2, n_steps=4)
        assert sim.run_policy_calls, "expected at least one rollout call"
        for call in sim.run_policy_calls:
            assert call["video"] is None

    def test_per_episode_video_paths_are_distinct(self) -> None:
        sim = _FakeSim()
        run_policy(
            sim,
            n_episodes=3,
            n_steps=4,
            video={"path": "/tmp/rollout.mp4", "fps": 30},
        )
        paths = [c["video"]["path"] for c in sim.run_policy_calls]
        assert paths == [
            "/tmp/rollout_ep0.mp4",
            "/tmp/rollout_ep1.mp4",
            "/tmp/rollout_ep2.mp4",
        ]
        # Non-path keys are preserved per episode.
        assert all(c["video"]["fps"] == 30 for c in sim.run_policy_calls)

    def test_rejects_non_dict_video(self) -> None:
        sim = _FakeSim()
        result = run_policy(sim, n_episodes=1, n_steps=4, video="rollout.mp4")  # type: ignore[arg-type]
        assert result["status"] == "error"
        assert "video must be a dict" in result["content"][0]["text"]
        assert sim.run_policy_calls == []

    def test_video_paths_reports_only_existing_files(self, tmp_path: Path) -> None:
        sim = _FakeSim()
        # _FakeSim never actually writes an MP4, so video_paths must be empty
        # (honest reporting: only files that landed on disk are listed).
        result = run_policy(
            sim,
            n_episodes=1,
            n_steps=4,
            video={"path": str(tmp_path / "rollout.mp4"), "fps": 30},
        )
        payload = result["content"][1]["json"]
        assert payload["video_paths"] == []

    def test_falsy_video_path_disables_recording(self) -> None:
        sim = _FakeSim()
        result = run_policy(sim, n_episodes=1, n_steps=4, video={"fps": 30})
        assert result["status"] == "success"
        # Dict without a usable path is forwarded (facade treats it as "off").
        assert sim.run_policy_calls[0]["video"] == {"fps": 30}
        assert result["content"][1]["json"]["video_paths"] == []


# --------------------------------------------------------------------------
# Success-payload content contract
# --------------------------------------------------------------------------


class TestSuccessPayloadContentShape:
    """Pin the two-part ``content`` contract of a successful rollout.

    A successful ``run_policy`` result carries EXACTLY two content parts: a
    human-readable ``{"text": summary}`` followed by a machine-readable
    ``{"json": payload}`` whose keys downstream verifiers parse (episode /
    frame counts, warnings, dataset root). Locking this shape guards against
    a regression to a single ``{"text": ...}`` payload, which would strand the
    parquet-truth fields the fabrication guard depends on.
    """

    def test_success_content_is_text_then_json(self) -> None:
        sim = _FakeSim()
        result = run_policy(sim, n_episodes=2, n_steps=5, dataset_root=None)

        assert result["status"] == "success"
        content = result["content"]
        assert len(content) == 2, "success payload must be [text, json], not a single text part"

        assert set(content[0]) == {"text"}
        assert isinstance(content[0]["text"], str) and content[0]["text"].strip()

        assert set(content[1]) == {"json"}
        payload = content[1]["json"]
        assert isinstance(payload, dict)
        # The parquet-truth / accounting keys a verifier reads off the payload.
        for key in (
            "n_episodes_requested",
            "n_episodes_actual",
            "n_frames_actual",
            "n_episodes_ok",
            "dataset_root",
            "warnings",
            "episodes",
        ):
            assert key in payload, f"success payload missing contract key {key!r}"
