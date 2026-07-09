"""Async-RTC chunk pipeline in :class:`PolicyRunner.run`.

The synchronous loop queries the policy, fully drains the returned chunk, then
re-queries - so inference never overlaps execution and a chunk-emitting VLA
shows a per-seam stall in sim that it would NOT show on real hardware (where an
async controller hides inference latency behind chunk execution).

``run(..., async_rtc=True)`` overlaps the two: while the current chunk drains,
the next ``get_actions`` is fired on a single background worker once the chunk
is ~50% consumed, then atomically swapped in. These tests pin:

* overlap: the next inference STARTS mid-chunk (before the current chunk drains)
* synchronous baseline: inference only starts after the chunk fully drains
* latency masking: async wall-time is materially lower than synchronous
* correctness/back-compat: identical step accounting; default is synchronous
"""

from __future__ import annotations

import inspect
import threading
import time
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import PolicyRunner

_CHUNK = 4
_INFER_SLEEP = 0.05
_EXEC_SLEEP = 0.02  # per send_action; chunk exec (~0.08s) > infer (0.05s) -> hidden


class _CountingSim(SimEngine):
    """Minimal ``SimEngine`` that counts ``send_action`` calls and can pace them.

    ``send_count`` is incremented BEFORE the optional per-action sleep so a
    concurrently-running policy that reads it observes the step index reached so
    far, which is what the overlap assertions key off.
    """

    def __init__(self, exec_sleep: float = 0.0) -> None:
        self._joint_names = ["j0", "j1", "j2"]
        self._robots = {"arm": self._joint_names}
        self._exec_sleep = exec_sleep
        self._lock = threading.Lock()
        self.send_count = 0

    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        return {"status": "success"}

    def step(self, n_steps: int = 1):
        return {"status": "success"}

    def get_state(self):
        return {"sim_time": 0.0, "step_count": self.send_count}

    def add_robot(self, name, **kw):
        return {"status": "success"}

    def remove_robot(self, name):
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return list(self._robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(self._robots.get(robot_name, []))

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        return {n: 0.0 for n in self._joint_names}

    def send_action(self, action, robot_name=None, n_substeps=1):
        with self._lock:
            self.send_count += 1
        if self._exec_sleep:
            time.sleep(self._exec_sleep)

    def render(self, camera_name="default", width=None, height=None):
        return {"image": np.zeros((height or 48, width or 64, 3), dtype=np.uint8)}


class _ChunkPolicy(Policy):
    """Emits fixed-size chunks; records the sim step index at each inference start.

    ``infer_starts[k]`` is the value of ``sim.send_count`` captured at the very
    first line of the k-th ``get_actions`` call (before the inference sleep). In
    the synchronous loop these are exact multiples of the chunk size (the chunk
    is always fully drained before the next query); the async pipeline fires the
    query mid-chunk, so at least one value is NOT a chunk multiple.
    """

    def __init__(self, sim: _CountingSim, chunk: int = _CHUNK, infer_sleep: float = _INFER_SLEEP) -> None:
        self._sim = sim
        self.actions_per_step = chunk
        self._chunk = chunk
        self._infer_sleep = infer_sleep
        self.robot_state_keys: list[str] = []
        self.infer_starts: list[int] = []
        self.observed_delays: list[int | None] = []
        self._lock = threading.Lock()

    @property
    def provider_name(self) -> str:
        return "chunk-test"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.robot_state_keys = robot_state_keys

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        with self._lock:
            self.infer_starts.append(self._sim.send_count)
            # The runtime sets the deterministic inference-delay step count just
            # before each call (see PolicyRunner._query_chunk).
            self.observed_delays.append(self.rtc_observed_delay_steps)
        if self._infer_sleep:
            time.sleep(self._infer_sleep)
        keys = self.robot_state_keys or ["j0", "j1", "j2"]
        return [{k: 0.0 for k in keys} for _ in range(self._chunk)]


def _run(
    async_rtc: bool, *, exec_sleep: float = _EXEC_SLEEP, n_steps: int = 16
) -> tuple[dict, _ChunkPolicy, _CountingSim]:
    sim = _CountingSim(exec_sleep=exec_sleep)
    policy = _ChunkPolicy(sim)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    duration = n_steps / 50.0
    result = PolicyRunner(sim).run(
        "arm",
        policy,
        duration=duration,
        control_frequency=50.0,
        action_horizon=_CHUNK,
        fast_mode=True,
        async_rtc=async_rtc,
    )
    return result, policy, sim


def test_async_rtc_starts_next_inference_mid_chunk() -> None:
    """The prefetched chunk-(N+1) inference genuinely overlaps chunk-N execution.

    Proven by a rendezvous, NOT a wall-clock timing race: the consumer blocks
    the final action of the first chunk until the prefetch worker has entered
    ``get_actions``, so the step index captured at inference start is guaranteed
    to be observed strictly before the chunk drains - deterministically,
    independent of thread-scheduling latency under CPU load. (Reading
    ``send_count`` on the worker thread and asserting ``< chunk`` without this
    handshake flaked under load: the worker could be scheduled only after the
    consumer had already drained the chunk, reading a boundary value.)

    The rendezvous also pins the mid-chunk trigger itself: were the prefetch
    submitted only at the chunk boundary (no overlap), the worker would never
    start before the blocked final send and the wait would time out.
    """
    infer_started = threading.Event()

    class _RendezvousSim(_CountingSim):
        def send_action(self, action, robot_name=None, n_substeps=1):
            # Block the final action of the first chunk until the prefetch
            # worker has begun inference. The worker is submitted at the
            # mid-chunk trigger (strictly before this step), so it cannot
            # deadlock; if overlap regressed to boundary-only prefetch this
            # wait times out and fails the test.
            with self._lock:
                about_to_send = self.send_count + 1
            if about_to_send == _CHUNK:
                assert infer_started.wait(timeout=5.0), "prefetch worker did not start before the first chunk drained"
            with self._lock:
                self.send_count += 1
            if self._exec_sleep:
                time.sleep(self._exec_sleep)

    class _RendezvousPolicy(_ChunkPolicy):
        async def get_actions(
            self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
        ) -> list[dict[str, Any]]:
            with self._lock:
                is_first_prefetch = len(self.infer_starts) == 1
                self.infer_starts.append(self._sim.send_count)
                self.observed_delays.append(self.rtc_observed_delay_steps)
            if is_first_prefetch:
                infer_started.set()
            if self._infer_sleep:
                time.sleep(self._infer_sleep)
            keys = self.robot_state_keys or ["j0", "j1", "j2"]
            return [{k: 0.0 for k in keys} for _ in range(self._chunk)]

    sim = _RendezvousSim(exec_sleep=_EXEC_SLEEP)
    policy = _RendezvousPolicy(sim)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    result = PolicyRunner(sim).run(
        "arm",
        policy,
        duration=16 / 50.0,
        control_frequency=50.0,
        action_horizon=_CHUNK,
        fast_mode=True,
        async_rtc=True,
    )
    assert result["status"] == "success"
    # More than one chunk was consumed, so the prefetch had to fire.
    assert len(policy.infer_starts) >= 2
    # The first prefetch's inference began strictly before the first chunk's
    # final action drained - genuine overlap, guaranteed by the rendezvous.
    assert policy.infer_starts[1] < _CHUNK, policy.infer_starts


def test_sync_only_queries_after_chunk_drains() -> None:
    """The synchronous baseline never overlaps: every query is on a boundary."""
    result, policy, sim = _run(async_rtc=False)
    assert result["status"] == "success"
    assert len(policy.infer_starts) >= 2
    assert all(c % _CHUNK == 0 for c in policy.infer_starts), policy.infer_starts


def test_async_rtc_masks_inference_latency() -> None:
    """Async wall-time is materially lower because inference hides behind exec."""
    t0 = time.perf_counter()
    sync_result, _, _ = _run(async_rtc=False)
    sync_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    async_result, _, _ = _run(async_rtc=True)
    async_elapsed = time.perf_counter() - t0

    assert sync_result["status"] == "success"
    assert async_result["status"] == "success"
    # 16 steps / chunk 4 => 4 chunks. Sync pays infer_sleep per chunk serially;
    # async hides all but (at most) the first. Saving >= 2 inference periods is
    # a conservative, non-flaky margin.
    assert sync_elapsed - async_elapsed > 2 * _INFER_SLEEP, (sync_elapsed, async_elapsed)


def test_async_rtc_matches_sync_step_accounting() -> None:
    """Both paths run the same number of steps and send the same action count."""
    sync_result, _, sync_sim = _run(async_rtc=False, exec_sleep=0.0)
    async_result, _, async_sim = _run(async_rtc=True, exec_sleep=0.0)

    sync_json = sync_result["content"][1]["json"]
    async_json = async_result["content"][1]["json"]
    assert sync_json["n_steps"] == async_json["n_steps"] == 16
    assert sync_sim.send_count == async_sim.send_count == 16
    assert async_json["action_errors"] == 0


def test_async_rtc_defaults_to_none_and_auto_resolves() -> None:
    """The default is ``None`` (auto-resolve), not a hardcoded ``False``.

    ``None`` means "let the policy decide" - chunk-emitting VLAs opt into
    latency masking automatically while single-step policies stay synchronous.
    """
    assert inspect.signature(PolicyRunner.run).parameters["async_rtc"].default is None
    assert inspect.signature(SimEngine.run_policy).parameters["async_rtc"].default is None


def test_sync_loop_supplies_zero_observed_delay() -> None:
    """Synchronous loop: the world is paused during inference, so the runtime
    tells the policy exactly 0 control steps elapsed (not a wall-clock guess)."""
    _, policy, _ = _run(async_rtc=False)
    assert policy.observed_delays, "policy was never queried"
    assert all(d == 0 for d in policy.observed_delays), policy.observed_delays


def test_async_rtc_supplies_deterministic_observed_delay() -> None:
    """Async pipeline: prefetched chunks carry the EXACT count of still-pending
    steps (len(chunk) - prefetch_trigger), a known integer independent of how
    long inference actually took. The cold-start and short-chunk re-queries are
    synchronous (delay 0)."""
    _, policy, _ = _run(async_rtc=True)
    assert policy.observed_delays, "policy was never queried"
    # prefetch_trigger = max(1, _CHUNK // 2); pending = _CHUNK - prefetch_trigger.
    expected_prefetch_delay = _CHUNK - max(1, _CHUNK // 2)
    assert all(d in (0, expected_prefetch_delay) for d in policy.observed_delays), policy.observed_delays
    # At least one prefetched (overlapped) query must carry the non-zero count -
    # otherwise the async path silently degraded to synchronous re-queries.
    assert any(d == expected_prefetch_delay for d in policy.observed_delays), policy.observed_delays


# --- is_chunk_emitting + auto-enable contract -----------------------------


class _SingleStepPolicy(Policy):
    """One action per ``get_actions`` (``MockPolicy`` shape) -> not chunk-emitting."""

    def __init__(self) -> None:
        self.actions_per_step = 1
        self.robot_state_keys: list[str] = []

    @property
    def provider_name(self) -> str:
        return "single-step-test"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.robot_state_keys = robot_state_keys

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        keys = self.robot_state_keys or ["j0", "j1", "j2"]
        return [{k: 0.0 for k in keys}]


def test_is_chunk_emitting_true_for_multistep_policy() -> None:
    """A policy whose execution horizon is > 1 reports itself chunk-emitting."""
    sim = _CountingSim()
    assert _ChunkPolicy(sim, chunk=4).is_chunk_emitting() is True


def test_is_chunk_emitting_false_for_single_step_policy() -> None:
    """A single-action policy is NOT chunk-emitting (overlap would not help)."""
    assert _SingleStepPolicy().is_chunk_emitting() is False


def test_async_rtc_auto_enabled_for_chunk_policy() -> None:
    """``async_rtc=None`` (default) auto-enables overlap for a chunk policy."""
    sim = _CountingSim(exec_sleep=_EXEC_SLEEP)
    policy = _ChunkPolicy(sim)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    # No async_rtc kwarg at all -> uses the None default -> resolves via the
    # policy. A chunk-emitting policy must enable the overlap.
    result = PolicyRunner(sim).run(
        "arm", policy, duration=16 / 50.0, control_frequency=50.0, action_horizon=_CHUNK, fast_mode=True
    )
    assert result["status"] == "success"
    telem = result["content"][1]["json"]
    assert telem["rtc_async_enabled"] is True
    # Prefetch fired mid-chunk (the overlap actually ran).
    assert any(c % _CHUNK != 0 for c in policy.infer_starts), policy.infer_starts


def test_async_rtc_auto_disabled_for_single_step_policy() -> None:
    """``async_rtc=None`` keeps a single-step policy on the synchronous loop."""
    sim = _CountingSim()
    policy = _SingleStepPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    result = PolicyRunner(sim).run("arm", policy, duration=8 / 50.0, control_frequency=50.0, fast_mode=True)
    assert result["status"] == "success"
    assert result["content"][1]["json"]["rtc_async_enabled"] is False


def test_async_overlap_enabled_but_seam_not_blended_for_non_rtc_chunk_policy() -> None:
    """Overlap auto-enable and RTC seam-blending are independent capabilities.

    ``run_policy``'s docstring distinguishes two things that a reader can easily
    conflate: the async OVERLAP (latency masking, auto-enabled for *any*
    chunk-emitting policy via ``is_chunk_emitting()``) and RTC SEAM BLENDING (a
    checkpoint-level property, ``supports_rtc``, that joins consecutive chunks
    smoothly and requires an enabled ``rtc_config``). A chunk-emitting policy
    that does NOT support RTC - the shape of the public ``lerobot/smolvla_base``
    checkpoint (``rtc_config=None``), MolmoAct2 (no ``rtc_config``), ACT and
    diffusion - must still get the overlap, but its seam is a plain chunk swap,
    not a blended one. This pins that ``rtc_async_enabled`` can be ``True`` while
    ``supports_rtc`` is ``False``, so the two never get collapsed back together.
    """
    sim = _CountingSim(exec_sleep=_EXEC_SLEEP)
    policy = _ChunkPolicy(sim)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))

    # A plain chunk emitter declares no RTC support (no enabled rtc_config).
    assert getattr(policy, "supports_rtc", False) is False
    assert policy.is_chunk_emitting() is True

    result = PolicyRunner(sim).run(
        "arm", policy, duration=16 / 50.0, control_frequency=50.0, action_horizon=_CHUNK, fast_mode=True
    )
    assert result["status"] == "success"
    telem = result["content"][1]["json"]
    # Overlap auto-enabled (latency masking) ...
    assert telem["rtc_async_enabled"] is True
    # ... yet the policy still reports no internal RTC seam blending.
    assert getattr(policy, "supports_rtc", False) is False


# --- telemetry block ------------------------------------------------------

_RTC_KEYS = {
    "rtc_async_enabled",
    "rtc_chunks_acquired",
    "rtc_prefetch_hits",
    "rtc_prefetch_blocks",
    "rtc_avg_inference_ms",
    "rtc_max_inference_ms",
}


def test_telemetry_fields_present_on_both_paths() -> None:
    """All six RTC telemetry fields appear in the json on sync AND async runs."""
    for async_rtc in (False, True):
        result, _, _ = _run(async_rtc=async_rtc, exec_sleep=0.0)
        telem = result["content"][1]["json"]
        assert _RTC_KEYS <= set(telem), (async_rtc, telem)
        assert telem["rtc_async_enabled"] is async_rtc


def test_async_rtc_prefetch_blocks_when_inference_slow() -> None:
    """Fake-slow policy: inference > chunk exec -> prefetch blocks at every seam,
    and total wall time tracks inference x n_chunks (not x 2 - no double pay)."""
    # chunk exec ~0.04s (4 x 0.01) << infer 0.10s -> inference cannot be hidden,
    # so every swap blocks. Keep exec tiny so wall time is dominated by inference.
    n_steps = 16
    n_chunks = n_steps // _CHUNK
    infer = 0.10
    sim = _CountingSim(exec_sleep=0.01)
    policy = _ChunkPolicy(sim, infer_sleep=infer)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    t0 = time.perf_counter()
    result = PolicyRunner(sim).run(
        "arm",
        policy,
        duration=n_steps / 50.0,
        control_frequency=50.0,
        action_horizon=_CHUNK,
        fast_mode=True,
        async_rtc=True,
    )
    elapsed = time.perf_counter() - t0
    assert result["status"] == "success"
    telem = result["content"][1]["json"]
    assert telem["rtc_prefetch_blocks"] > 0, telem
    # Each chunk's inference is paid ONCE (serialized behind the next), never
    # twice: wall time stays under the 2x-per-chunk ceiling the issue calls out.
    assert elapsed < 2 * infer * n_chunks, (elapsed, infer, n_chunks)


def test_async_rtc_prefetch_hits_when_inference_fast() -> None:
    """Fake-fast policy: inference << chunk exec -> every seam is a hit, no blocks."""
    n_steps = 16
    n_chunks = n_steps // _CHUNK
    # infer 0.005s << exec 0.08s/chunk -> the prefetch is always ready at swap.
    sim = _CountingSim(exec_sleep=0.02)
    policy = _ChunkPolicy(sim, infer_sleep=0.005)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    result = PolicyRunner(sim).run(
        "arm",
        policy,
        duration=n_steps / 50.0,
        control_frequency=50.0,
        action_horizon=_CHUNK,
        fast_mode=True,
        async_rtc=True,
    )
    assert result["status"] == "success"
    telem = result["content"][1]["json"]
    assert telem["rtc_prefetch_blocks"] == 0, telem
    assert telem["rtc_prefetch_hits"] == n_chunks - 1, telem


# --- empty-chunk drop-and-requery fallback --------------------------------


class _EmptyOnCallsPolicy(_ChunkPolicy):
    """Emits an EMPTY chunk on the configured (1-indexed) inference calls.

    Lets a test force a prefetched chunk to arrive empty and assert the runner
    degrades to one synchronous re-query rather than killing the rollout.
    """

    def __init__(self, sim: _CountingSim, empty_calls: set[int], **kw: Any) -> None:
        super().__init__(sim, **kw)
        self._empty_calls = empty_calls
        self._calls = 0

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._calls += 1
            call_no = self._calls
            self.infer_starts.append(self._sim.send_count)
            self.observed_delays.append(self.rtc_observed_delay_steps)
        if self._infer_sleep:
            time.sleep(self._infer_sleep)
        if call_no in self._empty_calls:
            return []
        keys = self.robot_state_keys or ["j0", "j1", "j2"]
        return [{k: 0.0 for k in keys} for _ in range(self._chunk)]


def test_async_rtc_empty_prefetch_degrades_to_resync() -> None:
    """An empty prefetched chunk triggers ONE synchronous re-query, not an error."""
    sim = _CountingSim(exec_sleep=0.0)
    # Call 1 = cold start (non-empty), call 2 = first prefetch (empty -> resync).
    policy = _EmptyOnCallsPolicy(sim, empty_calls={2}, infer_sleep=0.005)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    result = PolicyRunner(sim).run(
        "arm",
        policy,
        duration=16 / 50.0,
        control_frequency=50.0,
        action_horizon=_CHUNK,
        fast_mode=True,
        async_rtc=True,
    )
    assert result["status"] == "success", result
    assert result["content"][1]["json"]["n_steps"] == 16


def test_async_rtc_empty_twice_errors() -> None:
    """Empty prefetch AND empty re-query is fatal (structured error, no hang)."""
    sim = _CountingSim(exec_sleep=0.0)
    # Every inference after the cold start returns empty -> resync also empty.
    policy = _EmptyOnCallsPolicy(sim, empty_calls={2, 3, 4, 5, 6}, infer_sleep=0.005)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    result = PolicyRunner(sim).run(
        "arm",
        policy,
        duration=16 / 50.0,
        control_frequency=50.0,
        action_horizon=_CHUNK,
        fast_mode=True,
        async_rtc=True,
    )
    assert result["status"] == "error", result
    assert "empty action chunk" in result["content"][0]["text"]
    # The error result still carries the telemetry block.
    assert _RTC_KEYS <= set(result["content"][1]["json"])


# --- hard inference timeout -----------------------------------------------


def test_async_rtc_prefetch_timeout_errors_cleanly() -> None:
    """A stuck prefetch hits the hard timeout and returns a structured error
    instead of hanging the sim indefinitely."""
    infer = 0.15
    n_steps = 64  # many chunks remain; the timeout must NOT wait for all of them
    sim = _CountingSim(exec_sleep=0.0)
    policy = _ChunkPolicy(sim, infer_sleep=infer)
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    t0 = time.perf_counter()
    result = PolicyRunner(sim).run(
        "arm",
        policy,
        duration=n_steps / 50.0,
        control_frequency=50.0,
        action_horizon=_CHUNK,
        fast_mode=True,
        async_rtc=True,
        rtc_inference_timeout_s=0.02,
    )
    elapsed = time.perf_counter() - t0
    assert result["status"] == "error", result
    assert "rtc_inference_timeout_s" in result["content"][0]["text"]
    # The rollout aborts after the cold-start query plus the single in-flight
    # inference the executor joins on shutdown - bounded by ~2 inferences, NOT
    # the full 16-chunk rollout it would otherwise run.
    assert elapsed < 4 * infer, elapsed
    assert _RTC_KEYS <= set(result["content"][1]["json"])
