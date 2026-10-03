"""Multi-episode policy rollout - Strands Agent ``@tool`` wrapper.

Closes the fabrication vector identified in
`strands-labs/robots#708 <https://github.com/strands-labs/robots/issues/708>`_:

The existing ``Robot``/``Simulation`` AgentTool surface exposes a
**single-rollout** ``run_policy`` action (one ``duration``/``n_steps`` call
=> one trajectory). When a human asks an LLM agent to "run 20 episodes of
60 steps each", the LLM has no ``n_episodes`` knob to turn - it must
improvise. The historical failure mode (audited across 47 molmoact-e2e
runs, 16 falsely marked OK) is that the LLM dispatches **one** giant
``run_policy`` call, then **narrates** "20/20 episodes complete" - the
recorder sees one mega-episode of ``20x60=1200`` frames and writes
``info.json:total_episodes=1``.

The *recorder* side is already handled: per-episode ``save_episode``
boundaries are wired in ``PolicyRunner.evaluate`` / ``_evaluate_with_spec``.
This tool fixes the *exposure* side: it surfaces ``n_episodes`` explicitly and
drives the episode loop in deterministic Python - no LLM in the loop.

The tool also returns **parquet-truth**, not agent self-report: after the
final ``stop_recording`` it reads ``meta/info.json:total_episodes`` from
the dataset on disk and surfaces that count in the returned payload, so a
downstream verifier comparing "requested vs actual" catches any silent
collapse before status=OK is reported.

Design notes (the contract this tool pins):

* ``simulation`` is a **Python handle**, not an LLM-supplied string. Pass
  the live ``Simulation`` (or ``Robot``-compatible engine) constructed by
  the orchestrator. LLMs cannot synthesize this argument, by design - the
  tool is meant to be invoked from a deterministic outer loop in a
  scripted runner (the pattern voted in HB#349). Mesh-clients drive it
  through normal Python wiring.
* ``n_episodes`` is a required, validated integer. There is no fallback,
  no "infer from duration", no per-episode self-report - the loop iterates
  exactly ``n_episodes`` times, and the parquet-truth gate at the end
  catches any divergence.
* The episode loop calls ``simulation.run_policy(...)`` per iteration and
  invokes the ``PolicyRunner._finalize_recorder_episode`` helper between
  rollouts so each episode lands in its own parquet
  row. The trailing ``stop_recording`` flushes the final episode and
  closes the dataset. A boundary that FAILS stops the loop and is reported
  as ``recording_save_error``: the recorder closes itself on a failed flush,
  so the remaining episodes would record nothing and count no drops, and a
  count over them would be the very self-report this tool exists to replace.
* Recording is OPTIONAL. When ``dataset_root`` is provided we drive a full
  ``start_recording`` -> ``stop_recording`` cycle and report parquet-truth
  (``total_episodes``, ``total_frames``). When ``dataset_root`` is omitted
  we still run the N-episode loop but skip recording - useful for smoke
  tests where the goal is just to exercise the policy.

See ``strands-labs/robots#708`` for the full root-cause analysis and the
end-to-end agent-test fix history (HB#352 -> #716 -> this tool).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from strands.tools.decorator import tool

logger = logging.getLogger(__name__)


def _err(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": text}]}


#: Reported for a header the file declares no usable count under - an absent
#: key, or a declaration outside the count domain. ``declared_count`` refuses
#: every negative, so no real declaration can collide with this sentinel, and
#: the gate below already initialises both totals to it for the case where
#: there is no dataset to read at all.
_NO_COUNT = -1


def _read_parquet_truth(dataset_root: str | Path) -> dict[str, Any]:
    """Read ground-truth episode/frame counts from ``meta/info.json``.

    ``info.json`` is sync-flushed by LeRobot v3, so it is the authoritative
    source for ``total_episodes`` / ``total_frames`` immediately after
    ``stop_recording`` returns. The episodes/data parquet files are
    async-flushed and can lag (see HB#372 forensics + e2e verifier's
    two-phase wait pattern), so we explicitly DO NOT depend on them here.

    Both headers are graded by :func:`~strands_robots.utils.declared_count`, the
    one owner every reader of this file shares, rather than coerced with
    ``int()``. Coercing was destructive in both directions at the one surface
    that exists to catch a fabricated episode count:

    * ``2.5`` truncated to ``2`` and ``true`` counted as one episode, so a
      header no writer could have produced was reported to the caller AS parquet
      truth - and agreed with the requested count, which is the silent collapse
      this gate was written to catch. The same header is "metadata is corrupt"
      to :func:`~strands_robots.verify_dataset.verify_dataset` and to
      :func:`~strands_robots.dataset_metadata.read_dataset_episode_indices`.
    * ``1e400`` (a well-formed JSON number ``json.load`` parses to ``inf``),
      ``NaN`` and ``null`` raised ``OverflowError`` / ``ValueError`` /
      ``TypeError`` out of this function and past the tool envelope, from a file
      that was perfectly readable.

    A file that cannot be READ is reported the same way, and "read" is graded by
    ``ValueError`` rather than by ``json.JSONDecodeError`` alone: bytes the
    declared encoding does not describe raise ``UnicodeDecodeError`` and a number
    longer than ``sys.get_int_max_str_digits`` raises a plain ``ValueError``,
    neither of which is a ``json.JSONDecodeError``. Both escaped past the tool
    envelope from the gate that exists to answer whether the recording produced
    the episodes it was asked for - the same escape the graded headers above
    closed for a file that was perfectly readable.

    A header that declares something which is not a count is a third outcome,
    distinct from both a usable count and an absent header, so it is reported in
    ``info_problems`` - the spelling both other readers of this file already use
    - rather than collapsing into either. The count itself then reads
    :data:`_NO_COUNT`.

    Args:
        dataset_root: Dataset root holding ``meta/info.json``.

    Returns:
        ``info_present`` plus, when the header was readable, the two graded
        counts, any ``info_problems`` naming a declaration that is not a count,
        and the declared ``fps``.

    Returns a partial result on missing fields rather than raising, so the
    caller can surface a structured error instead of a stack trace.
    """
    # Lazy, like every other strands_robots import in this module: a caller who
    # runs without recording pulls in no extra module.
    from strands_robots.utils import declared_count

    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.is_file():
        return {"info_present": False, "info_path": str(info_path)}
    try:
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError) as e:
        return {"info_present": False, "info_path": str(info_path), "error": repr(e)}
    if not isinstance(info, dict):
        # A readable JSON document that is not an object carries no headers at
        # all - the same "nothing to verify against" as an unreadable file.
        # Reading it as one raised AttributeError past the tool envelope.
        return {
            "info_present": False,
            "info_path": str(info_path),
            "error": f"meta/info.json holds a JSON {type(info).__name__}, not an object",
        }

    counts: dict[str, int] = {}
    info_problems: list[str] = []
    for key in ("total_episodes", "total_frames"):
        raw = info.get(key)
        declared = declared_count(raw)
        counts[key] = _NO_COUNT if declared is None else declared
        if key in info and declared is None:
            info_problems.append(f"meta/info.json {key}={raw!r} is not a count - metadata is corrupt")
    return {
        "info_present": True,
        "info_path": str(info_path),
        "total_episodes": counts["total_episodes"],
        "total_frames": counts["total_frames"],
        "info_problems": info_problems,
        "fps": info.get("fps"),
    }


@tool
def run_policy(
    simulation: Any,
    *,
    robot_name: str | None = None,
    policy_provider: str = "mock",
    policy_config: dict[str, Any] | None = None,
    instruction: str = "",
    n_episodes: int = 1,
    n_steps: int = 60,
    control_frequency: float = 30.0,
    action_horizon: int = 8,
    fast_mode: bool = True,
    dataset_root: str | None = None,
    dataset_repo_id: str = "local/run_policy_rollout",
    dataset_task: str = "",
    dataset_fps: int = 30,
    dataset_cameras: list[str] | None = None,
    seed: int | None = None,
    policy_kwargs: dict[str, Any] | None = None,
    video: dict[str, Any] | None = None,
    stop_when: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Roll out a policy for ``n_episodes`` x ``n_steps`` with per-episode parquet boundaries.

    Pass-through wrapper around :meth:`Simulation.run_policy` that owns the
    multi-episode loop and the recording lifecycle, so an LLM agent never
    has to improvise either. Closes the #708 fabrication vector by:

    1. **Explicit ``n_episodes``** - the loop iterates exactly N times,
       no narrated counts.
    2. **Per-episode ``save_episode``** - each rollout lands in its own
       parquet row via ``PolicyRunner._finalize_recorder_episode``.
    3. **Parquet-truth return** - final payload carries
       ``total_episodes`` / ``total_frames`` read from
       ``meta/info.json`` AFTER ``stop_recording`` returns, NOT
       self-reported by the loop. Mismatch with ``n_episodes`` is surfaced
       as ``warnings=[...]`` for the verifier to act on. Both counts are
       graded by ``declared_count``, so a header that declares something
       which is not a count reads ``-1`` and is reported as corrupt
       metadata rather than coerced into agreement with the request.

    Args:
        simulation: Live ``Simulation`` (or compatible) handle.
            Constructed by the orchestrator - pass through a Python
            partial / closure, not from agent text. LLMs cannot
            synthesize this argument, which is the point: the episode
            loop runs in deterministic Python.
        robot_name: Robot to control. Forwarded to ``run_policy``.
            Required when the simulation hosts more than one robot.
        policy_provider: Provider name passed to ``create_policy``
            inside the engine (``"mock"`` / ``"lerobot_local"`` /
            ``"remote"`` / ``"wbc"`` / ...).
        policy_config: Provider-specific kwargs forwarded verbatim.
        instruction: Natural-language instruction for the policy.
        n_episodes: Number of reset -> rollout episodes. MUST be a
            positive int. There is no "guess from duration" fallback.
        n_steps: Hard cap on control steps per episode. Forwarded to
            ``run_policy`` as ``n_steps``.
        control_frequency: Target Hz for policy queries. Must be a finite
            number > 0; an unusable rate is reported before the rollout
            starts instead of aborting every episode mid-flight. When a
            recording is requested it must also EQUAL ``dataset_fps`` - see
            that parameter.
        action_horizon: Lower bound on actions consumed per policy call
            before re-querying; the effective interval is
            ``max(action_horizon, policy.execution_horizon)``, so a
            chunk-emitting policy always consumes its full chunk and a
            smaller value has no effect (see ``resolve_chunk_length``).
            Must be a positive integer, reported before the rollout starts
            for the same reason.
        fast_mode: Skip real-time sleep between steps (default True for
            rollouts - wall-clock pacing slows headless eval). Must be a
            boolean; it selects a posture rather than scaling a quantity, so
            any other type is reported before the rollout starts rather than
            read by truthiness - a truthy ``"false"`` would otherwise run the
            episodes unpaced.
        dataset_root: When set, the tool drives the full recording
            cycle: ``start_recording(root=dataset_root, ...)`` -> N
            rollouts with per-episode save_episode -> ``stop_recording``
            -> parquet-truth read. When ``None`` the loop runs without
            recording (smoke-test mode).
        dataset_repo_id: Forwarded to ``start_recording``.
        dataset_task: Task label forwarded to ``start_recording``.
        dataset_fps: Dataset FPS forwarded to ``start_recording``. Must be a
            positive whole number - reported by ``start_recording`` itself,
            which checks the rate before it touches the target directory, so
            an unusable value costs nothing. It must also EQUAL
            ``control_frequency`` whenever ``dataset_root`` is set: the
            recorder captures one frame per control step and never decimates,
            while LeRobot timestamps every frame from the declared rate, so a
            differing pair cannot be honored, only mislabelled. The
            disagreement is refused up front
            (:func:`~strands_robots.simulation.recording.requested_rate_mismatch_reason`)
            rather than by the per-episode rollout - which this tool reaches
            only after ``start_recording(overwrite=True)`` has replaced any
            dataset already at ``dataset_root`` with an empty one. Ignored
            entirely when ``dataset_root`` is ``None``.
        dataset_cameras: Camera names to record into the dataset.
            When set, forwarded as ``start_recording(cameras=...)``
            (supported by both the MuJoCo and Newton backends) to
            scope a policy-specific dataset to exactly the views the
            policy declares (e.g. ``["camera1", "camera2", "camera3"]``)
            and keep the implicit ``default`` free camera out of
            ``observation.images.*``. When ``None`` (default) no
            ``cameras`` kwarg is forwarded at all, so every scene
            camera is recorded and the call stays backend-agnostic
            across the MuJoCo and Newton engines.
        seed: Master RNG seed. Each episode derives a deterministic
            offset so rollouts are reproducible within a process.
        policy_kwargs: Optional per-call goal payload forwarded to
            every ``policy.get_actions`` call (the #300 goal keys).
        video: Optional rollout-video config forwarded to
            :meth:`Simulation.run_policy` (e.g.
            ``{"path": "/tmp/rollout.mp4", "fps": 30, "camera": "camera1",
            "width": 640, "height": 480}``). ``path`` is required to enable
            recording; a falsy/absent path disables it. For ``n_episodes > 1``
            an ``_ep<i>`` suffix is inserted into the path stem so each
            episode writes its own MP4 instead of overwriting. The returned
            payload carries ``video_paths`` (the MP4s that landed on disk).
        stop_when: Optional semantic early-return clause forwarded to every
            per-episode :meth:`Simulation.run_policy` call: the episode ends
            as soon as the condition holds in the sim, instead of only at the
            ``n_steps`` budget - a per-episode success gate for collection
            loops. Same predicate DSL as a benchmark spec's ``success``
            clause: a single call ``{"predicate": "grasped", "body": "cube",
            "gripper_prefix": "so101/gripper"}`` or an ``{"all": [...]}`` /
            ``{"any": [...]}`` group. Validated against the closed predicate
            registry up front (before any recording is started), so an
            unknown predicate name is rejected with the valid list while
            nothing has been set up yet. Each rollout's ``stopped_reason``
            (``'predicate'`` | ``'budget'`` | ``'cancelled'`` | ``'error'``)
            is reported per episode, as is ``stop_when_true_at_reset``: a
            clause the scene's initial state already satisfies is evaluated
            only AFTER an applied action, so it fires on the episode's first
            step whatever the policy commands - one recorded frame for that
            episode, tagged ``stopped_reason='predicate'`` and
            indistinguishable from an episode that reached the condition.
            Usually a threshold on the wrong side of the initial state (a
            ``body_above_z`` below where the object already rests). The
            payload aggregates ``episodes_stop_when_true_at_reset`` with
            ``stop_when_reset_warning``; it is deliberately not a ``warnings``
            entry, which would flip ``status`` to ``"error"``, because domain
            randomisation legitimately satisfies a clause on some draws.

    Returns:
        Standard ``{status, content}`` payload. On success the payload
        also carries::

            {
                "n_episodes_requested": int,
                "n_episodes_actual": int,      # parquet-truth, -1 if unread
                "n_frames_actual": int,        # parquet-truth, -1 if unread
                "dataset_root": str | None,
                "recording_save_error": str | None,   # None on a healthy run
                "warnings": [str, ...],        # mismatch flags
            "episodes_stop_when_true_at_reset": int,
            "stop_when_reset_warning": str | None,
                "episodes": [
                    {"index": int, "status": "success" | "error", ...},
                    ...
                ],
            }
    """
    # ---- 1. Validation ---------------------------------------------------
    if simulation is None:
        return _err(
            "run_policy: `simulation` is required (pass the live Simulation/Robot "
            "handle from the orchestrator). LLMs cannot synthesize this argument "
            "- that is the point: the episode loop must run in deterministic Python. "
            "See #708 for the fabrication vector this tool closes."
        )

    if not hasattr(simulation, "run_policy"):
        return _err(
            f"run_policy: `simulation` of type {type(simulation).__name__!r} does "
            "not expose .run_policy(). Pass a strands_robots Simulation or "
            "compatible engine."
        )

    if not isinstance(n_episodes, int) or isinstance(n_episodes, bool) or n_episodes < 1:
        return _err(
            f"run_policy: n_episodes must be a positive int, got {n_episodes!r}. "
            "This loop iterates exactly n_episodes times; there is no fallback."
        )

    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
        return _err(f"run_policy: n_steps must be a positive int, got {n_steps!r}.")

    # Same pre-flight reason as the checks below, plus the one that makes it
    # load-bearing rather than merely tidy: step 2 starts its recording with
    # ``overwrite=True``, so a rollout knob the facade refuses is discovered
    # inside the episode loop - AFTER an existing dataset at ``dataset_root``
    # was removed and replaced with an empty one. Both knobs are forwarded
    # verbatim to every episode's ``Simulation.run_policy`` and neither is
    # superseded by another argument, so both are checked unconditionally on the
    # shared domains the facade itself applies (``_validate_positive_frequency``
    # and ``_validate_action_horizon`` delegate to exactly these two). The
    # message is therefore byte-identical to the one the loop used to report;
    # only its timing changes, from after the destruction to before it.
    # ``strands_robots.utils`` imports nothing from the package, so this keeps
    # the module's lazy-import convention without pulling in the sim stack.
    from strands_robots.utils import positive_count_error, positive_finite_number_error

    if freq_error := positive_finite_number_error(control_frequency, "control_frequency", "run_policy"):
        return _err(freq_error)

    if horizon_error := positive_count_error(action_horizon, "action_horizon", "run_policy"):
        return _err(horizon_error)

    # The one rule between two of this tool's own parameters, and the only one
    # whose refusal no downstream guard can reach in time. Each rate is already
    # checked on its own domain - ``control_frequency`` just above,
    # ``dataset_fps`` by ``start_recording`` ahead of the target it resolves -
    # but the recorder is driven once per control step with no decimation and
    # LeRobot timestamps every frame from the declared rate, so the two must be
    # EQUAL. That equality is enforced by the rollout entry point, which this
    # tool reaches only inside the episode loop: measured against an existing
    # dataset of one episode / five frames, ``dataset_fps=30`` with
    # ``control_frequency=50.0`` wiped it to ``total_episodes=0,
    # total_frames=0`` and then reported ``0/2 episodes ok``. Gated on a
    # requested recording because ``dataset_fps`` is forwarded nowhere without
    # one, so a recording-less rollout is unaffected at any rate.
    if dataset_root is not None:
        from strands_robots.simulation.recording import requested_rate_mismatch_reason

        if rate_error := requested_rate_mismatch_reason(
            "run_policy", dataset_fps, control_frequency, fps_param="dataset_fps"
        ):
            return _err(rate_error)

    # Same pre-flight reason as the schema checks below: the per-episode seed is
    # derived arithmetically (``seed + ep``) OUTSIDE the per-episode try, so a
    # non-numeric seed raised out of this tool entirely - past the structured
    # result an agent reads - and a float/negative one reached NumPy inside the
    # loop after step 2 had already created a dataset. Shares the domain every
    # rollout surface uses, so a seed this loop accepts is one the facade it
    # forwards to can apply.
    # ``None`` is the documented "draw fresh entropy" spelling, so there is
    # nothing to check for it - and this module keeps every strands_robots import
    # lazy, so a caller who supplies no seed pulls in no extra module.
    if seed is not None:
        from strands_robots.simulation.base import MAX_EVAL_SEED, randomization_seed_error

        if seed_error := randomization_seed_error(seed, "run_policy", max_seed=MAX_EVAL_SEED):
            return _err(seed_error)

        # The value each episode applies is ``seed + ep``, not ``seed``, so a
        # seed inside the ceiling can still derive one above it - the same
        # accepted-but-unappliable gap, one derivation further on. The check
        # above has already established that ``seed`` is an integer, and
        # ``n_episodes`` is validated further up, so this is computable here.
        if randomization_seed_error(seed + n_episodes - 1, "run_policy", max_seed=MAX_EVAL_SEED):
            return _err(
                f"run_policy: episode seeds are derived as seed + episode index, so seed={seed!r} "
                f"with n_episodes={n_episodes} reaches {seed + n_episodes - 1} on the last episode - "
                f"above the {MAX_EVAL_SEED} ceiling every rollout surface accepts. Lower the seed or "
                "the episode count."
            )

    # The pacing posture is checked before step 2 for the same reason as the
    # options around it: the facade refuses a non-boolean fast_mode itself, but
    # only once this tool has already created the dataset it was asked to
    # record into, so every episode would then be reported as refused beside
    # an empty dataset. Read by truthiness, as it was, ``fast_mode="false"`` ran
    # every episode unpaced and ``fast_mode=0`` paced them, neither a declared
    # spelling. The domain is the shared one, so a spelling the facade refuses
    # is refused here with the same words.
    from strands_robots.utils import boolean_flag_error

    if flag_error := boolean_flag_error(fast_mode, "fast_mode", "run_policy"):
        return _err(flag_error)

    if video is not None and not isinstance(video, dict):
        return _err(
            "run_policy: video must be a dict (VideoConfig kwargs) or None, "
            f"got {type(video).__name__!r}. Example: "
            "video={'path': '/tmp/rollout.mp4', 'fps': 30, 'camera': 'camera1'}."
        )

    if video is not None:
        # Check the video schema here, not on the first forwarded run_policy
        # call: this tool may start a dataset recording (below) before the
        # episode loop, so a mistyped key must be rejected while nothing has
        # been set up yet. Imported lazily (like PolicyRunner in
        # _finalize_episode) and only when recording options were actually
        # supplied, so a rollout without video never touches the sim stack.
        from strands_robots.simulation.policy_runner import VideoConfig

        if video_error := VideoConfig.validation_error(video):
            return _err(f"run_policy: {video_error}")

    # Same reason, for the provider keyword bags: a non-mapping policy_config /
    # policy_kwargs must be rejected before step 2 creates a dataset and starts
    # recording, otherwise the tool leaves an empty dataset behind and reports
    # every episode as raised. Imported here rather than at module scope to
    # match the lazy-import convention the rest of this tool follows.
    from strands_robots.policies import policy_mapping_error

    for _param, _value in (("policy_config", policy_config), ("policy_kwargs", policy_kwargs)):
        if mapping_error := policy_mapping_error(_value, _param):
            return _err(f"run_policy: {mapping_error}")

    # Same reason again for the early-return clause: compile it against the
    # closed predicate registry BEFORE step 2 starts a recording, so an
    # unknown predicate name / bad kwargs is rejected while nothing has been
    # set up yet (instead of leaving an empty dataset behind and reporting
    # every episode as errored). The compiled callable is discarded - the
    # facade recompiles per call - because the dict form is what the
    # per-episode run_policy forwarding below passes through.
    if stop_when is not None:
        from strands_robots.simulation.benchmark_spec import compile_stop_when

        try:
            compile_stop_when(stop_when)
        except ValueError as e:
            return _err(f"run_policy: {e}")

    # The one input every check above leaves unjudged is the one that names
    # what will run: the policy. Its provider is resolved, its preflight run and
    # its constructor called inside every episode's ``Simulation.run_policy``,
    # so an unknown provider name, a camera the provider cannot route or a
    # checkpoint id that does not exist was discovered inside the loop - AFTER
    # step 2 had replaced the dataset at ``dataset_root`` with an empty one.
    # Measured against two recorded episodes of five frames: an unknown
    # provider and a missing ``lerobot_local`` checkpoint each left
    # ``total_episodes=0`` behind and reported ``0/1 episodes ok``. When a
    # recording is requested the policy is therefore judged and built HERE,
    # before anything touches disk, and the one built policy is handed to every
    # episode (``policy_object=``), which also pays a checkpoint load once per
    # call instead of once per episode. Without a recording nothing is at
    # stake on disk and the facade keeps reporting per episode as before.
    forwarded_policy: dict[str, Any] = {}
    if dataset_root is not None:
        built = _build_policy_before_recording(simulation, robot_name, policy_provider, policy_config, dataset_root)
        if isinstance(built, dict):
            return built
        forwarded_policy["policy_object"] = built

    # ---- 2. Optional: start recording -----------------------------------
    recording_started = False
    if dataset_root is not None:
        if not hasattr(simulation, "start_recording"):
            return _err(
                "run_policy: dataset_root requested but simulation does not "
                "expose .start_recording(). Install the [lerobot] extra or pass "
                "dataset_root=None for a recording-less rollout."
            )
        # Only forward ``cameras`` when the caller asked for a subset. Both
        # the MuJoCo and Newton backends' ``start_recording`` accept a
        # ``cameras=`` scope, so this tool stays backend-agnostic. Gating on
        # ``is not None`` keeps the default record-all path byte-for-byte
        # identical to pre-feature behaviour on every backend (no redundant
        # ``cameras=None`` forwarded).
        start_kwargs: dict[str, Any] = dict(
            repo_id=dataset_repo_id,
            task=dataset_task,
            fps=dataset_fps,
            root=dataset_root,
            overwrite=True,
        )
        if dataset_cameras is not None:
            start_kwargs["cameras"] = dataset_cameras
        start_result = simulation.start_recording(**start_kwargs)
        if start_result.get("status") != "success":
            # Surface the engine's own error verbatim - it already explains
            # missing extras, world not loaded, etc.
            return start_result
        recording_started = True

    # ---- 3. Episode loop ------------------------------------------------
    episodes: list[dict[str, Any]] = []
    requested_video_paths: list[str] = []
    recording_save_error: str | None = None
    try:
        for ep in range(n_episodes):
            ep_seed = None if seed is None else seed + ep
            if ep > 0 and "policy_object" in forwarded_policy:
                # The one built policy serves every episode, and ``PolicyRunner.run``
                # resets it only when a seed was given: unseeded, a history-keeping
                # provider (flux3_action, groot, any RTC policy) would condition
                # episode N+1 on episode N's frames while the scene has jumped
                # back to rest, and the drifted actions are what the dataset keeps.
                # Mirrors the facade's between-episode reset (SimEngine.run_policy);
                # best-effort like every other reset call site.
                try:
                    forwarded_policy["policy_object"].reset(seed=ep_seed)
                except Exception as e:  # noqa: BLE001 - reset is best-effort
                    logger.warning(
                        "policy.reset(seed=%s) raised %s before episode %d; continuing without per-episode policy reset",
                        ep_seed,
                        e,
                        ep + 1,
                    )
            try:
                ep_video, ep_video_path = _episode_video_config(video, ep, n_episodes)
                if ep_video_path:
                    requested_video_paths.append(ep_video_path)
                rollout = simulation.run_policy(
                    robot_name=robot_name,
                    policy_provider=policy_provider,
                    policy_config=policy_config,
                    instruction=instruction,
                    control_frequency=control_frequency,
                    action_horizon=action_horizon,
                    fast_mode=fast_mode,
                    n_steps=n_steps,
                    max_steps=n_steps,
                    policy_kwargs=policy_kwargs,
                    seed=ep_seed,
                    video=ep_video,
                    stop_when=stop_when,
                    **forwarded_policy,
                )
            except Exception as e:  # noqa: BLE001 - per-episode resilience
                logger.exception("Episode %d/%d raised: %s", ep + 1, n_episodes, e)
                rollout = {
                    "status": "error",
                    "content": [{"text": f"Episode {ep + 1} raised: {e!r}"}],
                }

            ep_record = {
                "index": ep,
                "status": rollout.get("status", "error"),
                "text": (rollout.get("content") or [{}])[0].get("text", "")[:500],
            }
            # Surface the rollout's termination attribution so a caller
            # gating episodes on stop_when can see which episodes hit the
            # predicate vs. ran out of budget without re-parsing text.
            rollout_json = next(
                (blk["json"] for blk in rollout.get("content") or [] if isinstance(blk, dict) and "json" in blk),
                None,
            )
            if isinstance(rollout_json, dict) and "stopped_reason" in rollout_json:
                ep_record["stopped_reason"] = rollout_json["stopped_reason"]
                ep_record["steps_used"] = rollout_json.get("steps_used")
                # A clause that already held at this episode's reset makes the
                # episode end after one step with stopped_reason="predicate" -
                # a one-frame episode tagged as having reached the condition.
                # Carried through per episode because a randomised initial state
                # makes it an EPISODE-level fact, not a run-level one.
                ep_record["stop_when_true_at_reset"] = bool(rollout_json.get("stop_when_true_at_reset", False))
                ep_record["stop_when_reset_warning"] = rollout_json.get("stop_when_reset_warning")
            episodes.append(ep_record)

            # Per-episode parquet boundary. This helper is wired inside
            # PolicyRunner.evaluate() / _evaluate_with_spec(), but bare
            # run_policy does NOT call it (single-rollout APIs assume the
            # caller owns episode framing). We do it here because we ARE the
            # caller.
            #
            # The helper lives on PolicyRunner. We get the active runner
            # via the simulation's policy-thread bookkeeping OR construct a
            # lightweight wrapper that reads the recorder out of
            # ``sim._world._backend_state``. The simplest path that respects
            # the existing API surface: build a transient PolicyRunner just
            # for the finalize call. It's stateless w.r.t. the recorder.
            if recording_started:
                recording_save_error = _finalize_episode(simulation)
                if recording_save_error is not None:
                    # This episode's frames did not reach the dataset and the
                    # recorder closed itself, so every later episode would run
                    # into a recorder that refuses its frames -
                    # burning the remaining budget to record nothing. Stop and
                    # report the reason, which is the posture every sibling
                    # flush takes (PolicyRunner.evaluate breaks here too).
                    recording_save_error = f"episode {ep}: {recording_save_error}"
                    logger.error("run_policy: stopping the rollout - %s", recording_save_error)
                    break

    finally:
        # ---- 4. Stop recording (always, on success or failure) ----------
        # idempotent on the simulation side ("Was not recording." path), so
        # safe to call even if the start_recording above failed silently.
        if recording_started and hasattr(simulation, "stop_recording"):
            stop_result = simulation.stop_recording()
            if stop_result.get("status") != "success":
                logger.warning(
                    "run_policy: stop_recording returned non-success: %s",
                    stop_result,
                )

    # ---- 5. Parquet-truth gate -----------------------------------------
    n_actual_eps = _NO_COUNT
    n_actual_frames = _NO_COUNT
    warnings_: list[str] = []
    truth: dict[str, Any] = {}

    # First, so it reads as the cause of the episode-count mismatch below
    # rather than as another symptom of it.
    if recording_save_error is not None:
        warnings_.append(
            f"the recorder could not flush an episode - {recording_save_error}. "
            f"The remaining episodes of {n_episodes} were not run: a closed recorder "
            "drops their frames without counting them."
        )

    if dataset_root is not None:
        truth = _read_parquet_truth(dataset_root)
        if not truth.get("info_present"):
            warnings_.append(
                f"meta/info.json missing under {dataset_root!r} - cannot "
                "verify episode count from parquet truth. "
                f"({truth.get('error', 'no error reported')})"
            )
        else:
            n_actual_eps = truth["total_episodes"]
            n_actual_frames = truth["total_frames"]
            # A header that declares something which is not a count is reported
            # as itself. The guard below can only compare a COUNT, and naming
            # the save_episode boundary for a corrupt header would report the
            # wrong cause - that boundary may have fired for every episode.
            warnings_.extend(truth["info_problems"])
            if n_actual_eps == _NO_COUNT:
                if not truth["info_problems"]:
                    warnings_.append(
                        f"meta/info.json under {dataset_root!r} declares no "
                        "total_episodes - cannot verify the episode count."
                    )
            elif n_actual_eps != n_episodes:
                warnings_.append(
                    f"FABRICATION GUARD: requested {n_episodes} episodes, "
                    f"meta/info.json:total_episodes={n_actual_eps}. "
                    + (
                        # It DID fire, and said why it failed. Reporting it as
                        # absent would send a reader after the tool's wiring
                        # instead of after the reason above.
                        f"The per-episode save_episode boundary failed: {recording_save_error}."
                        if recording_save_error is not None
                        else "The per-episode save_episode boundary did not fire as expected."
                    )
                    + " See #708."
                )

    # ---- 6. Build payload ----------------------------------------------
    n_ok = sum(1 for e in episodes if e["status"] == "success")
    # Episodes whose stop_when clause already held at reset, so they ended after
    # one step whatever the policy commanded. Reported beside the counts rather
    # than appended to ``warnings_``, which flips ``status`` to "error": a
    # randomised initial state legitimately satisfies a clause on some draws, so
    # this qualifies the episodes it names without failing the collection. Same
    # posture as PolicyRunner.evaluate's episodes_successful_at_reset.
    n_reset_true = sum(1 for e in episodes if e.get("stop_when_true_at_reset"))
    # Carried up from the rollout that reported it rather than re-derived here:
    # one source of truth for the text, and no import of the simulation
    # package - which this tool must keep working without.
    stop_when_reset_warning = next(
        (e["stop_when_reset_warning"] for e in episodes if e.get("stop_when_reset_warning")), None
    )
    summary_line = f"run_policy: {n_ok}/{n_episodes} episodes ok" + (
        f" | parquet-truth: total_episodes={n_actual_eps}, total_frames={n_actual_frames}"
        if dataset_root is not None
        else ""
    )
    if n_reset_true:
        summary_line += f" | stop_when_true_at_reset={n_reset_true}/{n_episodes}"
    if warnings_:
        summary_line += f" | warnings={len(warnings_)}"
    # The first failed episode's own words, on the line an agent reads. The
    # per-episode records carry every reason, but a caller that stops at the
    # summary was told only a count and went looking for the wiring.
    first_failed = next((e for e in episodes if e["status"] != "success"), None)
    if first_failed is not None:
        summary_line += f" | first error (episode {first_failed['index'] + 1}): {first_failed['text']}"

    payload = {
        "n_episodes_requested": n_episodes,
        "n_episodes_actual": n_actual_eps,
        "n_frames_actual": n_actual_frames,
        "n_episodes_ok": n_ok,
        "episodes_stop_when_true_at_reset": n_reset_true,
        "stop_when_reset_warning": stop_when_reset_warning,
        "dataset_root": dataset_root,
        "recording_save_error": recording_save_error,
        "warnings": warnings_,
        "episodes": episodes,
    }
    if video is not None:
        # Honest reporting: surface only the rollout MP4s that actually
        # landed on disk, not the paths we asked the facade to write.
        payload["video_paths"] = [pth for pth in requested_video_paths if Path(pth).exists()]
    if truth.get("info_path"):
        payload["info_path"] = truth["info_path"]

    out_status = "success" if (n_ok == n_episodes and not warnings_) else "error"
    return {
        "status": out_status,
        "content": [{"text": summary_line}, {"json": payload}],
    }


def _build_policy_before_recording(
    simulation: Any,
    robot_name: str | None,
    policy_provider: str,
    policy_config: dict[str, Any] | None,
    dataset_root: str,
) -> Any:
    """Judge and build the policy while the dataset at ``dataset_root`` is still whole.

    Three verdicts, in the order the facade itself applies them, each returned
    as the ``status=error`` envelope this tool answers with:

    * the provider name, by :func:`~strands_robots.policies.policy_provider_error`,
      the rule ``create_policy`` resolves with;
    * the provider's own pre-build hook, by
      :func:`~strands_robots.policies.preflight_reason`, fed this simulation's
      observation keys when the simulation can answer them (a stand-in with no
      ``get_observation`` is not refused; the hook is then run on nothing, as
      the facade does when the observation is not yet available);
    * the constructor, by calling :func:`~strands_robots.policies.create_policy`.
      Every raise is reported, not only the ``TypeError`` / ``ValueError`` the
      facade's envelope covers: this is the agent-tool surface, whose episode
      loop already turns any raise into an error record, and a missing
      checkpoint (``FileNotFoundError``), a server nobody listens on
      (``ConnectionError``) or a checkpoint directory without its ONNX
      (``RuntimeError``) are the measured shapes.

    Args:
        simulation: The live simulation handle the tool was given.
        robot_name: As passed to the tool; ``None`` lets the simulation pick its
            single robot, as ``run_policy`` does.
        policy_provider: As passed to the tool.
        policy_config: As passed to the tool, already known to be a mapping.
        dataset_root: Named in every refusal, so the caller knows what was kept.

    Returns:
        The built :class:`~strands_robots.policies.base.Policy`, or the error
        envelope when the policy cannot be built - in which case
        ``start_recording`` is never reached and the dataset is untouched.
    """
    from strands_robots.policies import create_policy, policy_provider_error, preflight_reason

    config = policy_config or {}
    untouched = f"No recording was started and the dataset at {dataset_root!r} is untouched."

    reason = policy_provider_error(policy_provider, **config)
    if reason is not None:
        return _err(f"run_policy: {reason} {untouched}")

    def observation_keys() -> set[str]:
        reader = getattr(simulation, "get_observation", None)
        if reader is None:
            return set()
        obs = reader(robot_name)
        return set(obs) if isinstance(obs, dict) else set()

    reason = preflight_reason(policy_provider, observation_keys, **config)
    if reason is not None:
        return _err(f"run_policy: {reason} {untouched}")

    try:
        return create_policy(policy_provider, **config)
    except Exception as exc:  # noqa: BLE001 - every constructor raise is this tool's to report
        logger.exception("run_policy: policy provider %r could not be built: %s", policy_provider, exc)
        return _err(
            f"run_policy: policy provider {policy_provider!r} refused its configuration, "
            f"so no rollout was started. {exc} {untouched}"
        )


def _episode_video_config(
    video: dict[str, Any] | None, ep: int, n_episodes: int
) -> tuple[dict[str, Any] | None, str | None]:
    """Derive a per-episode ``video`` config for ``Simulation.run_policy``.

    The facade records exactly one MP4 per ``run_policy`` call (at
    ``video["path"]``). This tool drives the episode loop itself, so for a
    multi-episode rollout every call would otherwise overwrite the same file.
    To keep one artifact per episode we insert an ``_ep<i>`` suffix into the
    path stem when ``n_episodes > 1``; a single-episode rollout uses the path
    verbatim so the common case writes exactly the file the caller named.

    Args:
        video: The caller's ``video`` dict (VideoConfig kwargs) or ``None``.
        ep: Zero-based episode index.
        n_episodes: Total episodes in this rollout.

    Returns:
        ``(config, path)`` where ``config`` is the per-episode dict to forward
        and ``path`` is its resolved MP4 path. ``(None, None)`` when video is
        disabled, and ``(config, None)`` when a dict was given without a usable
        ``path`` (the facade treats a falsy path as "recording off").
    """
    if not isinstance(video, dict):
        return None, None
    path = video.get("path")
    if not path:
        # Falsy path -> facade disables recording; forward as-is, track nothing.
        return dict(video), None
    cfg = dict(video)
    if n_episodes > 1:
        # Mirror Simulation._episode_video_config exactly: insert ``_ep{i}``
        # (no zero-padding) before the extension so the tool's per-episode
        # filenames match the facade's own multi-episode naming.
        root, ext = os.path.splitext(str(path))
        cfg["path"] = f"{root}_ep{ep}{ext or '.mp4'}"
    return cfg, cfg["path"]


def _finalize_episode(simulation: Any) -> str | None:
    """Invoke ``PolicyRunner._finalize_recorder_episode`` for ``simulation``.

    This helper is the canonical per-episode boundary on
    ``PolicyRunner`` (it reads the active recorder out of
    ``sim._world._backend_state["dataset_recorder"]`` and calls its
    ``save_episode``). Bare ``run_policy`` does not invoke it - it assumes
    the caller owns episode framing. Since this tool *is* the caller, we
    delegate to the same helper to keep the boundary logic in one place
    and to inherit its tolerance for absent/empty buffers.

    Returns:
        ``None`` when the episode was flushed, when there was nothing to
        flush, or when the boundary could not be attempted at all. The
        helper's reason string when the flush FAILED.

        A failed flush is not tolerable the way an absent boundary is, and
        :meth:`~strands_robots.simulation.policy_runner.PolicyRunner._finalize_recorder_episode`
        reports it by *returning* the reason rather than raising: the recorder
        marks itself closed, so every later episode's frames reach no dataset
        and are not counted as drops either. The reason is handed back for the
        episode loop to stop on, exactly as
        :meth:`~strands_robots.simulation.policy_runner.PolicyRunner.evaluate`
        stops on it. Being unable to *reach* the boundary is different and stays
        tolerated: no episode was flushed, so no recorder is poisoned, and the
        parquet-truth gate below reports the count that results.
    """
    try:
        from strands_robots.simulation.policy_runner import PolicyRunner
    except ImportError:
        logger.debug("PolicyRunner unavailable; skipping per-episode finalize")
        return None

    try:
        runner = PolicyRunner(simulation)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not construct PolicyRunner for finalize: %s", e)
        return None

    try:
        return runner._finalize_recorder_episode()  # noqa: SLF001 - this is the contract surface
    except Exception as e:  # noqa: BLE001 - documented not to raise; tolerated if it does
        logger.warning("Per-episode finalize raised: %s", e)
        return None
