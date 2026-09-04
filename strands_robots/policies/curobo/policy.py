"""CuroboPolicy - in-process collision-aware motion planning via NVIDIA cuRobo.

The policy reads a goal from the well-known ``**kwargs`` keys defined in
issue #300 (``target_pose``, ``target_joints``, ``world_update``), forwards
the request to a :class:`MotionPlanner` instance running in the same
process, caches the resulting collision-free trajectory, and yields
``action_horizon``-sized chunks per ``get_actions`` call so the 50Hz
execution loop in :class:`~strands_robots.robot.Robot` can stream per-step
joint targets without re-planning.

The on-device cuRobo APIs were restructured between the ``0.7.x`` series
and the current ``main`` branch (issue #421). This module targets the
restructured ``main`` API:

* ``curobo.motion_planner.MotionPlanner`` (was ``curobo.wrap.reacher.motion_gen.MotionGen``)
* ``curobo.motion_planner.MotionPlannerCfg`` (was ``MotionGenConfig``)
* ``curobo.types.DeviceCfg`` (was ``curobo.types.base.TensorDeviceType``)
* ``curobo.types.JointState`` (was ``curobo.types.state.JointState``)
* ``curobo.types.Pose`` + ``curobo.types.GoalToolPose`` (planning now needs the latter)

The legacy ``WorldConfig`` is gone; the collision scene flows through
``MotionPlannerCfg.create(scene_model=...)`` at config-build time, and
per-call refreshes go through ``MotionPlanner.update_scene`` (or the
legacy ``update_world`` shim for stub planners in unit tests).

Construction mirrors the other non-VLA providers - no service mode, since
cuRobo is a CUDA library rather than a sidecar:

.. code-block:: python

    from strands_robots.policies import create_policy

    policy = create_policy(
        "curobo",                                  # alias: "cumotion"
        robot_config="franka.yml",
        world_config={"cuboid": {...}},            # fed to scene_model
        action_horizon=16,
    )

    actions = policy.get_actions_sync(
        observation_dict={"observation.state": [0.0] * 7},
        instruction="reach for the red block",     # ignored by planners
        target_pose=[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
    )

The ``nvidia-curobo`` package on PyPI is an unrelated v0.1 squatter - the
real cuRobo is published only as source on GitHub. Users opt in by
installing it from source before constructing this policy::

    git clone https://github.com/NVlabs/curobo.git
    pip install -e ./curobo

The ``[curobo]`` extra in ``pyproject.toml`` is intentionally empty until
cuRobo publishes a real PyPI wheel. The policy module raises a clear
:class:`ImportError` (via :func:`require_optional`) on construction when
the ``curobo`` Python package is missing.

Stub seam: unit tests inject a fake planner via the ``motion_gen=`` kwarg
to avoid touching CUDA. The dispatch in :meth:`_plan_and_cache` prefers the
new ``plan_pose`` / ``plan_js`` entry points when the planner exposes
them, but falls back to the legacy ``plan_single`` / ``plan_single_js``
names so the stub-based test path keeps working without GPU.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from strands_robots.policies._log_safety import sanitize_log_value
from strands_robots.policies.base import Policy, chunk_count_error
from strands_robots.utils import name_list_error, require_optional

logger = logging.getLogger(__name__)


# Joint-name allowlist regex - matches the same pattern used by
# ``mesh.security.validate_command`` and :class:`MoveIt2Policy` for
# ``target_joints`` keys, so a value the mesh accepts can flow
# end-to-end without a second allowlist mismatch.
_JOINT_NAME_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]*\Z"

# Cap on how big a trajectory we will keep cached. cuRobo's interpolated
# plans are typically O(100s) of waypoints; this bound exists to fail
# loudly if a sidecar / config bug returns a multi-megabyte trajectory
# rather than silently consuming RAM.
_MAX_TRAJECTORY_WAYPOINTS = 100_000

# Well-known goal fields (issue #300) an LLM may pack into the natural-language
# instruction as a JSON object. Doubles as the pre-filter for the fallback
# parse: an instruction that mentions neither field cannot carry a goal, so no
# JSON decode is attempted for it.
_GOAL_PAYLOAD_KEYS = ("target_pose", "target_joints")

# Cap on how many ``{`` offsets the fallback parse will try to decode. A failed
# decode attempt costs O(len(instruction)) in the worst case, so an unbounded
# scan over an LLM-authored string is a quadratic-time surface; the cap makes
# the work linear in the instruction length. A goal that is not inside one of
# the first candidate objects is reported as unparseable rather than hunted for.
_MAX_GOAL_PAYLOAD_CANDIDATES = 32


def _opens_a_json_object(text: str, brace: int) -> bool:
    """True when ``text[brace]`` could start a JSON object.

    A JSON object is ``{`` then whitespace then either a ``"``-quoted key or the
    closing ``}``. Anything else (``{gripper}``, ``{{``) is prose and is not
    worth a decode attempt.
    """
    probe = brace + 1
    while probe < len(text) and text[probe] in " \t\r\n":
        probe += 1
    return probe < len(text) and text[probe] in '"}'


def _find_goal_payload(instruction: str) -> dict[str, Any] | None:
    """Return the JSON object embedded in ``instruction`` that carries a goal.

    Walks the ``{`` offsets in the instruction and decodes each one with
    :meth:`json.JSONDecoder.raw_decode`, which ends the object at its own
    closing brace instead of assuming the payload runs to the last ``}`` in the
    string. Instructions that mention no goal field at all are rejected without
    a decode. A decoded object without a top-level goal field is
    skipped whole - only a top-level ``target_pose`` / ``target_joints`` counts
    as a goal, so a payload that merely nests one is not a goal (unchanged).

    Returns ``None`` when no candidate yields such an object; when the
    instruction does mention a goal field, that outcome is logged so a
    malformed payload is not silently indistinguishable from no payload.
    """
    if not any(key in instruction for key in _GOAL_PAYLOAD_KEYS):
        return None
    decoder = json.JSONDecoder()
    idx = instruction.find("{")
    attempts = 0
    while idx != -1 and attempts < _MAX_GOAL_PAYLOAD_CANDIDATES:
        if not _opens_a_json_object(instruction, idx):
            # Prose braces (``{gripper}``, ``{{``) cannot open a JSON object, so
            # they are skipped without spending a decode attempt - the candidate
            # budget is a backstop against a pathological string, not a limit on
            # how much prose may precede the payload.
            idx = instruction.find("{", idx + 1)
            continue
        attempts += 1
        try:
            payload, end = decoder.raw_decode(instruction, idx)
        except ValueError:
            # Not the start of a complete JSON value (prose braces, a typo in
            # the payload). Try the next opening brace.
            idx = instruction.find("{", idx + 1)
            continue
        if isinstance(payload, dict) and any(key in payload for key in _GOAL_PAYLOAD_KEYS):
            return payload
        # A complete value that is not a goal: resume after it rather than
        # descending into the braces it contains.
        idx = instruction.find("{", max(end, idx + 1))
    logger.warning(
        "curobo: the instruction mentions a goal field but no JSON object carrying "
        "a top-level target_pose/target_joints could be decoded from it (%d candidate "
        "offsets tried over %d characters). Pass the goal as target_pose= / "
        "target_joints= kwargs instead of embedding it in the instruction text.",
        attempts,
        len(instruction),
    )
    return None


class CuroboPolicy(Policy):
    """In-process cuRobo ``MotionPlanner`` wrapper.

    The policy is intentionally thin - all motion-planning state lives in
    the :class:`MotionPlanner` instance owned by this object. Because
    cuRobo runs on a CUDA device, this policy is **not** thread-safe across
    processes; callers that want fan-out should construct one
    ``CuroboPolicy`` per worker.

    Args:
        robot_config: Path to (or in-memory dict of) the cuRobo robot
            description YAML. cuRobo ships configs for many arms under
            ``curobo/content/configs/robot/``; for example ``"franka.yml"``
            or ``"ur5e.yml"``. May also be a pre-loaded dict (skips
            disk I/O - useful for tests and embedded deployments).
        world_config: Initial collision scene. Forwarded as the
            ``scene_model=`` kwarg to ``MotionPlannerCfg.create``. cuRobo
            accepts a dict with ``"cuboid"`` / ``"mesh"`` / ``"sphere"`` /
            ``"capsule"`` keys whose values are mappings from name to
            geometry params, or ``None`` for free-space planning.
            Per-call overrides flow through the ``world_update`` kwarg on
            :meth:`get_actions` and are forwarded to
            ``MotionPlanner.update_scene``.
        action_horizon: Number of waypoints to yield per call to
            :meth:`get_actions`. Matches the chunked-action contract used
            by the 50Hz execution loop in :class:`~strands_robots.robot.Robot`.
            Default 16 - same as :class:`~strands_robots.policies.groot.policy.Gr00tPolicy`'s
            inner-loop horizon. Must be a positive ``int``: it is consumed as
            a slice bound over the cached trajectory, and it shares
            :func:`~strands_robots.policies.base.chunk_count_error` with the
            ``actions_per_step`` of every other provider, so the same chunk
            count cannot be refused by one policy and accepted by another.
        device_cfg: Optional cuRobo ``DeviceCfg`` controlling the device
            (e.g. ``torch.device('cuda:0')``) and dtype. When omitted, a
            ``DeviceCfg`` is constructed for ``cuda:0`` if CUDA is
            available, else ``cpu``. Passing a string (``"cuda:0"`` /
            ``"cpu"``) is also accepted; it is converted internally.
            Accepted under the legacy alias ``tensor_args=`` for
            backwards compatibility with code written against the
            ``0.7.x`` API; both kwargs map to the same parameter.
        motion_planner_kwargs: Optional extra kwargs forwarded to
            ``MotionPlannerCfg.create`` - e.g.
            ``{"interpolation_dt": 0.02, "use_cuda_graph": False}``.
            Reserved for advanced tuning; defaults are sensible.
            Accepted under the legacy alias ``motion_gen_kwargs=``.
        motion_gen: Pre-built ``MotionPlanner`` instance (kwarg name kept
            for backwards-compatibility with the ``0.7.x`` test-seam
            contract). When supplied, the policy skips its own
            ``MotionPlannerCfg.create`` + ``MotionPlanner(...)``
            construction. This is the seam unit tests use to inject a
            stub planner without a CUDA device. Production callers should
            leave this ``None`` and pass ``robot_config``.
        warmup: When ``True`` (default), call ``MotionPlanner.warmup()``
            after construction so the first ``get_actions`` call is not
            paying JIT-compile cost. Set ``False`` only for tests where
            warmup is expensive or undesirable.
        **kwargs: Forward-compatibility absorber for the smart-string
            resolution path. Per the #300 contract, providers MUST ignore
            unknown kwargs rather than raising.

    Raises:
        ImportError: If ``[curobo]`` extra is not installed and no
            pre-built ``motion_gen`` is supplied.
        ValueError: If ``action_horizon`` is not a positive ``int``, or both
            ``robot_config`` and ``motion_gen`` are missing.

    Examples:
        Direct construction::

            from strands_robots.policies.curobo import CuroboPolicy

            policy = CuroboPolicy(
                robot_config="franka.yml",
                action_horizon=16,
            )

        Via the registry::

            from strands_robots.policies import create_policy

            policy = create_policy("curobo", robot_config="franka.yml")
            policy = create_policy("cumotion", robot_config="franka.yml")  # alias

        Per-call goal::

            actions = policy.get_actions_sync(
                observation_dict={"observation.state": [0.0] * 7},
                instruction="",                               # unused
                target_pose=[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            )
    """

    def __init__(
        self,
        robot_config: str | dict[str, Any] | None = None,
        world_config: dict[str, Any] | None = None,
        action_horizon: int = 16,
        device_cfg: Any = None,
        motion_planner_kwargs: dict[str, Any] | None = None,
        motion_gen: Any = None,
        warmup: bool = True,
        # Legacy aliases preserved from the 0.7.x API surface so callers
        # that pass ``tensor_args=`` / ``motion_gen_kwargs=`` keep working
        # through the migration. Only one of each pair may be supplied.
        tensor_args: Any = None,
        motion_gen_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # ``action_horizon`` is the slice width ``_next_chunk`` takes out of the
        # cached trajectory, which is the same quantity every other provider
        # calls ``actions_per_step`` - so it shares their domain rather than
        # carrying one of its own. A bare ``action_horizon < 1`` admitted
        # everything that comparison happens to answer False for: ``True``
        # became a silent horizon of 1 while ``False`` was refused, and ``2.7``
        # was truncated to 2 by an ``int()`` coercion on the way to the
        # attribute - which this makes dead, since only an ``int`` now reaches
        # it. Everything the comparison cannot answer at all escaped this
        # constructor as whatever ``<`` or that ``int()`` raised: ``TypeError``
        # for a numeric string, ``None`` or a list, ``OverflowError`` for
        # ``inf``. None of those is the ``ValueError`` the ``Raises`` section
        # promises, and none of them names the parameter.
        if error := chunk_count_error(action_horizon, "action_horizon", "curobo"):
            raise ValueError(error)

        # Reconcile the legacy aliases. Supplying both forms is an error
        # rather than a silent precedence rule - the user almost certainly
        # has a typo or a stale call site.
        if device_cfg is not None and tensor_args is not None:
            raise ValueError(
                "CuroboPolicy: pass exactly one of device_cfg= (preferred) or the legacy alias tensor_args=, not both."
            )
        if motion_planner_kwargs is not None and motion_gen_kwargs is not None:
            raise ValueError(
                "CuroboPolicy: pass exactly one of motion_planner_kwargs= "
                "(preferred) or the legacy alias motion_gen_kwargs=, not both."
            )
        resolved_device_cfg = device_cfg if device_cfg is not None else tensor_args
        resolved_planner_kwargs = motion_planner_kwargs if motion_planner_kwargs is not None else motion_gen_kwargs

        self.robot_config = robot_config
        self.world_config = world_config
        self.action_horizon = action_horizon
        self._motion_planner_kwargs = dict(resolved_planner_kwargs or {})

        # State for trajectory chunking: cache the full plan, yield
        # ``action_horizon`` rows per call until exhausted, then re-plan
        # on the next call.
        self._robot_state_keys: list[str] = []
        self._cached_trajectory: list[list[float]] = []
        self._cached_cursor: int = 0

        # When the caller supplies a pre-built ``motion_gen`` (e.g. from
        # tests), use it directly. Otherwise build one from the cuRobo
        # APIs. The lazy import lives behind ``require_optional`` so the
        # error message points at the ``[curobo]`` extra cleanly. The
        # constructor kwarg keeps the ``motion_gen=`` name for
        # backwards-compatibility; internally we store it as
        # ``self._motion_planner`` to match the new cuRobo type name.
        if motion_gen is not None:
            self._motion_planner = motion_gen
        else:
            if robot_config is None:
                raise ValueError(
                    "CuroboPolicy requires either ``robot_config`` (path or dict) "
                    "or a pre-built ``motion_gen`` instance. Pass robot_config="
                    "'franka.yml' to load one of the cuRobo built-in configs."
                )
            self._motion_planner = self._build_motion_gen(
                robot_config=robot_config,
                world_config=world_config,
                device_cfg=resolved_device_cfg,
            )
            if warmup:
                self._safe_warmup()

        # Per the #300 contract: silently ignore unknown kwargs. The
        # smart-string resolver and ``register_policy`` may fan extra
        # kwargs through this constructor; the policy only consumes a
        # short, documented set.
        if kwargs:
            logger.debug(
                "CuroboPolicy ignoring unknown constructor kwargs: %s",
                sorted(kwargs.keys()),
            )

        logger.info(
            "CuroboPolicy ready [robot_config=%r action_horizon=%d]",
            robot_config if isinstance(robot_config, str) else "<dict>",
            self.action_horizon,
        )

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Registry key for this provider (``"curobo"``)."""
        return "curobo"

    @property
    def requires_images(self) -> bool:
        """cuRobo plans from joint state + collision world, never images.

        Returning ``False`` lets the simulation skip camera rendering for
        this provider - same throughput optimisation
        :class:`~strands_robots.policies.moveit2.MoveIt2Policy` and
        :class:`~strands_robots.policies.mock.MockPolicy` expose.
        """
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Configure the joint names this policy emits actions for.

        Used to map the per-row joint values cuRobo returns onto per-joint
        action dicts. When unset, ``get_actions`` falls back to
        ``observation.state`` length and emits ``"joint_<i>"`` keys
        (consistent with :class:`MockPolicy` / :class:`MoveIt2Policy`).

        Raises:
            ValueError: If ``robot_state_keys`` is not an ordered list of
                distinct non-blank names, per
                :func:`~strands_robots.utils.name_list_error`. A single name
                passed as a bare string is the mistake this catches: ``str`` is
                iterable per character, so it would bind one joint per letter.
        """
        if robot_state_keys and (
            error := name_list_error(robot_state_keys, "robot_state_keys", "set_robot_state_keys")
        ):
            raise ValueError(error)
        self._robot_state_keys = list(robot_state_keys)

    def reset(self, seed: int | None = None) -> None:
        """Drop the cached trajectory and reset cuRobo's per-episode state.

        cuRobo's ``MotionPlanner.reset()`` clears any seed / partial-plan
        state held by the planner. The ``seed`` argument is not forwarded
        to cuRobo (its trajopt RNG is configured at construction time);
        it is accepted for API parity with the rest of the non-VLA family.

        Best-effort - any failure (planner doesn't expose ``reset``,
        endpoint raises) is logged and swallowed. Eval correctness is
        preserved even when reset is a no-op (the next ``get_actions``
        call re-plans from the current observation).
        """
        # Always clear the cached trajectory so the next ``get_actions``
        # re-plans from the (likely-different) starting state.
        self._cached_trajectory = []
        self._cached_cursor = 0

        # Forward to ``MotionPlanner.reset`` if available. Older / stub
        # planners may not expose it; that's fine.
        reset_fn = getattr(self._motion_planner, "reset", None)
        if reset_fn is None:
            logger.debug("CuroboPolicy.reset: motion_planner has no reset(); cleared cache only")
            return
        try:
            reset_fn()
            logger.debug("CuroboPolicy.reset: forwarded to motion_planner (seed=%r)", seed)
        except Exception as e:  # noqa: BLE001 - reset is best-effort
            logger.info(
                "CuroboPolicy.reset: motion_planner.reset() raised (seed=%r): %s; "
                "continuing without per-episode planner-side reset",
                seed,
                e,
            )

    async def get_actions(
        self,
        observation_dict: dict[str, Any],
        instruction: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Plan a collision-free trajectory and yield ``action_horizon`` chunks.

        On the first call (or after the cached trajectory is exhausted),
        this method:

        1. Reads the goal from ``**kwargs`` (or, as a fallback for
           LLM-driven workflows, parses it out of ``instruction``).
        2. Optionally refreshes the cuRobo collision scene via
           ``world_update`` (forwarded to ``MotionPlanner.update_scene``,
           or the legacy ``update_world`` shim for stub planners).
        3. Builds a :class:`JointState` start configuration from
           ``observation_dict["observation.state"]``.
        4. Calls ``MotionPlanner.plan_pose`` (Cartesian goals) or
           ``plan_js`` (joint-space goals) and unpacks the interpolated
           plan into a list of joint-target rows.
        5. Caches the full trajectory.

        Every call (cache-miss or cache-hit) returns the next
        ``action_horizon`` rows from the cached trajectory as per-step
        action dicts. When the cache empties, the next call re-plans
        from the current observation.

        Args:
            observation_dict: Robot observation. ``observation.state`` is
                used as the start joint configuration. ``observation.velocity``
                is used as the start joint velocity if present, else zeros.
                The natural-language ``instruction`` is forwarded to the
                fallback :meth:`_parse_target` only if no structured
                kwargs are supplied.
            instruction: Natural-language instruction. Used only as a
                fallback parse target for LLM-driven workflows when
                neither ``target_pose`` nor ``target_joints`` is supplied
                via kwargs. Planner providers consume goals through
                structured kwargs; this fallback exists for API parity
                with the LLM-agent demos.
            **kwargs: Well-known goal payload from #300:

                * ``target_pose`` (``list[float]``):
                  ``[x, y, z, qw, qx, qy, qz]`` in the robot base frame.
                * ``target_joints`` (``dict[str, float]``): joint-space
                  goal keyed by joint name (radians / metres).
                * ``world_update`` (``dict | None``): per-call world
                  refresh for collision-aware planning.
                * ``replan`` (``bool``): force a re-plan even if the
                  cache still has waypoints. Default ``False``.

                Unknown kwargs are silently ignored.

        Returns:
            List of action dicts; up to ``action_horizon`` entries per
            call. May be shorter on the final chunk of a trajectory.

        Raises:
            ValueError: If neither structured goal nor a parseable
                ``instruction`` is provided, or if the goal payload is
                malformed.
            RuntimeError: If cuRobo returns ``success=False`` (no
                collision-free path).
        """
        # 1. Pull goals from kwargs (or fall back to the instruction).
        target_pose = kwargs.get("target_pose")
        target_joints = kwargs.get("target_joints")
        world_update = kwargs.get("world_update")
        replan = bool(kwargs.get("replan", False))

        if target_pose is None and target_joints is None:
            target_pose, target_joints = self._parse_target(instruction)

        if target_pose is None and target_joints is None:
            raise ValueError(
                "CuroboPolicy.get_actions requires at least one of "
                "target_pose=[x,y,z,qw,qx,qy,qz] or target_joints={joint:value}. "
                "These are the well-known kwargs from issue #300; the "
                "natural-language `instruction` is parsed as a fallback "
                "only when it contains a JSON object with a 'target_pose' "
                "or 'target_joints' field."
            )

        # 2. Validation - reject malformed goals up-front.
        if target_pose is not None:
            self._validate_target_pose(target_pose)
        if target_joints is not None:
            self._validate_target_joints(target_joints)

        # 3. Cache check - if the previous trajectory still has unyielded
        # rows AND no new goal forces a replan, just stream the next
        # chunk. Otherwise re-plan from the current state.
        if not self._cache_has_waypoints() or replan:
            joint_state = self._extract_joint_state(observation_dict)
            self._plan_and_cache(
                joint_state=joint_state,
                target_pose=target_pose,
                target_joints=target_joints,
                world_update=world_update,
            )

        return self._next_chunk()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_motion_gen(
        self,
        robot_config: str | dict[str, Any],
        world_config: dict[str, Any] | None,
        device_cfg: Any,
    ) -> Any:
        """Construct a ``MotionPlanner`` instance from the cuRobo APIs.

        Targets cuRobo's restructured ``main`` API (issue #421):

        * ``curobo.types.DeviceCfg`` (was ``TensorDeviceType``)
        * ``curobo.motion_planner.MotionPlannerCfg`` (was ``MotionGenConfig``)
        * ``curobo.motion_planner.MotionPlanner`` (was ``MotionGen``)
        * The collision scene flows through ``MotionPlannerCfg.create(scene_model=...)``;
          there is no longer a standalone ``WorldConfig``.

        Lives in its own method so the constructor seam stays clean and
        unit tests can override ``__init__`` paths without touching this
        path. Importing cuRobo is gated by :func:`require_optional` so
        the ``[curobo]`` extra is the actionable error.

        The method name ``_build_motion_gen`` is preserved for parity
        with the legacy 0.7.x test fixtures and external monkeypatches;
        the returned object is now a ``MotionPlanner``.
        """
        require_optional(
            "curobo",
            # cuRobo is NOT on PyPI (the ``nvidia-curobo`` v0.1 package
            # is an unrelated squatter). Real install is from source:
            #   git clone https://github.com/NVlabs/curobo.git
            #   pip install -e ./curobo
            pip_install="-e git+https://github.com/NVlabs/curobo.git#egg=curobo",
            extra="curobo",
            purpose="CuroboPolicy motion planning",
        )
        # Import lazily so module load doesn't pay the CUDA-init cost
        # for users who never construct a CuroboPolicy. The ``main``
        # cuRobo API exposes the high-level types directly under
        # ``curobo.types`` and ``curobo.motion_planner``.
        import torch  # type: ignore[import-not-found]
        from curobo.motion_planner import (  # type: ignore[import-not-found]
            MotionPlanner,
            MotionPlannerCfg,
        )
        from curobo.types import DeviceCfg  # type: ignore[import-not-found]

        resolved_device_cfg = self._resolve_device_cfg(device_cfg, DeviceCfg, torch)

        # ``world_config`` is forwarded as-is to ``scene_model=`` so a
        # ``MotionPlannerCfg`` can resolve the collision geometry. cuRobo
        # accepts the same dict shape it used to accept on
        # ``WorldConfig.from_dict`` (cuboid / mesh / sphere / capsule
        # mappings), so existing callers that built a dict for the
        # 0.7.x API continue to work without modification.
        scene_model = world_config

        # Build kwargs for ``MotionPlannerCfg.create``. ``robot=`` is the
        # canonical kwarg name on the ``main`` API; ``scene_model=`` is
        # only forwarded when the caller supplied one to keep the call
        # signature minimal for the free-space case.
        cfg_kwargs: dict[str, Any] = {
            "robot": robot_config,
            "device_cfg": resolved_device_cfg,
        }
        if scene_model is not None:
            cfg_kwargs["scene_model"] = scene_model
        cfg_kwargs.update(self._motion_planner_kwargs)

        cfg = MotionPlannerCfg.create(**cfg_kwargs)
        return MotionPlanner(cfg)

    @staticmethod
    def _resolve_device_cfg(device_cfg: Any, DeviceCfg: Any, torch: Any) -> Any:
        """Coerce a user-supplied device-config kwarg into a cuRobo ``DeviceCfg``.

        Accepts ``None`` (defaults to cuda:0 if available else cpu), a
        string ``"cuda:0"`` / ``"cpu"`` / ``"cuda"``, a
        ``torch.device`` instance, or a pre-built ``DeviceCfg``. The last
        case is detected duck-typed so we don't import ``DeviceCfg`` at
        the policy module top.
        """
        if device_cfg is None:
            default_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            return DeviceCfg(device=default_device)
        if isinstance(device_cfg, str):
            return DeviceCfg(device=torch.device(device_cfg))
        if isinstance(device_cfg, torch.device):
            return DeviceCfg(device=device_cfg)
        # Either a pre-built DeviceCfg or a legacy TensorDeviceType
        # the user is forwarding from older code. Pass through unchanged
        # and let the planner raise if the type is genuinely wrong - the
        # error message from cuRobo is clearer than anything we'd add.
        return device_cfg

    def _safe_warmup(self) -> None:
        """Call :meth:`MotionPlanner.warmup` if available; log on failure."""
        warmup_fn = getattr(self._motion_planner, "warmup", None)
        if warmup_fn is None:
            return
        try:
            warmup_fn()
        except Exception as e:  # noqa: BLE001 - warmup is best-effort
            logger.warning(
                "CuroboPolicy: motion_planner.warmup() raised (%s); first get_actions call will pay JIT-compile cost",
                e,
            )

    def _cache_has_waypoints(self) -> bool:
        return self._cached_cursor < len(self._cached_trajectory)

    def _next_chunk(self) -> list[dict[str, Any]]:
        """Yield the next ``action_horizon`` rows from the cached trajectory."""
        if not self._cache_has_waypoints():
            return []
        end = min(self._cached_cursor + self.action_horizon, len(self._cached_trajectory))
        rows = self._cached_trajectory[self._cached_cursor : end]
        self._cached_cursor = end
        keys = self._resolve_joint_keys(len(rows[0]) if rows else 0)
        actions: list[dict[str, Any]] = []
        for row in rows:
            actions.append({k: float(v) for k, v in zip(keys, row, strict=False)})
        return actions

    def _plan_and_cache(
        self,
        joint_state: list[float] | None,
        target_pose: list[float] | None,
        target_joints: dict[str, float] | None,
        world_update: dict[str, Any] | None,
    ) -> None:
        """Build the cuRobo request, dispatch the plan call, cache the result.

        Imports cuRobo types lazily so the smoke tests that inject a stub
        planner via ``motion_gen=`` never touch the cuRobo package at all.

        Dispatch order for Cartesian goals:

        1. New API: ``MotionPlanner.plan_pose(GoalToolPose, JointState)``
        2. Legacy / stub API: ``plan_single(JointState, Pose)``

        Dispatch order for joint-space goals:

        1. New API: ``MotionPlanner.plan_js(JointState, JointState)``
        2. Legacy / stub API: ``plan_single_js(JointState, JointState)``
        3. Last resort: ``plan_single(JointState, JointState)`` - a planner
           exposing neither joint-space entry point still plans a joint-space
           goal through its Cartesian-shaped entry point.

        Both joint-space names exist because cuRobo renamed that entry point
        between its legacy and ``main`` surfaces, so a planner is free to
        expose either one and the first it has is the one used.
        """
        # Refresh the collision scene if requested.
        if world_update is not None:
            self._apply_world_update(world_update)

        # Build the start state. cuRobo's ``JointState.from_position``
        # accepts a tensor; we leave the conversion to cuRobo so this
        # path stays small.
        start_state = self._build_start_state(joint_state)

        try:
            if target_pose is not None:
                goal = self._build_goal_pose(target_pose)
                # New API: ``plan_pose(goal, start)`` with the goal
                # built as a ``GoalToolPose`` (5D tensor shape). The
                # legacy ``plan_single(start, goal)`` accepts a flat
                # ``Pose`` and is kept as a fallback so the stub seam
                # in unit tests (``_StubMotionGen.plan_single``) keeps
                # working without any GPU.
                plan_pose_fn = getattr(self._motion_planner, "plan_pose", None)
                if plan_pose_fn is not None:
                    result = plan_pose_fn(goal, start_state)
                else:
                    result = self._motion_planner.plan_single(start_state, goal)
            else:
                # target_joints is non-None here (validated upstream).
                goal_js = self._build_goal_joint_state(target_joints or {})
                # New API: ``plan_js`` is the joint-space planner;
                # legacy / stub API is ``plan_single_js``. Both take
                # ``(start_state, goal_js)``. Fall back to
                # ``plan_single`` only if neither joint-space entry
                # point is exposed.
                plan_js_fn = getattr(self._motion_planner, "plan_js", None) or getattr(
                    self._motion_planner, "plan_single_js", None
                )
                if plan_js_fn is not None:
                    result = plan_js_fn(start_state, goal_js)
                else:
                    result = self._motion_planner.plan_single(start_state, goal_js)
        except Exception as e:
            # Re-raise as RuntimeError with goal context so the runner
            # gets a clear message instead of an opaque cuRobo trace.
            raise RuntimeError(
                f"CuroboPolicy planning failed: target_pose={target_pose!r}, target_joints={target_joints!r}: {e}"
            ) from e

        # cuRobo's ``main`` API returns ``None`` from ``plan_pose`` / ``plan_js``
        # when the planner finds no solution (e.g. an unreachable goal), rather
        # than a result object with ``success=False`` (the legacy 0.7.x shape).
        # Guard both so an unsolved plan surfaces as a clear RuntimeError instead
        # of an opaque "missing get_interpolated_plan" error downstream.
        if result is None:
            raise RuntimeError(
                "CuroboPolicy planning failed: planner returned no solution "
                f"(result is None) for target_pose={target_pose!r}, "
                f"target_joints={target_joints!r}"
            )

        if not getattr(result, "success", True):
            status = getattr(result, "status", "unknown")
            raise RuntimeError(
                f"CuroboPolicy planning failed: status={status!r}, "
                f"target_pose={target_pose!r}, target_joints={target_joints!r}"
            )

        trajectory = self._extract_trajectory(result)
        if len(trajectory) > _MAX_TRAJECTORY_WAYPOINTS:
            raise RuntimeError(
                f"CuroboPolicy got {len(trajectory)} waypoints, exceeds "
                f"{_MAX_TRAJECTORY_WAYPOINTS} guard. Likely a misconfigured "
                "interpolation_dt. Refusing to cache."
            )
        self._cached_trajectory = trajectory
        self._cached_cursor = 0

    def _apply_world_update(self, world_update: dict[str, Any]) -> None:
        """Forward a per-call collision-scene refresh to cuRobo.

        On the cuRobo ``main`` API the scene is owned by the
        ``MotionPlannerCfg`` and refreshed via
        ``MotionPlanner.update_scene(dict)``. We probe ``update_scene``
        first (new API), then ``update_world`` (legacy + stub seam) so a
        stub planner that only exposes ``update_world`` keeps working in
        unit tests.

        The callee gets the raw dict so test stubs can record it
        directly. Real cuRobo accepts the same dict shape it used to
        accept on ``WorldConfig.from_dict``.
        """
        update_fn = getattr(self._motion_planner, "update_scene", None) or getattr(
            self._motion_planner, "update_world", None
        )
        if update_fn is None:
            shown = sorted(world_update.keys()) if isinstance(world_update, dict) else world_update
            logger.warning(
                "CuroboPolicy: motion_planner exposes neither update_scene() "
                "nor update_world(); world_update=%s ignored",
                # %s over the sanitized repr renders what %r rendered, with the
                # observation-derived key names' line breaks escaped.
                sanitize_log_value(repr(shown)),
            )
            return
        update_fn(world_update)

    def _build_start_state(self, joint_state: list[float] | None) -> Any:
        """Build a cuRobo :class:`JointState` from a Python list.

        Targets the ``main`` API: ``curobo.types.JointState`` (was
        ``curobo.types.state.JointState``) and ``curobo.types.DeviceCfg``
        (was ``curobo.types.base.TensorDeviceType``).
        """
        if joint_state is None:
            # Without a start state, defer to whatever the planner has
            # configured (its retract config, typically). Stub planners
            # ignore the start state anyway.
            return None
        try:
            import torch  # type: ignore[import-not-found]
            from curobo.types import DeviceCfg, JointState  # type: ignore[import-not-found]
        except ImportError:
            # Stub-injection path: pass the raw list through. The stub
            # planner is responsible for interpreting it.
            return joint_state

        device, dtype = self._planner_tensor_kwargs(DeviceCfg, torch)
        position = torch.tensor(joint_state, device=device, dtype=dtype).unsqueeze(0)
        return JointState.from_position(position)

    def _build_goal_pose(self, target_pose: list[float]) -> Any:
        """Build a cuRobo ``GoalToolPose`` from ``[x, y, z, qw, qx, qy, qz]``.

        The ``main`` API requires a 5D tensor shape ``[B, H, L, G, 3]``
        (Batch / Horizon / Links / Goalset / 3) for the position and
        ``[B, H, L, G, 4]`` for the quaternion, plus an explicit
        ``tool_frames`` list. We resolve the tool frame from
        ``self._motion_planner.kinematics.tool_frames[0]`` (the canonical
        Thor-validated path); callers that need a non-default tool frame
        can override by setting ``self._motion_planner.kinematics.tool_frames``
        before constructing the policy.

        The stub seam (no cuRobo installed) returns the raw list so unit
        tests can introspect what was forwarded without importing cuRobo.
        """
        try:
            import torch  # type: ignore[import-not-found]
            from curobo.types import DeviceCfg, GoalToolPose  # type: ignore[import-not-found]
        except ImportError:
            # Stub path - pass the raw list through.
            return target_pose

        device, dtype = self._planner_tensor_kwargs(DeviceCfg, torch)

        # 5D shape: [B=1, H=1, L=1, G=1, 3] for position and 4 for quaternion.
        # The single batch / horizon / link / goalset entry corresponds to
        # the canonical "single-arm reach to one Cartesian goal" use case
        # this policy targets. Goal-set / batched planning is a separate
        # follow-up (out-of-scope per issue #421).
        pos = torch.tensor(target_pose[0:3], device=device, dtype=dtype).reshape(1, 1, 1, 1, 3)
        quat = torch.tensor(target_pose[3:7], device=device, dtype=dtype).reshape(1, 1, 1, 1, 4)

        tool_frames = self._resolve_tool_frames()
        return GoalToolPose(
            tool_frames=tool_frames,
            position=pos,
            quaternion=quat,
        )

    def _resolve_tool_frames(self) -> list[Any]:
        """Resolve the ``tool_frames`` argument for ``GoalToolPose``.

        cuRobo's ``MotionPlanner`` exposes ``planner.kinematics.tool_frames``
        on the ``main`` API. We pick the first entry by default - this
        matches the Thor-validated example for single-arm reach. Sites
        that need a non-default tool frame can mutate
        ``self._motion_planner.kinematics.tool_frames`` before
        constructing the policy.
        """
        kin = getattr(self._motion_planner, "kinematics", None)
        if kin is None:
            raise RuntimeError(
                "CuroboPolicy: motion_planner has no .kinematics attribute; "
                "cannot resolve tool_frames for GoalToolPose. This is expected "
                "only for stub planners that bypass _build_goal_pose entirely."
            )
        tool_frames = getattr(kin, "tool_frames", None)
        if not tool_frames:
            raise RuntimeError(
                "CuroboPolicy: motion_planner.kinematics.tool_frames is empty; "
                "cannot construct GoalToolPose. Ensure the robot YAML defines "
                "at least one tool frame."
            )
        return [tool_frames[0]]

    def _build_goal_joint_state(self, target_joints: dict[str, float]) -> Any:
        """Build a cuRobo :class:`JointState` from a name->value dict.

        Targets the ``main`` API: ``curobo.types.JointState`` and
        ``curobo.types.DeviceCfg`` import paths.
        """
        try:
            import torch  # type: ignore[import-not-found]
            from curobo.types import DeviceCfg, JointState  # type: ignore[import-not-found]
        except ImportError:
            return target_joints

        device, dtype = self._planner_tensor_kwargs(DeviceCfg, torch)
        # Order keys deterministically. If ``set_robot_state_keys`` was
        # called we honour that order; otherwise sorted for stability.
        if self._robot_state_keys and set(target_joints).issubset(set(self._robot_state_keys)):
            keys = [k for k in self._robot_state_keys if k in target_joints]
        else:
            keys = sorted(target_joints.keys())
        position = torch.tensor([target_joints[k] for k in keys], device=device, dtype=dtype).unsqueeze(0)
        return JointState.from_position(position, joint_names=keys)

    def _planner_tensor_kwargs(self, DeviceCfg: Any, torch: Any) -> tuple[Any, Any]:
        """Resolve ``(device, dtype)`` for tensor construction.

        Reads them off the planner's stored ``DeviceCfg`` (the canonical
        location on the ``main`` API). Falls back to a fresh
        ``DeviceCfg()`` default when the planner doesn't expose one
        (e.g. a future API rename or an unusual stub).
        """
        # Prefer the planner's resolved device_cfg so tensors land on
        # the same device the planner is bound to.
        dc = (
            getattr(self._motion_planner, "device_cfg", None)
            or getattr(self._motion_planner, "tensor_args", None)
            or DeviceCfg(device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        )
        device = getattr(dc, "device", None)
        dtype = getattr(dc, "dtype", None) or torch.float32
        if device is None:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        return device, dtype

    @staticmethod
    def _extract_trajectory(result: Any) -> list[list[float]]:
        """Pull the joint-position trajectory out of a cuRobo plan result.

        On the cuRobo ``main`` API the result of ``plan_pose`` /
        ``plan_js`` exposes ``get_interpolated_plan()`` returning a
        :class:`JointState` whose ``position`` is a
        ``[batch, horizon, T, ndof]`` tensor. The legacy ``MotionGenResult``
        exposed a 2D ``[T, ndof]`` tensor; this helper collapses any leading
        batch / horizon dimensions (taking the first plan) so callers always
        receive a flat ``[T, ndof]`` list-of-lists regardless of API era.

        Stub planners may emit a plain ``list[list[float]]`` directly via
        ``result.trajectory`` to keep the test seam lightweight.
        """
        # Stub path first - if the result already exposes a list-of-lists
        # at ``trajectory``, prefer it.
        traj = getattr(result, "trajectory", None)
        if isinstance(traj, list):
            return [[float(v) for v in row] for row in traj]

        # Real cuRobo path: ``get_interpolated_plan().position`` is a torch
        # tensor. On the ``main`` API its shape is ``[batch, horizon, T, ndof]``
        # (e.g. ``[1, 1, 61, 9]`` for Franka); the legacy API returned a 2D
        # ``[T, ndof]`` tensor. Either way we want the final ``[T, ndof]`` view.
        get_plan = getattr(result, "get_interpolated_plan", None)
        if get_plan is None:
            raise RuntimeError(
                "CuroboPolicy: planner result is missing both "
                "``trajectory`` (stub path) and ``get_interpolated_plan`` "
                "(real path); cannot extract waypoints"
            )
        plan = get_plan()
        position = getattr(plan, "position", plan)
        # Collapse leading batch / horizon dims (take the first plan) down to a
        # 2D ``[T, ndof]`` tensor. ``ndim`` exists on torch tensors and numpy
        # arrays; stub objects fall through to the list path below.
        ndim = getattr(position, "ndim", None)
        if ndim is not None:
            while position.ndim > 2:
                position = position[0]
        # ``position`` is typically ``torch.Tensor``; ``.cpu().tolist()``
        # produces a list-of-lists. Fall back to ``list(...)`` for stub
        # objects.
        try:
            return [list(map(float, row)) for row in position.cpu().tolist()]
        except AttributeError:
            return [list(map(float, row)) for row in position]

    def _extract_joint_state(self, observation_dict: dict[str, Any]) -> list[float] | None:
        """Pull ``observation.state`` out of the observation dict.

        Accepts list / tuple / numpy array / torch tensor; returns a plain
        Python list of floats so cuRobo's tensor builders get a known
        input shape.
        """
        state = observation_dict.get("observation.state")
        if state is None:
            return None
        try:
            if hasattr(state, "tolist"):
                state = state.tolist()
            return [float(x) for x in state]
        except (TypeError, ValueError) as e:
            logger.warning(
                "CuroboPolicy: failed to extract joint_state from observation.state=%s (%s); "
                "letting planner use its own retract configuration",
                sanitize_log_value(repr(state)),
                sanitize_log_value(e),
            )
            return None

    def _resolve_joint_keys(self, n: int) -> list[str]:
        """Resolve the joint key names for an n-element trajectory row.

        If ``set_robot_state_keys`` was called with a matching length,
        use those names; otherwise fall back to positional ``joint_<i>``
        labels (consistent with :class:`MockPolicy` and :class:`MoveIt2Policy`).
        """
        if self._robot_state_keys and len(self._robot_state_keys) == n:
            return list(self._robot_state_keys)
        return [f"joint_{i}" for i in range(n)]

    @staticmethod
    def _parse_target(instruction: str) -> tuple[list[float] | None, dict[str, float] | None]:
        """Best-effort fallback parse of the natural-language instruction.

        For LLM-driven workflows (``Robot.start_task(..., policy_provider="curobo")``),
        the agent may pack a goal into the instruction string as a JSON
        snippet. This helper extracts ``target_pose`` / ``target_joints``
        from such a payload so the LLM-agent demo path works without
        forcing the agent to learn a new kwargs API.

        Returns ``(None, None)`` when no goal is found - the caller will
        then raise :class:`ValueError`.

        Extraction is delegated to :func:`_find_goal_payload`, which bounds the
        object at its own closing brace; the goal survives prose that carries
        braces of its own on either side of it.
        """
        if not instruction or not isinstance(instruction, str):
            return None, None
        payload = _find_goal_payload(instruction)
        if payload is None:
            return None, None
        target_pose = payload.get("target_pose")
        target_joints = payload.get("target_joints")
        if isinstance(target_pose, list):
            tp: list[float] | None = [float(v) for v in target_pose]
        else:
            tp = None
        if isinstance(target_joints, dict):
            tj: dict[str, float] | None = {str(k): float(v) for k, v in target_joints.items()}
        else:
            tj = None
        return tp, tj

    @staticmethod
    def _validate_target_pose(target_pose: Any) -> None:
        """Validate ``target_pose`` is a 7-element list of finite floats."""
        try:
            poses = list(target_pose)
        except TypeError as e:
            raise ValueError(f"target_pose must be a 7-element list, got {type(target_pose).__name__}") from e
        if len(poses) != 7:
            raise ValueError(f"target_pose must have exactly 7 elements [x,y,z,qw,qx,qy,qz], got {len(poses)}")
        for i, v in enumerate(poses):
            try:
                f = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"target_pose[{i}] must be a number, got {type(v).__name__}") from e
            if math.isnan(f) or math.isinf(f):
                raise ValueError(f"target_pose[{i}]={f!r} must be finite")

    @staticmethod
    def _validate_target_joints(target_joints: Any) -> None:
        """Validate ``target_joints`` is a name->finite-float mapping."""
        if not isinstance(target_joints, dict):
            raise ValueError(f"target_joints must be a dict[str, float], got {type(target_joints).__name__}")
        pattern = re.compile(_JOINT_NAME_PATTERN)
        for k, v in target_joints.items():
            if not isinstance(k, str) or not pattern.match(k):
                raise ValueError(
                    f"target_joints key {k!r} must match {_JOINT_NAME_PATTERN!r} (letters, digits, underscore, hyphen)"
                )
            try:
                f = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"target_joints[{k!r}]={v!r} must be a number") from e
            if math.isnan(f) or math.isinf(f):
                raise ValueError(f"target_joints[{k!r}]={f!r} must be finite")


__all__ = ["CuroboPolicy"]
