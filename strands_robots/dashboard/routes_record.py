"""``/api/record``, ``/api/collect``, ``/api/replay`` and ``/api/datasets/labels`` - the record screen.

The session surface (open an arm pair, start/stop/redo/discard episodes, thumbnails, close) lives
in :mod:`strands_robots.dashboard.record_api` and is mounted here late-bound: the controller is
read off ``request.app.state.record`` per request, so the router exists before the app does and
a coordinator that swaps the bridge or the device manager does not pin a stale one.

Every path a client names is contained to the dataset home (``$HF_LEROBOT_HOME``, the same
place :mod:`strands_robots.dataset_source` writes to). A path outside it is refused with 400 and
one fixed body whether or not it exists; the dashboard is a network service and the filesystem
outside the dataset home is not its business to describe.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request

from strands_robots.dashboard import access, record_api
from strands_robots.dashboard.dataset_check import OUTSIDE_DATASET_HOME

logger = logging.getLogger(__name__)

router = APIRouter(tags=["record"], dependencies=[Depends(access.require_session)])


def dataset_home() -> Path:
    """Where datasets live on this machine - lerobot's own constant when it is installed."""
    from strands_robots.dataset_source import _lerobot_home

    return _lerobot_home().expanduser().resolve()


def contained_path(raw: Any) -> Path:
    """Resolve a client-named path and refuse it unless it sits under :func:`dataset_home`.

    ``resolve()`` follows symlinks and folds ``..`` before the containment check, so a link out of
    the home does not count as inside it. The refusal carries no trace of the target.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise HTTPException(422, "path required")
    try:
        target = Path(raw.strip()).expanduser().resolve()
    except (OSError, RuntimeError):
        raise HTTPException(400, OUTSIDE_DATASET_HOME) from None
    if not target.is_relative_to(dataset_home()):
        raise HTTPException(400, OUTSIDE_DATASET_HOME)
    return target


# ---------------------------------------------------------------- controller


def controller(request: Request) -> record_api.RecordController:
    """The app's controller, built on first use when :func:`attach` could not build it eagerly."""
    state = request.app.state
    existing = getattr(state, "record", None)
    if existing is None:
        existing = _build(request.app)
        state.record = existing
    return cast("record_api.RecordController", existing)


def _build(app: FastAPI) -> record_api.RecordController:
    devices = getattr(app.state, "devices", None)
    bridge = getattr(app.state, "bridge", None)
    return record_api.RecordController(devices, bridge=bridge)


def _activity(request: Request) -> Callable[..., None] | None:
    # Late-bound on purpose: capturing a bound method would pin the router to one bridge instance.
    bridge = getattr(request.app.state, "bridge", None)
    if bridge is None:
        return None
    return cast("Callable[..., None] | None", getattr(bridge, "record_activity", None))


def attach(app: FastAPI) -> None:
    """Set ``app.state.record``; when the device manager is not attached yet, finish at startup."""
    if getattr(app.state, "devices", None) is not None:
        app.state.record = _build(app)
        return

    async def _late() -> None:
        # The devices lane may attach after this one; rebuild once everything has run.
        if getattr(app.state, "record", None) is None or app.state.record._devices is None:
            app.state.record = _build(app)

    hooks = getattr(app.state, "startup_hooks", None)
    if isinstance(hooks, list):
        hooks.append(_late)
    else:
        app.state.record = _build(app)


router.include_router(record_api.build_router(controller, _activity, late_bound=True))


# ------------------------------------------------------------ one-shot runs


def _devices_or_503(request: Request) -> Any:
    devices = getattr(request.app.state, "devices", None)
    if devices is None:
        raise HTTPException(503, "no device manager is attached to this dashboard - nothing can run a one-shot sim")
    return devices


@router.post("/api/collect")
async def collect_episodes(request: Request, body: dict[str, Any]) -> dict[str, Any]:
    """Collect a policy-driven dataset in a one-shot mesh sim. run_policy drives exactly n_episodes
    rollouts with per-episode parquet boundaries and reports parquet-truth counts.
    """
    dataset_root = str(contained_path(body.get("dataset_root")))
    devices = _devices_or_503(request)
    # Remember the root so /api/training/datasets discovers the result even outside the default
    # scan paths. The training lane ships the memory; without it the collection still runs.
    try:
        from strands_robots.dashboard import training as _training

        _training.remember_dataset_root(dataset_root)
    except ImportError:
        logger.debug("[record] training module absent; %s will not be remembered for the picker", dataset_root)
    result = await asyncio.to_thread(
        lambda: devices.collect(
            dataset_root=dataset_root,
            dataset_repo_id=body.get("dataset_repo_id", "local/collected"),
            robot_name=body.get("robot_name") or "so101",
            policy_provider=body.get("policy_provider", "mock"),
            policy_config=body.get("policy_config"),
            instruction=body.get("instruction", ""),
            n_episodes=int(body.get("n_episodes", 5)),
            duration=float(body.get("duration", 10.0)),
            fps=int(body.get("fps", 30)),
        )
    )
    # Two recorders writing one dataset directory interleave episodes into each other's files.
    # 409 names the session already holding it.
    if result.get("already_running"):
        raise HTTPException(409, result)
    return cast("dict[str, Any]", result)


@router.post("/api/replay")
async def replay_episode(request: Request, body: dict[str, Any]) -> dict[str, Any]:
    """Replay a recorded LeRobotDataset episode in a one-shot mesh sim."""
    repo_id = (body.get("repo_id") or "").strip()
    if not repo_id:
        raise HTTPException(422, "repo_id required")
    root = body.get("root")
    if root is not None:
        # Containment first: validate_replay would otherwise say whether the directory exists.
        root = str(contained_path(root))
    devices = _devices_or_503(request)
    from strands_robots.dashboard.device_manager import validate_replay

    bad = validate_replay(repo_id, body.get("episode", 0), root, body.get("speed", 1.0))
    if bad:
        raise HTTPException(422, bad)
    result = await asyncio.to_thread(
        devices.replay,
        repo_id,
        int(body.get("episode", 0)),
        root,
        float(body.get("speed", 1.0)),
        body.get("robot_name") or "so101",
    )
    # 409, not an error-shaped 200: a second replay of the same episode is a conflict with
    # something that already exists, and the response names the peer already showing it.
    if result.get("already_running"):
        raise HTTPException(409, result)
    return cast("dict[str, Any]", result)


# ------------------------------------------------------------------ labels


@router.get("/api/datasets/labels")
async def dataset_labels(root: str | None = None, path: str | None = None) -> dict[str, Any]:
    """The episode label sidecar of one local dataset, as the label view renders it."""
    from strands_robots import episode_labels as _labels
    from strands_robots.dashboard.episode_label_view import label_view

    target = contained_path(root if root is not None else path)
    if not target.is_dir():
        raise HTTPException(404, "no dataset directory there")

    document: dict[str, Any] | None = None
    sidecar_error: str | None = None
    if _labels.labels_path(target).exists():
        try:
            document = _labels.read_labels(target)
        except Exception as e:  # noqa: BLE001 - a corrupt sidecar must not read as "no labels yet"
            sidecar_error = f"{type(e).__name__}: {e}"

    total: int | None = None
    try:
        total = json.loads((target / "meta" / "info.json").read_text(encoding="utf-8")).get("total_episodes")
    except Exception:  # noqa: BLE001 - a dataset mid-recording has no readable info.json yet
        pass

    return label_view(document, total_episodes=total, sidecar_error=sidecar_error)
